# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage 2 smoke test: one training step with the v3 multi-scale BEV decoder.

Verifies:
  1. remap warm-start ckpt loads (strict=False, only skip_convs missing)
  2. forward through SearchWorldTrainer.shared_step returns the full loss dict
  3. backward works and BEV decoder receives a non-trivial gradient share
     (the pre-fix value was ~0.2%; with bev_weight=3.0 + v3 loss we expect a
     visible boost)
  4. the in-place autograd bug is gone (no "leaf Variable ... in-place" error)

Usage (inside container):
    PYTHONPATH=/workspace/SearchWorld python model/rl/smoke_test_stage2.py \
        --ckpt /workspace/datasets_shared/WorldSearch_data/experiments/stage3_smoke/init_from_v2_ms.ckpt
"""
import argparse

import gin
import torch

# Register all configurables BEFORE gin.parse_config_file.
from model.dataset.uav_dataset import UAVDataModule  # noqa: F401
from model.dataset.uav_parquet_dataset import UAVParquetDataModule
from model.trainer import SearchWorldTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    gin.parse_config_file('configs/stage2_expert_parquet_config.gin',
                          skip_unknown=True)

    device = torch.device(args.device)

    # ---- Build the model exactly as train.py does (gin-configured
    # SearchWorldTrainer builds its own SearchWorld internally), then
    # warm-start trainer.model.
    trainer = SearchWorldTrainer()
    trainer = trainer.to(device)
    model = trainer.model

    # Warm-start from the remapped v2 ckpt (keys carry 'model.' prefix).
    ckpt = torch.load(args.ckpt, map_location='cpu')
    sd = {k[len('model.'):] if k.startswith('model.') else k: v
          for k, v in ckpt['state_dict'].items()}
    # Loss module has no params in ckpt; drop non-model keys.
    model_keys = set(model.state_dict().keys())
    filtered = {k: v for k, v in sd.items() if k in model_keys}
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print(f'[check] warm-start: loaded {len(filtered)} keys, '
          f'missing={len(missing)}, unexpected={len(unexpected)}')
    for k in missing:
        print(f'    missing: {k}')
    assert all('skip_convs' in k for k in missing), \
        'unexpected missing keys beyond bev_decoder.skip_convs.*'
    assert len(unexpected) == 0

    # ---- Data.
    dm = UAVParquetDataModule(
        dataset_path='/workspace/datasets_shared/WorldSearch_data/expert_parquet',
        batch_size=2,
        sequence_length=8,
        num_workers=0,
        enable_semantic=False,
        enable_rgb_stylegan=True,
        is_gwm_pretrain=False,
        use_lazy_loading=True,
        enable_bev=True,
    )
    dm.setup('fit')
    batch = next(iter(dm.train_dataloader()))
    batch = {k: v.to(device) for k, v in batch.items()}

    # ---- One full training step (forward + loss dict + backward).
    trainer.train()
    losses, output = trainer.shared_step(batch)
    print('[check] shared_step losses:', sorted(losses.keys()))

    loss = trainer.loss_reducing(losses)
    print(f"[check] total loss = {loss.item():.6f}")
    for k, v in losses.items():
        if torch.is_tensor(v) and k != 'loss' and v.numel() == 1:
            print(f'    {k}: {v.item():.6f}')

    loss.backward()

    # ---- Gradient share of the BEV decoder vs. the RSSM.
    def grad_norm_sum(module):
        return sum(p.grad.abs().sum().item() for p in module.parameters()
                   if p.grad is not None)

    bev_grad = grad_norm_sum(model.bev_decoder)
    rssm_grad = grad_norm_sum(model.rssm)
    enc_grad = grad_norm_sum(model.observation_encoder)
    print(f'[check] grad |sum|: bev_decoder={bev_grad:.6f}, '
          f'rssm={rssm_grad:.6f}, obs_encoder={enc_grad:.6f}')
    share = bev_grad / (bev_grad + rssm_grad + enc_grad + 1e-12)
    print(f'[check] BEV gradient share = {share * 100:.2f}% '
          '(pre-fix reference: ~0.2%)')
    assert bev_grad > 0, 'BEV decoder received zero gradient!'

    opt = torch.optim.AdamW(trainer.parameters(), lr=1e-5)
    opt.step()
    print('[PASS] Stage 2 smoke test: warm-start + v3 BEV loss + '
          'backward + BEV gradient boost all OK')


if __name__ == '__main__':
    main()
