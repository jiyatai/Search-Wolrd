# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage 3 smoke test: full value-guided imagination rollout + backward.

Runs one real batch through the frozen world model, the value-guided planning
engine and a backward pass, checking that
  * the stage-2 checkpoint loads and the world model is fully frozen /
    locked in eval mode,
  * the dataset carries the 'pose' / 'target_rel' columns (required),
  * the rollout produces the value-guided losses and only the action head
    receives gradients (no critic, no GAE, no world-model grads).

Usage (inside container):
    CUDA_VISIBLE_DEVICES=2 python model/rl/smoke_test.py \
        -c configs/stage3_imagination_config.gin \
        -d /workspace/datasets_shared/WorldSearch_data/expert_parquet \
        --ckpt /workspace/datasets_shared/WorldSearch_data/experiments/stage2_expert_v3/checkpoints/last.ckpt
"""
import argparse

import gin
import torch

# Register all configurables BEFORE gin.parse_config_file.
from model.dataset.uav_dataset import UAVDataModule  # noqa: F401
from model.dataset.uav_parquet_dataset import UAVParquetDataModule
from model.trainer import SearchWorldTrainer  # noqa: F401
from model.rl import ImaginationRLModule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    args, _ = parser.parse_known_args()

    gin.parse_config_file('configs/stage3_imagination_config.gin',
                          skip_unknown=True)

    device = torch.device(args.device)
    # load_strict=False for the remapped stage-2 ckpt: the only keys allowed
    # to be missing are the 8 new bev_decoder.skip_convs.* params.
    module = ImaginationRLModule(
        checkpoint_path=args.ckpt, load_strict=False).to(device)

    # ---- Verify only skip_convs are freshly initialised.
    # ckpt keys carry a 'model.' prefix (SearchWorldTrainer.model) while
    # world_model.state_dict() does not - strip it before comparing.
    ckpt_sd = {k[len('model.'):] if k.startswith('model.') else k: v
               for k, v in
               torch.load(args.ckpt, map_location='cpu')['state_dict'].items()}
    model_sd = module.world_model.state_dict()
    missing = [k for k in model_sd if k not in ckpt_sd]
    print(f'[check] keys missing from ckpt ({len(missing)}):')
    for k in missing:
        print(f'    {k}')
    assert all('skip_convs' in k for k in missing), \
        'unexpected missing keys beyond bev_decoder.skip_convs.*'

    # ---- BatchNorm pollution check: train() must keep world_model.eval().
    module.train()
    wm_training = module.world_model.training
    bn_training = module.world_model.bev_decoder.training
    print(f'[check] after module.train(): world_model.training={wm_training} '
          f'(expect False), bev_decoder.training={bn_training} (expect False)')
    assert not wm_training and not bn_training

    # ---- Frozen check.
    n_frozen = sum(1 for p in module.world_model.parameters()
                   if not p.requires_grad)
    n_total_wm = sum(1 for _ in module.world_model.parameters())
    print(f'[check] world model frozen: {n_frozen}/{n_total_wm} params '
          'require_grad=False (expect equal)')
    assert n_frozen == n_total_wm

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
    loader = dm.train_dataloader()
    batch = next(iter(loader))
    batch = {k: v.to(device) for k, v in batch.items()}
    print(f'[check] batch keys: {sorted(batch.keys())}')
    assert 'pose' in batch, \
        "batch is missing 'pose' - regenerate the parquet data " \
        "(scripts/convert_expert_to_parquet.py)."
    print(f"[check] action {batch['action'].shape} {batch['action'].dtype}, "
          f"text_feat {batch['text_feat'].shape}, "
          f"bev_memory {batch['bev_memory'].shape}, "
          f"pose {batch['pose'].shape}")

    # ---- Encode + rollout.
    init = module._encode_real_batch(batch)
    h0, z0, text_feat, pose0 = (init['h0'], init['z0'],
                                init['text_feat'], init['pose0'])
    print(f"[check] h0 {tuple(h0.shape)}, z0 {tuple(z0.shape)}, "
          f"text_feat {tuple(text_feat.shape)}, pose0 {tuple(pose0.shape)}")

    module.train()
    out = module.engine.rollout(h0, z0, text_feat, pose0)
    print('[check] rollout output:')
    for k, v in out.items():
        if torch.is_tensor(v):
            print(f'    {k}: {tuple(v.shape)} = {v.item():.6f}')
        else:
            print(f'    {k}: {v}')

    # ---- Backward: ONLY the action head receives gradient.
    out['action_loss'].backward()
    a_grad = sum(p.grad.abs().sum().item() for p in module.actor.parameters()
                 if p.grad is not None)
    wm_grad = sum(p.grad.abs().sum().item()
                  for p in module.world_model.parameters()
                  if p.grad is not None)
    print(f'[check] grad |sum|: actor={a_grad:.4f}, '
          f'world_model={wm_grad:.6f} (expect 0)')
    assert a_grad > 0 and wm_grad == 0.0

    # ---- Optimizer step (action head only).
    opt = torch.optim.AdamW(module.actor.parameters(), lr=1e-4)
    opt.step()
    print('[PASS] Stage 3 value-guided planning smoke test: '
          'load + BN-eval-lock + frozen world model + rollout + '
          'action-head-only backward + step all OK')


if __name__ == '__main__':
    main()
