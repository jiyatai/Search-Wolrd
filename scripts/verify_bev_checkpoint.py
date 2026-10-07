# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify BEV decoding from trained checkpoint in SearchWorld."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from PIL import Image
import gin


def visualize_bev(bev_np, output_path):
    """Visualize 3-channel BEV and save to file."""
    h, w = bev_np.shape[1:3] if bev_np.shape[0] == 3 else bev_np.shape[:2]

    # Handle (H, W, 3) or (3, H, W)
    if len(bev_np.shape) == 3 and bev_np.shape[0] == 3:
        bev = bev_np
    elif len(bev_np.shape) == 3 and bev_np.shape[2] == 3:
        bev = bev_np.transpose(2, 0, 1)
    elif len(bev_np.shape) == 4:
        bev = bev_np[0, 0] if bev_np.shape[1] == 3 else bev_np[0, :, :, 0]
    else:
        bev = bev_np

    # Create RGB visualization
    viz = np.zeros((h, w, 3), dtype=np.uint8)

    # Channel 0: Exploration (white = explored)
    explore = bev[0]
    viz[:, :, 0] = (explore * 255).astype(np.uint8)
    viz[:, :, 1] = (explore * 255).astype(np.uint8)
    viz[:, :, 2] = (explore * 255).astype(np.uint8)

    # Channel 1: Obstacle (red overlay)
    obstacle = bev[1]
    viz[:, :, 0] = np.maximum(viz[:, :, 0], (obstacle * 255).astype(np.uint8))

    # Channel 2: Value (green overlay)
    if bev.shape[0] >= 3:
        value = bev[2]
        # Value is typically 0-1
        viz[:, :, 1] = np.maximum(viz[:, :, 1], (value * 255).astype(np.uint8))

    Image.fromarray(viz).save(output_path)
    print(f"Saved BEV visualization to {output_path}")

    # Also save individual channels
    for c in range(min(3, bev.shape[0])):
        channel_viz = (bev[c] * 255).astype(np.uint8)
        channel_viz_rgb = np.stack([channel_viz]*3, axis=-1)
        channel_path = output_path.replace('.png', f'_ch{c}.png')
        Image.fromarray(channel_viz_rgb).save(channel_path)


def main():
    # Paths for SearchWorld project
    checkpoint_path = '/shared_disk/users/wenhao.lu/JYT/WorldSearch_data/experiments/gwm_pretrain_with_bev/checkpoints/last.ckpt'
    config_path = '/mnt/pfs/users/luwenhao/code_jyt/SearchWorld/configs/gwm_pretrain_config.gin'
    output_dir = '/shared_disk/users/wenhao.lu/JYT/WorldSearch_data/experiments/gwm_pretrain_with_bev/test_outputs'

    os.makedirs(output_dir, exist_ok=True)

    print("="*70)
    print("SearchWorld - BEV Checkpoint Verification")
    print("="*70)

    # 1. Load checkpoint
    print(f"\n[1] Loading checkpoint from {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print(f"  Checkpoint loaded! Keys: {list(checkpoint.keys())}")

    # 2. Inspect state dict
    print(f"\n[2] Inspecting state dict")
    if 'state_dict' in checkpoint:
        sd = checkpoint['state_dict']
        print(f"  State dict has {len(sd)} keys")

        # Look for BEV-related keys
        bev_keys = [k for k in sd.keys() if 'bev' in k.lower()]
        if bev_keys:
            print(f"\n  ✓ Found {len(bev_keys)} BEV-related keys:")
            for k in bev_keys[:15]:
                print(f"    {k}: {tuple(sd[k].shape)}")
            if len(bev_keys) > 15:
                print(f"    ... and {len(bev_keys)-15} more")

            # Also check if model has BEV decoder enabled
            has_bev_decoder = any('bev_decoder' in k for k in sd.keys())
            print(f"\n  BEV decoder present: {has_bev_decoder}")
        else:
            print(f"\n  ✗ No BEV keys found in state dict")

        # Save all state dict keys to a file
        with open(os.path.join(output_dir, 'state_dict_keys.txt'), 'w') as f:
            for k in sorted(sd.keys()):
                f.write(f"{k}: {tuple(sd[k].shape)}\n")
        print(f"\n  Saved state dict to {output_dir}/state_dict_keys.txt")

    # 3. Check hyperparameters
    if 'hyper_parameters' in checkpoint:
        hp = checkpoint['hyper_parameters']
        print(f"\n[3] Hyperparameters:")
        for k, v in hp.items():
            print(f"  {k}: {v}")

        with open(os.path.join(output_dir, 'hyperparams.txt'), 'w') as f:
            for k, v in hp.items():
                f.write(f"{k}: {v}\n")

    # 4. Try to load gin config
    print(f"\n[4] Loading gin config from {config_path}")
    try:
        gin.parse_config_file(config_path, skip_unknown=True)
        print("  ✓ Gin config loaded")
    except Exception as e:
        print(f"  Could not load gin config: {e}")

    # 5. Try to load the model
    print(f"\n[5] Attempting to load model...")
    try:
        from model.trainer import SearchWorldTrainer

        model = SearchWorldTrainer.load_from_checkpoint(
            checkpoint_path,
            map_location='cpu',
            strict=False
        )
        print("  ✓ SearchWorldTrainer loaded!")

        # Inspect the model
        xmob_model = model.model
        print(f"\n  Model: {type(xmob_model)}")

        # Check what decoders are enabled
        print(f"  Enable semantic: {getattr(xmob_model, 'enable_semantic', 'N/A')}")
        print(f"  Enable RGB stylegan: {getattr(xmob_model, 'enable_rgb_stylegan', 'N/A')}")
        print(f"  Enable RGB diffusion: {getattr(xmob_model, 'enable_rgb_diffusion', 'N/A')}")
        print(f"  Enable BEV decoder: {getattr(xmob_model, 'enable_bev_decoder', 'N/A')}")

        # Check if BEV decoder exists
        if hasattr(xmob_model, 'bev_decoder'):
            print(f"  ✓ BEV decoder found! Type: {type(xmob_model.bev_decoder)}")

            # 6. Test with dummy input
            print(f"\n[6] Testing model forward pass with dummy input...")

            # Set to eval mode
            model.eval()
            xmob_model.eval()

            # Create dummy batch
            batch_size = 1
            seq_len = 4
            img_h, img_w = 320, 512

            dummy_batch = {
                'image': torch.randn(batch_size, seq_len, 3, img_h, img_w),
                'relative_pose': torch.randn(batch_size, seq_len, 4),
                'text_feat': torch.randn(batch_size, seq_len, 768),
                'action': torch.randint(0, 8, (batch_size, seq_len)),
            }

            # Forward pass
            with torch.no_grad():
                output = xmob_model(dummy_batch)

            print(f"  ✓ Forward pass completed!")
            print(f"  Output keys: {list(output.keys())}")

            # Check for BEV output
            if 'bev_pred' in output:
                bev_pred = output['bev_pred']
                print(f"  ✓ BEV prediction found! Shape: {bev_pred.shape}")
                print(f"  BEV min/max: {bev_pred.min():.3f} / {bev_pred.max():.3f}")

                # Visualize
                bev_np = bev_pred[0, 0].numpy()  # First batch, first step
                viz_path = os.path.join(output_dir, 'bev_pred_dummy.png')
                visualize_bev(bev_np, viz_path)

                # Save raw BEV data
                np.save(os.path.join(output_dir, 'bev_pred_dummy.npy'), bev_np)

            else:
                print(f"  ✗ 'bev_pred' not found in model output")
                print(f"  Available outputs: {list(output.keys())}")

                # Also print what decoders the model has
                print(f"\n  Model attributes:")
                for attr in dir(xmob_model):
                    if not attr.startswith('_'):
                        print(f"    {attr}")

        else:
            print(f"  ✗ Model has no 'bev_decoder' attribute")
            print(f"  Model attributes: {[attr for attr in dir(xmob_model) if not attr.startswith('_')]}")

    except Exception as e:
        print(f"\n  Error loading model: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n" + "="*70)
    print(f"Verification complete! Results saved to {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
