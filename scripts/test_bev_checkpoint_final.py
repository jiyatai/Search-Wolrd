# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Final test script to verify BEV checkpoint and generate visualizations.
Run this in Docker container:
python scripts/test_bev_checkpoint_final.py
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import gin
import numpy as np
from PIL import Image


def visualize_bev(bev_tensor, prefix="bev", output_dir="test_outputs"):
    """Visualize BEV map and save to file."""
    os.makedirs(output_dir, exist_ok=True)

    # Handle different formats
    if isinstance(bev_tensor, torch.Tensor):
        bev_array = bev_tensor.detach().cpu().numpy()
    else:
        bev_array = np.array(bev_tensor)

    # Handle batch/sequence dimensions
    if len(bev_array.shape) == 5:  # (b, s, 3, h, w)
        bev_array = bev_array[0, 0]  # first batch, first step
    elif len(bev_array.shape) == 4:  # (b, 3, h, w)
        bev_array = bev_array[0]  # first batch
    elif len(bev_array.shape) == 3:  # (3, h, w)
        pass  # keep as is
    elif len(bev_array.shape) == 2:  # (h, w)
        # Single channel, duplicate to 3
        bev_array = np.stack([bev_array]*3, axis=0)

    h, w = bev_array.shape[1], bev_array.shape[2]

    # Create RGB visualization
    viz = np.zeros((h, w, 3), dtype=np.uint8)

    # Channel 0: Exploration (gray scale)
    explore = (np.clip(bev_array[0], 0, 1) * 255).astype(np.uint8)
    viz[:, :, 0] = explore
    viz[:, :, 1] = explore
    viz[:, :, 2] = explore

    # Channel 1: Obstacle (red overlay)
    obstacle = (np.clip(bev_array[1], 0, 1) * 255).astype(np.uint8)
    viz[:, :, 0] = np.maximum(viz[:, :, 0], obstacle)

    # Channel 2: Value (green overlay)
    if bev_array.shape[0] >= 3:
        value = bev_array[2]
        if value.min() < 0:
            # Value ranges from -1 to 1, normalize to 0-1
            value = (value + 1) / 2
        value = (np.clip(value, 0, 1) * 255).astype(np.uint8)
        viz[:, :, 1] = np.maximum(viz[:, :, 1], value)

    # Save combined
    output_path = os.path.join(output_dir, f"{prefix}_combined.png")
    Image.fromarray(viz).save(output_path)
    print(f"Saved BEV to: {output_path}")

    # Save individual channels
    for c in range(min(3, bev_array.shape[0])):
        ch_array = (np.clip(bev_array[c], 0, 1) * 255).astype(np.uint8)
        ch_viz = np.stack([ch_array]*3, axis=-1)
        ch_names = ['exploration', 'obstacle', 'value']
        ch_output = os.path.join(output_dir, f"{prefix}_{ch_names[c]}.png")
        Image.fromarray(ch_viz).save(ch_output)


def main():
    # Setup paths
    project_dir = Path(__file__).parent.parent
    checkpoint_path = '/shared_disk/users/wenhao.lu/JYT/WorldSearch_data/experiments/gwm_pretrain_with_bev_v2/checkpoints/epoch=73-step=4440.ckpt'
    base_config = project_dir / 'configs' / 'base_train_config.gin'
    parquet_config = project_dir / 'configs' / 'gwm_pretrain_parquet_config.gin'
    output_dir = '/shared_disk/users/wenhao.lu/JYT/WorldSearch_data/experiments/gwm_pretrain_with_bev/test_outputs'

    print("=" * 70)
    print("SearchWorld - BEV Checkpoint Verification")
    print("=" * 70)

    # Verify paths
    print(f"\nCheckpoint: {checkpoint_path} - {'✓' if os.path.exists(checkpoint_path) else '✗'}")
    print(f"Base config: {base_config} - {'✓' if base_config.exists() else '✗'}")
    print(f"Parquet config: {parquet_config} - {'✓' if parquet_config.exists() else '✗'}")

    if not os.path.exists(checkpoint_path):
        print("\nERROR: Checkpoint not found!")
        return

    # Step 1: Inspect checkpoint
    print(f"\n[1] Inspecting checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    print(f"  - Epoch: {checkpoint.get('epoch', 'n/a')}")
    print(f"  - Global step: {checkpoint.get('global_step', 'n/a')}")

    if 'state_dict' in checkpoint:
        sd = checkpoint['state_dict']
        print(f"  - State dict keys: {len(sd)}")

        # Find BEV keys
        bev_keys = [k for k in sd.keys() if 'bev' in k.lower()]
        semantic_keys = [k for k in sd.keys() if 'semantic' in k.lower()]
        rgb_keys = [k for k in sd.keys() if 'rgb_decoder' in k.lower()]

        print(f"\n[2] Decoders in checkpoint:")
        print(f"  - Semantic decoder: {len(semantic_keys) > 0}")
        print(f"  - RGB decoder: {len(rgb_keys) > 0}")
        print(f"  - BEV decoder: {len(bev_keys) > 0}")

        if bev_keys:
            print(f"\n[3] BEV decoder weights found:")
            for k in bev_keys[:15]:
                print(f"  - {k}: {sd[k].shape}")

        # Save state dict keys
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'state_dict_keys.txt'), 'w') as f:
            for k in sorted(sd.keys()):
                f.write(f"{k}: {tuple(sd[k].shape)}\n")
        print(f"\n[4] Saved state dict to {output_dir}/state_dict_keys.txt")

    # Step 2: Try to load model with correct gin config
    print(f"\n[5] Loading model with gin config...")

    try:
        from model.trainer import SearchWorldTrainer

        # Clear any existing config
        gin.clear_config()

        # Load actual training config files
        gin.parse_config_file(str(base_config), skip_unknown=True)
        gin.parse_config_file(str(parquet_config), skip_unknown=True)

        print("  ✓ Gin config loaded")

        # Load model
        model = SearchWorldTrainer.load_from_checkpoint(
            checkpoint_path,
            map_location='cpu',
            strict=False
        )
        print("  ✓ Model loaded")

        # Check model structure
        xmob = model.model
        print(f"\n[6] Model structure:")
        print(f"  - enable_semantic: {getattr(xmob, 'enable_semantic', 'n/a')}")
        print(f"  - enable_rgb_stylegan: {getattr(xmob, 'enable_rgb_stylegan', 'n/a')}")
        print(f"  - enable_bev_decoder: {getattr(xmob, 'enable_bev_decoder', 'n/a')}")
        print(f"  - has semantic_decoder: {hasattr(xmob, 'semantic_decoder')}")
        print(f"  - has rgb_decoder: {hasattr(xmob, 'rgb_decoder')}")
        print(f"  - has bev_decoder: {hasattr(xmob, 'bev_decoder')}")

        # Step 3: Test forward pass
        print(f"\n[7] Testing forward pass...")
        model.eval()
        xmob.eval()

        # Create dummy batch
        dummy_batch = {
            'image': torch.randn(1, 4, 3, 320, 512),
            'relative_pose': torch.randn(1, 4, 4),
            'text_feat': torch.randn(1, 4, 768),
            'action': torch.randint(0, 8, (1, 4)),
        }

        with torch.no_grad():
            output = xmob(dummy_batch)

        print(f"  ✓ Forward pass completed!")
        print(f"\n[8] Model outputs:")
        for k, v in output.items():
            if isinstance(v, torch.Tensor):
                print(f"  - {k}: {tuple(v.shape)}")
            elif isinstance(v, dict):
                print(f"  - {k}: dict with keys {list(v.keys())}")
            else:
                print(f"  - {k}: {type(v)}")

        # Check for BEV output
        if 'bev_pred' in output:
            print(f"\n[9] ✓ BEV output found!")
            bev_pred = output['bev_pred']
            print(f"  - Shape: {tuple(bev_pred.shape)}")
            print(f"  - Min: {bev_pred.min():.3f}")
            print(f"  - Max: {bev_pred.max():.3f}")

            # Visualize BEV
            visualize_bev(bev_pred, 'bev_prediction', output_dir)
            print(f"\n[10] Visualizations saved to {output_dir}")
        else:
            print(f"\n[9] ✗ BEV output not found")
            print(f"  - Check if enable_bev_decoder is True and the weights exist")

    except Exception as e:
        print(f"\n  ✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n" + "=" * 70)
    print("Verification complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
