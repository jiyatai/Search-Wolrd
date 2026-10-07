# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test script to verify BEV decoding from trained checkpoint."""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import gin
import numpy as np
import torch
from PIL import Image
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from model.trainer import SearchWorldTrainer
from model.dataset.uav_parquet_dataset import UAVParquetDataModule, UAVParquetDataset
from model.searchworld.bev_memory import BEVMemory


# BEV visualization colors
BEV_COLORS = np.array([
    [0, 0, 0],         # Channel 0: unexplored
    [200, 200, 200],  # Channel 0: explored
    [0, 0, 0],         # Channel 1: free
    [255, 0, 0],       # Channel 1: obstacle
    [0, 0, 0],         # Channel 2: low value
    [0, 255, 0],       # Channel 2: high value
], dtype=np.uint8)


def visualize_bev(bev_tensor, prefix="bev", output_dir="test_outputs"):
    """Visualize BEV map and save to file."""
    os.makedirs(output_dir, exist_ok=True)

    # Handle both (3, H, W) and (S, 3, H, W) or (B, S, 3, H, W)
    if len(bev_tensor.shape) == 4:
        bev = bev_tensor[0, 0]  # first batch, first seq
    elif len(bev_tensor.shape) == 3:
        bev = bev_tensor[0]  # first seq
    else:
        bev = bev_tensor

    if isinstance(bev, torch.Tensor):
        bev = bev.detach().cpu().numpy()

    h, w = bev.shape[1:3] if bev.shape[0] == 3 else bev.shape[2:4]
    if len(bev.shape) == 3 and bev.shape[0] != 3:
        # (H, W, 3) - swap
        bev = bev.transpose(2, 0, 1)

    # Create visualization
    viz = np.zeros((h, w, 3), dtype=np.uint8)

    # Channel 0: exploration layer (gray scale)
    explore = bev[0] if bev.shape[0] == 3 else bev[:, :, 0]
    viz[:, :, 0] = (explore * 200).astype(np.uint8)
    viz[:, :, 1] = (explore * 200).astype(np.uint8)
    viz[:, :, 2] = (explore * 200).astype(np.uint8)

    # Channel 1: obstacle layer (red overlay)
    obstacle = bev[1] if bev.shape[0] == 3 else bev[:, :, 1]
    viz[:, :, 0] = np.maximum(viz[:, :, 0], (obstacle * 255).astype(np.uint8))

    # Channel 2: value layer (green overlay)
    if bev.shape[0] >= 3:
        value = bev[2]
        viz[:, :, 1] = np.maximum(viz[:, :, 1], ((value + 1) / 2 * 255).astype(np.uint8))

    # Save
    output_path = os.path.join(output_dir, f"{prefix}.png")
    Image.fromarray(viz).save(output_path)
    print(f"Saved BEV visualization to {output_path}")

    # Also save individual channels
    for c in range(min(3, bev.shape[0])):
        channel = bev[c] if bev.shape[0] == 3 else bev[:, :, c]
        channel_viz = (channel * 255).astype(np.uint8)
        if len(channel_viz.shape) == 2:
            channel_viz = np.stack([channel_viz]*3, axis=-1)
        channel_path = os.path.join(output_dir, f"{prefix}_ch{c}.png")
        Image.fromarray(channel_viz).save(channel_path)


def load_checkpoint(checkpoint_path, config_path, device="cuda" if torch.cuda.is_available() else "cpu"):
    """Load model from checkpoint and gin config."""
    # Parse gin config
    if config_path and os.path.exists(config_path):
        print(f"Loading config from {config_path}")
        gin.parse_config_file(config_path)

    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Try to load as SearchWorldTrainer
    model = None
    try:
        model = SearchWorldTrainer.load_from_checkpoint(checkpoint_path, map_location=device)
        print("Successfully loaded as SearchWorldTrainer")
    except Exception as e:
        print(f"Could not load as SearchWorldTrainer: {e}")
        print("Trying direct model instantiation...")

        # Try to load state dict directly
        if model is None and 'state_dict' in checkpoint:
            from model.searchworld.searchworld import SearchWorld
            model = SearchWorld()
            state_dict = checkpoint['state_dict']
            # Remove 'model.' prefix if present
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('model.'):
                    new_state_dict[k[6:]] = v
                else:
                    new_state_dict[k] = v
            model.load_state_dict(new_state_dict, strict=False)
            print("Loaded model state dict directly")

    if model is None:
        raise ValueError("Could not load model from checkpoint")

    # Move to eval mode
    model = model.to(device)
    model.eval()
    if hasattr(model, 'model'):
        model.model.eval()

    return model


def test_bev_decoding(model, data_dir, output_dir="test_outputs", num_samples=5, device="cuda"):
    """Test BEV decoding with real data."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n=== Testing BEV Decoding ===")
    print(f"Output directory: {output_dir}")

    # Create dataset - use parquet with BEV
    try:
        dataset = UAVParquetDataset(
            dataset_path=os.path.join(data_dir, "test"),
            sequence_length=4,
            enable_semantic=True,
            enable_bev=True
        )
        print(f"Loaded dataset with {len(dataset)} samples")
    except Exception as e:
        print(f"Could not load UAVParquetDataset: {e}")
        print("Falling back to simple test...")
        return test_bev_decoding_simple(model, output_dir, num_samples, device)

    # Get a few samples
    for i in range(min(num_samples, len(dataset))):
        print(f"\nSample {i}")

        batch = dataset[i]

        # Add batch dimension
        batch = {k: v[None, ...] if isinstance(v, torch.Tensor) or isinstance(v, np.ndarray)
                 else [[v]] for k, v in batch.items()}

        # Move to device
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device)
            elif isinstance(batch[k], np.ndarray):
                batch[k] = torch.from_numpy(batch[k]).to(device)

        # Forward pass
        with torch.no_grad():
            if hasattr(model, 'model'):
                output = model.model(batch)
            else:
                output = model(batch)

        # Check output
        print(f"Output keys: {list(output.keys())}")

        if 'bev_pred' in output:
            bev_pred = output['bev_pred']
            print(f"BEV pred shape: {bev_pred.shape}")
            visualize_bev(bev_pred, f"bev_pred_sample{i}", output_dir)
        else:
            print("WARNING: 'bev_pred' not found in model output")

        if 'bev_gt' in batch or 'bev' in batch:
            bev_gt = batch.get('bev_gt', batch.get('bev'))
            print(f"BEV GT shape: {bev_gt.shape}")
            visualize_bev(bev_gt, f"bev_gt_sample{i}", output_dir)


def test_bev_decoding_simple(model, output_dir="test_outputs", num_samples=3, device="cuda"):
    """Simple test with dummy data if dataset not available."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n=== Running Simple BEV Test ===")

    # Create dummy input
    batch_size = 1
    seq_len = 4
    h, w = 320, 512

    dummy_batch = {
        'image': torch.randn(batch_size, seq_len, 3, h, w).to(device),
        'relative_pose': torch.randn(batch_size, seq_len, 4).to(device),
        'text_feat': torch.randn(batch_size, seq_len, 768).to(device),
        'action': torch.randint(0, 8, (batch_size, seq_len)).to(device),
    }

    # Forward pass
    with torch.no_grad():
        if hasattr(model, 'model'):
            output = model.model(dummy_batch)
        else:
            output = model(dummy_batch)

    print(f"Output keys: {list(output.keys())}")

    if 'bev_pred' in output:
        bev_pred = output['bev_pred']
        print(f"BEV pred shape: {bev_pred.shape}")
        print(f"BEV pred range: {bev_pred.min():.3f} to {bev_pred.max():.3f}")
        visualize_bev(bev_pred, "bev_pred_dummy", output_dir)
    else:
        print("WARNING: 'bev_pred' not found in model output")
        print("Check if enable_bev_decoder=True in config")

    return 'bev_pred' in output


def inspect_checkpoint(checkpoint_path):
    """Inspect what's in the checkpoint."""
    print(f"\n=== Inspecting Checkpoint ===")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    print(f"Top-level keys: {list(checkpoint.keys())}")

    if 'state_dict' in checkpoint:
        sd = checkpoint['state_dict']
        print(f"\nState dict keys (first 20):")
        for i, k in enumerate(list(sd.keys())[:30]):
            print(f"  {k}: {sd[k].shape}")

        print(f"\nTotal state dict keys: {len(sd)}")

        # Check for BEV-related weights
        bev_keys = [k for k in sd.keys() if 'bev' in k.lower()]
        if bev_keys:
            print(f"\nBEV-related keys:")
            for k in bev_keys:
                print(f"  {k}: {sd[k].shape}")
        else:
            print("\nNo BEV-related keys found in state dict")


def main():
    parser = argparse.ArgumentParser(description="Test BEV decoding from checkpoint")
    parser.add_argument('--checkpoint_path', type=str,
                        default='/shared_disk/users/wenhao.lu/JYT/WorldSearch_data/experiments/gwm_pretrain_with_bev/checkpoints/last.ckpt',
                        help='Path to checkpoint file')
    parser.add_argument('--config_path', type=str,
                        default='/mnt/pfs/users/luwenhao/code_jyt/SearchWorld/configs/gwm_pretrain_config.gin',
                        help='Path to gin config file')
    parser.add_argument('--data_dir', type=str,
                        default='/shared_disk/users/wenhao.lu/JYT/WorldSearch_data/BrushifyUrban_parquet_with_bev',
                        help='Path to test data directory')
    parser.add_argument('--output_dir', type=str,
                        default='/shared_disk/users/wenhao.lu/JYT/WorldSearch_data/experiments/gwm_pretrain_with_bev/test_outputs',
                        help='Path to output directory')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to test')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (default: auto-detect)')
    parser.add_argument('--inspect_only', action='store_true',
                        help='Just inspect the checkpoint without running inference')

    args = parser.parse_args()

    # Set device
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {args.device}")

    # Inspect checkpoint first
    inspect_checkpoint(args.checkpoint_path)

    if args.inspect_only:
        return

    # Load model
    try:
        model = load_checkpoint(args.checkpoint_path, args.config_path, args.device)
        print(f"Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test BEV decoding
    try:
        success = test_bev_decoding(model, args.data_dir, args.output_dir, args.num_samples, args.device)
    except Exception as e:
        print(f"Error in BEV testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
