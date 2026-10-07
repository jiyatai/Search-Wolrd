#!/usr/bin/env python3
"""Test decoding and visualize results."""

import os
import sys
from pathlib import Path

import gin
import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from model.trainer import SearchWorldTrainer
from model.dataset.uav_parquet_dataset import UAVParquetDataModule
from model.visualization import visualise_semantic


def save_image_grid(tensor, save_path, nrow=4):
    """Save image grid from tensor."""
    # tensor shape: [B, 3, H, W]
    tensor = tensor.cpu().float()
    tensor = (tensor * 255).byte()
    tensor = tensor.permute(0, 2, 3, 1)  # [B, H, W, 3]

    # Make grid
    b, h, w, c = tensor.shape
    ncol = min(b, nrow)
    nrow = (b + ncol - 1) // ncol
    grid = np.zeros((nrow * h, ncol * w, 3), dtype=np.uint8)

    for i in range(b):
        r = i // ncol
        cc = i % ncol
        grid[r*h:(r+1)*h, cc*w:(cc+1)*w] = tensor[i].numpy()

    Image.fromarray(grid).save(save_path)


def save_semantic_grid(tensor, save_path, nrow=4):
    """Save semantic segmentation grid."""
    # tensor shape: [B, H, W]
    # Use different colors for 6 classes
    colors = [
        [0, 0, 0],         # Sky/Background - black
        [128, 64, 128],    # Ground - dark purple
        [70, 70, 70],      # Building - dark gray
        [220, 20, 60],     # Human - red
        [0, 0, 142],       # Vehicle - blue
        [107, 142, 35],    # Obstacle - green
    ]

    tensor = tensor.cpu().numpy().astype(np.int64)
    b, h, w = tensor.shape

    rgb = np.zeros((b, h, w, 3), dtype=np.uint8)
    for i in range(b):
        for c in range(6):
            mask = tensor[i] == c
            rgb[i][mask] = colors[c]

    # Make grid
    ncol = min(b, nrow)
    nrow = (b + ncol - 1) // ncol
    grid = np.zeros((nrow * h, ncol * w, 3), dtype=np.uint8)

    for i in range(b):
        r = i // ncol
        cc = i % ncol
        grid[r*h:(r+1)*h, cc*w:(cc+1)*w] = rgb[i]

    Image.fromarray(grid).save(save_path)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', required=True, help='Config file')
    parser.add_argument('--dataset', '-d', required=True, help='Dataset path')
    parser.add_argument('--checkpoint', '-p', required=True, help='Checkpoint path')
    parser.add_argument('--output', '-o', default='./vis_results', help='Output dir')
    parser.add_argument('--num-batches', type=int, default=2, help='Number of batches to test')
    args = parser.parse_args()

    # Parse config
    gin.parse_config_file(args.config, skip_unknown=True)

    # Output dir
    os.makedirs(args.output, exist_ok=True)

    # Load model
    print(f"Loading model from {args.checkpoint}")
    model = SearchWorldTrainer.load_from_checkpoint(args.checkpoint, strict=False)
    model.eval()
    model.cuda()

    # Load data (small batch, no distributed)
    print(f"Loading data from {args.dataset}")
    data_module = UAVParquetDataModule(
        dataset_path=args.dataset,
        batch_size=4,
        sequence_length=4,
        num_workers=0,
        enable_semantic=True,
        use_lazy_loading=True,
    )
    data_module.setup()
    val_loader = data_module.val_dataloader()

    # Run some batches
    print("Running inference...")
    for batch_idx, batch in enumerate(val_loader):
        if batch_idx >= args.num_batches:
            break

        print(f"Batch {batch_idx}")

        # Move to GPU
        batch_gpu = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch_gpu[k] = v.cuda()

        # Forward pass
        with torch.no_grad():
            losses, output = model.shared_step(batch_gpu)

        # Print losses
        print(f"  Losses: {losses}")

        # Save input images
        # input image shape: [B, S, 3, H, W], we take first sequence element
        input_img = batch_gpu['image'][:, 0]  # [B, 3, H, W]
        save_image_grid(input_img, f"{args.output}/batch{batch_idx}_input.png")

        # Save semantic labels (input)
        if 'semantic_label_1' in batch_gpu:
            semantic_gt = batch_gpu['semantic_label_1'][:, 0]  # [B, H, W]
            save_semantic_grid(semantic_gt, f"{args.output}/batch{batch_idx}_semantic_gt.png")

        # Save semantic predictions
        if 'semantic_segmentation_1' in output:
            semantic_pred = output['semantic_segmentation_1'][:, 0].argmax(dim=1)  # [B, H, W]
            save_semantic_grid(semantic_pred, f"{args.output}/batch{batch_idx}_semantic_pred.png")

        # Save step-by-step predictions
        seq_len = batch_gpu['image'].shape[1]
        for s in range(seq_len):
            if 'semantic_segmentation_1' in output:
                semantic_pred_s = output['semantic_segmentation_1'][:, s].argmax(dim=1)
                save_semantic_grid(semantic_pred_s, f"{args.output}/batch{batch_idx}_step{s}_semantic_pred.png")

        print(f"  Saved results to {args.output}")

    print("Done!")


if __name__ == '__main__':
    main()
