# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare BEV prediction vs Ground Truth."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import gin
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# Discrete action label -> human-readable name (matches convert_uav_to_parquet.py)
ACTION_NAMES = [
    'forward', 'left', 'right', 'ascend', 'descend', 'rotl', 'rotr', 'stop',
]

# BEV channel names and their visualization colors (RGB)
CHANNEL_NAMES = ['exploration', 'obstacle', 'value']
CHANNEL_COLORS = [
    (0, 255, 0),    # exploration: green
    (255, 0, 0),    # obstacle: red
    (0, 0, 255),    # value: blue
]


def format_action(action):
    """Format an integer action label as 'index (name)' for display."""
    if action is None:
        return 'unknown'
    if isinstance(action, torch.Tensor):
        action = int(action.item())
    name = ACTION_NAMES[int(action)] if 0 <= int(action) < len(ACTION_NAMES) else '?'
    return f"{int(action)} ({name})"


def channel_to_rgb(channel, color):
    """Convert a single [0,1] channel to a colored RGB uint8 image."""
    if isinstance(channel, torch.Tensor):
        channel = channel.detach().cpu().numpy()
    ch = np.clip(channel, 0, 1)[..., None]  # (h, w, 1)
    return (ch * np.array(color, dtype=np.float32)).astype(np.uint8)


def denormalize_bev(bev_array):
    """Denormalize BEV channels to 0-255 for visualization."""
    if isinstance(bev_array, torch.Tensor):
        bev_array = bev_array.detach().cpu().numpy()

    c, h, w = bev_array.shape
    viz = np.zeros((h, w, 3), dtype=np.uint8)

    # Channel 0: Exploration (gray)
    explore = (np.clip(bev_array[0], 0, 1) * 255).astype(np.uint8)
    viz[:, :, 0], viz[:, :, 1], viz[:, :, 2] = explore, explore, explore

    # Channel 1: Obstacle (red overlay)
    obstacle = (np.clip(bev_array[1], 0, 1) * 255).astype(np.uint8)
    viz[:, :, 0] = np.maximum(viz[:, :, 0], obstacle)

    # Channel 2: Value (green overlay)
    if c >= 3:
        value = bev_array[2]
        value = (np.clip(value, 0, 1) * 255).astype(np.uint8)
        viz[:, :, 1] = np.maximum(viz[:, :, 1], value)

    return viz


def extract_bev_data(bev_tensor):
    """Extract first batch, first sequence as (3, H, W) numpy array."""
    if isinstance(bev_tensor, torch.Tensor):
        arr = bev_tensor.detach().cpu().numpy()
    else:
        arr = np.array(bev_tensor)

    if len(arr.shape) == 5:  # (b, s, 3, h, w)
        arr = arr[0, 0]
    elif len(arr.shape) == 4:  # (b, 3, h, w)
        arr = arr[0]
    return arr


def compute_bev_metrics(pred_bev, gt_bev):
    """Compute quantitative BEV metrics.

    - exploration / obstacle: IoU (binary at 0.5 threshold)
    - value: MAE (channels are in [0, 1] via sigmoid)
    """
    if isinstance(pred_bev, torch.Tensor):
        pred_bev = pred_bev.detach().cpu().numpy()
    if isinstance(gt_bev, torch.Tensor):
        gt_bev = gt_bev.detach().cpu().numpy()

    def iou(p, t):
        p_bin = (p > 0.5).astype(np.bool_)
        t_bin = (t > 0.5).astype(np.bool_)
        inter = np.logical_and(p_bin, t_bin).sum()
        union = np.logical_or(p_bin, t_bin).sum()
        return float(inter) / float(union) if union > 0 else float('nan')

    return {
        'exploration_iou': iou(pred_bev[0], gt_bev[0]),
        'obstacle_iou': iou(pred_bev[1], gt_bev[1]),
        'value_mae': float(np.mean(np.abs(pred_bev[2] - gt_bev[2]))),
    }


def create_comparison_image(pred_bev, gt_bev, step_idx, output_dir, action=None):
    """Create per-channel comparison of pred vs GT BEV, with action info.

    Layout: one row per BEV channel (exploration / obstacle / value), each row
    shows [Prediction | Ground Truth | Diff x3]. The title line carries the
    discrete action label.
    """
    os.makedirs(output_dir, exist_ok=True)

    if isinstance(pred_bev, torch.Tensor):
        pred_bev = pred_bev.detach().cpu().numpy()
    if isinstance(gt_bev, torch.Tensor):
        gt_bev = gt_bev.detach().cpu().numpy()

    c, h, w = pred_bev.shape

    # Convert each channel to a colored RGB image (per-channel separation).
    pred_viz = [channel_to_rgb(pred_bev[i], CHANNEL_COLORS[i]) for i in range(c)]
    gt_viz = [channel_to_rgb(gt_bev[i], CHANNEL_COLORS[i]) for i in range(c)]

    # Layout metrics
    title_h = 36
    header_h = 24
    row_label_w = 96
    n_cols = 3  # pred | gt | diff
    gap = 8

    canvas_w = row_label_w + n_cols * w + (n_cols + 1) * gap
    canvas_h = title_h + header_h + c * (h + gap) + 20

    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 240

    # 1) Fill the BEV image tiles into the numpy canvas FIRST.
    col_x0 = row_label_w
    for i in range(c):
        row_y = title_h + header_h + i * (h + gap)
        for j in range(n_cols):
            x = col_x0 + j * (w + gap)
            if j == 0:
                img = pred_viz[i]
            elif j == 1:
                img = gt_viz[i]
            else:
                # diff between pred and GT (grayscale, amplified x3)
                diff = np.abs(pred_bev[i].astype(float) - gt_bev[i].astype(float))
                diff = np.clip(diff * 3, 0, 1) * 255
                img = np.repeat(diff.astype(np.uint8)[..., None], 3, axis=-1)
            canvas[row_y:row_y + h, x:x + w] = img

    # 2) Convert to PIL only AFTER all numpy writes, then draw text on top.
    canvas_pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(canvas_pil)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
        font_small = ImageFont.truetype("DejaVuSans.ttf", 12)
    except Exception:
        font = ImageFont.load_default()
        font_small = font

    # Title line with action info
    title = f"BEV Pred vs GT (step {step_idx})"
    if action is not None:
        title += f"   |   action: {format_action(action)}"
    draw.text((10, 8), title, fill=(0, 0, 0), font=font)

    # Column headers
    headers = ['Prediction', 'Ground Truth', 'Diff x3']
    for j, name in enumerate(headers):
        x = col_x0 + j * (w + gap)
        draw.text((x + w // 2 - 40, title_h + 2), name, fill=(0, 0, 0), font=font_small)

    # Channel labels on the left
    for i in range(c):
        row_y = title_h + header_h + i * (h + gap)
        draw.text((8, row_y + h // 2 - 8), CHANNEL_NAMES[i],
                  fill=(0, 0, 0), font=font_small)

    save_path = os.path.join(output_dir, f"bev_compare_step{step_idx}.png")
    canvas_pil.save(save_path)
    print(f"  Saved: {save_path}")

    return canvas


def main():
    project_dir = Path(__file__).parent.parent
    checkpoint_path = '/shared_disk/users/wenhao.lu/JYT/WorldSearch_data/experiments/gwm_pretrain_with_bev_v3/checkpoints/epoch=90-step=2730.ckpt'
    output_dir = '/shared_disk/users/wenhao.lu/JYT/WorldSearch_data/experiments/gwm_pretrain_with_bev_v3/test_outputs'
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("BEV Prediction vs Ground Truth Comparison")
    print("=" * 60)

    # 1. Load config
    print("\n[1] Loading config...")
    gin.clear_config()

    # Import model classes FIRST so gin can bind to them
    from model.trainer import SearchWorldTrainer  # noqa - registers all sub-modules

    # Now parse config - all symbols are registered
    base_config = project_dir / 'configs' / 'base_train_config.gin'
    parquet_config = project_dir / 'configs' / 'gwm_pretrain_parquet_config.gin'
    gin.parse_config_file(str(base_config), skip_unknown=True)
    gin.parse_config_file(str(parquet_config), skip_unknown=True)

    # Use GPU if available (DINOv2+SigLIP2 inference on CPU is very slow)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Using device: {device}")

    # 2. Load model
    print("\n[2] Loading model...")
    model = SearchWorldTrainer.load_from_checkpoint(checkpoint_path, map_location=device, strict=False)
    model = model.to(device)
    model.eval()
    print("  Model loaded!")

    # 3. Test with real BEV data
    print("\n[3] Loading real data with BEV GT...")
    from model.dataset.uav_parquet_dataset import UAVParquetDataset

    data_dir = '/shared_disk/users/wenhao.lu/JYT/WorldSearch_data/BrushifyUrban_parquet_with_bev'
    dataset = UAVParquetDataset(
        dataset_path=os.path.join(data_dir, 'test'),
        sequence_length=4,
        enable_semantic=True,
        enable_rgb_stylegan=True,
        is_gwm_pretrain=True,
        enable_bev=True
    )
    print(f"  Dataset size: {len(dataset)}")

    # 4. Compare on samples
    print("\n[4] Running comparison...")
    num_samples = min(8, len(dataset))

    # Aggregate metrics across all steps / samples
    agg = {'exploration_iou': [], 'obstacle_iou': [], 'value_mae': []}
    evaluated_steps = 0

    for i in range(num_samples):
        # Use different steps within the sample to see variation
        sid = i * 10 % len(dataset)
        sample = dataset[sid]

        # Add batch dimension
        batch = {}
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v[:4].unsqueeze(0)  # use first 4 steps, add batch dim
        batch = {k: v.to(device) for k, v in batch.items()}

        # Run inference
        with torch.no_grad():
            output = model.model(batch)

        if 'bev_pred' not in output or 'bev_gt' not in batch:
            print(f"  Sample {i}: missing BEV data, skipping")
            continue

        # Compare all 4 sequence steps
        for s in range(min(4, output['bev_pred'].shape[1])):
            pred_bev = output['bev_pred'][0, s]  # (3, 256, 256)
            gt_bev = batch['bev_gt'][0, s]       # (3, 256, 256)

            # Extract the discrete action label for this step (if present).
            action = None
            if 'action' in batch:
                action = batch['action'][0, s]

            m = compute_bev_metrics(pred_bev, gt_bev)
            for k in agg:
                agg[k].append(m[k])
            evaluated_steps += 1

            create_comparison_image(pred_bev, gt_bev, f"{i}_seq{s}", output_dir,
                                    action=action)

        # Also create all-steps grid
        all_pred = output['bev_pred'][0].cpu().numpy()   # (4, 3, 256, 256)
        all_gt = batch['bev_gt'][0].cpu().numpy()        # (4, 3, 256, 256)

        # Grid: 2 rows (pred, GT) x 4 cols (steps)
        h, w = 256, 256
        top_h = 30
        grid = np.ones((top_h + h * 2 + 30, w * 4 + 30, 3), dtype=np.uint8) * 240

        # 1) Fill BEV tiles into the numpy grid FIRST.
        for s in range(4):
            x = 10 + s * (w + 5)
            grid[top_h:top_h+h, x:x+w] = denormalize_bev(all_pred[s])
            grid[top_h+h+10:top_h+h*2+10, x:x+w] = denormalize_bev(all_gt[s])

        # 2) Convert to PIL only AFTER all numpy writes, then draw labels.
        grid_pil = Image.fromarray(grid)
        draw = ImageDraw.Draw(grid_pil)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 12)
        except Exception:
            font = ImageFont.load_default()

        draw.text((10, 5), "PRED", fill=(0, 0, 0), font=font)
        for s in range(4):
            x = 10 + s * (w + 5)
            # Action label above each step column
            if 'action' in batch:
                action = batch['action'][0, s]
                draw.text((x + w // 2 - 30, 5), format_action(action),
                          fill=(0, 0, 0), font=font)

        draw.text((10, top_h + h + 8), "GT", fill=(0, 0, 0), font=font)
        grid_pil.save(os.path.join(output_dir, f"bev_grid_sample{i}.png"))
        print(f"  Saved grid: bev_grid_sample{i}.png")

    # Print aggregated metrics
    print(f"\n{'='*60}")
    print(f"Aggregated BEV metrics over {evaluated_steps} steps:")
    for k, vals in agg.items():
        vals = [v for v in vals if v == v]  # drop NaN
        if vals:
            print(f"  {k:20s}: mean={sum(vals)/len(vals):.4f}  "
                  f"min={min(vals):.4f}  max={max(vals):.4f}")
    print(f"{'='*60}")
    print(f"Done! All results saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
