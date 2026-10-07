# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Decode and visualize RGB / BEV / Semantic reconstructions from a GWM checkpoint.

Loads a GWM pretrain checkpoint (gwm_pretrain_parquet_config.gin) and runs a
forward pass on a few samples from the val/test parquet dataset.  For every
sample it saves, side-by-side with the ground truth:

  - RGB reconstruction          (rgb_1  vs  input image)
  - BEV map reconstruction      (bev_pred vs bev_gt, per channel)
  - Semantic segmentation       (semantic_segmentation_1 vs semantic_label_1)

and prints quantitative metrics (RGB PSNR, BEV IoU/MAE, semantic mIoU).

Run INSIDE the SearchWorld docker container (see README comment at bottom):

    cd /workspace
    export HF_ENDPOINT=https://hf-mirror.com
    python scripts/decode_rgb_bev.py
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# Defaults (mounted paths inside the container).
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).parent.parent
DEFAULT_CHECKPOINT = (
    "/shared_disk/users/wenhao.lu/JYT/WorldSearch_data/experiments/"
    "gwm_pretrain_with_bev_v3/checkpoints/last.ckpt"
)
DEFAULT_DATA_DIR = (
    "/shared_disk/users/wenhao.lu/JYT/WorldSearch_data/BrushifyUrban_parquet_with_bev"
)
DEFAULT_OUTPUT_DIR = (
    "/shared_disk/users/wenhao.lu/JYT/WorldSearch_data/experiments/"
    "gwm_pretrain_with_bev_v3/decode_outputs"
)

# Discrete action label -> human-readable name.
ACTION_NAMES = [
    'forward', 'left', 'right', 'ascend', 'descend', 'rotl', 'rotr', 'stop',
]

# Semantic colors matching isaac_sim_semantic_label.SEMANTIC_COLORS (first 6).
SEMANTIC_COLORS = np.array(
    [
        [128, 128, 128],  # 0 BACKGROUND
        [0, 255, 0],      # 1 NAVIGABLE
        [255, 165, 0],    # 2 FORKLIFT
        [0, 0,255],       # 3 PALLET
        [255, 255, 0],    # 4 CONE
        [255, 0, 255],    # 5 SIGN
    ],
    dtype=np.uint8,
)
SEMANTIC_NAMES = ['BACKGROUND', 'NAVIGABLE', 'FORKLIFT', 'PALLET', 'CONE', 'SIGN']

BEV_CHANNEL_NAMES = ['exploration', 'obstacle', 'value']


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _font(size=14):
    for name in ('DejaVuSans.ttf', 'DejaVuSans-Bold.ttf'):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def format_action(action):
    if action is None:
        return 'unknown'
    if isinstance(action, torch.Tensor):
        action = int(action.item())
    name = ACTION_NAMES[int(action)] if 0 <= int(action) < len(ACTION_NAMES) else '?'
    return f"{int(action)} ({name})"


def to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def rgb_to_uint8(rgb):
    """(3, H, W) float tensor/array in ~[0,1] -> (H, W, 3) uint8."""
    arr = to_np(rgb)
    if arr.shape[0] == 3:
        arr = arr.transpose(1, 2, 0)
    return np.clip(np.round(arr * 255.0), 0, 255).astype(np.uint8)


def bev_to_uint8(bev):
    """(3, H, W) BEV in [0,1] -> (H, W, 3) uint8 overlay viz."""
    arr = to_np(bev)
    h, w = arr.shape[1], arr.shape[2]
    viz = np.zeros((h, w, 3), dtype=np.uint8)
    explore = np.clip(arr[0], 0, 1) * 255
    viz[:, :, 0] = viz[:, :, 1] = viz[:, :, 2] = explore
    obstacle = np.clip(arr[1], 0, 1) * 255
    viz[:, :, 0] = np.maximum(viz[:, :, 0], obstacle)
    if arr.shape[0] >= 3:
        value = arr[2]
        if value.min() < 0:
            value = (value + 1) / 2
        viz[:, :, 1] = np.maximum(viz[:, :, 1], np.clip(value, 0, 1) * 255)
    return viz.astype(np.uint8)


def bev_channel_to_uint8(channel, color):
    """Single [0,1] channel -> colored (H, W, 3) uint8."""
    arr = to_np(channel)
    ch = np.clip(arr, 0, 1)[..., None]
    return (ch * np.array(color, dtype=np.float32)).astype(np.uint8)


def semantic_to_uint8(logits):
    """(C, H, W) logits -> (H, W, 3) uint8 colored argmax."""
    arr = to_np(logits)
    pred = arr.argmax(axis=0)
    n = SEMANTIC_COLORS.shape[0]
    pred = np.clip(pred, 0, n - 1)
    return SEMANTIC_COLORS[pred]


def semantic_label_to_uint8(label):
    """(H, W) int label -> (H, W, 3) uint8 colored."""
    arr = to_np(label).astype(np.int64)
    n = SEMANTIC_COLORS.shape[0]
    arr = np.clip(arr, 0, n - 1)
    return SEMANTIC_COLORS[arr]


def hstack_images(imgs, gap=4, pad_color=(255, 255, 255)):
    imgs = [np.asarray(im) for im in imgs]
    h = max(im.shape[0] for im in imgs)
    w = sum(im.shape[1] for im in imgs) + gap * (len(imgs) - 1)
    canvas = np.ones((h, w, 3), dtype=np.uint8) * np.array(pad_color, dtype=np.uint8)
    x = 0
    for im in imgs:
        ih, iw = im.shape[:2]
        y = (h - ih) // 2
        canvas[y:y + ih, x:x + iw] = im
        x += iw + gap
    return canvas


def _draw_labels(canvas, labels, tile_w, tile_h, y_off=0, gap=4):
    """Draw a text label above each tile (tiles laid out horizontally)."""
    pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil)
    f = _font(12)
    for i, lab in enumerate(labels):
        x = i * (tile_w + gap)
        draw.text((x + 4, y_off + 2), str(lab), fill=(0, 0, 0), font=f)
    return np.asarray(pil)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def rgb_psnr(pred, gt):
    p = to_np(pred).astype(np.float32)
    g = to_np(gt).astype(np.float32)
    p = np.clip(p, 0, 1)
    mse = np.mean((p - g) ** 2)
    if mse == 0:
        return float('inf')
    return float(10 * np.log10(1.0 / mse))


def bev_iou(p, t, thresh=0.5):
    p = (to_np(p) > thresh)
    t = (to_np(t) > thresh)
    inter = np.logical_and(p, t).sum()
    union = np.logical_or(p, t).sum()
    return float(inter / union) if union > 0 else float('nan')


def semantic_iou(pred_logits, gt_label, n_classes=None):
    pred = to_np(pred_logits).argmax(axis=0)
    gt = to_np(gt_label).astype(np.int64)
    n_classes = n_classes or SEMANTIC_COLORS.shape[0]
    ious = {}
    for c in range(n_classes):
        p = pred == c
        g = gt == c
        inter = np.logical_and(p, g).sum()
        union = np.logical_or(p, g).sum()
        ious[c] = float(inter / union) if union > 0 else float('nan')
    return ious


# ---------------------------------------------------------------------------
# Model / data loading
# ---------------------------------------------------------------------------
def load_model(checkpoint_path, config_paths, device):
    import gin
    gin.clear_config()

    # Import registers the gin-configurable classes before parsing config.
    from model.trainer import SearchWorldTrainer  # noqa: F401

    for cfg in config_paths:
        gin.parse_config_file(cfg, skip_unknown=True)

    model = SearchWorldTrainer.load_from_checkpoint(
        checkpoint_path, map_location=device, strict=False)
    model = model.to(device)
    model.eval()
    if hasattr(model, 'model'):
        model.model.eval()
    return model


def load_dataset(data_dir, split, device):
    from model.dataset.uav_parquet_dataset import UAVParquetDataset

    dataset = UAVParquetDataset(
        dataset_path=os.path.join(data_dir, split),
        sequence_length=4,
        enable_semantic=True,
        enable_rgb_stylegan=True,
        is_gwm_pretrain=True,
        enable_bev=True,
    )
    return dataset


def make_batch(sample, device):
    batch = {}
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v[:4].unsqueeze(0)  # first 4 steps + batch dim
    return {k: v.to(device) for k, v in batch.items()}


# ---------------------------------------------------------------------------
# Per-sample rendering
# ---------------------------------------------------------------------------
def render_rgb_grid(images, preds, actions, sample_idx, out_dir):
    """images/preds: list of (3,H,W) arrays over seq steps."""
    h, w = images[0].shape[-2:]
    top = 28
    gap = 6
    n = len(images)
    canvas = np.ones((top + 2 * h + 2 * gap, n * w + (n + 1) * gap, 3),
                     dtype=np.uint8) * 240
    for s in range(n):
        x = gap + s * (w + gap)
        canvas[top:top + h, x:x + w] = rgb_to_uint8(images[s])
        canvas[top + h + gap:top + 2 * h + gap, x:x + w] = rgb_to_uint8(preds[s])
    pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil)
    f = _font(13)
    draw.text((8, 4), f"RGB  GT (top) / Pred (bottom)  sample {sample_idx}",
              fill=(0, 0, 0), font=f)
    for s in range(n):
        x = gap + s * (w + gap)
        draw.text((x + 4, top - 20), format_action(actions[s]), fill=(0, 0, 0), font=_font(11))
    out = os.path.join(out_dir, f"rgb_sample{sample_idx}.png")
    pil.save(out)
    return out


def render_bev_grid(bev_preds, bev_gts, actions, sample_idx, out_dir):
    """BEV 3-channel comparison: rows = channels, cols = [pred, gt]."""
    n = len(bev_preds)
    # Use single-step per-channel layout for the whole sequence:
    # For each channel: one row with n preds and n gts is too wide; instead
    # produce one image per channel with 2 rows (pred/gt) x n steps.
    outs = []
    for c in range(3):
        h, w = bev_preds[0].shape[-2:]
        top = 28
        gap = 6
        canvas = np.ones((top + 2 * h + 2 * gap, n * w + (n + 1) * gap, 3),
                         dtype=np.uint8) * 240
        color = [(255, 255, 255), (255, 0, 0), (0, 255, 0)][c]
        for s in range(n):
            x = gap + s * (w + gap)
            p = bev_channel_to_uint8(to_np(bev_preds[s])[c], color)
            g = bev_channel_to_uint8(to_np(bev_gts[s])[c], color)
            canvas[top:top + h, x:x + w] = p
            canvas[top + h + gap:top + 2 * h + gap, x:x + w] = g
        pil = Image.fromarray(canvas)
        draw = ImageDraw.Draw(pil)
        f = _font(13)
        draw.text((8, 4), f"BEV {BEV_CHANNEL_NAMES[c]}  Pred(top)/GT(bottom)  sample {sample_idx}",
                  fill=(0, 0, 0), font=f)
        for s in range(n):
            x = gap + s * (w + gap)
            draw.text((x + 4, top - 20), format_action(actions[s]), fill=(0, 0, 0), font=_font(11))
        out = os.path.join(out_dir, f"bev_{BEV_CHANNEL_NAMES[c]}_sample{sample_idx}.png")
        pil.save(out)
        outs.append(out)
    return outs


def render_bev_overlay_grid(bev_preds, bev_gts, actions, sample_idx, out_dir):
    """Overlay (3-channel combined) BEV: 2 rows (pred/gt) x n steps."""
    n = len(bev_preds)
    h, w = bev_preds[0].shape[-2:]
    top = 28
    gap = 6
    canvas = np.ones((top + 2 * h + 2 * gap, n * w + (n + 1) * gap, 3),
                     dtype=np.uint8) * 240
    for s in range(n):
        x = gap + s * (w + gap)
        canvas[top:top + h, x:x + w] = bev_to_uint8(bev_preds[s])
        canvas[top + h + gap:top + 2 * h + gap, x:x + w] = bev_to_uint8(bev_gts[s])
    pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil)
    draw.text((8, 4), f"BEV overlay  Pred(top)/GT(bottom)  sample {sample_idx}",
              fill=(0, 0, 0), font=_font(13))
    for s in range(n):
        x = gap + s * (w + gap)
        draw.text((x + 4, top - 20), format_action(actions[s]), fill=(0, 0, 0), font=_font(11))
    out = os.path.join(out_dir, f"bev_overlay_sample{sample_idx}.png")
    pil.save(out)
    return out


def render_semantic_grid(pred_logits, gt_labels, actions, sample_idx, out_dir):
    n = len(pred_logits)
    h, w = pred_logits[0].shape[-2:]
    top = 28
    gap = 6
    canvas = np.ones((top + 2 * h + 2 * gap, n * w + (n + 1) * gap, 3),
                     dtype=np.uint8) * 240
    for s in range(n):
        x = gap + s * (w + gap)
        canvas[top:top + h, x:x + w] = semantic_to_uint8(pred_logits[s])
        canvas[top + h + gap:top + 2 * h + gap, x:x + w] = semantic_label_to_uint8(gt_labels[s])
    pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil)
    draw.text((8, 4), f"Semantic  Pred(top)/GT(bottom)  sample {sample_idx}",
              fill=(0, 0, 0), font=_font(13))
    for s in range(n):
        x = gap + s * (w + gap)
        draw.text((x + 4, top - 20), format_action(actions[s]), fill=(0, 0, 0), font=_font(11))
    out = os.path.join(out_dir, f"semantic_sample{sample_idx}.png")
    pil.save(out)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Decode RGB/BEV/Semantic from GWM checkpoint")
    parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT)
    parser.add_argument('--data-dir', default=DEFAULT_DATA_DIR)
    parser.add_argument('--split', default='val', choices=['val', 'test', 'train'])
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--num-samples', type=int, default=8)
    parser.add_argument('--device', default=None, help="cuda / cpu (default: cuda if available)")
    parser.add_argument('--stride', type=int, default=1,
                        help="sample index stride to spread across the split")
    args = parser.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("SearchWorld GWM - RGB / BEV / Semantic decoding")
    print("=" * 70)
    print(f"  checkpoint : {args.checkpoint}")
    print(f"  data       : {args.data_dir}/{args.split}")
    print(f"  output     : {args.output_dir}")
    print(f"  device     : {device}")
    print(f"  samples    : {args.num_samples}")

    # 1. Load model
    print("\n[1] Loading model ...")
    config_paths = [
        str(PROJECT_DIR / 'configs' / 'base_train_config.gin'),
        str(PROJECT_DIR / 'configs' / 'gwm_pretrain_parquet_config.gin'),
    ]
    model = load_model(args.checkpoint, config_paths, device)
    xmob = model.model
    print(f"  enable_semantic      : {getattr(xmob, 'enable_semantic', None)}")
    print(f"  enable_rgb_stylegan  : {getattr(xmob, 'enable_rgb_stylegan', None)}")
    print(f"  enable_rgb_diffusion : {getattr(xmob, 'enable_rgb_diffusion', None)}")
    print(f"  enable_bev_decoder   : {getattr(xmob, 'enable_bev_decoder', None)}")

    # 2. Load dataset
    print(f"\n[2] Loading dataset ({args.split}) ...")
    dataset = load_dataset(args.data_dir, args.split, device)
    print(f"  dataset size : {len(dataset)}")
    n_samples = min(args.num_samples, len(dataset))

    # 3. Run decoding
    print(f"\n[3] Running decoding on {n_samples} samples ...")
    agg = {
        'rgb_psnr': [],
        'bev_exploration_iou': [],
        'bev_obstacle_iou': [],
        'bev_value_mae': [],
        'semantic_iou': {c: [] for c in range(SEMANTIC_COLORS.shape[0])},
    }

    for i in range(n_samples):
        idx = (i * args.stride) % len(dataset)
        sample = dataset[idx]
        batch = make_batch(sample, device)

        with torch.no_grad():
            output = xmob(batch)

        # --- extract per-step tensors ---
        # RGB
        images = [batch['image'][0, s].cpu().numpy() for s in range(batch['image'].shape[1])]
        rgb_preds = [output['rgb_1'][0, s].cpu().numpy() for s in range(output['rgb_1'].shape[1])]
        actions = [batch['action'][0, s] for s in range(batch['action'].shape[1])]

        # BEV
        bev_preds = [output['bev_pred'][0, s].cpu().numpy() for s in range(output['bev_pred'].shape[1])]
        bev_gts = [batch['bev_gt'][0, s].cpu().numpy() for s in range(batch['bev_gt'].shape[1])]

        # Semantic
        sem_preds = [output['semantic_segmentation_1'][0, s].cpu().numpy()
                     for s in range(output['semantic_segmentation_1'].shape[1])]
        sem_gts = [batch['semantic_label_1'][0, s].cpu().numpy()
                   for s in range(batch['semantic_label_1'].shape[1])]

        # --- render ---
        render_rgb_grid(images, rgb_preds, actions, i, args.output_dir)
        render_bev_overlay_grid(bev_preds, bev_gts, actions, i, args.output_dir)
        render_bev_grid(bev_preds, bev_gts, actions, i, args.output_dir)
        render_semantic_grid(sem_preds, sem_gts, actions, i, args.output_dir)

        # --- metrics ---
        for s in range(len(images)):
            agg['rgb_psnr'].append(rgb_psnr(rgb_preds[s], images[s]))
            agg['bev_exploration_iou'].append(bev_iou(bev_preds[s][0], bev_gts[s][0]))
            agg['bev_obstacle_iou'].append(bev_iou(bev_preds[s][1], bev_gts[s][1]))
            agg['bev_value_mae'].append(
                float(np.mean(np.abs(bev_preds[s][2] - bev_gts[s][2]))))
            siou = semantic_iou(sem_preds[s], sem_gts[s])
            for c, v in siou.items():
                agg['semantic_iou'][c].append(v)

        print(f"  sample {i:2d} (idx {idx}): rgb_psnr={agg['rgb_psnr'][-1]:.2f}")

    # 4. Summary
    print("\n" + "=" * 70)
    print("Aggregated metrics")
    print("=" * 70)

    def _mean(vals):
        vals = [v for v in vals if v == v and np.isfinite(v)]
        return sum(vals) / len(vals) if vals else float('nan')

    summary = {}
    summary['rgb_psnr'] = _mean(agg['rgb_psnr'])
    summary['bev_exploration_iou'] = _mean(agg['bev_exploration_iou'])
    summary['bev_obstacle_iou'] = _mean(agg['bev_obstacle_iou'])
    summary['bev_value_mae'] = _mean(agg['bev_value_mae'])
    for c in range(SEMANTIC_COLORS.shape[0]):
        summary[f'semantic_iou_{SEMANTIC_NAMES[c]}'] = _mean(agg['semantic_iou'][c])

    for k, v in summary.items():
        print(f"  {k:26s}: {v:.4f}")

    with open(os.path.join(args.output_dir, 'metrics_summary.txt'), 'w') as f:
        for k, v in summary.items():
            f.write(f"{k}: {v:.4f}\n")

    print(f"\nDone! Outputs saved to: {args.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# How to run (inside the SearchWorld container):
#
#   # on the host, if the container is not running:
#   docker start adoring_lamarr
#   docker exec -it adoring_lamarr bash
#
#   # inside the container:
#   cd /workspace
#   export HF_ENDPOINT=https://hf-mirror.com
#   export TOKENIZERS_PARALLELISM=false
#   python scripts/decode_rgb_bev.py \
#       --checkpoint /shared_disk/users/wenhao.lu/JYT/WorldSearch_data/experiments/gwm_pretrain_with_bev_v3/checkpoints/last.ckpt \
#       --split val \
#       --num-samples 8
# ---------------------------------------------------------------------------
