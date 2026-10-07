# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Interactive BEV imagination: input an action, the world model outputs a predicted BEV.

This script demonstrates the SearchWorld world model in "imagination" mode:

  1. Load a real observation (a single frame) and encode it into an initial
     RSSM posterior state (h_0, sample_0).
  2. The user inputs a discrete action (forward / left / right / ascend /
     descend / rotl / rotr / stop).
  3. The RSSM advances the latent state using its *prior* via ``imagine_step``
     (no new observation is needed).
  4. The BEV decoder decodes the new state into a 3-channel BEV map
     (exploration / obstacle / value), which is rendered and saved in real time.

Usage:
    # Interactive mode (default)
    python scripts/interactive_bev_imagination.py

    # One-shot: roll out a fixed action sequence of length N
    python scripts/interactive_bev_imagination.py --actions 1 3 2 5 7 --sample 0

    # Deterministic imagination (use prior mean instead of sampling)
    python scripts/interactive_bev_imagination.py --no-sample
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Set offline env vars early so the DINOv2 / SigLIP2 encoders use cached weights.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, str(Path(__file__).parent.parent))

import gin  # noqa: E402


# Discrete action label -> human-readable name (matches convert_uav_to_parquet.py).
ACTION_NAMES = [
    'forward', 'left', 'right', 'ascend', 'descend', 'rotl', 'rotr', 'stop',
]

# BEV channel names and their visualization colors (RGB).
CHANNEL_NAMES = ['exploration', 'obstacle', 'value']
CHANNEL_COLORS = [
    (0, 255, 0),    # exploration: green
    (255, 0, 0),    # obstacle: red
    (0, 0, 255),    # value: blue
]

CHECKPOINT_PATH = (
    '/shared_disk/users/wenhao.lu/JYT/WorldSearch_data/experiments/'
    'gwm_pretrain_with_bev_v3/checkpoints/epoch=90-step=2730.ckpt'
)
DATA_DIR = (
    '/shared_disk/users/wenhao.lu/JYT/WorldSearch_data/BrushifyUrban_parquet_with_bev'
)
OUTPUT_DIR = (
    '/shared_disk/users/wenhao.lu/JYT/WorldSearch_data/experiments/'
    'gwm_pretrain_with_bev_v3/imagination_outputs'
)


def parse_action(text):
    """Parse a user action from a name or integer index."""
    text = text.strip().lower()
    if text in ('q', 'quit', 'exit'):
        return None
    if text.isdigit():
        idx = int(text)
        if 0 <= idx < len(ACTION_NAMES):
            return idx
        print(f"  [!] Action index out of range (0-{len(ACTION_NAMES) - 1}).")
        return None
    if text in ACTION_NAMES:
        return ACTION_NAMES.index(text)
    print(f"  [!] Unknown action '{text}'. Valid: {ACTION_NAMES}")
    return None


def format_action(idx):
    return f"{idx} ({ACTION_NAMES[idx]})"


def bev_to_rgb(bev):
    """Convert a (3, H, W) BEV tensor/array in [0, 1] to a colored RGB uint8 image.

    exploration -> green, obstacle -> red, value -> blue, additive overlay.
    """
    if isinstance(bev, torch.Tensor):
        bev = bev.detach().cpu().numpy()
    c, h, w = bev.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    for i in range(min(c, 3)):
        ch = np.clip(bev[i], 0, 1)          # (h, w) in [0, 1]
        rgb += ch[..., None] * np.array(CHANNEL_COLORS[i], dtype=np.float32)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb


def render_pair(pred_bev, gt_bev, step_idx, out_dir, action=None):
    """Render prediction + optional GT side by side and save a PNG."""
    pred_rgb = bev_to_rgb(pred_bev)
    gt_rgb = bev_to_rgb(gt_bev) if gt_bev is not None else None

    label_h = 24
    pad = 8
    h, w = pred_rgb.shape[:2]

    n_cols = 2 if gt_rgb is not None else 1
    canvas_w = n_cols * w + (n_cols + 1) * pad
    canvas_h = label_h + h + 2 * pad
    canvas = np.full((canvas_h, canvas_w, 3), 240, dtype=np.uint8)

    canvas[label_h:label_h + h, pad:pad + w] = pred_rgb
    if gt_rgb is not None:
        canvas[label_h:label_h + h, 2 * pad + w:2 * pad + 2 * w] = gt_rgb

    img = Image.fromarray(canvas)
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    title = f"BEV imagination step {step_idx}"
    if action is not None:
        title += f"   |   action: {format_action(action)}"
    draw.text((pad, 3), title, fill=(0, 0, 0))
    draw.text((pad, label_h - 14), "prediction", fill=(0, 0, 0))
    if gt_rgb is not None:
        draw.text((2 * pad + w, label_h - 14), "ground truth", fill=(0, 0, 0))

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"bev_imagine_step{step_idx}.png")
    img.save(path)
    print(f"  Saved: {path}")
    return img


def load_model(device, checkpoint_path):
    from model.trainer import SearchWorldTrainer

    print("[1] Loading config...")
    gin.clear_config()
    # Import model classes first so gin can bind to them.
    from model.trainer import SearchWorldTrainer as _  # noqa: F401
    base_config = Path(__file__).parent.parent / 'configs' / 'base_train_config.gin'
    parquet_config = Path(__file__).parent.parent / 'configs' / 'gwm_pretrain_parquet_config.gin'
    gin.parse_config_file(str(base_config), skip_unknown=True)
    gin.parse_config_file(str(parquet_config), skip_unknown=True)

    print("[2] Loading model...")
    model = SearchWorldTrainer.load_from_checkpoint(
        checkpoint_path, map_location=device, strict=False)
    model = model.to(device).eval()
    print("  Model loaded!")
    return model


def encode_initial_state(model, dataset, sample_idx, device):
    """Encode a single real frame into an initial RSSM posterior state.

    Returns (h_t, sample_t) and the sample's full action / GT-BEV sequences so
    that we can optionally compare imagination against real rollouts.
    """
    sample = dataset[sample_idx]
    seq_len = sample['image'].shape[0]

    # Build a single-frame batch (s=1) for the observation encoder.
    batch = {}
    for k in ['image', 'relative_pose', 'text_feat']:
        batch[k] = sample[k][:1].unsqueeze(0).to(device)  # (1, 1, ...)
    # BEV encoder needs bev_memory when enable_bev=True.
    if 'bev_memory' in sample:
        batch['bev_memory'] = sample['bev_memory'][:1].unsqueeze(0).to(device)
    elif 'bev_gt' in sample:
        batch['bev_memory'] = sample['bev_gt'][:1].unsqueeze(0).to(device)

    rssm = model.model.rssm
    obs_encoder = model.model.observation_encoder

    with torch.no_grad():
        obs_dict = obs_encoder(batch)
        embedding_0 = obs_dict['embedding'][:, 0]  # (1, embedding_dim)

        h_t = embedding_0.new_zeros((1, rssm.hidden_state_dim))
        sample_t = embedding_0.new_zeros((1, rssm.state_dim))
        action_0 = embedding_0.new_full((1,), 7, dtype=torch.long)  # sentinel stop
        out = rssm.observe_step(h_t, sample_t, action_0, embedding_0,
                                use_sample=False)
        posterior = out['posterior']
        h_t = posterior['hidden_state']
        sample_t = posterior['sample']

    # Keep GT sequences for optional comparison.
    gt_bevs = sample.get('bev_gt')            # (seq_len, 3, 256, 256)
    gt_actions = sample.get('action')         # (seq_len,)

    print(f"  Initial state encoded from sample {sample_idx} (sequence {seq_len} frames).")
    print(f"  GT action sequence: {gt_actions.tolist()}")
    print(f"  GT action names   : {[ACTION_NAMES[a] for a in gt_actions.tolist()]}")
    return h_t, sample_t, gt_bevs, gt_actions, seq_len


def imagine_step(model, h_t, sample_t, action_idx, use_sample):
    """Advance latent state with a discrete action and decode BEV."""
    rssm = model.model.rssm
    bev_decoder = model.model.bev_decoder

    action_t = h_t.new_tensor([action_idx], dtype=torch.long)  # (1,)
    with torch.no_grad():
        prior = rssm.imagine_step(h_t, sample_t, action_t, use_sample=use_sample)
        h_new = prior['hidden_state']
        sample_new = prior['sample']
        state = torch.cat([h_new, sample_new], dim=-1)  # (1, state_dim)
        bev = bev_decoder(state)                        # (1, 3, 256, 256)
    return h_new, sample_new, bev[0]


def print_bev_stats(bev):
    if isinstance(bev, torch.Tensor):
        bev = bev.detach().cpu().numpy()
    for c in range(3):
        ch = bev[c]
        print(f"    ch{c} ({CHANNEL_NAMES[c]:11s}): "
              f"min={ch.min():.3f} max={ch.max():.3f} mean={ch.mean():.3f}")


def run_interactive(model, h_t, sample_t, gt_bevs, gt_actions, seq_len,
                    use_sample, out_dir):
    print("\n" + "=" * 60)
    print("Interactive BEV imagination")
    print("  Input an action to imagine the next BEV.")
    print("  Actions: " + ", ".join(f"{i}={n}" for i, n in enumerate(ACTION_NAMES)))
    print("  Type 'q' to quit.")
    print("=" * 60)

    step = 0
    while True:
        raw = input(f"\n[step {step}] action > ")
        action_idx = parse_action(raw)
        if action_idx is None:
            if raw.strip().lower() in ('q', 'quit', 'exit'):
                print("Bye.")
                break
            continue

        h_t, sample_t, bev = imagine_step(model, h_t, sample_t, action_idx,
                                          use_sample)
        step += 1
        print(f"  -> action {format_action(action_idx)}")
        print_bev_stats(bev)

        # Compare against real GT if still within the loaded sequence.
        gt_bev = None
        if gt_bevs is not None and step < seq_len:
            gt_bev = gt_bevs[step]
            print(f"    (GT action at step {step}: {format_action(int(gt_actions[step]))})")

        render_pair(bev, gt_bev, step, out_dir, action=action_idx)


def run_oneshot(model, h_t, sample_t, gt_bevs, gt_actions, seq_len,
                actions, use_sample, out_dir):
    print("\n" + "=" * 60)
    print(f"One-shot roll-out of {len(actions)} actions:")
    print("  " + " ".join(format_action(a) for a in actions))
    print("=" * 60)

    for i, action_idx in enumerate(actions):
        h_t, sample_t, bev = imagine_step(model, h_t, sample_t, action_idx,
                                          use_sample)
        step = i + 1
        print(f"\n[step {step}] action {format_action(action_idx)}")
        print_bev_stats(bev)
        gt_bev = None
        if gt_bevs is not None and step < seq_len:
            gt_bev = gt_bevs[step]
        render_pair(bev, gt_bev, step, out_dir, action=action_idx)


def main():
    parser = argparse.ArgumentParser(description="Interactive BEV imagination.")
    parser.add_argument('--sample', type=int, default=0,
                        help='dataset sample index for the initial observation')
    parser.add_argument('--actions', nargs='+', type=int, default=None,
                        help='one-shot action sequence (non-interactive)')
    parser.add_argument('--no-sample', dest='use_sample', action='store_false',
                        help='use prior mean (deterministic) instead of sampling')
    parser.set_defaults(use_sample=True)
    parser.add_argument('--checkpoint', default=CHECKPOINT_PATH)
    parser.add_argument('--data-dir', default=DATA_DIR)
    parser.add_argument('--output-dir', default=OUTPUT_DIR)
    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    data_dir = args.data_dir
    output_dir = args.output_dir

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = load_model(device, checkpoint_path)

    print("[3] Loading dataset...")
    from model.dataset.uav_parquet_dataset import UAVParquetDataset
    dataset = UAVParquetDataset(
        dataset_path=os.path.join(data_dir, 'test'),
        sequence_length=4,
        enable_semantic=True,
        enable_rgb_stylegan=True,
        is_gwm_pretrain=True,
        enable_bev=True,
    )
    print(f"  Dataset size: {len(dataset)}")

    print("[4] Encoding initial observation...")
    h_t, sample_t, gt_bevs, gt_actions, seq_len = encode_initial_state(
        model, dataset, args.sample, device)

    if args.actions is not None:
        run_oneshot(model, h_t, sample_t, gt_bevs, gt_actions, seq_len,
                    args.actions, args.use_sample, output_dir)
    else:
        run_interactive(model, h_t, sample_t, gt_bevs, gt_actions, seq_len,
                        args.use_sample, output_dir)


if __name__ == "__main__":
    main()
