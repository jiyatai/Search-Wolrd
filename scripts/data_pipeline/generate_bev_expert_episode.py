#!/usr/bin/env python3
"""
Generate per-step BEV images for one expert episode (4cam jsonl format).

Reuses the exact BEVGenerator code path from convert_expert_to_parquet.py
(numpy version, identical to the training-data GT generation):
    - relative frame anchored at episode start (rel_pos, rel_yaw)
    - depth.npz uint8 front channel -> meters (u8/255*100)
    - exploration: FOV(90deg, 20m) decay, cumulative max over steps
    - obstacle: depth projection + Gaussian blur, cumulative max
    - value: feasibility-masked blend of remaining exploration and the goal
      Gaussian (paper eq. valuetarget), min-max normalized

Output (per step t):
    <out_dir>/bev_step_%04d.png             3 channels + composite, UAV/target markers
    <out_dir>/bev_step_%04d.npz             raw (3, 256, 256) float32 BEV
    <out_dir>/ch_exploration/bev_step_%04d.png   channel 0 alone
    <out_dir>/ch_obstacle/bev_step_%04d.png      channel 1 alone
    <out_dir>/ch_value/bev_step_%04d.png         channel 2 alone
    <out_dir>/bev_composite.gif             optional gif via --gif
    <out_dir>/bev_exploration.gif           optional per-channel gif via --gif
    <out_dir>/bev_obstacle.gif              optional per-channel gif via --gif
    <out_dir>/bev_value.gif                 optional per-channel gif via --gif

Usage:
    python scripts/generate_bev_expert_episode.py \
        --episode /shared_disk/users/wenhao.lu/JYT/WorldSearch_data/expert_data/BrushifyUrban_test/ep_154
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from convert_uav_to_parquet import BEVGenerator  # noqa: E402
from convert_expert_to_parquet import decode_depth  # noqa: E402

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# BEV geometry constants (must match BEVGenerator)
BEV_SIZE = 256
WORLD_SCALE = 0.4
CENTER = BEV_SIZE // 2
FOV_RANGE_M = 20.0
FOV_HALF_ANGLE_DEG = 45.0

# ---- Custom per-channel color spec ----
# exploration (ch0): deep saturated green -> white (high contrast vs value)
CMAP_EXPL = LinearSegmentedColormap.from_list(
    'expl_green', [(0.0, 0.30, 0.0), (1.0, 1.0, 1.0)])
# value (ch2): white -> orange -> deep orange-red (strong vs green exploration)
CMAP_VALUE = LinearSegmentedColormap.from_list(
    'value_orange_red', [(1.0, 1.0, 1.0), (1.0, 0.45, 0.0), (0.78, 0.08, 0.0)])
# obstacle (ch1) is rendered as RGB via obstacle_rgb():
#   unexplored=gray, explored-free=white, obstacle=black (accumulated)
GRAY = np.array([0.65, 0.65, 0.65], dtype=np.float32)
WHITE = np.array([1.0, 1.0, 1.0], dtype=np.float32)


def obstacle_rgb(obst, expl):
    """Render obstacle channel as RGB using exploration as background.

    Background blends gray (unexplored) -> white (explored) by the
    exploration value; obstacle intensity darkens it toward black.
    Obstacles accumulate over steps (cumulative max in BEVGenerator).
    """
    bg = GRAY + (WHITE - GRAY) * expl[..., None]
    rgb = bg * (1.0 - obst[..., None])
    return np.clip(rgb, 0.0, 1.0)


def render_channel(bev, ch):
    """Render one BEV channel to an (H, W, 3) RGB image per the color spec."""
    if ch == 0:
        return CMAP_EXPL(bev[0])[..., :3]
    if ch == 1:
        return obstacle_rgb(bev[1], bev[0])
    return CMAP_VALUE(bev[2])[..., :3]


def bev_to_pixel(x_rel: float, y_rel: float):
    """World (X=forward, Y=left, episode frame) -> BEV pixel (u, v)."""
    u = CENTER + x_rel / WORLD_SCALE
    v = CENTER - y_rel / WORLD_SCALE
    return u, v


def draw_markers(ax, rel_pos, rel_yaw, target_rel,
                 show_target=True, show_uav=True):
    """Mark UAV (red dot + FOV cone) and target (yellow star)."""
    uav_u, uav_v = bev_to_pixel(rel_pos[0], rel_pos[1])
    if show_uav:
        ax.add_patch(Circle((uav_u, uav_v), 3, color='red', zorder=5))

        # FOV cone boundaries (yaw is CCW-positive in world; BEV v-axis flipped)
        for sign in (-1, 1):
            ang = rel_yaw + sign * math.radians(FOV_HALF_ANGLE_DEG)
            r_pix = FOV_RANGE_M / WORLD_SCALE
            du = r_pix * math.cos(ang)
            dv = -r_pix * math.sin(ang)
            ax.plot([uav_u, uav_u + du], [uav_v, uav_v + dv],
                    'r-', alpha=0.5, linewidth=1)

    if show_target:
        tgt_u, tgt_v = bev_to_pixel(target_rel[0], target_rel[1])
        if 0 <= tgt_u < BEV_SIZE and 0 <= tgt_v < BEV_SIZE:
            ax.plot(tgt_u, tgt_v, marker='*', color='yellow',
                    markersize=12, markeredgecolor='black', zorder=5)


def make_channel_figure(bev, ch, name, rel_pos, rel_yaw, target_rel,
                        rec, step_idx, show_target=True, show_uav=True):
    """Single-channel map with UAV/target markers (custom color spec)."""
    rgb = render_channel(bev, ch)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(rgb, origin='upper')
    ax.set_title(f'{name}  (step {step_idx}, act={rec.get("act", "?")})')
    draw_markers(ax, rel_pos, rel_yaw, target_rel, show_target, show_uav)
    plt.tight_layout()
    return fig


def make_figure(bev, rel_pos, rel_yaw, target_rel, rec, step_idx,
                show_target=True, show_uav=True):
    """3 channel maps (custom colors) + RGB-like composite with markers."""
    composite = render_channel(bev, 1)  # obstacle layer = composite view

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.6))
    panels = [
        (0, 'Exploration (ch0): green=unexplored white=explored'),
        (1, 'Obstacle (ch1): gray=unexplored white=free black=obstacle'),
        (2, 'Value (ch2): white=low orange=high'),
    ]
    for ax, (ch, title) in zip(axes[:3], panels):
        ax.imshow(render_channel(bev, ch), origin='upper')
        ax.set_title(title, fontsize=9)
        draw_markers(ax, rel_pos, rel_yaw, target_rel, show_target, show_uav)

    ax = axes[3]
    ax.imshow(composite, origin='upper')
    ax.set_title('Obstacle-layer view (accumulated)')
    draw_markers(ax, rel_pos, rel_yaw, target_rel, show_target, show_uav)

    act = rec.get('act', '?')
    phase = rec.get('phase', '?')
    fig.suptitle(
        f'ep step {step_idx}  act={act}  phase={phase}  '
        f'rel_pos=({rel_pos[0]:.1f}, {rel_pos[1]:.1f})m  '
        f'rel_yaw={math.degrees(rel_yaw):.1f}deg  '
        f'target_rel=({target_rel[0]:.1f}, {target_rel[1]:.1f})m',
        fontsize=10)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episode', '-e', required=True,
                        help='Expert episode dir (contains episode.jsonl + depth.npz)')
    parser.add_argument('--output', '-o', default=None,
                        help='Output dir (default: <episode>/bev_vis)')
    parser.add_argument('--gif', action='store_true',
                        help='Also save an animated gif of the composite')
    parser.add_argument('--no-target', action='store_true',
                        help='Do not draw the yellow target star marker')
    parser.add_argument('--no-uav', action='store_true',
                        help='Do not draw the UAV red dot + FOV cone marker')
    parser.add_argument('--diffuse', type=float, default=None,
                        metavar='SIGMA',
                        help='Gaussian-diffuse each per-step exploration '
                             'observation before max-accumulation '
                             '(spread the observation footprint smoothly)')
    args = parser.parse_args()

    episode_path = Path(args.episode)
    out_dir = Path(args.output) if args.output else episode_path / 'bev_vis'
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load summary + jsonl records (same as convert_expert_to_parquet) ----
    with open(episode_path / 'episode_summary.json') as f:
        summary = json.load(f)
    records = []
    with open(episode_path / 'episode.jsonl') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        raise ValueError(f'empty episode.jsonl in {episode_path}')

    depth_npz = np.load(episode_path / 'depth.npz')

    start_pos = np.array(summary['start_position'], dtype=np.float64)
    target_pos = np.array(summary['target_position'][0], dtype=np.float64)
    qx, qy, qz, qw = summary['start_quaternion']
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    start_yaw = math.atan2(siny_cosp, cosy_cosp)
    target_rel = (target_pos - start_pos)[:2]

    # ---- Per-step BEV generation (identical to parquet GT path) ----
    bev_gen = BEVGenerator()
    prev_bev = None
    frames = []          # composite gif frames
    channel_frames = {   # per-channel gif frames
        'exploration': [],
        'obstacle': [],
        'value': [],
    }
    # channel subdir, channel index, gif name
    channel_cfg = [
        ('ch_exploration', 0, 'exploration'),
        ('ch_obstacle', 1, 'obstacle'),
        ('ch_value', 2, 'value'),
    ]
    channel_dirs = {name: out_dir / sub for sub, _, name in channel_cfg}
    for d in channel_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    for rec in records:
        t = rec['t']
        if t >= depth_npz['front'].shape[0]:
            break

        depth_m = decode_depth(depth_npz['front'][t])
        pos = np.array(rec['pos'], dtype=np.float64)
        quat = rec['quat']  # [x, y, z, w]
        siny_cosp = 2.0 * (quat[3] * quat[2] + quat[0] * quat[1])
        cosy_cosp = 1.0 - 2.0 * (quat[1] * quat[1] + quat[2] * quat[2])
        yaw = math.atan2(siny_cosp, cosy_cosp)

        rel_pos = pos[:2] - start_pos[:2]
        rel_yaw = ((yaw - start_yaw + math.pi) % (2 * math.pi)) - math.pi

        pose = np.array([rel_pos[0], rel_pos[1], 0.0, rel_yaw], dtype=np.float32)
        bev = bev_gen.generate_bev(pose, depth_m, target_rel, prev_bev,
                                   expl_diffuse_sigma=args.diffuse)
        prev_bev = bev

        show_target = not args.no_target
        show_uav = not args.no_uav
        fig = make_figure(bev, rel_pos, rel_yaw, target_rel, rec, t,
                          show_target, show_uav)
        fig_path = out_dir / f'bev_step_{t:04d}.png'
        fig.savefig(fig_path, dpi=130, bbox_inches='tight')
        plt.close(fig)

        np.savez_compressed(out_dir / f'bev_step_{t:04d}.npz', bev=bev)

        # Per-channel standalone images
        for sub, ch, name in channel_cfg:
            fig = make_channel_figure(
                bev, ch, name, rel_pos, rel_yaw, target_rel, rec, t,
                show_target, show_uav)
            fig_path_ch = channel_dirs[name] / f'bev_step_{t:04d}.png'
            fig.savefig(fig_path_ch, dpi=130, bbox_inches='tight')
            plt.close(fig)

        # Collect frames for optional gifs (custom color spec)
        frames.append(
            (render_channel(bev, 1) * 255).clip(0, 255).astype(np.uint8))
        for _, ch, name in channel_cfg:
            channel_frames[name].append(
                (render_channel(bev, ch) * 255).clip(0, 255).astype(np.uint8))
        print(f'[t={t:02d}] act={rec.get("act"):<8} saved {fig_path.name}')

    if args.gif and frames:
        try:
            import imageio.v2 as imageio
            imageio.mimsave(out_dir / 'bev_composite.gif', frames, fps=4)
            print(f'Saved {out_dir / "bev_composite.gif"}')
            for name, ch_frames in channel_frames.items():
                imageio.mimsave(
                    out_dir / f'bev_{name}.gif', ch_frames, fps=4)
                print(f'Saved {out_dir / f"bev_{name}.gif"}')
        except ImportError:
            print('imageio not installed, skip gif')

    print(f'\nDone: {len(frames)} steps -> {out_dir}')


if __name__ == '__main__':
    main()
