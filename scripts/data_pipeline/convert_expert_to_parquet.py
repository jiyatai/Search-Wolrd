#!/usr/bin/env python3
"""
Convert expert UAV dataset (4cam jsonl format) to Parquet format.

Expert format (per episode):
    dataset_path/
        ep_1000/
            episode_summary.json   # description / target / start pose
            episode.jsonl          # per-step: t, act, pos, quat, mv, ...
            rgb_front/0000.jpg     # (512, 512) RGB
            depth.npz              # uint8, keys: front/left/right/down
                                   # depth_meters = uint8 / 255.0 * 100.0
            trajectory.png

Parquet format:
    output_path/
        train/
            ep_1000/
                output_0000.pqt
                output_0000_metadata.json
            ...
        val/
        test/

Notes:
    - No semantic columns: stage-2 training disables the semantic head.
    - BEV generation reuses BEVGenerator from convert_uav_to_parquet.py.
    - Action mapping (must match dataset/妯″瀷涓ゅ):
        start:0 forward:1 left:2 right:3 ascend:4 descend:5 rotl:6 rotr:7
    - 'stop' is encoded as its own terminal class (7); 'start' is a
      first-frame sentinel and is encoded as the terminal class.
    - task_description uses episode_summary.json 'description' (object
      description) for language conditioning.
"""

import argparse
import io
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# Reuse the BEV generator from the random-data converter.
sys.path.insert(0, str(Path(__file__).parent))
from convert_uav_to_parquet import BEVGenerator, split_episodes  # noqa: E402

# depth.npz uint8 -> meters (see AirVLNSimulatorClientTool.getImageResponses)
DEPTH_MAX_RANGE = 100.0

ACTION_TO_IDX = {
    'forward': 0,
    'left': 1,
    'right': 2,
    'ascend': 3,
    'descend': 4,
    'rotl': 5,
    'rotr': 6,
    'stop': 7,
    'start': 7,
}


def _quat_to_yaw(qx, qy, qz, qw):
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def decode_depth(depth_u8: np.ndarray) -> np.ndarray:
    """uint8 depth (0-255) -> float32 meters (0-100)."""
    return depth_u8.astype(np.float32) / 255.0 * DEPTH_MAX_RANGE


def load_expert_episode(episode_path: Path,
                        bev_gen: BEVGenerator,
                        enable_bev: bool = True):
    """Load one expert episode (4cam jsonl format).

    Returns (summary, steps_data). steps_data rows contain the parquet
    columns: driving_command, ego_speed, camera_image (+ bev_map,
    bev_map_shape when enable_bev).
    """
    summary_path = episode_path / 'episode_summary.json'
    with open(summary_path, 'r') as f:
        summary = json.load(f)

    # Read per-step records.
    jsonl_path = episode_path / 'episode.jsonl'
    records = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        raise ValueError(f'empty episode.jsonl in {episode_path}')

    # Relative frame: target & poses expressed w.r.t. episode start.
    start_pos = np.array(summary['start_position'], dtype=np.float64)
    target_pos = np.array(summary['target_position'][0], dtype=np.float64)
    start_yaw = _quat_to_yaw(*summary['start_quaternion'])
    target_rel = (target_pos - start_pos)[:2]

    # Optional depth stack for BEV obstacle layer.
    depth_npz = None
    if enable_bev:
        depth_path = episode_path / 'depth.npz'
        if depth_path.exists():
            depth_npz = np.load(depth_path)

    steps_data = []
    prev_bev = None
    for rec in records:
        t = rec['t']

        # RGB front image -> PNG bytes.
        img_path = episode_path / 'rgb_front' / f'{t:04d}.jpg'
        if not img_path.exists():
            # tolerate missing trailing frames
            break
        with Image.open(img_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            camera_image = buf.getvalue()

        # Relative pose for BEV generation.
        pos = np.array(rec['pos'], dtype=np.float64)
        quat = rec['quat']  # [x, y, z, w]
        yaw = _quat_to_yaw(*quat)
        rel_pos = pos[:2] - start_pos[:2]
        rel_yaw = yaw - start_yaw
        rel_yaw = ((rel_yaw + np.pi) % (2 * np.pi)) - np.pi

        step_data = {
            'driving_command': ACTION_TO_IDX.get(rec['act'], ACTION_TO_IDX['stop']),
            # 'mv' is the cumulative move distance in meters; per-step speed
            # proxy: difference to previous step (m/step). Fallback 0.
            'ego_speed': 0.0,
            'camera_image': camera_image,
            # Allocentric agent pose (x, y, yaw) relative to the episode
            # start (metres, radians). Stage-3 value-guided planning anchors
            # the action motion footprints here; the observation encoder's
            # 'relative_pose' input intentionally stays zeros (checkpoint
            # compatibility), so this column is additive.
            'pose': [float(rel_pos[0]), float(rel_pos[1]), float(rel_yaw)],
            # Ground-truth target position (x, y) in the same allocentric
            # frame (metres relative to episode start = BEV grid centre).
            # TRAINING-ONLY: used to build the value-layer target V* of
            # eq. (valuetarget); never an inference input.
            'target_rel': [float(target_rel[0]), float(target_rel[1])],
        }

        if enable_bev:
            depth_m = None
            if depth_npz is not None and t < depth_npz['front'].shape[0]:
                depth_m = decode_depth(depth_npz['front'][t])
            pose = np.array(
                [rel_pos[0], rel_pos[1], 0.0, rel_yaw], dtype=np.float32)
            bev = bev_gen.generate_bev(pose, depth_m, target_rel, prev_bev)
            prev_bev = bev
            step_data['bev_map'] = bev.flatten().tolist()
            step_data['bev_map_shape'] = [3, 256, 256]

        steps_data.append(step_data)

    # Fill ego_speed from cumulative move distance afterwards.
    for i in range(len(steps_data)):
        mv_now = records[i].get('mv', 0.0)
        if i > 0:
            mv_prev = records[i - 1].get('mv', 0.0)
        else:
            # mv before first move is 0 (first record is pre-move state)
            mv_prev = 0.0
        steps_data[i]['ego_speed'] = float(max(0.0, mv_now - mv_prev))

    return summary, steps_data


def convert_episode_to_parquet(episode_path: Path,
                               output_episode_path: Path,
                               samples_per_file: int = 50,
                               enable_bev: bool = True,
                               bev_gen: BEVGenerator = None):
    """Convert a single expert episode to Parquet files."""
    output_episode_path.mkdir(parents=True, exist_ok=True)

    if bev_gen is None:
        bev_gen = BEVGenerator()

    try:
        summary, steps_data = load_expert_episode(
            episode_path, bev_gen, enable_bev=enable_bev)
    except Exception as e:
        print(f"Error loading {episode_path}: {e}")
        import traceback
        traceback.print_exc()
        return 0

    if not steps_data:
        print(f"Warning: no valid steps in {episode_path}")
        return 0

    # Language conditioning text: object description from the summary.
    task_description = summary.get('description', 'Search for the target object')

    num_files = (len(steps_data) + samples_per_file - 1) // samples_per_file
    for file_idx in range(num_files):
        start_idx = file_idx * samples_per_file
        end_idx = min((file_idx + 1) * samples_per_file, len(steps_data))
        chunk_steps = steps_data[start_idx:end_idx]

        df = pd.DataFrame(chunk_steps)
        parquet_path = output_episode_path / f'output_{file_idx:04d}.pqt'
        df.to_parquet(parquet_path, engine='pyarrow')

        metadata = {
            'episode_id': episode_path.name,
            'file_index': file_idx,
            'num_samples': len(chunk_steps),
            'task_description': task_description,
            'object_name': summary.get('object_name', ''),
            'scene_name': summary.get('scene_name', ''),
            'outcome': summary.get('outcome', ''),
        }
        metadata_path = output_episode_path / f'output_{file_idx:04d}_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    return len(steps_data)


def main():
    parser = argparse.ArgumentParser(
        description='Convert expert UAV dataset (4cam jsonl) to Parquet format')
    parser.add_argument('--input', '-i', required=True,
                        help='Path to expert dataset root (contains ep_*/)')
    parser.add_argument('--output', '-o', required=True,
                        help='Path to output Parquet dataset root')
    parser.add_argument('--samples-per-file', type=int, default=100,
                        help='Number of samples per parquet file')
    parser.add_argument('--no-bev', action='store_true',
                        help='Disable BEV map generation')
    parser.add_argument('--max-episodes', type=int, default=0,
                        help='Only convert first N episodes (0 = all)')
    parser.add_argument('--episode-glob', default='ep_*',
                        help='Episode directory glob pattern')
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    episodes = sorted(input_path.glob(args.episode_glob))
    episodes = [e for e in episodes if e.is_dir()]
    if args.max_episodes > 0:
        episodes = episodes[:args.max_episodes]

    if not episodes:
        print(f"No episodes matching '{args.episode_glob}' found in {input_path}")
        return

    print(f"Found {len(episodes)} episodes")
    print(f"BEV generation: {'OFF' if args.no_bev else 'ON'}")

    bev_gen = None if args.no_bev else BEVGenerator()
    splits = split_episodes(episodes)

    total_steps = 0
    for split_name, split_eps in splits.items():
        if not split_eps:
            continue
        split_output_path = output_path / split_name
        split_output_path.mkdir(parents=True, exist_ok=True)

        print(f"\nConverting {split_name} split ({len(split_eps)} episodes)")
        for episode_path in tqdm(split_eps, desc=split_name):
            n = convert_episode_to_parquet(
                episode_path,
                split_output_path / episode_path.name,
                samples_per_file=args.samples_per_file,
                enable_bev=not args.no_bev,
                bev_gen=bev_gen,
            )
            total_steps += n

    print(f"\nDone! Dataset saved to {output_path}")
    print(f"Total steps converted: {total_steps}")
    for split_name, split_eps in splits.items():
        split_path = output_path / split_name
        if split_path.exists():
            num_episodes = len([f for f in split_path.iterdir() if f.is_dir()])
            print(f"  {split_name}: {num_episodes} episodes")


if __name__ == '__main__':
    main()

