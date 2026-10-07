#!/usr/bin/env python3
"""Convert UAV dataset to Parquet format with BEV.

NOTE: superseded by ``convert_uav_to_parquet.py``, which is the converter
described in ``README.md`` and whose ``BEVGenerator`` is the shared BEV
definition reused by ``convert_expert_to_parquet.py``.  This file is kept for
reference only; prefer the canonical converter for any new parquet export.

The grid below is affine (0.4 m/cell, ``X = (u - W/2) * res``,
``Y = (H/2 - v) * res``) so that it matches ``model/loss/bev_grid.py``.
``linspace(-51.2, 51.2, 256)`` would give a 0.4016 m step instead and misalign
the layers with the affine map by up to half a cell at the edges.
"""

import argparse
import io
import json
import math
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from scipy.ndimage import gaussian_filter
from tqdm import tqdm


class BEVGenerator:
    """Generate BEV from UAV data."""
    def __init__(self, bev_size=(256, 256), world_scale=0.4):
        self.bev_h, self.bev_w = bev_size
        self.world_scale = world_scale
        x = (np.arange(self.bev_w, dtype=np.float64) - self.bev_w / 2.0) * world_scale
        y = (self.bev_h / 2.0 - np.arange(self.bev_h, dtype=np.float64)) * world_scale
        xx, yy = np.meshgrid(x, y)
        self.grid_x = xx
        self.grid_y = yy

    def compute_exploration_layer(self, pose):
        uav_x, uav_y, _, uav_yaw = pose
        dx = self.grid_x - uav_x
        dy = self.grid_y - uav_y
        dist = np.sqrt(dx**2 + dy**2)
        angle_to_point = np.arctan2(dy, dx)
        angle_diff = angle_to_point - uav_yaw
        angle_diff = ((angle_diff + np.pi) % (2 * np.pi)) - np.pi
        in_fov = (np.abs(angle_diff) <= np.radians(45)) & (dist <= 20.0)
        distance_decay = np.clip(1.0 - (dist / 20.0), 0.0, 1.0)
        angle_decay = np.clip(np.cos(angle_diff * (np.pi / np.radians(90.0))), 0.0, 1.0)
        exploration = np.zeros_like(dist)
        exploration[in_fov] = distance_decay[in_fov] * angle_decay[in_fov]
        return exploration

    def compute_obstacle_layer(self, depth, pose):
        obstacle = np.zeros((self.bev_h, self.bev_w), dtype=np.float32)
        if depth is None:
            return obstacle
        h, w = depth.shape
        v_pix, u_pix = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        u_norm = (u_pix / (w - 1)) * 2 - 1
        v_norm = (v_pix / (h - 1)) * 2 - 1
        valid = (depth > 0.1) & (depth < 30.0)
        valid_flat = valid.flatten()
        depth_flat = depth.flatten()[valid_flat]
        u_norm_flat = u_norm.flatten()[valid_flat]
        v_norm_flat = v_norm.flatten()[valid_flat]
        if len(depth_flat) == 0:
            return obstacle
        Xw_rel = depth_flat
        Yw_rel = -u_norm_flat * depth_flat * 0.5
        cos_yaw = math.cos(pose[3])
        sin_yaw = math.sin(pose[3])
        Xw = Xw_rel * cos_yaw - Yw_rel * sin_yaw + pose[0]
        Yw = Xw_rel * sin_yaw + Yw_rel * cos_yaw + pose[1]
        Zw = 1.0 - v_norm_flat * depth_flat * 0.3
        is_obstacle = Zw > 0.5
        u_bev = (128 + (Xw / self.world_scale)).astype(np.int64)
        v_bev = (128 - (Yw / self.world_scale)).astype(np.int64)
        valid_bev = (u_bev >= 0) & (u_bev < self.bev_w) & (v_bev >= 0) & (v_bev < self.bev_h)
        u_bev = u_bev[valid_bev & is_obstacle]
        v_bev = v_bev[valid_bev & is_obstacle]
        if len(u_bev) > 0:
            obstacle[v_bev, u_bev] = 1.0
        obstacle = gaussian_filter(obstacle, sigma=1.0)
        obstacle = np.clip(obstacle, 0.0, 1.0)
        return obstacle

    def compute_value_layer(self, exploration, obstacle, target_pos):
        """Value layer = the training target V* (paper eq. valuetarget).

        V*(x) = Norm[(1 - M_obs) * ((1 - M_expl) + G_g(x))], with the goal
        Gaussian of width sigma_g = 10 m. Keep in sync with
        model/loss/value_target.py::build_value_target.
        """
        target_x, target_y = target_pos[0], target_pos[1]
        dist2 = (self.grid_x - target_x) ** 2 + (self.grid_y - target_y) ** 2
        goal = np.exp(-dist2 / (2.0 * 10.0 ** 2))
        feasibility = 1.0 - obstacle
        value = feasibility * ((1.0 - exploration) + goal)
        # Norm rescales the layer to unit maximum and never subtracts a
        # minimum (mirrors build_value_target's `value / vmax`).
        vmax = float(value.max())
        if vmax > 1e-8:
            value = value / vmax
        return value

    def generate_bev(self, pose, depth, target_pos, prev_bev=None):
        exploration = self.compute_exploration_layer(pose)
        obstacle = self.compute_obstacle_layer(depth, pose)
        value = self.compute_value_layer(exploration, obstacle, target_pos)
        if prev_bev is not None:
            exploration = np.maximum(prev_bev[0], exploration)
        return np.stack([exploration, obstacle, value], axis=0)


def _quat_to_yaw(qx, qy, qz, qw):
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def load_uav_episode(episode_path: Path):
    summary_path = episode_path / 'episode_summary.json'
    with open(summary_path, 'r') as f:
        summary = json.load(f)

    step_folders = sorted(
        [f for f in episode_path.iterdir() if f.name.startswith('step_')],
        key=lambda x: int(x.name.split('_')[1])
    )

    target_pos = np.array(summary['target_position'][0])[:2]
    start_pos = np.array(summary['start_position'])[:2]
    target_rel = target_pos - start_pos
    start_yaw = _quat_to_yaw(*summary['start_quaternion'])
    bev_gen = BEVGenerator()
    prev_bev = None
    steps_data = []

    for step_folder in step_folders:
        step_path = step_folder / 'state.json'
        with open(step_path, 'r') as f:
            state = json.load(f)
        step_data = {}

        img_path = step_folder / 'rgb_front.png'
        with Image.open(img_path) as img:
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            step_data['camera_image'] = img_byte_arr.getvalue()

        semantic_path = step_folder / 'semantic_front.npy'
        if semantic_path.exists():
            semantic = np.load(semantic_path)
            step_data['semantic_labels'] = semantic.flatten().tolist()
            step_data['perspective_semantic_image_shape'] = list(semantic.shape)
        else:
            step_data['semantic_labels'] = []
            step_data['perspective_semantic_image_shape'] = [0, 0]

        depth_path = step_folder / 'depth_front.npy'
        depth = None
        if depth_path.exists():
            depth = np.load(depth_path)
            step_data['depth_image'] = depth.flatten().tolist()
            step_data['depth_image_shape'] = list(depth.shape)

        pos = np.array([state['position']['x'], state['position']['y'], state['position']['z']])
        q = state['attitude_quaternion']
        cur_yaw = _quat_to_yaw(q['x'], q['y'], q['z'], q['w'])
        rel_pos = pos[:2] - start_pos
        rel_yaw = cur_yaw - start_yaw
        rel_yaw = ((rel_yaw + np.pi) % (2 * np.pi)) - np.pi
        pose = np.array([rel_pos[0], rel_pos[1], 0.0, rel_yaw], dtype=np.float32)

        bev = bev_gen.generate_bev(pose, depth, target_rel, prev_bev)
        step_data['bev_map'] = bev.flatten().tolist()
        step_data['bev_map_shape'] = [3, 256, 256]
        prev_bev = bev

        action_str = state.get('action', 'start')
        action_map = {'forward':0,'left':1,'right':2,'ascend':3,'descend':4,'rotl':5,'rotr':6,'stop':7,'start':7}
        step_data['driving_command'] = action_map.get(action_str, action_map['stop'])
        step_data['ego_speed'] = 0.0
        step_data['path'] = [0.0, 0.0, 0.0, 0.0]
        step_data['route_poses'] = [0.0, 0.0, 0.0, 0.0]
        steps_data.append(step_data)

    return summary, steps_data


def convert_episode_to_parquet(
    episode_path: Path,
    output_episode_path: Path,
    task_description: str,
    samples_per_file: int = 100
):
    output_episode_path.mkdir(parents=True, exist_ok=True)
    try:
        summary, steps_data = load_uav_episode(episode_path)
    except Exception as e:
        print(f"Error loading {episode_path}: {e}")
        import traceback
        traceback.print_exc()
        return

    num_files = (len(steps_data) + samples_per_file - 1) // samples_per_file
    for file_idx in range(num_files):
        start_idx = file_idx * samples_per_file
        end_idx = min((file_idx + 1) * samples_per_file, len(steps_data))
        df = pd.DataFrame(steps_data[start_idx:end_idx])
        parquet_filename = f'output_{file_idx:04d}.pqt'
        parquet_path = output_episode_path / parquet_filename
        df.to_parquet(parquet_path, engine='pyarrow')
        metadata = {
            'episode_id': episode_path.name,
            'file_index': file_idx,
            'num_samples': len(df),
            'task_description': task_description,
        }
        metadata_path = output_episode_path / f'output_{file_idx:04d}_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True)
    parser.add_argument('--output', '-o', required=True)
    parser.add_argument('--samples-per-file', type=int, default=100)
    parser.add_argument('--task-description', default='Search for target object')
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    episodes = sorted([f for f in input_path.iterdir() if f.name.startswith('episode_') and f.is_dir()])
    print(f"Found {len(episodes)} episodes")

    n = len(episodes)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    splits = {
        'train': episodes[:n_train],
        'val': episodes[n_train:n_train + n_val],
        'test': episodes[n_train + n_val:],
    }

    for split_name, split_episodes in splits.items():
        if not split_episodes:
            continue
        split_output_path = output_path / split_name
        split_output_path.mkdir(parents=True, exist_ok=True)
        print(f"\nConverting {split_name} split ({len(split_episodes)} episodes)")
        for episode_path in tqdm(split_episodes, desc=split_name):
            output_episode_path = split_output_path / episode_path.name
            convert_episode_to_parquet(
                episode_path,
                output_episode_path,
                args.task_description,
                args.samples_per_file
            )

    print(f"\nDone! Dataset saved to {output_path}")


if __name__ == '__main__':
    main()

