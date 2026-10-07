#!/usr/bin/env python3
"""
Convert UAV dataset format to Parquet format for faster loading.

UAV format:
    dataset_path/
        episode_0001/
            episode_summary.json
            step_0000/
                rgb_front.png
                depth_front.npy (optional)
                semantic_front.npy (optional)
                state.json

Parquet format:
    output_path/
        train/
            episode_0001/
                output_0000.pqt
                output_0000_metadata.json
            ...
        val/
        test/
"""

import argparse
import io
import json
import os
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


# --- Explicit spatial-cognition memory specification (paper Table tab:specs) --
# Keep in sync with model/loss/bev_grid.py, model/rl/footprint.py and
# model/searchworld/bev_memory.py.
BEV_SIZE = 256                  # cells per side (256 x 256)
BEV_RESOLUTION = 0.4            # metres per cell (102.4 m x 102.4 m coverage)
FOV_RANGE_M = 20.0              # rho   : sensing range of the exploration FOV
FOV_ANGLE_DEG = 90.0            # phi   : angular width of the FOV
FOV_HALF_ANGLE_DEG = FOV_ANGLE_DEG / 2.0
Z_OBS_M = 0.5                   # z_obs : obstacle vertical clearance threshold
SIGMA_G_M = 10.0                # sigma_g: width of the goal Gaussian in V*
LAMBDA_EXPL = 1.0               # lambda_expl in eq. (valuetarget)
LAMBDA_GOAL = 1.0               # lambda_goal in eq. (valuetarget)
DEPTH_MIN_M = 0.1               # valid depth window of the depth camera
DEPTH_MAX_M = 30.0


class BEVGenerator:
    """Generate the three-channel BEV memory (paper Section 3.2).

    Channel 0 (exploration): deterministic function of the pose and the camera
    FOV, textbook formula eq. (expl).
    Channel 1 (obstacle): height/clearance test on the depth image
    back-projected with the camera intrinsics.
    Channel 2 (value): the training target V* of eq. (valuetarget). It is
    stored as metadata only: the training loss rebuilds V* from channels 0/1
    plus the ``target_rel`` column, so nothing but visualisation depends on
    this channel (see ``model/loss/value_target.py``).

    Geometry single source of truth: ``model/loss/bev_grid.py``.
    """

    # Front depth camera model. The defaults are exactly the AirSim front-view
    # field of view that the earlier heuristic hard-coded as the magic numbers
    # 0.5 / 0.3 (the tangents of the half-FOV), so the pinhole back-projection
    # below reproduces the previous numerics while making the intrinsics
    # explicit and overridable from the real AirSim settings.
    DEFAULT_HFOV_DEG = 2.0 * math.degrees(math.atan(0.5))   # 53.13 deg
    DEFAULT_VFOV_DEG = 2.0 * math.degrees(math.atan(0.3))   # 33.40 deg

    def __init__(self, bev_size=(BEV_SIZE, BEV_SIZE), world_scale=BEV_RESOLUTION,
                 hfov_deg=None, vfov_deg=None,
                 z_obs=Z_OBS_M, camera_height=1.0):
        self.bev_h, self.bev_w = bev_size
        self.world_scale = world_scale
        self.hfov_deg = (self.DEFAULT_HFOV_DEG if hfov_deg is None
                         else float(hfov_deg))
        self.vfov_deg = (self.DEFAULT_VFOV_DEG if vfov_deg is None
                         else float(vfov_deg))
        self.z_obs = z_obs
        self.camera_height = camera_height

        # Cell-centre world coordinates of the affine grid, EXACTLY the map of
        # model/loss/bev_grid.py::grid_world_coords:
        #     u = W/2 + X/res   <=>   X = (u - W/2) * res   (+X = forward/right)
        #     v = H/2 - Y/res   <=>   Y = (H/2 - v) * res   (+Y = left)
        # np.linspace(-51.2, 51.2, 256) would instead give a 102.4/255 =
        # 0.40157 m step, which drifts up to half a cell (~0.2 m) against the
        # affine map shared by the observation encoder, the value target and
        # the stage-3 motion footprints.
        res = self.world_scale
        xs = (np.arange(self.bev_w, dtype=np.float64) - self.bev_w / 2.0) * res
        ys = (self.bev_h / 2.0 - np.arange(self.bev_h, dtype=np.float64)) * res
        self.grid_x, self.grid_y = np.meshgrid(xs, ys)  # both (H, W)

    def world_to_bev(self, Xw, Yw):
        """World metres -> BEV (col, row) as floats; inverse of the grid."""
        return (self.bev_w / 2.0 + Xw / self.world_scale,
                self.bev_h / 2.0 - Yw / self.world_scale)

    def compute_exploration_layer(self, pose):
        """Exploration layer of eq. (expl) in the paper.

        M_expl* (c) = max_{tau <= t} [ (1 - d_tau(c)/rho) *
                      cos(pi * dtheta_tau(c) / phi) ] *
                      1[ d_tau(c) <= rho, |dtheta_tau(c)| <= phi/2 ]

        with sensing range rho = FOV_RANGE_M and FOV width phi =
        FOV_ANGLE_DEG (phi/2 = FOV_HALF_ANGLE_DEG). The per-step observation is
        the bracketed term masked by the indicator; the caller accumulates the
        running maximum over steps (``generate_bev``).
        """
        uav_x = pose[0]
        uav_y = pose[1]
        uav_yaw = pose[3]

        dx = self.grid_x - uav_x
        dy = self.grid_y - uav_y
        dist = np.sqrt(dx**2 + dy**2)

        angle_to_point = np.arctan2(dy, dx)
        angle_diff = angle_to_point - uav_yaw
        angle_diff = ((angle_diff + np.pi) % (2 * np.pi)) - np.pi

        # Indicator: |dtheta| <= phi/2 and d <= rho.
        in_fov_angle = np.abs(angle_diff) <= np.radians(FOV_HALF_ANGLE_DEG)
        in_fov_range = dist <= FOV_RANGE_M
        in_fov = in_fov_angle & in_fov_range

        # Linear radial attenuation (1 - d/rho), clipped to [0, 1].
        distance_decay = 1.0 - (dist / FOV_RANGE_M)
        distance_decay = np.clip(distance_decay, 0.0, 1.0)

        # Cosine angular attenuation cos(pi * dtheta / phi). Inside the FOV
        # mask |pi*dtheta/phi| <= pi/2, so the clip is a numerical no-op.
        angle_decay = np.cos(angle_diff * (np.pi / np.radians(FOV_ANGLE_DEG)))
        angle_decay = np.clip(angle_decay, 0.0, 1.0)

        exploration = np.zeros_like(dist)
        exploration[in_fov] = distance_decay[in_fov] * angle_decay[in_fov]
        return exploration

    def compute_obstacle_layer(self, depth, pose):
        """Obstacle layer M_obs*: vertical clearance test on the depth cloud.

        Matches the paper's description: the front-view depth image is
        back-projected to a metric 3-D point cloud *with the camera
        intrinsics*, transformed into the allocentric BEV frame by the pose,
        orthographically projected onto the grid, and a cell is marked as an
        obstacle when the projected point lies more than z_obs above the
        ground.

        Camera model (pinhole, zero distortion, zero pitch/roll):
            fx = (W - 1) / 2 / tan(hfov / 2),   cx = (W - 1) / 2
            fy = (H - 1) / 2 / tan(vfov / 2),   cy = (H - 1) / 2
            X_cam =  D                        (forward)
            Y_cam = -(u - cx) / fx * D        (+Y_cam = left)
            Z_cam = -(v - cy) / fy * D        (+Z_cam = up, v grows downward)

        The point's height above the ground is ``camera_height + Z_cam`` and a
        point is an obstacle when that exceeds ``z_obs``. Points whose depth
        falls outside [DEPTH_MIN_M, DEPTH_MAX_M] are discarded as invalid.
        """
        obstacle = np.zeros((self.bev_h, self.bev_w), dtype=np.float32)

        if depth is None:
            return obstacle

        h, w = depth.shape

        # Camera intrinsics derived from the front-view field of view.
        cx = (w - 1) / 2.0
        cy = (h - 1) / 2.0
        fx = cx / math.tan(math.radians(self.hfov_deg) / 2.0)
        fy = cy / math.tan(math.radians(self.vfov_deg) / 2.0)

        # Create pixel coordinates
        v_pix, u_pix = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

        # Filter valid depth
        valid = (depth > DEPTH_MIN_M) & (depth < DEPTH_MAX_M)
        depth_flat = depth[valid].astype(np.float64)
        u_pix_flat = u_pix[valid].astype(np.float64)
        v_pix_flat = v_pix[valid].astype(np.float64)

        if len(depth_flat) == 0:
            return obstacle

        # Back-project the depth image to a metric point cloud in the camera
        # frame (forward / left / up), then express it in the BEV frame
        # (X = forward, Y = left) by rotating with the yaw and translating
        # with the UAV position.
        Xw_rel = depth_flat
        Yw_rel = -(u_pix_flat - cx) / fx * depth_flat
        Zw_rel = -(v_pix_flat - cy) / fy * depth_flat

        # Rotate
        cos_yaw = math.cos(pose[3])
        sin_yaw = math.sin(pose[3])
        Xw = Xw_rel * cos_yaw - Yw_rel * sin_yaw + pose[0]
        Yw = Xw_rel * sin_yaw + Yw_rel * cos_yaw + pose[1]

        # Height above ground / clearance test.
        Zw = self.camera_height + Zw_rel
        is_obstacle = Zw > self.z_obs

        # Orthographic projection onto the grid via the shared affine map.
        u_bev_f, v_bev_f = self.world_to_bev(Xw, Yw)
        # np.floor selects the cell that CONTAINS the point. For in-bounds
        # points u/v >= 0 anyway, so this equals the previous truncating cast.
        u_bev = np.floor(u_bev_f).astype(np.int64)
        v_bev = np.floor(v_bev_f).astype(np.int64)

        # Filter in bounds
        valid_bev = (u_bev >= 0) & (u_bev < self.bev_w) & (v_bev >= 0) & (v_bev < self.bev_h)
        u_bev = u_bev[valid_bev & is_obstacle]
        v_bev = v_bev[valid_bev & is_obstacle]

        # Mark
        if len(u_bev) > 0:
            obstacle[v_bev, u_bev] = 1.0

        # Gaussian blur
        from scipy.ndimage import gaussian_filter
        obstacle = gaussian_filter(obstacle, sigma=1.0)
        obstacle = np.clip(obstacle, 0.0, 1.0)
        return obstacle

    def compute_value_layer(self, exploration, obstacle, target_pos):
        """Value layer = the training target V* (paper eq. valuetarget).

        V*(x) = Norm[(1 - M_obs) * (lambda_expl * (1 - M_expl) + G_g(x))],
        with lambda_expl = lambda_goal = 1.0 and the goal Gaussian of width
        sigma_g = 10 m. Keep in sync with
        model/loss/value_target.py::build_value_target.
        """
        target_x, target_y = target_pos[0], target_pos[1]

        dist2 = (self.grid_x - target_x) ** 2 + (self.grid_y - target_y) ** 2
        goal = np.exp(-dist2 / (2.0 * SIGMA_G_M ** 2))

        # Feasibility-masked blend of remaining exploration and the goal prior.
        feasibility = 1.0 - obstacle
        value = feasibility * ((1.0 - exploration) + goal)

        # Norm[...] rescales the layer to UNIT MAXIMUM (build_value_target
        # divides by amax and never subtracts a minimum), so the minimum is
        # preserved instead of being mapped to 0.
        vmax = float(value.max())
        if vmax > 1e-8:
            value = value / vmax

        return value

    def generate_bev(self, pose, depth, target_pos, prev_bev=None,
                     expl_diffuse_sigma=None):
        """Generate full BEV.

        When prev_bev is given, layers accumulate over steps:
            - exploration: cumulative max of per-step FOV coverage
            - obstacle:    cumulative max of depth-projected obstacles
            - value:       computed from the ACCUMULATED exploration, so
              explored areas keep a persistently low value across steps

        expl_diffuse_sigma: optional Gaussian sigma applied to the PER-STEP
            exploration observation BEFORE accumulation. The observation's
            decay tail then spreads smoothly outward (past the hard 20 m /
            45 deg boundaries) instead of stopping abruptly, while the
            cumulative max keeps explored areas at their peak value.
        """
        exploration = self.compute_exploration_layer(pose)
        obstacle = self.compute_obstacle_layer(depth, pose)

        # Gaussian-diffuse the single-step observation before accumulation
        if expl_diffuse_sigma is not None and expl_diffuse_sigma > 0:
            from scipy.ndimage import gaussian_filter
            exploration = gaussian_filter(
                exploration, sigma=expl_diffuse_sigma)
            exploration = np.clip(exploration, 0.0, 1.0)

        # Accumulate exploration and obstacle from previous steps
        if prev_bev is not None:
            exploration = np.maximum(prev_bev[0], exploration)
            obstacle = np.maximum(prev_bev[1], obstacle)

        # Value must be computed AFTER exploration accumulation, otherwise
        # the value layer would only see the single-step FOV and explored
        # areas would not keep their accumulated decrease.
        value = self.compute_value_layer(exploration, obstacle, target_pos)

        bev = np.stack([exploration, obstacle, value], axis=0)  # (3, 256, 256)
        return bev


def split_episodes(
    episodes: List[Path],
    split_ratios: Dict[str, float] = None
) -> Dict[str, List[Path]]:
    """Split episodes into train/val/test."""
    if split_ratios is None:
        split_ratios = {'train': 0.8, 'val': 0.1, 'test': 0.1}

    n = len(episodes)
    n_train = int(n * split_ratios.get('train', 0.8))
    n_val = int(n * split_ratios.get('val', 0.1))

    return {
        'train': episodes[:n_train],
        'val': episodes[n_train:n_train + n_val],
        'test': episodes[n_train + n_val:],
    }


def _quat_to_yaw(qx, qy, qz, qw):
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def load_uav_episode(episode_path: Path):
    """Load an episode from UAV format."""
    summary_path = episode_path / 'episode_summary.json'
    with open(summary_path, 'r') as f:
        summary = json.load(f)

    # Find all step folders
    step_folders = sorted(
        [f for f in episode_path.iterdir() if f.name.startswith('step_')],
        key=lambda x: int(x.name.split('_')[1])
    )

    # Get target position
    target_pos = np.array(summary['target_position'][0])[:2]
    start_pos = np.array(summary['start_position'])[:2]
    target_rel = target_pos - start_pos

    # Get start yaw
    start_yaw = _quat_to_yaw(*summary['start_quaternion'])

    # Initialize BEV generator
    bev_gen = BEVGenerator()
    prev_bev = None

    steps_data = []
    for step_folder in step_folders:
        step_data = load_uav_step(step_folder, start_pos, start_yaw, target_rel, bev_gen, prev_bev)
        steps_data.append(step_data)
        prev_bev = step_data['_bev']  # Keep for next step

    return summary, steps_data


def load_uav_step(step_path: Path, start_pos, start_yaw, target_rel, bev_gen, prev_bev):
    """Load a single step from UAV format."""
    step_data = {}

    # Load state
    with open(step_path / 'state.json', 'r') as f:
        state = json.load(f)

    # Load RGB image (convert to PNG bytes)
    img_path = step_path / 'rgb_front.png'
    if img_path.exists():
        with Image.open(img_path) as img:
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            step_data['camera_image'] = img_byte_arr
    else:
        step_data['camera_image'] = None

    # Load semantic label if exists
    semantic_path = step_path / 'semantic_front.npy'
    if semantic_path.exists():
        semantic = np.load(semantic_path)
        step_data['semantic_labels'] = semantic.flatten().tolist()
        step_data['perspective_semantic_image_shape'] = list(semantic.shape)
    else:
        step_data['semantic_labels'] = []
        step_data['perspective_semantic_image_shape'] = [0, 0]

    # Load depth if exists
    depth_path = step_path / 'depth_front.npy'
    depth = None
    if depth_path.exists():
        depth = np.load(depth_path)
        step_data['depth_image'] = depth.flatten().tolist()
        step_data['depth_image_shape'] = list(depth.shape)

    # Get pose
    pos = np.array([state['position']['x'], state['position']['y'], state['position']['z']])
    q = state['attitude_quaternion']
    cur_yaw = _quat_to_yaw(q['x'], q['y'], q['z'], q['w'])
    rel_pos = pos[:2] - start_pos
    rel_yaw = cur_yaw - start_yaw
    rel_yaw = ((rel_yaw + np.pi) % (2 * np.pi)) - np.pi

    # Generate BEV
    pose = np.array([rel_pos[0], rel_pos[1], 0.0, rel_yaw], dtype=np.float32)
    bev = bev_gen.generate_bev(pose, depth, target_rel, prev_bev)
    step_data['_bev'] = bev  # Keep temp for next step
    step_data['bev_map'] = bev.flatten().tolist()
    step_data['bev_map_shape'] = [3, 256, 256]

    # Map action to driving_command
    action_str = state.get('action', 'start')
    action_map = {
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
    step_data['driving_command'] = action_map.get(action_str, action_map['stop'])

    # ego_speed (set to 0 as placeholder)
    step_data['ego_speed'] = 0.0

    # Allocentric agent pose (x, y, yaw) relative to the episode start
    # (metres, radians): stage-3 footprint anchor.
    step_data['pose'] = [float(rel_pos[0]), float(rel_pos[1]), float(rel_yaw)]
    # Ground-truth target position (x, y) in the same allocentric frame
    # (metres relative to episode start = BEV grid centre). TRAINING-ONLY
    # supervision for the value layer (paper eq. valuetarget).
    step_data['target_rel'] = [float(target_rel[0]), float(target_rel[1])]

    # path and route_poses (placeholders for compatibility)
    step_data['path'] = [0.0, 0.0, 0.0, 0.0]
    step_data['route_poses'] = [0.0, 0.0, 0.0, 0.0]

    return step_data


def convert_episode_to_parquet(
    episode_path: Path,
    output_episode_path: Path,
    task_description: str,
    samples_per_file: int = 50
):
    """Convert a single episode to Parquet files."""
    output_episode_path.mkdir(parents=True, exist_ok=True)

    try:
        summary, steps_data = load_uav_episode(episode_path)
    except Exception as e:
        print(f"Error loading {episode_path}: {e}")
        import traceback
        traceback.print_exc()
        return

    # Split into chunks and save as parquet
    num_files = (len(steps_data) + samples_per_file - 1) // samples_per_file

    for file_idx in range(num_files):
        start_idx = file_idx * samples_per_file
        end_idx = min((file_idx + 1) * samples_per_file, len(steps_data))
        chunk_steps = steps_data[start_idx:end_idx]

        # Remove temp bev before saving
        for step in chunk_steps:
            if '_bev' in step:
                del step['_bev']

        # Create DataFrame
        df = pd.DataFrame(chunk_steps)

        # Save parquet
        parquet_filename = f'output_{file_idx:04d}.pqt'
        parquet_path = output_episode_path / parquet_filename
        df.to_parquet(parquet_path, engine='pyarrow')

        # Save metadata
        metadata = {
            'episode_id': episode_path.name,
            'file_index': file_idx,
            'num_samples': len(chunk_steps),
            'task_description': task_description,
        }
        metadata_path = output_episode_path / f'output_{file_idx:04d}_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description='Convert UAV dataset to Parquet format'
    )
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Path to UAV dataset root'
    )
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Path to output Parquet dataset root'
    )
    parser.add_argument(
        '--samples-per-file',
        type=int,
        default=100,
        help='Number of samples per parquet file'
    )
    parser.add_argument(
        '--task-description',
        default='Search for the target object',
        help='Task description for text encoder'
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    # Find all episodes
    episodes = sorted([
        f for f in input_path.iterdir()
        if f.name.startswith('episode_') and f.is_dir()
    ])

    if not episodes:
        print(f"No episodes found in {input_path}")
        return

    print(f"Found {len(episodes)} episodes")

    # Split into train/val/test
    splits = split_episodes(episodes)

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
    for split_name in splits:
        split_path = output_path / split_name
        if split_path.exists():
            num_episodes = len([f for f in split_path.iterdir() if f.is_dir()])
            print(f"  {split_name}: {num_episodes} episodes")


if __name__ == '__main__':
    main()

