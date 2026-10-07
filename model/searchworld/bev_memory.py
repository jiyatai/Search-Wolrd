# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gin
import torch
from torch import nn
import torch.nn.functional as F
import math

from model.loss.bev_grid import grid_world_coords


@gin.configurable
class BEVMemory(nn.Module):
    '''BEV Memory for UAV target search.

    BEV Specifications:
    - Size: 256 x 256 pixels
    - Resolution: 0.4 meters per pixel (affine: u = 128 + X/0.4, v = 128 - Y/0.4)
    - Coverage: a 102.4 m x 102.4 m square; cell centres span X in
      [-51.2 m, +50.8 m] and Y in [+51.2 m, -50.8 m]
    - Origin cell (0,0) at world (-51.2 m, +51.2 m)
    - Center cell (128,128) at world (0 m, 0 m)

    Three channels:
    [0]: exploration  - 已探索区域，20米视野范围内衰减
    [1]: obstacle     - 障碍物，从深度图生成
    [2]: value        - 价值，从探索层和目标位置计算

    Args:
        bev_size: tuple (height, width) of BEV map
        world_scale: meters per pixel in BEV
        fov_range: maximum FOV range in meters
    '''
    def __init__(self,
                 bev_size: tuple = (256, 256),
                 world_scale: float = 0.4,
                 fov_range: float = 20.0,
                 fov_angle: float = 90.0):
        super().__init__()

        self.bev_h, self.bev_w = bev_size
        self.world_scale = world_scale
        self.fov_range = fov_range
        self.fov_angle = fov_angle

        # Precompute coordinate grids, affine-consistent with
        # u = 128 + X/res and v = 128 - Y/res (single source of truth:
        # model/loss/bev_grid.py). NOTE: linspace(-51.2, 51.2, 256) gives a
        # 0.4016 m step instead of 0.4 m, which misaligns the rendered layers
        # with the converter's affine map by up to half a cell at the edges.
        X, Y = grid_world_coords(self.bev_w, self.world_scale)
        self.register_buffer('bev_x', X.reshape(-1).clone())   # (W,) world X
        self.register_buffer('bev_y', Y.reshape(-1).clone())   # (H,) world Y

        # BEV coordinates in world units
        xx, yy = torch.meshgrid(self.bev_x, self.bev_y, indexing='xy')
        self.register_buffer('grid_x', xx)  # (H, W)
        self.register_buffer('grid_y', yy)  # (H, W)

    def world_to_bev(self, Xw, Yw):
        '''
        Convert world coordinates to BEV pixel coordinates.

        Args:
            Xw: world X coordinate (forward)
            Yw: world Y coordinate (left)

        Returns:
            u: BEV column (0-255)
            v: BEV row (0-255)
        '''
        u = 128 + (Xw / self.world_scale)
        v = 128 - (Yw / self.world_scale)
        return u, v

    def bev_to_world(self, u, v):
        '''Convert BEV pixel coordinates to world coordinates.'''
        Xw = (u - 128) * self.world_scale
        Yw = (128 - v) * self.world_scale
        return Xw, Yw

    def reset_memory(self, batch_size: int, device: torch.device):
        '''Initialize empty BEV memory.'''
        return torch.zeros(batch_size, 3, self.bev_h, self.bev_w, device=device)

    def compute_exploration_layer(self, pose):
        '''
        Compute exploration layer from UAV pose with 20m FOV decay.

        Args:
            pose: (b, 4) UAV pose (Xw, Yw, Zw, yaw)

        Returns:
            exploration: (b, 256, 256) exploration probability map
        '''
        b = pose.shape[0]
        device = pose.device

        # UAV position in world coordinates
        uav_x = pose[:, 0]  # (b,)
        uav_y = pose[:, 1]  # (b,)
        uav_yaw = pose[:, 3]  # (b,), radians

        # Expand grids for batch
        grid_x = self.grid_x[None, :, :].expand(b, -1, -1)  # (b, 256, 256)
        grid_y = self.grid_y[None, :, :].expand(b, -1, -1)

        # Relative position from UAV
        dx = grid_x - uav_x[:, None, None]
        dy = grid_y - uav_y[:, None, None]

        # Distance from UAV
        distance = torch.sqrt(dx**2 + dy**2)  # (b, 256, 256)

        # Angle from UAV forward direction
        angle_to_point = torch.atan2(dy, dx)  # (b, 256, 256)
        angle_diff = angle_to_point - uav_yaw[:, None, None]
        angle_diff = ((angle_diff + math.pi) % (2 * math.pi)) - math.pi  # wrap to [-pi, pi]

        # FOV mask: within angle and range
        in_fov_angle = torch.abs(angle_diff) < math.radians(self.fov_angle / 2)
        in_fov_range = distance <= self.fov_range
        in_fov = in_fov_angle & in_fov_range  # (b, 256, 256)

        # Distance-based decay (linear decay from 1 to 0)
        distance_decay = 1.0 - (distance / self.fov_range)
        distance_decay = torch.clamp(distance_decay, 0.0, 1.0)

        # Angle-based decay (cosine decay from 1 to 0 at edges)
        angle_decay = torch.cos(angle_diff * (math.pi / math.radians(self.fov_angle)))
        angle_decay = torch.clamp(angle_decay, 0.0, 1.0)

        # Combine decays
        exploration = torch.zeros_like(distance)
        exploration[in_fov] = distance_decay[in_fov] * angle_decay[in_fov]

        return exploration

    def update_exploration(self, old_exploration, pose):
        '''
        Update exploration layer with new observation.

        Args:
            old_exploration: (b, 256, 256)
            pose: (b, 4) UAV pose

        Returns:
            new_exploration: (b, 256, 256)
        '''
        new_observation = self.compute_exploration_layer(pose)
        new_exploration = torch.max(old_exploration, new_observation)
        return new_exploration

    def compute_obstacle_layer_from_depth(self, depth_map, pose, camera_intrinsics=None):
        '''
        Compute obstacle layer from depth map (for ground truth generation).

        Args:
            depth_map: (b, h, w) depth map in meters
            pose: (b, 4) UAV pose
            camera_intrinsics: optional camera intrinsics

        Returns:
            obstacle: (b, 256, 256) obstacle probability map
        '''
        b = depth_map.shape[0]
        device = depth_map.device

        obstacle = torch.zeros(b, self.bev_h, self.bev_w, device=device)

        # Simple heuristic: high depth values = obstacle
        # Project depth points to BEV (simplified version)
        h, w = depth_map.shape[1], depth_map.shape[2]

        # Create pixel coordinates
        v, u = torch.meshgrid(torch.arange(h, device=device),
                              torch.arange(w, device=device),
                              indexing='ij')

        # Normalize to [-1, 1]
        u_norm = (u.float() / (w - 1)) * 2 - 1
        v_norm = (v.float() / (h - 1)) * 2 - 1

        # Assume camera is facing forward (simplified projection)
        # Xw = depth (forward), Yw from u (horizontal), height from v
        for i in range(b):
            depth = depth_map[i]  # (h, w)

            # Only consider points with valid depth
            valid = (depth > 0.1) & (depth < 30.0)

            if valid.sum() > 0:
                # Project to world coordinates relative to UAV
                Xw_rel = depth[valid]  # forward = depth
                Yw_rel = -u_norm[valid] * depth[valid] * 0.5  # scale factor

                # Add UAV pose
                Xw = Xw_rel * torch.cos(pose[i, 3]) - Yw_rel * torch.sin(pose[i, 3]) + pose[i, 0]
                Yw = Xw_rel * torch.sin(pose[i, 3]) + Yw_rel * torch.cos(pose[i, 3]) + pose[i, 1]

                # Convert to BEV coordinates
                u_bev, v_bev = self.world_to_bev(Xw, Yw)

                # Clamp to BEV bounds
                u_bev = torch.clamp(u_bev.long(), 0, self.bev_w - 1)
                v_bev = torch.clamp(v_bev.long(), 0, self.bev_h - 1)

                # Mark as obstacle (height threshold)
                # Assume points above 0.5m are obstacles
                Zw = 1.0 - v_norm[valid] * depth[valid] * 0.3  # approximate height
                is_obstacle = Zw > 0.5

                # Update obstacle layer
                obstacle[i, v_bev[is_obstacle], u_bev[is_obstacle]] = 1.0

        # Apply Gaussian blur to smooth obstacle boundaries
        obstacle = self._gaussian_blur(obstacle.unsqueeze(1), sigma=1.0).squeeze(1)

        return obstacle

    def _gaussian_blur(self, x, sigma=1.0):
        '''Apply Gaussian blur to tensor.'''
        kernel_size = int(sigma * 3) * 2 + 1
        channels = x.shape[1]

        # Create Gaussian kernel
        kernel = torch.arange(kernel_size, device=x.device) - kernel_size // 2
        kernel = torch.exp(-kernel**2 / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        kernel_2d = kernel[:, None] * kernel[None, :]
        kernel_2d = kernel_2d[None, None, :, :].repeat(channels, 1, 1, 1)

        # Apply convolution
        padding = kernel_size // 2
        blurred = F.conv2d(x, kernel_2d, padding=padding, groups=channels)
        return blurred

    def compute_value_layer_gt(self, exploration, obstacle, target_position):
        '''
        Ground-truth value layer, built as the training target V*.

        Delegates to model.loss.value_target.build_value_target so that this
        module and the training loss share one implementation of
        eq. (valuetarget): explored cells decay, obstacle cells are infeasible,
        cells near the target are boosted by a Gaussian prior, and the layer is
        rescaled to unit maximum.

        Args:
            exploration: (b, H, W) exploration layer in [0, 1]
            obstacle: (b, H, W) obstacle layer in [0, 1]
            target_position: (b, 2) target (Xw, Yw) in world coordinates

        Returns:
            value: (b, H, W) value layer in [0, 1]
        '''
        from model.loss.value_target import build_value_target
        return build_value_target(
            exploration, obstacle, target_position,
            bev_size=self.bev_h, bev_resolution=self.world_scale)

    def forward(self, pose, depth_map=None, target_position=None, bev_memory=None):
        '''
        Update BEV memory from observations.

        Args:
            pose: (b, 4) UAV pose
            depth_map: (b, h, w) optional depth map for obstacle layer
            target_position: (b, 2) optional target position for value layer GT
            bev_memory: (b, 3, 256, 256) previous BEV memory, or None

        Returns:
            bev_memory: (b, 3, 256, 256) updated BEV memory
        '''
        b = pose.shape[0]
        device = pose.device

        if bev_memory is None:
            bev_memory = self.reset_memory(b, device)

        # Update exploration layer (always)
        bev_memory[:, 0] = self.update_exploration(bev_memory[:, 0], pose)

        # Update obstacle layer (if depth map is provided)
        if depth_map is not None:
            bev_memory[:, 1] = self.compute_obstacle_layer_from_depth(depth_map, pose)

        # Update value layer (if target position is provided, for GT)
        if target_position is not None:
            bev_memory[:, 2] = self.compute_value_layer_gt(
                bev_memory[:, 0], bev_memory[:, 1], target_position)

        return bev_memory

