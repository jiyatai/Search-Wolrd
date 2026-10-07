"""
Simple BEV Memory module without gin dependency for testing.
"""
import torch
from torch import nn
import torch.nn.functional as F
import math


class BEVMemory(nn.Module):
    """BEV Memory for UAV target search.

    BEV Specifications:
    - Size: 256 x 256 pixels
    - Resolution: 0.4 meters per pixel
    - Coverage: 102.4m x 102.4m (from -51.2m to +51.2m)
    - Origin (0,0) in BEV = (-51.2m, +51.2m) in world
    - Center (128,128) in BEV = (0m, 0m) in world

    Three channels:
    [0]: exploration  - 已探索区域，20米视野范围内衰减
    [1]: obstacle     - 障碍物，从深度图生成
    [2]: value        - 价值，从探索层和目标位置计算

    Args:
        bev_size: tuple (height, width) of BEV map
        world_scale: meters per pixel in BEV
        fov_range: maximum FOV range in meters
    """
    def __init__(self,
                 bev_size=(256, 256),
                 world_scale=0.4,
                 fov_range=20.0,
                 fov_angle=90.0):
        super().__init__()

        self.bev_h, self.bev_w = bev_size
        self.world_scale = world_scale
        self.fov_range = fov_range
        self.fov_angle = fov_angle

        # Precompute coordinate grids
        # Note: these will be created on the correct device when reset_memory is called
        self.register_buffer('grid_x', torch.zeros(bev_size[1], bev_size[0]))
        self.register_buffer('grid_y', torch.zeros(bev_size[1], bev_size[0]))

        # Initialize grids properly
        self._init_grids()

    def _init_grids(self):
        """Initialize coordinate grids (affine, 0.4 m/cell).

        Single source of truth: ``model/loss/bev_grid.py``, i.e. cell centres
        ``X = (u - W/2) * res`` and ``Y = (H/2 - v) * res``.  NOTE:
        ``linspace(-51.2, 51.2, 256)`` would give a 0.4016 m step instead of
        0.4 m and misalign the layers with the converter's affine map by up to
        half a cell at the edges.
        """
        res = self.world_scale
        bev_x = (torch.arange(self.bev_w, dtype=torch.float32)
                 - self.bev_w / 2.0) * res
        bev_y = (self.bev_h / 2.0
                 - torch.arange(self.bev_h, dtype=torch.float32)) * res
        xx, yy = torch.meshgrid(bev_x, bev_y, indexing='xy')
        self.grid_x.copy_(xx)
        self.grid_y.copy_(yy)

    def to(self, *args, **kwargs):
        """Override to to ensure grids are moved too."""
        self = super().to(*args, **kwargs)
        self._init_grids()
        self.grid_x = self.grid_x.to(*args, **kwargs)
        self.grid_y = self.grid_y.to(*args, **kwargs)
        return self

    def world_to_bev(self, Xw, Yw):
        """
        Convert world coordinates to BEV pixel coordinates.

        Args:
            Xw: world X coordinate (forward)
            Yw: world Y coordinate (left)

        Returns:
            u: BEV column (0-255)
            v: BEV row (0-255)
        """
        u = self.bev_w / 2.0 + (Xw / self.world_scale)
        v = self.bev_h / 2.0 - (Yw / self.world_scale)
        return u, v

    def bev_to_world(self, u, v):
        """Convert BEV pixel coordinates to world coordinates."""
        Xw = (u - self.bev_w / 2.0) * self.world_scale
        Yw = (self.bev_h / 2.0 - v) * self.world_scale
        return Xw, Yw

    def reset_memory(self, batch_size, device):
        """Initialize empty BEV memory."""
        return torch.zeros(batch_size, 3, self.bev_h, self.bev_w, device=device)

    def compute_exploration_layer(self, pose):
        """
        Compute exploration layer from UAV pose with 20m FOV decay.

        Args:
            pose: (b, 4) UAV pose (Xw, Yw, Zw, yaw)

        Returns:
            exploration: (b, 256, 256) exploration probability map
        """
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
        # Normalize angle difference to [-pi, pi]
        angle_diff = ((angle_diff + math.pi) % (2 * math.pi)) - math.pi

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
        """
        Update exploration layer with new observation.

        Args:
            old_exploration: (b, 256, 256)
            pose: (b, 4) UAV pose

        Returns:
            new_exploration: (b, 256, 256)
        """
        new_observation = self.compute_exploration_layer(pose)
        new_exploration = torch.max(old_exploration, new_observation)
        return new_exploration

    def compute_obstacle_layer_from_depth(self, depth_map, pose):
        """
        Compute obstacle layer from depth map (for ground truth generation).

        Args:
            depth_map: (b, h, w) depth map in meters
            pose: (b, 4) UAV pose

        Returns:
            obstacle: (b, 256, 256) obstacle probability map (new observation only)
        """
        b = depth_map.shape[0]
        device = depth_map.device

        obstacle = torch.zeros(b, self.bev_h, self.bev_w, device=device)

        # Project depth points to BEV
        h, w = depth_map.shape[1], depth_map.shape[2]

        for i in range(b):
            depth = depth_map[i]  # (h, w)

            # Only consider points with valid depth
            valid = (depth > 0.1) & (depth < 30.0)

            if valid.sum() > 0:
                # Create pixel coordinates
                v_pix, u_pix = torch.meshgrid(torch.arange(h, device=device),
                                              torch.arange(w, device=device),
                                              indexing='ij')

                # Normalize to [-1, 1] for easier calculations
                u_norm = (u_pix.float() / (w - 1)) * 2 - 1
                v_norm = (v_pix.float() / (h - 1)) * 2 - 1

                # Filter valid points
                valid_flat = valid.flatten()
                depth_flat = depth.flatten()[valid_flat]
                u_norm_flat = u_norm.flatten()[valid_flat]
                v_norm_flat = v_norm.flatten()[valid_flat]

                # Project to world coordinates relative to UAV
                # Simplified camera model: forward = X, left = Y
                Xw_rel = depth_flat
                Yw_rel = -u_norm_flat * depth_flat * 0.5

                # Rotate by UAV yaw
                cos_yaw = torch.cos(pose[i, 3])
                sin_yaw = torch.sin(pose[i, 3])
                Xw = Xw_rel * cos_yaw - Yw_rel * sin_yaw + pose[i, 0]
                Yw = Xw_rel * sin_yaw + Yw_rel * cos_yaw + pose[i, 1]

                # Approximate height from v coordinate
                Zw = 1.0 - v_norm_flat * depth_flat * 0.3

                # Convert to BEV coordinates
                u_bev = 128 + (Xw / self.world_scale)
                v_bev = 128 - (Yw / self.world_scale)

                # Clamp to BEV bounds
                u_bev_clamped = torch.clamp(u_bev.long(), 0, self.bev_w - 1)
                v_bev_clamped = torch.clamp(v_bev.long(), 0, self.bev_h - 1)

                # Mark as obstacle if above height threshold
                is_obstacle = Zw > 0.5

                # Update obstacle layer
                obstacle[i, v_bev_clamped[is_obstacle], u_bev_clamped[is_obstacle]] = 1.0

        # Apply Gaussian blur to smooth
        obstacle = self._gaussian_blur(obstacle.unsqueeze(1), sigma=1.0).squeeze(1)

        return obstacle

    def update_obstacle(self, old_obstacle, new_obstacle):
        """
        Update obstacle layer with new observation, retaining previous info.

        Args:
            old_obstacle: (b, 256, 256)
            new_obstacle: (b, 256, 256)

        Returns:
            new_obstacle: (b, 256, 256)
        """
        return torch.max(old_obstacle, new_obstacle)

    def _gaussian_blur(self, x, sigma=1.0):
        """Apply Gaussian blur to tensor."""
        kernel_size = int(sigma * 3) * 2 + 1
        if kernel_size < 3:
            kernel_size = 3
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
        """
        Ground-truth value layer = the training target V* (paper eq. valuetarget).

        V*(x) = Norm[(1 - M_obs*) * (lambda_expl * (1 - M_expl*) + lambda_goal * G_g)]
        with G_g the goal Gaussian (sigma_g = 10 m) and Norm rescaling to unit
        maximum.  Kept inline so this offline copy stays dependency-free; keep
        it in sync with model/loss/value_target.py::build_value_target.

        Args:
            exploration: (b, H, W) exploration layer in [0, 1]
            obstacle: (b, H, W) obstacle layer in [0, 1]
            target_position: (b, 2) target (Xw, Yw) in world coordinates

        Returns:
            value: (b, H, W) value layer in [0, 1]
        """
        b = exploration.shape[0]
        tx = target_position[:, 0].reshape(b, 1, 1)
        ty = target_position[:, 1].reshape(b, 1, 1)

        dist2 = (self.grid_x[None] - tx) ** 2 + (self.grid_y[None] - ty) ** 2
        goal = torch.exp(-dist2 / (2.0 * 10.0 ** 2))

        feasibility = 1.0 - obstacle
        value = feasibility * ((1.0 - exploration) + goal)
        vmax = value.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
        return value / vmax

    def forward(self, pose, depth_map=None, target_position=None, bev_memory=None):
        """
        Update BEV memory from observations.

        Args:
            pose: (b, 4) UAV pose
            depth_map: (b, h, w) optional depth map for obstacle layer
            target_position: (b, 2) optional target position for value layer GT
            bev_memory: (b, 3, 256, 256) previous BEV memory, or None

        Returns:
            bev_memory: (b, 3, 256, 256) updated BEV memory
        """
        b = pose.shape[0]
        device = pose.device

        if bev_memory is None:
            bev_memory = self.reset_memory(b, device)

        # Update exploration layer (always)
        bev_memory[:, 0] = self.update_exploration(bev_memory[:, 0], pose)

        # Update obstacle layer (if depth map is provided)
        if depth_map is not None:
            new_obstacle = self.compute_obstacle_layer_from_depth(depth_map, pose)
            bev_memory[:, 1] = self.update_obstacle(bev_memory[:, 1], new_obstacle)

        # Update value layer (if target position is provided, for GT)
        if target_position is not None:
            bev_memory[:, 2] = self.compute_value_layer_gt(
                bev_memory[:, 0], bev_memory[:, 1], target_position)

        return bev_memory

