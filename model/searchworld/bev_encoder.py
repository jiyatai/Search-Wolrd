# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Dict

import gin
import torch
import torch.nn.functional as F
from torch import nn

from model.searchworld.utils import pack_sequence_dim, unpack_sequence_dim


@gin.configurable
class BEVEncoder(nn.Module):
    '''BEV encoder - encodes exploration + obstacle layers together.

    The full 3-channel BEV is preserved for decoder use.

    Architecture:
        Input: (N, 2, 256, 256) [exploration, obstacle]
        -> Conv(32, 4x4, stride=2) -> BN -> ReLU
        -> Conv(64, 4x4, stride=2) -> BN -> ReLU
        -> Conv(128, 4x4, stride=2) -> BN -> ReLU
        -> Conv(256, 4x4, stride=2) -> BN -> ReLU
        -> Conv(512, 4x4, stride=2) -> BN -> ReLU
        -> Flatten -> Linear(512*8*8 -> 1024) -> out

    Input:
        bev_spatial: (b, s, 2, H, W) or (b, s, 3, H, W)
                    - if 3-channel, first 2 are used

    Output:
        bev_features: (b, s, out_dim)
    '''
    def __init__(self,
                 bev_size: tuple = (256, 256),
                 hidden_dims: list = [32, 64, 128, 256, 512],
                 out_dim: int = 1024):
        super().__init__()

        layers = []
        in_channels = 2
        for h_dim in hidden_dims:
            layers.append(nn.Conv2d(in_channels, h_dim, kernel_size=4, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(h_dim))
            layers.append(nn.ReLU(True))
            in_channels = h_dim

        self.conv_layers = nn.Sequential(*layers)

        # Compute flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, 2, bev_size[0], bev_size[1])
            feat = self.conv_layers(dummy)
            self.flat_dim = feat.shape[1] * feat.shape[2] * feat.shape[3]

        self.proj = nn.Sequential(
            nn.Linear(self.flat_dim, out_dim * 2),
            nn.ReLU(True),
            nn.Linear(out_dim * 2, out_dim),
        )

        self.out_dim = out_dim

    def forward(self, bev_spatial: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            bev_spatial: (b, s, 2, H, W) or (b, s, 3, H, W)

        Returns:
            bev_features: (b, s, out_dim)
        '''
        # Handle single-step case without sequence dim
        has_seq_dim = len(bev_spatial.shape) == 5
        if not has_seq_dim:
            bev_spatial = bev_spatial.unsqueeze(1)

        b, s = bev_spatial.shape[:2]

        # Pack sequence
        bev = pack_sequence_dim(bev_spatial)  # (b*s, C, H, W)

        # Use only first 2 channels (exploration + obstacle)
        bev = bev[:, :2]  # (b*s, 2, H, W)

        # Encode
        x = self.conv_layers(bev)
        x = x.flatten(1)
        x = self.proj(x)  # (N, out_dim)

        # Unpack sequence
        x = unpack_sequence_dim(x, b, s)

        # Remove sequence dim if it wasn't there
        if not has_seq_dim:
            x = x.squeeze(1)

        return x

