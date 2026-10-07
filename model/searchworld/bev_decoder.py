# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gin
import torch
import torch.nn.functional as F
from torch import nn


@gin.configurable
class BEVDecoder(nn.Module):
    '''BEV decoder - decodes RSSM state to 3-channel BEV map.

    Architecture:
        Input: (N, state_dim) RSSM latent state
        -> Linear(state_dim -> 512*8*8)
        -> Reshape (512, 8, 8)
        -> ConvTranspose(512, 256, 4x4, stride=2, padding=1) -> BN -> ReLU
        -> ConvTranspose(256, 128, 4x4, stride=2, padding=1) -> BN -> ReLU
        -> ConvTranspose(128, 64, 4x4, stride=2, padding=1) -> BN -> ReLU
        -> ConvTranspose(64, 32, 4x4, stride=2, padding=1) -> BN -> ReLU
        -> ConvTranspose(32, 3, 4x4, stride=2, padding=1)
        -> Output: (N, 3, 256, 256)

    Channel 0: exploration (sigmoid: [0, 1])
    Channel 1: obstacle (sigmoid: [0, 1])
    Channel 2: value (tanh: [-1, 1] or sigmoid: [0, 1])

    Input:
        state: (b, s, state_dim) RSSM latent state

    Output:
        bev_pred: (b, s, 3, 256, 256) predicted BEV
    '''
    def __init__(self,
                 state_dim: int = 1536,
                 bev_size: tuple = (256, 256),
                 hidden_dims: list = [512, 256, 128, 64, 32],
                 value_use_tanh: bool = False):
        super().__init__()

        self.bev_h, self.bev_w = bev_size
        self.value_use_tanh = value_use_tanh

        # Initial projection
        initial_size = (hidden_dims[0], 8, 8)
        self.proj = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0] * 8 * 8),
            nn.ReLU(True),
        )
        self.initial_shape = initial_size

        # Decoder layers
        layers = []
        in_dim = hidden_dims[0]
        for out_dim in hidden_dims[1:]:
            layers.append(nn.ConvTranspose2d(in_dim, out_dim, kernel_size=4, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_dim))
            layers.append(nn.ReLU(True))
            in_dim = out_dim

        self.decoder_layers = nn.Sequential(*layers)

        # Final convolution to 3 channels
        self.final_conv = nn.ConvTranspose2d(hidden_dims[-1], 3, kernel_size=4, stride=2, padding=1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            state: (b, s, state_dim) or (b, state_dim) RSSM latent state

        Returns:
            bev_pred: (b, s, 3, 256, 256) or (b, 3, 256, 256)
        '''
        # Handle single-step case without sequence dim
        has_seq_dim = len(state.shape) == 3
        if not has_seq_dim:
            state = state.unsqueeze(1)

        b, s = state.shape[:2]

        # Pack sequence
        state_packed = state.reshape(b * s, -1)  # (b*s, state_dim)

        # Project and reshape
        x = self.proj(state_packed)
        x = x.reshape(b * s, self.initial_shape[0], self.initial_shape[1], self.initial_shape[2])

        # Decode
        x = self.decoder_layers(x)
        x = self.final_conv(x)  # (b*s, 3, 256, 256)

        # Split into channels with appropriate activations
        exploration = torch.sigmoid(x[:, 0:1])  # (b*s, 1, 256, 256)
        obstacle = torch.sigmoid(x[:, 1:2])    # (b*s, 1, 256, 256)

        if self.value_use_tanh:
            value = torch.tanh(x[:, 2:3])  # (b*s, 1, 256, 256) - [-1, 1]
        else:
            value = torch.sigmoid(x[:, 2:3])  # (b*s, 1, 256, 256) - [0, 1]

        # Combine channels
        bev_pred = torch.cat([exploration, obstacle, value], dim=1)  # (b*s, 3, 256, 256)

        # Unpack sequence
        bev_pred = bev_pred.reshape(b, s, 3, self.bev_h, self.bev_w)

        # Remove sequence dim if it wasn't there
        if not has_seq_dim:
            bev_pred = bev_pred.squeeze(1)

        return bev_pred


@gin.configurable
class BEVDecoderMultiScale(nn.Module):
    '''BEV decoder with multi-scale supervision.

    Architecture:
        Input: (N, state_dim) RSSM latent state
        -> Linear(state_dim -> 512*8*8)
        -> Reshape (512, 8, 8)
        -> ConvTranspose(512, 256, 4x4, stride=2, padding=1) -> BN -> ReLU
           └-> skip conv -> 3-channel BEV (16x16) - supervision
        -> ConvTranspose(256, 128, 4x4, stride=2, padding=1) -> BN -> ReLU
           └-> skip conv -> 3-channel BEV (32x32) - supervision
        -> ConvTranspose(128, 64, 4x4, stride=2, padding=1) -> BN -> ReLU
           └-> skip conv -> 3-channel BEV (64x64) - supervision
        -> ConvTranspose(64, 32, 4x4, stride=2, padding=1) -> BN -> ReLU
           └-> skip conv -> 3-channel BEV (128x128) - supervision
        -> ConvTranspose(32, 3, 4x4, stride=2, padding=1)
        -> Output: (N, 3, 256, 256)

    Input:
        state: (b, s, state_dim) RSSM latent state

    Output:
        bev_pred: dict containing:
            'bev_256': (b, s, 3, 256, 256)
            'bev_128': (b, s, 3, 128, 128)
            'bev_64': (b, s, 3, 64, 64)
            'bev_32': (b, s, 3, 32, 32)
            'bev_16': (b, s, 3, 16, 16)
    '''
    def __init__(self,
                 state_dim: int = 1536,
                 bev_size: tuple = (256, 256),
                 hidden_dims: list = [512, 256, 128, 64, 32],
                 value_use_tanh: bool = False):
        super().__init__()

        self.bev_h, self.bev_w = bev_size
        self.value_use_tanh = value_use_tanh

        # Initial projection
        self.proj = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0] * 8 * 8),
            nn.ReLU(True),
        )
        self.initial_shape = (hidden_dims[0], 8, 8)

        # Decoder layers with skip connections
        self.decoder_blocks = nn.ModuleList()
        self.skip_convs = nn.ModuleList()

        in_dim = hidden_dims[0]
        for i, out_dim in enumerate(hidden_dims[1:]):
            self.decoder_blocks.append(nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(True),
            ))
            # Skip connection to predict BEV at this scale
            self.skip_convs.append(nn.Conv2d(out_dim, 3, kernel_size=1))
            in_dim = out_dim

        # Final convolution
        self.final_conv = nn.ConvTranspose2d(hidden_dims[-1], 3, kernel_size=4, stride=2, padding=1)

    def forward(self, state: torch.Tensor) -> dict:
        '''
        Args:
            state: (b, s, state_dim) RSSM latent state

        Returns:
            bev_pred: dict with multi-scale BEV predictions
        '''
        has_seq_dim = len(state.shape) == 3
        if not has_seq_dim:
            state = state.unsqueeze(1)

        b, s = state.shape[:2]
        state_packed = state.reshape(b * s, -1)

        # Project and reshape
        x = self.proj(state_packed)
        x = x.reshape(b * s, self.initial_shape[0], self.initial_shape[1], self.initial_shape[2])

        # Decode with multi-scale supervision
        outputs = {}
        scales = [16, 32, 64, 128]

        for i, (block, skip_conv) in enumerate(zip(self.decoder_blocks, self.skip_convs)):
            x = block(x)
            # Predict BEV at this scale
            bev_skip = skip_conv(x)
            # Apply activations.
            # NOTE: build the activated tensor with torch.cat instead of
            # in-place slice assignment (`bev_skip[:, 0:1] = ...`), which bumps
            # the version counter of the conv output and can make autograd fail
            # with "one of the variables needed for gradient computation has
            # been modified by an inplace operation".
            expl = torch.sigmoid(bev_skip[:, 0:1])
            obst = torch.sigmoid(bev_skip[:, 1:2])
            if self.value_use_tanh:
                value = torch.tanh(bev_skip[:, 2:3])
            else:
                value = torch.sigmoid(bev_skip[:, 2:3])
            bev_skip = torch.cat([expl, obst, value], dim=1)
            # Reshape and store
            bev_skip = bev_skip.reshape(b, s, 3, scales[i], scales[i])
            outputs[f'bev_{scales[i]}'] = bev_skip

        # Final BEV
        x = self.final_conv(x)
        exploration = torch.sigmoid(x[:, 0:1])
        obstacle = torch.sigmoid(x[:, 1:2])
        if self.value_use_tanh:
            value = torch.tanh(x[:, 2:3])
        else:
            value = torch.sigmoid(x[:, 2:3])
        bev_pred = torch.cat([exploration, obstacle, value], dim=1)
        bev_pred = bev_pred.reshape(b, s, 3, 256, 256)
        outputs['bev_256'] = bev_pred
        if not has_seq_dim:
            for k in outputs:
                outputs[k] = outputs[k].squeeze(1)

        return outputs

