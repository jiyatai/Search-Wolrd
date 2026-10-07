# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lightweight BEV grid constants and world<->cell affine map.

Kept dependency-free (pure torch) so both the loss package and the stage-3
planning package can share the same grid definition without importing the
heavy training modules.

The grid is the allocentric BEV used by the converters and the BEV decoder:
256 x 256 cells at 0.4 m/cell, centred on the episode start, with

    u = size/2 + X / res     (column, world +X to the right)
    v = size/2 - Y / res     (row,    world +Y up)
"""
import torch

BEV_SIZE = 256
BEV_RESOLUTION = 0.4          # metres per cell
BEV_EXTENT_M = BEV_SIZE * BEV_RESOLUTION   # 102.4 m


def grid_world_coords(size: int,
                      resolution: float,
                      device=None,
                      dtype=torch.float32):
    """World coordinates (metres) of every cell of a ``size x size`` BEV grid.

    Returns:
        X: (1, size) horizontal world coordinate per column.
        Y: (size, 1) vertical world coordinate per row.
    """
    half = size / 2.0
    idx = torch.arange(size, device=device, dtype=dtype)
    X = (idx - half) * resolution           # columns
    Y = (half - idx) * resolution           # rows
    return X.view(1, size), Y.view(size, 1)
