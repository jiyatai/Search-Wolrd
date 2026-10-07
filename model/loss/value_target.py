# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Training-time supervision target for the decoded spatial value layer.

Implements the paper's eq. (valuetarget):

    V*(x) = Norm[ (1 - M_obs*(x)) * ( lambda_expl * (1 - M_expl*(x))
                                       + lambda_goal * G_g(x) ) ]
    G_g(x) = exp( -||x - g||^2 / (2 sigma_g^2) )

with ``Norm`` rescaling the layer to unit maximum.  The exploration and
obstacle layers are the geometrically constructed ground-truth labels (the
starred quantities); ``g`` is the ground-truth target position expressed in the
same allocentric BEV frame as the grid (metres relative to the episode start,
which is the grid centre).  The target coordinate enters only through the
training label: at inference the decoder runs on the observation streams alone.

The grid uses the converter's affine map (``grid_world_coords`` in
``model/loss/bev_grid.py``): ``u = size/2 + X/res``, ``v = size/2 - Y/res``.
"""
import torch

from model.loss.bev_grid import grid_world_coords


def build_value_target(exploration_gt: torch.Tensor,
                       obstacle_gt: torch.Tensor,
                       target_rel: torch.Tensor,
                       lambda_expl: float = 1.0,
                       lambda_goal: float = 1.0,
                       sigma_g: float = 10.0,
                       bev_size: int = 256,
                       bev_resolution: float = 0.4,
                       eps: float = 1e-8) -> torch.Tensor:
    """Build the value-layer supervision target ``V*``.

    Args:
        exploration_gt: (..., H, W) exploration label ``M_expl*`` in [0, 1].
        obstacle_gt: (..., H, W) obstacle label ``M_obs*`` in [0, 1].
        target_rel: (..., 2) target position ``g`` in world metres relative to
            the grid centre (per-sample leading dims must match the layers).
        lambda_expl / lambda_goal / sigma_g: target coefficients.
        bev_size / bev_resolution: grid geometry (cell -> metre conversion).

    Returns:
        (..., H, W) target in [0, 1] with unit maximum (zero everywhere if the
        feasibility mask is empty).
    """
    h, w = exploration_gt.shape[-2:]
    dev = exploration_gt.device
    dtype = exploration_gt.dtype

    X, Y = grid_world_coords(w, bev_resolution, device=dev, dtype=dtype)
    X = X.view(1, w)
    Y = Y.view(h, 1)

    tx = target_rel[..., 0].reshape(*target_rel.shape[:-1], 1, 1)
    ty = target_rel[..., 1].reshape(*target_rel.shape[:-1], 1, 1)
    gaussian = torch.exp(-((X - tx) ** 2 + (Y - ty) ** 2) /
                         (2.0 * float(sigma_g) ** 2))

    feasibility = 1.0 - obstacle_gt
    value = feasibility * (float(lambda_expl) * (1.0 - exploration_gt)
                           + float(lambda_goal) * gaussian)
    vmax = value.amax(dim=(-2, -1), keepdim=True).clamp_min(eps)
    return value / vmax
