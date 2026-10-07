# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Motion-footprint geometry for value-guided imagination planning.

This module implements the *closed-form, non-learned* footprint geometry that
the paper's appendix ("Action footprint geometry") formalizes.  The decoded
task-aware spatial value layer lives on the allocentric BEV grid; a candidate
movement action ``a`` is scored by pooling that value layer over the action's
motion footprint ``K^a``:

    q_V(s_t, a) = sum_x K^a_{t+1}(x) * V_hat^{sp,a}_{t+1}(x).

Everything here is pure geometry (units: metres, radians):

  * the allocentric BEV grid is 256 x 256 cells at 0.4 m/cell and is centred on
    the episode start, matching ``scripts/convert_*_to_parquet.py``::

        u = size/2 + X / res      (column, world +X to the right)
        v = size/2 - Y / res      (row,    world +Y up)

  * the sensor fan is the canonical ego-frame sector of half-width
    ``FOV_HALF_ANGLE_DEG = 45 deg`` (= phi/2, phi = 90 deg) and range
    ``SENSOR_RANGE = 20 m`` (rho), matching the exploration layer in
    ``BEVGenerator``.

  * each atomic action carries a benchmark-fixed motion delta (see
    ``ACTION_DELTAS``).  The footprint is the fan *swept* along the motion
    delta: a translation sweeps the fan through space, a rotation sweeps it in
    place through the rotation angle.  ``n_sweep`` samples along the sweep are
    unioned; ``n_sweep == 1`` reduces to the fan at the initial pose.

NOTE ON THE ACTION DELTAS: the paper's appendix states 5 m translations,
2 m vertical translations and 15 deg rotations, fixed by the UAV-ON benchmark.
The values below are the ones we could verify in this repository (the discrete
action ids used by the converters and the world model).  If the training
machine's simulator uses different magnitudes, override ``ACTION_DELTAS`` /
``D_TRANS`` / ``D_VERT`` / ``THETA_ROT`` here (single source of truth) rather
than editing the engine.
"""
import math
from typing import Dict, Sequence, Tuple

import torch

from model.loss.bev_grid import BEV_RESOLUTION, BEV_SIZE, grid_world_coords

# ---------------------------------------------------------------------------
# BEV grid / sensing geometry (paper appendix "Implementation Details").
# ---------------------------------------------------------------------------
SENSOR_RANGE = 20.0           # metres, rho
FOV_HALF_ANGLE_DEG = 45.0     # degrees, phi/2 (full FOV = 90 deg)

# Number of samples along the motion delta used to approximate the swept
# union.  n_sweep == 1 -> the fan at the initial pose (no sweep).
N_SWEEP = 5

# ---------------------------------------------------------------------------
# Discrete action set.  Action ids match scripts/convert_*_to_parquet.py:
#   0 forward, 1 left, 2 right, 3 ascend, 4 descend, 5 rotl, 6 rotr,
#   7 stop. The stop class is excluded from movement reweighting.
# ---------------------------------------------------------------------------
D_TRANS = 5.0                                   # metres
D_VERT = 2.0                                    # metres
THETA_ROT = math.radians(15.0)                  # radians

N_ACTIONS = 8
STOP_ACTION = 7
MOVE_ACTIONS: Tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)

# Ego frame: +x forward, +y left.  Values are (dx, dy, dtheta).
#   forward  : 5 m along the heading
#   left     : 5 m to the left   (strafe)
#   right    : 5 m to the right  (strafe)
#   ascend / descend : 2 m vertical.  The BEV is a top-down planar grid, so the
#       ground projection of the fan is unchanged; the footprint is the fan at
#       the current (x, y) (paper: "the fan projected at the new altitude").
#   rotl / rotr : rotate the fan in place by +/- 15 deg.
ACTION_DELTAS: Dict[int, Tuple[float, float, float]] = {
    0: (D_TRANS, 0.0, 0.0),
    1: (0.0, D_TRANS, 0.0),
    2: (0.0, -D_TRANS, 0.0),
    3: (0.0, 0.0, 0.0),
    4: (0.0, 0.0, 0.0),
    5: (0.0, 0.0, THETA_ROT),
    6: (0.0, 0.0, -THETA_ROT),
    7: (0.0, 0.0, 0.0),
}


def wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    """Wrap angle(s) to (-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def world_to_ego(X: torch.Tensor,
                 Y: torch.Tensor,
                 pose: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Map world coordinates to the ego frame of ``pose = (x, y, yaw)``.

    Ego frame: +x forward, +y left.  ``e = R(-yaw) (P - p)``.
    """
    x, y, yaw = pose[0], pose[1], pose[2]
    vx = X - x
    vy = Y - y
    c = torch.cos(yaw)
    s = torch.sin(yaw)
    ego_x = c * vx + s * vy
    ego_y = -s * vx + c * vy
    return ego_x, ego_y


def fan_membership(ego_x: torch.Tensor,
                   ego_y: torch.Tensor,
                   sensor_range: float,
                   half_angle_rad: float) -> torch.Tensor:
    """Boolean mask of points inside the canonical forward sensor fan."""
    r = torch.sqrt(ego_x * ego_x + ego_y * ego_y)
    ang = torch.atan2(ego_y, ego_x)
    return (r <= sensor_range) & (ang.abs() <= half_angle_rad)


def footprint_mask(pose: torch.Tensor,
                   action: int,
                   bev_size: int = BEV_SIZE,
                   bev_resolution: float = BEV_RESOLUTION,
                   sensor_range: float = SENSOR_RANGE,
                   fov_half_angle_deg: float = FOV_HALF_ANGLE_DEG,
                   n_sweep: int = N_SWEEP,
                   action_deltas: Dict[int, Tuple[float, float, float]] = None
                   ) -> torch.Tensor:
    """Binary motion footprint of one action at one pose, on the BEV grid.

    Args:
        pose: (3,) tensor ``(x, y, yaw)`` in the allocentric world frame
            (metres, radians), matching the BEV grid centre.
        action: discrete action id.
        bev_size / bev_resolution: grid geometry.
        sensor_range / fov_half_angle_deg: sensor fan geometry.
        n_sweep: number of samples along the motion delta (1 = no sweep).
        action_deltas: optional override of ``ACTION_DELTAS``.

    Returns:
        (bev_size, bev_size) float tensor in {0, 1}; out-of-grid cells are
        implicitly dropped because only in-grid cells exist.
    """
    deltas = ACTION_DELTAS if action_deltas is None else action_deltas
    dx, dy, dpsi = deltas[int(action)]

    X, Y = grid_world_coords(bev_size, bev_resolution,
                             device=pose.device, dtype=pose.dtype)
    ego_x, ego_y = world_to_ego(X, Y, pose)

    half = math.radians(float(fov_half_angle_deg))
    mask = torch.zeros_like(ego_x, dtype=torch.bool)

    ts = torch.linspace(0.0, 1.0, int(n_sweep),
                        device=pose.device, dtype=pose.dtype)
    for t in ts:
        tf = float(t)
        # Undo the translation: the fan at sweep position t covers q iff the
        # query point translated back by t*delta is inside the fan at t=0.
        qx = ego_x - tf * dx
        qy = ego_y - tf * dy
        # Undo the in-place rotation: the fan at angle t*dpsi covers q iff
        # R(-t*dpsi) q is inside the canonical fan.
        th = tf * dpsi
        ct = math.cos(th)
        st = math.sin(th)
        rx = ct * qx + st * qy
        ry = -st * qx + ct * qy
        mask = mask | fan_membership(rx, ry, sensor_range, half)

    return mask.to(pose.dtype)


def footprint_masks(poses: torch.Tensor,
                    action: int,
                    bev_size: int = BEV_SIZE,
                    bev_resolution: float = BEV_RESOLUTION,
                    sensor_range: float = SENSOR_RANGE,
                    fov_half_angle_deg: float = FOV_HALF_ANGLE_DEG,
                    n_sweep: int = N_SWEEP,
                    action_deltas: Dict[int, Tuple[float, float, float]] = None
                    ) -> torch.Tensor:
    """Vectorized :func:`footprint_mask` over ``poses`` (N, 3) -> (N, H, W)."""
    deltas = ACTION_DELTAS if action_deltas is None else action_deltas
    dx, dy, dpsi = deltas[int(action)]

    n = poses.shape[0]
    X, Y = grid_world_coords(bev_size, bev_resolution,
                             device=poses.device, dtype=poses.dtype)
    X = X.view(1, 1, bev_size)
    Y = Y.view(1, bev_size, 1)
    x = poses[:, 0].view(n, 1, 1)
    y = poses[:, 1].view(n, 1, 1)
    yaw = poses[:, 2].view(n, 1, 1)

    vx = X - x
    vy = Y - y
    c = torch.cos(yaw)
    s = torch.sin(yaw)
    ego_x = c * vx + s * vy
    ego_y = -s * vx + c * vy

    half = math.radians(float(fov_half_angle_deg))
    mask = torch.zeros(n, bev_size, bev_size,
                       dtype=torch.bool, device=poses.device)
    ts = torch.linspace(0.0, 1.0, int(n_sweep),
                        device=poses.device, dtype=poses.dtype)
    for t in ts:
        tf = float(t)
        qx = ego_x - tf * dx
        qy = ego_y - tf * dy
        th = tf * dpsi
        ct = math.cos(th)
        st = math.sin(th)
        rx = ct * qx + st * qy
        ry = -st * qx + ct * qy
        mask = mask | fan_membership(rx, ry, sensor_range, half)
    return mask.to(poses.dtype)


def candidate_footprints(poses: torch.Tensor,
                         move_actions: Sequence[int] = MOVE_ACTIONS,
                         **kwargs) -> torch.Tensor:
    """All candidate movement footprints for a batch of poses.

    Returns (N, len(move_actions), bev_size, bev_size).
    """
    per_action = [footprint_masks(poses, a, **kwargs) for a in move_actions]
    return torch.stack(per_action, dim=1)


def step_pose(pose: torch.Tensor,
              action: int,
              action_deltas: Dict[int, Tuple[float, float, float]] = None
              ) -> torch.Tensor:
    """Compose one pose (3,) with the ego-frame delta of ``action``.

    ``p' = p (+) Delta_a``: the successor pose.  World-frame displacement is
    ``R(yaw) @ (dx, dy)``; yaw is advanced by ``dpsi``.
    """
    deltas = ACTION_DELTAS if action_deltas is None else action_deltas
    dx, dy, dpsi = deltas[int(action)]
    x, y, yaw = pose[0], pose[1], pose[2]
    c = torch.cos(yaw)
    s = torch.sin(yaw)
    wx = c * dx - s * dy
    wy = s * dx + c * dy
    out = torch.stack([x + wx, y + wy, yaw + dpsi])
    out = torch.stack([out[0], out[1], wrap_angle(out[2])])
    return out


def step_pose_batch(poses: torch.Tensor,
                    actions: torch.Tensor,
                    action_deltas: Dict[int, Tuple[float, float, float]] = None
                    ) -> torch.Tensor:
    """Vectorized :func:`step_pose` for (N, 3) poses and (N,) action ids."""
    deltas = ACTION_DELTAS if action_deltas is None else action_deltas
    dev = poses.device
    dx = torch.tensor([deltas[0][0], deltas[1][0], deltas[2][0], deltas[3][0],
                       deltas[4][0], deltas[5][0], deltas[6][0], deltas[7][0]],
                      device=dev, dtype=poses.dtype)
    dy = torch.tensor([deltas[0][1], deltas[1][1], deltas[2][1], deltas[3][1],
                       deltas[4][1], deltas[5][1], deltas[6][1], deltas[7][1]],
                      device=dev, dtype=poses.dtype)
    dpsi = torch.tensor([deltas[0][2], deltas[1][2], deltas[2][2], deltas[3][2],
                         deltas[4][2], deltas[5][2], deltas[6][2], deltas[7][2]],
                        device=dev, dtype=poses.dtype)
    yaw = poses[:, 2]
    c = torch.cos(yaw)
    s = torch.sin(yaw)
    adx = dx[actions]
    ady = dy[actions]
    wx = c * adx - s * ady
    wy = s * adx + c * ady
    x = poses[:, 0] + wx
    y = poses[:, 1] + wy
    yaw_out = wrap_angle(yaw + dpsi[actions])
    return torch.stack([x, y, yaw_out], dim=-1)


# ---------------------------------------------------------------------------
# Value-guided policy improvement (paper eq. pi-plus / eq: actionloss).
# ---------------------------------------------------------------------------
def build_pi_plus(pi_bc: torch.Tensor,
                  q_v: torch.Tensor,
                  tau: float,
                  move_actions: Sequence[int] = MOVE_ACTIONS,
                  stop_action: int = STOP_ACTION,
                  eps: float = 1e-8):
    """Construct the value-guided target policy ``pi+``.

    ``pi+(a) = pi_BC(a) * exp(q_V(a) / tau)`` over movement actions only,
    renormalized within the movement sub-distribution, with ``stop`` held at
    ``pi_BC(stop)`` (paper eq. pi-plus and appendix "The value-guided update").

    Args:
        pi_bc: (..., A) full behaviour-cloned distribution over A actions
            (probabilities, sums to 1 over all actions).
        q_v: (..., M) action utilities for the ``move_actions`` (M = |A_move|).
        tau: temperature (> 0).

    Returns:
        pi_plus_full: (..., A) with ``pi_plus_full[..., stop] == pi_bc[..., stop]``
            and the movement entries summing to ``1 - pi_bc(stop)``.
        pi_plus_move: (..., M) movement sub-distribution (sums to 1).
    """
    move_actions = list(move_actions)
    bc_move = pi_bc[..., move_actions]                       # (..., M)
    stop_mass = pi_bc[..., stop_action]                      # (...)
    logits = torch.log(bc_move.clamp_min(eps)) + q_v / tau
    pi_plus_move = torch.softmax(logits, dim=-1)             # (..., M)

    full = torch.zeros_like(pi_bc)
    full[..., move_actions] = pi_plus_move * (1.0 - stop_mass).unsqueeze(-1)
    full[..., stop_action] = stop_mass
    return full, pi_plus_move


def movement_kl(pi_plus_move: torch.Tensor,
                pi_theta_logits: torch.Tensor,
                move_actions: Sequence[int] = MOVE_ACTIONS,
                stop_action: int = STOP_ACTION,
                eps: float = 1e-8) -> torch.Tensor:
    """KL(pi+ || pi_theta) restricted to the movement actions (paper eq. actionloss).

    Compares the *conditional* movement sub-distributions, so the pinned stop
    probability does not enter the loss (both pi+ and pi_theta carry stop mass;
    only movement masses are reallocated).

    Returns: (...,) per-sample KL values.
    """
    move_actions = list(move_actions)
    theta = torch.softmax(pi_theta_logits, dim=-1)
    stop_theta = theta[..., stop_action]
    denom = (1.0 - stop_theta).clamp_min(eps).unsqueeze(-1)
    theta_move = theta[..., move_actions] / denom
    log_p = torch.log(pi_plus_move.clamp_min(eps))
    log_q = torch.log(theta_move.clamp_min(eps))
    return (pi_plus_move * (log_p - log_q)).sum(dim=-1)


def policy_entropy(logits: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Categorical entropy H(softmax(logits)) for monitoring."""
    logp = torch.log_softmax(logits, dim=-1)
    p = logp.exp()
    return -(p * logp).sum(dim=-1)
