# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Value-guided imagination planning engine (stage 3).

This replaces the old exploration-gain REINFORCE engine.  Stage 3 carries

    * NO reward function,
    * NO scalar critic and no critic regression loss,
    * NO GAE / lambda-returns,
    * NO entropy bonus and no collision-termination pose head.

Instead the frozen world model imagines successor states, the frozen decoded
task-aware spatial value layer (the third BEV channel) is pooled over each
candidate action's motion footprint, and the behaviour-cloned policy is
reweighted by the resulting utilities:

    q_V(s_t, a) = sum_x K^a_{t+1}(x) * V_hat^{sp,a}_{t+1}(x)
    pi+(a)      ∝ pi_BC(a) * exp(Q_V^{(H)}(s_t, a) / tau)     (A_move only)
    L_action    = KL( pi+ || pi_theta )                       (A_move only)

with ``stop`` held at ``pi_BC(stop)`` and the movement probabilities
renormalized within the movement sub-distribution (paper eq. pi-plus and
eq. actionloss; appendix "Theoretical Details").

What is frozen / trainable
--------------------------
* RSSM, observation encoder and BEV (value) decoder: FROZEN, eval mode.
* The action head (``ContinuedActionPolicy`` warm-started from the Stage-2 BC
  head, or ``ImagRLActor`` for the fresh-actor ablation): TRAINABLE.
* The Stage-2 behaviour-cloned head ``pi_BC``: FROZEN; it defines the task
  prior and the pinned stop probability.

Pose handling
-------------
The motion footprints are anchored to the agent pose ``p_t = (x_t, y_t,
psi_t)`` associated with the current (posterior) state.  Stage 3 starts from
posterior states of real batches, so ``p_0`` is the recorded pose of the last
real step and is propagated through the imagined actions by composing the
benchmark-fixed motion deltas (``model/rl/footprint.py``).  The pose is a
training-time geometry input only.

Gradients
---------
Only the actor logits carry gradient.  Imagined states, decoded value layers
and footprints are computed under ``torch.no_grad``.
"""
from typing import Dict, List, Sequence

import gin
import torch
from torch import nn

from model.rl.footprint import (BEV_RESOLUTION, BEV_SIZE, FOV_HALF_ANGLE_DEG,
                                MOVE_ACTIONS, N_SWEEP, SENSOR_RANGE,
                                STOP_ACTION, build_pi_plus, candidate_footprints,
                                movement_kl, policy_entropy, step_pose_batch)


def _policy_logits(policy: nn.Module,
                   latent_state: torch.Tensor,
                   text_feat: torch.Tensor) -> torch.Tensor:
    """Uniform logits interface for ActionPolicy / ContinuedActionPolicy / ImagRLActor."""
    if hasattr(policy, 'policy_state_fusion'):
        # Stage-2 ActionPolicy: fusion + MLP command head.
        policy_state = policy.policy_state_fusion(latent_state, text_feat)
        return policy.policy_mlp(policy_state)['command']
    return policy(latent_state, text_feat)


@gin.configurable
class ValueGuidedImaginationEngine(nn.Module):
    """Prior-only imagined rollouts scored by the decoded spatial value layer.

    The engine owns no parameters and registers no submodules.  Its four module
    references are attached with ``object.__setattr__`` because the referenced
    modules are already registered elsewhere (``world_model.rssm`` /
    ``world_model.bev_decoder`` and the LightningModule's ``actor`` /
    ``bc_policy``), and ``state_dict`` recursion does not de-duplicate shared
    submodules: a plain assignment here would store those weights twice in
    every checkpoint.
    """

    def __init__(self,
                 rssm: nn.Module,
                 bev_decoder: nn.Module,
                 actor: nn.Module,
                 bc_policy: nn.Module,
                 imagination_horizon: int = 8,
                 gamma: float = 0.99,
                 tau: float = 1000.0,
                 value_channel: int = 2,
                 bev_size: int = BEV_SIZE,
                 bev_resolution: float = BEV_RESOLUTION,
                 sensor_range: float = SENSOR_RANGE,
                 fov_half_angle_deg: float = FOV_HALF_ANGLE_DEG,
                 n_sweep: int = N_SWEEP,
                 move_actions: Sequence[int] = MOVE_ACTIONS,
                 stop_action: int = STOP_ACTION,
                 action_deltas: Dict = None):
        super().__init__()
        # object.__setattr__ keeps these out of this module's submodule
        # registry; see the class docstring.
        object.__setattr__(self, 'rssm', rssm)
        object.__setattr__(self, 'bev_decoder', bev_decoder)
        object.__setattr__(self, 'actor', actor)
        object.__setattr__(self, 'bc_policy', bc_policy)
        self.horizon = int(imagination_horizon)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.value_channel = int(value_channel)
        self.bev_size = int(bev_size)
        self.bev_resolution = float(bev_resolution)
        self.sensor_range = float(sensor_range)
        self.fov_half_angle_deg = float(fov_half_angle_deg)
        self.n_sweep = int(n_sweep)
        self.move_actions = tuple(int(a) for a in move_actions)
        self.stop_action = int(stop_action)
        # Optional override (single source of truth = model/rl/footprint.py).
        self.action_deltas = action_deltas
        # Lookup table: action id -> index within move_actions (-1 for stop).
        lut = torch.full((max(self.move_actions) + 1,), -1, dtype=torch.long)
        for idx, a_c in enumerate(self.move_actions):
            lut[a_c] = idx
        # Plain attribute rather than a registered buffer, so it stays out of
        # the state dict; the rollout moves it to the active device explicitly.
        self._move_lut = lut

    # ------------------------------------------------------------------
    def _footprints(self, poses: torch.Tensor) -> torch.Tensor:
        """(N, 3) poses -> candidate movement footprints (N, M, H, W)."""
        return candidate_footprints(
            poses, move_actions=self.move_actions,
            bev_size=self.bev_size, bev_resolution=self.bev_resolution,
            sensor_range=self.sensor_range,
            fov_half_angle_deg=self.fov_half_angle_deg,
            n_sweep=self.n_sweep, action_deltas=self.action_deltas)

    def _decode_value(self, states: torch.Tensor) -> torch.Tensor:
        """Decode the value channel for (N, d) states -> (N, H, W)."""
        out = self.bev_decoder(states)
        bev = out['bev_256'] if isinstance(out, dict) else out
        return bev[:, self.value_channel]

    # ------------------------------------------------------------------
    def rollout(self,
                h0: torch.Tensor,
                z0: torch.Tensor,
                text_feat: torch.Tensor,
                pose0: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Imagined rollout + value-guided action loss.

        Args:
            h0, z0: (b, d) initial posterior state (detached from the world model).
            text_feat: (b, text_feat_dim) instruction conditioning.
            pose0: (b, 3) agent pose ``(x, y, yaw)`` of the initial real state.
        """
        b = h0.shape[0]
        device = h0.device
        h = h0.contiguous()
        z = z0.contiguous()
        pose = pose0.to(dtype=h0.dtype).contiguous()
        m = len(self.move_actions)

        states: List[torch.Tensor] = []
        logits_steps: List[torch.Tensor] = []
        bc_logits_steps: List[torch.Tensor] = []
        q_steps: List[torch.Tensor] = []          # (b, M) one-step utilities
        chosen_q_steps: List[torch.Tensor] = []   # (b,)
        entropy_steps: List[torch.Tensor] = []

        for _k in range(self.horizon):
            state = torch.cat([h, z], dim=-1).detach()
            states.append(state)
            logits = _policy_logits(self.actor, state, text_feat)  # grad
            logits_steps.append(logits)

            with torch.no_grad():
                prob = torch.softmax(logits.detach(), dim=-1)
                a = torch.multinomial(prob, 1).squeeze(1)          # (b,)
                entropy_steps.append(policy_entropy(logits.detach()))
                bc_logits = _policy_logits(self.bc_policy, state, text_feat)
                bc_logits_steps.append(bc_logits.detach())

                # --- Candidate one-step branches (frozen world + value decoder).
                cand_states = []
                for a_c in self.move_actions:
                    a_t = torch.full((b,), a_c, dtype=torch.long, device=device)
                    out = self.rssm.imagine_step(h, z, a_t, use_sample=True)
                    cand_states.append(
                        torch.cat([out['hidden_state'], out['sample']], dim=-1))
                cand_states = torch.stack(cand_states, dim=1)       # (b, M, d)
                flat = cand_states.reshape(b * m, -1).contiguous()
                value = self._decode_value(flat).reshape(
                    b, m, self.bev_size, self.bev_size)
                foot = self._footprints(pose)                       # (b, M, H, W)
                q = (foot * value).sum(dim=(-1, -2))                # (b, M)

                # sampled action's one-step utility (0 for stop) -> continuation
                sel = self._move_lut.to(device)[a]
                chosen = torch.where(
                    sel >= 0,
                    q.gather(1, sel.clamp_min(0).unsqueeze(1)).squeeze(1),
                    torch.zeros(b, device=device))
            q_steps.append(q)
            chosen_q_steps.append(chosen)

            # Advance the trajectory with the sampled action.
            with torch.no_grad():
                out = self.rssm.imagine_step(h, z, a, use_sample=True)
                h = out['hidden_state'].contiguous()
                z = out['sample'].contiguous()
            pose = step_pose_batch(pose, a, self.action_deltas)

        # --- H-step discounted accumulation Q_V^{(H)}.
        # The continuation term is shared across candidate actions at a given
        # state, so it is a constant shift of the utility ranking and cancels
        # inside the pi+ softmax; the action-dependent signal is q_V(s_k, .).
        q_stack = torch.stack(q_steps, dim=1)                       # (b, T, M)
        chosen_stack = torch.stack(chosen_q_steps, dim=1)           # (b, T)
        q_horizon = torch.zeros_like(q_stack)
        cont = torch.zeros(b, device=device)
        for k in reversed(range(self.horizon)):
            q_horizon[:, k] = q_stack[:, k] + self.gamma * cont.unsqueeze(1)
            cont = chosen_stack[:, k] + self.gamma * cont

        # --- pi+ target + KL(pi+ || pi_theta) on movement actions only.
        kl_per_step = []
        pi_plus_stop = []
        for k in range(self.horizon):
            with torch.no_grad():
                bc = torch.softmax(bc_logits_steps[k], dim=-1)      # frozen prior
                pi_plus_full, pi_plus_move = build_pi_plus(
                    bc, q_horizon[:, k].detach(), self.tau,
                    self.move_actions, self.stop_action)
            kl = movement_kl(pi_plus_move, logits_steps[k],
                             self.move_actions, self.stop_action)
            kl_per_step.append(kl.mean())
            pi_plus_stop.append(pi_plus_full[:, self.stop_action].mean())
        action_loss = torch.stack(kl_per_step).mean()

        # --- Monitoring (all detached).
        q_flat = q_stack.detach()
        entropy = torch.stack(entropy_steps).mean()
        return {
            'action_loss': action_loss,
            'kl_action': action_loss.detach(),
            'q_v_mean': q_flat.mean(),
            'q_v_std': q_flat.std(),
            'q_v_max': q_flat.amax(),
            'q_v_min': q_flat.amin(),
            'q_v_chosen_mean': chosen_stack.detach().mean(),
            'q_v_best_margin': (q_flat.amax(dim=-1)
                                - q_flat.mean(dim=-1)).mean(),
            'policy_entropy': entropy,
            'pi_plus_stop': torch.stack(pi_plus_stop).mean(),
            'policy_stop': torch.softmax(
                torch.stack(logits_steps, dim=1).detach(), dim=-1
            )[:, :, self.stop_action].mean(),
        }
