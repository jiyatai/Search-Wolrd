# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Action heads for stage-3 value-guided imagination planning.

Stage 3 trains ONLY the action head; the RSSM and the decoded spatial value
decoder are frozen (paper Section "Three-Stage Training Curriculum", Stage 3).
There is no scalar critic, no GAE and no reward head here.

Two interchangeable actor heads are provided:

* :class:`ContinuedActionPolicy` (default) -- a deep copy of the Stage-2 BC
  action head (``ActionPolicy``), warm-started and then refined.  The
  behaviour-cloned original stays inside the frozen world model and supplies
  the ``pi_BC`` prior / pinned ``stop`` probability of the ``pi+`` target.

* :class:`ImagRLActor` -- a fresh 8-way head used only for the ablation that
  removes the warm start (``use_fresh_actor=True``).  It shares the
  ``(latent_state, text_feat) -> logits`` interface.

Both mirror the ``ActionPolicy`` fusion interface so the conditioning pathway
is preserved and a stage-3 checkpoint's policy weights drop back into the
deployment bundle unchanged (see ``scripts/export_stage3_policy.py``).
"""
import copy

import gin
import torch
from torch import nn


class ContinuedActionPolicy(nn.Module):
    '''Warm-start wrapper: the Stage-2 BC action head, continued in Stage 3.

    Wraps the loaded (frozen) ActionPolicy's fusion + command head in a
    TRAINABLE copy so Stage 3 can fine-tune the very head that deployment
    executes. The BC original stays frozen inside the world model as the
    behavior-cloned prior ``pi_BC`` for the value-guided target.

    Architecture note: ActionPolicy uses self_attn fusion by default
    (PolicyStateSelfAttentionFusion: MultiheadAttention over 2 tokens)
    followed by MLPPolicy.command_fc. We copy the modules wholesale rather
    than reimplementing them, so the continued head is architecturally
    IDENTICAL to the deployed Stage-2 head; a stage-3 checkpoint's policy
    weights drop back into the deployment bundle unchanged (see
    scripts/export_stage3_policy.py).

    forward(latent_state, text_feat) -> logits (b, n_actions); mirrors
    ImagRLActor's interface so the engine can use either.
    '''

    def __init__(self, action_policy: nn.Module):
        super().__init__()
        # Deep-copy the BC head (fusion + MLP policy); this copy trains.
        self.fusion = copy.deepcopy(action_policy.policy_state_fusion)
        self.policy_mlp = copy.deepcopy(action_policy.policy_mlp)

    def forward(self, latent_state: torch.Tensor,
                text_feat: torch.Tensor) -> torch.Tensor:
        policy_state = self.fusion(latent_state, text_feat)
        return self.policy_mlp(policy_state)['command']


@gin.configurable
class ImagRLActor(nn.Module):
    '''8-way categorical actor for the fresh-head (no warm start) ablation.

    forward(latent_state, text_feat) -> logits (b, 8)
    The signature intentionally mirrors ActionPolicy's fusion inputs so the
    text conditioning path is preserved from stage 2.
    '''

    def __init__(self,
                 latent_state_dim: int = 1536,
                 text_feat_dim: int = 768,
                 hidden_dim: int = 512,
                 n_actions: int = 8):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(latent_state_dim + text_feat_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
        )
        self.head = nn.Linear(hidden_dim, n_actions)
        # Small init for stable early logits.
        nn.init.orthogonal_(self.head.weight, gain=0.01)
        nn.init.zeros_(self.head.bias)

    def forward(self, latent_state: torch.Tensor,
                text_feat: torch.Tensor) -> torch.Tensor:
        x = torch.cat([latent_state, text_feat], dim=-1)
        return self.head(self.fusion(x))
