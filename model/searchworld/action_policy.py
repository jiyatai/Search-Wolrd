# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

import gin
import torch
from torch import nn

from model.searchworld.diffusion_policy import DiffusionPolicy
from model.searchworld.utils import pack_sequence_dim, unpack_sequence_dim


class PolicyStateConcatenateFusion(nn.Module):
    ''' Concatenation based fusion for policy input features.
    '''
    def __init__(self, latent_state_dim, text_feat_dim, fusion_dim):
        super().__init__()
        self.fc_fused = nn.Linear(latent_state_dim + text_feat_dim,
                                  fusion_dim)

    def forward(self, latent_state, text_feat):
        return self.fc_fused(torch.cat([latent_state, text_feat], dim=1))


class PolicyStateMLPAttentionFusion(nn.Module):
    ''' MLP attention based fusion for policy input features.
    '''
    def __init__(self, latent_state_dim, text_feat_dim, fusion_dim):
        super().__init__()
        self.fc_latent_state = nn.Linear(latent_state_dim, fusion_dim)
        self.fc_text_feat = nn.Linear(text_feat_dim, fusion_dim)
        self.attn = nn.Linear(fusion_dim * 2, 1)

    def forward(self, latent_state, text_feat):
        latent_state_proj = self.fc_latent_state(latent_state)
        text_proj = self.fc_text_feat(text_feat)
        combined = torch.cat((latent_state_proj, text_proj), dim=1)
        attn_weights = torch.sigmoid(self.attn(combined))
        fused_embedding = attn_weights * latent_state_proj + (
            1 - attn_weights) * text_proj
        return fused_embedding


class PolicyStateSelfAttentionFusion(nn.Module):
    ''' Scaled dot-product self attention based fusion for policy input features.
    '''
    def __init__(self, latent_state_dim, text_feat_dim, fusion_dim):
        super().__init__()
        self.fc_latent_state = nn.Linear(latent_state_dim, fusion_dim)
        self.fc_text_feat = nn.Linear(text_feat_dim, fusion_dim)
        self.fc_fused = nn.Linear(2 * fusion_dim, fusion_dim)
        self.attn = nn.MultiheadAttention(embed_dim=fusion_dim, num_heads=4)

    def forward(self, latent_state, text_feat):
        latent_state_proj = self.fc_latent_state(latent_state)
        text_proj = self.fc_text_feat(text_feat)
        combined = torch.cat(
            (latent_state_proj.unsqueeze(0), text_proj.unsqueeze(0)), dim=0)
        attn_output, _ = self.attn(combined, combined, combined)
        fused_embedding = torch.cat((attn_output[0, :], attn_output[1, :]),
                                    dim=-1)
        fused_embedding = self.fc_fused(fused_embedding)
        return fused_embedding


@gin.configurable
class MLPPolicy(nn.Module):
    '''MLP based policy network.

        Args:
            in_channels (int): input channels size
            command_n_channels (int): output command tensor size
            path_n_channels (int): output path tensor size

        Inputs:
            x: policy_state fused from latent state and text features.

        Returns:
            policys: dict of policy outputs
    '''
    def __init__(self, in_channels: int, command_n_channels: int,
                 path_n_channels: int):
        super().__init__()
        self.command_fc = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(True),
            nn.Linear(in_channels, in_channels),
            nn.ReLU(True),
            nn.Linear(in_channels, in_channels // 2),
            nn.ReLU(True),
            # Output raw logits over discrete action classes.
            nn.Linear(in_channels // 2, command_n_channels),
        )
        self.path_fc = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(True),
            nn.Linear(in_channels, in_channels),
            nn.ReLU(True),
            nn.Linear(in_channels, in_channels // 2),
            nn.ReLU(True),
            nn.Linear(in_channels // 2, path_n_channels),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor):
        return {'command': self.command_fc(x), 'path': self.path_fc(x)}


@gin.configurable
class ActionPolicy(nn.Module):
    '''Action policy network for UAV target search.

        Args:
            latent_state_dim (int): input latent state feature size
            policy_state_dim (int): policy state dimension
            text_feat_dim (int): text feature dimension (precomputed offline)
            policy_state_fusion_mode (str): how to fuse latent state and text
            enable_policy_diffusion(bool): use diffusion policy or simple MLP

        Inputs:
            batch(dict): input batch containing text_feat
            latent_state(torch.Tensor): latent state from RSSM

        Returns:
            output(dict): policy action and path
    '''
    def __init__(self,
                 latent_state_dim: int,
                 policy_state_dim: int,
                 text_feat_dim: int = 768,
                 policy_state_fusion_mode: str = "concatenate",
                 enable_policy_diffusion: bool = True):
        super().__init__()

        self.enable_policy_diffusion = enable_policy_diffusion
        self.poly_state_dim = policy_state_dim

        if policy_state_fusion_mode == 'mlp_attn':
            self.policy_state_fusion = PolicyStateMLPAttentionFusion(
                latent_state_dim=latent_state_dim,
                text_feat_dim=text_feat_dim,
                fusion_dim=policy_state_dim)
        elif policy_state_fusion_mode == 'self_attn':
            self.policy_state_fusion = PolicyStateSelfAttentionFusion(
                latent_state_dim=latent_state_dim,
                text_feat_dim=text_feat_dim,
                fusion_dim=policy_state_dim)
        else:
            self.policy_state_fusion = PolicyStateConcatenateFusion(
                latent_state_dim=latent_state_dim,
                text_feat_dim=text_feat_dim,
                fusion_dim=policy_state_dim)

        if self.enable_policy_diffusion:
            self.policy_diffuser = DiffusionPolicy(
                latent_state_dim=policy_state_dim)
        else:
            self.policy_mlp = MLPPolicy(in_channels=policy_state_dim)

    def forward(self, batch, latent_state):
        b, s = batch['text_feat'].shape[:2]

        text_feat = pack_sequence_dim(batch['text_feat'])

        output = {}

        policy_state = self.policy_state_fusion(latent_state, text_feat)
        if self.enable_policy_diffusion:
            policy_diffuser_output = self.policy_diffuser(batch, policy_state)
            output = {**output, **policy_diffuser_output}
            noise = torch.randn((policy_state.shape[0],
                                 self.policy_diffuser.num_input_channels()),
                                device=policy_state.device)
            diffusion_policy_out = self.policy_diffuser.denoising_and_decode(
                noise, policy_state)
            output['action'] = unpack_sequence_dim(
                diffusion_policy_out['actions'], b, s)
            output['path'] = unpack_sequence_dim(diffusion_policy_out['paths'],
                                                 b, s)
        else:
            mlp_policy_out = self.policy_mlp(policy_state)
            output['action'] = unpack_sequence_dim(mlp_policy_out['command'],
                                                   b, s)
            output['path'] = unpack_sequence_dim(mlp_policy_out['path'], b, s)

        return output

    @torch.inference_mode()
    def inference(self, latent_state: torch.Tensor, batch: dict):
        b, s = batch['text_feat'].shape[:2]
        text_feat = pack_sequence_dim(batch['text_feat'])
        policy_state = self.policy_state_fusion(latent_state, text_feat)

        if self.enable_policy_diffusion:
            diffusion_policy_out = self.policy_diffuser.denoising_and_decode(
                batch['policy_noise'], policy_state, denoising_steps=5)
            command_output = unpack_sequence_dim(
                diffusion_policy_out['actions'], b, s)
            path_output = unpack_sequence_dim(diffusion_policy_out['paths'], b,
                                              s)
        else:
            mlp_policy_out = self.policy_mlp(policy_state)
            command_output = unpack_sequence_dim(mlp_policy_out['command'], b,
                                                 s)
            path_output = unpack_sequence_dim(mlp_policy_out['path'], b, s)

        return command_output, path_output
