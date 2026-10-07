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
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict

import gin
import torch
import torch.nn.functional as F
from torch import nn
from transformers import (AutoModel, DepthAnythingForDepthEstimation,
                          Dinov2Model)

from model.searchworld.utils import pack_sequence_dim, unpack_sequence_dim
from model.searchworld.bev_encoder import BEVEncoder

# Need to compute this dynamically and adapt to different image size.
DEPTH_ANYTHING_IMAGE_SIZE = [322, 518]


@gin.configurable
class SpeedEncoder(nn.Module):
    '''Encoder of robot's speed

        Args:
            out_channels (int): output channels size
            speed_normalisation (float): speed normialisation factor
    '''
    def __init__(self, out_channels: int, speed_normalisation: float):
        super().__init__()
        self.speed_encoder = nn.Sequential(
            nn.Linear(1, out_channels),
            nn.ReLU(True),
            nn.Linear(out_channels, out_channels),
            nn.ReLU(True),
        )
        self.out_channels = out_channels
        self.speed_normalisation = speed_normalisation

    def forward(self, speed: torch.Tensor) -> torch.Tensor:
        return self.speed_encoder(speed / self.speed_normalisation)


@gin.configurable
class PoseEncoder(nn.Module):
    '''Encoder of UAV relative pose w.r.t. the episode start frame.

        Input (per step):
            relative_pose: (..., 4) = [dx, dy, dz, dyaw]
                dx, dy, dz : position offset from the episode start (world frame)
                dyaw       : heading offset (radians) from the start yaw

        Notes:
            - Relative-to-start avoids absolute-coordinate sim2real mismatch,
              as long as the sim/real episode starts are aligned.
            - Values can grow with episode length, so we normalise by a
              configurable scene scale before the MLP.
    '''
    def __init__(self,
                 out_channels: int,
                 pose_normalisation: float = 50.0,
                 in_channels: int = 4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pose_normalisation = pose_normalisation
        self.pose_encoder = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(True),
            nn.Linear(out_channels, out_channels),
            nn.ReLU(True),
        )

    def forward(self, relative_pose: torch.Tensor) -> torch.Tensor:
        return self.pose_encoder(relative_pose / self.pose_normalisation)


@gin.configurable
class ImageDINOEncoder(nn.Module):
    ''' Image encoding with DINO v2
    '''
    def __init__(self, enable_fine_tune=False):
        super().__init__()
        self.dino_model = Dinov2Model.from_pretrained('facebook/dinov2-small',
                                                      output_attentions=True,
                                                      attn_implementation="eager")
        # Freeze the pre-trained dino model.
        if not enable_fine_tune:
            for param in self.dino_model.parameters():
                param.requires_grad = False
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.out_channels = self.dino_model.config.hidden_size * 2

    def forward(self, image):
        # Preprocess image.
        dino_image_size = self.dino_model.config.image_size
        processed_image = F.interpolate(
            image,
            size=[dino_image_size, dino_image_size],
            mode='bicubic',
            align_corners=False)
        processed_image = (processed_image - self.mean) / self.std

        # Encode image.
        outputs = self.dino_model(processed_image)

        # Extract features.
        last_hidden_states = outputs[0]
        cls_token = last_hidden_states[:, 0]
        patch_tokens = last_hidden_states[:, 1:]
        features = torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=1)

        # Extract attentions
        n, _, h, w = processed_image.shape
        avg_attension = torch.mean(outputs.attentions[-1], dim=1)
        cls_attention = avg_attension[:, 0, 1:].view(n, h // 14, w // 14)

        return {'image_features': features, 'image_attentions': cls_attention}


@gin.configurable
class ImageDepthAnythingEncoder(nn.Module):
    ''' Image encoding with Depth Anything.
    '''
    def __init__(self, enable_fine_tune=False):
        super().__init__()
        self.enable_fine_tune = enable_fine_tune
        self.depth_model = DepthAnythingForDepthEstimation.from_pretrained(
            "LiheYoung/depth-anything-small-hf",
            output_hidden_states=True,
            output_attentions=True)
        if not enable_fine_tune:
            for param in self.depth_model.parameters():
                param.requires_grad = False
        else:
            self.depth_gt_model = DepthAnythingForDepthEstimation.from_pretrained(
                "LiheYoung/depth-anything-small-hf",
                output_hidden_states=False,
                output_attentions=False)
            for param in self.depth_gt_model.parameters():
                param.requires_grad = False
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.out_channels = self.depth_model.backbone.config.hidden_size * 2

    def forward(self, image):
        # Preprocess image.
        processed_image = F.interpolate(image,
                                        size=DEPTH_ANYTHING_IMAGE_SIZE,
                                        mode='bicubic',
                                        align_corners=False)
        processed_image = (processed_image - self.mean) / self.std

        # Encode image.
        outputs = self.depth_model(processed_image)

        # Extract features.
        last_hidden_states = outputs.hidden_states[-1]
        cls_token = last_hidden_states[:, 0]
        patch_tokens = last_hidden_states[:, 1:]
        features = torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=1)

        # Extract attentions
        n, _, h, w = processed_image.shape
        avg_attension = torch.mean(outputs.attentions[-1], dim=1)
        cls_attention = avg_attension[:, 0, 1:].view(n, h // 14, w // 14)

        ret_dict = {
            'image_features': features,
            'image_attentions': cls_attention,
            'depth': outputs.predicted_depth,
        }

        # Get depth GT based on the pretrained DepthAnything model.
        if self.enable_fine_tune:
            depth_gt = self.depth_gt_model(processed_image).predicted_depth
            ret_dict['depth_gt'] = depth_gt

        return ret_dict


@gin.configurable
class ImageSigLIPEncoder(nn.Module):
    ''' Image encoding with SigLIP2 (frozen) for language-aligned features.

        Model: google/siglip2-base-patch16-224
            - Input image: 224x224, model-internal normalisation
            - Output: 768-d pooled feature (L2-normalised)

        Note: in this SearchWorld pipeline, the SigLIP image feature is
        aggregated into the RSSM embedding but cosine similarity with the
        text feature is NOT computed inside the model (handled downstream
        if needed).
    '''
    def __init__(self,
                 model_name: str = 'google/siglip2-base-patch16-224',
                 enable_fine_tune: bool = False):
        super().__init__()
        self.siglip_model = AutoModel.from_pretrained(model_name)
        if not enable_fine_tune:
            for param in self.siglip_model.parameters():
                param.requires_grad = False
        self.model_name = model_name
        # siglip2-base-patch16-224 -> 768
        self.out_channels = self.siglip_model.config.text_config.hidden_size \
            if hasattr(self.siglip_model.config, 'text_config') \
            else self.siglip_model.config.hidden_size

    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        '''
        Args:
            image: (N, 3, H, W) RGB tensor in [0, 1].

        Returns:
            dict with 'image_features': (N, 768) L2-normalised feature.
        '''
        target_size = 224
        if image.shape[-1] != target_size or image.shape[-2] != target_size:
            processed_image = F.interpolate(
                image, size=[target_size, target_size],
                mode='bicubic', align_corners=False)
        else:
            processed_image = image

        # Only use the vision model, not the text model
        if hasattr(self.siglip_model, 'vision_model'):
            vision_outputs = self.siglip_model.vision_model(pixel_values=processed_image)
            image_features = vision_outputs.pooler_output
        else:
            # Fallback: if the model is already a vision-only model
            outputs = self.siglip_model(pixel_values=processed_image)
            image_features = outputs.image_embeds if hasattr(
                outputs, 'image_embeds') and outputs.image_embeds is not None \
                else outputs.last_hidden_state.mean(dim=1)
        image_features = F.normalize(image_features, dim=-1)
        return {'image_features': image_features}


@gin.configurable
class UAVObservationEncoder(nn.Module):
    '''Observation encoder for the UAV target-search task.

        Branches (all merged into one embedding, then fed to RSSM):
            - DINOv2 (frozen):       spatial/geometric features + cls_attention
            - SigLIP2 (frozen):     language-aligned image features (768)
            - PoseEncoder:           4D relative pose w.r.t. episode start
                                    [dx, dy, dz, dyaw]  -> 64
            - text_feat (precomputed by DataLoader):
                                    SigLIP text embedding of the target
                                    description (768).  Produced offline by
                                    running SigLIP's text encoder on
                                    `task.description`; the model receives
                                    the vector directly, no text encoder is
                                    loaded here.
            - BEVEncoder:            encodes BEV exploration + obstacle layers

        Fusion:
            concat(dino 768, siglip 768, pose 64, text 768, bev 1024)
                -> Linear -> embedding_dim
                -> RSSM

        Inputs (batch dict):
            image:         (b, s, 3, H, W) front-camera RGB
            relative_pose: (b, s, 4)       [dx, dy, dz, dyaw] vs. start
            text_feat:     (b, s, 768)     SigLIP text embedding of
                                          task.description (offline)
            bev_memory:    (b, s, 3, 256, 256) [optional] BEV map
            action:        (b, s, A)       NOT consumed here; passed to RSSM

        Returns (dict):
            embedding:       (b, s, embedding_dim)
            dino_features:   (b, s, 768)
            siglip_features: (b, s, 768)  L2-normalised
            pose_features:   (b, s, 64)
            text_features:   (b, s, 768)  echo back for downstream
            bev_features:    (b, s, bev_dim) [if enabled]
            dino_attentions: (b, s, h, w) for semantic decoder
    '''
    def __init__(self,
                 dino_encoder: nn.Module = ImageDINOEncoder,
                 siglip_encoder: nn.Module = ImageSigLIPEncoder,
                 pose_out_channels: int = 64,
                 text_feat_dim: int = 768,
                 embedding_dim: int = 1024,
                 enable_bev: bool = False,
                 bev_dim: int = 1024):
        super().__init__()

        # Image: DINOv2 (spatial / decoder-conditioning).
        self.dino_encoder = dino_encoder()
        dino_dim = self.dino_encoder.out_channels  # 768

        # Image: SigLIP2 (language-aligned).
        self.siglip_encoder = siglip_encoder()
        siglip_dim = self.siglip_encoder.out_channels  # 768

        # Pose.
        self.pose_encoder = PoseEncoder(out_channels=pose_out_channels)
        pose_dim = self.pose_encoder.out_channels  # 64

        # BEV encoder (optional).
        self.enable_bev = enable_bev
        if enable_bev:
            self.bev_encoder = BEVEncoder(out_dim=bev_dim)
        else:
            self.bev_encoder = None

        # Fusion projection.
        total_dim = dino_dim + siglip_dim + pose_dim + text_feat_dim
        if enable_bev:
            total_dim += bev_dim

        self.fusion = nn.Linear(total_dim, embedding_dim)

        self.text_feat_dim = text_feat_dim
        self.embedding_dim = embedding_dim
        self.bev_dim = bev_dim if enable_bev else 0

    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        b, s = batch['image'].shape[:2]

        # Pack (b, s, ...) -> (b*s, ...) for the image / pose branches.
        image = pack_sequence_dim(batch['image'])
        relative_pose = pack_sequence_dim(batch['relative_pose'])
        text_feat = pack_sequence_dim(batch['text_feat'])  # (N, 768)

        # DINOv2.
        dino_out = self.dino_encoder(image)
        dino_feat = dino_out['image_features']             # (N, 768)
        dino_attn = dino_out['image_attentions']           # (N, h, w)

        # SigLIP2.
        siglip_out = self.siglip_encoder(image)
        siglip_feat = siglip_out['image_features']         # (N, 768)

        # Pose.
        pose_feat = self.pose_encoder(relative_pose)       # (N, 64)

        # Collect features.
        features_list = [dino_feat, siglip_feat, pose_feat, text_feat]

        outputs = {
            'embedding': None,
            'dino_features': unpack_sequence_dim(dino_feat, b, s),
            'siglip_features': unpack_sequence_dim(siglip_feat, b, s),
            'pose_features': unpack_sequence_dim(pose_feat, b, s),
            'text_features': unpack_sequence_dim(text_feat, b, s),
            'dino_attentions': unpack_sequence_dim(dino_attn, b, s),
        }

        # BEV encoding (if enabled).
        if self.enable_bev:
            bev_memory = batch.get('bev_memory', batch.get('bev_gt'))
            if bev_memory is None:
                raise RuntimeError(
                    "UAVObservationEncoder.enable_bev=True but batch has no "
                    "'bev_memory' or 'bev_gt' key. Ensure the DataModule is "
                    "constructed with enable_bev=True and the dataset provides "
                    "BEV data.")
            bev_feat = self.bev_encoder(bev_memory)  # (b, s, bev_dim)
            bev_feat_packed = pack_sequence_dim(bev_feat)  # (b*s, bev_dim)
            features_list.append(bev_feat_packed)
            outputs['bev_features'] = bev_feat

        # Aggregate.
        fused = torch.cat(features_list, dim=-1)
        embedding = self.fusion(fused)                     # (N, embedding_dim)

        outputs['embedding'] = unpack_sequence_dim(embedding, b, s)

        return outputs
