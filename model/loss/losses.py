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
import torchvision.models
from torch import nn

from model.loss.bev_grid import BEV_RESOLUTION, BEV_SIZE
from model.loss.value_target import build_value_target
from model.searchworld.utils import pack_sequence_dim


@gin.configurable
class SegmentationLoss(nn.Module):
    '''Loss for semantic segmentaiton

        Inputs:
            prediction: predicted semantic image (b, s, c_s, h, w)
            target: ground-truth semantic image (b, s, h, w)

        Returns:
            loss: float
    '''
    def __init__(self, use_top_k: bool, top_k_ratio: float, use_poly_one: bool,
                 poly_one_coefficient: float, use_weights: bool,
                 semantic_weights: list):
        super().__init__()
        self.use_top_k = use_top_k
        self.top_k_ratio = top_k_ratio
        self.use_weights = use_weights
        self.use_poly_one = use_poly_one
        self.poly_one_coefficient = poly_one_coefficient

        if self.use_weights:
            self.weights = semantic_weights

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> float:
        b, s, c, h, w = prediction.shape
        prediction = prediction.view(b * s, c, h, w)
        target = target.view(b * s, h, w).long()

        weights = torch.tensor(
            self.weights, dtype=prediction.dtype,
            device=prediction.device) if self.use_weights else None

        loss = F.cross_entropy(
            prediction,
            target,
            reduction='none',
            weight=weights,
        )
        loss = loss[~torch.isnan(loss)]
        if self.use_poly_one:
            prob = torch.exp(-loss)
            loss_poly_one = self.poly_one_coefficient * (1 - prob)
            loss = loss + loss_poly_one
        loss = loss.view(b, s, -1)
        if self.use_top_k:
            # Penalises the top-k hardest pixels
            k = int(self.top_k_ratio * loss.shape[2])
            loss = loss.topk(k, dim=-1)[0]

        return torch.mean(loss)


class PerceptualLoss(nn.Module):
    '''Perceptural loss for RGB reconstruction.
       Ref paper: Perceptual Losses for Real-Time Style Transfer and Super-Resolution

    Args:
        x: pred image (N, 3, H, W)
        y: target image (N, 3, H, W)
    '''
    def __init__(self):
        super().__init__()

        # VGG blocks
        blocks = []
        blocks.append(
            torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(
            torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(
            torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(
            torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for block in blocks:
            for p in block.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)

        # Normalization parameters for VGG-16
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, target):
        # Normalize input images
        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std

        loss = 0.0
        x = pred
        y = target
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += torch.nn.functional.l1_loss(x, y)
        return loss


@gin.configurable
class RgbLoss(nn.Module):
    '''Loss for RGB prediction

    Inputs:
        prediction: predicted rgb image (b, s, 3, h, w)
        target: ground-truth rgb image (b, s, 3, h, w)

    Returns:
        loss: loss for RGB
    '''
    def __init__(self):
        super().__init__()
        self.perceptual_loss = PerceptualLoss()
        self.l1_loss = F.l1_loss

    def forward(self, prediction: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        assert len(prediction.shape) == 5, 'Prediction must be a 5D tensor'
        l1_loss = self.l1_loss(prediction, target, reduction='none')
        l1_loss = torch.sum(l1_loss, dim=-3, keepdims=True).mean()
        perceptual_loss = self.perceptual_loss(pack_sequence_dim(prediction),
                                               pack_sequence_dim(target))
        return l1_loss + perceptual_loss


@gin.configurable
class ActionLoss(nn.Module):
    '''Loss for discrete action classification.

        Inputs:
            prediction: predicted action logits (b, s, num_classes)
            target: ground-truth action class indices (b, s)

        Returns:
            loss: cross-entropy loss for action
    '''
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, prediction: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        assert len(prediction.shape) == 3, (
            'Prediction must be a 3D tensor (b, s, num_classes)')
        assert len(target.shape) == 2, (
            'Target must be a 2D tensor (b, s) of class indices')

        # Flatten (b, s) so that CrossEntropyLoss treats every timestep
        # in the sequence as an independent classification sample.
        prediction = prediction.reshape(-1, prediction.shape[-1])
        target = target.reshape(-1).long()
        return self.loss_fn(prediction, target)


class PathLoss(nn.Module):
    '''Loss for path regression

        Inputs:
            prediction: predicted action (b, s, c_p)
            target: ground-truth action (b, s, c_p)

        Returns:
            loss: loss for path
    '''
    def __init__(self, channel_dim=-1):
        super().__init__()
        self.channel_dim = channel_dim
        self.loss_fn = F.mse_loss

    def forward(self, prediction: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        loss = self.loss_fn(prediction, target, reduction='none')

        # Sum channel dimension
        loss = torch.sum(loss, dim=self.channel_dim, keepdims=True)
        return loss.mean()


class ProbabilisticLoss(nn.Module):
    ''' KL divergence loss between a prior distribution and a posterior distribution

        Inputs:
            prior_mu, prior_sigma: Prior distributions
            posterior_mu, posterior_sigma: Poserior distributions

        Returns:
            loss: KL divergence between the two distributions.
    '''
    def forward(self, prior_mu: torch.Tensor, prior_sigma: torch.Tensor,
                posterior_mu: torch.Tensor,
                posterior_sigma: torch.Tensor) -> torch.Tensor:
        posterior_var = posterior_sigma[:, 1:]**2
        prior_var = prior_sigma[:, 1:]**2

        posterior_log_sigma = torch.log(posterior_sigma[:, 1:])
        prior_log_sigma = torch.log(prior_sigma[:, 1:])

        kl_div = (prior_log_sigma - posterior_log_sigma - 0.5 +
                  (posterior_var +
                   (posterior_mu[:, 1:] - prior_mu[:, 1:])**2) /
                  (2 * prior_var))
        first_kl = -posterior_log_sigma[:, :1] - 0.5 + (
            posterior_var[:, :1] + posterior_mu[:, :1]**2) / 2
        kl_div = torch.cat([first_kl, kl_div], dim=1)

        # Sum across channel dimension
        # Average across batch dimension, keep time dimension for monitoring
        kl_loss = torch.mean(torch.sum(kl_div, dim=-1))
        return kl_loss


@gin.configurable
class KLLoss(nn.Module):
    ''' Balanced loss for KL divergence

        Inputs:
            prior: Prior distributions
            posterio: Poserior distributions

        Returns:
            loss: Balanced KL divergence between the two distributions.
    '''
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
        self.loss = ProbabilisticLoss()

    def forward(self, prior: Dict, posterior: Dict) -> float:
        prior_mu, prior_sigma = prior['mu'], prior['sigma']
        posterior_mu, posterior_sigma = posterior['mu'], posterior['sigma']
        prior_loss = self.loss(prior_mu, prior_sigma, posterior_mu.detach(),
                               posterior_sigma.detach())
        posterior_loss = self.loss(prior_mu.detach(), prior_sigma.detach(),
                                   posterior_mu, posterior_sigma)

        return self.alpha * prior_loss + (1 - self.alpha) * posterior_loss


@gin.configurable
class DiffusionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = F.mse_loss

    def forward(self, noise: torch.tensor, noise_pred: torch.tensor) -> float:
        loss = self.loss(noise, noise_pred)
        return loss


@gin.configurable
class DepthLoss(nn.Module):
    ''' Loss for depth prediction.
    '''
    def __init__(self, alpha=1.0, beta=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.l1_loss = nn.L1Loss()

    def forward(self, predicted, target):
        # L1 loss
        l1_loss = self.l1_loss(predicted, target)

        # Gradient loss
        gradient_loss = self.compute_gradient_loss(predicted, target)

        # Combined loss
        return self.alpha * l1_loss + self.beta * gradient_loss

    def compute_gradient_loss(self, predicted, target):
        pred_dx = torch.abs(predicted[:, :, :-1] - predicted[:, :, 1:])
        pred_dy = torch.abs(predicted[:, :-1, :] - predicted[:, 1:, :])
        target_dx = torch.abs(target[:, :, :-1] - target[:, :, 1:])
        target_dy = torch.abs(target[:, :-1, :] - target[:, 1:, :])

        grad_loss_x = self.l1_loss(pred_dx, target_dx)
        grad_loss_y = self.l1_loss(pred_dy, target_dy)

        gradient_loss = grad_loss_x + grad_loss_y
        return gradient_loss


@gin.configurable
class BEVLoss(nn.Module):
    ''' Loss for BEV prediction (value layer + exploration/obstacle).

    Three channels:
        0: exploration (binary mask, cumulative max)
        1: obstacle (binary mask, transient overwrite)
        2: value layer (continuous [0, 1]; the decoded task-aware spatial value
           layer, paper eq. value)

    The third channel is supervised by the training target of eq. (valuetarget)
    (see ``model/loss/value_target.py``)::

        V*(x) = Norm[ (1 - M_obs*(x)) ( lambda_expl (1 - M_expl*(x))
                                         + lambda_goal * G_g(x) ) ]

    where ``G_g`` is a Gaussian centred on the ground-truth target position
    ``g`` (available only at training time) and ``Norm`` rescales the layer to
    unit maximum.  The exploration and obstacle labels are the geometry-built
    ground-truth layers (channels 0 and 1 of ``target``); the target position
    comes from the ``target_rel`` dataset column (metres relative to the grid
    centre).  The target coordinate never enters the model at inference.

    Other details:
        - The robust loss l_rob of eq. (valueloss) is a Huber (smooth-L1) loss
          with ``value_huber_beta``.
        - Dice loss is mixed with weighted BCE for the two binary channels:
          both are extremely sparse (measured positive rate 1.3% / 0.3%), and
          Dice is insensitive to class imbalance.
        - Frames whose GT is all-zero for a binary channel (no observation in
          the window) are excluded from that channel's loss instead of being
          forced to predict all-zeros (obstacle is transient and frequently
          empty early in a window).
        - Multi-scale supervision: when the model provides a multi-scale dict
          (bev_16 ... bev_256), the GT is area-downsampled to each scale and
          the per-scale losses are combined with geometric weights.

    Inputs:
        prediction: either a tensor (b, s, 3, H, W) or a dict of multi-scale
            tensors {bev_16: (b,s,3,16,16), ..., bev_256: (b,s,3,256,256)}
        target: target BEV (b, s, 3, H, W) at full resolution
        target_rel: (b, s, 2) target position in world metres relative to the
            grid centre (training-only; ``None`` disables the value channel
            when ``require_target=False``)

    Returns:
        loss: dict of losses (bev_exploration / bev_obstacle / bev_value /
            bev_total) plus a detached 'bev_value_corr' monitoring entry.
    '''
    def __init__(self,
                 exploration_weight: float = 1.0,
                 obstacle_weight: float = 1.0,
                 value_weight: float = 1.0,
                 use_weighted_bce: bool = True,
                 use_dice: bool = True,
                 dice_weight: float = 1.0,
                 skip_all_zero_frames: bool = True,
                 lambda_expl: float = 1.0,
                 lambda_goal: float = 1.0,
                 sigma_g: float = 10.0,
                 value_huber_beta: float = 0.1,
                 require_target: bool = True,
                 multiscale_weights: dict = None):
        super().__init__()
        self.exploration_weight = exploration_weight
        self.obstacle_weight = obstacle_weight
        self.value_weight = value_weight
        self.use_weighted_bce = use_weighted_bce
        self.use_dice = use_dice
        self.dice_weight = dice_weight
        self.skip_all_zero_frames = skip_all_zero_frames
        self.lambda_expl = lambda_expl
        self.lambda_goal = lambda_goal
        self.sigma_g = sigma_g
        self.value_huber_beta = value_huber_beta
        self.require_target = require_target
        # Full-res scale -> loss weight. bev_256 is the primary output.
        self.multiscale_weights = multiscale_weights or {
            256: 1.0,
            128: 0.5,
            64: 0.25,
            32: 0.125,
            16: 0.0625,
        }
        self._warned_missing_target = False

    @staticmethod
    def _bce_with_pos_weight(pred: torch.Tensor,
                             target: torch.Tensor) -> torch.Tensor:
        """Weighted BCE for already-sigmoid-ed BEV channels.

        Plain ``F.binary_cross_entropy`` (not ``_with_logits``):
          1) has no ``pos_weight`` argument, so we apply the class weight manually
             (``pos_weight`` on the positive class, 1.0 elsewhere);
          2) is flagged "unsafe to autocast" by AMP and raises a RuntimeError
             under 16-mixed precision, so we run it with autocast disabled. The
             inputs are already float32 (sigmoid outputs), so this costs nothing.
        """
        pos_weight = torch.sum(target < 0.5) / (torch.sum(target >= 0.5) + 1)
        pos_weight = pos_weight.clamp(0.1, 10)
        with torch.cuda.amp.autocast(enabled=False):
            bce = F.binary_cross_entropy(pred, target, reduction='none')
            weight = torch.where(target >= 0.5, pos_weight, 1.0)
            return (bce * weight).mean()

    @staticmethod
    def _dice_loss(pred: torch.Tensor, target: torch.Tensor,
                   eps: float = 1.0) -> torch.Tensor:
        """Soft Dice loss for sparse binary channels.

        Args:
            pred: (n, 1, h, w) in [0, 1] (already sigmoid-ed)
            target: (n, 1, h, w) binary-ish in [0, 1]
        """
        pred = pred.flatten(1)
        target = target.flatten(1)
        inter = (pred * target).sum(dim=-1)
        union = pred.sum(dim=-1) + target.sum(dim=-1)
        dice = 1.0 - (2.0 * inter + eps) / (union + eps)
        return dice.mean()

    def _binary_channel_loss(self, pred: torch.Tensor,
                             target: torch.Tensor) -> torch.Tensor:
        """Weighted BCE (+ Dice) for one binary channel.

        Optionally skips frames whose GT is entirely zero (no observation).
        Returns a zero-valued tensor with a graph connection when no frame is
        valid, so autograd never sees a parameterless loss.
        """
        if self.skip_all_zero_frames:
            valid = target.flatten(1).sum(dim=1) > 0
            if valid.sum() == 0:
                return pred.sum() * 0.0
            pred = pred[valid]
            target = target[valid]

        if self.use_weighted_bce:
            loss = self._bce_with_pos_weight(pred, target)
        else:
            with torch.cuda.amp.autocast(enabled=False):
                loss = F.binary_cross_entropy(pred, target)

        if self.use_dice:
            loss = loss + self.dice_weight * self._dice_loss(pred, target)
        return loss

    def _value_channel_loss(self, pred: torch.Tensor,
                            exploration_gt: torch.Tensor,
                            obstacle_gt: torch.Tensor,
                            target_rel: torch.Tensor = None):
        """Supervise the third channel as the decoded task-aware value layer.

        Builds the training target of eq. (valuetarget) from the geometry-built
        exploration/obstacle labels and the (training-only) target position,
        then applies the robust loss of eq. (valueloss). Returns (loss, corr)
        where corr is a detached Pearson correlation for monitoring.
        """
        if target_rel is None:
            if self.require_target:
                raise ValueError(
                    "BEVLoss requires 'target_rel' (ground-truth target "
                    "position in the BEV frame) to build the value-layer "
                    "target V*. Regenerate the dataset with the converters "
                    "(scripts/convert_expert_to_parquet.py / "
                    "scripts/convert_uav_to_parquet.py), or set "
                    "BEVLoss.require_target=False to train without value "
                    "supervision.")
            if not self._warned_missing_target:
                print('[BEVLoss] target_rel missing: value-layer supervision '
                      'is DISABLED. Regenerate the parquet data to train the '
                      'value decoder (paper Stage 1/2).')
                self._warned_missing_target = True
            return pred.sum() * 0.0, pred.new_zeros(())

        h = pred.shape[-2]
        # Cell size at this scale: the full 256-cell grid spans 102.4 m.
        cell = (BEV_SIZE * BEV_RESOLUTION) / h
        # target_rel arrives as (n, 2); the per-channel tensors are (n, 1, h, w),
        # so align the leading dims to (n, 1, 2) for broadcasting.
        if target_rel.dim() == pred.dim() - 2:
            target_rel = target_rel.unsqueeze(-2)
        value_gt = build_value_target(
            exploration_gt, obstacle_gt, target_rel,
            lambda_expl=self.lambda_expl, lambda_goal=self.lambda_goal,
            sigma_g=self.sigma_g, bev_size=h, bev_resolution=cell)

        loss = F.smooth_l1_loss(pred, value_gt, beta=self.value_huber_beta)

        with torch.no_grad():
            pv = pred.flatten()
            tv = value_gt.flatten()
            pv = pv - pv.mean()
            tv = tv - tv.mean()
            corr = (pv * tv).sum() / (pv.norm() * tv.norm() + 1e-8)

        return loss, corr

    def _single_scale_loss(self, pred: torch.Tensor, target: torch.Tensor,
                           target_rel: torch.Tensor = None):
        """Per-scale channel losses.

        Args:
            pred: (b, s, 3, h, w) prediction at this scale
            target: (b, s, 3, h, w) GT downsampled to this scale
            target_rel: (b, s, 2) target position (world metres, grid frame)
        """
        b, s, c, h, w = pred.shape
        # Cast to float32 so half-precision predictions (from AMP) match the
        # float32 labels ("Found dtype Float but expected Half").
        pred_packed = pred.reshape(b * s, c, h, w).float()
        target_packed = target.reshape(b * s, c, h, w).float()
        tr_packed = None if target_rel is None else target_rel.reshape(
            b * s, target_rel.shape[-1]).float()

        exploration = self._binary_channel_loss(pred_packed[:, 0:1],
                                                target_packed[:, 0:1])
        obstacle = self._binary_channel_loss(pred_packed[:, 1:2],
                                             target_packed[:, 1:2])
        value, corr = self._value_channel_loss(
            pred_packed[:, 2:3], target_packed[:, 0:1],
            target_packed[:, 1:2], tr_packed)
        return exploration, obstacle, value, corr

    def forward(self, prediction, target: torch.Tensor,
                target_rel: torch.Tensor = None) -> dict:
        # Accept a plain tensor (single scale) for backward compatibility.
        if isinstance(prediction, torch.Tensor):
            prediction = {'bev_256': prediction}

        b, s = target.shape[:2]
        full_res = target.shape[-1]

        total_exploration = 0.0
        total_obstacle = 0.0
        total_value = 0.0
        value_corr = None

        for scale, scale_weight in self.multiscale_weights.items():
            key = f'bev_{scale}'
            if key not in prediction:
                continue

            pred_s = prediction[key]
            if scale == full_res:
                target_s = target
            else:
                # Area downsampling = per-cell coverage ratio for the binary
                # channels and expected value for the value channel.
                target_s = F.interpolate(
                    target.reshape(b * s, *target.shape[2:]),
                    size=(scale, scale),
                    mode='area',
                ).reshape(b, s, *target.shape[2:-2], scale, scale)

            exploration, obstacle, value, corr = self._single_scale_loss(
                pred_s, target_s, target_rel)
            total_exploration = total_exploration + scale_weight * exploration
            total_obstacle = total_obstacle + scale_weight * obstacle
            total_value = total_value + scale_weight * value
            if scale == full_res:
                value_corr = corr

        losses = {}
        losses['bev_exploration'] = self.exploration_weight * total_exploration
        losses['bev_obstacle'] = self.obstacle_weight * total_obstacle
        losses['bev_value'] = self.value_weight * total_value
        losses['bev_total'] = losses['bev_exploration'] + losses[
            'bev_obstacle'] + losses['bev_value']
        # Monitoring only (detached); NOT part of bev_total.
        losses['bev_value_corr'] = value_corr.detach() \
            if isinstance(value_corr, torch.Tensor) else value_corr
        return losses


@gin.configurable
class SearchWorldLoss(nn.Module):
    ''' Aggregated loss for the SearchWorld model training.

        Inputs:
            output (dict): dict of model outputs
                action: (b, s, c_a)
                prior: dict of prior state estimator
                posterior: dict of posterior state estimator
                semantic_segmentation_1:  (b, s, c_semantic, h, w)
                rgb_1: (b, s, 3, h, w)
                bev_pred: (b, s, 3, 256, 256)
            batch (Dict): dict of the target tensors:
                action: (b, s, c_a)
                semantic_label: (b, s, h, w)
                bev_gt: (b, s, 3, 256, 256)

        Returns:
            losses: dict of loss items (action, kl, semantic_segmentation)
    '''
    def __init__(self,
                 action_weight: float,
                 path_weight: float,
                 kl_weight: float,
                 semantic_weight: float,
                 rgb_weight: float,
                 diffusion_weight: float,
                 depth_weight: float,
                 bev_weight: float = 1.0,
                 bev_exploration_weight: float = 1.0,
                 bev_obstacle_weight: float = 1.0,
                 bev_value_weight: float = 1.0,
                 enable_semantic: bool = False,
                 enable_rgb_stylegan: bool = False,
                 enable_rgb_diffusion: bool = True,
                 enable_policy_diffusion: bool = False,
                 enable_bev_decoder: bool = False,
                 is_gwm_pretrain: bool = True):
        super().__init__()
        self.action_weight = action_weight
        self.path_weight = path_weight
        self.kl_weight = kl_weight
        self.semantic_weight = semantic_weight
        self.rgb_weight = rgb_weight
        self.diffusion_weight = diffusion_weight
        self.depth_weight = depth_weight
        self.bev_weight = bev_weight
        self.bev_exploration_weight = bev_exploration_weight
        self.bev_obstacle_weight = bev_obstacle_weight
        self.bev_value_weight = bev_value_weight
        self.enable_semantic = enable_semantic
        self.enable_rgb_stylegan = enable_rgb_stylegan
        self.enable_rgb_diffusion = enable_rgb_diffusion
        self.enable_policy_diffusion = enable_policy_diffusion
        self.enable_bev_decoder = enable_bev_decoder
        self.is_gwm_pretrain = is_gwm_pretrain

        if not self.is_gwm_pretrain:
            if self.enable_policy_diffusion:
                self.policy_diffusion_loss = DiffusionLoss()
            else:
                self.action_loss = ActionLoss()
                self.path_loss = PathLoss()

        self.kl_loss = KLLoss()

        if self.enable_semantic:
            self.segmentation_loss = SegmentationLoss()
        if self.enable_rgb_stylegan:
            self.rgb_loss = RgbLoss()
        if self.enable_rgb_diffusion:
            self.diffusion_loss = DiffusionLoss()

        if self.enable_bev_decoder:
            self.bev_loss = BEVLoss()

        self.depth_loss = DepthLoss()

    def forward(self, output: Dict, batch: Dict) -> Dict:
        losses = {}

        if not self.is_gwm_pretrain:
            if self.enable_policy_diffusion:
                losses[
                    'action'] = self.action_weight * self.policy_diffusion_loss(
                        output['action_noise'], output['action_noise_pred'])
                losses['path'] = self.path_weight * self.policy_diffusion_loss(
                    output['path_noise'], output['path_noise_pred'])
            else:
                losses['action'] = self.action_weight * self.action_loss(
                    output['action'], batch['action'])
                if 'path' in output and 'path' in batch:
                    losses['path'] = self.path_weight * self.path_loss(
                        output['path'], batch['path'])

        losses['kl'] = self.kl_weight * self.kl_loss(output['prior'],
                                                     output['posterior'])

        # Semantic segmentation loss.
        if self.enable_semantic:
            for downsampling_factor in [1, 2, 4]:
                if f"semantic_segmentation_{downsampling_factor}" not in output:
                    continue
                semantic_segmentation_loss = self.segmentation_loss(
                    prediction=output[
                        f"semantic_segmentation_{downsampling_factor}"],
                    target=batch[f"semantic_label_{downsampling_factor}"],
                )
                discount = 1 / downsampling_factor
                losses[f"semantic_segmentation_{downsampling_factor}"] = (
                    discount * self.semantic_weight *
                    semantic_segmentation_loss)

        # StyleGan RGB regression loss.
        if self.enable_rgb_stylegan:
            for downsampling_factor in [1, 2, 4]:
                if f"rgb_{downsampling_factor}" not in output:
                    continue
                discount = 1 / downsampling_factor
                rgb_loss = self.rgb_loss(
                    prediction=output[f"rgb_{downsampling_factor}"],
                    target=batch[f"rgb_label_{downsampling_factor}"],
                )
                losses[
                    f"rgb_{downsampling_factor}"] = discount * self.rgb_weight * rgb_loss

        # Diffusion RGB loss.
        if self.enable_rgb_diffusion:
            losses['diffusion'] = self.diffusion_weight * self.diffusion_loss(
                output['rgb_noise'], output['rgb_noise_pred'])

        # Depth loss.
        if 'depth' in output and 'depth_gt' in output:
            losses['depth'] = self.depth_weight * self.depth_loss(
                output['depth'], output['depth_gt'])

        # BEV loss.
        if self.enable_bev_decoder and 'bev_pred' in output and 'bev_gt' in batch:
            # Prefer the multi-scale dict when the model provides it; fall back
            # to the plain tensor for older checkpoints / eval scripts.
            bev_prediction = output.get('bev_pred_ms', output['bev_pred'])
            # target_rel (target position in the BEV frame) is a training-only
            # supervision input used to build the value-layer target V*.
            bev_loss_dict = self.bev_loss(bev_prediction, batch['bev_gt'],
                                          batch.get('target_rel'))
            losses['bev_exploration'] = bev_loss_dict['bev_exploration']
            losses['bev_obstacle'] = bev_loss_dict['bev_obstacle']
            losses['bev_value'] = bev_loss_dict['bev_value']
            losses['bev'] = self.bev_weight * bev_loss_dict['bev_total']
            # Monitoring-only entry (detached). 'monitor_' keys are excluded
            # from the summed total by SearchWorldTrainer.loss_reducing.
            if 'bev_value_corr' in bev_loss_dict:
                losses['monitor_bev_value_corr'] = bev_loss_dict[
                    'bev_value_corr']

        return losses
