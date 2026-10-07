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

import gin
import pytorch_lightning as pl
import torch
import wandb
from torch import nn

from model.loss.losses import SearchWorldLoss
from model.searchworld.searchworld import SearchWorld
from model.eval.searchworld_metrics import SearchWorldMetrics
from model.visualization import visualise_rgb, visualise_semantic, visualise_depth, visualise_attention


@gin.configurable
class SearchWorldTrainer(pl.LightningModule):
    '''PyTorch Lightning module for training the SearchWorld model.
    '''
    def __init__(self, weight_decay: float, lr: float,
                 scheduler_pct_start: float):
        super().__init__()
        self.weight_decay = weight_decay
        self.lr = lr
        self.scheduler_pct_start = scheduler_pct_start
        # Save hyperparameters.
        self.save_hyperparameters()

        # Model
        self.model = SearchWorld()

        # Losses
        self.loss = SearchWorldLoss()

        # Metrics calculator.
        self.metrics = SearchWorldMetrics()

    def forward(self, batch):
        return self.model(batch)

    def inference_prediction(self,
                             batch,
                             enable_semantic_inference=True,
                             enable_rgb_inference=False):
        return self.model.inference_prediction(batch,
                                               enable_semantic_inference,
                                               enable_rgb_inference)

    def inference(self, batch, enable_semantic, enable_rgb, enable_depth):
        return self.model.inference(batch, enable_semantic, enable_rgb,
                                    enable_depth)

    def shared_step(self, batch):
        output = self.forward(batch)
        losses = self.loss(output, batch)
        return losses, output

    def training_step(self, batch, batch_idx):
        losses, output = self.shared_step(batch)
        self.log_and_visualize(batch,
                               output,
                               losses,
                               batch_idx,
                               prefix='train')
        return self.loss_reducing(losses)

    def validation_step(self, batch, batch_idx):
        losses, output = self.shared_step(batch)
        self.log_and_visualize(batch,
                               output,
                               losses,
                               batch_idx,
                               prefix='validation')
        val_loss = self.loss_reducing(losses)
        return {'val_loss': val_loss}

    def test_step(self, batch, batch_idx):
        losses, output = self.shared_step(batch)
        self.log_and_visualize(batch, output, losses, batch_idx, prefix='test')

    def log_and_visualize(self,
                          batch,
                          output,
                          losses,
                          batch_idx,
                          prefix='train'):
        # Log losses
        for key, value in losses.items():
            self.log(f'{prefix}/losses/{key}', value, sync_dist=True)

        # Log evaluation metrics
        metrics = self.metrics.evaluate(batch, output, prefix)
        self.log_dict(metrics, sync_dist=True, on_epoch=True, on_step=False)

        # Log total loss and videos for validation.
        if prefix == 'validation':
            # Log total validation loss.
            self.log('val_loss',
                     self.loss_reducing(losses),
                     sync_dist=True,
                     on_epoch=True)
            # Log video for the first batch of validation.
            if batch_idx == 0:
                self.visualise(batch, output, batch_idx, prefix=prefix)

    def log_video(self, name, viz):
        if torch.distributed.get_rank() == 0:
            wandb.log({name: wandb.Video(viz.cpu())}, step=self.global_step)

    def visualise(self, batch, output, batch_idx, prefix='train'):
        if 'semantic_segmentation_1' in output:
            # Visualize semantic prediction
            semantic_viz = visualise_semantic(batch, output)
            semantic_name = f'{prefix}/videos/semantic/epoch_{self.current_epoch}_batch_{batch_idx}'
            self.log_video(semantic_name, semantic_viz)

        if 'rgb_1' in output:
            # Visualize rgb prediction.
            rgb_viz = visualise_rgb(batch, output)
            rgb_name = f'{prefix}/videos/rgb/epoch_{self.current_epoch}_batch_{batch_idx}'
            self.log_video(rgb_name, rgb_viz)

        if 'depth' in output:
            # Visualize depth prediction.
            depth_viz = visualise_depth(batch, output)
            depth_name = f'{prefix}/videos/depth/epoch_{self.current_epoch}_batch_{batch_idx}'
            self.log_video(depth_name, depth_viz)

        if 'image_attentions' in output:
            # Visualize attention.
            att_viz = visualise_attention(batch, output)
            att_name = f'{prefix}/videos/attention/epoch_{self.current_epoch}_batch_{batch_idx}'
            self.logger.experiment.log({att_name: att_viz})

    def loss_reducing(self, loss: torch.Tensor):
        # 'monitor_' prefixed entries are detached diagnostics, not losses.
        total_loss = sum([
            x for k, x in loss.items() if not k.startswith('monitor_')
        ])
        return total_loss

    def configure_optimizers(self):
        # Weight decay
        parameters = self._add_weight_decay(
            self.model,
            self.weight_decay,
            skip_list=['relative_position_bias_table'],
        )
        # Optimizer
        optimizer = torch.optim.AdamW(parameters, lr=self.lr, weight_decay=0.0)
        # scheduler
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=self.scheduler_pct_start,
        )
        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]

    #  Do not decay batch norm parameters and biases:
    #  https://discuss.pytorch.org/t/weight-decay-in-the-optimizers-is-a-bad\
    # -idea-especially-with-batchnorm/16994/2
    def _add_weight_decay(self,
                          model: nn.Module,
                          weight_decay: float = 0.01,
                          skip_list: list = None):
        no_decay = []
        decay = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if len(param.shape) == 1 or any(x in name for x in skip_list):
                no_decay.append(param)
            else:
                decay.append(param)
        return [
            {
                'params': no_decay,
                'weight_decay': 0.
            },
            {
                'params': decay,
                'weight_decay': weight_decay
            },
        ]
