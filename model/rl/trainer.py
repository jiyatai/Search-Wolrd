# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

'''Stage-3 value-guided imagination planning Lightning module.

Pipeline per training step:
  1. Load a real batch (expert parquet; no reward/done in data).
  2. Run the FROZEN world model's observation encoder + RSSM forward
     (posterior) on the real batch to obtain an initial latent state and the
     recorded agent pose.
  3. Roll the FROZEN prior (RSSM.imagine_step) forward under the actor's
     sampled actions; for every candidate movement action, imagine the
     successor, decode the frozen value layer, and pool it over the action's
     motion footprint.
  4. Build the value-guided target pi+ and train ONLY the action head with
     KL(pi+ || pi_theta) on movement actions, stop pinned at pi_BC(stop).

There is NO reward function, NO scalar critic, NO GAE and NO collision/
pose head. The only trainable parameters are the action-head weights.
'''
from typing import Dict

import gin
import pytorch_lightning as pl
import torch

from model.rl.actor_critic import ContinuedActionPolicy, ImagRLActor
from model.rl.imagination import ValueGuidedImaginationEngine
from model.trainer import SearchWorldTrainer
from model.searchworld.searchworld import SearchWorld


@gin.configurable
class ImaginationRLModule(pl.LightningModule):
    '''Value-guided imagination planning on a frozen world model.'''

    def __init__(self,
                 checkpoint_path: str,
                 weight_decay: float = 0.01,
                 lr: float = 1e-4,
                 action_loss_weight: float = 1.0,
                 load_strict: bool = True,
                 use_fresh_actor: bool = False):
        super().__init__()
        self.save_hyperparameters()

        # ---- Frozen world model (loaded from the stage-2 checkpoint).
        world_model_trainer = SearchWorldTrainer.load_from_checkpoint(
            checkpoint_path, map_location='cpu', strict=load_strict)
        self.world_model = world_model_trainer.model
        self.world_model.eval()

        # Dimensions from the loaded model (avoid hard-coding mismatches).
        rssm = self.world_model.rssm
        self.state_dim = rssm.hidden_state_dim + rssm.state_dim
        text_feat_dim = self.world_model.observation_encoder.text_feat_dim

        # ---- Actor: the Stage-2 BC action head, continued (default), or a
        # fresh head (warm-start ablation). The continued actor deep-copies
        # the deployed BC head BEFORE it is frozen, so the copy is trainable
        # while the BC original stays frozen as the behaviour-cloning prior
        # that defines pi_BC and the pinned stop probability.
        self.use_fresh_actor = use_fresh_actor
        if use_fresh_actor:
            self.actor = ImagRLActor(latent_state_dim=self.state_dim,
                                     text_feat_dim=text_feat_dim)
        else:
            self.actor = ContinuedActionPolicy(
                self.world_model.action_policy)
            # Re-enable gradients on the continued copy (deepcopy would
            # otherwise inherit whatever the source had; be explicit).
            for p in self.actor.parameters():
                p.requires_grad_(True)
        # Frozen pi_BC anchor: the ORIGINAL BC head inside the world model.
        self.bc_policy = self.world_model.action_policy

        # ---- Now freeze the world model (BC anchor included).
        for p in self.world_model.parameters():
            p.requires_grad = False

        # ---- Value-guided imagination engine (orchestrator; Rssm and value
        # decoder frozen, no parameters of its own besides the actor).
        self.engine = ValueGuidedImaginationEngine(
            rssm=self.world_model.rssm,
            bev_decoder=self.world_model.bev_decoder,
            actor=self.actor,
            bc_policy=self.bc_policy)

        self.action_loss_weight = action_loss_weight
        self.weight_decay = weight_decay
        self.lr = lr

    def train(self, mode: bool = True):
        '''Keep the frozen world model in eval mode regardless of what
        Lightning flips on the outer module.

        Lightning calls `.train()` on the whole module each epoch (and after
        validation). If the frozen world model were flipped to train mode,
        the BEV decoder's BatchNorm running stats would be updated with
        imagined-rollout statistics and the RSSM's latent dropout would
        activate - both corrupt the frozen model. Overriding train() is
        order-proof: ANY .train()/.eval() call keeps world_model.eval().
        '''
        super().train(mode)
        self.world_model.eval()
        return self

    def on_train_epoch_start(self):
        self.world_model.eval()

    def on_validation_epoch_start(self):
        self.world_model.eval()

    def _encode_real_batch(self, batch: Dict) -> Dict[str, torch.Tensor]:
        '''Run the frozen observation encoder + RSSM posterior over the real
        sequence; return the final posterior state, text_feat and the pose of
        that final real step (the footprint anchor for the imagined rollout).

        The RSSM resets h_t every 8-step window internally, matching how the
        world model was trained in stage 2 (window memory resets; the
        BEV-encoding branch is the only episode-level history channel).
        '''
        if 'pose' not in batch:
            raise RuntimeError(
                "Stage-3 value-guided planning needs the agent pose (x, y, "
                "yaw) at each real step ('pose' parquet column) to anchor the "
                "action motion footprints. Regenerate the dataset with "
                "scripts/convert_expert_to_parquet.py (see its 'pose' column).")
        with torch.no_grad():
            obs = self.world_model.observation_encoder(batch)
            embedding = obs['embedding']                 # (b, s, emb_dim)
            action = batch['action']                     # (b, s) int64
            text_feat = batch['text_feat'][:, 0]         # (b, 768) per-seq constant
            out = self.world_model.rssm(embedding, action, use_sample=False)
            h_last = out['posterior']['hidden_state'][:, -1]    # (b, hidden_dim)
            z_last = out['posterior']['sample'][:, -1]          # (b, state_dim)
        return {
            'h0': h_last.detach(),
            'z0': z_last.detach(),
            'text_feat': text_feat.detach(),
            'pose0': batch['pose'][:, -1].detach().float(),     # (b, 3)
        }

    def shared_step(self, batch) -> Dict:
        init = self._encode_real_batch(batch)
        return self.engine.rollout(init['h0'], init['z0'],
                                   init['text_feat'], init['pose0'])

    def training_step(self, batch, batch_idx):
        losses = self.shared_step(batch)
        for key, value in losses.items():
            self.log(f'train/plan/{key}', value, sync_dist=True)
        total = self.action_loss_weight * losses['action_loss']
        self.log('train/plan/total_loss', total, sync_dist=True)
        return total

    def validation_step(self, batch, batch_idx):
        losses = self.shared_step(batch)
        for key, value in losses.items():
            self.log(f'validation/plan/{key}', value, sync_dist=True)
        total = self.action_loss_weight * losses['action_loss']
        self.log('val_loss', total, sync_dist=True, on_epoch=True)
        return total

    def configure_optimizers(self):
        # ONLY the action head receives gradients (paper Stage 3).
        params = [
            {'params': self.actor.parameters(),
             'weight_decay': self.weight_decay},
        ]
        optimizer = torch.optim.AdamW(params, lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.estimated_stepping_batches)
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]
