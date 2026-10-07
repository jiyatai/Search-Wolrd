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

'''Dataset for the UAV target-search task.

Data layout (under ``dataset_path``):
    episode_0001/
        episode_summary.json     # episode-level metadata
        step_0000/
            rgb_front.png        # front-camera RGB
            depth_front.npy      # GT depth (256x256 float32)
            semantic_front.npy   # Semantic label (h, w uint8)
            state.json           # per-step state (pose, action, task...)
        step_0001/ ... step_0049/

The dataset returns a dict of tensors with sequence dimension prepended,
ready to be consumed by ``UAVObservationEncoder`` and ``RSSM``.
'''

import bisect
import json
import os
from typing import Dict, Optional, Tuple

import gin
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from model.dataset.data_constants import INPUT_IMAGE_SIZE


# Discrete action label mapping. Indices follow the paper/UAV-ON interface:
# seven movement actions plus an independent terminal ``stop`` action.
# ``start`` is a first-frame sentinel and is encoded as ``stop`` because the
# model has no eighth non-terminal action.
ACTION_TO_IDX: Dict[str, int] = {
    'forward': 0,
    'left':    1,
    'right':   2,
    'ascend':  3,
    'descend': 4,
    'rotl':    5,
    'rotr':    6,
    'stop':    7,
    'start':   7,
}
IDX_TO_ACTION: Dict[int, str] = {v: k for k, v in ACTION_TO_IDX.items()}
NUM_DISCRETE_ACTIONS = len(ACTION_TO_IDX)  # 8


def interpolate_resize(x, size, mode='bilinear'):
    '''Resize the tensor with interpolation'''
    # Handle both (C, H, W) and (S, C, H, W) formats
    orig_shape = x.shape
    if len(orig_shape) == 3:
        # Single image: (C, H, W) -> (1, C, H, W)
        x = x[None, ...]
    if mode == 'nearest':
        resized = F.interpolate(x, size, mode='nearest')
    else:
        resized = F.interpolate(x, size, mode='bilinear', align_corners=False)
    if len(orig_shape) == 3:
        return resized[0]
    else:
        return resized


def _quat_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    '''Extract yaw angle (radians) from a quaternion (x, y, z, w).'''
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return float(np.arctan2(siny_cosp, cosy_cosp))


@gin.configurable
class UAVDataset(Dataset):
    '''UAV target-search dataset.

    Inputs (constructor):
        dataset_root:         root directory with episode_* folders
        sequence_length:      number of consecutive steps per sample
        split:                one of {'train', 'val', 'test'}
        split_ratios:         (train, val, test) ratios; must sum to 1.0
        camera_view:          which RGB/depth view to load (e.g. 'front')
        text_encoder_name:    HF model id for the SigLIP text encoder used
                              to pre-compute per-episode ``text_feat``.
        enable_depth:         if True, also load ``depth_<view>.npy`` into
                              batch['depth'].
        enable_semantic:      if True, also load ``semantic_<view>.npy`` into
                              batch['semantic_label']
        seed:                 RNG seed for reproducible split assignment.

    Returned batch (each value is sequence-first):
        image:                (S, 3, H, W) float32 in [0, 1]
        relative_pose:        (S, 4)       float32 [dx, dy, dz, dyaw] vs
                                          episode start
        text_feat:            (S, 768)     float32, SigLIP text embedding
                                          of task description
        action:               (S,)         int64 discrete action label
        distance_to_target:   (S, 1)       float32 metres (for reward/eval)
        depth:                (S, H_d, W_d) float32 (only if enable_depth)
        semantic_label_1:     (S, H, W)    long (only if enable_semantic)
        semantic_label_2:     (S, H/2, W/2) long (only if enable_semantic)
        semantic_label_4:     (S, H/4, W/4) long (only if enable_semantic)
    '''
    def __init__(self,
                 dataset_path: str,
                 sequence_length: int,
                 split: str = 'train',
                 split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
                 camera_view: str = 'front',
                 text_encoder_name: str = 'google/siglip2-base-patch16-224',
                 enable_depth: bool = False,
                 enable_semantic: bool = False,
                 seed: int = 42):
        super().__init__()
        assert split in ('train', 'val', 'test')
        assert abs(sum(split_ratios) - 1.0) < 1e-6

        self.dataset_path = dataset_path
        self.sequence_length = sequence_length
        self.split = split
        self.camera_view = camera_view
        self.enable_depth = enable_depth
        self.enable_semantic = enable_semantic

        # 1) Index all episodes and sort deterministically.
        all_episodes = sorted(
            d for d in os.listdir(dataset_path)
            if d.startswith('episode_') and os.path.isdir(
                os.path.join(dataset_path, d)))
        if not all_episodes:
            raise FileNotFoundError(
                f'No episode_XXXX folders found under {dataset_path}')

        # 2) Deterministic split by episode id.
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(all_episodes))
        n_train = int(len(all_episodes) * split_ratios[0])
        n_val = int(len(all_episodes) * split_ratios[1])
        train_idx = perm[:n_train]
        val_idx = perm[n_train:n_train + n_val]
        test_idx = perm[n_train + n_val:]
        if split == 'train':
            sel = train_idx
        elif split == 'val':
            sel = val_idx
        else:
            sel = test_idx
        self.episodes = [all_episodes[i] for i in sorted(sel)]

        # 3) Read episode summaries: start pose + per-episode metadata.
        self.start_position: Dict[str, np.ndarray] = {}
        self.start_yaw: Dict[str, float] = {}
        self.total_steps: Dict[str, int] = {}
        self.task_description: Dict[str, str] = {}
        for ep in self.episodes:
            summary_path = os.path.join(dataset_path, ep,
                                        'episode_summary.json')
            with open(summary_path, 'r') as f:
                summary = json.load(f)
            self.start_position[ep] = np.array(summary['start_position'],
                                               dtype=np.float32)
            self.start_yaw[ep] = _quat_to_yaw(*summary['start_quaternion'])
            self.total_steps[ep] = int(summary['total_steps'])
            self.task_description[ep] = summary['description']

        # 4) Pre-compute per-episode text_feat via frozen SigLIP text encoder.
        # Use cache if available
        cache_path = os.path.join(dataset_path, f'text_feat_cache_{split}.pt')
        if os.path.exists(cache_path):
            print(f'[UAVDataset] Loading text features from cache: {cache_path}')
            cache_data = torch.load(cache_path)
            self.text_feat_dim = cache_data['text_feat_dim']
            self.text_feat = cache_data['text_feat']
            # Skip initializing text encoder when using cache
            self._text_processor = None
            self._text_model = None
        else:
            print(f'[UAVDataset] Computing text features (will cache to: {cache_path})')
            self.text_feat_dim = self._init_text_encoder(text_encoder_name)
            self.text_feat: Dict[str, torch.Tensor] = {}
            for ep in self.episodes:
                self.text_feat[ep] = self._encode_text(self.task_description[ep])
            # Save cache
            torch.save({
                'text_feat_dim': self.text_feat_dim,
                'text_feat': self.text_feat,
            }, cache_path)
            print(f'[UAVDataset] Cached text features to: {cache_path}')

        # 5) Build flat (episode, seq_start) index for random sampling.
        self._seq_index: list = []
        for ep in self.episodes:
            n_steps = self.total_steps[ep]
            n_seqs = max(0, n_steps - self.sequence_length + 1)
            for s in range(n_seqs):
                self._seq_index.append((ep, s))

    def _init_text_encoder(self, model_name: str) -> int:
        '''Load the SigLIP text encoder (frozen) and stash it on self.

        Note: ``google/siglip2-base-patch16-224`` uses a GemmaTokenizer
        under the hood, so we load the tokenizer via ``AutoTokenizer``
        (not ``AutoProcessor``) and pass ``padding='max_length'`` plus
        ``truncation=True`` to obtain a fixed-size input the model
        expects.
        '''
        from transformers import AutoModel, AutoTokenizer
        self._text_processor = AutoTokenizer.from_pretrained(model_name)
        self._text_model = AutoModel.from_pretrained(model_name)
        self._text_model.eval()
        for p in self._text_model.parameters():
            p.requires_grad = False
        # Probe the embedding dimension with a dummy call.
        with torch.no_grad():
            tok = self._text_processor(
                text='probe', return_tensors='pt',
                padding=True, truncation=True, max_length=64)
            text_emb = F.normalize(
                self._text_model.get_text_features(**tok), dim=-1)
        return text_emb.shape[-1]

    @torch.no_grad()
    def _encode_text(self, text: str) -> torch.Tensor:
        device = next(self._text_model.parameters()).device
        tok = self._text_processor(
            text=text, return_tensors='pt',
            padding=True, truncation=True, max_length=64)
        tok = {k: v.to(device) for k, v in tok.items()}
        text_emb = F.normalize(
            self._text_model.get_text_features(**tok), dim=-1)
        return text_emb[0].detach().cpu()

    def __len__(self) -> int:
        return len(self._seq_index)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        ep, seq_start = self._seq_index[index]
        ep_path = os.path.join(self.dataset_path, ep)
        start_pos = self.start_position[ep]
        start_yaw = self.start_yaw[ep]
        text_feat = self.text_feat[ep]

        images = []
        relative_poses = []
        actions = []
        distances = []
        depths = []
        semantic_labels = []

        for i in range(self.sequence_length):
            step_idx = seq_start + i
            step_path = os.path.join(ep_path, f'step_{step_idx:04d}')

            with open(os.path.join(step_path, 'state.json'), 'r') as f:
                state = json.load(f)

            # Image: HxWx4 RGBA uint8 -> 3xHxW float32 in [0, 1].
            # The source PNGs are stored as RGBA; convert to RGB so the
            # channel count matches the (3, H, W) layout the encoders
            # expect.
            img_path = os.path.join(step_path, f'rgb_{self.camera_view}.png')
            with Image.open(img_path) as img:
                img = img.convert('RGB')
                img = np.array(img, dtype=np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))
            images.append(img)

            # Relative pose: [dx, dy, dz, dyaw] vs episode start.
            pos = np.array([
                state['position']['x'], state['position']['y'],
                state['position']['z']
            ],
                           dtype=np.float32)
            q = state['attitude_quaternion']
            cur_yaw = _quat_to_yaw(q['x'], q['y'], q['z'], q['w'])
            rel_pos = pos - start_pos
            rel_yaw = cur_yaw - start_yaw
            # Wrap to (-pi, pi].
            rel_yaw = (rel_yaw + np.pi) % (2.0 * np.pi) - np.pi
            relative_poses.append(
                np.concatenate([rel_pos, [rel_yaw]], axis=0))

            # Action: string -> int label.
            action_str = state.get('action', 'start')
            actions.append(int(ACTION_TO_IDX.get(action_str, ACTION_TO_IDX['stop'])))

            # Distance to target (for reward/eval only, not supervision).
            distances.append(float(state.get('distance_to_target', 0.0)))

            # Depth (optional GT supervision).
            if self.enable_depth:
                depth = np.load(
                    os.path.join(step_path, f'depth_{self.camera_view}.npy'))
                depths.append(depth)

            # Semantic label (optional GT supervision).
            if self.enable_semantic:
                semantic_path = os.path.join(step_path, f'semantic_{self.camera_view}.npy')
                if os.path.exists(semantic_path):
                    semantic = np.load(semantic_path)
                else:
                    # Fallback: if semantic label doesn't exist, use zeros
                    semantic = np.zeros(img.shape[1:], dtype=np.uint8)
                semantic_labels.append(semantic)

        batch = {}
        batch['image'] = torch.from_numpy(np.stack(images))                # (S, 3, H, W)
        batch['relative_pose'] = torch.from_numpy(
            np.stack(relative_poses)).float()                              # (S, 4)
        batch['action'] = torch.tensor(actions, dtype=torch.int64)         # (S,)
        batch['distance_to_target'] = torch.tensor(distances,
                                                dtype=torch.float32).unsqueeze(-1)  # (S, 1)
        batch['text_feat'] = text_feat.unsqueeze(0).expand(
            self.sequence_length, -1).clone()                              # (S, 768)

        if self.enable_depth:
            batch['depth'] = torch.from_numpy(np.stack(depths)).float()

        if self.enable_semantic:
            self._compose_semantic_labels(batch, semantic_labels)

        # Downsample input images
        self._down_sample_input_image(batch)

        return batch

    def _down_sample_input_image(self, batch):
        size = INPUT_IMAGE_SIZE  # (h, w)
        batch['image'] = interpolate_resize(
            batch['image'],
            size,
            mode='bilinear',
        )

    def _compose_semantic_labels(self, batch, semantic_labels):
        '''Compose semantic labels at different downsampling factors.'''
        # Start with original size
        batch['semantic_label'] = torch.from_numpy(np.stack(semantic_labels)).long()  # (S, h, w)

        # Create different scales
        batch['semantic_label_1'] = batch['semantic_label']
        h, w = batch['semantic_label_1'].shape[-2:]

        for downsample_factor in [2, 4]:
            size = (h // downsample_factor, w // downsample_factor)
            previous_label_factor = downsample_factor // 2
            # Use nearest neighbor to resize
            resized = interpolate_resize(
                batch[f'semantic_label_{previous_label_factor}'].float(),
                size,
                mode='nearest'
            )
            batch[f'semantic_label_{downsample_factor}'] = resized.long()


@gin.configurable
class UAVDataModule(pl.LightningDataModule):
    '''Lightning DataModule wrapping train/val/test :class:`UAVDataset` instances.'''
    def __init__(self,
                 dataset_path: str,
                 batch_size: int,
                 sequence_length: int,
                 num_workers: int,
                 camera_view: str = 'front',
                 text_encoder_name: str = 'google/siglip2-base-patch16-224',
                 enable_depth: bool = False,
                 enable_semantic: bool = False,
                 split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
                 seed: int = 42):
        super().__init__()
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.num_workers = num_workers
        self.camera_view = camera_view
        self.text_encoder_name = text_encoder_name
        self.enable_depth = enable_depth
        self.enable_semantic = enable_semantic
        self.split_ratios = split_ratios
        self.seed = seed
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        common = dict(
            dataset_path=self.dataset_path,
            sequence_length=self.sequence_length,
            camera_view=self.camera_view,
            text_encoder_name=self.text_encoder_name,
            enable_depth=self.enable_depth,
            enable_semantic=self.enable_semantic,
            split_ratios=self.split_ratios,
            seed=self.seed,
        )
        if stage == 'fit' or stage is None:
            self.train_dataset = UAVDataset(split='train', **common)
            self.val_dataset = UAVDataset(split='val', **common)
        if stage == 'test' or stage is None:
            self.test_dataset = UAVDataset(split='test', **common)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          drop_last=True,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          drop_last=True,
                          shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          drop_last=True,
                          shuffle=False)

