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

"""
Dataset for UAV data in Parquet format for faster loading.
"""

import bisect
import io
import json
import os
from typing import Dict, Optional, Tuple

import gin
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torchvision.transforms.functional as tvf
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from model.dataset.data_constants import INPUT_IMAGE_SIZE


def interpolate_resize(x, size, mode=tvf.InterpolationMode.NEAREST):
    '''Resize the tensor with interpolation'''
    x = tvf.resize(x, size, interpolation=mode, antialias=True)
    return x


@gin.configurable
class UAVParquetDataModule(pl.LightningDataModule):
    '''Lightning DataModule for UAV data in Parquet format.'''
    def __init__(self,
                 dataset_path: str,
                 batch_size: int,
                 sequence_length: int,
                 num_workers: int,
                 enable_semantic: bool = False,
                 enable_rgb_stylegan: bool = False,
                 is_gwm_pretrain: bool = False,
                 precomputed_semantic_label: bool = True,
                 use_lazy_loading: bool = True,
                 text_encoder_name: str = 'google/siglip2-base-patch16-224',
                 enable_bev: bool = False):
        super().__init__()
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.num_workers = num_workers
        self.enable_semantic = enable_semantic
        self.enable_rgb_stylegan = enable_rgb_stylegan
        self.is_gwm_pretrain = is_gwm_pretrain
        self.precomputed_semantic_label = precomputed_semantic_label
        self.use_lazy_loading = use_lazy_loading
        self.text_encoder_name = text_encoder_name
        self.enable_bev = enable_bev
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        common = dict(
            sequence_length=self.sequence_length,
            enable_semantic=self.enable_semantic,
            enable_rgb_stylegan=self.enable_rgb_stylegan,
            is_gwm_pretrain=self.is_gwm_pretrain,
            precomputed_semantic_label=self.precomputed_semantic_label,
            use_lazy_loading=self.use_lazy_loading,
            text_encoder_name=self.text_encoder_name,
            enable_bev=self.enable_bev,
        )
        if stage == 'fit' or stage is None:
            train_path = os.path.join(self.dataset_path, 'train')
            self.train_dataset = UAVParquetDataset(dataset_path=train_path, **common)
            val_path = os.path.join(self.dataset_path, 'val')
            self.val_dataset = UAVParquetDataset(dataset_path=val_path, **common)
        if stage == 'test' or stage is None:
            test_path = os.path.join(self.dataset_path, 'test')
            self.test_dataset = UAVParquetDataset(dataset_path=test_path, **common)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                         batch_size=self.batch_size,
                         num_workers=self.num_workers,
                         pin_memory=True,
                         persistent_workers=self.num_workers > 0,
                         prefetch_factor=4 if self.num_workers > 0 else None,
                         drop_last=True,
                         shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                         batch_size=self.batch_size,
                         num_workers=self.num_workers,
                         pin_memory=True,
                         persistent_workers=self.num_workers > 0,
                         prefetch_factor=4 if self.num_workers > 0 else None,
                         drop_last=True,
                         shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                         batch_size=self.batch_size,
                         num_workers=self.num_workers,
                         pin_memory=True,
                         persistent_workers=self.num_workers > 0,
                         prefetch_factor=4 if self.num_workers > 0 else None,
                         drop_last=True,
                         shuffle=False)


class UAVParquetDataset(Dataset):
    '''Dataset for UAV data in Parquet format.'''
    def __init__(self,
                 dataset_path: str,
                 sequence_length: int,
                 enable_semantic: bool = False,
                 enable_rgb_stylegan: bool = False,
                 is_gwm_pretrain: bool = False,
                 precomputed_semantic_label: bool = True,
                 use_lazy_loading: bool = True,
                 text_encoder_name: str = 'google/siglip2-base-patch16-224',
                 enable_bev: bool = False):
        super().__init__()
        self.sequence_length = sequence_length
        self.enable_semantic = enable_semantic
        self.enable_rgb_stylegan = enable_rgb_stylegan
        self.is_gwm_pretrain = is_gwm_pretrain
        self.precomputed_semantic_label = precomputed_semantic_label
        self.use_lazy_loading = use_lazy_loading
        self.enable_bev = enable_bev

        # Define required columns
        self.required_columns = ['driving_command', 'ego_speed', 'camera_image']
        if self.enable_semantic and self.precomputed_semantic_label:
            self.required_columns += ['semantic_labels', 'perspective_semantic_image_shape']
        if self.enable_bev:
            self.required_columns += ['bev_map', 'bev_map_shape']
        # Optional geometry/supervision columns written by the converters:
        #   pose       (3,) agent (x, y, yaw) in the allocentric BEV frame;
        #              anchors the stage-3 action motion footprints.
        #   target_rel (2,) ground-truth target position in that frame, used
        #              ONLY to build the value-layer training target V*.
        # They are absent in older parquet dumps, so they are added to the
        # per-file read only when the file actually carries them.
        self.optional_columns = ['pose', 'target_rel']
        self._schema_columns_cache = {}

        # Try to load cached text features first.
        # Supported locations (first match wins):
        #   1) <dataset_path>/text_feat_cache.pt
        #   2) <parent>/text_feat_cache_<split>.pt  (e.g. root/train -> root/text_feat_cache_train.pt)
        cache_candidates = [
            os.path.join(dataset_path, 'text_feat_cache.pt'),
            os.path.join(os.path.dirname(dataset_path.rstrip('/')),
                         f"text_feat_cache_{os.path.basename(dataset_path.rstrip('/'))}.pt"),
        ]
        cache_path = next((p for p in cache_candidates if os.path.exists(p)), None)
        if cache_path is not None:
            print(f"Loading text features from cache: {cache_path}")
            cache_data = torch.load(cache_path)
            self.text_feat_dim = cache_data.get('text_feat_dim', 768)
            self.text_feat = cache_data.get('text_feat', {})
            self.default_text_feat = np.zeros(self.text_feat_dim, dtype=np.float32)
        else:
            # Try to initialize text encoder
            try:
                self._init_text_encoder(text_encoder_name)
                self.text_feat = {}
            except Exception as e:
                print(f"Warning: Could not initialize text encoder: {e}")
                self.text_feat_dim = 768
                self.text_feat = {}
                self.default_text_feat = np.zeros(self.text_feat_dim, dtype=np.float32)

        if self.use_lazy_loading:
            self.file_paths = []
            self.file_sizes = []
            self.file_metadata = []  # Store task description for each file
            self.accumulated_sample_sizes = [0]
            self.num_samples = 0

            if not os.path.exists(dataset_path):
                return

            for episode in sorted(os.listdir(dataset_path)):
                episode_path = os.path.join(dataset_path, episode)
                if not os.path.isdir(episode_path):
                    continue

                pqt_files = sorted([
                    f for f in os.listdir(episode_path)
                    if f.endswith('.pqt')
                ])

                for pqt_file in pqt_files:
                    parquet_path = os.path.join(episode_path, pqt_file)

                    # Try to load metadata
                    metadata_path = parquet_path.replace('.pqt', '_metadata.json')
                    metadata = {}
                    if os.path.exists(metadata_path):
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)

                    sample_count = self._get_sample_count(parquet_path)
                    self.file_paths.append(parquet_path)
                    self.file_sizes.append(sample_count)
                    self.file_metadata.append(metadata)

                    usable_samples = sample_count // self.sequence_length
                    self.num_samples += usable_samples
                    self.accumulated_sample_sizes.append(self.num_samples)
        else:
            self.dfs = []
            self.df_text_feats = []  # Text feat for each df
            self.accumulated_sample_sizes = [0]
            self.num_samples = 0

            if not os.path.exists(dataset_path):
                return

            for episode in tqdm(sorted(os.listdir(dataset_path)),
                              desc=f"Loading {os.path.basename(dataset_path)}"):
                episode_path = os.path.join(dataset_path, episode)
                if not os.path.isdir(episode_path):
                    continue

                pqt_files = sorted([
                    f for f in os.listdir(episode_path)
                    if f.endswith('.pqt')
                ])

                for pqt_file in pqt_files:
                    parquet_path = os.path.join(episode_path, pqt_file)

                    # Try to load metadata
                    metadata_path = parquet_path.replace('.pqt', '_metadata.json')
                    task_desc = None
                    if os.path.exists(metadata_path):
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                            task_desc = metadata.get('task_description')

                    df = pd.read_parquet(
                        parquet_path,
                        columns=self._parquet_columns(parquet_path),
                        engine='pyarrow'
                    )
                    self.dfs.append(df)

                    # Get or compute text feat for this episode
                    self.df_text_feats.append(
                        self._get_text_feat_for_desc(task_desc)
                    )

                    self.num_samples += len(df) // self.sequence_length
                    self.accumulated_sample_sizes.append(self.num_samples)

    def _init_text_encoder(self, model_name: str):
        from transformers import AutoModel, AutoTokenizer
        self._text_processor = AutoTokenizer.from_pretrained(model_name)
        self._text_model = AutoModel.from_pretrained(model_name)
        self._text_model.eval()
        for p in self._text_model.parameters():
            p.requires_grad = False
        with torch.no_grad():
            tok = self._text_processor(
                text='probe', return_tensors='pt',
                padding=True, truncation=True, max_length=64
            )
            text_emb = self._normalize_text_feat(
                self._text_model.get_text_features(**tok)
            )
        self.text_feat_dim = text_emb.shape[-1]
        self.default_text_feat = np.zeros(self.text_feat_dim, dtype=np.float32)

    def _normalize_text_feat(self, text_emb):
        import torch.nn.functional as F
        # transformers >= 5.x returns a ModelOutput from get_text_features
        # instead of a bare tensor. The equivalent of the old (4.x) return
        # value is `pooler_output` (SigLIP2-base has no text projection,
        # so `text_embeds` is None).
        if not torch.is_tensor(text_emb):
            if getattr(text_emb, 'pooler_output', None) is not None:
                text_emb = text_emb.pooler_output
            elif getattr(text_emb, 'text_embeds', None) is not None:
                text_emb = text_emb.text_embeds
            else:
                raise ValueError(
                    f"Cannot extract text embedding from {type(text_emb)}")
        return F.normalize(text_emb, dim=-1).cpu().numpy()

    def _get_text_feat_for_desc(self, task_desc):
        '''Get text feature for a task description.

        Lookup order: text_feat cache dict (from text_feat_cache.pt or
        memoized online encodings) -> online encoding -> zero fallback.
        '''
        if task_desc:
            feat = self.text_feat.get(task_desc)
            if feat is not None:
                if isinstance(feat, torch.Tensor):
                    feat = feat.cpu().numpy()
                feat = np.asarray(feat, dtype=np.float32).reshape(-1)
                return feat
            if hasattr(self, '_text_model'):
                feat = self._encode_text(task_desc)
                # Memoize so each unique description is encoded only once.
                self.text_feat[task_desc] = feat
                return feat
        return self.default_text_feat

    @torch.no_grad()
    def _encode_text(self, text: str):
        if not hasattr(self, '_text_model'):
            return self.default_text_feat

        device = next(self._text_model.parameters()).device
        tok = self._text_processor(
            text=text, return_tensors='pt',
            padding=True, truncation=True, max_length=64
        )
        tok = {k: v.to(device) for k, v in tok.items()}
        text_emb = self._normalize_text_feat(
            self._text_model.get_text_features(**tok)
        )
        return text_emb[0]

    def _parquet_columns(self, parquet_path):
        '''Columns to read from a parquet file: required + present optional.

        Reads the parquet schema once per file (cached) so optional columns
        (e.g. 'pose', 'target_rel') can be picked up without breaking older
        datasets that lack them.
        '''
        cols = self._schema_columns_cache.get(parquet_path)
        if cols is None:
            import pyarrow.parquet as pq
            names = set(pq.read_schema(parquet_path).names)
            cols = self.required_columns + [
                c for c in self.optional_columns if c in names]
            self._schema_columns_cache[parquet_path] = cols
        return cols

    def _get_sample_count(self, parquet_path):
        import pyarrow.parquet as pq
        metadata = pq.read_metadata(parquet_path)
        return metadata.num_rows

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        batch = {}

        if self.use_lazy_loading:
            file_idx = bisect.bisect_right(self.accumulated_sample_sizes, index) - 1
            file_path = self.file_paths[file_idx]
            file_metadata = self.file_metadata[file_idx]
            relative_index = index - self.accumulated_sample_sizes[file_idx]
            sequence_start = relative_index * self.sequence_length

            df = pd.read_parquet(
                file_path,
                columns=self._parquet_columns(file_path),
                engine='pyarrow'
            )
            sequence_end = sequence_start + self.sequence_length
            sequence_df = df.iloc[sequence_start:sequence_end]

            # Get text feat
            task_desc = file_metadata.get('task_description')
            text_feat = self._get_text_feat_for_desc(task_desc)

            for seq_idx in range(self.sequence_length):
                element = self._get_element(sequence_df, seq_idx)
                for k, v in element.items():
                    batch[k] = batch.get(k, []) + [v]
        else:
            df_idx = bisect.bisect_left(self.accumulated_sample_sizes, index + 1) - 1
            sample_idx = (index - self.accumulated_sample_sizes[df_idx]) * self.sequence_length

            text_feat = self.df_text_feats[df_idx]

            for seq_idx in range(self.sequence_length):
                element = self._get_element(self.dfs[df_idx], sample_idx + seq_idx)
                for k, v in element.items():
                    batch[k] = batch.get(k, []) + [v]

        for k, v in batch.items():
            if k == 'action':
                batch[k] = torch.tensor(np.stack(v), dtype=torch.int64)
            else:
                batch[k] = torch.from_numpy(np.stack(v)).type(torch.float32)

        # Add text feat (same for all sequence steps)
        batch['text_feat'] = torch.from_numpy(text_feat).type(torch.float32)
        batch['text_feat'] = batch['text_feat'].unsqueeze(0).repeat(
            self.sequence_length, 1
        )

        self._down_sample_input_image(batch)

        if self.enable_rgb_stylegan:
            self._compose_rgb_labels(batch)

        if self.enable_semantic:
            self._compose_semantic_labels(batch)

        return batch

    def _get_element(self, df, sample_index):
        sample = df.iloc[sample_index]
        element = {}

        element['action'] = self._get_action(sample)
        element['image'] = self._get_rgb_image(sample)
        element['relative_pose'] = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # Optional geometry columns written by the converters:
        #   pose       (3,) agent (x, y, yaw) in the allocentric BEV frame
        #              (stage-3 motion-footprint anchor).
        #   target_rel (2,) ground-truth target position in that frame,
        #              training-only supervision for the value layer.
        # Absent in older parquet files (pre-regeneration).
        if 'pose' in sample.index and sample['pose'] is not None:
            element['pose'] = np.array(sample['pose'], dtype=np.float32)
        if 'target_rel' in sample.index and sample['target_rel'] is not None:
            element['target_rel'] = np.array(
                sample['target_rel'], dtype=np.float32)  # (2,)

        if self.enable_semantic:
            element['semantic_label'] = self._get_semantic_label(sample)

        if self.enable_bev:
            bev = self._get_bev(sample)
            element['bev_gt'] = bev      # (3, 256, 256) supervision target
            element['bev_memory'] = bev  # (3, 256, 256) observation input for BEV encoder

        return element

    def _get_bev(self, sample):
        bev_flat = np.array(sample['bev_map'], dtype=np.float32)
        bev_shape = sample['bev_map_shape']
        bev = bev_flat.reshape(bev_shape)  # (3, 256, 256)
        return bev

    def _get_rgb_image(self, sample):
        rgb_image = Image.open(io.BytesIO(sample['camera_image']))
        # Convert to RGB (in case it's RGBA)
        if rgb_image.mode != 'RGB':
            rgb_image = rgb_image.convert('RGB')
        return np.transpose(np.array(rgb_image), (2, 0, 1)) / 255.0

    def _get_semantic_label(self, sample):
        semantic_labels = np.array(sample['semantic_labels'], dtype=np.uint8)
        semantic_labels = semantic_labels.reshape(
            sample['perspective_semantic_image_shape'])
        return semantic_labels

    def _get_action(self, sample):
        return sample['driving_command']

    def _down_sample_input_image(self, batch):
        size = INPUT_IMAGE_SIZE
        batch['image'] = interpolate_resize(
            batch['image'],
            size,
            mode=tvf.InterpolationMode.BILINEAR,
        )
        if 'semantic_label' in batch:
            batch['semantic_label'] = interpolate_resize(
                batch['semantic_label'],
                size,
                mode=tvf.InterpolationMode.NEAREST)

    def _compose_semantic_labels(self, batch):
        batch['semantic_label_1'] = batch['semantic_label']
        h, w = batch['semantic_label_1'].shape[-2:]
        for downsample_factor in [2, 4]:
            size = h // downsample_factor, w // downsample_factor
            previous_label_factor = downsample_factor // 2
            batch[f'semantic_label_{downsample_factor}'] = interpolate_resize(
                batch[f'semantic_label_{previous_label_factor}'],
                size,
                mode=tvf.InterpolationMode.NEAREST)

    def _compose_rgb_labels(self, batch):
        batch['rgb_label_1'] = batch['image']
        h, w = batch['rgb_label_1'].shape[-2:]
        for downsample_factor in [2, 4]:
            size = h // downsample_factor, w // downsample_factor
            previous_label_factor = downsample_factor // 2
            batch[f'rgb_label_{downsample_factor}'] = interpolate_resize(
                batch[f'rgb_label_{previous_label_factor}'],
                size,
                mode=tvf.InterpolationMode.BILINEAR)
