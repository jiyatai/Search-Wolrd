# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

'''Test script to verify UAV dataset with semantic labels.'''

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from model.dataset.uav_dataset import UAVDataset


def test_dataset():
    '''Test UAV dataset loading with semantic labels.'''
    dataset_path = '/shared_disk/users/wenhao.lu/JYT/WorldSearch_data/BrushifyUrban_test'

    print('Testing UAV dataset with semantic labels...')
    print(f'Dataset path: {dataset_path}')

    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f'ERROR: Dataset not found at {dataset_path}')
        return False

    # Create dataset
    dataset = UAVDataset(
        dataset_path=dataset_path,
        sequence_length=2,
        split='train',
        split_ratios=(0.8, 0.1, 0.1),
        enable_semantic=True
    )

    print(f'Number of samples: {len(dataset)}')
    print(f'Number of episodes: {len(dataset.episodes)}')

    if len(dataset) == 0:
        print('ERROR: No samples found!')
        return False

    # Get a sample
    sample = dataset[0]

    print('\nSample keys:')
    for k, v in sorted(sample.items()):
        if isinstance(v, torch.Tensor):
            print(f'  {k}: {v.shape}, dtype={v.dtype}')
        else:
            print(f'  {k}: {type(v)}')

    # Check required keys
    required_keys = ['image', 'relative_pose', 'text_feat', 'action', 'distance_to_target']
    missing_keys = [k for k in required_keys if k not in sample]
    if missing_keys:
        print(f'ERROR: Missing required keys: {missing_keys}')
        return False

    # Check semantic keys
    semantic_keys = ['semantic_label_1', 'semantic_label_2', 'semantic_label_4']
    for k in semantic_keys:
        if k in sample:
            v = sample[k]
            print(f'\n{k}:')
            print(f'  shape: {v.shape}')
            print(f'  dtype: {v.dtype}')
            print(f'  min: {v.min().item()}, max: {v.max().item()}')
            print(f'  unique labels: {torch.unique(v).tolist()}')
        else:
            print(f'WARNING: {k} not found in sample')

    # Verify label values are in valid range (0-5)
    if 'semantic_label_1' in sample:
        labels = sample['semantic_label_1']
        valid_mask = (labels >= 0) & (labels <= 5)
        if not torch.all(valid_mask):
            print('ERROR: Invalid label values found!')
            return False

    print('\n✓ Dataset test passed!')
    return True


def count_labels():
    '''Count how many semantic labels exist in the dataset.'''
    dataset_path = '/shared_disk/users/wenhao.lu/JYT/WorldSearch_data/BrushifyUrban_test'

    print('\nCounting semantic labels...')

    total_steps = 0
    labeled_steps = 0

    episodes = [d for d in sorted(os.listdir(dataset_path))
                if d.startswith('episode_') and os.path.isdir(os.path.join(dataset_path, d))]

    for ep in episodes:
        ep_dir = os.path.join(dataset_path, ep)
        steps = [d for d in sorted(os.listdir(ep_dir))
                 if d.startswith('step_') and os.path.isdir(os.path.join(ep_dir, d))]
        for step in steps:
            total_steps += 1
            semantic_path = os.path.join(ep_dir, step, 'semantic_front.npy')
            if os.path.exists(semantic_path):
                labeled_steps += 1

    print(f'Total steps: {total_steps}')
    print(f'With semantic labels: {labeled_steps}')
    print(f'Percentage: {100 * labeled_steps / total_steps:.1f}%')


if __name__ == '__main__':
    count_labels()
    print()
    success = test_dataset()
    sys.exit(0 if success else 1)
