# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Smoke test for UAVDataset.  Run from the project root with the right
# Python environment (e.g. the conda env that has torch / transformers
# / pytorch-lightning installed):
#
#     python scripts/smoke_test_uav_dataset.py
#
# Exits 0 on success, raises AssertionError with diagnostic message on
# any problem.  Adjust DATA_PATH below if your data lives elsewhere.

import os
import sys

# Make the project root importable when run as a plain script.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch

from model.dataset.uav_dataset import (IDX_TO_ACTION, NUM_DISCRETE_ACTIONS,
                                       UAVDataset)

DATA_PATH = '/shared_disk/users/wenhao.lu/JYT/WorldSearch_data/BrushifyUrban_test'
SEQUENCE_LENGTH = 4


def _header(text: str) -> None:
    print()
    print('=' * 60)
    print(text)
    print('=' * 60)


def main() -> int:
    _header(f'Loading UAVDataset from {DATA_PATH}')
    ds = UAVDataset(
        dataset_path=DATA_PATH,
        sequence_length=SEQUENCE_LENGTH,
        split='train',
    )
    print(f'Train split size: {len(ds)} sequences')
    print(f'Text-feat dim: {ds.text_feat_dim}')
    print(f'Num discrete actions: {NUM_DISCRETE_ACTIONS}')
    print(f'Episodes in split: {len(ds.episodes)} '
          f'(first 3: {ds.episodes[:3]})')

    # --- 1) Index 0 (start of first train episode) -------------------
    _header('Sample 0: first sequence of first train episode')
    s0 = ds[0]
    for k, v in s0.items():
        if isinstance(v, torch.Tensor):
            print(f'  {k:22s} shape={tuple(v.shape)}  dtype={v.dtype}')
        else:
            print(f'  {k:22s} type={type(v).__name__}')

    # shape assertions
    assert s0['image'].shape == (SEQUENCE_LENGTH, 3, 512, 512), \
        f"image shape wrong: {s0['image'].shape}"
    assert s0['relative_pose'].shape == (SEQUENCE_LENGTH, 4)
    assert s0['text_feat'].shape == (SEQUENCE_LENGTH, ds.text_feat_dim)
    assert s0['action'].shape == (SEQUENCE_LENGTH,)
    assert s0['distance_to_target'].shape == (SEQUENCE_LENGTH, 1)
    assert s0['action'].dtype == torch.int64
    assert s0['image'].dtype == torch.float32
    assert s0['relative_pose'].dtype == torch.float32

    # relative_pose at seq start should be ~0 (start of episode)
    rp0 = s0['relative_pose'][0]
    print(f'\nrelative_pose[0] (should be near 0): {rp0.tolist()}')
    assert torch.allclose(rp0, torch.zeros(4), atol=2.0), \
        f'relative_pose[0] should be near zero, got {rp0}'

    # text_feat should be L2-normalised (SigLIP text_embeds is normalised)
    norm = s0['text_feat'][0].norm().item()
    print(f'text_feat[0] L2 norm: {norm:.4f} (expect ~1.0)')
    assert abs(norm - 1.0) < 0.05, f'text_feat not L2-normalised: {norm}'

    # image value range
    img_min, img_max = s0['image'].min().item(), s0['image'].max().item()
    print(f'image value range: [{img_min:.3f}, {img_max:.3f}]')
    assert 0.0 <= img_min and img_max <= 1.0, \
        f'image not in [0, 1]: [{img_min}, {img_max}]'

    # action labels
    actions = s0['action'].tolist()
    action_names = [IDX_TO_ACTION[a] for a in actions]
    print(f'action labels (idx): {actions}')
    print(f'action labels (str): {action_names}')

    # distance to target over the sequence
    dist = s0['distance_to_target'].squeeze(-1).tolist()
    print(f'distance_to_target per step: {dist}')

    # --- 2) Different sample should have different text_feat ---------
    _header('Sample 1 vs Sample 0: text_feat should differ for different '
            'episodes')
    s1 = ds[1]
    same_text = torch.allclose(s0['text_feat'][0], s1['text_feat'][0], atol=1e-4)
    print(f'  sample0 text_feat[0,:5]: {s0["text_feat"][0, :5].tolist()}')
    print(f'  sample1 text_feat[0,:5]: {s1["text_feat"][0, :5].tolist()}')
    if same_text:
        # Could happen if the RNG seed produces same-ep adjacent samples;
        # warn but don't fail.
        print('  WARNING: text features identical for sample 0 and 1 '
              '(may share the same episode).')
    else:
        print('  OK: text features differ.')

    # --- 3) Train/val/test splits disjoint ---------------------------
    _header('Split sizes (train/val/test)')
    train_ds = UAVDataset(DATA_PATH, SEQUENCE_LENGTH, split='train')
    val_ds = UAVDataset(DATA_PATH, SEQUENCE_LENGTH, split='val')
    test_ds = UAVDataset(DATA_PATH, SEQUENCE_LENGTH, split='test')
    print(f'  train: {len(train_ds)} seqs  episodes={len(train_ds.episodes)}')
    print(f'  val:   {len(val_ds)} seqs  episodes={len(val_ds.episodes)}')
    print(f'  test:  {len(test_ds)} seqs  episodes={len(test_ds.episodes)}')
    assert set(train_ds.episodes).isdisjoint(set(val_ds.episodes))
    assert set(train_ds.episodes).isdisjoint(set(test_ds.episodes))
    assert set(val_ds.episodes).isdisjoint(set(test_ds.episodes))

    _header('All smoke-test assertions passed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
