# Changes for Semantic Segmentation Support

## Summary

This document describes the changes made to enable semantic segmentation decoder and generate pseudo labels for the UAV dataset.

## Changes Made

### 1. `scripts/data_pipeline/generate_pseudo_semantic_labels_dino.py` (Modified)
- Updated semantic classes to 6 classes specific to UAV/BrushifyUrban dataset
- Improved sky detection using color heuristics
- Added vehicle, building, human, obstacle detection using Grounding DINO
- Updated to use INPUT_IMAGE_SIZE (320, 512)
- Added support for processing subsets of episodes

### 2. `scripts/data_pipeline/visualize_semantic_labels.py` (New)
- Visualization script to verify generated semantic labels
- Shows RGB, semantic label, and overlay side-by-side
- Includes color-coded legend

### 3. `scripts/data_pipeline/test_uav_dataset.py` (New)
- Test script to verify dataset loading with semantic labels

### 4. `scripts/data_pipeline/README_SEMANTIC.md` (New)
- Usage documentation for semantic label generation and training

### 5. `model/dataset/uav_dataset.py` (Modified)
- Added `enable_semantic` parameter to load semantic labels
- Added `_compose_semantic_labels()` to create multi-scale labels (1x, 2x, 4x)
- Added `interpolate_resize()` helper function
- Semantic labels are saved as `semantic_front.npy` in each step folder

### 6. `model/dataset/__init__.py` (Modified)
- Added exports for `UAVDataModule` and `UAVDataset`

### 7. `configs/uav_train_config.gin` (New)
- Configuration for UAV dataset training with semantic labels enabled

## Semantic Classes

| ID | Class | Description |
|----|-------|-------------|
| 0 | SKY | Sky regions |
| 1 | GROUND | Ground/navigable surface |
| 2 | BUILDING | Buildings, walls, structures |
| 3 | HUMAN | People |
| 4 | VEHICLE | Cars, trucks, etc. |
| 5 | OBSTACLE | Playground equipment, statues, trees, etc. |

## Next Steps

1. Generate semantic labels:
   ```bash
   conda activate dataset
   python scripts/generate_pseudo_semantic_labels_dino.py     --dataset_root /shared_disk/users/wenhao.lu/JYT/WorldSearch_data/BrushifyUrban_test     --device cuda
   ```

2. Verify labels:
   ```bash
   python scripts/visualize_semantic_labels.py --dataset_root /shared_disk/users/wenhao.lu/JYT/WorldSearch_data/BrushifyUrban_test --episode episode_0001 --output_dir /tmp/vis
   ```

3. Train model:
   ```bash
   python train.py --config-files configs/uav_train_config.gin --dataset-path /shared_disk/users/wenhao.lu/JYT/WorldSearch_data/BrushifyUrban_test --output-dir /path/to/output
   ```
