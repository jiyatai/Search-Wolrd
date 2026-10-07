# Semantic Label Generation and Usage Guide

This guide describes how to generate pseudo semantic labels for the UAV dataset and use them for training.

## Semantic Classes (6 classes)

| Label | Class Name | Description |
|-------|------------|-------------|
| 0     | SKY        | Sky regions (detected via color heuristics) |
| 1     | GROUND     | Ground/navigable surface (default label) |
| 2     | BUILDING   | Buildings, walls, towers |
| 3     | HUMAN      | People, humans |
| 4     | VEHICLE    | Cars, trucks, other vehicles |
| 5     | OBSTACLE   | Playground equipment, statues, trees, etc. |

## Step 1: Generate Semantic Labels

Run the label generation script:

```bash
cd /mnt/pfs/users/luwenhao/code_jyt/SearchWorld

# Install dependencies if needed
pip install transformers torch torchvision pillow numpy tqdm accelerate

# Generate labels for all episodes
python scripts/data_pipeline/generate_pseudo_semantic_labels_dino.py \
    --dataset_root /shared_disk/users/wenhao.lu/JYT/WorldSearch_data/BrushifyUrban_test

# Optional: Overwrite existing labels
python scripts/data_pipeline/generate_pseudo_semantic_labels_dino.py \
    --dataset_root /shared_disk/users/wenhao.lu/JYT/WorldSearch_data/BrushifyUrban_test \
    --overwrite

# Optional: Process a subset of episodes
python scripts/data_pipeline/generate_pseudo_semantic_labels_dino.py \
    --dataset_root /shared_disk/users/wenhao.lu/JYT/WorldSearch_data/BrushifyUrban_test \
    --start_episode 0 \
    --end_episode 10
```

## Step 2: Verify Generated Labels

Visualize the generated labels to verify quality:

```bash
# Visualize a single episode
python scripts/data_pipeline/visualize_semantic_labels.py \
    --dataset_root /shared_disk/users/wenhao.lu/JYT/WorldSearch_data/BrushifyUrban_test \
    --episode episode_0001 \
    --output_dir /tmp/semantic_vis

# Visualize all episodes
python scripts/data_pipeline/visualize_semantic_labels.py \
    --dataset_root /shared_disk/users/wenhao.lu/JYT/WorldSearch_data/BrushifyUrban_test \
    --all_episodes \
    --output_dir /tmp/semantic_vis
```

## Step 3: Train with Semantic Labels

Use the UAV config for training with semantic labels:

```bash
python train.py \
    --config-files configs/uav_train_config.gin \
    --dataset-path /shared_disk/users/wenhao.lu/JYT/WorldSearch_data/BrushifyUrban_test \
    --output-dir /path/to/output/dir \
    --wandb-entity-name your_entity \
    --wandb-project-name searchworld_uav
```

## Dataset Structure

After generating labels, your dataset will have:

```
BrushifyUrban_test/
├── episode_0001/
│   ├── episode_summary.json
│   ├── step_0000/
│   │   ├── rgb_front.png
│   │   ├── depth_front.npy
│   │   ├── semantic_front.npy    <-- Generated semantic label (uint8, hxw)
│   │   └── state.json
│   ├── step_0001/
│   └── ...
└── ...
```

## Configuration Details

The UAV config (`configs/uav_train_config.gin`) enables:
- `ENABLE_SEMANTIC=True`: Enable semantic segmentation decoder
- `ENABLE_RGB_STYLEGAN=False`: Disable RGB StyleGAN decoder
- `ENABLE_RGB_DIFFUSION=False`: Disable RGB diffusion decoder
- `NUM_SEMANTIC_CLASSES=6`: 6 semantic classes

## Files Modified/Added

| File | Description |
|------|-------------|
| `scripts/data_pipeline/generate_pseudo_semantic_labels_dino.py` | Label generation script |
| `scripts/data_pipeline/visualize_semantic_labels.py` | Visualization script |
| `scripts/README_SEMANTIC_LABELS.md` | This document |
| `model/dataset/uav_dataset.py` | Updated UAV dataset with semantic support |
| `model/dataset/__init__.py` | Updated imports |
| `configs/uav_train_config.gin` | UAV training config |
