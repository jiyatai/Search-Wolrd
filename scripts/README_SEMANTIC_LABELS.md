# Improved Semantic Label Generation
==================================

## Problem with Original Implementation

The original `scripts/data_pipeline/generate_pseudo_semantic_labels_dino.py` uses **bounding boxes only**, which causes:
- Blocky, imprecise segmentation
- Labels that don't follow object boundaries
- Poor visual quality for training

## Improved Versions

We provide **two improved versions**:

### Version 1: `scripts/data_pipeline/generate_pseudo_semantic_labels_improved.py` (SAM Version)
- Uses **Grounding DINO + SAM (Segment Anything Model)**
- Pixel-precise segmentation
- Better sky/ground detection
- Depth-aware refinement
- **Best quality but requires SAM/MobileSAM installation**

### Version 2: `scripts/data_pipeline/generate_pseudo_semantic_labels_v2.py` (Lightweight Version)
- Uses **Grounding DINO + GrabCut/color-based refinement**
- No SAM dependency - easier to install
- Faster runtime
- Still much better than original

## Installation

### Common Dependencies (for both versions)
```bash
pip install transformers torch torchvision pillow numpy tqdm accelerate scipy
```

### For V2 (Lightweight, recommended first)
```bash
# Optional but recommended for GrabCut
pip install opencv-python
```

### For Improved (SAM Version)
```bash
# Option A: MobileSAM (faster, recommended)
pip install git+https://github.com/ChaoningZhang/MobileSAM.git

# Option B: Original SAM
pip install git+https://github.com/facebookresearch/segment-anything.git
```

The script will automatically download model weights on first use.

## Usage

### Quick Start - Try V2 First (Lightweight)

```bash
# Generate labels for first few episodes with V2
python scripts/data_pipeline/generate_pseudo_semantic_labels_v2.py \
    --dataset_root /path/to/dataset \
    --start_episode 0 \
    --end_episode 2 \
    --save_vis \
    --overwrite
```

### Full Usage - V2 (Lightweight)
```bash
python scripts/data_pipeline/generate_pseudo_semantic_labels_v2.py \
    --dataset_root /path/to/dataset \
    --model IDEA-Research/grounding-dino-base \
    --output_size 320 512 \
    --overwrite \
    --save_vis
```

Options:
- `--no_grabcut`: Disable GrabCut (faster, but less accurate boundaries)
- `--no_depth`: Don't use depth information
- `--save_vis`: Save color visualization
- `--overwrite`: Overwrite existing labels

### Full Usage - Improved (SAM Version)
```bash
python scripts/data_pipeline/generate_pseudo_semantic_labels_improved.py \
    --dataset_root /path/to/dataset \
    --sam_model mobile \
    --overwrite \
    --save_vis
```

Options:
- `--sam_model`: Choose 'mobile' (MobileSAM, faster) or 'sam' (original SAM)
- `--no_sam`: Disable SAM entirely (use boxes only)
- `--dino_model`: Grounding DINO model to use

## Compare Results

Use the comparison tool to see differences:

```bash
python compare_semantic_labels.py \
    --dataset_root /path/to/dataset \
    --output_dir /tmp/semantic_comparison
```

This creates side-by-side comparisons of:
- Original RGB image
- Original labels (bounding boxes)
- New labels (with SAM/GrabCut refinement)
- Overlay on RGB

## What's Improved

| Feature | Original | V2 | Improved (SAM) |
|---------|---------|----|---------------|
| Segmentation | Bounding boxes only | GrabCut/color-based | SAM pixel-level |
| Sky detection | Simple color heuristic | Multi-criteria + morphological | Multi-criteria + morphological |
| Ground detection | None | Depth + color | Depth + color |
| NMS | No | Yes | Yes |
| Post-processing | None | Edge-aware smoothing | Edge-aware smoothing |
| Depth-aware | No | Yes | Yes |
| Model | Grounding DINO tiny | Grounding DINO base | Grounding DINO base |

## Labeling Strategy

1. **Sky Detection**:
   - Blue channel dominance
   - Brightness/saturation checks
   - Sky color similarity
   - Morphological cleanup (keep large regions only)
   - Depth confirmation (sky should be far away)

2. **Object Detection**:
   - Grounding DINO with improved text queries
   - NMS to remove overlapping boxes
   - Higher thresholds for better precision

3. **Mask Refinement**:
   - V2: GrabCut (OpenCV) + color similarity
   - Improved: SAM (best)
   - Priority order: Human > Vehicle > Obstacle > Building

4. **Final Polish**:
   - Edge-aware smoothing
   - Depth discontinuity preservation
   - Ensure sky stays in upper region

## Performance Notes

- **V2**: ~1-2 FPS on GPU
- **Improved (SAM)**: ~0.3-0.5 FPS on GPU
- Both are much slower than original, but produce much higher quality labels

Tip: Generate on a small subset first to verify quality!

## Example Workflow

```bash
# Step 1: Test on a single episode
python scripts/data_pipeline/generate_pseudo_semantic_labels_v2.py \
    --dataset_root /path/to/dataset \
    --start_episode 0 \
    --end_episode 1 \
    --save_vis \
    --overwrite

# Step 2: Compare
python compare_semantic_labels.py \
    --dataset_root /path/to/dataset \
    --output_dir ./comparison_results

# Step 3: If satisfied, run on full dataset
python scripts/data_pipeline/generate_pseudo_semantic_labels_v2.py \
    --dataset_root /path/to/dataset \
    --overwrite
```

## Troubleshooting

### SAM won't download
If MobileSAM/SAM fails to download automatically:
- Download manually from their GitHub repos
- Place checkpoint in working directory

### Out of memory
- Use V2 instead of SAM version
- Reduce batch size (not applicable here, single image processing)
- Use smaller model: `--model IDEA-Research/grounding-dino-tiny`

### OpenCV not found
Install with:
```bash
pip install opencv-python
```
Or run with `--no_grabcut`

### Scipy not found
Install with:
```bash
pip install scipy
```
