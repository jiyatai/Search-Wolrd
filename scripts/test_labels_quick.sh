#!/bin/bash
# Quick test script for semantic label generation

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_ROOT="/shared_disk/users/wenhao.lu/JYT/WorldSearch_data/BrushifyUrban_test"

echo "========================================="
echo "Semantic Label Generation - Quick Test"
echo "========================================="
echo ""
echo "Dataset: $DATASET_ROOT"
echo ""

# Check dataset exists
if [ ! -d "$DATASET_ROOT" ]; then
    echo "ERROR: Dataset not found at $DATASET_ROOT"
    exit 1
fi

echo "Available episodes:"
ls -la "$DATASET_ROOT" | grep "episode_" | head -10
echo ""

cat << 'EOF'
Quick Start Options:
====================

1. Test V2 (Lightweight, recommended first)
   -----------------------------------------
   # Install dependencies first
   pip install transformers torch torchvision pillow numpy tqdm accelerate scipy opencv-python

   # Run on first 2 episodes
   python generate_pseudo_semantic_labels_v2.py \
       --start_episode 0 \
       --end_episode 2 \
       --save_vis

2. Test Improved (SAM version)
   -----------------------------
   # Install SAM first
   pip install git+https://github.com/ChaoningZhang/MobileSAM.git

   # Run
   python generate_pseudo_semantic_labels_improved.py \
       --start_episode 0 \
       --end_episode 2 \
       --save_vis

3. Compare results
   -----------------
   python compare_semantic_labels.py \
       --output_dir /tmp/semantic_comparison

EOF

echo ""
echo "Checking existing semantic labels..."
first_ep=$(ls -d "$DATASET_ROOT"/episode_* 2>/dev/null | head -1)
if [ -n "$first_ep" ]; then
    first_step=$(ls -d "$first_ep"/step_* 2>/dev/null | head -1)
    if [ -n "$first_step" ]; then
        echo "  Found: $first_step"
        if [ -f "$first_step/semantic_front.npy" ]; then
            echo "  ✓ Has semantic_front.npy"
        else
            echo "  ✗ No semantic_front.npy"
        fi
        if [ -f "$first_step/rgb_front.png" ]; then
            echo "  ✓ Has rgb_front.png"
        fi
        if [ -f "$first_step/depth_front.npy" ]; then
            echo "  ✓ Has depth_front.npy"
        fi
    fi
fi

echo ""
echo "========================================="
echo "Ready! Use the commands above to test."
echo "========================================="
