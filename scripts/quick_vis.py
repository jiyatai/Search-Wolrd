#!/usr/bin/env python
# Quick visualization script for semantic labels

import os
import numpy as np
from PIL import Image


# Semantic classes
class SemanticLabel:
    SKY = 0
    GROUND = 1
    BUILDING = 2
    HUMAN = 3
    VEHICLE = 4
    OBSTACLE = 5


# Color map for visualization (RGB)
SEMANTIC_COLORS = np.array([
    [135, 206, 235],   # Sky - light blue
    [34, 139, 34],     # Ground - forest green
    [139, 69, 19],     # Building - saddle brown
    [255, 0, 0],       # Human - red
    [0, 0, 255],       # Vehicle - blue
    [255, 165, 0],     # Obstacle - orange
], dtype=np.uint8)


def apply_color_map(semantic_label):
    """Apply color map to semantic label."""
    h, w = semantic_label.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for label in range(6):
        mask = semantic_label == label
        colored[mask] = SEMANTIC_COLORS[label]
    return colored


def visualize_episode(dataset_root, episode_name, output_dir, max_steps=5):
    """Visualize semantic labels for an episode."""
    episode_dir = os.path.join(dataset_root, episode_name)

    # Collect steps
    steps = []
    for d in sorted(os.listdir(episode_dir)):
        if d.startswith("step_") and os.path.isdir(os.path.join(episode_dir, d)):
            steps.append(d)

    steps = steps[:max_steps]

    # Create output dir
    os.makedirs(output_dir, exist_ok=True)

    print("Visualizing %s, %d steps..." % (episode_name, len(steps)))

    for step_idx, step in enumerate(steps):
        step_dir = os.path.join(episode_dir, step)
        img_path = os.path.join(step_dir, "rgb_front.png")
        semantic_path = os.path.join(step_dir, "semantic_front.npy")

        if not os.path.exists(img_path):
            continue

        # Load RGB
        rgb_pil = Image.open(img_path).convert("RGB")
        rgb = np.array(rgb_pil)
        h, w = rgb.shape[:2]

        # Load semantic if available
        has_semantic = os.path.exists(semantic_path)
        if has_semantic:
            semantic = np.load(semantic_path)
            colored_semantic = apply_color_map(semantic)
            colored_pil = Image.fromarray(colored_semantic)
            colored_pil = colored_pil.resize(rgb_pil.size, Image.NEAREST)
            colored_semantic = np.array(colored_pil)

            # Create overlay
            overlay = (0.6 * rgb + 0.4 * colored_semantic).astype(np.uint8)
        else:
            print("  Warning: No semantic label for %s" % step)
            continue

        # Create combined image
        combined = np.ones((h, w * 3 + 20, 3), dtype=np.uint8) * 255

        # Place images
        combined[:, :w] = rgb
        combined[:, w+5:w*2+5] = colored_semantic
        combined[:, w*2+10:w*3+10] = overlay

        # Save
        output_path = os.path.join(output_dir, "%s_%s.png" % (episode_name, step))
        Image.fromarray(combined).save(output_path)
        print("  Saved: %s" % output_path)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/shared_disk/users/wenhao.lu/JYT/WorldSearch_data/BrushifyUrban_test",
        help="Root directory with episode_* folders"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/tmp/semantic_vis",
        help="Output directory"
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=2,
        help="Max episodes"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=5,
        help="Max steps per episode"
    )
    args = parser.parse_args()

    # Get episodes
    episodes = []
    for d in sorted(os.listdir(args.dataset_root)):
        if d.startswith("episode_") and os.path.isdir(os.path.join(args.dataset_root, d)):
            episodes.append(d)
    episodes = episodes[:args.max_episodes]

    for ep in episodes:
        visualize_episode(args.dataset_root, ep, args.output_dir, args.max_steps)

    print("\nDone! Check %s for results." % args.output_dir)


if __name__ == "__main__":
    main()
