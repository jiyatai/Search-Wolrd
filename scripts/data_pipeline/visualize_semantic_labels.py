# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

'''Script to visualize semantic labels alongside RGB images (no GUI required).'''

import argparse
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont


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


SEMANTIC_NAMES = [
    'Sky',
    'Ground',
    'Building',
    'Human',
    'Vehicle',
    'Obstacle',
]


def apply_color_map(semantic_label):
    """Apply color map to semantic label."""
    h, w = semantic_label.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for label in range(6):
        mask = semantic_label == label
        colored[mask] = SEMANTIC_COLORS[label]
    return colored


def create_legend_image():
    """Create a legend image as numpy array."""
    legend_height = 30 * len(SEMANTIC_NAMES) + 20
    legend_width = 200
    legend = np.ones((legend_height, legend_width, 3), dtype=np.uint8) * 255

    for i, (name, color) in enumerate(zip(SEMANTIC_NAMES, SEMANTIC_COLORS)):
        y = 10 + i * 30
        # Draw color box
        legend[y:y+20, 10:30] = color
        # Draw text (simple, no font dependency)
        # We'll create a PIL image for text

    # Use PIL for proper text rendering
    img_legend = Image.fromarray(legend)
    draw = ImageDraw.Draw(img_legend)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except:
        font = ImageFont.load_default()

    for i, (name, color) in enumerate(zip(SEMANTIC_NAMES, SEMANTIC_COLORS)):
        y = 10 + i * 30
        draw.rectangle([10, y, 30, y+20], fill=tuple(color), outline=(0,0,0))
        draw.text((40, y+2), name, fill=(0,0,0), font=font)

    return np.array(img_legend)


def visualize_episode(dataset_root, episode_name, output_dir, max_steps=10):
    """Visualize semantic labels for an episode and save to files."""
    episode_dir = os.path.join(dataset_root, episode_name)

    # Collect steps
    steps = []
    for d in sorted(os.listdir(episode_dir)):
        if d.startswith("step_") and os.path.isdir(os.path.join(episode_dir, d)):
            steps.append(d)

    steps = steps[:max_steps]

    # Create output dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"Visualizing {episode_name}, {len(steps)} steps...")

    # Get legend
    legend = create_legend_image()

    # Visualize each step
    for step in steps:
        step_dir = os.path.join(episode_dir, step)
        img_path = os.path.join(step_dir, "rgb_front.png")
        semantic_path = os.path.join(step_dir, "semantic_front.npy")

        if not os.path.exists(img_path):
            continue

        # Load RGB
        rgb_pil = Image.open(img_path).convert("RGB")
        rgb = np.array(rgb_pil)

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
            print(f"  Warning: No semantic label for {step}")
            continue

        # Create combined image
        h, w = rgb.shape[:2]
        combined = np.ones((h, w * 3 + legend.shape[1] + 20, 3), dtype=np.uint8) * 255

        # Place images
        combined[:, :w] = rgb
        combined[:, w+5:w*2+5] = colored_semantic
        combined[:, w*2+10:w*3+10] = overlay

        # Place legend
        legend_h = legend.shape[0]
        if legend_h <= h:
            combined[:legend_h, w*3+15:w*3+15+legend.shape[1]] = legend

        # Add titles at top
        combined_pil = Image.fromarray(combined)
        draw = ImageDraw.Draw(combined_pil)
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()

        draw.text((w//2 - 30, 5), 'RGB', fill=(255,255,255), font=font, stroke_fill=(0,0,0), stroke_width=2)
        draw.text((w + 5 + w//2 - 50, 5), 'Semantic Label', fill=(255,255,255), font=font, stroke_fill=(0,0,0), stroke_width=2)
        draw.text((w*2 + 10 + w//2 - 30, 5), 'Overlay', fill=(255,255,255), font=font, stroke_fill=(0,0,0), stroke_width=2)

        # Save
        output_path = os.path.join(output_dir, f"{episode_name}_{step}.png")
        combined_pil.save(output_path)

    print(f"  Saved to {output_dir}")

    # Also create a single grid image with all steps
    if len(steps) > 0:
        create_grid_image(dataset_root, episode_name, steps, output_dir, legend)


def create_grid_image(dataset_root, episode_name, steps, output_dir, legend):
    """Create a grid image of all steps."""
    from math import ceil, sqrt

    n = len(steps)
    cols = min(5, ceil(sqrt(n)))
    rows = ceil(n / cols)

    # Load first image to get size
    first_step = steps[0]
    img_path = os.path.join(dataset_root, episode_name, first_step, "rgb_front.png")
    rgb_pil = Image.open(img_path).convert("RGB")
    w, h = rgb_pil.size

    # Grid dimensions (each cell has RGB + semantic side by side)
    cell_w = w * 2 + 10
    cell_h = h + 25
    grid_w = cols * cell_w + 10
    grid_h = rows * cell_h + 10

    grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 240

    for i, step in enumerate(steps):
        row = i // cols
        col = i % cols

        step_dir = os.path.join(dataset_root, episode_name, step)
        img_path = os.path.join(step_dir, "rgb_front.png")
        semantic_path = os.path.join(step_dir, "semantic_front.npy")

        rgb_pil = Image.open(img_path).convert("RGB")
        rgb = np.array(rgb_pil)

        if os.path.exists(semantic_path):
            semantic = np.load(semantic_path)
            colored_semantic = apply_color_map(semantic)
            colored_pil = Image.fromarray(colored_semantic)
            colored_pil = colored_pil.resize(rgb_pil.size, Image.NEAREST)
            colored_semantic = np.array(colored_pil)

        # Position
        y0 = 10 + row * cell_h
        x0 = 10 + col * cell_w

        # Place RGB
        grid[y0+20:y0+20+h, x0:x0+w] = rgb
        # Place semantic
        grid[y0+20:y0+20+h, x0+w+5:x0+w+5+w] = colored_semantic

        # Add step label
        grid_pil = Image.fromarray(grid)
        draw = ImageDraw.Draw(grid_pil)
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except:
            font = ImageFont.load_default()
        draw.text((x0, y0), step, fill=(0,0,0), font=font)
        grid = np.array(grid_pil)

    # Save grid
    grid_pil = Image.fromarray(grid)
    grid_output = os.path.join(output_dir, f"{episode_name}_grid.png")
    grid_pil.save(grid_output)
    print(f"  Grid saved to {grid_output}")


def compute_statistics(dataset_root, episodes, output_dir):
    """Compute and save label statistics."""
    from collections import defaultdict

    print("\nComputing label statistics...")

    label_counts = defaultdict(int)
    total_pixels = 0

    for ep in episodes:
        episode_dir = os.path.join(dataset_root, ep)
        steps = [d for d in sorted(os.listdir(episode_dir))
                 if d.startswith("step_") and os.path.isdir(os.path.join(episode_dir, d))]

        for step in steps[:1]:  # Just sample first step of each episode for stats
            semantic_path = os.path.join(episode_dir, step, "semantic_front.npy")
            if os.path.exists(semantic_path):
                semantic = np.load(semantic_path)
                for label in range(6):
                    label_counts[label] += np.sum(semantic == label)
                total_pixels += semantic.size

    if total_pixels > 0:
        stats_file = os.path.join(output_dir, "label_statistics.txt")
        with open(stats_file, "w") as f:
            f.write("=" * 50 + "\n")
            f.write("Semantic Label Statistics\n")
            f.write("=" * 50 + "\n\n")

            for label in range(6):
                count = label_counts[label]
                percent = 100 * count / total_pixels
                f.write(f"{SEMANTIC_NAMES[label]:10s}: {count:10d} pixels ({percent:5.2f}%)\n")

            f.write(f"\nTotal: {total_pixels} pixels\n")

        print(f"Statistics saved to {stats_file}")
        print("\nLabel distribution:")
        for label in range(6):
            count = label_counts[label]
            percent = 100 * count / total_pixels
            print(f"  {SEMANTIC_NAMES[label]:10s}: {percent:5.2f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/shared_disk/users/wenhao.lu/JYT/WorldSearch_data/BrushifyUrban_test",
        help="Root directory with episode_* folders"
    )
    parser.add_argument(
        "--episode",
        type=str,
        default=None,
        help="Episode name to visualize (default: first 3 episodes)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/tmp/semantic_vis",
        help="Output directory to save visualizations"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=10,
        help="Maximum number of steps per episode to visualize"
    )
    parser.add_argument(
        "--all_episodes",
        action="store_true",
        help="Visualize all episodes (warning: slow)"
    )
    args = parser.parse_args()

    # Get episodes
    all_episodes = []
    for d in sorted(os.listdir(args.dataset_root)):
        if d.startswith("episode_") and os.path.isdir(os.path.join(args.dataset_root, d)):
            all_episodes.append(d)

    if args.all_episodes:
        episodes = all_episodes
    elif args.episode:
        episodes = [args.episode]
    else:
        episodes = all_episodes[:3]  # First 3 episodes by default

    print("=" * 60)
    print("Semantic Label Visualizer")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    print(f"Episodes to visualize: {len(episodes)}")
    print()

    # Visualize each episode
    for ep in episodes:
        visualize_episode(args.dataset_root, ep, args.output_dir, args.max_steps)

    # Compute statistics
    compute_statistics(args.dataset_root, all_episodes[:20], args.output_dir)

    print()
    print("=" * 60)
    print(f"Done! Check {args.output_dir} for results.")
    print("=" * 60)


if __name__ == "__main__":
    main()
