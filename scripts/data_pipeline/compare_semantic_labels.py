# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Script to compare different semantic label generation methods side by side

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


def create_legend_image(height=200):
    """Create a legend image."""
    legend_width = 180
    legend = np.ones((height, legend_width, 3), dtype=np.uint8) * 255

    img_legend = Image.fromarray(legend)
    draw = ImageDraw.Draw(img_legend)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except:
        font = ImageFont.load_default()

    box_size = 20
    spacing = 30
    start_y = 15

    for i, (name, color) in enumerate(zip(SEMANTIC_NAMES, SEMANTIC_COLORS)):
        y = start_y + i * spacing
        # Draw color box
        draw.rectangle([10, y, 10 + box_size, y + box_size],
                      fill=tuple(color), outline=(0, 0, 0))
        # Draw text
        draw.text((40, y + 2), name, fill=(0, 0, 0), font=font)

    return np.array(img_legend)


def compute_label_stats(labels: np.ndarray) -> dict:
    """Compute statistics for a label map."""
    total = labels.size
    stats = {}
    for i, name in enumerate(SEMANTIC_NAMES):
        count = np.sum(labels == i)
        stats[name] = {
            'count': count,
            'percent': 100 * count / total
        }
    return stats


def list_episodes(dataset_root: str):
    episodes = []
    for d in sorted(os.listdir(dataset_root)):
        if d.startswith("episode_") and os.path.isdir(os.path.join(dataset_root, d)):
            episodes.append(d)
    return episodes


def list_step_dirs(episode_dir: str):
    steps = []
    for d in sorted(os.listdir(episode_dir)):
        if d.startswith("step_") and os.path.isdir(os.path.join(episode_dir, d)):
            steps.append(d)
    return steps


def compare_single_image(
    img_path: str,
    label_path: str,
    output_path: str,
    label_v2_path: str = None,
    label_improved_path: str = None
):
    """Compare a single image's labels."""
    # Load RGB
    rgb_pil = Image.open(img_path).convert("RGB")
    rgb = np.array(rgb_pil)
    h, w = rgb.shape[:2]

    # Load labels
    labels = {}
    if os.path.exists(label_path):
        labels['Original'] = np.load(label_path)
    if label_v2_path and os.path.exists(label_v2_path):
        labels['V2'] = np.load(label_v2_path)
    elif label_path.replace('semantic_front', 'semantic_front_v2'):
        p = label_path.replace('semantic_front', 'semantic_front_v2')
        if os.path.exists(p):
            labels['V2'] = np.load(p)
    if label_improved_path and os.path.exists(label_improved_path):
        labels['Improved'] = np.load(label_improved_path)

    if not labels:
        print(f"No labels found for {img_path}")
        return

    # Create colored labels
    colored = {}
    for name, label in labels.items():
        colored[name] = apply_color_map(label)
        # Resize to match RGB if needed
        if colored[name].shape[:2] != (h, w):
            colored_pil = Image.fromarray(colored[name])
            colored_pil = colored_pil.resize((w, h), Image.NEAREST)
            colored[name] = np.array(colored_pil)

    # Create overlay for each
    overlays = {}
    for name, c in colored.items():
        overlays[name] = (0.6 * rgb + 0.4 * c).astype(np.uint8)

    # Build output image
    num_labels = len(labels)
    # Layout: RGB | (Label1 | Overlay1) | (Label2 | Overlay2) ... | Legend
    cell_width = w
    panel_width = w * 2  # Label + Overlay per method
    legend = create_legend_image(h)
    total_width = cell_width + num_labels * panel_width + legend.shape[1] + 20 * (num_labels + 1)

    output = np.ones((h, total_width, 3), dtype=np.uint8) * 240

    # Place RGB
    x = 10
    output[:, x:x+w] = rgb
    x += w + 10

    # Place each method
    for name in labels.keys():
        output[:, x:x+w] = colored[name]
        x += w
        output[:, x:x+w] = overlays[name]
        x += w + 10

    # Place legend
    output[:, -legend.shape[1]:] = legend

    # Add titles
    output_pil = Image.fromarray(output)
    draw = ImageDraw.Draw(output_pil)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
        title_font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
        title_font = ImageFont.load_default()

    # Draw titles with background
    def draw_text_with_bg(draw, pos, text, font, bg_color=(0, 0, 0), text_color=(255, 255, 255)):
        x, y = pos
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.rectangle([x-5, y-2, x+text_w+5, y+text_h+4], fill=bg_color)
        draw.text((x, y), text, fill=text_color, font=font)

    y_text = 10
    x = 10 + w // 2 - 30
    draw_text_with_bg(draw, (x, y_text), "RGB", title_font)

    x = 10 + w + 10
    for name in labels.keys():
        # Label title
        draw_text_with_bg(draw, (x + w//2 - 40, y_text), f"{name}", title_font)
        # Overlay title
        draw_text_with_bg(draw, (x + w + w//2 - 40, y_text), "Overlay", title_font)
        x += panel_width + 10

    # Save
    output_pil.save(output_path)
    print(f"Saved comparison to {output_path}")

    return labels


def create_summary_grid(
    comparisons: list,
    output_dir: str,
    max_per_row: int = 3
):
    """Create a summary grid of all comparisons."""
    # Not implemented yet - placeholder
    pass


def main():
    parser = argparse.ArgumentParser(
        description="Compare semantic label generation methods"
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/shared_disk/users/wenhao.lu/JYT/WorldSearch_data/BrushifyUrban_test",
        help="Root directory with episode_* folders"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/tmp/semantic_comparison",
        help="Output directory"
    )
    parser.add_argument(
        "--episode",
        type=str,
        default=None,
        help="Specific episode to visualize"
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=3,
        help="Max episodes to process"
    )
    parser.add_argument(
        "--max_steps_per_episode",
        type=int,
        default=5,
        help="Max steps per episode"
    )
    parser.add_argument(
        "--label_suffix",
        type=str,
        default=None,
        help="Suffix for alternative labels (e.g., '_v2' for semantic_front_v2.npy)"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    episodes = list_episodes(args.dataset_root)
    if args.episode:
        episodes = [args.episode]
    else:
        episodes = episodes[:args.max_episodes]

    print(f"Processing {len(episodes)} episodes")

    all_comparisons = []

    for ep in episodes:
        episode_dir = os.path.join(args.dataset_root, ep)
        steps = list_step_dirs(episode_dir)
        steps = steps[:args.max_steps_per_episode]

        ep_output_dir = os.path.join(args.output_dir, ep)
        os.makedirs(ep_output_dir, exist_ok=True)

        print(f"Processing {ep}, {len(steps)} steps")

        for step in steps:
            step_dir = os.path.join(episode_dir, step)
            img_path = os.path.join(step_dir, "rgb_front.png")
            label_path = os.path.join(step_dir, "semantic_front.npy")

            if not os.path.exists(img_path):
                continue

            output_path = os.path.join(ep_output_dir, f"{step}_comparison.png")

            # Look for alternative labels
            label_v2_path = os.path.join(step_dir, "semantic_front_v2.npy")
            label_improved_path = os.path.join(step_dir, "semantic_front_improved.npy")

            labels = compare_single_image(
                img_path,
                label_path,
                output_path,
                label_v2_path if os.path.exists(label_v2_path) else None,
                label_improved_path if os.path.exists(label_improved_path) else None
            )

            if labels:
                all_comparisons.append((img_path, labels))

    print(f"\nDone! Comparisons saved to {args.output_dir}")

    # Print summary stats
    if all_comparisons:
        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)

        # Aggregate stats across all images
        agg_stats = {}
        for img_path, labels in all_comparisons:
            for method, label_map in labels.items():
                if method not in agg_stats:
                    agg_stats[method] = {}
                    for name in SEMANTIC_NAMES:
                        agg_stats[method][name] = []
                stats = compute_label_stats(label_map)
                for name in SEMANTIC_NAMES:
                    agg_stats[method][name].append(stats[name]['percent'])

        # Print
        for method in agg_stats.keys():
            print(f"\n{method}:")
            for name in SEMANTIC_NAMES:
                avg = np.mean(agg_stats[method][name])
                print(f"  {name:10s}: {avg:5.2f}%")


if __name__ == "__main__":
    main()
