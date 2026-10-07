# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

'''Simple script to check semantic labels (minimal dependencies).'''

import argparse
import os
import json


def load_npy(path):
    """Simple numpy loader with fallback for pickle."""
    try:
        import numpy as np
        return np.load(path)
    except ImportError:
        print("numpy not found, trying alternative...")
        return None


def check_labels(dataset_root, output_dir, num_episodes=5, num_steps=5):
    """Check semantic labels."""
    os.makedirs(output_dir, exist_ok=True)

    episodes = []
    for d in sorted(os.listdir(dataset_root)):
        if d.startswith("episode_") and os.path.isdir(os.path.join(dataset_root, d)):
            episodes.append(d)

    episodes = episodes[:num_episodes]
    print(f"Checking {len(episodes)} episodes...")
    print()

    try:
        import numpy as np
        has_numpy = True
    except ImportError:
        has_numpy = False
        print("numpy not available - doing basic check only")
        print()

    stats = {
        'total_files': 0,
        'episodes_checked': 0,
        'steps_checked': 0,
    }

    label_counts = {}

    for ep in episodes:
        episode_dir = os.path.join(dataset_root, ep)
        steps = []
        for d in sorted(os.listdir(episode_dir)):
            if d.startswith("step_") and os.path.isdir(os.path.join(episode_dir, d)):
                steps.append(d)
        steps = steps[:num_steps]

        print(f"Episode {ep}: {len(steps)} steps")
        stats['episodes_checked'] += 1

        for step in steps:
            step_dir = os.path.join(episode_dir, step)
            semantic_path = os.path.join(step_dir, "semantic_front.npy")

            if os.path.exists(semantic_path):
                stats['total_files'] += 1
                stats['steps_checked'] += 1

                if has_numpy:
                    try:
                        semantic = np.load(semantic_path)
                        print(f"  {step}: shape={semantic.shape}, dtype={semantic.dtype}, "
                              f"labels={np.unique(semantic).tolist()}")

                        for label in np.unique(semantic):
                            label_counts[label] = label_counts.get(label, 0) + np.sum(semantic == label)
                    except Exception as e:
                        print(f"  {step}: ERROR loading - {e}")
                else:
                    print(f"  {step}: OK (file exists)")
            else:
                print(f"  {step}: MISSING")

        print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Episodes checked: {stats['episodes_checked']}")
    print(f"Steps checked: {stats['steps_checked']}")
    print()

    if has_numpy and label_counts:
        label_names = ['Sky', 'Ground', 'Building', 'Human', 'Vehicle', 'Obstacle']
        total = sum(label_counts.values())
        print("Label distribution:")
        for label in sorted(label_counts.keys()):
            name = label_names[label] if label < len(label_names) else f'Unknown-{label}'
            percent = 100 * label_counts[label] / total
            print(f"  {name:10s} ({label}): {percent:5.2f}%")

    # Save summary
    summary_file = os.path.join(output_dir, "check_summary.json")
    with open(summary_file, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nSummary saved to {summary_file}")


def main():
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
        "--num_episodes",
        type=int,
        default=5,
        help="Number of episodes to check"
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=5,
        help="Number of steps per episode to check"
    )
    args = parser.parse_args()

    check_labels(args.dataset_root, args.output_dir, args.num_episodes, args.num_steps)


if __name__ == "__main__":
    main()
