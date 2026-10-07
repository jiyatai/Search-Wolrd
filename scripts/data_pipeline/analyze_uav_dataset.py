#!/usr/bin/env python3
"""Analyze UAV dataset: check heights, positions, etc."""

import json
from pathlib import Path
import numpy as np
from tqdm import tqdm


def main():
    dataset_path = Path("/shared_disk/users/wenhao.lu/JYT/WorldSearch_data/BrushifyUrban_test")

    heights = []
    z_values = []
    positions = []

    print(f"Checking dataset at {dataset_path}")

    episodes = sorted([d for d in dataset_path.iterdir() if d.name.startswith("episode_")])
    print(f"Found {len(episodes)} episodes")

    for episode_path in tqdm(episodes[:10], desc="Analyzing episodes"):
        summary_path = episode_path / "episode_summary.json"
        if not summary_path.exists():
            continue

        with open(summary_path, 'r') as f:
            summary = json.load(f)

        start_pos = summary["start_position"]
        # AirSim coordinate system: z is up-down, negative is up!
        start_height = -start_pos[2]  # z is negative when up
        heights.append(start_height)

        # Check each step
        step_folders = sorted([d for d in episode_path.iterdir() if d.name.startswith("step_")])
        for step_folder in step_folders:
            state_path = step_folder / "state.json"
            if not state_path.exists():
                continue

            with open(state_path, 'r') as f:
                state = json.load(f)

            pos = state["position"]
            z_value = pos[2]
            z_values.append(z_value)
            positions.append(pos)

    print()
    print("=" * 60)
    print("ANALYSIS RESULTS")
    print("=" * 60)
    print()

    heights = np.array(heights)
    print(f"Start heights (meters):")
    print(f"  Min: {heights.min():.2f}")
    print(f"  Max: {heights.max():.2f}")
    print(f"  Mean: {heights.mean():.2f}")
    print(f"  Median: {np.median(heights):.2f}")
    print()

    z_values = np.array(z_values)
    print(f"All z-values (raw, AirSim coordinate):")
    print(f"  Min: {z_values.min():.2f}")
    print(f"  Max: {z_values.max():.2f}")
    print(f"  Mean: {z_values.mean():.2f}")
    print(f"  Median: {np.median(z_values):.2f}")
    print()

    positions = np.array(positions)
    print(f"Position (x, y, z) ranges:")
    print(f"  X: [{positions[:,0].min():.2f}, {positions[:,0].max():.2f}]")
    print(f"  Y: [{positions[:,1].min():.2f}, {positions[:,1].max():.2f}]")
    print(f"  Z: [{positions[:,2].min():.2f}, {positions[:,2].max():.2f}]")
    print()

    print("=" * 60)
    print("Camera information from AirSim settings:")
    print("=" * 60)
    print("- Camera 0: Front (Pitch=0, X=1, Y=0, Z=0)")
    print("- Camera 3: Down (Pitch=-90, X=0, Y=0, Z=0)")
    print("- FOV: 90 degrees")
    print("- Image size: 512 x 512")
    print()


if __name__ == "__main__":
    main()
