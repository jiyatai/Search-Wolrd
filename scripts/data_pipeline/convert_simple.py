#!/usr/bin/env python3
"""Simple converter - no functions defined after use."""

import argparse
import io
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(
        description='Convert UAV dataset to Parquet format'
    )
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Path to UAV dataset root'
    )
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Path to output Parquet dataset root'
    )
    parser.add_argument(
        '--samples-per-file',
        type=int,
        default=100,
        help='Number of samples per parquet file'
    )
    parser.add_argument(
        '--task-description',
        default='Search for the target object',
        help='Task description for text encoder'
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    # Find all episodes
    episodes = sorted([
        f for f in input_path.iterdir()
        if f.name.startswith('episode_') and f.is_dir()
    ])

    if not episodes:
        print(f"No episodes found in {input_path}")
        return

    print(f"Found {len(episodes)} episodes")

    # Split into train/val/test
    n = len(episodes)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    splits = {
        'train': episodes[:n_train],
        'val': episodes[n_train:n_train + n_val],
        'test': episodes[n_train + n_val:],
    }

    for split_name, split_episodes in splits.items():
        if not split_episodes:
            continue

        split_output_path = output_path / split_name
        split_output_path.mkdir(parents=True, exist_ok=True)

        print(f"\nConverting {split_name} split ({len(split_episodes)} episodes)")

        for episode_path in tqdm(split_episodes, desc=split_name):
            output_episode_path = split_output_path / episode_path.name
            output_episode_path.mkdir(parents=True, exist_ok=True)

            # Load episode
            try:
                summary_path = episode_path / 'episode_summary.json'
                with open(summary_path, 'r') as f:
                    summary = json.load(f)

                # Find all step folders
                step_folders = sorted(
                    [f for f in episode_path.iterdir() if f.name.startswith('step_')],
                    key=lambda x: int(x.name.split('_')[1])
                )

                steps_data = []
                for step_folder in step_folders:
                    step_data = {}

                    # Load state
                    with open(step_folder / 'state.json', 'r') as f:
                        state = json.load(f)

                    # Load RGB image (convert to PNG bytes)
                    img_path = step_folder / 'rgb_front.png'
                    if img_path.exists():
                        with Image.open(img_path) as img:
                            img_byte_arr = io.BytesIO()
                            img.save(img_byte_arr, format='PNG')
                            img_byte_arr = img_byte_arr.getvalue()
                            step_data['camera_image'] = img_byte_arr

                    # Load semantic label if exists
                    semantic_path = step_folder / 'semantic_front.npy'
                    if semantic_path.exists():
                        semantic = np.load(semantic_path)
                        step_data['semantic_labels'] = semantic.flatten().tolist()
                        step_data['perspective_semantic_image_shape'] = list(semantic.shape)
                    else:
                        step_data['semantic_labels'] = []
                        step_data['perspective_semantic_image_shape'] = [0, 0]

                    # Map action to driving_command
                    action_str = state.get('action', 'start')
                    action_map = {
                        'forward': 0,
                        'left': 1,
                        'right': 2,
                        'ascend': 3,
                        'descend': 4,
                        'rotl': 5,
                        'rotr': 6,
                        'stop': 7,
                        'start': 7,
                    }
                    step_data['driving_command'] = action_map.get(action_str, action_map['stop'])
                    step_data['ego_speed'] = 0.0
                    step_data['path'] = [0.0, 0.0, 0.0, 0.0]
                    step_data['route_poses'] = [0.0, 0.0, 0.0, 0.0]

                    steps_data.append(step_data)

                # Save to parquet files
                num_files = (len(steps_data) + args.samples_per_file - 1) // args.samples_per_file

                for file_idx in range(num_files):
                    start_idx = file_idx * args.samples_per_file
                    end_idx = min((file_idx + 1) * args.samples_per_file, len(steps_data))
                    chunk_steps = steps_data[start_idx:end_idx]

                    # Create DataFrame
                    df = pd.DataFrame(chunk_steps)

                    # Save parquet
                    parquet_filename = f'output_{file_idx:04d}.pqt'
                    parquet_path = output_episode_path / parquet_filename
                    df.to_parquet(parquet_path, engine='pyarrow')

                    # Save metadata
                    metadata = {
                        'episode_id': episode_path.name,
                        'file_index': file_idx,
                        'num_samples': len(chunk_steps),
                        'task_description': args.task_description,
                    }
                    metadata_path = output_episode_path / f'output_{file_idx:04d}_metadata.json'
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=2)

            except Exception as e:
                print(f"Error converting {episode_path}: {e}")
                import traceback
                traceback.print_exc()

    print(f"\nDone! Dataset saved to {output_path}")
    for split_name in splits:
        split_path = output_path / split_name
        if split_path.exists():
            num_episodes = len([f for f in split_path.iterdir() if f.is_dir()])
            print(f"  {split_name}: {num_episodes} episodes")


if __name__ == '__main__':
    main()

