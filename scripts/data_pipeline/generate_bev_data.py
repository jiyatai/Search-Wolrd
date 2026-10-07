#!/usr/bin/env python
"""
Generate BEV memory data for decoder training.
"""
import os
import sys
import json
import numpy as np
import torch
import math

# Add current dir
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from bev_memory_simple import BEVMemory


def load_episode_data(episode_path, max_steps=50):
    """Load all steps from an episode."""
    # Load episode summary
    summary_path = os.path.join(episode_path, 'episode_summary.json')
    with open(summary_path, 'r') as f:
        summary = json.load(f)

    start_pos = np.array(summary['start_position'])[:2]  # x, y
    target_pos = np.array(summary['target_position'][0])[:2]  # x, y

    # Load each step
    steps = []
    for step_idx in range(min(max_steps, summary['total_steps'])):
        step_path = os.path.join(episode_path, f'step_{step_idx:04d}')
        if not os.path.exists(step_path):
            continue

        state_path = os.path.join(step_path, 'state.json')
        with open(state_path, 'r') as f:
            state = json.load(f)

        # Get absolute position
        abs_x = state['position']['x']
        abs_y = state['position']['y']

        # Convert to relative position (relative to start)
        rel_x = abs_x - start_pos[0]
        rel_y = abs_y - start_pos[1]

        # Get yaw
        yaw_rad = math.radians(state['attitude_euler_deg']['yaw_deg'])

        # Load depth maps
        depth_front = np.load(os.path.join(step_path, 'depth_front.npy'))

        steps.append({
            'step_idx': step_idx,
            'rel_x': rel_x,
            'rel_y': rel_y,
            'yaw_rad': yaw_rad,
            'depth_front': depth_front,
        })

    # Convert target to relative
    target_rel = target_pos - start_pos

    return steps, target_rel, start_pos, target_pos


def generate_bev_for_episode(episode_path, output_dir, device='cuda'):
    """Generate BEV data for an episode."""
    os.makedirs(output_dir, exist_ok=True)

    # Load episode data
    steps, target_rel, start_pos, target_pos = load_episode_data(episode_path)

    # Initialize BEV memory
    bev_memory = BEVMemory(bev_size=(256, 256), world_scale=0.4, fov_range=20.0, fov_angle=90.0).to(device)
    bev_state = None

    # Save episode metadata
    metadata = {
        'start_pos': start_pos.tolist(),
        'target_pos': target_pos.tolist(),
        'target_rel': target_rel.tolist(),
        'num_steps': len(steps),
    }
    np.save(os.path.join(output_dir, 'metadata.npy'), metadata)

    # Process each step
    bev_sequence = []
    for step_data in steps:
        step_idx = step_data['step_idx']

        # Pose: [rel_x, rel_y, z, yaw]
        pose = torch.tensor([[step_data['rel_x'], step_data['rel_y'], 0.0, step_data['yaw_rad']]],
                           dtype=torch.float32, device=device)

        # Depth map (front view)
        depth_front = torch.tensor(step_data['depth_front'][np.newaxis, ...], dtype=torch.float32, device=device)

        # Target position for value layer GT
        target_pos_tensor = torch.tensor([[target_rel[0], target_rel[1]]], dtype=torch.float32, device=device)

        # Update BEV
        bev_state = bev_memory(pose, depth_front, target_pos_tensor, bev_state)

        # Save BEV state for this step
        bev_np = bev_state[0].cpu().numpy()  # (3, 256, 256)
        bev_sequence.append(bev_np)

    # Save BEV sequence as a single array
    bev_sequence_np = np.stack(bev_sequence, axis=0)  # (T, 3, 256, 256)
    np.save(os.path.join(output_dir, 'bev_sequence.npy'), bev_sequence_np)

    print(f'Saved {len(steps)} steps to {output_dir}')
    return bev_sequence_np


def main():
    # Configuration
    dataset_root = '/shared_disk/users/wenhao.lu/JYT/WorldSearch_data/BrushifyUrban_test'
    output_root = os.path.join(dataset_root, 'bev_data')

    # Find all episodes
    episode_dirs = []
    for f in os.listdir(dataset_root):
        if f.startswith('episode_') and os.path.isdir(os.path.join(dataset_root, f)):
            episode_dirs.append(f)
    episode_dirs.sort()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    print(f'Found {len(episode_dirs)} episodes')

    # Generate BEV for each episode
    for episode_dir in episode_dirs:
        episode_idx = int(episode_dir.split('_')[1])
        episode_path = os.path.join(dataset_root, episode_dir)
        output_dir = os.path.join(output_root, episode_dir)

        # Skip if already exists
        if os.path.exists(os.path.join(output_dir, 'bev_sequence.npy')):
            print(f'Skipping {episode_dir} (already exists)')
            continue

        print(f'Processing {episode_dir}...')
        try:
            generate_bev_for_episode(episode_path, output_dir, device=device)
        except Exception as e:
            print(f'Error processing {episode_dir}: {e}')
            import traceback
            traceback.print_exc()

    print('Done!')


if __name__ == '__main__':
    main()

