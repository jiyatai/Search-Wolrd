#!/usr/bin/env python
"""
Generate BEV memory from dataset and visualize results.
"""
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import torch
import torch.nn.functional as F
from PIL import Image
import math

# Add project path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from bev_memory_simple import BEVMemory


def load_npy(npy_path):
    """Load numpy array from file."""
    return np.load(npy_path)


def load_depth(depth_path):
    """Load depth map (in meters)."""
    depth = np.load(depth_path)
    return depth


def quaternion_to_yaw(qx, qy, qz, qw):
    """Convert quaternion to yaw angle in radians."""
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return yaw


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
        depth_front = load_depth(os.path.join(step_path, 'depth_front.npy'))
        depth_left = load_depth(os.path.join(step_path, 'depth_left.npy'))
        depth_right = load_depth(os.path.join(step_path, 'depth_right.npy'))

        # Load RGB for visualization
        rgb_front = np.array(Image.open(os.path.join(step_path, 'rgb_front.png')))

        steps.append({
            'step_idx': step_idx,
            'abs_x': abs_x,
            'abs_y': abs_y,
            'rel_x': rel_x,
            'rel_y': rel_y,
            'yaw_rad': yaw_rad,
            'depth_front': depth_front,
            'depth_left': depth_left,
            'depth_right': depth_right,
            'rgb_front': rgb_front,
        })

    # Convert target to relative
    target_rel = target_pos - start_pos

    return steps, target_rel, start_pos, target_pos


def generate_bev_for_episode(episode_path, output_dir, device='cuda'):
    """Generate BEV for an episode and save visualizations."""
    os.makedirs(output_dir, exist_ok=True)

    # Load episode data
    steps, target_rel, start_pos, target_pos = load_episode_data(episode_path)

    # Initialize BEV memory
    bev_memory = BEVMemory(bev_size=(256, 256), world_scale=0.4, fov_range=20.0, fov_angle=90.0).to(device)
    bev_state = None

    # Process each step
    for step_data in steps:
        step_idx = step_data['step_idx']
        print(f'Processing step {step_idx}...')

        # Pose: [rel_x, rel_y, z, yaw]
        pose = torch.tensor([[step_data['rel_x'], step_data['rel_y'], 0.0, step_data['yaw_rad']]],
                           dtype=torch.float32, device=device)

        # Depth map (front view)
        depth_front = torch.tensor(step_data['depth_front'][np.newaxis, ...], dtype=torch.float32, device=device)

        # Target position for value layer GT
        target_pos_tensor = torch.tensor([[target_rel[0], target_rel[1]]], dtype=torch.float32, device=device)

        # Update BEV
        bev_state = bev_memory(pose, depth_front, target_pos_tensor, bev_state)

        # Visualize
        fig = visualize_bev_step(bev_state, step_data, target_rel, step_idx)
        fig.savefig(os.path.join(output_dir, f'bev_step_{step_idx:04d}.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)

        # Also save RGB comparison
        fig = visualize_rgb_bev_comparison(bev_state, step_data, target_rel, step_idx)
        fig.savefig(os.path.join(output_dir, f'comparison_step_{step_idx:04d}.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)

    print(f'Saved {len(steps)} BEV visualizations to {output_dir}')
    return bev_state


def visualize_bev_step(bev_state, step_data, target_rel, step_idx):
    """Visualize BEV state for a single step."""
    bev = bev_state[0].cpu().numpy()  # (3, 256, 256)
    exploration = bev[0]
    obstacle = bev[1]
    value = bev[2]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # 1. Exploration layer
    im1 = axes[0].imshow(exploration, cmap='viridis', vmin=0, vmax=1, origin='upper')
    axes[0].set_title(f'Exploration\nStep {step_idx}')
    axes[0].set_xlabel('BEV X')
    axes[0].set_ylabel('BEV Y')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # 2. Obstacle layer
    im2 = axes[1].imshow(obstacle, cmap='plasma', vmin=0, vmax=1, origin='upper')
    axes[1].set_title('Obstacle')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    # 3. Value layer
    im3 = axes[2].imshow(value, cmap='RdYlGn', vmin=0, vmax=1, origin='upper')
    axes[2].set_title('Value')
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    # 4. Composite view
    composite = np.zeros((256, 256, 3))
    composite[..., 0] = obstacle  # Red = obstacle
    composite[..., 1] = exploration  # Green = exploration
    composite[..., 2] = value  # Blue = value
    axes[3].imshow(composite, origin='upper')
    axes[3].set_title('Composite\n(R=Obstacle, G=Exploration, B=Value)')

    # Mark UAV position on all plots
    uav_x = 128 + (step_data['rel_x'] / 0.4)
    uav_y = 128 - (step_data['rel_y'] / 0.4)

    # Mark target position
    target_x = 128 + (target_rel[0] / 0.4)
    target_y = 128 - (target_rel[1] / 0.4)

    for ax in axes:
        # UAV position (red)
        uav_circle = Circle((uav_x, uav_y), 3, color='red', fill=True, alpha=0.8, label='UAV')
        ax.add_patch(uav_circle)

        # Target position (yellow)
        if 0 <= target_x < 256 and 0 <= target_y < 256:
            target_circle = Circle((target_x, target_y), 5, color='yellow', fill=True, alpha=0.8, label='Target')
            ax.add_patch(target_circle)

        # Draw FOV cone
        yaw = step_data['yaw_rad']
        fov_angle = np.radians(45)
        fov_range = 20 / 0.4  # meters to pixels
        # Draw FOV boundaries
        for angle_offset in [-fov_angle, fov_angle]:
            end_x = uav_x + fov_range * np.cos(yaw + angle_offset)
            end_y = uav_y - fov_range * np.sin(yaw + angle_offset)  # minus because BEV Y is flipped
            ax.plot([uav_x, end_x], [uav_y, end_y], 'r-', alpha=0.5, linewidth=1)

    plt.tight_layout()
    return fig


def visualize_rgb_bev_comparison(bev_state, step_data, target_rel, step_idx):
    """Visualize RGB and BEV side by side."""
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1])

    # RGB front view
    ax1 = fig.add_subplot(gs[:, 0])
    ax1.imshow(step_data['rgb_front'])
    ax1.set_title(f'Front View\nStep {step_idx}')
    ax1.axis('off')

    # BEV layers
    bev = bev_state[0].cpu().numpy()
    layer_names = ['Exploration', 'Obstacle', 'Value']
    cmaps = ['viridis', 'plasma', 'RdYlGn']

    for i in range(3):
        ax = fig.add_subplot(gs[i // 2, 1 + (i % 2)])
        im = ax.imshow(bev[i], cmap=cmaps[i], vmin=0, vmax=1, origin='upper')
        ax.set_title(layer_names[i])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Mark UAV position
        uav_x = 128 + (step_data['rel_x'] / 0.4)
        uav_y = 128 - (step_data['rel_y'] / 0.4)
        uav_circle = Circle((uav_x, uav_y), 3, color='red', fill=True, alpha=0.8)
        ax.add_patch(uav_circle)

        # Mark target
        target_x = 128 + (target_rel[0] / 0.4)
        target_y = 128 - (target_rel[1] / 0.4)
        if 0 <= target_x < 256 and 0 <= target_y < 256:
            target_circle = Circle((target_x, target_y), 5, color='yellow', fill=True, alpha=0.8)
            ax.add_patch(target_circle)

    # Add pose info
    pose_text = (f"UAV rel pos: ({step_data['rel_x']:.1f}, {step_data['rel_y']:.1f}) m\n"
                f"Yaw: {np.degrees(step_data['yaw_rad']):.1f} deg\n"
                f"Target rel pos: ({target_rel[0]:.1f}, {target_rel[1]:.1f}) m")
    fig.text(0.5, 0.01, pose_text, ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    return fig


def main():
    # Configuration
    dataset_root = '/shared_disk/users/wenhao.lu/JYT/WorldSearch_data/BrushifyUrban_test'
    output_root = os.path.join(dataset_root, 'bev_vis')

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
        if os.path.exists(output_dir) and len(os.listdir(output_dir)) >= 100:
            print(f'Skipping {episode_dir} (already exists)')
            continue

        print(f'Processing {episode_dir}...')
        try:
            generate_bev_for_episode(episode_path, output_dir, device=device)
        except Exception as e:
            print(f'Error processing {episode_dir}: {e}')

    print('Done!')


if __name__ == '__main__':
    main()

