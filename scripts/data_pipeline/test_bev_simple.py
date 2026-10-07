#!/usr/bin/env python
"""
Simple test script to check depth format and basic BEV generation.
"""
import os
import sys
import json
import numpy as np
import math

# Add current dir
sys.path.insert(0, os.path.dirname(__file__))

from bev_memory_simple import BEVMemory

# Test with a single step
dataset_root = '/shared_disk/users/wenhao.lu/JYT/WorldSearch_data/BrushifyUrban_test'
episode_path = os.path.join(dataset_root, 'episode_0001')

# Load episode summary
summary_path = os.path.join(episode_path, 'episode_summary.json')
with open(summary_path, 'r') as f:
    summary = json.load(f)

start_pos = np.array(summary['start_position'])[:2]  # x, y
target_pos = np.array(summary['target_position'][0])[:2]  # x, y
target_rel = target_pos - start_pos

print(f'Start pos: {start_pos}')
print(f'Target pos: {target_pos}')
print(f'Target rel: {target_rel}')

# Load step 0
step_path = os.path.join(episode_path, 'step_0000')
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

print(f'\nStep 0:')
print(f'  Abs pos: ({abs_x:.2f}, {abs_y:.2f})')
print(f'  Rel pos: ({rel_x:.2f}, {rel_y:.2f})')
print(f'  Yaw: {math.degrees(yaw_rad):.2f} deg')

# Load depth map
depth_front = np.load(os.path.join(step_path, 'depth_front.npy'))
print(f'\nDepth map shape: {depth_front.shape}')
print(f'Depth dtype: {depth_front.dtype}')
print(f'Depth range: [{depth_front.min():.3f}, {depth_front.max():.3f}]')
print(f'Depth mean: {depth_front.mean():.3f}')
print(f'Depth stats (non-zero):')
valid = depth_front > 0
print(f'  Valid pixels: {valid.sum()}/{valid.size}')
if valid.sum() > 0:
    print(f'  Valid range: [{depth_front[valid].min():.3f}, {depth_front[valid].max():.3f}]')

# Now try BEV generation with CPU
import torch
device = 'cpu'

print(f'\nUsing device: {device}')

# Initialize BEV memory
bev_memory = BEVMemory(bev_size=(256, 256), world_scale=0.4, fov_range=20.0, fov_angle=90.0).to(device)
bev_state = None

# Test 1: exploration only first
print('\n=== Test 1: Exploration only ===')
pose = torch.tensor([[rel_x, rel_y, 0.0, yaw_rad]], dtype=torch.float32, device=device)
bev_state = bev_memory(pose, depth_map=None, target_position=None, bev_memory=bev_state)
print(f'BEV state shape: {bev_state.shape}')
print(f'Exploration min/max: {bev_state[0, 0].min():.3f}/{bev_state[0, 0].max():.3f}')
print(f'Exploration non-zero: {(bev_state[0, 0] > 0).sum().item()}')

# Test 2: with depth
print('\n=== Test 2: With depth ===')
depth_front_t = torch.tensor(depth_front[np.newaxis, ...], dtype=torch.float32, device=device)
bev_state = bev_memory(pose, depth_map=depth_front_t, target_position=None, bev_memory=bev_state)
print(f'Obstacle min/max: {bev_state[0, 1].min():.3f}/{bev_state[0, 1].max():.3f}')
print(f'Obstacle non-zero: {(bev_state[0, 1] > 0).sum().item()}')

# Test 3: with target
print('\n=== Test 3: With target ===')
target_pos_t = torch.tensor([[target_rel[0], target_rel[1]]], dtype=torch.float32, device=device)
bev_state = bev_memory(pose, depth_map=depth_front_t, target_position=target_pos_t, bev_memory=bev_state)
print(f'Value min/max: {bev_state[0, 2].min():.3f}/{bev_state[0, 2].max():.3f}')

print('\nDone! BEV generation works.')

# Save as numpy for visualization
output_dir = '/shared_disk/users/wenhao.lu/NYZ/CityBot/bev_vis/episode_0001'
os.makedirs(output_dir, exist_ok=True)
np.save(os.path.join(output_dir, 'bev_step_0000.npy'), bev_state[0].cpu().numpy())
print(f'Saved to {output_dir}/bev_step_0000.npy')

# Now do simple matplotlib visualization
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

bev = bev_state[0].cpu().numpy()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Exploration
im0 = axes[0].imshow(bev[0], cmap='viridis', vmin=0, vmax=1)
axes[0].set_title('Exploration')
plt.colorbar(im0, ax=axes[0])

# Obstacle
im1 = axes[1].imshow(bev[1], cmap='plasma', vmin=0, vmax=1)
axes[1].set_title('Obstacle')
plt.colorbar(im1, ax=axes[1])

# Value
im2 = axes[2].imshow(bev[2], cmap='RdYlGn', vmin=0, vmax=1)
axes[2].set_title('Value')
plt.colorbar(im2, ax=axes[2])

# Mark UAV and target
uav_x = 128 + (rel_x / 0.4)
uav_y = 128 - (rel_y / 0.4)
target_x = 128 + (target_rel[0] / 0.4)
target_y = 128 - (target_rel[1] / 0.4)

for ax in axes:
    uav_circle = Circle((uav_x, uav_y), 3, color='red', fill=True, alpha=0.8, label='UAV')
    ax.add_patch(uav_circle)
    if 0 <= target_x < 256 and 0 <= target_y < 256:
        target_circle = Circle((target_x, target_y), 5, color='yellow', fill=True, alpha=0.8, label='Target')
        ax.add_patch(target_circle)

plt.savefig(os.path.join(output_dir, 'bev_test_0000.png'), dpi=150, bbox_inches='tight')
print(f'Plot saved to {output_dir}/bev_test_0000.png')

