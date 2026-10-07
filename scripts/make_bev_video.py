#!/usr/bin/env python
"""
Make a video from BEV visualizations.
"""
import os
import sys
import subprocess

output_dir = '/shared_disk/users/wenhao.lu/NYZ/CityBot/bev_vis/episode_0001'
video_path = os.path.join(output_dir, 'bev_evolution.mp4')

# Use ffmpeg to make video
cmd = [
    'ffmpeg',
    '-y',
    '-framerate', '5',
    '-i', os.path.join(output_dir, 'comparison_step_%04d.png'),
    '-c:v', 'libx264',
    '-pix_fmt', 'yuv420p',
    video_path
]

print(' '.join(cmd))
subprocess.run(cmd, check=True)
print(f'Video saved to {video_path}')

