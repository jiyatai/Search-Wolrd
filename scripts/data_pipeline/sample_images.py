# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

'''Sample a few images to understand the dataset.'''

import os
from PIL import Image
import json

dataset_root = "/shared_disk/users/wenhao.lu/JYT/WorldSearch_data/BrushifyUrban_test"

# Check a few episodes
episodes = ['episode_0001', 'episode_0010', 'episode_0050', 'episode_0100']

print("Dataset sample:")
print("=" * 60)

for ep in episodes:
    ep_dir = os.path.join(dataset_root, ep)
    summary_file = os.path.join(ep_dir, "episode_summary.json")

    if os.path.exists(summary_file):
        with open(summary_file) as f:
            summary = json.load(f)
        print(f"\n{ep}:")
        print(f"  Object: {summary.get('object_name', 'N/A')}")
        print(f"  Description: {summary.get('description', 'N/A')[:100]}")

    # Check image size
    step_dir = os.path.join(ep_dir, "step_0000")
    img_path = os.path.join(step_dir, "rgb_front.png")
    if os.path.exists(img_path):
        img = Image.open(img_path)
        print(f"  Image size: {img.size}")

print("\n" + "=" * 60)
