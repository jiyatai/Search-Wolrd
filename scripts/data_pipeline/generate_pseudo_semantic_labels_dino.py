# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Script to generate pseudo semantic labels for UAV data using:
# Grounding DINO + simple heuristics
# Requires: transformers, torch, torchvision, PIL, numpy, tqdm
#
# Install:
# pip install transformers torch torchvision pillow numpy tqdm accelerate

import argparse
import os
from typing import List
from tqdm import tqdm
import numpy as np
from PIL import Image

import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


# Semantic classes for BrushifyUrban dataset (6 classes)
class SemanticLabel:
    SKY = 0
    GROUND = 1
    BUILDING = 2
    HUMAN = 3
    VEHICLE = 4
    OBSTACLE = 5


class GroundingLabelGenerator:
    """Generate pseudo-labels using Grounding DINO + heuristics."""

    def __init__(self, model_id="IDEA-Research/grounding-dino-tiny", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model {model_id} on {self.device}...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
        self.model.eval()

        # Category mapping for 6 classes
        # Format: (text_queries, class_id, box_threshold)
        self.class_configs = [
            # Sky is handled with heuristics, not detection
            ("human, person, people, man, woman", SemanticLabel.HUMAN, 0.25),
            ("car, vehicle, automobile, sedan, truck, bus", SemanticLabel.VEHICLE, 0.25),
            ("building, house, wall, structure, tower", SemanticLabel.BUILDING, 0.20),
            ("swing, playground, slide, seesaw, climber, play equipment", SemanticLabel.OBSTACLE, 0.25),
            ("statue, angel, sculpture, monument", SemanticLabel.OBSTACLE, 0.25),
            ("tree, plant, bush", SemanticLabel.OBSTACLE, 0.20),
        ]

    def detect_objects(self, img_pil, text_queries, box_threshold=0.25):
        """Detect objects for given text queries."""
        try:
            inputs = self.processor(
                images=img_pil, text=text_queries, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            # Try different API variants
            try:
                # Newer API
                results = self.processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    threshold=box_threshold
                )
            except TypeError:
                try:
                    # Try without keyword names
                    results = self.processor.post_process_grounded_object_detection(
                        outputs,
                        inputs.input_ids,
                        box_threshold,
                        box_threshold
                    )
                except:
                    # Manual post-processing
                    logits = outputs.logits[0]
                    boxes = outputs.pred_boxes[0]
                    scores = torch.sigmoid(logits.max(dim=-1)[0])
                    keep = scores > box_threshold
                    boxes = boxes[keep]
                    # Convert from normalized to pixel coordinates
                    w, h = img_pil.size
                    boxes = boxes * torch.tensor([w, h, w, h], device=self.device)
                    # Convert from cxcywh to xyxy
                    boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
                    boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
                    boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
                    boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
                    return boxes.cpu().numpy(), scores[keep].cpu().numpy()
            return results[0]["boxes"].cpu().numpy(), results[0]["scores"].cpu().numpy()
        except Exception as e:
            print(f"Error detecting objects: {e}")
            return np.array([]), np.array([])

    @torch.no_grad()
    def __call__(self, img_path: str, img_size=(320, 512)) -> np.ndarray:
        """Generate semantic label for an image."""
        img_pil = Image.open(img_path).convert("RGB")
        img_pil = img_pil.resize(img_size, Image.BILINEAR)
        w, h = img_pil.size

        # Initialize labels - default to ground
        labels = np.full((h, w), SemanticLabel.GROUND, dtype=np.uint8)

        # 1. Sky detection using color heuristic on upper half
        img_np = np.array(img_pil)
        img_upper = img_np[:h//2, :, :]

        # Sky detection using blue channel dominance
        # Sky is blue: blue - max(red, green) > threshold
        r = img_upper[:, :, 0].astype(np.int16)
        g = img_upper[:, :, 1].astype(np.int16)
        b = img_upper[:, :, 2].astype(np.int16)

        # Blue minus max(red, green)
        blue_diff = b - np.maximum(r, g)
        # Also require minimum brightness
        sky_mask = (blue_diff > 15) & (b > 80)
        # Apply to upper half
        labels[:h//2, :][sky_mask] = SemanticLabel.SKY

        # 2. Ground detection - lower half with no other labels will stay ground

        # 3. Detect objects using Grounding DINO
        all_boxes = []
        all_classes = []

        for text_queries, class_id, threshold in self.class_configs:
            boxes, _ = self.detect_objects(img_pil, text_queries, threshold)
            if len(boxes) > 0:
                for box in boxes:
                    all_boxes.append(box)
                    all_classes.append(class_id)

        # 4. Assign labels from bounding boxes
        # Sort by class priority (higher priority first)
        priority_order = [SemanticLabel.HUMAN, SemanticLabel.VEHICLE,
                         SemanticLabel.BUILDING, SemanticLabel.OBSTACLE]

        for priority_class in priority_order:
            for box, class_id in zip(all_boxes, all_classes):
                if class_id != priority_class:
                    continue
                x_min, y_min, x_max, y_max = [int(round(c)) for c in box]
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(w, x_max)
                y_max = min(h, y_max)
                if x_max > x_min and y_max > y_min:
                    labels[y_min:y_max, x_min:x_max] = class_id

        # 5. Post-processing: make building transition smoother in lower half
        # Lower part that's not sky and not object -> ground
        # Ensure buildings are in middle/upper
        return labels


def list_episodes(dataset_root: str) -> List[str]:
    episodes = []
    for d in sorted(os.listdir(dataset_root)):
        if d.startswith("episode_") and os.path.isdir(os.path.join(dataset_root, d)):
            episodes.append(d)
    return episodes


def list_step_dirs(episode_dir: str) -> List[str]:
    steps = []
    for d in sorted(os.listdir(episode_dir)):
        if d.startswith("step_") and os.path.isdir(os.path.join(episode_dir, d)):
            steps.append(d)
    return steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/shared_disk/users/wenhao.lu/JYT/WorldSearch_data/BrushifyUrban_test",
        help="Root directory with episode_* folders"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing semantic_front.npy"
    )
    parser.add_argument(
        "--output_size",
        type=int,
        nargs=2,
        default=(320, 512),
        help="Output size (height, width) for semantic labels"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="IDEA-Research/grounding-dino-tiny",
        help="HuggingFace model id for Grounding DINO"
    )
    parser.add_argument(
        "--start_episode",
        type=int,
        default=0,
        help="Start from this episode index"
    )
    parser.add_argument(
        "--end_episode",
        type=int,
        default=None,
        help="End at this episode index (exclusive)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu, default: auto-detect)"
    )
    args = parser.parse_args()

    # Initialize label generator
    label_generator = GroundingLabelGenerator(model_id=args.model_id, device=args.device)

    episodes = list_episodes(args.dataset_root)
    print(f"Found {len(episodes)} episodes")

    # Apply episode range
    if args.end_episode is None:
        args.end_episode = len(episodes)
    episodes = episodes[args.start_episode:args.end_episode]
    print(f"Processing episodes {args.start_episode} to {args.end_episode - 1} ({len(episodes)} episodes)")

    # Count total steps
    total_steps = 0
    for ep in episodes:
        total_steps += len(list_step_dirs(os.path.join(args.dataset_root, ep)))

    print(f"Processing {total_steps} total steps")
    processed, skipped = 0, 0

    for ep in tqdm(episodes, desc="Episodes"):
        episode_dir = os.path.join(args.dataset_root, ep)
        steps = list_step_dirs(episode_dir)

        for step in steps:  # Simplified output for GPU speed
            step_dir = os.path.join(episode_dir, step)
            img_path = os.path.join(step_dir, "rgb_front.png")
            label_path = os.path.join(step_dir, "semantic_front.npy")

            if not os.path.exists(img_path):
                skipped += 1
                continue

            if os.path.exists(label_path) and not args.overwrite:
                skipped += 1
                continue

            # Generate pseudo-label
            try:
                labels = label_generator(img_path, (args.output_size[1], args.output_size[0]))

                # Resize to target output size if needed
                if labels.shape != tuple(args.output_size):
                    img_pil = Image.fromarray(labels)
                    img_pil = img_pil.resize(args.output_size[::-1], Image.NEAREST)
                    labels = np.array(img_pil)

                # Save as numpy array (uint8) - save as (h, w)
                np.save(label_path, labels.astype(np.uint8))
                processed += 1
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                skipped += 1
                continue

    print(f"Done! Processed {processed} images, skipped {skipped}")


if __name__ == "__main__":
    main()
