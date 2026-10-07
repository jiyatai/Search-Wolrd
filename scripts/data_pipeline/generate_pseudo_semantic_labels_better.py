# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Improved script for pseudo semantic labels - Python 2.7 compatible
# Uses Grounding DINO + better heuristics
#
# Install:
# pip install transformers torch torchvision pillow numpy tqdm accelerate

import argparse
import os
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


# Color map for visualization (RGB)
SEMANTIC_COLORS = np.array([
    [135, 206, 235],   # Sky - light blue
    [34, 139, 34],     # Ground - forest green
    [139, 69, 19],     # Building - saddle brown
    [255, 0, 0],       # Human - red
    [0, 0, 255],       # Vehicle - blue
    [255, 165, 0],     # Obstacle - orange
], dtype=np.uint8)


class BetterLabelGenerator:
    """
    Better label generator with improved heuristics.
    - Better sky detection
    - Better box refinement using color similarity
    - Post-processing smoothing
    """

    def __init__(self, model_id="IDEA-Research/grounding-dino-base", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print("Loading Grounding DINO %s on %s..." % (model_id, self.device))
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
        self.dino_model.to(self.device)
        self.dino_model.eval()

        # Category mapping with better queries and thresholds
        self.class_configs = [
            ("human, person, people, man, woman", SemanticLabel.HUMAN, 0.3),
            ("car, vehicle, automobile, sedan, truck, bus", SemanticLabel.VEHICLE, 0.3),
            ("building, house, wall, structure, tower", SemanticLabel.BUILDING, 0.25),
            ("swing, playground, slide, seesaw, play equipment", SemanticLabel.OBSTACLE, 0.3),
            ("statue, angel, sculpture, monument", SemanticLabel.OBSTACLE, 0.3),
            ("tree, plant, bush, vegetation", SemanticLabel.OBSTACLE, 0.25),
        ]

    def detect_objects(self, img_pil, text_queries, box_threshold=0.3):
        """Detect objects with improved NMS"""
        try:
            inputs = self.processor(
                images=img_pil, text=text_queries, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                outputs = self.dino_model(**inputs)

            # Try different API variants
            try:
                results = self.processor.post_process_grounded_object_detection(
                    outputs, inputs.input_ids, threshold=box_threshold
                )
                boxes = results[0]["boxes"].cpu().numpy()
                scores = results[0]["scores"].cpu().numpy()
            except TypeError:
                try:
                    results = self.processor.post_process_grounded_object_detection(
                        outputs, inputs.input_ids, box_threshold, box_threshold
                    )
                    boxes = results[0]["boxes"].cpu().numpy()
                    scores = results[0]["scores"].cpu().numpy()
                except:
                    # Manual post-processing
                    logits = outputs.logits[0]
                    boxes = outputs.pred_boxes[0]
                    scores = torch.sigmoid(logits.max(dim=-1)[0])
                    keep = scores > box_threshold
                    boxes = boxes[keep]
                    scores = scores[keep]
                    w, h = img_pil.size
                    boxes = boxes * torch.tensor([w, h, w, h], device=self.device)
                    boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
                    boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
                    boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
                    boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
                    boxes = boxes.cpu().numpy()
                    scores = scores.cpu().numpy()

            # Apply simple NMS
            if len(boxes) > 0:
                boxes, scores = self._simple_nms(boxes, scores)

            return boxes, scores

        except Exception as e:
            print("Error detecting objects: %s" % e)
            return np.array([]), np.array([])

    def _simple_nms(self, boxes, scores, iou_threshold=0.5):
        """Simple NMS implementation"""
        if len(boxes) == 0:
            return boxes, scores

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h

            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= iou_threshold)[0]
            order = order[inds + 1]

        return boxes[keep], scores[keep]

    def _detect_sky_improved(self, img_np, depth_np=None):
        """Improved sky detection using multiple heuristics"""
        h, w = img_np.shape[:2]
        sky_mask = np.zeros((h, w), dtype=bool)

        # Only process upper region for sky (2/3 of image)
        sky_region_end = int(h * 0.7)
        img_upper = img_np[:sky_region_end, :, :]

        # Convert to float
        img_float = img_upper.astype(np.float32)
        r = img_float[:, :, 0]
        g = img_float[:, :, 1]
        b = img_float[:, :, 2]

        # Multiple sky indicators
        blue_diff = b - np.maximum(r, g)
        blue_ratio = b / (np.maximum(r, g) + 1e-6)
        brightness = (r + g + b) / 3

        # Sky blue reference
        sky_r, sky_g, sky_b = 135, 206, 235
        color_dist = np.sqrt(
            (r - sky_r)**2 +
            (g - sky_g)**2 +
            (b - sky_b)**2
        )

        # Combine criteria
        sky_candidate = np.zeros_like(blue_diff, dtype=bool)

        # Strong sky pixels
        strong_sky = (
            (blue_diff > 15) &
            (blue_ratio > 1.1) &
            (brightness > 80) &
            (brightness < 250) &
            (color_dist < 100)
        )
        sky_candidate |= strong_sky

        # Moderate sky pixels (near strong ones)
        moderate_sky = (
            (blue_diff > 5) &
            (blue_ratio > 1.05) &
            (brightness > 70) &
            (color_dist < 130)
        )
        sky_candidate |= moderate_sky

        sky_mask[:sky_region_end, :] = sky_candidate

        # Simple morphological cleanup - dilate to connect regions
        sky_mask = self._simple_cleanup(sky_mask)

        return sky_mask

    def _simple_cleanup(self, mask):
        """Simple morphological cleanup without scipy"""
        h, w = mask.shape
        result = mask.copy()

        # Simple 3x3 majority vote to remove noise
        padded = np.pad(result, 1, mode='edge')
        for i in range(h):
            for j in range(w):
                region = padded[i:i+3, j:j+3]
                if np.sum(region) >= 5:
                    result[i, j] = True
                elif np.sum(region) <= 2:
                    result[i, j] = False

        return result

    def _refine_box_mask(self, box, img_np):
        """Refine box mask using color similarity"""
        h, w = img_np.shape[:2]
        x1, y1, x2, y2 = box.astype(int)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        if x2 <= x1 or y2 <= y1:
            mask = np.zeros((h, w), dtype=bool)
            mask[y1:y2, x1:x2] = True
            return mask

        # Shrink box slightly for reliable color sample
        bw, bh = x2 - x1, y2 - y1
        shrink = 0.15
        x1s = int(x1 + bw * shrink)
        y1s = int(y1 + bh * shrink)
        x2s = int(x2 - bw * shrink)
        y2s = int(y2 - shrink)
        x1s, y1s = max(0, x1s), max(0, y1s)
        x2s, y2s = min(w, x2s), min(h, y2s)

        if x2s <= x1s or y2s <= y1s:
            x1s, y1s, x2s, y2s = x1, y1, x2, y2

        # Get sample region color stats
        sample_region = img_np[y1s:y2s, x1s:x2s]
        if sample_region.size == 0:
            mask = np.zeros((h, w), dtype=bool)
            mask[y1:y2, x1:x2] = True
            return mask

        mean_color = np.mean(sample_region, axis=(0, 1))
        std_color = np.std(sample_region, axis=(0, 1))
        std_color = std_color * 1.5 + 10

        # Compute color distance in box region
        box_region = img_np[y1:y2, x1:x2]
        color_dist = np.sqrt(np.sum((box_region - mean_color)**2, axis=-1))

        # Threshold
        max_dist = np.sqrt(np.sum(std_color**2)) * 1.5
        object_mask = color_dist < max_dist

        # Also use brightness similarity
        brightness_box = np.mean(box_region, axis=-1)
        brightness_sample = np.mean(sample_region)
        brightness_diff = np.abs(brightness_box - brightness_sample)
        brightness_mask = brightness_diff < 60

        combined_mask = object_mask | brightness_mask

        mask = np.zeros((h, w), dtype=bool)
        mask[y1:y2, x1:x2] = combined_mask

        return mask

    def _smooth_labels(self, labels):
        """Simple label smoothing using majority vote"""
        h, w = labels.shape
        labels_smooth = labels.copy()

        # 3x3 majority vote
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                region = labels[i-1:i+2, j-1:j+2]
                values, counts = np.unique(region, return_counts=True)
                if len(counts) > 0:
                    labels_smooth[i, j] = values[np.argmax(counts)]

        return labels_smooth

    def __call__(self, img_path, depth_path=None, img_size=(320, 512)):
        """Generate semantic label"""
        img_pil = Image.open(img_path).convert("RGB")
        img_pil = img_pil.resize(img_size, Image.BILINEAR)
        w, h = img_pil.size
        img_np = np.array(img_pil)

        # Load depth if available
        depth_np = None
        if depth_path is not None and os.path.exists(depth_path):
            try:
                depth_np = np.load(depth_path)
                depth_pil = Image.fromarray(depth_np)
                depth_pil = depth_pil.resize(img_size, Image.NEAREST)
                depth_np = np.array(depth_pil)
            except Exception as e:
                depth_np = None

        # Initialize with ground
        labels = np.full((h, w), SemanticLabel.GROUND, dtype=np.uint8)

        # 1. Sky detection (improved)
        sky_mask = self._detect_sky_improved(img_np, depth_np)
        labels[sky_mask] = SemanticLabel.SKY

        # 2. Detect objects
        all_boxes = []
        all_classes = []

        for text_queries, class_id, threshold in self.class_configs:
            boxes, scores = self.detect_objects(img_pil, text_queries, threshold)
            if len(boxes) > 0:
                for box in boxes:
                    all_boxes.append(box)
                    all_classes.append(class_id)

        # 3. Create refined masks for each detection
        object_masks = []
        for box, class_id in zip(all_boxes, all_classes):
            mask = self._refine_box_mask(box, img_np)
            object_masks.append((mask, class_id))

        # 4. Apply masks in priority order
        priority_order = [
            SemanticLabel.HUMAN,
            SemanticLabel.VEHICLE,
            SemanticLabel.OBSTACLE,
            SemanticLabel.BUILDING,
            SemanticLabel.GROUND,
            SemanticLabel.SKY
        ]

        for priority_class in priority_order:
            for mask, class_id in object_masks:
                if class_id == priority_class:
                    labels[mask] = class_id

        # 5. Also apply raw boxes as fallback for any missed regions
        for priority_class in priority_order:
            for box, class_id in zip(all_boxes, all_classes):
                if class_id == priority_class:
                    x1, y1, x2, y2 = box.astype(int)
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(w, x2)
                    y2 = min(h, y2)
                    if x2 > x1 and y2 > y1:
                        region = labels[y1:y2, x1:x2]
                        fill_mask = region == SemanticLabel.GROUND
                        labels[y1:y2, x1:x2][fill_mask] = class_id

        # 6. Final smoothing
        labels = self._smooth_labels(labels)

        # Ensure sky remains in upper part
        sky_region = int(h * 0.15)
        labels[:sky_region, :][sky_mask[:sky_region, :]] = SemanticLabel.SKY

        return labels


def list_episodes(dataset_root):
    episodes = []
    for d in sorted(os.listdir(dataset_root)):
        if d.startswith("episode_") and os.path.isdir(os.path.join(dataset_root, d)):
            episodes.append(d)
    return episodes


def list_step_dirs(episode_dir):
    steps = []
    for d in sorted(os.listdir(episode_dir)):
        if d.startswith("step_") and os.path.isdir(os.path.join(episode_dir, d)):
            steps.append(d)
    return steps


def apply_color_map(semantic_label):
    """Apply color map to semantic label"""
    h, w = semantic_label.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for label in range(6):
        mask = semantic_label == label
        colored[mask] = SEMANTIC_COLORS[label]
    return colored


def main():
    parser = argparse.ArgumentParser(
        description="Better pseudo semantic label generation (Python 2 compatible)"
    )
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
        "--model",
        type=str,
        default="IDEA-Research/grounding-dino-base",
        help="Grounding DINO model"
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
    parser.add_argument(
        "--save_vis",
        action="store_true",
        help="Save visualization alongside labels"
    )
    args = parser.parse_args()

    # Initialize label generator
    label_generator = BetterLabelGenerator(
        model_id=args.model,
        device=args.device
    )

    episodes = list_episodes(args.dataset_root)
    print("Found %d episodes" % len(episodes))

    # Apply episode range
    if args.end_episode is None:
        args.end_episode = len(episodes)
    episodes = episodes[args.start_episode:args.end_episode]
    print("Processing episodes %d to %d (%d episodes)" % (
        args.start_episode, args.end_episode - 1, len(episodes)
    ))

    # Count total steps
    total_steps = 0
    for ep in episodes:
        total_steps += len(list_step_dirs(os.path.join(args.dataset_root, ep)))

    print("Processing %d total steps" % total_steps)
    processed, skipped = 0, 0

    for ep in tqdm(episodes, desc="Episodes"):
        episode_dir = os.path.join(args.dataset_root, ep)
        steps = list_step_dirs(episode_dir)

        for step in steps:
            step_dir = os.path.join(episode_dir, step)
            img_path = os.path.join(step_dir, "rgb_front.png")
            depth_path = os.path.join(step_dir, "depth_front.npy")
            label_path = os.path.join(step_dir, "semantic_front.npy")

            if not os.path.exists(img_path):
                skipped += 1
                continue

            if os.path.exists(label_path) and not args.overwrite:
                skipped += 1
                continue

            # Generate pseudo-label
            try:
                labels = label_generator(
                    img_path,
                    depth_path if os.path.exists(depth_path) else None,
                    (args.output_size[1], args.output_size[0])
                )

                # Resize to target output size if needed
                if labels.shape != tuple(args.output_size):
                    img_pil = Image.fromarray(labels)
                    img_pil = img_pil.resize(args.output_size[::-1], Image.NEAREST)
                    labels = np.array(img_pil)

                # Save as numpy array (uint8)
                np.save(label_path, labels.astype(np.uint8))

                # Save visualization if requested
                if args.save_vis:
                    vis_path = os.path.join(step_dir, "semantic_vis_better.png")
                    colored = apply_color_map(labels)
                    Image.fromarray(colored).save(vis_path)

                processed += 1
            except Exception as e:
                print("Error processing %s: %s" % (img_path, e))
                skipped += 1
                continue

    print("Done! Processed %d images, skipped %d" % (processed, skipped))


if __name__ == "__main__":
    main()
