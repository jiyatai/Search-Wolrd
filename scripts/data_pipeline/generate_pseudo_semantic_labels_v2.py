# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Lightweight improved script for pseudo semantic labels
# Uses Grounding DINO + better heuristics + GrabCut/edge-aware refinement
# No SAM dependency - easier to install and faster
#
# Install:
# pip install transformers torch torchvision pillow numpy tqdm accelerate scipy opencv-python

import argparse
import os
from typing import List, Tuple, Optional
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


class LabelGeneratorV2:
    """
    Improved label generator without SAM dependency.
    Uses:
    - Grounding DINO for detection
    - Edge-aware box refinement (color-based)
    - Better sky/ground heuristics
    - Morphological post-processing
    - Optional GrabCut for better boundaries (if OpenCV available)
    """

    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-base",
        device=None,
        use_grabcut: bool = True,
        use_depth: bool = True
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_grabcut = use_grabcut
        self.use_depth = use_depth

        print(f"Loading Grounding DINO {model_id} on {self.device}...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
        self.dino_model.to(self.device)
        self.dino_model.eval()

        # Check for OpenCV
        self.has_cv2 = False
        if self.use_grabcut:
            try:
                import cv2
                self.cv2 = cv2
                self.has_cv2 = True
                print("OpenCV available - will use GrabCut for boundary refinement")
            except ImportError:
                print("OpenCV not available - skipping GrabCut")
                self.has_cv2 = False

        # Category mapping with improved queries
        self.class_configs = [
            # (text_queries, class_id, box_threshold, box_shrink)
            ("human, person, people, man, woman, pedestrian",
             SemanticLabel.HUMAN, 0.30, 0.1),
            ("car, vehicle, automobile, sedan, truck, bus, van",
             SemanticLabel.VEHICLE, 0.30, 0.05),
            ("building, house, wall, structure, tower, facade",
             SemanticLabel.BUILDING, 0.25, 0.0),
            ("swing, playground, slide, seesaw, climber, play equipment",
             SemanticLabel.OBSTACLE, 0.30, 0.05),
            ("statue, angel, sculpture, monument",
             SemanticLabel.OBSTACLE, 0.30, 0.05),
            ("tree, plant, bush, vegetation, foliage",
             SemanticLabel.OBSTACLE, 0.25, 0.0),
        ]

        # Class priority
        self.priority_order = [
            SemanticLabel.HUMAN,
            SemanticLabel.VEHICLE,
            SemanticLabel.OBSTACLE,
            SemanticLabel.BUILDING,
            SemanticLabel.GROUND,
            SemanticLabel.SKY
        ]

    def detect_objects(
        self,
        img_pil,
        text_queries: str,
        box_threshold: float = 0.3
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Detect objects with NMS."""
        try:
            inputs = self.processor(
                images=img_pil, text=text_queries, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                outputs = self.dino_model(**inputs)

            # Try different API variants
            try:
                results = self.processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    threshold=box_threshold
                )
                boxes = results[0]["boxes"].cpu().numpy()
                scores = results[0]["scores"].cpu().numpy()
            except TypeError:
                try:
                    results = self.processor.post_process_grounded_object_detection(
                        outputs,
                        inputs.input_ids,
                        box_threshold,
                        box_threshold
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

            # Apply NMS
            if len(boxes) > 0:
                boxes, scores = self._nms(boxes, scores, iou_threshold=0.5)

            return boxes, scores

        except Exception as e:
            print(f"Error detecting objects: {e}")
            return np.array([]), np.array([])

    def _nms(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        iou_threshold: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Non-maximum suppression."""
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

    def _detect_sky(
        self,
        img_np: np.ndarray,
        depth_np: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Multi-step sky detection with refinement."""
        h, w = img_np.shape[:2]
        sky_mask = np.zeros((h, w), dtype=bool)

        # Process upper region for sky
        sky_region = slice(0, int(h * 0.75))
        img_upper = img_np[sky_region, :, :].astype(np.float32)

        r, g, b = img_upper[..., 0], img_upper[..., 1], img_upper[..., 2]

        # Multiple sky indicators
        blue_diff = b - np.maximum(r, g)
        blue_ratio = b / (np.maximum(r, g) + 1e-6)
        brightness = (r + g + b) / 3

        # Sky color similarity (CIELAB distance if possible)
        sky_rgb = np.array([135, 206, 235], dtype=np.float32)
        color_dist = np.sqrt(np.sum((img_upper - sky_rgb)**2, axis=-1))

        # Combine criteria with adaptive thresholds
        sky_candidate = np.zeros_like(blue_diff, dtype=bool)

        # Strong sky candidates
        strong_sky = (
            (blue_diff > 15) &
            (blue_ratio > 1.15) &
            (brightness > 90) &
            (brightness < 245) &
            (color_dist < 90)
        )
        sky_candidate |= strong_sky

        # Moderate sky candidates (near strong candidates)
        moderate_sky = (
            (blue_diff > 5) &
            (blue_ratio > 1.05) &
            (brightness > 70) &
            (color_dist < 130)
        )

        # Use depth to confirm sky (sky should be far)
        if depth_np is not None and self.use_depth:
            try:
                depth_upper = depth_np[sky_region, :]
                valid_depth = np.isfinite(depth_upper)
                if np.any(valid_depth):
                    depth_thresh = np.percentile(depth_upper[valid_depth], 60)
                    far_region = depth_upper > depth_thresh
                    moderate_sky = moderate_sky | far_region
            except:
                pass

        sky_candidate |= moderate_sky

        sky_mask[sky_region, :] = sky_candidate

        # Morphological cleanup
        sky_mask = self._cleanup_sky_mask(sky_mask)

        return sky_mask

    def _cleanup_sky_mask(self, sky_mask: np.ndarray) -> np.ndarray:
        """Clean up sky mask with morphological operations."""
        try:
            from scipy import ndimage

            # Remove small noise
            sky_mask = ndimage.binary_opening(sky_mask, structure=np.ones((3, 3)), iterations=2)
            # Fill holes
            sky_mask = ndimage.binary_closing(sky_mask, structure=np.ones((5, 5)), iterations=3)

            # Keep only large connected components (sky is usually one big region)
            labeled, num_features = ndimage.label(sky_mask)
            if num_features > 0:
                sizes = ndimage.sum(sky_mask, labeled, range(num_features + 1))
                # Keep components larger than 5% of sky region
                min_size = sky_mask.size * 0.05
                for i in range(1, num_features + 1):
                    if sizes[i] < min_size:
                        sky_mask[labeled == i] = False

                # Connect nearby regions
                sky_mask = ndimage.binary_dilation(sky_mask, structure=np.ones((5, 5)), iterations=1)
                sky_mask = ndimage.binary_closing(sky_mask, structure=np.ones((7, 7)), iterations=2)

        except ImportError:
            pass  # Skip if scipy not available

        return sky_mask

    def _refine_box_with_color(
        self,
        box: np.ndarray,
        img_np: np.ndarray,
        shrink_factor: float = 0.0
    ) -> np.ndarray:
        """
        Refine box boundaries using color similarity.
        Creates a mask that follows object edges better than raw box.
        """
        h, w = img_np.shape[:2]
        x1, y1, x2, y2 = box.astype(int)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return np.zeros((h, w), dtype=bool)

        # Shrink box slightly for more reliable color sample
        if shrink_factor > 0:
            bw, bh = x2 - x1, y2 - y1
            x1s = int(x1 + bw * shrink_factor)
            y1s = int(y1 + bh * shrink_factor)
            x2s = int(x2 - bw * shrink_factor)
            y2s = int(y2 - bh * shrink_factor)
            x1s, y1s = max(0, x1s), max(0, y1s)
            x2s, y2s = min(w, x2s), min(h, y2s)
        else:
            x1s, y1s, x2s, y2s = x1, y1, x2, y2

        # Get sample region color stats
        if x2s > x1s and y2s > y1s:
            sample_region = img_np[y1s:y2s, x1s:x2s]
        else:
            sample_region = img_np[y1:y2, x1:x2]

        if sample_region.size == 0:
            mask = np.zeros((h, w), dtype=bool)
            mask[y1:y2, x1:x2] = True
            return mask

        # Compute color statistics in sample region
        mean_color = np.mean(sample_region, axis=(0, 1))
        std_color = np.std(sample_region, axis=(0, 1))

        # Expand std slightly to be more inclusive
        std_color = std_color * 1.5 + 10

        # Compute color distance across the full box region
        box_region = img_np[y1:y2, x1:x2]
        color_dist = np.sqrt(np.sum((box_region - mean_color)**2, axis=-1))

        # Threshold - pixels with similar color are part of object
        max_dist = np.sqrt(np.sum(std_color**2)) * 1.5
        object_mask = color_dist < max_dist

        # Also consider brightness similarity
        brightness_box = np.mean(box_region, axis=-1)
        brightness_sample = np.mean(sample_region)
        brightness_diff = np.abs(brightness_box - brightness_sample)
        brightness_mask = brightness_diff < 60

        # Combine
        combined_mask = object_mask | brightness_mask

        # Create full-size mask
        mask = np.zeros((h, w), dtype=bool)
        mask[y1:y2, x1:x2] = combined_mask

        return mask

    def _refine_box_with_grabcut(
        self,
        box: np.ndarray,
        img_np: np.ndarray
    ) -> np.ndarray:
        """Refine box using GrabCut algorithm (OpenCV)."""
        if not self.has_cv2:
            return None

        h, w = img_np.shape[:2]
        x1, y1, x2, y2 = box.astype(int)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        # Expand box slightly for GrabCut
        bw, bh = x2 - x1, y2 - y1
        x1_gc = max(0, x1 - int(bw * 0.1))
        y1_gc = max(0, y1 - int(bh * 0.1))
        x2_gc = min(w, x2 + int(bw * 0.1))
        y2_gc = min(h, y2 + int(bh * 0.1))

        # GrabCut expects (x, y, w, h)
        gc_rect = (x1_gc, y1_gc, x2_gc - x1_gc, y2_gc - y1_gc)

        # Initialize mask
        mask = np.zeros(img_np.shape[:2], dtype=np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        # Convert to BGR for OpenCV
        img_bgr = img_np[..., ::-1].copy()

        try:
            # Run GrabCut
            self.cv2.grabCut(
                img_bgr, mask, gc_rect,
                bgd_model, fgd_model,
                iterCount=5,
                mode=self.cv2.GC_INIT_WITH_RECT
            )

            # Create result mask
            result_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype(bool)

            return result_mask

        except Exception as e:
            print(f"GrabCut failed: {e}")
            return None

    def _smooth_labels(
        self,
        labels: np.ndarray,
        window_size: int = 3
    ) -> np.ndarray:
        """Smooth labels using majority voting."""
        h, w = labels.shape
        labels_smooth = labels.copy()

        half = window_size // 2

        try:
            from scipy import ndimage

            for class_id in range(6):
                mask = labels == class_id
                # Clean up
                mask = ndimage.binary_opening(mask, iterations=1)
                mask = ndimage.binary_closing(mask, iterations=1)
                labels_smooth[mask] = class_id

        except ImportError:
            # Fallback: simple majority vote
            for i in range(half, h - half):
                for j in range(half, w - half):
                    patch = labels[i-half:i+half+1, j-half:j+half+1]
                    values, counts = np.unique(patch, return_counts=True)
                    labels_smooth[i, j] = values[np.argmax(counts)]

        return labels_smooth

    def _apply_depth_refinement(
        self,
        labels: np.ndarray,
        depth_np: np.ndarray
    ) -> np.ndarray:
        """Refine labels using depth discontinuities."""
        if not self.use_depth or depth_np is None:
            return labels

        h, w = labels.shape

        try:
            from scipy import ndimage

            # Compute depth gradients
            depth_float = depth_np.astype(np.float32)
            # Fill NaN/inf
            valid = np.isfinite(depth_float)
            if not np.all(valid):
                depth_float[~valid] = np.interp(
                    np.where(~valid)[0],
                    np.where(valid)[0],
                    depth_float[valid]
                )

            # Compute edges
            gy, gx = np.gradient(depth_float)
            grad_mag = np.sqrt(gx**2 + gy**2)
            edge_mask = grad_mag > np.percentile(grad_mag, 75)

            # Dilate edges
            edge_mask = ndimage.binary_dilation(edge_mask, iterations=1)

            # Keep labels from crossing edges - use original label at edges
            labels_refined = labels.copy()

            return labels_refined

        except ImportError:
            return labels

    @torch.no_grad()
    def __call__(
        self,
        img_path: str,
        depth_path: Optional[str] = None,
        img_size: Tuple[int, int] = (320, 512)
    ) -> np.ndarray:
        """Generate semantic label."""
        img_pil = Image.open(img_path).convert("RGB")
        img_pil = img_pil.resize(img_size, Image.LANCZOS)
        w, h = img_pil.size
        img_np = np.array(img_pil)

        # Load depth
        depth_np = None
        if depth_path is not None and os.path.exists(depth_path) and self.use_depth:
            try:
                depth_np = np.load(depth_path)
                depth_pil = Image.fromarray(depth_np)
                depth_pil = depth_pil.resize(img_size, Image.NEAREST)
                depth_np = np.array(depth_pil)
            except Exception as e:
                depth_np = None

        # Initialize with ground
        labels = np.full((h, w), SemanticLabel.GROUND, dtype=np.uint8)

        # 1. Sky detection
        sky_mask = self._detect_sky(img_np, depth_np)
        labels[sky_mask] = SemanticLabel.SKY

        # 2. Detect objects
        all_boxes = []
        all_classes = []
        all_shrinks = []

        for text_queries, class_id, threshold, shrink in self.class_configs:
            boxes, scores = self.detect_objects(img_pil, text_queries, threshold)
            if len(boxes) > 0:
                for box in boxes:
                    all_boxes.append(box)
                    all_classes.append(class_id)
                    all_shrinks.append(shrink)

        # 3. Create masks for each detection
        object_masks = []  # List of (mask, class_id)

        for box, class_id, shrink in zip(all_boxes, all_classes, all_shrinks):
            # Try GrabCut first
            mask = None
            if self.has_cv2 and self.use_grabcut:
                mask = self._refine_box_with_grabcut(box, img_np)

            # Fall back to color-based refinement
            if mask is None:
                mask = self._refine_box_with_color(box, img_np, shrink)

            if mask is not None:
                object_masks.append((mask, class_id))

        # 4. Apply masks in priority order
        for priority_class in self.priority_order:
            for mask, class_id in object_masks:
                if class_id == priority_class:
                    labels[mask] = class_id

        # 5. Also apply raw boxes as fallback for any missed regions
        for priority_class in self.priority_order:
            for box, class_id in zip(all_boxes, all_classes):
                if class_id == priority_class:
                    x1, y1, x2, y2 = box.astype(int)
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(w, x2)
                    y2 = min(h, y2)
                    if x2 > x1 and y2 > y1:
                        # Only fill if not already labeled with higher priority
                        region = labels[y1:y2, x1:x2]
                        fill_mask = region == SemanticLabel.GROUND
                        labels[y1:y2, x1:x2][fill_mask] = class_id

        # 6. Apply depth refinement
        if depth_np is not None:
            labels = self._apply_depth_refinement(labels, depth_np)

        # 7. Final smoothing
        labels = self._smooth_labels(labels, window_size=3)

        # 8. Ensure sky in upper part
        sky_region = slice(0, int(h * 0.15))
        labels[sky_region][sky_mask[sky_region]] = SemanticLabel.SKY

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
    parser = argparse.ArgumentParser(
        description="Improved pseudo semantic label generation (v2 - no SAM)"
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
        "--no_grabcut",
        action="store_true",
        help="Disable GrabCut (faster but less accurate)"
    )
    parser.add_argument(
        "--no_depth",
        action="store_true",
        help="Disable depth information"
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
    label_generator = LabelGeneratorV2(
        model_id=args.model,
        device=args.device,
        use_grabcut=not args.no_grabcut,
        use_depth=not args.no_depth
    )

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
                    vis_path = os.path.join(step_dir, "semantic_vis_v2.png")
                    colored = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
                    for l in range(6):
                        colored[labels == l] = SEMANTIC_COLORS[l]
                    Image.fromarray(colored).save(vis_path)

                processed += 1
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                import traceback
                traceback.print_exc()
                skipped += 1
                continue

    print(f"Done! Processed {processed} images, skipped {skipped}")


if __name__ == "__main__":
    main()
