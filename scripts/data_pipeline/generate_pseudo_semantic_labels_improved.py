# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Improved script to generate pseudo semantic labels for UAV data using:
# Grounding DINO + SAM (Segment Anything Model) + improved heuristics
#
# Install:
# pip install transformers torch torchvision pillow numpy tqdm accelerate
# pip install git+https://github.com/facebookresearch/segment-anything.git
# Or install mobile SAM for faster inference:
# pip install git+https://github.com/ChaoningZhang/MobileSAM.git

import argparse
import os
from typing import List, Tuple, Optional
from tqdm import tqdm
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
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


class SAMWrapper:
    """Wrapper for Segment Anything Model (SAM) or MobileSAM."""

    def __init__(self, model_type: str = "mobile", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        self.predictor = None
        self._load_model()

    def _load_model(self):
        """Load SAM/MobileSAM model."""
        try:
            if self.model_type == "mobile":
                # Try MobileSAM first (faster)
                try:
                    from mobile_sam import sam_model_registry, SamPredictor
                    print("Loading MobileSAM...")
                    sam_checkpoint = "mobile_sam.pt"
                    model_type_mobile = "vit_t"

                    # Download if not present
                    if not os.path.exists(sam_checkpoint):
                        print("Downloading MobileSAM checkpoint...")
                        import urllib.request
                        url = "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
                        try:
                            urllib.request.urlretrieve(url, sam_checkpoint)
                        except:
                            print("MobileSAM download failed, trying regular SAM...")
                            raise FileNotFoundError("MobileSAM not available")

                    mobile_sam = sam_model_registry[model_type_mobile](checkpoint=sam_checkpoint)
                    mobile_sam.to(device=self.device)
                    mobile_sam.eval()
                    self.predictor = SamPredictor(mobile_sam)
                    print("MobileSAM loaded successfully!")
                    return
                except Exception as e:
                    print(f"MobileSAM failed: {e}, trying regular SAM...")

            # Fall back to regular SAM
            print("Loading SAM (vit_b)...")
            from segment_anything import sam_model_registry, SamPredictor

            sam_checkpoint = "sam_vit_b_01ec64.pth"
            if not os.path.exists(sam_checkpoint):
                print("Downloading SAM checkpoint...")
                import urllib.request
                url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
                try:
                    urllib.request.urlretrieve(url, sam_checkpoint)
                except Exception as e:
                    print(f"SAM download failed: {e}")
                    print("Will use bounding box mode with post-processing instead.")
                    self.predictor = None
                    return

            sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
            sam.to(device=self.device)
            self.predictor = SamPredictor(sam)
            print("SAM loaded successfully!")

        except Exception as e:
            print(f"Failed to load SAM: {e}")
            print("Will use bounding box mode with post-processing instead.")
            self.predictor = None

    def set_image(self, img_np: np.ndarray):
        """Set image for prediction."""
        if self.predictor is not None:
            self.predictor.set_image(img_np)

    def segment_box(self, box: np.ndarray) -> Optional[np.ndarray]:
        """Segment object within bounding box."""
        if self.predictor is None:
            return None

        try:
            masks, scores, _ = self.predictor.predict(
                box=box,
                multimask_output=False
            )
            return masks[0]  # Return first mask
        except Exception as e:
            print(f"SAM segmentation failed: {e}")
            return None


class ImprovedLabelGenerator:
    """Generate improved pseudo-labels using Grounding DINO + SAM + better heuristics."""

    def __init__(
        self,
        dino_model_id: str = "IDEA-Research/grounding-dino-base",
        sam_model_type: str = "mobile",
        device=None,
        use_sam: bool = True,
        use_depth: bool = True
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_sam = use_sam
        self.use_depth = use_depth

        print(f"Loading Grounding DINO {dino_model_id} on {self.device}...")
        self.processor = AutoProcessor.from_pretrained(dino_model_id)
        self.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_model_id)
        self.dino_model.to(self.device)
        self.dino_model.eval()

        # Category mapping with improved queries and thresholds
        self.class_configs = [
            # (text_queries, class_id, box_threshold, box_expansion)
            ("human, person, people, man, woman, pedestrian",
             SemanticLabel.HUMAN, 0.30, 0.05),
            ("car, vehicle, automobile, sedan, truck, bus, van",
             SemanticLabel.VEHICLE, 0.30, 0.05),
            ("building, house, wall, structure, tower, facade",
             SemanticLabel.BUILDING, 0.25, 0.0),
            ("swing, playground, slide, seesaw, climber, play equipment",
             SemanticLabel.OBSTACLE, 0.30, 0.05),
            ("statue, angel, sculpture, monument",
             SemanticLabel.OBSTACLE, 0.30, 0.05),
            ("tree, plant, bush, vegetation, foliage",
             SemanticLabel.OBSTACLE, 0.25, 0.02),
        ]

        # SAM wrapper for instance segmentation
        self.sam = None
        if self.use_sam:
            self.sam = SAMWrapper(model_type=sam_model_type, device=self.device)

        # Class priority (higher = more important, overwrites lower)
        self.priority_order = [
            SemanticLabel.HUMAN,
            SemanticLabel.VEHICLE,
            SemanticLabel.OBSTACLE,
            SemanticLabel.BUILDING,
            SemanticLabel.GROUND,
            SemanticLabel.SKY
        ]

        # Store current image for SAM
        self.current_img_np = None

    def detect_objects(
        self,
        img_pil,
        text_queries: str,
        box_threshold: float = 0.3
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Detect objects with improved NMS."""
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
                    # Convert from normalized to pixel coordinates
                    w, h = img_pil.size
                    boxes = boxes * torch.tensor([w, h, w, h], device=self.device)
                    # Convert from cxcywh to xyxy
                    boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
                    boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
                    boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
                    boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
                    boxes = boxes.cpu().numpy()
                    scores = scores.cpu().numpy()

            # Apply NMS to remove overlapping boxes
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
        """Non-maximum suppression to remove overlapping boxes."""
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

    def _expand_box(
        self,
        box: np.ndarray,
        expansion: float,
        img_w: int,
        img_h: int
    ) -> np.ndarray:
        """Expand box slightly for better SAM results."""
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1

        x1 = max(0, x1 - w * expansion)
        y1 = max(0, y1 - h * expansion)
        x2 = min(img_w, x2 + w * expansion)
        y2 = min(img_h, y2 + h * expansion)

        return np.array([x1, y1, x2, y2])

    def _detect_sky_improved(
        self,
        img_np: np.ndarray,
        depth_np: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Improved sky detection using color + gradient + depth."""
        h, w = img_np.shape[:2]
        sky_mask = np.zeros((h, w), dtype=bool)

        # Only process upper 2/3 of image for sky
        sky_region = slice(0, int(h * 0.7))
        img_upper = img_np[sky_region, :, :]

        # Convert to float
        img_float = img_upper.astype(np.float32)
        r = img_float[:, :, 0]
        g = img_float[:, :, 1]
        b = img_float[:, :, 2]

        # 1. Blue dominance
        blue_diff = b - np.maximum(r, g)
        blue_ratio = b / (np.maximum(r, g) + 1e-6)

        # 2. Brightness check (sky is usually bright)
        brightness = (r + g + b) / 3

        # 3. Saturation check (sky is not too saturated)
        max_rgb = np.maximum(np.maximum(r, g), b)
        min_rgb = np.minimum(np.minimum(r, g), b)
        saturation = (max_rgb - min_rgb) / (max_rgb + 1e-6)

        # 4. Color similarity to sky blue
        sky_r, sky_g, sky_b = 135, 206, 235
        color_dist = np.sqrt(
            (r - sky_r)**2 +
            (g - sky_g)**2 +
            (b - sky_b)**2
        )

        # Combine heuristics
        sky_candidate = (
            (blue_diff > 10) &
            (blue_ratio > 1.1) &
            (brightness > 80) &
            (brightness < 250) &
            (saturation < 0.5) &
            (color_dist < 100)
        )

        # 5. If depth is available, sky should be far/inf
        if depth_np is not None and self.use_depth:
            depth_upper = depth_np[sky_region, :]
            # Sky typically has maximum or very high depth
            depth_mean = np.mean(depth_upper)
            depth_std = np.std(depth_upper)
            sky_depth = depth_upper > (depth_mean + 0.5 * depth_std)
            sky_candidate = sky_candidate | sky_depth

        sky_mask[sky_region, :] = sky_candidate

        # Post-process: morphological operations to clean up
        sky_mask = self._morphological_cleanup(sky_mask, mode="sky")

        return sky_mask

    def _detect_ground_improved(
        self,
        img_np: np.ndarray,
        depth_np: Optional[np.ndarray] = None,
        existing_labels: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Improved ground detection."""
        h, w = img_np.shape[:2]
        ground_mask = np.zeros((h, w), dtype=bool)

        # Process lower half
        ground_region = slice(int(h * 0.4), h)
        img_lower = img_np[ground_region, :, :]

        img_float = img_lower.astype(np.float32)
        r = img_float[:, :, 0]
        g = img_float[:, :, 1]
        b = img_float[:, :, 2]

        # Green dominance (for grass, etc.)
        green_diff = g - np.maximum(r, b)
        green_ratio = g / (np.maximum(r, b) + 1e-6)

        # Ground colors are typically darker/less saturated
        brightness = (r + g + b) / 3

        # Ground candidate
        ground_candidate = (
            (green_diff > -30) &  # Not too anti-green
            (brightness > 30) &
            (brightness < 200)
        )

        # If depth is available, ground is usually close/medium depth
        if depth_np is not None and self.use_depth:
            depth_lower = depth_np[ground_region, :]
            # Ground has moderate depth (not too close, not too far)
            valid_depth = np.isfinite(depth_lower)
            if np.any(valid_depth):
                depth_mean = np.mean(depth_lower[valid_depth])
                ground_depth = (depth_lower > depth_mean * 0.3) & (depth_lower < depth_mean * 2.0)
                ground_candidate = ground_candidate | (ground_depth & valid_depth)

        ground_mask[ground_region, :] = ground_candidate

        # Avoid areas already labeled as something else
        if existing_labels is not None:
            ground_mask = ground_mask & (existing_labels == SemanticLabel.GROUND)

        return ground_mask

    def _morphological_cleanup(
        self,
        mask: np.ndarray,
        mode: str = "default"
    ) -> np.ndarray:
        """Apply morphological operations to clean up mask."""
        try:
            from scipy import ndimage

            if mode == "sky":
                # Remove small noise, keep large connected regions
                mask = ndimage.binary_opening(mask, iterations=2)
                # Fill holes
                mask = ndimage.binary_closing(mask, iterations=3)

                # Keep only largest connected component for sky
                labeled, num_features = ndimage.label(mask)
                if num_features > 0:
                    sizes = ndimage.sum(mask, labeled, range(num_features + 1))
                    largest_label = sizes.argmax()
                    mask = labeled == largest_label
            else:
                # General cleanup
                mask = ndimage.binary_opening(mask, iterations=1)
                mask = ndimage.binary_closing(mask, iterations=1)

        except ImportError:
            # Fallback: simple smoothing with convolution
            kernel = np.ones((3, 3), np.float32) / 9
            for _ in range(2):
                mask_padded = np.pad(mask.astype(np.float32), 1, mode='edge')
                mask_smooth = np.zeros_like(mask_padded)
                for i in range(1, mask_padded.shape[0] - 1):
                    for j in range(1, mask_padded.shape[1] - 1):
                        mask_smooth[i, j] = np.sum(mask_padded[i-1:i+2, j-1:j+2] * kernel)
                mask = mask_smooth[1:-1, 1:-1] > 0.5

        return mask

    def _fill_boundary_with_sam(
        self,
        labels: np.ndarray,
        img_np: np.ndarray,
        boxes: List[np.ndarray],
        classes: List[int],
        img_w: int,
        img_h: int
    ) -> np.ndarray:
        """Fill labels using SAM for precise segmentation."""
        if self.sam is None or self.sam.predictor is None:
            return labels

        try:
            # Set image for SAM
            self.sam.set_image(img_np)

            # Process by priority
            for priority_class in self.priority_order:
                for box, class_id in zip(boxes, classes):
                    if class_id != priority_class:
                        continue

                    # Expand box slightly for better SAM results
                    expansion = next(
                        (cfg[3] for cfg in self.class_configs if cfg[1] == class_id),
                        0.05
                    )
                    box_expanded = self._expand_box(box, expansion, img_w, img_h)

                    # Get segmentation mask from SAM
                    mask = self.sam.segment_box(box_expanded)

                    if mask is not None:
                        # Resize mask if needed
                        if mask.shape != labels.shape:
                            mask_pil = Image.fromarray(mask.astype(np.uint8))
                            mask_pil = mask_pil.resize((img_w, img_h), Image.NEAREST)
                            mask = np.array(mask_pil).astype(bool)

                        # Apply mask to labels
                        labels[mask] = class_id

        except Exception as e:
            print(f"SAM filling failed: {e}, falling back to boxes")

        return labels

    def _fill_boundary_boxes(
        self,
        labels: np.ndarray,
        boxes: List[np.ndarray],
        classes: List[int],
        img_w: int,
        img_h: int
    ) -> np.ndarray:
        """Fill labels using bounding boxes with improved edge handling."""
        # Process by priority
        for priority_class in self.priority_order:
            for box, class_id in zip(boxes, classes):
                if class_id != priority_class:
                    continue

                x_min, y_min, x_max, y_max = [int(round(c)) for c in box]
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(img_w, x_max)
                y_max = min(img_h, y_max)

                if x_max > x_min and y_max > y_min:
                    # Fill center region
                    labels[y_min:y_max, x_min:x_max] = class_id

        return labels

    def _smooth_edges(
        self,
        labels: np.ndarray
    ) -> np.ndarray:
        """Smooth label edges using majority voting."""
        try:
            from scipy import ndimage

            h, w = labels.shape
            labels_smooth = labels.copy()

            # For each class, apply majority voting
            for class_id in range(6):
                mask = labels == class_id
                # Dilate slightly and then vote
                mask_dilated = ndimage.binary_dilation(mask, iterations=1)
                labels_smooth[mask_dilated] = class_id

            return labels_smooth

        except ImportError:
            # Simple convolution fallback
            return labels

    def _refine_with_depth(
        self,
        labels: np.ndarray,
        depth_np: np.ndarray
    ) -> np.ndarray:
        """Refine labels using depth information."""
        if not self.use_depth or depth_np is None:
            return labels

        h, w = labels.shape
        labels_refined = labels.copy()

        # Create regions based on depth discontinuities
        depth_grad = np.abs(np.gradient(depth_np.astype(np.float32)))
        depth_grad_mag = np.sqrt(depth_grad[0]**2 + depth_grad[1]**2)

        # High gradient means depth discontinuity (likely object boundary)
        depth_edge = depth_grad_mag > np.percentile(depth_grad_mag, 70)

        # Smooth near edges - don't let labels cross depth discontinuities
        labels_refined = self._edge_aware_smoothing(labels_refined, depth_edge)

        return labels_refined

    def _edge_aware_smoothing(
        self,
        labels: np.ndarray,
        edge_mask: np.ndarray
    ) -> np.ndarray:
        """Smooth labels while respecting edges."""
        h, w = labels.shape
        labels_smooth = labels.copy()

        # Simple 3x3 majority vote, but not across edges
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                # If not at an edge, can smooth
                if not edge_mask[i, j]:
                    patch = labels[i-1:i+2, j-1:j+2]
                    # Find most common label in patch
                    values, counts = np.unique(patch, return_counts=True)
                    labels_smooth[i, j] = values[np.argmax(counts)]

        return labels_smooth

    @torch.no_grad()
    def __call__(
        self,
        img_path: str,
        depth_path: Optional[str] = None,
        img_size: Tuple[int, int] = (320, 512)
    ) -> np.ndarray:
        """Generate improved semantic label for an image."""
        img_pil = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img_pil.size
        img_pil = img_pil.resize(img_size, Image.LANCZOS)  # Use LANCZOS for better quality
        w, h = img_pil.size
        img_np = np.array(img_pil)

        # Load depth if available
        depth_np = None
        if depth_path is not None and os.path.exists(depth_path) and self.use_depth:
            try:
                depth_np = np.load(depth_path)
                depth_pil = Image.fromarray(depth_np)
                depth_pil = depth_pil.resize(img_size, Image.NEAREST)
                depth_np = np.array(depth_pil)
            except Exception as e:
                print(f"Could not load depth: {e}")
                depth_np = None

        # Initialize labels
        labels = np.full((h, w), SemanticLabel.GROUND, dtype=np.uint8)

        # 1. Sky detection (improved)
        sky_mask = self._detect_sky_improved(img_np, depth_np)
        labels[sky_mask] = SemanticLabel.SKY

        # 2. Detect objects using Grounding DINO
        all_boxes = []
        all_classes = []

        for text_queries, class_id, threshold, _ in self.class_configs:
            boxes, scores = self.detect_objects(img_pil, text_queries, threshold)
            if len(boxes) > 0:
                for box in boxes:
                    all_boxes.append(box)
                    all_classes.append(class_id)

        # 3. Fill labels
        if self.use_sam and self.sam is not None and self.sam.predictor is not None:
            # Try SAM first for precise segmentation
            labels = self._fill_boundary_with_sam(labels, img_np, all_boxes, all_classes, w, h)

        # Always fill with boxes as fallback/SAM supplement
        labels = self._fill_boundary_boxes(labels, all_boxes, all_classes, w, h)

        # 4. Refine ground detection
        ground_mask = self._detect_ground_improved(img_np, depth_np, labels)
        # Only apply ground where not already labeled
        ground_region = (labels == SemanticLabel.GROUND)
        labels[ground_region & ~ground_mask] = SemanticLabel.GROUND

        # 5. Refine with depth if available
        if depth_np is not None:
            labels = self._refine_with_depth(labels, depth_np)

        # 6. Final edge smoothing
        labels = self._smooth_edges(labels)

        # Ensure sky remains in upper region
        labels[:int(h*0.2), :][sky_mask[:int(h*0.2), :]] = SemanticLabel.SKY

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
        description="Improved pseudo semantic label generation using Grounding DINO + SAM"
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
        "--dino_model",
        type=str,
        default="IDEA-Research/grounding-dino-base",
        help="Grounding DINO model (base is better than tiny)"
    )
    parser.add_argument(
        "--sam_model",
        type=str,
        default="mobile",
        choices=["mobile", "sam"],
        help="SAM model type (mobile = MobileSAM, sam = SAM vit_b)"
    )
    parser.add_argument(
        "--no_sam",
        action="store_true",
        help="Disable SAM (use boxes only)"
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
    label_generator = ImprovedLabelGenerator(
        dino_model_id=args.dino_model,
        sam_model_type=args.sam_model,
        device=args.device,
        use_sam=not args.no_sam,
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
                    vis_path = os.path.join(step_dir, "semantic_vis.png")
                    # Create colored label
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
