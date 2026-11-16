"""
SAM 2 Leaf Segmentation Module
================================
Uses Meta's Segment Anything Model 2 (SAM 2.1 Hiera Small) to isolate
the diseased leaf / foliage canopy from cluttered agricultural field backgrounds
(soil, dirt, sky, human hands, greenhouse frames, grass).

Supports:
- Automatic mode: Bounding-box & centroid foliage guidance (preserves all necrotic lesions)
- Point prompt: User clicks on or specifies coordinates of a specific leaf
- Box prompt: User specifies [x1, y1, x2, y2] bounding box
- Fallback: Graceful OpenCV GrabCut if weights are missing
"""
import os
import time
import cv2
import numpy as np
from PIL import Image

# Global SAM 2 predictor cache
_SAM2_AVAILABLE = False
_sam2_predictor = None
_sam2_model_info = None

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    _SAM2_AVAILABLE = True
except ImportError:
    print("[WARNING] sam2 package not found in Python environment.")


def load_sam2_model(checkpoint_path=None, config=None, device="cpu"):
    """
    Load SAM 2 model into memory once and cache it.
    """
    global _sam2_predictor, _sam2_model_info

    if not _SAM2_AVAILABLE:
        print("[WARNING] SAM 2 is not installed. Using fallback.")
        return None

    if _sam2_predictor is not None:
        return _sam2_predictor

    if checkpoint_path is None or config is None:
        from config import SAM2_CHECKPOINT, SAM2_CONFIG
        checkpoint_path = SAM2_CHECKPOINT
        config = SAM2_CONFIG

    if not os.path.isfile(checkpoint_path):
        print(f"[WARNING] SAM 2 checkpoint not found at: {checkpoint_path}")
        return None

    try:
        t0 = time.time()
        sam2_model = build_sam2(config, checkpoint_path, device=device)
        total_params = sum(p.numel() for p in sam2_model.parameters())
        _sam2_predictor = SAM2ImagePredictor(sam2_model)
        _sam2_model_info = {
            "name": "Meta SAM 2.1 Hiera Small",
            "checkpoint": checkpoint_path,
            "params": total_params,
            "device": device,
            "load_time": time.time() - t0,
        }
        print(f"[INFO] Meta SAM 2 loaded successfully: {total_params:,} parameters on {device} ({time.time()-t0:.2f}s)")
        return _sam2_predictor
    except Exception as e:
        print(f"[WARNING] Failed to load SAM 2: {e}")
        return None


def get_sam2_info():
    """Return verified metadata about the active SAM 2 engine."""
    predictor = load_sam2_model()
    if predictor is not None and _sam2_model_info is not None:
        return {
            "active": True,
            "engine": "Meta SAM 2.1 Hiera Small",
            "params": f"{_sam2_model_info['params']:,}",
            "checkpoint": os.path.basename(_sam2_model_info["checkpoint"]),
            "device": _sam2_model_info["device"],
        }
    return {
        "active": False,
        "engine": "OpenCV Fallback (GrabCut)",
        "params": "N/A",
        "checkpoint": "None",
        "device": "cpu",
    }


def segment_leaf_sam2(image, point_coords=None, point_labels=None,
                      box=None, device="cpu"):
    """
    Segment diseased leaf / plant foliage using Meta SAM 2.1.
    Guarantees that necrotic brown, yellow, and red disease spots are preserved
    without being cut off by artificial background points.
    
    Args:
        image: PIL Image or numpy array (RGB)
        point_coords: Optional Nx2 numpy array of point prompts [[x, y], ...]
        point_labels: Optional N-length array (1=foreground, 0=background)
        box: Optional [x1, y1, x2, y2] bounding box prompt
        device: 'cpu' or 'cuda'
    
    Returns:
        tuple: (segmented_image: PIL Image, mask: numpy uint8 array 0/255)
    """
    predictor = load_sam2_model(device=device)

    if isinstance(image, Image.Image):
        img_rgb = np.array(image.convert("RGB"))
    else:
        img_rgb = image.copy()

    h, w = img_rgb.shape[:2]
    total_pixels = h * w

    if predictor is None:
        print("[INFO] SAM 2 predictor unavailable, using fallback segmentation.")
        return _fallback_segment(img_rgb)

    t0 = time.time()
    # Feed image into SAM 2 Hiera vision transformer backbone
    predictor.set_image(img_rgb)

    if point_coords is not None or box is not None:
        # User-guided prompt (point click or bounding box)
        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=True,
        )
        best_idx = np.argmax(scores)
        mask = (masks[best_idx] > 0).astype(np.uint8) * 255
    else:
        # Automatic foliage guidance:
        # Detect plant foliage across green, chlorotic yellow, and necrotic brown
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        green = cv2.inRange(hsv, (18, 20, 20), (95, 255, 255))
        yellow_brown = cv2.inRange(hsv, (8, 25, 25), (25, 255, 255))
        r = img_rgb[:, :, 0].astype(np.float32)
        g = img_rgb[:, :, 1].astype(np.float32)
        b = img_rgb[:, :, 2].astype(np.float32)
        exg = ((2 * g - r - b) > 0).astype(np.uint8) * 255

        foliage = cv2.bitwise_or(green, yellow_brown)
        foliage = cv2.bitwise_or(foliage, exg)

        ys, xs = np.where(foliage > 0)

        if len(xs) > int(total_pixels * 0.015):
            # Foliage cluster found: compute bounding box and centroid
            x1 = max(0, int(np.percentile(xs, 1)) - 6)
            y1 = max(0, int(np.percentile(ys, 1)) - 6)
            x2 = min(w, int(np.percentile(xs, 99)) + 6)
            y2 = min(h, int(np.percentile(ys, 99)) + 6)
            auto_box = np.array([x1, y1, x2, y2])

            cx = int(np.median(xs))
            cy = int(np.median(ys))
            auto_pts = np.array([[cx, cy]])
            auto_lbls = np.array([1])

            # Predict using box + centroid without any artificial negative prompts
            masks, scores, _ = predictor.predict(
                point_coords=auto_pts,
                point_labels=auto_lbls,
                box=auto_box,
                multimask_output=False,
            )
            mask = (masks[0] > 0).astype(np.uint8) * 255
        else:
            # Full-frame leaf: use 5-point positive prompt
            pts = np.array([
                [int(w * 0.5), int(h * 0.5)],
                [int(w * 0.3), int(h * 0.3)],
                [int(w * 0.7), int(h * 0.3)],
                [int(w * 0.3), int(h * 0.7)],
                [int(w * 0.7), int(h * 0.7)],
            ])
            lbls = np.array([1, 1, 1, 1, 1])
            masks, scores, _ = predictor.predict(
                point_coords=pts,
                point_labels=lbls,
                multimask_output=False,
            )
            mask = (masks[0] > 0).astype(np.uint8) * 255

    # Gentle morphological cleanup: close small 3x3 gaps without collapsing arches
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Filter out tiny noise flecks (area < 0.1% of total image)
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = total_pixels * 0.001
    filtered = np.zeros_like(clean)
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            cv2.drawContours(filtered, [c], -1, 255, thickness=cv2.FILLED)

    if np.sum(filtered > 0) == 0:
        filtered = clean if np.sum(clean > 0) > 0 else np.ones((h, w), dtype=np.uint8) * 255

    segmented = img_rgb.copy()
    segmented[filtered == 0] = [255, 255, 255]

    coverage = np.mean(filtered > 0) * 100
    elapsed = time.time() - t0
    print(f"[SAM 2 ACTIVE] Segmented leaf in {elapsed:.2f}s (coverage: {coverage:.1f}%)")

    return Image.fromarray(segmented), filtered


def segment_leaf_auto(image, device="cpu", min_area_ratio=0.05, max_area_ratio=0.85):
    """
    Automatic foliage segmentation using Meta SAM 2.1.
    """
    return segment_leaf_sam2(image, device=device)


def _fallback_segment(img_rgb):
    """
    Graceful fallback segmentation using GrabCut if SAM 2 is missing.
    """
    h, w = img_rgb.shape[:2]
    try:
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        rect = (int(w * 0.05), int(h * 0.05), int(w * 0.90), int(h * 0.90))
        mask_gc = np.zeros((h, w), np.uint8)
        cv2.grabCut(img_rgb, mask_gc, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        mask = np.where((mask_gc == 2) | (mask_gc == 0), 0, 255).astype(np.uint8)
    except Exception:
        mask = np.ones((h, w), dtype=np.uint8) * 255

    if np.mean(mask > 0) < 0.05:
        mask = np.ones((h, w), dtype=np.uint8) * 255

    segmented = img_rgb.copy()
    segmented[mask == 0] = [255, 255, 255]
    return Image.fromarray(segmented), mask


def crop_to_mask(image, mask, padding=15):
    """
    Crop image to the bounding box of the mask with padding.
    """
    if isinstance(image, Image.Image):
        img = np.array(image)
    else:
        img = image.copy()

    h, w = mask.shape[:2]
    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        return Image.fromarray(img)

    x1 = max(0, xs.min() - padding)
    y1 = max(0, ys.min() - padding)
    x2 = min(w, xs.max() + padding)
    y2 = min(h, ys.max() + padding)

    cropped = img[y1:y2, x1:x2]
    return Image.fromarray(cropped)
