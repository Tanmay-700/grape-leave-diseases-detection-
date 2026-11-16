"""
End-to-End Inference Pipeline
=============================
Full CLI Pipeline: Field Photo → SAM 2 Segmentation → Dual-Model Ensemble → Grad-CAM

Usage:
    python pipeline.py path/to/leaf.jpg
    python pipeline.py path/to/leaf.jpg --crop grape
    python pipeline.py path/to/leaf.jpg --point 300,400
"""
import os
import argparse
import time
import numpy as np
from PIL import Image

from config import OUTPUTS_DIR
from segment import segment_leaf_sam2, segment_leaf_auto, crop_to_mask
from classify import classify_crop_ensemble, classify_auto_detect
from gradcam import generate_model_gradcam


def run_pipeline(image_path, crop=None, point=None, box=None, device="cpu",
                 save_outputs=True, output_dir=None):
    if output_dir is None:
        output_dir = OUTPUTS_DIR
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    total_start = time.time()

    print("=" * 65)
    print("  LEAF DISEASE CLASSIFICATION PIPELINE (CLI)")
    print(f"  Input: {image_path}")
    print("=" * 65)

    # ---- Step 1: Load Image ----
    print("\n[1/4] Loading image...")
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    print(f"  Resolution: {w}x{h}")

    # ---- Step 2: SAM 2 Segmentation ----
    print("\n[2/4] Segmenting leaf with Meta SAM 2...")
    seg_start = time.time()

    if point is not None:
        point_coords = np.array([point])
        point_labels = np.array([1])
        segmented, mask = segment_leaf_sam2(
            image, point_coords=point_coords, point_labels=point_labels, device=device
        )
        print(f"  Mode: Point prompt at ({point[0]}, {point[1]})")
    elif box is not None:
        box_np = np.array(box)
        segmented, mask = segment_leaf_sam2(image, box=box_np, device=device)
        print(f"  Mode: Box prompt [{box[0]}, {box[1]}, {box[2]}, {box[3]}]")
    else:
        segmented, mask = segment_leaf_auto(image, device=device)
        print("  Mode: Automatic Foliage Detection")

    seg_time = time.time() - seg_start
    cropped = crop_to_mask(segmented, mask, padding=15)
    print(f"  Segmentation time: {seg_time:.2f}s")

    if save_outputs:
        segmented.save(os.path.join(output_dir, f"{base_name}_segmented.png"))
        cropped.save(os.path.join(output_dir, f"{base_name}_cropped.png"))

    # ---- Step 3: Dual-Model Classification ----
    print("\n[3/4] Classifying with Dual-Model Ensemble...")
    cls_start = time.time()

    if crop and "grape" in crop.lower():
        result = classify_crop_ensemble(cropped, crop="grape")
        detected_crop = "Grape"
    elif crop and "apple" in crop.lower():
        result = classify_crop_ensemble(cropped, crop="apple")
        detected_crop = "Apple"
    else:
        result = classify_auto_detect(cropped)
        detected_crop = result.get("auto_crop", "Detected")

    cls_time = time.time() - cls_start

    consensus_pred = result.get("consensus_label", "N/A")
    consensus_conf = result.get("consensus_confidence", 0.0)
    m1 = result.get("model_1") or {}
    m2 = result.get("model_2") or {}
    is_agreement = (m1.get("predicted_label") == m2.get("predicted_label")) if (m1 and m2) else False

    print(f"  Crop: {detected_crop.title()}")
    print(f"  Consensus Diagnosis: {consensus_pred}")
    print(f"  Ensemble Confidence: {consensus_conf*100:.1f}%")
    print(f"  Model Agreement    : {'YES (High Confidence)' if is_agreement else 'NO (Conflicting)'}")
    if m1:
        print(f"    - Model 1 ({m1.get('arch', 'N/A').upper()}): {m1.get('predicted_label')} ({m1.get('confidence', 0)*100:.1f}%)")
    if m2:
        print(f"    - Model 2 ({m2.get('arch', 'N/A').upper()}): {m2.get('predicted_label')} ({m2.get('confidence', 0)*100:.1f}%)")
    print(f"  Classification time: {cls_time:.2f}s")

    # ---- Step 4: Dual Grad-CAM ----
    print("\n[4/4] Generating Side-by-Side Grad-CAM heatmaps...")
    cam_start = time.time()

    if m1.get("model_key"):
        try:
            cam1, _, _ = generate_model_gradcam(cropped, m1["model_key"], alpha=0.5)
            if cam1 and save_outputs:
                cam1.save(os.path.join(output_dir, f"{base_name}_gradcam_model1.png"))
        except Exception as e:
            print(f"  [WARN] Grad-CAM 1 failed: {e}")

    if m2.get("model_key"):
        try:
            cam2, _, _ = generate_model_gradcam(cropped, m2["model_key"], alpha=0.5)
            if cam2 and save_outputs:
                cam2.save(os.path.join(output_dir, f"{base_name}_gradcam_model2.png"))
        except Exception as e:
            print(f"  [WARN] Grad-CAM 2 failed: {e}")

    cam_time = time.time() - cam_start
    print(f"  Grad-CAM time: {cam_time:.2f}s")

    total_time = time.time() - total_start
    print(f"\n{'='*65}")
    print("  PIPELINE COMPLETE")
    print(f"  Total time: {total_time:.2f}s")
    if save_outputs:
        print(f"  Outputs saved to: {output_dir}")
    print(f"{'='*65}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Leaf Disease Classification Pipeline")
    parser.add_argument("image", help="Path to input field photograph")
    parser.add_argument("--crop", type=str, default=None, choices=["grape", "apple"],
                        help="Crop selection (optional, default: auto-detect)")
    parser.add_argument("--point", type=str, default=None,
                        help="Point prompt as 'x,y' (e.g., --point 300,400)")
    parser.add_argument("--box", type=str, default=None,
                        help="Box prompt as 'x1,y1,x2,y2' (e.g., --box 100,100,500,400)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    point = None
    if args.point:
        x, y = map(int, args.point.split(","))
        point = (x, y)

    box = None
    if args.box:
        box = tuple(map(int, args.box.split(",")))

    run_pipeline(
        args.image,
        crop=args.crop,
        point=point,
        box=box,
        device=args.device,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
