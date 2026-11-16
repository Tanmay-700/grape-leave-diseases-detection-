"""
FastAPI REST API Backend for Leaf Disease Classifier
=====================================================
Provides headless REST endpoints for:
- Meta SAM 2.1 Leaf Segmentation
- Dual-Model Ensemble Classification (Grape & Apple)
- Side-by-Side Grad-CAM Heatmaps

Run with:
    uvicorn api:app --host 0.0.0.0 --port 8080 --reload
Or:
    python api.py
"""
import io
import base64
import numpy as np
from PIL import Image
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from segment import segment_leaf_sam2, segment_leaf_auto, crop_to_mask, get_sam2_info
from classify import classify_crop_ensemble, classify_auto_detect
from gradcam import generate_model_gradcam

app = FastAPI(
    title="Leaf Disease Classification & Segmentation API",
    description="Dual-Model Ensemble + Meta SAM 2 + Grad-CAM Explainability Backend",
    version="2.0.0",
)

# Enable CORS for external frontends (React, Vue, mobile apps, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = "cpu"


def pil_to_base64(img: Image.Image) -> str:
    """Convert PIL image to base64 string."""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


@app.get("/")
def root():
    return {
        "service": "Leaf Disease Classification & SAM 2 API Backend",
        "status": "online",
        "docs_url": "/docs",
        "endpoints": {
            "/predict": "POST - Upload leaf image for segmentation, classification, and Grad-CAM",
            "/health": "GET - Model and SAM 2 loading status",
        },
    }


@app.get("/health")
def health_check():
    sam_info = get_sam2_info()
    return {
        "status": "healthy",
        "sam2": sam_info,
        "models": {
            "grape_efficientnet_b2": "models/grape_efficientnet_b2.pth",
            "grape_resnet50": "models/grape_resnet50.pth",
            "apple_efficientnet_b3": "models/apple_efficientnet_b3.pth",
            "apple_resnet50": "models/apple_resnet50.pth",
        },
    }


@app.post("/predict")
async def predict_leaf(
    file: UploadFile = File(...),
    crop_mode: str = Form("auto"),  # 'auto', 'grape', or 'apple'
    point_x: Optional[int] = Form(None),
    point_y: Optional[int] = Form(None),
    gradcam_alpha: float = Form(0.55),
):
    """
    Complete end-to-end inference pipeline:
    1. Read input image
    2. Segment with Meta SAM 2
    3. Crop to segmented leaf
    4. Dual-model ensemble classification
    5. Side-by-side Grad-CAM heatmaps
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File uploaded must be an image.")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")

    # 1. SAM 2 Segmentation
    try:
        if point_x is not None and point_y is not None:
            point_coords = np.array([[int(point_x), int(point_y)]])
            point_labels = np.array([1])
            segmented, mask = segment_leaf_sam2(
                image, point_coords=point_coords, point_labels=point_labels, device=DEVICE
            )
        else:
            segmented, mask = segment_leaf_auto(image, device=DEVICE)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {str(e)}")

    cropped = crop_to_mask(segmented, mask, padding=15)

    # 2. Dual-Model Classification & Ensemble
    try:
        mode_str = crop_mode.lower()
        if "auto" in mode_str:
            cls_result = classify_auto_detect(cropped)
        elif "grape" in mode_str:
            cls_result = classify_crop_ensemble(cropped, crop="grape")
        else:
            cls_result = classify_crop_ensemble(cropped, crop="apple")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

    # 3. Side-by-Side Grad-CAM
    cam1_base64 = None
    cam2_base64 = None

    if cls_result.get("model_1"):
        try:
            m1_name = cls_result["model_1"]["model_name"]
            crop_name = cls_result.get("crop", "grape")
            cam1, _, _ = generate_model_gradcam(
                cropped, crop=crop_name, model_type=m1_name, alpha=gradcam_alpha, device=DEVICE
            )
            if cam1 is not None:
                cam1_base64 = pil_to_base64(cam1)
        except Exception:
            pass

    if cls_result.get("model_2"):
        try:
            m2_name = cls_result["model_2"]["model_name"]
            crop_name = cls_result.get("crop", "grape")
            cam2, _, _ = generate_model_gradcam(
                cropped, crop=crop_name, model_type=m2_name, alpha=gradcam_alpha, device=DEVICE
            )
            if cam2 is not None:
                cam2_base64 = pil_to_base64(cam2)
        except Exception:
            pass

    response_payload = {
        "status": "success",
        "crop": cls_result.get("crop"),
        "ensemble": cls_result.get("ensemble"),
        "model_1": {
            "name": cls_result.get("model_1", {}).get("model_name"),
            "prediction": cls_result.get("model_1", {}).get("prediction"),
            "confidence": cls_result.get("model_1", {}).get("confidence"),
            "top_predictions": cls_result.get("model_1", {}).get("top_predictions", []),
        },
        "model_2": {
            "name": cls_result.get("model_2", {}).get("model_name"),
            "prediction": cls_result.get("model_2", {}).get("prediction"),
            "confidence": cls_result.get("model_2", {}).get("confidence"),
            "top_predictions": cls_result.get("model_2", {}).get("top_predictions", []),
        },
        "images": {
            "segmented_base64": pil_to_base64(segmented),
            "cropped_base64": pil_to_base64(cropped),
            "gradcam_model_1_base64": cam1_base64,
            "gradcam_model_2_base64": cam2_base64,
        },
    }

    return JSONResponse(content=response_payload)


if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("Starting FastAPI Backend Server on http://0.0.0.0:8000")
    print("Interactive Swagger Documentation: http://localhost:8000/docs")
    print("=" * 60)
    uvicorn.run("api:app", host="0.0.0.0", port=8080, reload=True)
