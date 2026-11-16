"""
Configuration for Dual-Model Ensemble (Setup B)
=================================================
Crop 1: Grape (4 classes)
  1. EfficientNet-B2
  2. ResNet-50

Crop 2: Apple (4 classes)
  1. EfficientNet-B3
  2. ResNet-50
"""
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SPLITS_GRAPE_DIR = os.path.join(DATA_DIR, "splits_grape")
SPLITS_APPLE_DIR = os.path.join(DATA_DIR, "splits_apple")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
SAM2_DIR = os.path.join(PROJECT_ROOT, "sam2_weights")

for d in [DATA_DIR, SPLITS_GRAPE_DIR, SPLITS_APPLE_DIR, MODELS_DIR, OUTPUTS_DIR, SAM2_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================
# Grape Classes (4 classes)
# ============================================================
GRAPE_CLASSES = [
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
]

GRAPE_DISPLAY_NAMES = [
    "Grape — Black Rot",
    "Grape — Esca (Black Measles)",
    "Grape — Leaf Blight",
    "Grape — Healthy",
]

GRAPE_EFF_PATH = os.path.join(MODELS_DIR, "grape_efficientnet_b2.pth")
GRAPE_RESNET_PATH = os.path.join(MODELS_DIR, "grape_resnet50.pth")

# ============================================================
# Apple Classes (4 classes: PlantVillage + Cornell AgriTech FGVC8)
# ============================================================
APPLE_CLASSES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
]

APPLE_DISPLAY_NAMES = [
    "Apple — Apple Scab",
    "Apple — Black Rot / Frogeye Spot",
    "Apple — Cedar Apple Rust",
    "Apple — Healthy",
]

APPLE_EFF_PATH = os.path.join(MODELS_DIR, "apple_efficientnet_b3.pth")
APPLE_RESNET_PATH = os.path.join(MODELS_DIR, "apple_resnet50.pth")

# SAM 2
SAM2_CHECKPOINT = os.path.join(SAM2_DIR, "sam2.1_hiera_small.pt")
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
