"""
Dual-Model Classifier Engine (Setup B: 2 Models Per Crop)
==========================================================
Grape:
  - Model 1: EfficientNet-B2
  - Model 2: ResNet-50

Apple:
  - Model 1: EfficientNet-B3
  - Model 2: ResNet-50
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import models, transforms

from config import (
    GRAPE_CLASSES, GRAPE_DISPLAY_NAMES, GRAPE_EFF_PATH, GRAPE_RESNET_PATH,
    APPLE_CLASSES, APPLE_DISPLAY_NAMES, APPLE_EFF_PATH, APPLE_RESNET_PATH,
    IMAGENET_MEAN, IMAGENET_STD,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_loaded_models = {}


def get_inference_transform(arch="resnet50"):
    """Get the inference transform for the specific architecture."""
    if arch == "efficientnet_b3":
        size = 300
    elif arch == "efficientnet_b2":
        size = 260
    else:
        size = 224

    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_model_instance(model_key):
    """
    Load a specific model instance:
    - 'grape_eff' (EfficientNet-B2)
    - 'grape_resnet' (ResNet-50)
    - 'apple_eff' (EfficientNet-B3)
    - 'apple_resnet' (ResNet-50)
    """
    if model_key in _loaded_models:
        return _loaded_models[model_key]

    if model_key == "grape_eff":
        path = GRAPE_EFF_PATH
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Model not found at: {path}")
        model = models.efficientnet_b2(weights=None)
        in_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(in_ftrs, 512), nn.SiLU(),
            nn.Dropout(0.2), nn.Linear(512, len(GRAPE_CLASSES))
        )
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        arch = "efficientnet_b2"

    elif model_key == "grape_resnet":
        path = GRAPE_RESNET_PATH
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Model not found at: {path}")
        model = models.resnet50(weights=None)
        in_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(in_ftrs, 512), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(512, len(GRAPE_CLASSES))
        )
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        arch = "resnet50"

    elif model_key == "apple_eff":
        path = APPLE_EFF_PATH
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Model not found at: {path}")
        model = models.efficientnet_b3(weights=None)
        in_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(in_ftrs, 512), nn.SiLU(),
            nn.Dropout(0.2), nn.Linear(512, len(APPLE_CLASSES))
        )
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        arch = "efficientnet_b3"

    elif model_key == "apple_resnet":
        path = APPLE_RESNET_PATH
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Model not found at: {path}")
        model = models.resnet50(weights=None)
        in_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(in_ftrs, 512), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(512, len(APPLE_CLASSES))
        )
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        arch = "resnet50"

    else:
        raise ValueError(f"Unknown model_key: {model_key}")

    model = model.to(DEVICE)
    model.eval()
    _loaded_models[model_key] = (model, arch)
    print(f"[INFO] Loaded model: {model_key} ({arch}) from {path}")
    return model, arch


@torch.no_grad()
def predict_single_model(image, model_key):
    """Run prediction on a single model."""
    model, arch = load_model_instance(model_key)
    crop = "grape" if "grape" in model_key else "apple"
    class_names = GRAPE_CLASSES if crop == "grape" else APPLE_CLASSES
    display_names = GRAPE_DISPLAY_NAMES if crop == "grape" else APPLE_DISPLAY_NAMES

    transform = get_inference_transform(arch)
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)

    output = model(tensor)
    probs = F.softmax(output, dim=1).cpu().numpy()[0]
    best_idx = probs.argmax()

    top_k = []
    for idx in probs.argsort()[::-1]:
        top_k.append({
            'class': class_names[idx],
            'label': display_names[idx],
            'confidence': float(probs[idx]),
        })

    return {
        'model_key': model_key,
        'arch': arch,
        'predicted_label': display_names[best_idx],
        'confidence': float(probs[best_idx]),
        'probabilities': probs,
        'top_k': top_k,
    }


def classify_crop_ensemble(image, crop="grape"):
    """
    Run both models for the given crop and compute their ensemble consensus.
    """
    crop = crop.lower()
    keys = ["grape_eff", "grape_resnet"] if crop == "grape" else ["apple_eff", "apple_resnet"]
    class_names = GRAPE_CLASSES if crop == "grape" else APPLE_CLASSES
    display_names = GRAPE_DISPLAY_NAMES if crop == "grape" else APPLE_DISPLAY_NAMES

    results = []
    for k in keys:
        try:
            res = predict_single_model(image, k)
            results.append(res)
        except Exception as e:
            print(f"[WARN] Error running {k}: {e}")

    if not results:
        missing = [f"{k}.pth" for k in keys]
        raise FileNotFoundError(
            f"No trained models found for {crop.title()}! Expected: {', '.join(missing)} in models/."
        )

    if len(results) == 2:
        blended_probs = (results[0]['probabilities'] + results[1]['probabilities']) / 2.0
    else:
        blended_probs = results[0]['probabilities']

    best_idx = blended_probs.argmax()
    ensemble_top_k = [
        {'label': display_names[i], 'confidence': float(blended_probs[i])}
        for i in blended_probs.argsort()[::-1]
    ]

    return {
        'crop': crop,
        'consensus_label': display_names[best_idx],
        'consensus_confidence': float(blended_probs[best_idx]),
        'ensemble_top_k': ensemble_top_k,
        'model_1': results[0] if len(results) >= 1 else None,
        'model_2': results[1] if len(results) >= 2 else None,
        'num_models_active': len(results),
    }


def classify_auto_detect(image):
    """
    Auto-detect crop by comparing confidence between Grape and Apple ensembles.
    """
    res_grape = None
    res_apple = None

    try:
        res_grape = classify_crop_ensemble(image, crop="grape")
    except Exception:
        pass

    try:
        res_apple = classify_crop_ensemble(image, crop="apple")
    except Exception:
        pass

    if res_grape is None and res_apple is None:
        raise FileNotFoundError(
            "No models found! Please train and place model checkpoints in models/."
        )

    if res_grape and not res_apple:
        res_grape['auto_crop'] = "🍇 Grape (only Grape trained)"
        return res_grape
    if res_apple and not res_grape:
        res_apple['auto_crop'] = "🍎 Apple (only Apple trained)"
        return res_apple

    # Both available: compare consensus confidence
    if res_grape['consensus_confidence'] >= res_apple['consensus_confidence']:
        winner = res_grape
        winner['auto_crop'] = "🍇 Grape"
    else:
        winner = res_apple
        winner['auto_crop'] = "🍎 Apple"

    return winner
