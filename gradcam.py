"""
Universal Grad-CAM Module for Setup B (Dual Models per Crop)
=============================================================
Target layers:
- ResNet-50:        model.layer4[-1]
- EfficientNet-B2:  model.features[-1]
- EfficientNet-B3:  model.features[-1]
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from classify import load_model_instance, get_inference_transform

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class UniversalGradCAM:
    """Grad-CAM for ResNet-50 and EfficientNet variants (B2 and B3)."""

    def __init__(self, model, arch="resnet50"):
        self.model = model
        self.arch = arch
        self.model.eval()

        if arch == "resnet50":
            self.target_layer = model.layer4[-1]
        elif "efficientnet" in arch:
            self.target_layer = model.features[-1]
        else:
            self.target_layer = list(model.children())[-2]

        self.gradients = None
        self.activations = None
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def generate(self, input_tensor, target_class=None):
        input_tensor = input_tensor.to(DEVICE)
        input_tensor.requires_grad_(True)

        output = self.model(input_tensor)
        probs = F.softmax(output, dim=1)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        confidence = probs[0, target_class].item()
        self.model.zero_grad()
        output[0, target_class].backward()

        gradients = self.gradients[0]
        activations = self.activations[0]

        weights = gradients.mean(dim=(1, 2))
        cam = torch.zeros(activations.shape[1:], device=DEVICE)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = F.relu(cam).cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam, target_class, confidence


def create_heatmap_overlay(image, heatmap, alpha=0.5):
    """Overlay heatmap on RGB image."""
    if isinstance(image, Image.Image):
        img = np.array(image.convert("RGB"))
    else:
        img = image.copy()

    h, w = img.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    colored = cv2.applyColorMap((heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    overlay = (img * (1 - alpha) + colored * alpha).astype(np.uint8)
    return Image.fromarray(overlay)


def generate_model_gradcam(image, model_key, alpha=0.5):
    """
    Generate Grad-CAM overlay for a specific model key ('grape_eff', 'apple_eff', 'apple_resnet', etc.).
    """
    model, arch = load_model_instance(model_key)
    transform = get_inference_transform(arch)

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image_rgb = image.convert("RGB")
    tensor = transform(image_rgb).unsqueeze(0)

    gradcam = UniversalGradCAM(model, arch=arch)
    try:
        heatmap, pred_idx, conf = gradcam.generate(tensor)
    finally:
        gradcam.remove_hooks()

    if arch == "efficientnet_b3":
        size = 300
    elif arch == "efficientnet_b2":
        size = 260
    else:
        size = 224

    img_sized = image_rgb.resize((size, size))
    overlay = create_heatmap_overlay(img_sized, heatmap, alpha=alpha)

    return overlay, heatmap, conf
