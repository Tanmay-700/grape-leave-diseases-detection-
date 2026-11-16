import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

from config import GRAPE_CLASSES, GRAPE_DISPLAY_NAMES, IMAGENET_MEAN, IMAGENET_STD
from classify import load_model_instance

DEVICE = torch.device("cpu")
TEST_DIR = "d:/gt/data/splits_grape/test"

print("Evaluating Grape Models on Held-Out Test Set (756 images)...")

# Load models
eff_model, _ = load_model_instance("grape_eff")
res_model, _ = load_model_instance("grape_resnet")
eff_model.eval()
res_model.eval()

# Transforms
tf_eff = transforms.Compose([
    transforms.Resize((260, 260)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
tf_res = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

ds_eff = datasets.ImageFolder(TEST_DIR, transform=tf_eff)
ds_res = datasets.ImageFolder(TEST_DIR, transform=tf_res)

loader_eff = DataLoader(ds_eff, batch_size=32, shuffle=False, num_workers=0)
loader_res = DataLoader(ds_res, batch_size=32, shuffle=False, num_workers=0)

y_true = []
eff_probs_list = []
res_probs_list = []

print("Running EfficientNet-B2...")
with torch.no_grad():
    for x, y in loader_eff:
        out = eff_model(x.to(DEVICE))
        probs = F.softmax(out, dim=1).cpu().numpy()
        eff_probs_list.append(probs)
        y_true.extend(y.numpy())

print("Running ResNet-50...")
with torch.no_grad():
    for x, _ in loader_res:
        out = res_model(x.to(DEVICE))
        probs = F.softmax(out, dim=1).cpu().numpy()
        res_probs_list.append(probs)

eff_probs = np.vstack(eff_probs_list)
res_probs = np.vstack(res_probs_list)
ens_probs = (eff_probs + res_probs) / 2.0
y_true = np.array(y_true)

eff_preds = eff_probs.argmax(axis=1)
res_preds = res_probs.argmax(axis=1)
ens_preds = ens_probs.argmax(axis=1)

eff_acc = (eff_preds == y_true).mean() * 100
res_acc = (res_preds == y_true).mean() * 100
ens_acc = (ens_preds == y_true).mean() * 100

print(f"\n{'='*60}")
print(f"GRAPE TEST ACCURACY REPORT (N = {len(y_true)} images):")
print(f"{'='*60}")
print(f"Model 1: EfficientNet-B2  : {eff_acc:.2f}%")
print(f"Model 2: ResNet-50        : {res_acc:.2f}%")
print(f"Ensemble (Blended Soft-Vote): {ens_acc:.2f}%")
print(f"{'='*60}")

print("\nDetailed Ensemble Classification Report:")
print(classification_report(y_true, ens_preds, target_names=GRAPE_DISPLAY_NAMES, digits=4))

print("Confusion Matrix (Rows: Ground Truth, Cols: Predicted):")
cm = confusion_matrix(y_true, ens_preds)
print(cm)
