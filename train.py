"""
Training Script for Leaf Disease Classifier
==============================================
ResNet50 with transfer learning, weighted sampling for natural images,
and comprehensive logging.

Usage:
    python train.py                    # Full training (Phase 1 + Phase 2)
    python train.py --phase1-only      # PlantVillage pre-training only
    python train.py --resume models/leaf_disease_resnet50_best.pth
"""
import os
import time
import copy
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from config import (
    SPLITS_DIR, MODELS_DIR, OUTPUTS_DIR,
    CLASS_NAMES, DISPLAY_NAMES, NUM_CLASSES,
    IMAGE_SIZE, BATCH_SIZE, NUM_EPOCHS,
    LEARNING_RATE, FINE_TUNE_LR, WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE, NATURAL_WEIGHT_MULTIPLIER,
    IMAGENET_MEAN, IMAGENET_STD,
    CLASSIFIER_CHECKPOINT, CLASSIFIER_BEST,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ============================================================
# Data Transforms
# ============================================================
def get_transforms():
    """Get train and validation/test transforms."""
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(25),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.15)),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    return train_transform, eval_transform


# ============================================================
# Weighted Sampler for Natural Images
# ============================================================
def create_weighted_sampler(dataset, natural_weight=NATURAL_WEIGHT_MULTIPLIER):
    """
    Create a WeightedRandomSampler that:
    1. Balances classes (oversamples minority classes)
    2. Gives extra weight to natural-background images
    """
    # Load natural image list
    natural_list_path = os.path.join(SPLITS_DIR, "natural_images.txt")
    natural_files = set()
    if os.path.exists(natural_list_path):
        with open(natural_list_path, 'r') as f:
            natural_files = set(line.strip() for line in f if line.strip())

    # Count per-class samples
    class_counts = [0] * NUM_CLASSES
    for _, label in dataset.samples:
        class_counts[label] += 1

    # Compute weight per sample
    class_weights = [1.0 / max(c, 1) for c in class_counts]
    sample_weights = []

    for path, label in dataset.samples:
        weight = class_weights[label]
        # Check if this is a natural-background image
        rel_path = os.path.relpath(path, os.path.join(SPLITS_DIR, "train"))
        if rel_path.replace("\\", "/") in natural_files or \
           rel_path.replace("/", "\\") in natural_files or \
           os.path.basename(path).startswith("gvlid_"):
            weight *= natural_weight
        sample_weights.append(weight)

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


# ============================================================
# Model
# ============================================================
def create_model():
    """Create ResNet50 with custom classifier head."""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    
    # Freeze early layers (conv1 through layer2)
    for name, param in model.named_parameters():
        if 'layer3' not in name and 'layer4' not in name and 'fc' not in name:
            param.requires_grad = False

    # Replace classifier
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(512, NUM_CLASSES),
    )

    return model.to(DEVICE)


def unfreeze_all(model):
    """Unfreeze all layers for fine-tuning phase."""
    for param in model.parameters():
        param.requires_grad = True
    return model


# ============================================================
# Training Loop
# ============================================================
def train_one_epoch(model, dataloader, criterion, optimizer, desc="Training"):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=desc, leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100*correct/total:.1f}%")

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, dataloader, criterion, desc="Validation"):
    """Evaluate on validation/test set."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for inputs, labels in tqdm(dataloader, desc=desc, leave=False):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc, np.array(all_preds), np.array(all_labels)


# ============================================================
# Training Manager
# ============================================================
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                num_epochs, save_path, phase_name="Training"):
    """Full training loop with early stopping."""
    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print(f"\n{'='*60}")
    print(f"  {phase_name} — {num_epochs} epochs")
    print(f"{'='*60}")

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 40)

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer,
            desc=f"[{phase_name}] Train"
        )

        # Validate
        val_loss, val_acc, _, _ = evaluate(
            model, val_loader, criterion,
            desc=f"[{phase_name}] Val"
        )

        # Step scheduler
        if scheduler is not None:
            if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Log
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {100*train_acc:.2f}%")
        print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {100*val_acc:.2f}%")
        print(f"  LR: {current_lr:.6f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save({
                'model_state_dict': best_model_wts,
                'class_names': CLASS_NAMES,
                'display_names': DISPLAY_NAMES,
                'num_classes': NUM_CLASSES,
                'epoch': epoch,
                'val_acc': val_acc,
            }, save_path)
            print(f"  ✓ New best model saved! (Val Acc: {100*val_acc:.2f}%)")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{EARLY_STOPPING_PATIENCE})")

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\n  Early stopping triggered after {epoch+1} epochs.")
            break

    # Load best weights
    model.load_state_dict(best_model_wts)
    print(f"\n  Best {phase_name} Val Accuracy: {100*best_val_acc:.2f}%")
    return model, history


# ============================================================
# Plot & Report
# ============================================================
def plot_history(history, save_path):
    """Plot training curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot([a * 100 for a in history['train_acc']], label='Train Acc')
    ax2.plot([a * 100 for a in history['val_acc']], label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy Curves')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Training plot saved to: {save_path}")


def final_evaluation(model, test_loader, criterion):
    """Run final evaluation on test set with full report."""
    print("\n" + "=" * 60)
    print("  FINAL TEST SET EVALUATION")
    print("=" * 60)

    test_loss, test_acc, preds, labels = evaluate(model, test_loader, criterion, "Testing")
    
    print(f"\n  Test Loss: {test_loss:.4f}")
    print(f"  Test Acc:  {100*test_acc:.2f}%")
    
    # Classification report
    report = classification_report(labels, preds, target_names=DISPLAY_NAMES, digits=3)
    print(f"\n{report}")

    # Save report
    report_path = os.path.join(OUTPUTS_DIR, "classification_report.txt")
    with open(report_path, 'w') as f:
        f.write(f"Test Accuracy: {100*test_acc:.2f}%\n\n")
        f.write(report)
    print(f"  Report saved to: {report_path}")

    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=range(NUM_CLASSES), yticks=range(NUM_CLASSES),
           xticklabels=DISPLAY_NAMES, yticklabels=DISPLAY_NAMES,
           xlabel='Predicted', ylabel='True', title='Confusion Matrix')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    
    plt.tight_layout()
    cm_path = os.path.join(OUTPUTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved to: {cm_path}")

    return test_acc


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Train Leaf Disease Classifier")
    parser.add_argument("--phase1-only", action="store_true",
                        help="Only run Phase 1 (no fine-tuning with all layers unfrozen)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    args = parser.parse_args()

    # Verify dataset exists
    train_dir = os.path.join(SPLITS_DIR, "train")
    val_dir = os.path.join(SPLITS_DIR, "val")
    test_dir = os.path.join(SPLITS_DIR, "test")

    if not os.path.isdir(train_dir):
        print("ERROR: Dataset splits not found. Run prepare_dataset.py first!")
        return

    # Transforms
    train_transform, eval_transform = get_transforms()

    # Datasets
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=eval_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=eval_transform)

    # Verify class order matches config
    detected_classes = train_dataset.classes
    print(f"\nDetected classes: {detected_classes}")
    print(f"Config classes:   {CLASS_NAMES}")
    
    if detected_classes != CLASS_NAMES:
        print("\n[WARNING] Class order mismatch! The model will use the detected order.")
        print("  Update CLASS_NAMES in config.py to match if needed.")

    # Weighted sampler
    train_sampler = create_weighted_sampler(train_dataset)

    # DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=0, pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=0, pin_memory=False,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=0, pin_memory=False,
    )

    print(f"\nTrain: {len(train_dataset)} images")
    print(f"Val:   {len(val_dataset)} images")
    print(f"Test:  {len(test_dataset)} images")

    # Create model
    model = create_model()
    
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Resumed from: {args.resume}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # ---- Phase 1: Train with frozen early layers ----
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=WEIGHT_DECAY,
    )
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

    model, history1 = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        num_epochs=args.epochs, save_path=CLASSIFIER_BEST,
        phase_name="Phase 1 (Partial Freeze)",
    )
    plot_history(history1, os.path.join(OUTPUTS_DIR, "phase1_training_curves.png"))

    # ---- Phase 2: Fine-tune all layers with lower LR ----
    if not args.phase1_only:
        model = unfreeze_all(model)
        optimizer_ft = optim.AdamW(
            model.parameters(), lr=FINE_TUNE_LR, weight_decay=WEIGHT_DECAY,
        )
        scheduler_ft = lr_scheduler.ReduceLROnPlateau(
            optimizer_ft, mode='min', factor=0.5, patience=3, verbose=True,
        )

        model, history2 = train_model(
            model, train_loader, val_loader, criterion, optimizer_ft, scheduler_ft,
            num_epochs=args.epochs, save_path=CLASSIFIER_BEST,
            phase_name="Phase 2 (Full Fine-Tune)",
        )
        plot_history(history2, os.path.join(OUTPUTS_DIR, "phase2_training_curves.png"))

    # ---- Final Evaluation ----
    test_acc = final_evaluation(model, test_loader, criterion)

    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': detected_classes,
        'display_names': DISPLAY_NAMES,
        'num_classes': NUM_CLASSES,
        'test_acc': test_acc,
    }, CLASSIFIER_CHECKPOINT)
    print(f"\nFinal model saved to: {CLASSIFIER_CHECKPOINT}")


if __name__ == "__main__":
    main()
