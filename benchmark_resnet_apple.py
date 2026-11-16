import os
import sys
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets
from sklearn.metrics import classification_report, confusion_matrix

# Ensure utf-8 encoding for stdout on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from classify import load_model_instance, get_inference_transform
from config import APPLE_CLASSES, APPLE_DISPLAY_NAMES, SPLITS_APPLE_DIR

def run_test():
    test_dir = os.path.join(SPLITS_APPLE_DIR, 'test')
    print("Loading ResNet-50 Apple model...", flush=True)
    model, arch = load_model_instance('apple_resnet')
    model.eval()
    
    tf = get_inference_transform(arch)
    test_ds = datasets.ImageFolder(test_dir, tf)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)
    
    total = len(test_ds)
    print(f"Evaluating {total} held-out test images across 4 classes...", flush=True)
    
    y_true = []
    y_pred = []
    
    t0 = time.time()
    with torch.inference_mode():
        for i, (images, labels) in enumerate(test_loader):
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            if (i + 1) % 5 == 0 or (i + 1) == len(test_loader):
                elapsed = time.time() - t0
                pct = ((i + 1) / len(test_loader)) * 100
                print(f"  Progress: {pct:5.1f}% ({len(y_true)}/{total} images evaluated in {elapsed:.1f}s)", flush=True)
                
    elapsed_total = time.time() - t0
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    np.save('d:/gt/apple_resnet_y_true.npy', y_true)
    np.save('d:/gt/apple_resnet_y_pred.npy', y_pred)
    
    accuracy = (y_pred == y_true).mean() * 100
    
    report = classification_report(y_true, y_pred, target_names=APPLE_DISPLAY_NAMES, digits=4)
    cm = confusion_matrix(y_true, y_pred)
    
    out_lines = []
    out_lines.append("="*60)
    out_lines.append(f"[APPLE] RESNET-50 TEST EVALUATION RESULTS")
    out_lines.append(f"Total Test Images: {total}")
    out_lines.append(f"Overall Accuracy:  {accuracy:.2f}%")
    out_lines.append(f"Inference Time:    {elapsed_total:.1f}s ({elapsed_total/total*1000:.1f}ms per image)")
    out_lines.append("="*60)
    out_lines.append("\nDetailed Classification Report:")
    out_lines.append(report)
    out_lines.append("\nConfusion Matrix:")
    header = f"{'True Class':<35} | " + " | ".join([f"P:{i}" for i in range(len(APPLE_CLASSES))])
    out_lines.append(header)
    out_lines.append("-" * len(header))
    for i, row in enumerate(cm):
        row_str = " | ".join([f"{val:4d}" for val in row])
        out_lines.append(f"{APPLE_DISPLAY_NAMES[i]:<35} | {row_str}")
        
    out_lines.append("\nClass Index Legend:")
    for i, name in enumerate(APPLE_DISPLAY_NAMES):
        out_lines.append(f"  P:{i} = {name}")
        
    output_text = "\n".join(out_lines)
    
    with open('d:/gt/benchmark_resnet_results.txt', 'w', encoding='utf-8') as f:
        f.write(output_text)
        
    print("\n" + output_text, flush=True)

if __name__ == '__main__':
    run_test()
