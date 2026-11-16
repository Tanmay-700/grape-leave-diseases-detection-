"""
Prepare Apple Dataset (Lab + In-Field Fusion)
==============================================
Merges:
  1. Controlled (Lab): PlantVillage Apple (3,171 images)
     - D:\\Leaf_diseases\\dataset\\PlantVillage-Dataset\\raw\\color
  2. Uncontrolled (Field): FGVC8 Plant Pathology 2021 (14,491 clean images at 512x512)
     - D:\\gt\\data\\archive.zip (img_sz_512) + D:\\gt\\data\\train.csv

Generates:
  - Stratified 80 / 10 / 10 splits (train, val, test)
  - Balanced across all 4 classes
  - Tagged by origin ('lab_' vs 'field_') for 2.0x field sampling
  - Output: D:\\gt\\data\\splits_apple and splits_apple.zip
"""
import os
import io
import shutil
import zipfile
import random
import pandas as pd
from collections import defaultdict
from PIL import Image

random.seed(42)

# Paths
BASE_DIR = r"D:\gt\data"
PV_DIR = r"D:\Leaf_diseases\dataset\PlantVillage-Dataset\raw\color"
ARCHIVE_ZIP = os.path.join(BASE_DIR, "archive.zip")
TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
SPLITS_DIR = os.path.join(BASE_DIR, "splits_apple")
OUTPUT_ZIP = os.path.join(BASE_DIR, "splits_apple.zip")

# Standard class mapping
CLASS_MAP_FIELD = {
    "scab": "Apple___Apple_scab",
    "frog_eye_leaf_spot": "Apple___Black_rot",
    "rust": "Apple___Cedar_apple_rust",
    "healthy": "Apple___healthy",
}

CLASS_MAP_LAB = {
    "Apple___Apple_scab": "Apple___Apple_scab",
    "Apple___Black_rot": "Apple___Black_rot",
    "Apple___Cedar_apple_rust": "Apple___Cedar_apple_rust",
    "Apple___healthy": "Apple___healthy",
}

CLASSES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
]


def prepare_dataset():
    print("[1/5] Reading FGVC8 train.csv labels...")
    df = pd.read_csv(TRAIN_CSV)
    df_clean = df[df["labels"].isin(CLASS_MAP_FIELD.keys())].copy()
    print(f"  Total valid single-label field images: {len(df_clean)}")
    for lbl, count in df_clean["labels"].value_counts().items():
        print(f"    - {lbl} -> {CLASS_MAP_FIELD[lbl]}: {count}")

    # Reset output directory
    if os.path.exists(SPLITS_DIR):
        print(f"  Cleaning existing {SPLITS_DIR}...")
        shutil.rmtree(SPLITS_DIR)

    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            os.makedirs(os.path.join(SPLITS_DIR, split, cls), exist_ok=True)

    # Collect all image items: list of (source_type, source_path_or_zipinfo, class_name, filename)
    items_by_class_and_origin = defaultdict(list)

    # 1. Collect Field images from archive.zip
    print("\n[2/5] Indexing field images from archive.zip...")
    field_img_to_class = dict(zip(df_clean["image"], df_clean["labels"].map(CLASS_MAP_FIELD)))
    
    with zipfile.ZipFile(ARCHIVE_ZIP, "r") as zf:
        namelist = set(zf.namelist())
        for img_name, cls_name in field_img_to_class.items():
            zip_entry = f"img_sz_512/{img_name}"
            if zip_entry in namelist:
                items_by_class_and_origin[(cls_name, "field")].append((zip_entry, img_name))

    # 2. Collect Lab images from PlantVillage
    print("\n[3/5] Indexing lab images from PlantVillage...")
    for pv_cls, target_cls in CLASS_MAP_LAB.items():
        pv_folder = os.path.join(PV_DIR, pv_cls)
        if not os.path.isdir(pv_folder):
            continue
        for fname in os.listdir(pv_folder):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                fpath = os.path.join(pv_folder, fname)
                items_by_class_and_origin[(target_cls, "lab")].append((fpath, fname))

    # Print summary of collected items
    print("\n  Summary of Collected Images:")
    total_imgs = 0
    for cls in CLASSES:
        n_field = len(items_by_class_and_origin[(cls, "field")])
        n_lab = len(items_by_class_and_origin[(cls, "lab")])
        n_total = n_field + n_lab
        total_imgs += n_total
        print(f"    * {cls}: {n_total} total ({n_field} field + {n_lab} lab)")
    print(f"  Grand Total Images to Split: {total_imgs}")

    # 3. Stratified 80 / 10 / 10 split
    print("\n[4/5] Performing stratified 80 / 10 / 10 split and writing files...")
    split_counts = defaultdict(lambda: defaultdict(int))

    with zipfile.ZipFile(ARCHIVE_ZIP, "r") as zf:
        for (cls, origin), item_list in items_by_class_and_origin.items():
            random.shuffle(item_list)
            n = len(item_list)
            n_train = int(n * 0.80)
            n_val = int(n * 0.10)
            # test gets remaining

            splits = {
                "train": item_list[:n_train],
                "val": item_list[n_train:n_train + n_val],
                "test": item_list[n_train + n_val:],
            }

            for split_name, sublist in splits.items():
                dest_dir = os.path.join(SPLITS_DIR, split_name, cls)
                for src, fname in sublist:
                    out_fname = f"{origin}_{fname}"
                    dest_path = os.path.join(dest_dir, out_fname)

                    if origin == "field":
                        # Read from zip and write
                        with zf.open(src) as z_in, open(dest_path, "wb") as f_out:
                            f_out.write(z_in.read())
                    else:
                        # Copy lab file
                        shutil.copy2(src, dest_path)

                    split_counts[split_name][cls] += 1

    print("\n  Split Distribution:")
    for split in ["train", "val", "test"]:
        print(f"    [{split.upper()}] Total: {sum(split_counts[split].values())}")
        for cls in CLASSES:
            print(f"      - {cls}: {split_counts[split][cls]}")

    # 4. Create ZIP archive for Colab upload
    print(f"\n[5/5] Compressing {SPLITS_DIR} into {OUTPUT_ZIP}...")
    if os.path.exists(OUTPUT_ZIP):
        os.remove(OUTPUT_ZIP)

    shutil.make_archive(SPLITS_DIR, "zip", SPLITS_DIR)
    zip_size_mb = os.path.getsize(OUTPUT_ZIP) / (1024 * 1024)
    print(f"  Successfully created: {OUTPUT_ZIP} ({zip_size_mb:.1f} MB)")
    print("\n[COMPLETE] Apple dataset preparation is finished!")


if __name__ == "__main__":
    prepare_dataset()
