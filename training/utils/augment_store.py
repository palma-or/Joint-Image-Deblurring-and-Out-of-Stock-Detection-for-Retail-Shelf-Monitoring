"""
augment_store.py
================
Apply the Albumentations pipeline (HorizontalMotionBlur -> GaussNoise -> NeonLighting)
to all images in <store_dir>/images/ and save the results to <output_dir>/.

Output structure:
  <output_dir>/
      images/   <- augmented images (same filenames as the originals)
      labels/   <- YOLO .txt files copied unchanged (bounding boxes don't change)

Usage:
    python augment_store.py \\
        --store-dir  <path/to/Store> \\
        --output-dir <path/to/Store_augmented>

Requirements:
    pip install albumentations opencv-python numpy
"""

import os
import argparse
import random
import shutil
import cv2
import numpy as np
import albumentations as A

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

SEED = 42


def parse_args():
    p = argparse.ArgumentParser(
        description='Apply the Pepper-calibrated Albumentations pipeline to a Store image set.'
    )
    p.add_argument('--store-dir', type=str, required=True,
                    help='Root directory containing images/ and labels/ subfolders.')
    p.add_argument('--output-dir', type=str, required=True,
                    help='Output directory where augmented images/ and labels/ will be saved.')
    p.add_argument('--seed', type=int, default=SEED,
                    help=f'Random seed for the augmentation pipeline (default: {SEED}).')
    return p.parse_args()


# -----------------------------------------------------------------------------
# Pepper-calibrated augmentations
# -----------------------------------------------------------------------------

class HorizontalMotionBlur(A.ImageOnlyTransform):
    """Pure horizontal directional blur calibrated on Pepper (kernel 7-13px)."""
    def __init__(self, length_range=(7, 13), p=1.0):
        super().__init__(p=p)
        self.length_range = length_range

    def apply(self, img, length=10, **params):
        k = np.zeros((length, length), dtype=np.float32)
        k[length // 2, :] = 1.0 / length
        return cv2.filter2D(img, -1, k)

    def get_params(self):
        length = random.randint(*self.length_range)
        if length % 2 == 0:
            length += 1
        return {"length": length}

    def get_transform_init_args_names(self):
        return ("length_range",)


class NeonLighting(A.ImageOnlyTransform):
    """Simulate supermarket neon lighting with a vignette and colour cast."""
    def __init__(self, p=1.0):
        super().__init__(p=p)

    def apply(self, img, cx_ratio=0.5, cy_ratio=0.15,
              sigma_x_ratio=0.7, sigma_y_ratio=0.6,
              warm=False, **params):
        out = img.astype(np.float32)
        h, w = out.shape[:2]

        if warm:
            out[:, :, 1] = np.clip(out[:, :, 1] * 1.04, 0, 255)
            out[:, :, 2] = np.clip(out[:, :, 2] * 1.08, 0, 255)
        else:
            out[:, :, 1] = np.clip(out[:, :, 1] * 1.06, 0, 255)
            out[:, :, 2] = np.clip(out[:, :, 2] * 0.95, 0, 255)

        Y, X = np.mgrid[0:h, 0:w]
        cx = w * cx_ratio
        cy = h * cy_ratio
        sx = w * sigma_x_ratio
        sy = h * sigma_y_ratio
        vignette = np.exp(-0.5 * (((X - cx) / sx) ** 2 + ((Y - cy) / sy) ** 2))
        vignette  = 0.88 + 0.14 * vignette
        out = out * vignette[:, :, np.newaxis]
        return np.clip(out, 0, 255).astype(np.uint8)

    def get_params(self):
        return {
            "cx_ratio":      random.uniform(0.3, 0.7),
            "cy_ratio":      random.uniform(0.05, 0.30),
            "sigma_x_ratio": random.uniform(0.5, 0.9),
            "sigma_y_ratio": random.uniform(0.4, 0.8),
            "warm":          random.random() > 0.5,
        }

    def get_transform_init_args_names(self):
        return ()


PIPELINE = A.Compose([
    HorizontalMotionBlur(length_range=(7, 13), p=1.0),
    A.GaussNoise(std_range=(5.0 / 255.0, 20.0 / 255.0), p=0.4),
    NeonLighting(p=1.0),
])

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

def is_image(fname: str) -> bool:
    return os.path.splitext(fname)[1].lower() in IMG_EXTS


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    args = parse_args()
    random.seed(args.seed)

    src_images_dir = os.path.join(args.store_dir, "images")
    src_labels_dir = os.path.join(args.store_dir, "labels")

    dst_images_dir = os.path.join(args.output_dir, "images")
    dst_labels_dir = os.path.join(args.output_dir, "labels")

    os.makedirs(dst_images_dir, exist_ok=True)
    os.makedirs(dst_labels_dir, exist_ok=True)

    if not os.path.isdir(src_images_dir):
        raise FileNotFoundError(f"Image directory not found: {src_images_dir}")

    image_files = sorted([f for f in os.listdir(src_images_dir) if is_image(f)])
    total = len(image_files)
    print(f"Images found: {total}")
    print(f"Output -> {args.output_dir}\n")

    aug_ok = 0
    lbl_ok = 0
    warn   = 0

    for i, fname in enumerate(image_files, 1):
        src_img_path = os.path.join(src_images_dir, fname)

        # --- Augment the image ---
        img = cv2.imread(src_img_path)
        if img is None:
            print(f"  [WARN] unable to read: {src_img_path}")
            warn += 1
            continue

        aug = PIPELINE(image=img)["image"]
        dst_img_path = os.path.join(dst_images_dir, fname)
        cv2.imwrite(dst_img_path, aug)
        aug_ok += 1

        # --- Copy the matching label file (if it exists) ---
        stem = os.path.splitext(fname)[0]
        src_lbl_path = os.path.join(src_labels_dir, stem + ".txt")
        if os.path.isfile(src_lbl_path):
            dst_lbl_path = os.path.join(dst_labels_dir, stem + ".txt")
            shutil.copy2(src_lbl_path, dst_lbl_path)
            lbl_ok += 1

        if i % 100 == 0 or i == total:
            print(f"  [{i}/{total}] processed...")

    print(f"\nDone.")
    print(f"  Augmented images saved : {aug_ok}")
    print(f"  Labels copied          : {lbl_ok}")
    if warn:
        print(f"  Warnings (skipped images): {warn}")


if __name__ == "__main__":
    main()
