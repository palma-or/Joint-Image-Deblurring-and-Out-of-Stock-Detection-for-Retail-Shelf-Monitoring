"""
step1_build_dataset.py
======================
STEP 1 of 2 — Build trainA and trainD from extracted video frames.

What it does:
  1. Reads all frames from the given frames directory
  2. Computes the Laplacian variance (blur score) for each frame
  3. Splits frames into two sets based on the chosen threshold:
       - score <  threshold → B (blurry) → trainA
       - score >= threshold → S (sharp)  → trainD
  4. Subsamples the larger set to reach the B:S = 6:4 ratio
     recommended in the Blur2Blur paper (Table 3)
  5. Creates the trainA and trainD folders
     (trainB and trainC are generated in step 2 using the GPU)

Note: choose the optimal threshold by running blur_analysis.py first.

Output structure:
    <output_dir>/
    ├── trainA/  ← B: blurry frames  (score < threshold, subsampled to 6:4)
    ├── trainB/  ← empty, populated by step2_build_k.py (GPU required)
    ├── trainC/  ← empty, populated by step2_build_k.py (GPU required)
    └── trainD/  ← S: sharp frames   (score >= threshold)

Usage:
    python step1_build_dataset.py \\
        --frames_dir <path/to/frames> \\
        --output_dir <path/to/output> \\
        --threshold 200
"""

import os
import shutil
import random
import argparse
import numpy as np
import cv2
from pathlib import Path


# ── CONFIG ────────────────────────────────────────────────────────────────────
DEFAULT_THRESHOLD = 200   # Laplacian variance — calibrate with blur_analysis.py
RATIO_B           = 6    # Recommended B:S ratio (Blur2Blur paper, Table 3)
RATIO_S           = 4
SEED              = 42
# ─────────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description='Build trainA and trainD splits for Blur2Blur training.'
    )
    p.add_argument(
        '--frames_dir',
        type=str,
        required=True,
        help='Directory containing all extracted video frames.',
    )
    p.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for the trainA/B/C/D splits.',
    )
    p.add_argument(
        '--threshold',
        type=float,
        default=DEFAULT_THRESHOLD,
        help=(
            'Laplacian variance threshold: score < threshold → B (blurry), '
            'score >= threshold → S (sharp). '
            'Choose the value after running blur_analysis.py. '
            f'Default: {DEFAULT_THRESHOLD}'
        ),
    )
    return p.parse_args()


def compute_blur_score(path: Path) -> float:
    """
    Compute the Laplacian variance of a grayscale image.

    A lower score indicates more blur; a higher score indicates sharpness.

    Args:
        path: Path to the image file.

    Returns:
        Laplacian variance as float, or 0.0 if the image cannot be read.
    """
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    return cv2.Laplacian(img, cv2.CV_64F).var()


def split_by_threshold(
    frames_dir: Path,
    threshold: float,
) -> tuple[list[Path], list[Path]]:
    """
    Compute blur scores for all frames and split them into blurry and sharp sets.

    Why the Laplacian threshold works:
      Laplacian variance measures the presence of high-frequency edge content.
      A blurry frame (motion or focus blur) has smoothed edges → low variance.
      A sharp frame has crisp edges → high variance.

    Args:
        frames_dir: Directory containing image files.
        threshold:  Laplacian variance cut-off for blurry/sharp split.

    Returns:
        Tuple (blurry_images, sharp_images), each a list of Path objects.
        - blurry_images: sorted by score ascending  (blurriest first)
        - sharp_images:  sorted by score descending (sharpest first)
    """
    all_frames: list[Path] = []
    for ext in ('*.jpg', '*.jpeg', '*.png'):
        all_frames.extend(frames_dir.glob(ext))

    print(f"  Total frames found in {frames_dir}: {len(all_frames)}")

    blurry_candidates: list[tuple[Path, float]] = []
    sharp_candidates:  list[tuple[Path, float]] = []
    skipped = 0

    for i, img_path in enumerate(all_frames):
        if i % 500 == 0:
            print(f"    {i}/{len(all_frames)}...", end='\r')
        score = compute_blur_score(img_path)
        if score == 0.0 and not img_path.exists():
            skipped += 1
            continue
        if score < threshold:
            blurry_candidates.append((img_path, score))
        else:
            sharp_candidates.append((img_path, score))

    print(f"    Done: {len(all_frames)} frames analysed")

    # Sort so the most extreme examples come first within each set
    sharp_candidates.sort(key=lambda x: x[1], reverse=True)   # sharpest first
    blurry_candidates.sort(key=lambda x: x[1], reverse=False)  # blurriest first

    blurry_images = [x[0] for x in blurry_candidates]
    sharp_images  = [x[0] for x in sharp_candidates]

    print(f"\n  Split with threshold = {threshold}:")
    print(f"    B (blurry, score <  {threshold}): {len(blurry_images)} frames")
    print(f"    S (sharp,  score >= {threshold}): {len(sharp_images)} frames")
    if skipped:
        print(f"    Skipped (unreadable files): {skipped}")

    return blurry_images, sharp_images


def apply_ratio(
    blurry_images: list[Path],
    sharp_images:  list[Path],
    ratio_b: int = RATIO_B,
    ratio_s: int = RATIO_S,
) -> tuple[list[Path], list[Path]]:
    """
    Subsample the larger set to reach the target B:S ratio.

    Strategy:
      - If B is too large relative to S → randomly subsample B
      - If S is too large relative to B → randomly subsample S
      - If the current ratio is already within 0.05 of the target → no change

    The Blur2Blur paper (Table 3) recommends ratio B:S = 6:4 = 1.5.

    Args:
        blurry_images: List of paths to blurry frames.
        sharp_images:  List of paths to sharp frames.
        ratio_b:       Numerator of the target ratio (default: 6).
        ratio_s:       Denominator of the target ratio (default: 4).

    Returns:
        Tuple (blurry_images, sharp_images) after subsampling.
    """
    target_ratio   = ratio_b / ratio_s   # 1.5
    n_b            = len(blurry_images)
    n_s            = len(sharp_images)
    current_ratio  = n_b / max(n_s, 1)

    print(f"\n  Current B/S ratio:  {current_ratio:.2f}")
    print(f"  Target  B/S ratio:  {target_ratio:.2f} ({ratio_b}:{ratio_s})")

    if abs(current_ratio - target_ratio) < 0.05:
        print(f"  → Ratio already within tolerance, no subsampling needed")
        return blurry_images, sharp_images

    random.seed(SEED)

    if current_ratio > target_ratio:
        # B is too large → subsample B
        n_b_target   = int(n_s * target_ratio)
        blurry_images = random.sample(blurry_images, min(n_b_target, n_b))
        print(f"  → Subsampled B: {n_b} → {len(blurry_images)}")
    else:
        # S is too large → subsample S
        n_s_target  = int(n_b / target_ratio)
        sharp_images = random.sample(sharp_images, min(n_s_target, n_s))
        print(f"  → Subsampled S: {n_s} → {len(sharp_images)}")

    effective_ratio = len(blurry_images) / max(len(sharp_images), 1)
    print(f"  → Effective ratio after subsampling: {effective_ratio:.2f}")

    return blurry_images, sharp_images


def build_split_folders(
    blurry_images: list[Path],
    sharp_images:  list[Path],
    output_dir:    Path,
) -> None:
    """
    Create trainA (B) and trainD (S) directories and populate them.

    trainB and trainC are left empty — they will be populated by
    step2_build_k.py using the GPU-based KernelWizard.

    Args:
        blurry_images: List of paths to blurry frames → trainA.
        sharp_images:  List of paths to sharp frames  → trainD.
        output_dir:    Root output directory.
    """
    # Recreate all four split directories from scratch
    for folder in ['trainA', 'trainB', 'trainC', 'trainD']:
        folder_path = output_dir / folder
        if folder_path.exists():
            shutil.rmtree(folder_path)
        folder_path.mkdir(parents=True)

    print("\n  Copying B → trainA (blurry frames)...")
    for i, img in enumerate(blurry_images):
        shutil.copy2(img, output_dir / 'trainA' / f"B_{i:05d}_{img.name}")
    print(f"    ✓ {len(blurry_images)} frames")

    print("  Copying S → trainD (sharp frames)...")
    for i, img in enumerate(sharp_images):
        shutil.copy2(img, output_dir / 'trainD' / f"S_{i:05d}_{img.name}")
    print(f"    ✓ {len(sharp_images)} frames")


def main():
    args = parse_args()

    random.seed(SEED)
    np.random.seed(SEED)

    frames_dir = Path(args.frames_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()

    print(f"\n{'='*60}")
    print(f"  STEP 1 — Build Blur2Blur dataset from video frames")
    print(f"  Source:    {frames_dir}")
    print(f"  Output:    {output_dir}")
    print(f"  Threshold: {args.threshold} (Laplacian variance)")
    print(f"  Ratio B:S = {RATIO_B}:{RATIO_S} (Blur2Blur paper)")
    print(f"{'='*60}\n")

    if not frames_dir.exists():
        raise FileNotFoundError(
            f"Directory not found: {frames_dir}\n"
            f"Make sure the frames have been extracted before running this script."
        )

    # Step 1 — Score frames and split by threshold
    print("[ 1/3 ] Computing blur scores and splitting frames...")
    blurry_images, sharp_images = split_by_threshold(frames_dir, args.threshold)

    if not blurry_images:
        raise RuntimeError(
            f"No blurry frames found with threshold={args.threshold}.\n"
            f"Try increasing the threshold (e.g. --threshold 300)."
        )
    if not sharp_images:
        raise RuntimeError(
            f"No sharp frames found with threshold={args.threshold}.\n"
            f"Try decreasing the threshold (e.g. --threshold 100)."
        )

    # Step 2 — Apply B:S ratio
    print(f"\n[ 2/3 ] Applying B:S ratio = {RATIO_B}:{RATIO_S}...")
    blurry_images, sharp_images = apply_ratio(blurry_images, sharp_images, RATIO_B, RATIO_S)

    # Step 3 — Build folder structure
    print("\n[ 3/3 ] Building folder structure...")
    build_split_folders(blurry_images, sharp_images, output_dir)

    n_b = len(blurry_images)
    n_s = len(sharp_images)

    print(f"""
{'='*60}
  STEP 1 complete

  Created in {output_dir}:
    trainA/ : {n_b} frames  ← B: blurry (score < {args.threshold})
    trainB/ : 0 frames  ← to be generated by step2_build_k.py (GPU)
    trainC/ : 0 frames  ← to be generated by step2_build_k.py (GPU)
    trainD/ : {n_s} frames  ← S: sharp  (score >= {args.threshold})

  B:S ratio = {n_b}:{n_s} ≈ {n_b/max(n_s,1):.2f}  (target: {RATIO_B/RATIO_S:.2f})

  Next step:
    python step2_build_k.py --data_dir {output_dir}
{'='*60}
""")


if __name__ == '__main__':
    main()