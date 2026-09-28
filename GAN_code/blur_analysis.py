"""
blur_analysis.py
================
STEP 0 — Exploratory blur analysis on video frames.

Run this script BEFORE step1_build_dataset.py to choose the Laplacian
variance threshold that splits frames into:
  - Blurry  (score <  threshold) → trainA (B)
  - Sharp   (score >= threshold) → trainD (S)

Produces plots and statistics to guide threshold selection.
The recommended threshold is the one that brings the B:S ratio closest
to 6:4 = 1.5, as recommended by the Blur2Blur paper (Table 3).
After choosing the threshold, pass it to step1_build_dataset.py.

Output:
    <output_dir>/
    ├── 01_frame_distribution.png   ← blur score distribution across all frames
    ├── 02_threshold_sensitivity.png← how B, S, and ratio change with the threshold
    ├── 03_blurriest_examples.png   ← visual examples of the blurriest frames
    ├── 04_sharpest_examples.png    ← visual examples of the sharpest frames
    └── blur_statistics.txt         ← full statistics + recommended threshold

Workflow:
    1. python blur_analysis.py                                ← this script
    2. Read blur_statistics.txt and pick the threshold from the plots
    3. python step1_build_dataset.py --threshold <THRESHOLD>

Usage:
    python blur_analysis.py --frames_dir <path/to/frames> --output_dir <path/to/output>
    python blur_analysis.py --target_ratio 1.5
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for cluster use
import matplotlib.pyplot as plt
from pathlib import Path

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    print("[ERROR] cv2 not found. Install with: pip install opencv-python-headless")
    exit(1)


# ── Plot style ────────────────────────────────────────────────────────────────
COLORS = {
    'frame':  '#3498DB',   # blue  — all frames
    'blurry': '#E74C3C',   # red   — frames below threshold (B)
    'sharp':  '#2ECC71',   # green — frames above threshold (S)
    'thresh': '#F39C12',   # orange— threshold line
    'ratio':  '#9B59B6',   # purple— ratio curve
    'bg':     '#F8F9FA',
}
plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         11,
    'axes.titlesize':    13,
    'axes.labelsize':    11,
    'figure.dpi':        150,
    'axes.spines.top':   False,
    'axes.spines.right': False,
})
# ─────────────────────────────────────────────────────────────────────────────


def blur_score(img_path: Path) -> float | None:
    """
    Compute the Laplacian variance of a grayscale image.

    A lower value indicates more blur (smoothed edges → low high-frequency energy).
    A higher value indicates a sharper image (crisp edges → high variance).

    Args:
        img_path: Path to the image file.

    Returns:
        Laplacian variance as a float, or None if the image cannot be read.
    """
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return cv2.Laplacian(img, cv2.CV_64F).var()


def collect_scores(frames_dir: Path) -> list[tuple[Path, float]]:
    """
    Compute blur scores for all frames in *frames_dir*.

    Args:
        frames_dir: Directory containing extracted video frames.

    Returns:
        List of (path, score) tuples for successfully read images.
    """
    all_frames = []
    for ext in ('*.jpg', '*.jpeg', '*.png'):
        all_frames.extend(frames_dir.glob(ext))

    print(f"  Frames found: {len(all_frames)}")

    results = []
    for i, img_path in enumerate(all_frames):
        if i % 500 == 0:
            print(f"    {i}/{len(all_frames)}...", end='\r')
        score = blur_score(img_path)
        if score is not None:
            results.append((img_path, score))

    print(f"    {len(results)}/{len(all_frames)} frames processed successfully")
    return results


def print_statistics(scores: list[float], f=None) -> np.ndarray:
    """
    Print and optionally save descriptive statistics of the blur score distribution.

    Args:
        scores: List of Laplacian variance values.
        f:      Optional open file handle for saving the output.

    Returns:
        NumPy array of scores.
    """
    s = np.array(scores)
    lines = [
        f"\n{'='*60}",
        f"  BLUR STATISTICS — Video Frames",
        f"{'='*60}",
        f"  Frames analysed:  {len(s)}",
        f"  Min (blurriest):  {s.min():.2f}",
        f"  Max (sharpest):   {s.max():.2f}",
        f"  Mean:             {s.mean():.2f}",
        f"  Median:           {np.median(s):.2f}",
        f"  Std dev:          {s.std():.2f}",
        f"  10th percentile:  {np.percentile(s, 10):.2f}",
        f"  25th percentile:  {np.percentile(s, 25):.2f}",
        f"  75th percentile:  {np.percentile(s, 75):.2f}",
        f"  90th percentile:  {np.percentile(s, 90):.2f}",
        f"  95th percentile:  {np.percentile(s, 95):.2f}",
        f"{'='*60}",
        f"  Distribution at common thresholds:",
    ]
    for threshold in [50, 100, 150, 200, 250, 300, 400, 500]:
        n_b   = (s < threshold).sum()
        n_s   = (s >= threshold).sum()
        ratio = n_b / max(n_s, 1)
        pct_b = (s < threshold).mean() * 100
        lines.append(
            f"  Threshold {threshold:4d}:  B={n_b:5d} ({pct_b:5.1f}%)  "
            f"S={n_s:5d} ({100-pct_b:5.1f}%)  ratio={ratio:.2f}"
        )
    lines.append(f"{'='*60}")

    for line in lines:
        print(line)
        if f:
            f.write(line + '\n')

    return s


def recommend_threshold(
    scores: list[float],
    target_ratio: float = 1.5,
    f=None,
) -> tuple[int, int, int]:
    """
    Find the threshold that brings the B/S ratio closest to *target_ratio*.

    The Blur2Blur paper (Table 3) recommends a 6:4 ratio (target_ratio = 1.5).

    Args:
        scores:       List of Laplacian variance values.
        target_ratio: Desired B/S split ratio (default: 1.5).
        f:            Optional open file handle for saving the output.

    Returns:
        Tuple of (best_threshold, n_blurry, n_sharp).
    """
    s = np.array(scores)
    candidate_thresholds = np.arange(10, 800, 5)
    best_threshold = None
    best_diff = float('inf')

    for threshold in candidate_thresholds:
        n_b   = (s < threshold).sum()
        n_s   = (s >= threshold).sum()
        ratio = n_b / max(n_s, 1)
        diff  = abs(ratio - target_ratio)
        if diff < best_diff:
            best_diff = diff
            best_threshold = threshold

    n_b_best  = int((s < best_threshold).sum())
    n_s_best  = int((s >= best_threshold).sum())
    ratio_eff = n_b_best / max(n_s_best, 1)

    lines = [
        f"\n{'='*60}",
        f"  RECOMMENDED THRESHOLD (target B/S ratio = {target_ratio:.1f})",
        f"{'='*60}",
        f"  Optimal threshold:       {best_threshold}",
        f"  → B (blurry → trainA):   {n_b_best} frames",
        f"  → S (sharp  → trainD):   {n_s_best} frames",
        f"  → Effective B/S ratio:   {ratio_eff:.2f}  (target: {target_ratio:.2f})",
        f"",
        f"  Next step:",
        f"  python step1_build_dataset.py --threshold {best_threshold}",
        f"{'='*60}\n",
    ]
    for line in lines:
        print(line)
        if f:
            f.write(line + '\n')

    return best_threshold, n_b_best, n_s_best


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_frame_distribution(
    results: list[tuple[Path, float]],
    scores: list[float],
    recommended_threshold: int,
    output_path: Path,
) -> None:
    """
    Plot the blur score distribution across all frames, with threshold lines
    and colour-coded bins based on the recommended threshold.

    Args:
        results:                List of (path, score) tuples.
        scores:                 List of raw score values.
        recommended_threshold:  Threshold returned by recommend_threshold().
        output_path:            Destination PNG path.
    """
    s = np.array(scores)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle('Blur Distribution — Video Frames', fontweight='bold')

    thresholds_to_show = sorted({100, 200, 300, int(recommended_threshold)})

    # Linear-scale histogram
    ax = axes[0]
    n, bins, patches = ax.hist(s, bins=100, color=COLORS['frame'],
                               edgecolor='white', alpha=0.85)
    for patch, left_edge in zip(patches, bins[:-1]):
        if left_edge < recommended_threshold:
            patch.set_facecolor(COLORS['blurry'])
            patch.set_alpha(0.7)
        else:
            patch.set_facecolor(COLORS['sharp'])
            patch.set_alpha(0.7)

    for threshold in thresholds_to_show:
        pct = (s < threshold).mean() * 100
        ls  = '-' if threshold == recommended_threshold else '--'
        lw  = 2.2 if threshold == recommended_threshold else 1.5
        label = (
            f'Threshold {threshold} ← RECOMMENDED'
            if threshold == recommended_threshold
            else f'Threshold {threshold} ({pct:.0f}% blurry)'
        )
        ax.axvline(threshold, color=COLORS['thresh'], linestyle=ls,
                   linewidth=lw, label=label)

    ax.set_xlabel('Laplacian variance  (low = blurry,  high = sharp)')
    ax.set_ylabel('Frame count')
    ax.set_title('Linear scale  |  red = B (blurry),  green = S (sharp)')
    ax.legend(fontsize=9)
    ax.set_facecolor(COLORS['bg'])

    # Log-scale histogram + CDF
    ax = axes[1]
    ax.hist(s, bins=100, color=COLORS['frame'], edgecolor='white', alpha=0.85)
    for threshold in thresholds_to_show:
        ls = '-' if threshold == recommended_threshold else '--'
        ax.axvline(threshold, color=COLORS['thresh'], linestyle=ls, linewidth=1.8)
    ax.set_yscale('log')
    ax.set_xlabel('Laplacian variance')
    ax.set_ylabel('Frame count (log scale)')
    ax.set_title('Log scale')
    ax.set_facecolor(COLORS['bg'])

    sorted_s = np.sort(s)
    cdf_ax = axes[1].twinx()
    cdf_ax.plot(sorted_s, np.arange(1, len(sorted_s) + 1) / len(sorted_s),
                color='#2C3E50', linewidth=1.5, alpha=0.6, label='CDF')
    cdf_ax.set_ylabel('CDF', color='#2C3E50')
    cdf_ax.tick_params(axis='y', labelcolor='#2C3E50')
    cdf_ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_threshold_sensitivity(
    scores: list[float],
    recommended_threshold: int,
    output_path: Path,
) -> None:
    """
    Show how the sizes of B, S, and the B/S ratio change across all thresholds.

    This plot is essential for making an informed threshold choice.

    Args:
        scores:                 List of Laplacian variance values.
        recommended_threshold:  Threshold returned by recommend_threshold().
        output_path:            Destination PNG path.
    """
    s = np.array(scores)
    thresholds = np.arange(10, 800, 5)
    n_B   = [(s < t).sum() for t in thresholds]
    n_S   = [(s >= t).sum() for t in thresholds]
    ratio = [b / max(ss, 1) for b, ss in zip(n_B, n_S)]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Laplacian Threshold Sensitivity Analysis', fontweight='bold')

    for ax, y, ylabel, title, color in [
        (axes[0], n_B,   'Frames in B (blurry)', 'trainA size (B)',           COLORS['blurry']),
        (axes[1], n_S,   'Frames in S (sharp)',  'trainD size (S)',           COLORS['sharp']),
        (axes[2], ratio, 'B/S ratio',            'B/S ratio vs threshold',    COLORS['ratio']),
    ]:
        ax.plot(thresholds, y, color=color, linewidth=2)
        ax.axvline(recommended_threshold, color=COLORS['thresh'], linestyle='-',
                   linewidth=2, label=f'Recommended threshold ({recommended_threshold})')
        if ylabel == 'B/S ratio':
            ax.axhline(1.5, color='gray', linestyle='--', linewidth=1.5,
                       label='Target ratio = 1.5 (paper Table 3)')
            ax.axhline(1.0, color='lightgray', linestyle=':', linewidth=1)
            ax.set_ylim(0, min(6, max(ratio) * 1.1))
        ax.set_xlabel('Laplacian variance threshold')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.set_facecolor(COLORS['bg'])

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_examples(
    results: list[tuple[Path, float]],
    title: str,
    output_path: Path,
    n: int = 6,
    blurriest_first: bool = True,
) -> None:
    """
    Display a grid of example frames sorted by blur score.

    Args:
        results:         List of (path, score) tuples.
        title:           Plot title string.
        output_path:     Destination PNG path.
        n:               Number of examples to show (default: 6).
        blurriest_first: If True, show blurriest frames; otherwise show sharpest.
    """
    sorted_results = sorted(results, key=lambda x: x[1], reverse=not blurriest_first)
    sample = sorted_results[:n]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(title, fontweight='bold', fontsize=13)

    for i, (img_path, score) in enumerate(sample):
        ax = axes[i // 3][i % 3]
        img = cv2.imread(str(img_path))
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img_rgb.shape[:2]
            cy, cx = h // 2, w // 2
            half = 128
            # Centre crop for a closer look at blur/sharpness
            crop = img_rgb[max(0, cy - half):cy + half, max(0, cx - half):cx + half]
            ax.imshow(crop)
        ax.set_title(f'Score: {score:.1f}\n{img_path.name[:35]}', fontsize=8)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Analyse blur scores on video frames to choose a Laplacian threshold.'
    )
    parser.add_argument(
        '--frames_dir',
        type=str,
        required=True,
        help='Directory containing all extracted video frames.',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory where plots and statistics will be saved.',
    )
    parser.add_argument(
        '--target_ratio',
        type=float,
        default=1.5,
        help='Target B/S split ratio (default: 1.5 = 6:4, Blur2Blur paper).',
    )
    args = parser.parse_args()

    frames_dir = Path(args.frames_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Blur Analysis — Video Frames")
    print(f"  Source:       {frames_dir}")
    print(f"  Output:       {output_dir}")
    print(f"  Target B/S ratio: {args.target_ratio:.1f} (Blur2Blur paper)")
    print(f"{'='*60}\n")

    if not frames_dir.exists():
        raise FileNotFoundError(f"Directory not found: {frames_dir}")

    # Step 1 — Collect scores
    print("[ 1/4 ] Computing blur scores for all frames...")
    results = collect_scores(frames_dir)

    if not results:
        raise RuntimeError(f"No frames found in {frames_dir}")

    scores = [s for _, s in results]

    # Step 2 — Statistics and recommended threshold
    print("\n[ 2/4 ] Computing statistics...")
    txt_path = output_dir / 'blur_statistics.txt'
    with open(txt_path, 'w') as f:
        print_statistics(scores, f)
        recommended_threshold, n_blurry, n_sharp = recommend_threshold(
            scores, args.target_ratio, f
        )
    print(f"  Statistics saved to: {txt_path}")

    # Step 3 — Plots
    print("\n[ 3/4 ] Generating plots...")
    plot_frame_distribution(
        results, scores, recommended_threshold,
        output_dir / '01_frame_distribution.png',
    )
    plot_threshold_sensitivity(
        scores, recommended_threshold,
        output_dir / '02_threshold_sensitivity.png',
    )

    # Step 4 — Visual examples
    print("\n[ 4/4 ] Generating visual examples...")
    plot_examples(
        results,
        title=f'Blurriest frames (low score → trainA)',
        output_path=output_dir / '03_blurriest_examples.png',
        blurriest_first=True,
    )
    plot_examples(
        results,
        title=f'Sharpest frames (high score → trainD)',
        output_path=output_dir / '04_sharpest_examples.png',
        blurriest_first=False,
    )

    print(f"""
{'='*60}
  Analysis complete.
  Output: {output_dir}

  Read blur_statistics.txt and choose the threshold from the plots,
  then run:
    python step1_build_dataset.py --threshold <CHOSEN_THRESHOLD> \\
        --frames_dir {frames_dir} --output_dir <OUTPUT_DIR>
{'='*60}
""")


if __name__ == '__main__':
    main()