"""
viz_utils.py

Shared helper functions for all thesis figure-generation scripts
(OOS detection project, Pepper robot / YOLO26 / Blur2Blur / TP-Diff).

Import this module from the individual fig_*.py scripts, e.g.:

    from viz_utils import load_image_rgb, laplacian_variance, save_figure

All functions work with RGB uint8 numpy arrays (H, W, 3) unless stated
otherwise. Bounding boxes are always [x1, y1, x2, y2] in pixel coordinates.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------

def load_image_rgb(path):
    """Load an image from disk and return it as RGB uint8 (H, W, 3)."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_figure(fig, path, dpi=300):
    """Save a matplotlib figure, creating the parent directory if needed."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"[saved] {path}")


# --------------------------------------------------------------------------
# Resize / crop helpers
# --------------------------------------------------------------------------

def resize_to(img, size):
    """Resize image to (width, height)."""
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def center_crop(img, crop_w, crop_h):
    """Center-crop an image to (crop_w, crop_h). Pads if the image is smaller."""
    h, w = img.shape[:2]
    if h < crop_h or w < crop_w:
        pad_h = max(0, crop_h - h)
        pad_w = max(0, crop_w - w)
        img = cv2.copyMakeBorder(
            img, pad_h // 2, pad_h - pad_h // 2,
            pad_w // 2, pad_w - pad_w // 2,
            cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        h, w = img.shape[:2]
    y0 = (h - crop_h) // 2
    x0 = (w - crop_w) // 2
    return img[y0:y0 + crop_h, x0:x0 + crop_w]


def resize_crop_to_match(img_a, img_b, target_size=None):
    """
    Bring two images to the same resolution and aspect ratio, so they are
    directly comparable side by side.

    If target_size=(w, h) is given, both images are resized to it.
    Otherwise the smaller of the two resolutions (by area) is used as target.
    """
    if target_size is None:
        area_a = img_a.shape[0] * img_a.shape[1]
        area_b = img_b.shape[0] * img_b.shape[1]
        h, w = (img_a.shape[:2] if area_a <= area_b else img_b.shape[:2])
        target_size = (w, h)
    return resize_to(img_a, target_size), resize_to(img_b, target_size)


def zoom_crop(image, bbox, margin=0.4, min_size=64):
    """
    Crop a region around a bounding box with extra margin, useful to show
    a zoomed-in detail of a false positive / false negative.

    bbox: [x1, y1, x2, y2] in pixel coordinates.
    margin: fraction of the box size added on every side.
    min_size: minimum crop width/height in pixels (avoids tiny crops).
    """
    h_img, w_img = image.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    mx, my = bw * margin, bh * margin

    cx1 = max(0, int(x1 - mx))
    cy1 = max(0, int(y1 - my))
    cx2 = min(w_img, int(x2 + mx))
    cy2 = min(h_img, int(y2 + my))

    # enforce a minimum crop size, centered on the box
    if cx2 - cx1 < min_size:
        cx_center = (x1 + x2) // 2
        cx1 = max(0, int(cx_center - min_size / 2))
        cx2 = min(w_img, cx1 + min_size)
    if cy2 - cy1 < min_size:
        cy_center = (y1 + y2) // 2
        cy1 = max(0, int(cy_center - min_size / 2))
        cy2 = min(h_img, cy1 + min_size)

    return image[cy1:cy2, cx1:cx2]


# --------------------------------------------------------------------------
# Sharpness metric
# --------------------------------------------------------------------------

def laplacian_variance(image):
    """Variance of the Laplacian, standard proxy for image sharpness."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    return cv2.Laplacian(gray, cv2.CV_64F).var()


# --------------------------------------------------------------------------
# Bounding boxes
# --------------------------------------------------------------------------

def draw_bboxes_mpl(ax, boxes, color="lime", label=None, linewidth=2, fontsize=8):
    """
    Draw bounding boxes on a matplotlib Axes.

    boxes: list of [x1, y1, x2, y2].
    label: optional text drawn once near the legend corner (e.g. "GT" or "pred").
    """
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=linewidth, edgecolor=color, facecolor="none"
        )
        ax.add_patch(rect)
    if label is not None and len(boxes) > 0:
        x1, y1, _, _ = boxes[0]
        ax.text(
            x1, max(y1 - 4, 0), label, color=color, fontsize=fontsize,
            fontweight="bold", va="bottom",
            bbox=dict(facecolor="black", alpha=0.4, pad=1, edgecolor="none")
        )


def draw_bboxes_cv2(image, boxes, color=(0, 255, 0), thickness=2):
    """Draw boxes directly on a copy of the image with OpenCV (RGB in/out)."""
    out = image.copy()
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    return out


# --------------------------------------------------------------------------
# Plot layout helpers
# --------------------------------------------------------------------------

def new_row_figure(n_panels, panel_size=(4, 4), suptitle=None):
    """Create a 1xN figure of Axes with images turned off (no ticks/frame)."""
    fig, axes = plt.subplots(1, n_panels, figsize=(panel_size[0] * n_panels, panel_size[1]))
    if n_panels == 1:
        axes = [axes]
    for ax in axes:
        ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    return fig, axes


def new_grid_figure(n_rows, n_cols, panel_size=(4, 4), suptitle=None):
    """Create an n_rows x n_cols figure of Axes with images turned off."""
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(panel_size[0] * n_cols, panel_size[1] * n_rows)
    )
    axes = np.array(axes).reshape(n_rows, n_cols)
    for ax in axes.flat:
        ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    return fig, axes
