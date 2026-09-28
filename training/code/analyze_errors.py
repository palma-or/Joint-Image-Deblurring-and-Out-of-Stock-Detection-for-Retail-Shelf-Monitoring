"""
analyze_errors.py
=================
Automated detection error analysis (FP and FN) on a set of test images.

FP categories — the detector predicts 'empty' but there is no matching GT:
  dark_region     : mean crop brightness < THR_DARK    (black area, shelf edge)
  motion_blur     : Laplacian variance < THR_BLUR       (Pepper camera motion blur)
  occupied_shelf  : high saturation + high texture      (area with products)
  refrigerator    : dominant blue/green tint + uniform brightness (fridge cell)
  other_fp        : everything else

FN categories — there is a GT 'empty' box but the detector misses it (iou > 0 with no prediction):
  small_box       : GT area < THR_SMALL_PX²             (small target)
  dark_fn         : mean crop brightness < THR_DARK     (shadowed area)
  low_contrast    : pixel variance < THR_CONTRAST       (low-contrast shelf)
  occluded        : IoU with other GT boxes > THR_OCC   (partially occluded)
  other_fn        : everything else

FNR is computed as FN / #GT_total (not 1 - Recall).

Output:
  <output_dir>/fp_errors.csv              — one row per FP with category and features
  <output_dir>/fn_errors.csv              — one row per FN with category and features
  <output_dir>/error_statistics.txt       — counts and percentages by category
  <output_dir>/error_statistics.png       — visual summary table
  <output_dir>/panels_fp/<cat>/           — sampled crops for each FP category
  <output_dir>/panels_fn/<cat>/           — sampled crops for each FN category

Usage:
  python analyze_errors.py \\
      --runs-dir <path/to/runs> \\
      --image-dir <path/to/dataset/test/images> \\
      --label-dir <path/to/dataset/test/labels> \\
      --output <path/to/error_analysis> \\
      --runs oos_gan_frozen oos_joint
"""
import os
import csv
import argparse
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from collections import defaultdict
from collections import Counter

# ══════════════════════════════════════════════════════════════
#  CONFIG (non-path constants — thresholds and fixed parameters)
# ══════════════════════════════════════════════════════════════

CONF_THRESH = 0.25    # YOLO prediction confidence threshold
IOU_THRESH  = 0.0     # IoU threshold: match if iou > 0 (any positive overlap)
IMGSZ       = 640
SEED        = 42

# Heuristic thresholds for error categorisation (pixel values in [0, 255])
THR_DARK        = 40     # mean brightness below which a crop is "dark"
THR_BLUR        = 50     # Laplacian variance below which it's "motion blur"
THR_CONTRAST    = 200    # pixel variance below which it's "low_contrast" (FN)
THR_SMALL_PX    = 32     # minimum side in pixels below which it's "small_box"
THR_SAT_HIGH    = 40     # mean saturation above which it's "occupied_shelf"
THR_TEXTURE     = 400    # texture variance above which it's "occupied_shelf"
THR_OCC         = 0.3    # IoU with other GT above which it's "occluded"
THR_REFRIG_B    = 0.38   # blue channel fraction for "refrigerator"

N_PANELS_PER_CAT = 1000   # crops to save per category

# Populated from CLI args at the start of main()
FONT_BOLD = None
FONT_REG  = None


# ══════════════════════════════════════════════════════════════
#  ARGPARSE
# ══════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description='Automated FP/FN error analysis for one or more training runs.'
    )
    p.add_argument('--runs-dir', type=str, required=True,
                    help='Directory containing the run folders (each with weights/best.pt).')
    p.add_argument('--image-dir', type=str, required=True,
                    help='Path to the test images directory.')
    p.add_argument('--label-dir', type=str, required=True,
                    help='Path to the test labels directory (YOLO format).')
    p.add_argument('--output', type=str, required=True,
                    help='Base output directory for CSVs, panels, and statistics.')
    p.add_argument('--run', type=str, default=None,
                    help='Name of a single run folder under --runs-dir. Overrides --runs if set.')
    p.add_argument('--runs', type=str, nargs='+', default=['oos_gan_frozen_store5', 'oos_joint_store5'],
                    help='List of run folder names to analyze (default: gan_frozen + joint).')
    p.add_argument('--conf', type=float, default=CONF_THRESH,
                    help=f'Confidence threshold for YOLO predictions (default: {CONF_THRESH}).')
    p.add_argument('--iou', type=float, default=IOU_THRESH,
                    help=f'IoU threshold for matching (default: {IOU_THRESH}).')
    p.add_argument('--seed', type=int, default=SEED,
                    help=f'Random seed for panel sampling (default: {SEED}).')
    p.add_argument('--font-bold', type=str,
                    default='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
                    help='Path to a bold TTF font used for table/panel titles.')
    p.add_argument('--font-regular', type=str,
                    default='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                    help='Path to a regular TTF font used for table/panel body text.')
    return p.parse_args()


# ══════════════════════════════════════════════════════════════
#  UTILITIES
# ══════════════════════════════════════════════════════════════

def _font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()


def read_yolo_labels(label_path, img_w, img_h):
    """Read GT boxes from a YOLO .txt file. Returns a list of dicts."""
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            xc, yc, bw, bh = (float(x) for x in parts[1:5])
            x1 = int((xc - bw / 2) * img_w)
            y1 = int((yc - bh / 2) * img_h)
            x2 = int((xc + bw / 2) * img_w)
            y2 = int((yc + bh / 2) * img_h)
            boxes.append({'cls': cls,
                          'x1': max(0, x1), 'y1': max(0, y1),
                          'x2': min(img_w, x2), 'y2': min(img_h, y2)})
    return boxes


def iou_box(a, b):
    """IoU between two box dicts with x1,y1,x2,y2."""
    ix1 = max(a['x1'], b['x1'])
    iy1 = max(a['y1'], b['y1'])
    ix2 = min(a['x2'], b['x2'])
    iy2 = min(a['y2'], b['y2'])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = max(1, (a['x2'] - a['x1']) * (a['y2'] - a['y1']))
    area_b = max(1, (b['x2'] - b['x1']) * (b['y2'] - b['y1']))
    return inter / (area_a + area_b - inter)


# ══════════════════════════════════════════════════════════════
#  VISUAL HEURISTICS
# ══════════════════════════════════════════════════════════════

def _crop_arr(img_arr, box, margin=4):
    """Crop the bounding box region with a minimum margin."""
    h, w = img_arr.shape[:2]
    x1 = max(0, box['x1'] - margin)
    y1 = max(0, box['y1'] - margin)
    x2 = min(w, box['x2'] + margin)
    y2 = min(h, box['y2'] + margin)
    if x2 <= x1 or y2 <= y1:
        return None
    return img_arr[y1:y2, x1:x2]


def _features(crop_arr):
    """
    Compute the visual features used for categorization.
    Returns a dict with: brightness, laplacian_var, saturation,
    texture_var, blue_ratio.
    """
    if crop_arr is None or crop_arr.size == 0:
        return {}
    gray = crop_arr.mean(axis=2)                     # brightness
    brightness = float(gray.mean())

    # Laplacian for blur detection
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    from scipy.ndimage import convolve
    lap = convolve(gray.astype(np.float32), kernel)
    laplacian_var = float(lap.var())

    # Saturation (from HSV): max - min per pixel
    r, g, b = crop_arr[:, :, 0], crop_arr[:, :, 1], crop_arr[:, :, 2]
    sat = (crop_arr.max(axis=2).astype(np.float32)
           - crop_arr.min(axis=2).astype(np.float32))
    saturation = float(sat.mean())

    # Texture (local variance of the grayscale pixels)
    texture_var = float(gray.var())

    # Blue channel fraction (useful for fridge cells)
    total = r.astype(np.float32) + g.astype(np.float32) + b.astype(np.float32) + 1e-6
    blue_ratio = float((b.astype(np.float32) / total).mean())

    return {
        'brightness':    brightness,
        'laplacian_var': laplacian_var,
        'saturation':    saturation,
        'texture_var':   texture_var,
        'blue_ratio':    blue_ratio,
    }


def classify_fp(feat):
    """
    Categorize an FP based on its visual features.
    Priority: dark > blur > refrigerator > occupied > other.
    """
    if not feat:
        return 'other_fp'
    if feat['brightness'] < THR_DARK:
        return 'dark_region'
    if feat['laplacian_var'] < THR_BLUR:
        return 'motion_blur'
    if (feat['blue_ratio'] > THR_REFRIG_B
            and feat['saturation'] < THR_SAT_HIGH
            and feat['brightness'] > 80):
        return 'refrigerator'
    if (feat['saturation'] > THR_SAT_HIGH
            and feat['texture_var'] > THR_TEXTURE):
        return 'occupied_shelf'
    return 'other_fp'


def classify_fn(box, gt_boxes, feat):
    """
    Categorize an FN based on the GT box properties and its crop.
    Priority: small > dark > occluded > low_contrast > other.
    """
    if not feat:
        return 'other_fn'
    bw = box['x2'] - box['x1']
    bh = box['y2'] - box['y1']
    if min(bw, bh) < THR_SMALL_PX:
        return 'small_box'
    if feat['brightness'] < THR_DARK:
        return 'dark_fn'
    # Check occlusion: IoU with other GT boxes
    other_gt = [b for b in gt_boxes if b is not box]
    max_iou = max((iou_box(box, o) for o in other_gt), default=0.0)
    if max_iou > THR_OCC:
        return 'occluded'
    if feat['texture_var'] < THR_CONTRAST:
        return 'low_contrast'
    return 'other_fn'


# ══════════════════════════════════════════════════════════════
#  TP / FP / FN MATCHING
# ══════════════════════════════════════════════════════════════

def match_boxes(gt_boxes, pred_boxes, iou_thresh):
    """
    Match predictions to GT via mat_pred / mat_gt (non-greedy).

    Definitions aligned with _compute_tp_fp_fn in visualize_detector_only.py:
      - A prediction is FP if mat_pred[pi] is an empty list
        (no GT with iou > 0, or iou >= iou_thresh if iou_thresh > 0)
      - A GT box is FN if mat_gt[gi] is an empty list
        (no prediction covers it)

    Match condition:
      (iou >= iou_thresh and iou_thresh > 0) or (iou > 0 and iou_thresh == 0)

    Returns:
      tp_gt_indices   : GT indices with at least one matched prediction
      fp_pred_indices : prediction indices with an empty list (FP)
      fn_gt_indices   : GT indices with an empty list (FN)
    """
    n_preds = len(pred_boxes)
    n_gts   = len(gt_boxes)

    mat_pred = [[] for _ in range(n_preds)]  # mat_pred[pi] -> list of matched gi
    mat_gt   = [[] for _ in range(n_gts)]    # mat_gt[gi]   -> list of matched pi

    for pi, pred in enumerate(pred_boxes):
        for gi, gt in enumerate(gt_boxes):
            iou = iou_box(pred, gt)
            if (iou_thresh > 0 and iou >= iou_thresh) or \
               (iou_thresh == 0 and iou > 0):
                mat_pred[pi].append(gi)
                mat_gt[gi].append(pi)

    tp_gt_indices   = [gi for gi in range(n_gts)   if len(mat_gt[gi])   > 0]
    fp_pred_indices = [pi for pi in range(n_preds) if len(mat_pred[pi]) == 0]
    fn_gt_indices   = [gi for gi in range(n_gts)   if len(mat_gt[gi])   == 0]

    return tp_gt_indices, fp_pred_indices, fn_gt_indices


# ══════════════════════════════════════════════════════════════
#  ERROR COLLECTION
# ══════════════════════════════════════════════════════════════

def collect_errors(img_paths, lbl_dir, model, conf, iou_thresh, device, imgsz):
    """
    Iterate over all images, run prediction, and collect FP and FN.
    Returns two lists of dicts: fp_records, fn_records.
    """
    from ultralytics import YOLO as _YOLO  # local import to avoid a top-level import

    fp_records = []
    fn_records = []

    n_gt_total  = 0  # total GT counter for FNR
    n_fn_total  = 0

    for img_path in img_paths:
        img_path = Path(img_path)
        lbl_path = Path(lbl_dir) / (img_path.stem + '.txt')

        img_pil = Image.open(img_path).convert('RGB')
        W, H    = img_pil.size
        img_arr = np.array(img_pil)

        gt_boxes = read_yolo_labels(str(lbl_path), W, H)
        # Keep only class 0 (empty)
        gt_boxes = [b for b in gt_boxes if b['cls'] == 0]

        # YOLO predictions
        results = model.predict(source=img_pil, imgsz=imgsz,
                                conf=conf, device=device, verbose=False)
        pred_boxes = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                if int(box.cls[0].cpu()) != 0:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                pred_boxes.append({
                    'cls':  0,
                    'conf': float(box.conf[0].cpu()),
                    'x1': max(0, x1), 'y1': max(0, y1),
                    'x2': min(W, x2), 'y2': min(H, y2),
                })

        _, fp_idxs, fn_idxs = match_boxes(gt_boxes, pred_boxes, iou_thresh)

        n_gt_total += len(gt_boxes)
        n_fn_total += len(fn_idxs)

        # ── Record FP ──────────────────────────────────────
        for pi in fp_idxs:
            pred = pred_boxes[pi]
            crop = _crop_arr(img_arr, pred)
            feat = _features(crop)
            cat  = classify_fp(feat)
            fp_records.append({
                'img':          img_path.name,
                'category':    cat,
                'conf':         round(pred['conf'], 4),
                'x1': pred['x1'], 'y1': pred['y1'],
                'x2': pred['x2'], 'y2': pred['y2'],
                'width':        pred['x2'] - pred['x1'],
                'height':       pred['y2'] - pred['y1'],
                'brightness':   round(feat.get('brightness', -1), 2),
                'laplacian_var': round(feat.get('laplacian_var', -1), 2),
                'saturation':   round(feat.get('saturation', -1), 2),
                'texture_var':  round(feat.get('texture_var', -1), 2),
                'blue_ratio':   round(feat.get('blue_ratio', -1), 4),
                '_img_arr':     img_arr,   # removed before CSV export
                '_box':         pred,
                '_gt_boxes':    gt_boxes,
            })

        # ── Record FN ──────────────────────────────────────
        for gi in fn_idxs:
            gt = gt_boxes[gi]
            crop = _crop_arr(img_arr, gt)
            feat = _features(crop)
            cat  = classify_fn(gt, gt_boxes, feat)
            bw   = gt['x2'] - gt['x1']
            bh   = gt['y2'] - gt['y1']
            fn_records.append({
                'img':          img_path.name,
                'category':    cat,
                'x1': gt['x1'], 'y1': gt['y1'],
                'x2': gt['x2'], 'y2': gt['y2'],
                'width':        bw,
                'height':       bh,
                'area':         bw * bh,
                'brightness':   round(feat.get('brightness', -1), 2),
                'laplacian_var': round(feat.get('laplacian_var', -1), 2),
                'saturation':   round(feat.get('saturation', -1), 2),
                'texture_var':  round(feat.get('texture_var', -1), 2),
                'blue_ratio':   round(feat.get('blue_ratio', -1), 4),
                '_img_arr':     img_arr,
                '_box':         gt,
                '_gt_boxes':    gt_boxes,
            })

    return fp_records, fn_records, n_gt_total, n_fn_total


# ══════════════════════════════════════════════════════════════
#  CSV EXPORT
# ══════════════════════════════════════════════════════════════

_FP_FIELDS = ['img', 'category', 'conf', 'x1', 'y1', 'x2', 'y2',
              'width', 'height', 'brightness', 'laplacian_var',
              'saturation', 'texture_var', 'blue_ratio']
_FN_FIELDS = ['img', 'category', 'x1', 'y1', 'x2', 'y2',
              'width', 'height', 'area', 'brightness', 'laplacian_var',
              'saturation', 'texture_var', 'blue_ratio']


def save_csv(records, fields, path):
    """
    Save a list of dicts to a CSV file using *fields* as column names.

        Args:
            records: List of dicts (extra keys are silently ignored).
            fields:  Ordered list of field names for the CSV header.
            path:    Destination file path.
    """
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(records)
    print(f"  CSV saved: {path} ({len(records)} rows)")


# ══════════════════════════════════════════════════════════════
#  PANELS — FULL IMAGE + CIRCLE + CROP
# ══════════════════════════════════════════════════════════════

def save_crop_panels(records, kind, output_dir, n_per_cat, seed):
    """
    For each sampled error, save a panel with:
      - Left column  : full image with a coloured circle on the bbox
      - Right column : enlarged crop of the bbox (max 300px)
      - Info row     : category, file name, numeric features

    The circle has radius = max(w,h)/2 + margin, drawn over the bbox
    to make it visible even for small boxes.
    """
    random.seed(seed)

    per_cat = defaultdict(list)
    for r in records:
        per_cat[r['category']].append(r)

    colors = {
        # FP categories
        'dark_region':    (30,  30, 200),
        'motion_blur':    (200, 200,  30),
        'occupied_shelf': (200,  80,  30),
        'refrigerator':   (30,  180, 200),
        'other_fp':       (150, 150, 150),
        # FN categories
        'small_box':      (200,  30,  30),
        'dark_fn':        (30,   30, 150),
        'low_contrast':   (150,  30, 200),
        'occluded':       (200, 130,  30),
        'other_fn':       (150, 150, 150),
    }

    base = os.path.join(output_dir, f'panels_{kind}')
    os.makedirs(base, exist_ok=True)

    # CSV mapping panel -> original image
    mapping_path = os.path.join(base, f'mapping_{kind}.csv')
    _mapping_rows = []

    f_bold = _font(FONT_BOLD, 11)
    f_reg  = _font(FONT_REG,  10)

    # Fixed panel dimensions
    IMG_MAX   = 480   # max side for the full image (scaled)
    CROP_MAX  = 300   # max side for the enlarged crop
    INFO_H    = 58    # info text row height
    HDR_H     = 22    # column header height
    PAD       = 8
    BG        = (25, 25, 25)
    C_WHITE   = (240, 240, 240)
    C_GRAY    = (160, 160, 160)

    for cat, recs in per_cat.items():
        cat_dir = os.path.join(base, cat)
        os.makedirs(cat_dir, exist_ok=True)

        sample = random.sample(recs, min(n_per_cat, len(recs)))
        color    = colors.get(cat, (180, 180, 180))

        for idx, r in enumerate(sample):
            img_arr = r['_img_arr']
            box     = r['_box']

            img_h_orig, img_w_orig = img_arr.shape[:2]

            # ── Scaled full image ───────────────────────
            scale_img = min(IMG_MAX / max(img_w_orig, 1),
                            IMG_MAX / max(img_h_orig, 1), 1.0)
            disp_w = max(1, int(img_w_orig * scale_img))
            disp_h = max(1, int(img_h_orig * scale_img))

            img_pil  = Image.fromarray(img_arr).resize((disp_w, disp_h), Image.LANCZOS)
            draw_img = ImageDraw.Draw(img_pil)

            # Scaled bbox coordinates
            bx1 = int(box['x1'] * scale_img)
            by1 = int(box['y1'] * scale_img)
            bx2 = int(box['x2'] * scale_img)
            by2 = int(box['y2'] * scale_img)
            bcx = (bx1 + bx2) // 2
            bcy = (by1 + by2) // 2

            # Bbox rectangle (FP or FN — category colour)
            draw_img.rectangle([bx1, by1, bx2, by2], outline=color, width=2)

            # GT boxes in green
            C_GT = (0, 220, 80)
            for gt in r.get('_gt_boxes', []):
                gx1 = int(gt['x1'] * scale_img)
                gy1 = int(gt['y1'] * scale_img)
                gx2 = int(gt['x2'] * scale_img)
                gy2 = int(gt['y2'] * scale_img)
                draw_img.rectangle([gx1, gy1, gx2, gy2], outline=C_GT, width=2)
                draw_img.text((gx1 + 2, gy1 + 2), "GT", fill=C_GT, font=f_reg)

            # Circle around the bbox — radius = half diagonal + 12px margin
            bw_s = max(bx2 - bx1, 1)
            bh_s = max(by2 - by1, 1)
            radius = int(((bw_s**2 + bh_s**2) ** 0.5) / 2) + 14
            draw_img.ellipse(
                [bcx - radius, bcy - radius, bcx + radius, bcy + radius],
                outline=color, width=3
            )

            # ── Enlarged crop ───────────────────────────────
            crop = _crop_arr(img_arr, box, margin=10)
            if crop is None or crop.size == 0:
                crop = img_arr  # fallback: full image

            ch, cw = crop.shape[:2]
            scale_crop = min(CROP_MAX / max(cw, 1),
                             CROP_MAX / max(ch, 1), 4.0)  # max 4x upscale
            nw = max(1, int(cw * scale_crop))
            nh = max(1, int(ch * scale_crop))
            crop_pil  = Image.fromarray(crop).resize((nw, nh), Image.LANCZOS)
            draw_crop = ImageDraw.Draw(crop_pil)
            # Coloured border on the crop
            draw_crop.rectangle([0, 0, nw - 1, nh - 1], outline=color, width=3)

            # ── Final canvas ─────────────────────────────────
            canvas_w = PAD + disp_w + PAD + nw + PAD
            canvas_h = HDR_H + max(disp_h, nh) + PAD + INFO_H
            canvas   = Image.new('RGB', (canvas_w, canvas_h), BG)
            draw     = ImageDraw.Draw(canvas)

            # Column headers
            draw.rectangle([PAD, 0, PAD + disp_w, HDR_H], fill=(50, 50, 60))
            draw.text((PAD + 4, 4), "Full image", fill=C_WHITE, font=f_reg)
            x_crop = PAD + disp_w + PAD
            draw.rectangle([x_crop, 0, x_crop + nw, HDR_H], fill=(50, 50, 60))
            draw.text((x_crop + 4, 4), f"Crop ×{scale_crop:.1f}", fill=C_WHITE, font=f_reg)

            # Paste images
            canvas.paste(img_pil,  (PAD,    HDR_H))
            canvas.paste(crop_pil, (x_crop, HDR_H))

            # Info text row
            ty = HDR_H + max(disp_h, nh) + 4
            draw.text((PAD, ty),
                      f"{cat}  |  {r['img']}",
                      fill=color, font=f_bold)
            feat_txt = (f"brightness={r['brightness']:.0f}  "
                        f"laplacian={r['laplacian_var']:.0f}  "
                        f"saturation={r['saturation']:.0f}  "
                        f"texture={r['texture_var']:.0f}")
            draw.text((PAD, ty + 16), feat_txt, fill=C_GRAY, font=f_reg)
            if kind == 'fp':
                extra = (f"conf={r.get('conf', 0):.3f}  "
                         f"size={r['width']}×{r['height']}px")
            else:
                extra = (f"size={r['width']}×{r['height']}px  "
                         f"area={r.get('area', r['width']*r['height'])}px²")
            draw.text((PAD, ty + 30), extra, fill=C_GRAY, font=f_reg)

            img_stem = Path(r['img']).stem
            panel_name = f'{kind}_{cat}_{idx:03d}__{img_stem}.png'
            out_path = os.path.join(cat_dir, panel_name)
            canvas.save(out_path)

            # Row for the mapping CSV
            row_map = {
                'panel':      panel_name,
                'category':  cat,
                'kind':       kind,
                'img':        r['img'],
                'x1': box['x1'], 'y1': box['y1'],
                'x2': box['x2'], 'y2': box['y2'],
            }
            if kind == 'fp':
                row_map['conf'] = r.get('conf', '')
            else:
                row_map['area'] = r.get('area', r['width'] * r['height'])
            _mapping_rows.append(row_map)

    # Write mapping CSV
    if _mapping_rows:
        fieldnames = list(_mapping_rows[0].keys())
        with open(mapping_path, 'w', newline='') as mf:
            writer = csv.DictWriter(mf, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(_mapping_rows)
        print(f"  Mapping saved: {mapping_path}")

    print(f"  {kind.upper()} panels saved to: {base}/")


# ══════════════════════════════════════════════════════════════
#  TEXT STATISTICS
# ══════════════════════════════════════════════════════════════

def print_statistics(fp_records, fn_records, n_gt_total, n_fn_total, run_name, conf, iou_thresh, output_dir):
    """
    Print and save a text statistics report for one run.

        Computes FNR, per-category FP/FN counts, and FPPI, then writes the
        report to both stdout and a text file in *output_dir*.

        Args:
            fp_records:   List of FP error dicts.
            fn_records:   List of FN error dicts.
            n_gt_total:   Total number of GT boxes.
            n_fn_total:   Total number of FN boxes.
            run_name:     Run identifier string (used in headers).
            conf:         Confidence threshold used in this evaluation.
            iou_thresh:   IoU threshold used for matching.
            output_dir:   Directory where the .txt report is saved.

        Returns:
            Tuple (fp_counter, fn_counter, fnr, n_fp).
    """
    

    n_fp = len(fp_records)
    fnr  = n_fn_total / n_gt_total if n_gt_total > 0 else 0.0

    fp_cnt = Counter(r['category'] for r in fp_records)
    fn_cnt = Counter(r['category'] for r in fn_records)

    fp_cats_ordered = ['dark_region', 'motion_blur', 'occupied_shelf',
                       'refrigerator', 'other_fp']
    fn_cats_ordered = ['small_box', 'dark_fn', 'low_contrast',
                       'occluded', 'other_fn']

    lines = []
    lines.append(f"\n{'='*60}")
    lines.append(f"  ERROR ANALYSIS — {run_name}")
    iou_label = (f"IoU>0 (any positive overlap)"
                 if iou_thresh == 0.0 else f"IoU≥{iou_thresh}")
    lines.append(f"  conf≥{conf}   {iou_label}")
    lines.append(f"{'='*60}")
    lines.append(f"\n  GT totali:   {n_gt_total}")
    lines.append(f"  FN totali:   {n_fn_total}")
    lines.append(f"  FNR = FN/GT: {fnr:.4f}  ({fnr*100:.2f}%)")
    lines.append(f"  FP totali:   {n_fp}")
    lines.append(f"  FPPI:        {n_fp / max(1, len(set(r['img'] for r in fp_records + fn_records))):.3f}")

    lines.append(f"\n  ── FALSE POSITIVES by category ─────────────")
    lines.append(f"  {'Category':<20} {'N':>6} {'%FP':>8}  Description")
    lines.append(f"  {'─'*20} {'─'*6} {'─'*8}  {'─'*30}")
    desc_fp = {
        'dark_region':    'Dark region / shelf edge',
        'motion_blur':    'Camera motion blur',
        'occupied_shelf': 'Shelf with products',
        'refrigerator':   'Refrigerator / cold background',
        'other_fp': 'Other',
    }
    for cat in fp_cats_ordered:
        n   = fp_cnt.get(cat, 0)
        pct = 100 * n / n_fp if n_fp > 0 else 0
        lines.append(f"  {cat:<20} {n:>6} {pct:>7.1f}%  {desc_fp[cat]}")

    lines.append(f"\n  ── FALSE NEGATIVES by category ─────────────")
    lines.append(f"  {'Category':<20} {'N':>6} {'%FN':>8}  Description")
    lines.append(f"  {'─'*20} {'─'*6} {'─'*8}  {'─'*30}")
    desc_fn = {
        'small_box':    'GT box too small (<32px side)',
        'dark_fn':      'Shadow / low illumination',
        'low_contrast': 'Low-contrast shelf',
        'occluded':     'Partially occluded by another GT box',
        'other_fn': 'Other',
    }
    for cat in fn_cats_ordered:
        n   = fn_cnt.get(cat, 0)
        pct = 100 * n_fn_total / n_gt_total if cat == 'all' else (
              100 * n / n_fn_total if n_fn_total > 0 else 0)
        lines.append(f"  {cat:<20} {n:>6} {pct:>7.1f}%  {desc_fn[cat]}")

    lines.append(f"\n  Heuristic thresholds used:")
    lines.append(f"    THR_DARK={THR_DARK}  THR_BLUR={THR_BLUR}  "
                 f"THR_CONTRAST={THR_CONTRAST}")
    lines.append(f"    THR_SMALL_PX={THR_SMALL_PX}  THR_SAT_HIGH={THR_SAT_HIGH}  "
                 f"THR_TEXTURE={THR_TEXTURE}")
    lines.append(f"    THR_OCC={THR_OCC}  THR_REFRIG_B={THR_REFRIG_B}")
    lines.append(f"{'='*60}\n")

    summary_text = '\n'.join(lines)
    print(summary_text)

    txt_path = os.path.join(output_dir, 'error_statistics.txt')
    with open(txt_path, 'w') as f:
        f.write(summary_text)
    print(f"  Statistics TXT: {txt_path}")

    return fp_cnt, fn_cnt, fnr, n_fp


# ══════════════════════════════════════════════════════════════
#  PNG TABLE
# ══════════════════════════════════════════════════════════════

def render_stats_png(fp_cnt, fn_cnt, fnr, n_fp, n_gt_total, n_fn_total, run_name, output_dir):
    """
    Render a paper-style summary PNG table for a single run.

        Args:
            fp_cnt:      Counter of FP counts by category.
            fn_cnt:      Counter of FN counts by category.
            fnr:         False Negative Rate (FN / GT).
            n_fp:        Total number of false positives.
            n_gt_total:  Total number of GT boxes.
            n_fn_total:  Total number of false negatives.
            run_name:    Run identifier string (shown in table title).
            output_dir:  Directory where the PNG is saved.
    """    

    fp_cats = ['dark_region', 'motion_blur', 'occupied_shelf',
               'refrigerator', 'other_fp']
    fn_cats = ['small_box', 'dark_fn', 'low_contrast', 'occluded', 'other_fn']

    desc_fp = {
        'dark_region':    'Dark region / shelf edge',
        'motion_blur':    'Camera motion blur',
        'occupied_shelf': 'Shelf with products',
        'refrigerator':   'Refrigerator',
        'other_fp': 'Other',
    }
    desc_fn = {
        'small_box':    'GT box too small (<32px)',
        'dark_fn':      'Shadow area',
        'low_contrast': 'Low contrast',
        'occluded':     'GT occlusion',
        'other_fn': 'Other',
    }

    # Layout constants
    ROW_H  = 28
    HDR_H  = 34
    GRP_H  = 36
    PAD    = 14
    COL_W  = [180, 60, 80, 240]   # Category, N, %, Description
    total_w = sum(COL_W) + PAD * 2

    n_rows = len(fp_cats) + len(fn_cats) + 2  # +2 gruppi header
    total_h = HDR_H + n_rows * ROW_H + 2 * GRP_H + 60

    C_BG      = (255, 252, 248)
    C_HDR_BG  = (210, 180, 150)
    C_GRP_BG  = (240, 225, 210)
    C_ROW1    = (255, 252, 248)
    C_ROW2    = (248, 242, 235)
    C_TEXT    = (40,  40,  40)
    C_BEST    = (0,  130,   0)
    C_LINE    = (200, 185, 170)
    C_WHITE   = (255, 255, 255)

    try:
        F_HDR  = ImageFont.truetype(FONT_BOLD, 11)
        F_GRP  = ImageFont.truetype(FONT_BOLD, 12)
        F_CELL = ImageFont.truetype(FONT_REG,  11)
        F_BOLD = ImageFont.truetype(FONT_BOLD, 11)
    except Exception:
        F_HDR = F_GRP = F_CELL = F_BOLD = ImageFont.load_default()

    img  = Image.new('RGB', (total_w, total_h), C_BG)
    draw = ImageDraw.Draw(img)

    def x_col(i):
        return PAD + sum(COL_W[:i])

    def draw_cell(text, x, y, w, h, font, color=C_TEXT, bg=None, align='left'):
        if bg:
            draw.rectangle([x, y, x + w, y + h], fill=bg)
        try:
            bb = font.getbbox(str(text))
            tw = bb[2] - bb[0]
        except Exception:
            tw = len(str(text)) * 7
        tx = x + 6 if align == 'left' else x + (w - tw) // 2
        ty = y + (h - 13) // 2
        draw.text((tx, ty), str(text), fill=color, font=font)

    def draw_hline(y_):
        draw.line([(PAD, y_), (total_w - PAD, y_)], fill=C_LINE, width=1)

    # Title
    draw.text((PAD, 6),
              f"Error Analysis — {run_name}",
              fill=C_TEXT, font=F_HDR)

    # Column headers
    headers = ['Category', 'N', '%', 'Description']
    y = 20
    for ci, (h_txt, w) in enumerate(zip(headers, COL_W)):
        draw_cell(h_txt, x_col(ci), y, w, HDR_H, F_HDR,
                  bg=C_HDR_BG, color=C_WHITE,
                  align='center' if ci > 0 else 'left')
    draw_hline(y + HDR_H)
    y += HDR_H

    # Overall summary
    summary_bg = (230, 218, 205)
    draw.rectangle([PAD, y, total_w - PAD, y + ROW_H], fill=summary_bg)
    draw_cell(f"GT={n_gt_total}  FN={n_fn_total}  FNR={fnr*100:.2f}%  "
              f"FP={n_fp}  FPPI={n_fp/max(1,n_gt_total):.3f}",
              x_col(0), y, total_w - 2*PAD, ROW_H, F_BOLD, color=(60, 40, 20))
    draw_hline(y + ROW_H)
    y += ROW_H

    def section(title, cats, cnt, n_tot, desc_map):
        nonlocal y
        draw.rectangle([PAD, y, total_w - PAD, y + GRP_H], fill=C_GRP_BG)
        draw.text((PAD + 8, y + (GRP_H - 13) // 2), title, fill=C_TEXT, font=F_GRP)
        draw_hline(y + GRP_H)
        y += GRP_H

        # find the max value for the green highlight
        max_n = max((cnt.get(c, 0) for c in cats), default=0)

        for ri, cat in enumerate(cats):
            n   = cnt.get(cat, 0)
            pct = 100 * n / n_tot if n_tot > 0 else 0
            bg  = C_ROW1 if ri % 2 == 0 else C_ROW2
            draw.rectangle([PAD, y, total_w - PAD, y + ROW_H], fill=bg)
            color_n = C_BEST if (n == max_n and max_n > 0) else C_TEXT
            draw_cell(cat,            x_col(0), y, COL_W[0], ROW_H, F_CELL)
            draw_cell(str(n),         x_col(1), y, COL_W[1], ROW_H, F_BOLD,
                      color=color_n, align='center')
            draw_cell(f"{pct:.1f}%",  x_col(2), y, COL_W[2], ROW_H, F_CELL,
                      color=color_n, align='center')
            draw_cell(desc_map.get(cat, ''), x_col(3), y, COL_W[3], ROW_H, F_CELL)
            draw_hline(y + ROW_H)
            y += ROW_H

    section('FALSE POSITIVES (FP)', fp_cats, fp_cnt, n_fp,   desc_fp)
    section('FALSE NEGATIVES (FN)', fn_cats, fn_cnt, n_fn_total, desc_fn)

    # Threshold legend
    draw.text((PAD, y + 4),
              f"Thresholds: dark<{THR_DARK}  blur_lap<{THR_BLUR}  "
              f"contrast_var<{THR_CONTRAST}  small<{THR_SMALL_PX}px  "
              f"occ_iou>{THR_OCC}  refrig_b>{THR_REFRIG_B}",
              fill=(130, 110, 90), font=F_CELL)

    draw.rectangle([PAD, 0, total_w - PAD, y + 50], outline=C_LINE, width=1)

    png_path = os.path.join(output_dir, 'error_statistics.png')
    img.save(png_path)
    print(f"  PNG table: {png_path}")


# ══════════════════════════════════════════════════════════════
#  COMPARATIVE PNG TABLE (multiple runs)
# ══════════════════════════════════════════════════════════════

def render_comparative_png(results, output_dir):
    """
    PNG table placing the results of multiple runs side by side.
    results: list of dicts with keys:
      run_name, fp_cnt, fn_cnt, fnr, n_fp, n_gt_total, n_fn_total
    """
    fp_cats = ['dark_region', 'motion_blur', 'occupied_shelf',
               'refrigerator', 'other_fp']
    fn_cats = ['small_box', 'dark_fn', 'low_contrast', 'occluded', 'other_fn']

    desc = {
        'dark_region':    'Dark region / shelf edge',
        'motion_blur':    'Camera motion blur',
        'occupied_shelf': 'Shelf with products',
        'refrigerator':   'Refrigerator',
        'other_fp':       'Other FP',
        'small_box':      'Small GT box (<32px)',
        'dark_fn':        'Shadow',
        'low_contrast':   'Low contrast',
        'occluded':       'GT occlusion',
        'other_fn':       'Other FN',
    }

    n_runs  = len(results)
    ROW_H   = 28
    HDR_H   = 40
    GRP_H   = 32
    PAD     = 14
    CAT_W   = 175
    DESC_W  = 200

    try:
        F_HDR  = ImageFont.truetype(FONT_BOLD, 11)
        F_GRP  = ImageFont.truetype(FONT_BOLD, 12)
        F_CELL = ImageFont.truetype(FONT_REG,  10)
        F_BOLD = ImageFont.truetype(FONT_BOLD, 10)
    except Exception:
        F_HDR = F_GRP = F_CELL = F_BOLD = ImageFont.load_default()

    def _text_w(text, font):
        try:
            bb = font.getbbox(str(text))
            return bb[2] - bb[0]
        except Exception:
            return len(str(text)) * 7

    # RUN_W must fit the longest run name (header) plus padding, otherwise
    # long run names (e.g. "oos_detector_only_ALBSV-Noaug") overflow into
    # the neighboring column instead of being clipped or wrapped.
    RUN_W = max(130, max((_text_w(r['run_name'], F_HDR) for r in results), default=0) + 20)
    total_w = PAD + CAT_W + n_runs * RUN_W + DESC_W + PAD

    n_rows  = n_runs + 1 + len(fp_cats) + 1 + len(fn_cats)
    total_h = HDR_H + n_rows * ROW_H + 2 * GRP_H + 50

    C_BG     = (255, 252, 248)
    C_HDR_BG = (210, 180, 150)
    C_GRP_BG = (240, 225, 210)
    C_ROW1   = (255, 252, 248)
    C_ROW2   = (248, 242, 235)
    C_TEXT   = (40,  40,  40)
    C_LINE   = (200, 185, 170)
    C_WHITE  = (255, 255, 255)
    C_BEST   = (0,  130,   0)

    img  = Image.new('RGB', (total_w, total_h), C_BG)
    draw = ImageDraw.Draw(img)

    def draw_cell(text, x, y, w, h, font, color=C_TEXT, bg=None, align='left'):
        if bg:
            draw.rectangle([x, y, x + w, y + h], fill=bg)
        try:
            bb = font.getbbox(str(text))
            tw = bb[2] - bb[0]
        except Exception:
            tw = len(str(text)) * 7
        tx = x + 6 if align == 'left' else x + (w - tw) // 2
        ty = y + (h - 13) // 2
        draw.text((tx, ty), str(text), fill=color, font=font)

    def draw_hline(y_):
        draw.line([(PAD, y_), (total_w - PAD, y_)], fill=C_LINE, width=1)

    # Header
    y = 0
    draw_cell('Category', PAD, y, CAT_W, HDR_H, F_HDR, bg=C_HDR_BG, color=C_WHITE)
    for ri, r in enumerate(results):
        rx = PAD + CAT_W + ri * RUN_W
        draw_cell(r['run_name'], rx, y, RUN_W, HDR_H,
                  F_HDR, bg=C_HDR_BG, color=C_WHITE, align='center')
    draw_cell('Description', PAD + CAT_W + n_runs * RUN_W, y,
              DESC_W, HDR_H, F_HDR, bg=C_HDR_BG, color=C_WHITE)
    draw_hline(HDR_H)
    y += HDR_H

    # Summary (one line per run so long run names never overflow the canvas)
    for r in results:
        draw.rectangle([PAD, y, total_w - PAD, y + ROW_H], fill=(230, 218, 205))
        line = (f"{r['run_name']}: GT={r['n_gt_total']} FP={r['n_fp']} "
                f"FN={r['n_fn_total']} FNR={r['fnr']*100:.1f}%")
        draw_cell(line, PAD, y, total_w - 2*PAD, ROW_H, F_BOLD, color=(60, 40, 20))
        y += ROW_H
    draw_hline(y)

    def section(title, cats, kind):
        nonlocal y
        draw.rectangle([PAD, y, total_w - PAD, y + GRP_H], fill=C_GRP_BG)
        draw.text((PAD + 8, y + (GRP_H - 13) // 2),
                  title, fill=C_TEXT, font=F_GRP)
        draw_hline(y + GRP_H)
        y += GRP_H
        for ri_cat, cat in enumerate(cats):
            ns   = [r['fp_cnt' if kind == 'fp' else 'fn_cnt'].get(cat, 0)
                    for r in results]
            max_n = max(ns) if ns else 0
            bg = C_ROW1 if ri_cat % 2 == 0 else C_ROW2
            draw.rectangle([PAD, y, total_w - PAD, y + ROW_H], fill=bg)
            draw_cell(cat, PAD, y, CAT_W, ROW_H, F_CELL)
            for ri, r in enumerate(results):
                cnt   = r['fp_cnt' if kind == 'fp' else 'fn_cnt']
                tot   = r['n_fp'] if kind == 'fp' else r['n_fn_total']
                n     = cnt.get(cat, 0)
                pct   = 100 * n / tot if tot > 0 else 0
                rx    = PAD + CAT_W + ri * RUN_W
                color = C_BEST if (n == max_n and max_n > 0) else C_TEXT
                draw_cell(f"{n} ({pct:.0f}%)", rx, y, RUN_W, ROW_H,
                          F_CELL, color=color, align='center')
            draw_cell(desc.get(cat, ''),
                      PAD + CAT_W + n_runs * RUN_W, y, DESC_W, ROW_H, F_CELL)
            draw_hline(y + ROW_H)
            y += ROW_H

    section('FALSE POSITIVES (FP)', fp_cats, 'fp')
    section('FALSE NEGATIVES (FN)', fn_cats, 'fn')

    draw.text((PAD, y + 4),
              f"Green=highest value per row  |  "
              f"Thresholds: dark<{THR_DARK} blur<{THR_BLUR} "
              f"small<{THR_SMALL_PX}px sat>{THR_SAT_HIGH} "
              f"tex>{THR_TEXTURE} occ>{THR_OCC} refrig>{THR_REFRIG_B}",
              fill=(130, 110, 90), font=F_CELL)
    draw.rectangle([PAD, 0, total_w - PAD, y + 46], outline=C_LINE, width=1)

    png_path = os.path.join(output_dir, 'comparative_statistics.png')
    img.save(png_path)
    print(f"  Comparative PNG table: {png_path}")


# ══════════════════════════════════════════════════════════════
#  MATPLOTLIB CHARTS
# ══════════════════════════════════════════════════════════════

FP_CATS = ['dark_region', 'motion_blur', 'refrigerator', 'occupied_shelf', 'other_fp']
FN_CATS = ['small_box', 'dark_fn', 'occluded', 'low_contrast', 'other_fn']

# Colors per category — consistent across all charts
CAT_COLORS_FP = {
    'dark_region':    '#9B59B6',   # purple
    'motion_blur':    '#8FBC44',   # olive green
    'refrigerator':   '#D4AC0D',   # ochre yellow
    'occupied_shelf': '#E07B54',   # orange
    'other_fp':       '#E74C3C',   # red
}
CAT_COLORS_FN = {
    'small_box':    '#8E44AD',     # dark purple
    'dark_fn':      '#2ECC71',     # green
    'occluded':     '#95A5A6',     # gray
    'low_contrast': '#5DADE2',     # light blue
    'other_fn':     '#3498DB',     # blue
}


def plot_bar_charts(results, output_dir):
    """
    Produce a PNG with 4 subplots (2 rows x n_runs columns):
      row 0: FP histograms for each run
      row 1: FN histograms for each run
    """
    matplotlib.use('Agg')

    n_runs = len(results)
    fig, axes = plt.subplots(2, n_runs,
                             figsize=(7 * n_runs, 10),
                             constrained_layout=True)
    if n_runs == 1:
        axes = [[axes[0]], [axes[1]]]

    fig.suptitle(f"Error Analysis — {n_runs} models ({IMGSZ}px images)",
                 fontsize=14, fontweight='bold')

    for ci, r in enumerate(results):
        # ── FP ────────────────────────────────────────────────
        ax_fp = axes[0][ci]
        fp_vals = [r['fp_cnt'].get(c, 0) for c in FP_CATS]
        fp_pcts = [100 * v / r['n_fp'] if r['n_fp'] > 0 else 0
                   for v in fp_vals]
        colors_fp = [CAT_COLORS_FP[c] for c in FP_CATS]
        bars = ax_fp.barh(FP_CATS, fp_vals, color=colors_fp, alpha=0.85)
        for bar, pct in zip(bars, fp_pcts):
            ax_fp.text(bar.get_width() + max(fp_vals) * 0.01,
                       bar.get_y() + bar.get_height() / 2,
                       f"{pct:.1f}%", va='center', fontsize=10)
        ax_fp.set_title(f"{r['run_name']}\nFP totali: {r['n_fp']}",
                        fontsize=11)
        ax_fp.set_xlabel('Count', fontsize=9)
        ax_fp.set_ylabel('False Positives', fontsize=10)
        ax_fp.invert_yaxis()
        ax_fp.set_xlim(0, max(fp_vals) * 1.18 if fp_vals else 1)
        ax_fp.spines['top'].set_visible(False)
        ax_fp.spines['right'].set_visible(False)

        # ── FN ────────────────────────────────────────────────
        ax_fn = axes[1][ci]
        fn_vals = [r['fn_cnt'].get(c, 0) for c in FN_CATS]
        fn_pcts = [100 * v / r['n_fn_total'] if r['n_fn_total'] > 0 else 0
                   for v in fn_vals]
        colors_fn = [CAT_COLORS_FN[c] for c in FN_CATS]
        bars = ax_fn.barh(FN_CATS, fn_vals, color=colors_fn, alpha=0.85)
        for bar, pct in zip(bars, fn_pcts):
            ax_fn.text(bar.get_width() + max(fn_vals) * 0.01,
                       bar.get_y() + bar.get_height() / 2,
                       f"{pct:.1f}%", va='center', fontsize=10)
        ax_fn.set_title(f"FN totali: {r['n_fn_total']}  (FNR {r['fnr']*100:.1f}%)",
                        fontsize=11)
        ax_fn.set_xlabel('Count', fontsize=9)
        ax_fn.set_ylabel('False Negatives', fontsize=10)
        ax_fn.invert_yaxis()
        ax_fn.set_xlim(0, max(fn_vals) * 1.18 if fn_vals else 1)
        ax_fn.spines['top'].set_visible(False)
        ax_fn.spines['right'].set_visible(False)

    png_path = os.path.join(output_dir, 'error_chart.png')
    fig.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Bar chart:          {png_path}")


def plot_delta_chart(results, output_dir):
    """
    Produce a PNG with 2 side-by-side subplots (delta FP% and delta FN%):
    compares run[1] vs run[0] — negative = fewer errors = better.
    Only meaningful with exactly 2 runs.
    """
    if len(results) != 2:
        return

    matplotlib.use('Agg')

    r0, r1 = results[0], results[1]

    def pct(cnt, cat, tot):
        return 100 * cnt.get(cat, 0) / tot if tot > 0 else 0.0

    delta_fp = {c: pct(r1['fp_cnt'], c, r1['n_fp'])
                 - pct(r0['fp_cnt'], c, r0['n_fp'])
                for c in FP_CATS}
    delta_fn = {c: pct(r1['fn_cnt'], c, r1['n_fn_total'])
                 - pct(r0['fn_cnt'], c, r0['n_fn_total'])
                for c in FN_CATS}

    fig, (ax_fp, ax_fn) = plt.subplots(1, 2, figsize=(13, 5),
                                        constrained_layout=True)
    fig.suptitle(f"error delta: {r1['run_name']} vs {r0['run_name']}\n"
                 f"(negative = fewer errors = better)",
                 fontsize=13, fontweight='bold')

    def _bar_delta(ax, cats, deltas, colors_map, xlabel):
        vals   = [deltas[c] for c in cats]
        colors = ['#E74C3C' if v > 0 else '#3498DB' for v in vals]
        ax.barh(cats, vals, color=colors, alpha=0.80)
        ax.axvline(0, color='black', linewidth=1, linestyle='--')
        ax.set_xlabel(xlabel, fontsize=10)
        ax.invert_yaxis()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # value labels
        for i, v in enumerate(vals):
            ax.text(v + (0.02 if v >= 0 else -0.02), i,
                    f"{v:+.2f}", va='center',
                    ha='left' if v >= 0 else 'right', fontsize=9)

    _bar_delta(ax_fp, FP_CATS, delta_fp, CAT_COLORS_FP, 'delta percentage points')
    _bar_delta(ax_fn, FN_CATS, delta_fn, CAT_COLORS_FN, 'delta percentage points')
    ax_fp.set_title('FP delta (%)', fontsize=11)
    ax_fn.set_title('FN delta (%)', fontsize=11)

    png_path = os.path.join(output_dir, 'delta_chart.png')
    fig.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Delta chart:        {png_path}")


# ══════════════════════════════════════════════════════════════
#  SINGLE RUN ANALYSIS
# ══════════════════════════════════════════════════════════════

def analyse_run(run_name, runs_dir, image_dir, label_dir, output_base, conf, iou_thresh, seed, device):
    """
    Run the full analysis on a single run and return the results dict
    needed for the comparative table.
    """
    run_dir    = os.path.join(runs_dir, run_name)
    weights    = os.path.join(run_dir, 'weights', 'best.pt')
    img_dir    = image_dir
    lbl_dir    = label_dir
    output_dir = os.path.join(output_base, run_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Error Analysis — {run_name}")
    print(f"  Weights: {weights}")
    print(f"  Output:  {output_dir}")
    print(f"{'='*60}\n")

    for p, name in [(weights, 'best.pt'), (img_dir, 'img_dir'), (lbl_dir, 'lbl_dir')]:
        if not os.path.exists(p):
            print(f"  [ERROR] {name} not found: {p}")
            return None

    model = YOLO(weights)
    print(f"  Model loaded: {weights}")

    ext = ('.jpg', '.jpeg', '.png')
    img_paths = sorted(p for p in Path(img_dir).iterdir()
                       if p.suffix.lower() in ext)
    print(f"  Images: {len(img_paths)}")

    iou_label = f"IoU>0" if iou_thresh == 0.0 else f"IoU>={iou_thresh}"
    print(f"\n  Analysis in progress (conf>={conf}, {iou_label})...")
    fp_records, fn_records, n_gt_total, n_fn_total = collect_errors(
        img_paths, lbl_dir, model,
        conf=conf, iou_thresh=iou_thresh,
        device=device, imgsz=IMGSZ,
    )
    print(f"  FP found: {len(fp_records)}")
    print(f"  FN found: {n_fn_total}  /  total GT: {n_gt_total}")

    save_csv(fp_records, _FP_FIELDS, os.path.join(output_dir, 'fp_errors.csv'))
    save_csv(fn_records, _FN_FIELDS, os.path.join(output_dir, 'fn_errors.csv'))

    save_crop_panels(fp_records, 'fp', output_dir, N_PANELS_PER_CAT, seed)
    save_crop_panels(fn_records, 'fn', output_dir, N_PANELS_PER_CAT, seed)

    fp_cnt, fn_cnt, fnr, n_fp = print_statistics(
        fp_records, fn_records, n_gt_total, n_fn_total,
        run_name, conf, iou_thresh, output_dir,
    )

    render_stats_png(fp_cnt, fn_cnt, fnr, n_fp, n_gt_total, n_fn_total,
                run_name, output_dir)

    print(f"\n  Done — {run_name}. Output in: {output_dir}\n")

    return {
        'run_name':   run_name,
        'fp_cnt':     fp_cnt,
        'fn_cnt':     fn_cnt,
        'fnr':        fnr,
        'n_fp':       n_fp,
        'n_gt_total': n_gt_total,
        'n_fn_total': n_fn_total,
    }


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    global FONT_BOLD, FONT_REG

    args = parse_args()
    FONT_BOLD = args.font_bold
    FONT_REG  = args.font_regular

    random.seed(args.seed)

    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # A single --run overrides --runs
    runs = [args.run] if args.run else args.runs

    output_base = args.output
    os.makedirs(output_base, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Error Analysis — {len(runs)} run(s)")
    for r in runs:
        print(f"    - {r}")
    print(f"  Output base: {output_base}")
    print(f"{'='*60}")

    results = []
    for run_name in runs:
        res = analyse_run(run_name, args.runs_dir, args.image_dir, args.label_dir,
                           output_base, args.conf, args.iou, args.seed, device)
        if res is not None:
            results.append(res)

    if len(results) >= 2:
        print(f"\n  Generating comparative table...")
        render_comparative_png(results, output_base)
        plot_bar_charts(results, output_base)
        plot_delta_chart(results, output_base)
    elif len(results) == 1:
        plot_bar_charts(results, output_base)

    print(f"\n{'='*60}")
    print(f"  Analysis complete. Output: {output_base}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()