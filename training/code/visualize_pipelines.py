"""
visualize_pipelines.py
======================
Generate visual panels and summary tables comparing the three
training pipelines defined in train_joint_v2.py:

  detector_only → YOLO26n on original images (baseline)
  gan_frozen    → GAN pre-processes video frames; GAN weights frozen
  joint         → GAN pre-processes video frames; GAN weights updated
                  by the detection loss (end-to-end joint training)

Tables produced:
  Table 1 — Main metrics:
    Precision | Recall | F1
    mAP@50 | mAP@50-95 | empty (per-class mAP@50)
    box_loss | cls_loss | dfl_loss | tot_loss

  Tables 2a/2b/2c — FPR / FNR / TPR (non-greedy matching):
    FPR@c{0.25,0.5} × IoU{0.0,0.25,0.5,0.75}
    FNR@c{0.25,0.5} × IoU{0.0,0.25,0.5,0.75}
    TPR@c{0.25,0.5} × IoU{0.0,0.25,0.5,0.75}
    #FP | #FN | #TP | #GT

    FPR = (predictions with empty match list) / #GT
    FNR = (GT boxes with empty match list)    / #GT
    IoU threshold 0.0: a match requires only iou > 0 (any positive overlap)

Output:
  <output_dir>/panels/<run>/compare_<name>.png
  <output_dir>/table1_metrics.png
  <output_dir>/table2a_fpr.png
  <output_dir>/table2b_fnr.png
  <output_dir>/table2c_tpr.png
  <output_dir>/summary_table.txt
  <output_dir>/table*.tsv

Usage:
  python visualize_pipelines.py \\
      --images <path/to/test/images> \\
      --labels <path/to/test/labels> \\
      --weights-detector <path/to/detector_only/weights/best.pt> \\
      --weights-frozen   <path/to/gan_frozen/weights/best.pt> \\
      --weights-joint    <path/to/joint/weights/best.pt> \\
      --gan-ckpt         <path/to/latest_net_G.pth> \\
      --blur2blur-dir    <path/to/Blur2Blur> \\
      --data-yaml        <path/to/data.yaml> \\
      --output           <path/to/results/visualizations_pipelines>
"""
import os
import sys
import csv
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'utils'))
from viz_utils import draw_bboxes_mpl, new_row_figure, save_figure

GAN_SIZE    = 256

CONF_THRESHOLDS  = [0.25, 0.5]              # confidence thresholds for FPR/FNR
IOU_THRESHOLDS   = [0.0, 0.25, 0.5, 0.75]  # IoU thresholds for FPR/FNR matching
CONF_PREDICT_MIN = min(CONF_THRESHOLDS)     # = 0.25

CLASS_NAMES = {0: 'empty'}

# Populated from CLI args at the start of main()
FONT_BOLD = None
FONT_REG  = None


# ══════════════════════════════════════════════════════════════
#  ARGPARSE
# ══════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description='Compare detector_only / gan_frozen / joint pipelines with panels and summary tables.'
    )
    p.add_argument('--images', type=str, required=True,
                    help='Path to the test images directory.')
    p.add_argument('--labels', type=str, required=True,
                    help='Path to the test labels directory (YOLO format).')
    p.add_argument('--weights-detector', type=str, required=True,
                    help='Path to the detector_only run weights/best.pt.')
    p.add_argument('--weights-frozen', type=str, required=True,
                    help='Path to the gan_frozen run weights/best.pt.')
    p.add_argument('--weights-joint', type=str, required=True,
                    help='Path to the joint run weights/best.pt.')
    p.add_argument('--gan-ckpt', type=str, required=True,
                    help='Pre-trained GAN checkpoint (e.g. latest_net_G.pth), used for gan_frozen.')
    p.add_argument('--gan-ckpt-joint', type=str, default=None,
                    help='GAN checkpoint updated by joint training. If omitted, falls back to --gan-ckpt.')
    p.add_argument('--blur2blur-dir', type=str, required=True,
                    help='Path to the Blur2Blur repository root.')
    p.add_argument('--data-yaml', type=str, required=True,
                    help='Path to data.yaml, used to derive the val/test splits for the summary table.')
    p.add_argument('--output', type=str, required=True,
                    help='Directory where panels, tables, and TSV files will be saved.')
    p.add_argument('--n-samples', type=int, default=10,
                    help='Number of test images to sample for visual panels (default: 10).')
    p.add_argument('--conf', type=float, default=CONF_PREDICT_MIN,
                    help=f'Confidence threshold for prediction panels (default: {CONF_PREDICT_MIN}).')
    p.add_argument('--imgsz', type=int, default=640,
                    help='Inference image size (default: 640).')
    p.add_argument('--seed', type=int, default=42,
                    help='Random seed for image sampling (default: 42).')
    p.add_argument('--skip-panels', action='store_true',
                    help='Skip visual panel generation, only produce the summary tables.')
    p.add_argument('--skip-table', action='store_true',
                    help='Skip the summary table generation, only produce visual panels.')
    p.add_argument('--font-bold', type=str,
                    default='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
                    help='Path to a bold TTF font used for table/panel titles.')
    p.add_argument('--font-regular', type=str,
                    default='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                    help='Path to a regular TTF font used for table/panel body text.')
    return p.parse_args()

# ══════════════════════════════════════════════════════════════
#  GAN
# ══════════════════════════════════════════════════════════════

def load_gan(gan_ckpt_path, device, blur2blur_root=None):
    """
    Instantiate Blur2BlurWrapper and load generator weights from
    gan_ckpt_path via gan.load_generator_weights(), which already handles
    a possible 'model.' key-prefix mismatch internally.

    Args:
        gan_ckpt_path:  Path to the generator checkpoint (.pth/.pt).
        device:         Torch device.
        blur2blur_root: Path to the Blur2Blur repository root, forwarded
                         to Blur2BlurWrapper (overrides its own default
                         path guess).

    Returns:
        Blur2BlurWrapper with generator loaded onto *device*.
    """
    from gan_wrapper import Blur2BlurWrapper
    gan = Blur2BlurWrapper(blur2blur_root=blur2blur_root)

    if os.path.exists(gan_ckpt_path):
        gan.load_generator_weights(gan_ckpt_path)
        print(f"  GAN loaded: {gan_ckpt_path}")
    else:
        print(f"  [WARN] GAN checkpoint not found: {gan_ckpt_path} "
              f"-- generator keeps its freshly-initialized weights")

    gan.generator.to(device)
    gan.generator.eval()
    return gan


def load_gan_joint(gan_ckpt_joint_path, device, blur2blur_root=None):
    """
    Load the GAN with weights updated by joint training.

    Different from the pre-trained GAN: the weights have been modified
    by backpropagation of YOLO26's detection loss.
    Returns None if the checkpoint doesn't exist (falls back to the
    pre-trained GAN).
    """
    if not gan_ckpt_joint_path or not os.path.exists(gan_ckpt_joint_path):
        print(f"  [INFO] gan_ckpt_joint not found — using the pre-trained GAN for joint as well")
        return None
    return load_gan(gan_ckpt_joint_path, device, blur2blur_root=blur2blur_root)


def compute_gan_statistics(img_orig, img_gan, label=""):
    """
    Compute quantitative statistics on the GAN output:
    - PSNR (Peak Signal-to-Noise Ratio): how different the enhanced image is
    - Variance: measures the image's detail/sharpness
    - Mean pixel difference
    """
    a = np.array(img_orig).astype(np.float32) / 255.0
    b = np.array(img_gan).astype(np.float32) / 255.0
    mse = np.mean((a - b) ** 2)
    psnr = 10 * np.log10(1.0 / mse) if mse > 1e-10 else float('inf')
    var_orig = float(np.var(a))
    var_gan  = float(np.var(b))
    mean_diff = float(np.abs(a - b).mean()) * 255
    return {
        'label':     label,
        'psnr':      psnr,
        'var_orig':  var_orig,
        'var_gan':   var_gan,
        'mean_diff': mean_diff,
    }


def apply_gan(gan, img_tensor, device):
    """(1,3,H,W) in [0,1] → (1,3,H,W) in [0,1]"""
    with torch.no_grad():
        norm    = img_tensor * 2.0 - 1.0
        res     = F.interpolate(norm, size=(GAN_SIZE, GAN_SIZE),
                                mode='bilinear', align_corners=False)
        out     = gan.generator(res)
        if isinstance(out, (list, tuple)):
            out = out[-1]
        out = F.interpolate(out, size=(img_tensor.shape[2], img_tensor.shape[3]),
                            mode='bilinear', align_corners=False)
        return torch.clamp((out + 1.0) / 2.0, 0.0, 1.0)


def tensor_to_pil(t):
    """Convert a float tensor (1, 3, H, W) in [0, 1] to a PIL image."""
    arr = (t.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def pil_to_tensor(img_pil, device):
    """Convert a PIL image to a float tensor (1, 3, H, W) in [0, 1] on *device*."""
    arr = np.array(img_pil).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


# ══════════════════════════════════════════════════════════════
#  YOLO LABELS
# ══════════════════════════════════════════════════════════════

def read_yolo_labels(label_path, img_w, img_h):
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = int((xc - w / 2) * img_w)
            y1 = int((yc - h / 2) * img_h)
            x2 = int((xc + w / 2) * img_w)
            y2 = int((yc + h / 2) * img_h)
            boxes.append({'cls': cls, 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})
    return boxes


# ══════════════════════════════════════════════════════════════
#  YOLO PREDICTIONS
# ══════════════════════════════════════════════════════════════

def predict_yolo(model, img_pil, args, device):
    results = model.predict(source=img_pil, imgsz=args.imgsz,
                            conf=args.conf, device=device, verbose=False)
    boxes = []
    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            boxes.append({
                'cls':  int(box.cls[0].cpu()),
                'conf': float(box.conf[0].cpu()),
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
            })
    return boxes


# ══════════════════════════════════════════════════════════════
#  VISUAL PANEL
# ══════════════════════════════════════════════════════════════

def create_panel(img_orig_pil, img_gan_frozen_pil, img_gan_joint_pil,
                  gt_boxes, pred_detector, pred_frozen, pred_joint,
                  img_name, args):
    """
    Build a fig07-style side-by-side comparison figure (matplotlib):

      detector_only | gan_frozen | joint

    Each panel shows the image that pipeline's detector actually sees
    (original for detector_only, GAN output for gan_frozen/joint), with
    GT boxes in green and that pipeline's predictions in red.

    Returns a matplotlib Figure.
    """
    def to_pixel_boxes(boxes, conf_thresh):
        return [[b['x1'], b['y1'], b['x2'], b['y2']]
                for b in boxes if b.get('conf', 1.0) >= conf_thresh]

    gt_pixel = [[b['x1'], b['y1'], b['x2'], b['y2']] for b in gt_boxes]

    panels = [
        ('Detector only', np.array(img_orig_pil),       to_pixel_boxes(pred_detector, args.conf)),
        ('GAN frozen',    np.array(img_gan_frozen_pil),  to_pixel_boxes(pred_frozen,   args.conf)),
        ('GAN joint',     np.array(img_gan_joint_pil),   to_pixel_boxes(pred_joint,    args.conf)),
    ]

    fig, axes = new_row_figure(3, panel_size=(4.5, 4.5))
    for i, (ax, (title, img, pred_pixel)) in enumerate(zip(axes, panels)):
        ax.imshow(img)
        ax.set_title(f"{title}\nGT:{len(gt_pixel)}  Pred:{len(pred_pixel)}")
        if gt_pixel:
            # only label the legend once, on the first panel, to avoid clutter
            draw_bboxes_mpl(ax, gt_pixel, color='lime', label='GT' if i == 0 else None)
        if pred_pixel:
            draw_bboxes_mpl(ax, pred_pixel, color='red', label='pred' if i == 0 else None)

    fig.suptitle(f"OOS Pipeline Comparison — {img_name}  (conf>={args.conf})",
                 fontsize=13, fontweight='bold', y=1.05)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════
#  METRICS — FPR/FNR with non-greedy multi-IoU matching
# ══════════════════════════════════════════════════════════════

def _iou(a, b):
    ix1 = max(a['x1'], b['x1']); iy1 = max(a['y1'], b['y1'])
    ix2 = min(a['x2'], b['x2']); iy2 = min(a['y2'], b['y2'])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    aa = (a['x2'] - a['x1']) * (a['y2'] - a['y1'])
    ab = (b['x2'] - b['x1']) * (b['y2'] - b['y1'])
    return inter / (aa + ab - inter + 1e-6)


def _compute_tp_fp_fn(pred_batch, gt_batch, iou_thresh):
    """
    Non-greedy matching (mat_pred/mat_gt) per image at a given IoU threshold.

    FP = a prediction with no matched GT   → mat_pred[i] is empty
    FN = a GT box with no matched prediction → mat_gt[j] is empty

    FPR = #FP / #GT
    FNR = #FN / #GT
    """
    n_tp = 0; n_fp = 0; n_fn = 0
    n_gt_total = 0; n_pred_total = 0

    for preds, gts in zip(pred_batch, gt_batch):
        n_preds = len(preds); n_gts = len(gts)
        n_gt_total   += n_gts
        n_pred_total += n_preds

        mat_pred = [[] for _ in range(n_preds)]
        mat_gt   = [[] for _ in range(n_gts)]

        for pi, p in enumerate(preds):
            for gi, g in enumerate(gts):
                iou = _iou(p, g)
                if (iou_thresh > 0 and iou >= iou_thresh) or \
                   (iou_thresh == 0 and iou > 0):
                    n_tp += 1
                    mat_pred[pi].append(gi)
                    mat_gt[gi].append(pi)

        n_fp += sum(1 for lst in mat_pred if len(lst) == 0)
        n_fn += sum(1 for lst in mat_gt   if len(lst) == 0)

    print(f"  IoU≥{iou_thresh:.2f}: TP={n_tp}  FP={n_fp}  FN={n_fn}  #GT={n_gt_total}  #pred={n_pred_total}")
    return n_tp, n_fp, n_fn


def _fpr_fnr_multi_iou(model, data_yaml, split, imgsz, device):
    """
    Compute FPR and FNR for all IoU x conf-threshold combinations.
    Predicts with conf=CONF_PREDICT_MIN, then filters per confidence threshold.
    """
    import yaml
    from PIL import Image as _Img

    try:
        with open(data_yaml) as f:
            cfg = yaml.safe_load(f)
        base    = Path(cfg.get('path', ''))
        rel     = cfg.get(split, '')
        img_dir = (base / rel) if not Path(rel).is_absolute() else Path(rel)
        if not img_dir.exists():
            img_dir = base / rel / 'images'
        if not img_dir.exists():
            print(f"  [WARN] img dir not found: {img_dir}")
            return None
    except Exception as e:
        print(f"  [WARN] error reading data.yaml: {e}")
        return None

    IMG_EXT   = {'.jpg', '.jpeg', '.png', '.bmp'}
    img_paths = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT)
    if not img_paths:
        print(f"  [WARN] no images found in {img_dir}")
        return None

    lbl_dir = Path(str(img_dir).replace('images', 'labels'))
    if not lbl_dir.exists():
        lbl_dir = img_dir.parent / 'labels'
    if not lbl_dir.exists():
        print(f"  [WARN] labels dir not found: {lbl_dir}")
        return None

    print(f"    FPR/FNR: {len(img_paths)} images (predict conf={CONF_PREDICT_MIN})...", flush=True)

    all_preds_raw = []
    gt_batch      = []
    n_gt_total    = 0

    for img_path in img_paths:
        lbl_path = lbl_dir / (img_path.stem + '.txt')
        gts = []
        if lbl_path.exists():
            try:
                with _Img.open(img_path) as im:
                    W, H = im.size
            except Exception:
                W, H = imgsz, imgsz
            with open(lbl_path) as f:
                for line in f:
                    p = line.strip().split()
                    if len(p) < 5:
                        continue
                    xc, yc, w, h = float(p[1]), float(p[2]), float(p[3]), float(p[4])
                    gts.append({'x1': int((xc - w/2)*W), 'y1': int((yc - h/2)*H),
                                'x2': int((xc + w/2)*W), 'y2': int((yc + h/2)*H)})
        gt_batch.append(gts)
        n_gt_total += len(gts)

        res = model.predict(source=str(img_path), imgsz=imgsz,
                            conf=CONF_PREDICT_MIN, device=device, verbose=False)
        preds_raw = []
        if res and res[0].boxes is not None:
            for box in res[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                preds_raw.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                                  'conf': float(box.conf[0].cpu())})
        all_preds_raw.append(preds_raw)

    gt_tot   = n_gt_total if n_gt_total > 0 else 1
    fpr_d    = {}
    fnr_d    = {}
    tpr_d    = {}
    n_fp_ref = {}
    n_fn_ref = {}
    n_tp_ref = {}

    for conf_t in CONF_THRESHOLDS:
        pred_batch_filtered = [
            [p for p in img_preds if p['conf'] >= conf_t]
            for img_preds in all_preds_raw
        ]
        for iou_t in IOU_THRESHOLDS:
            print(f"    conf≥{conf_t} | IoU≥{iou_t}:", end='  ', flush=True)
            tp, fp, fn = _compute_tp_fp_fn(pred_batch_filtered, gt_batch, iou_t)
            fpr_d[(conf_t, iou_t)] = float(fp / gt_tot)
            fnr_d[(conf_t, iou_t)] = float(fn / gt_tot)
            tpr_d[(conf_t, iou_t)] = float(tp / gt_tot)
            n_fp_ref[(conf_t, iou_t)] = fp
            n_fn_ref[(conf_t, iou_t)] = fn
            n_tp_ref[(conf_t, iou_t)] = tp

    return {
        'fpr':  fpr_d,
        'fnr':  fnr_d,
        'tpr':  tpr_d,
        'n_fp': n_fp_ref,
        'n_fn': n_fn_ref,
        'n_tp': n_tp_ref,
        'n_gt': n_gt_total,
    }


def _val_model(model, data_yaml, split, imgsz, device):
    """Run model.val + multi-IoU FPR/FNR. Returns a metrics dict."""
    res     = model.val(data=data_yaml, split=split, imgsz=imgsz,
                        device=device, verbose=True)
    mp      = float(res.box.mp)
    mr      = float(res.box.mr)
    map50   = float(res.box.map50)
    map5095 = float(res.box.map)
    f1      = 2 * mp * mr / (mp + mr) if (mp + mr) > 0 else 0.0

    per_cls = {}
    if hasattr(res.box, 'ap_class_index'):
        for i, idx in enumerate(res.box.ap_class_index):
            name = CLASS_NAMES.get(int(idx), f'cls{idx}')
            per_cls[name] = float(res.box.ap50[i])

    rates = _fpr_fnr_multi_iou(model, data_yaml, split, imgsz, device)

    result = {
        'precision': mp,
        'recall':    mr,
        'f1':        f1,
        'map50':     map50,
        'map5095':   map5095,
        'per_cls':   per_cls,
    }

    if rates is not None:
        result['n_gt'] = rates['n_gt']
        for conf_t in CONF_THRESHOLDS:
            for iou_t in IOU_THRESHOLDS:
                result[f'fpr_{conf_t}_{iou_t}']  = rates['fpr'].get((conf_t, iou_t))
                result[f'fnr_{conf_t}_{iou_t}']  = rates['fnr'].get((conf_t, iou_t))
                result[f'tpr_{conf_t}_{iou_t}']  = rates['tpr'].get((conf_t, iou_t))
                result[f'n_fp_{conf_t}_{iou_t}'] = rates['n_fp'].get((conf_t, iou_t))
                result[f'n_fn_{conf_t}_{iou_t}'] = rates['n_fn'].get((conf_t, iou_t))
                result[f'n_tp_{conf_t}_{iou_t}'] = rates['n_tp'].get((conf_t, iou_t))
    else:
        result['n_gt'] = None
        for conf_t in CONF_THRESHOLDS:
            for iou_t in IOU_THRESHOLDS:
                result[f'fpr_{conf_t}_{iou_t}']  = None
                result[f'fnr_{conf_t}_{iou_t}']  = None
                result[f'tpr_{conf_t}_{iou_t}']  = None
                result[f'n_fp_{conf_t}_{iou_t}'] = None
                result[f'n_fn_{conf_t}_{iou_t}'] = None
                result[f'n_tp_{conf_t}_{iou_t}'] = None

    return result


def read_run_csv(run_dir):
    """Read the last row of results.csv and return box/cls/dfl/tot loss."""
    csv_path = os.path.join(run_dir, 'results.csv')
    if not os.path.exists(csv_path):
        print(f"  [WARN] results.csv not found in {run_dir}")
        return {}
    with open(csv_path, newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    last = rows[-1]

    def _get(row, key):
        for k, v in row.items():
            if k.strip() == key and v and v.strip():
                try:
                    return float(v.strip())
                except ValueError:
                    return None
        return None

    bl = _get(last, 'val/box_loss')
    cl = _get(last, 'val/cls_loss')
    dl = _get(last, 'val/dfl_loss')
    tl = (bl + cl + dl) if (bl is not None and cl is not None and dl is not None) else None
    return {'box_loss': bl, 'cls_loss': cl, 'dfl_loss': dl, 'tot_loss': tl}


def _fmt(v, decimals=3):
    if v is None:
        return '—'
    return f"{v:.{decimals}f}"


# ══════════════════════════════════════════════════════════════
#  PNG TABLE RENDERING
# ══════════════════════════════════════════════════════════════

def _split_header(col):
    """
    Split a column name into (row1, row2) for two-row headers.
    'FPR_IoU@0.5/c0.25' → ('FPR IoU@0.5', 'c0.25')
    'Run'               → ('Run', '')
    """
    if '/c' in col:
        a, b = col.split('/c', 1)
        return a.replace('_', ' '), 'c' + b
    return col, ''


def _make_png_table(title_text, COLS, COL_W, all_rows, get_metric_fn,
                    METRIC_COLS, LOSS_COLS, int_cols, footer_text, output_path):
    """
    PNG table with 3-level headers for columns grouped by IoU.
    Simple columns use a single-row header.

    Special rows: ('__group__', 'LABEL') → section separator band
    (e.g. VALIDATION SET / TEST SET). Excluded from best-value computation
    and from row colour alternation.
    """
    ROW_H   = 30
    GRP_H   = 26
    HDR_R1  = 20
    HDR_R2  = 20
    HDR_H   = HDR_R1 + HDR_R2
    PAD_L   = 14
    TITLE_H = 24

    n_data  = sum(1 for r in all_rows if r[0] != '__group__')
    n_grp   = sum(1 for r in all_rows if r[0] == '__group__')
    total_h = TITLE_H + HDR_H + n_data * ROW_H + n_grp * GRP_H + 44
    total_w = sum(COL_W) + PAD_L * 2 + 4

    C_BG      = (255, 252, 248)
    C_HDR_GRP = (180, 140, 100)
    C_HDR_SUB = (210, 180, 150)
    C_GRP_BG  = (235, 220, 200)
    C_ROW1    = (255, 252, 248)
    C_ROW2    = (248, 242, 235)
    C_TEXT    = (40,  40,  40)
    C_BEST    = (0,  140,   0)
    C_LINE    = (200, 185, 170)
    C_SEP     = (150, 110,  70)
    C_WHITE   = (255, 255, 255)
    C_TITLE   = (80,  50,  20)

    try:
        F_HDR1  = ImageFont.truetype(FONT_BOLD, 11)
        F_HDR2  = ImageFont.truetype(FONT_REG,  10)
        F_CELL  = ImageFont.truetype(FONT_REG,  11)
        F_MODE  = ImageFont.truetype(FONT_BOLD, 11)
        F_TITLE = ImageFont.truetype(FONT_BOLD, 13)
    except Exception:
        F_HDR1 = F_HDR2 = F_CELL = F_MODE = F_TITLE = ImageFont.load_default()

    img  = Image.new('RGB', (total_w, total_h), C_BG)
    draw = ImageDraw.Draw(img)

    def x_col(ci):
        return PAD_L + sum(COL_W[:ci])

    def text_w(text, font):
        try:
            bb = font.getbbox(str(text))
            return bb[2] - bb[0]
        except Exception:
            return len(str(text)) * 7

    def draw_text_centered(text, x, y, w, h, font, color):
        tw = text_w(text, font)
        tx = x + (w - tw) // 2
        ty = y + (h - 13) // 2
        draw.text((tx, ty), str(text), fill=color, font=font)

    def draw_hline(y, width=1, color=None):
        draw.line([(PAD_L, y), (total_w - PAD_L, y)],
                  fill=color or C_LINE, width=width)

    def draw_vline(x, y_top, y_bot, width=2, color=None):
        draw.line([(x, y_top), (x, y_bot)], fill=color or C_SEP, width=width)

    def is_grouped(col):
        return '/c' in col

    # ── Title ─────────────────────────────────────────────────
    draw.text((PAD_L, 4), title_text, fill=C_TITLE, font=F_TITLE)

    # ── 3-level header ─────────────────────────────────────────
    y_hdr  = TITLE_H
    n_conf = len(CONF_THRESHOLDS)

    draw.rectangle([PAD_L, y_hdr, total_w - PAD_L, y_hdr + HDR_H], fill=C_HDR_SUB)

    grouped_count = 0
    for ci, (col, w) in enumerate(zip(COLS, COL_W)):
        x = x_col(ci)
        if not is_grouped(col):
            draw.rectangle([x, y_hdr, x + w, y_hdr + HDR_H], fill=C_HDR_GRP)
            align_x = x + 6 if ci <= 1 else x + (w - text_w(col, F_HDR1)) // 2
            ty = y_hdr + (HDR_H - 13) // 2
            draw.text((align_x, ty), col, fill=C_WHITE, font=F_HDR1)
        else:
            ci_in_block = grouped_count % n_conf
            if ci_in_block == 0:
                group_w  = sum(COL_W[ci:ci + n_conf])
                draw.rectangle([x, y_hdr, x + group_w, y_hdr + HDR_R1], fill=C_HDR_GRP)
                iou_label = col.split('/c')[0].replace('_', ' ')
                tw = text_w(iou_label, F_HDR1)
                tx = x + (group_w - tw) // 2
                ty = y_hdr + (HDR_R1 - 13) // 2
                draw.text((tx, ty), iou_label, fill=C_WHITE, font=F_HDR1)
                draw_vline(x, y_hdr, y_hdr + HDR_H)
            conf_label = 'c' + col.split('/c')[1]
            draw.rectangle([x, y_hdr + HDR_R1, x + w, y_hdr + HDR_H], fill=C_HDR_SUB)
            tw2 = text_w(conf_label, F_HDR2)
            tx2 = x + (w - tw2) // 2
            ty2 = y_hdr + HDR_R1 + (HDR_R2 - 11) // 2
            draw.text((tx2, ty2), conf_label, fill=C_WHITE, font=F_HDR2)
            draw.line([(x, y_hdr + HDR_R1), (x + w, y_hdr + HDR_R1)],
                      fill=C_SEP, width=1)
            grouped_count += 1

    draw_hline(y_hdr + HDR_H, width=2)
    y = y_hdr + HDR_H

    # ── Best-value computation (excludes group rows) ──────────
    data_rows = [(l, s) for l, s in all_rows if l != '__group__']

    def best_col(col):
        vals = [get_metric_fn(l, s, col) for l, s in data_rows]
        vals = [v for v in vals if v is not None]
        if not vals:
            return None
        return min(vals) if col in LOSS_COLS else max(vals)

    bests = {col: best_col(col) for col in METRIC_COLS + LOSS_COLS}

    data_idx = 0
    for label, split in all_rows:

        # ── Section separator row ─────────────────────────────
        if label == '__group__':
            draw.rectangle([PAD_L, y, total_w - PAD_L, y + GRP_H], fill=C_GRP_BG)
            draw.text((PAD_L + 8, y + (GRP_H - 13) // 2),
                      split, fill=C_TITLE, font=F_MODE)
            draw_hline(y + GRP_H, width=2)
            y += GRP_H
            continue

        # ── Normal data row ───────────────────────────────────
        bg = C_ROW1 if data_idx % 2 == 0 else C_ROW2
        draw.rectangle([PAD_L, y, total_w - PAD_L, y + ROW_H], fill=bg)

        draw.text((x_col(0) + 6, y + (ROW_H - 13) // 2), label, fill=C_TEXT, font=F_MODE)
        draw.text((x_col(1) + 6, y + (ROW_H - 13) // 2), split, fill=C_TEXT, font=F_CELL)

        for ci, col in enumerate(COLS[2:], start=2):
            val  = get_metric_fn(label, split, col)
            txt  = str(int(val)) if (col in int_cols and val is not None) \
                   else (_fmt(val) if val is not None else '—')
            best    = bests.get(col)
            is_best = (val is not None and best is not None and abs(val - best) < 1e-6)
            color   = C_BEST if is_best else C_TEXT
            draw_text_centered(txt, x_col(ci), y, COL_W[ci], ROW_H, F_CELL, color)

        grouped_count_row = 0
        for ci, col in enumerate(COLS[2:], start=2):
            if is_grouped(col):
                if grouped_count_row % n_conf == 0:
                    draw_vline(x_col(ci), y, y + ROW_H, width=1, color=C_LINE)
                grouped_count_row += 1

        draw_hline(y + ROW_H)
        y += ROW_H
        data_idx += 1

    draw.rectangle([PAD_L, TITLE_H, total_w - PAD_L, y + 4], outline=C_LINE, width=1)
    draw.text((PAD_L, y + 8), footer_text, fill=(120, 100, 80), font=F_CELL)

    img.save(output_path)
    return output_path


# ══════════════════════════════════════════════════════════════
#  SUMMARY TABLE
# ══════════════════════════════════════════════════════════════

def generate_summary_table(run_configs, device, output_dir, imgsz=640):
    """
    run_configs: list of dicts with keys:
      label, run_dir, data_yaml, splits_to_eval
    """
    print("\nGenerating summary table...")
    from ultralytics import YOLO

    results_by_run   = {}
    losses = {}

    for cfg in run_configs:
        label     = cfg['label']
        run_dir   = cfg['run_dir']
        data_yaml = cfg['data_yaml']
        weights   = os.path.join(run_dir, 'weights', 'best.pt')

        losses[label] = read_run_csv(run_dir)
        results_by_run[label]   = {}

        if not os.path.exists(weights):
            print(f"  [WARN] {label}: best.pt not found ({weights})")
            for split, _ in cfg['splits_to_eval']:
                results_by_run[label][split] = None
            continue

        model = YOLO(weights)
        for split, split_label in cfg['splits_to_eval']:
            print(f"  {label} / {split}...", end=' ', flush=True)
            try:
                m = _val_model(model, data_yaml, split, imgsz, device)
                results_by_run[label][split] = m
                print(f"mAP@50={m['map50']:.3f}  P={m['precision']:.3f}  R={m['recall']:.3f}")
            except Exception as e:
                import traceback
                print(f"ERROR: {e}")
                traceback.print_exc()
                results_by_run[label][split] = None

    # ── TXT table ─────────────────────────────────────────────
    # Rows ordered by split (val first, then test) and then by run
    rows_data = []
    for split_key, split_label in [('val', 'VAL'), ('test', 'TEST')]:
        for cfg in run_configs:
            rows_data.append((cfg['label'], split_key, split_label))

    def _rate_header_txt(metric):
        grp_row  = ''
        conf_row = ''
        for iou_t in IOU_THRESHOLDS:
            grp_label   = f'{metric} IoU@{iou_t}'
            conf_labels = [f'c{c}' for c in CONF_THRESHOLDS]
            group_w     = len(CONF_THRESHOLDS) * 9 - 1
            grp_row  += f'  {grp_label:^{group_w}}'
            conf_row += '  ' + ' '.join(f'{cl:>8}' for cl in conf_labels)
        return grp_row, conf_row

    fpr_grp, fpr_conf = _rate_header_txt('FPR')
    fnr_grp, fnr_conf = _rate_header_txt('FNR')
    tpr_grp, tpr_conf = _rate_header_txt('TPR')
    fp_grp,  fp_conf  = _rate_header_txt('#FP')
    fn_grp,  fn_conf  = _rate_header_txt('#FN')
    tp_grp,  tp_conf  = _rate_header_txt('#TP')

    hdr1 = (f"{'Run':<26} {'Split':<6} "
            f"{'':>10} {'':>7} {'':>7}  "
            f"{fpr_grp}  {fnr_grp}  {tpr_grp}  {fp_grp}  {fn_grp}  {tp_grp}  {'':>6}  "
            f"{'':>8} {'':>10} {'':>7}  {'':>9} {'':>9} {'':>9} {'':>9}")
    hdr2 = (f"{'Run':<26} {'Split':<6} "
            f"{'Precision':>10} {'Recall':>7} {'F1':>7}  "
            f"{fpr_conf}  {fnr_conf}  {tpr_conf}  {fp_conf}  {fn_conf}  {tp_conf}  {'#GT':>6}  "
            f"{'mAP@50':>8} {'mAP@50-95':>10} {'empty':>7}  "
            f"{'box_loss':>9} {'cls_loss':>9} {'dfl_loss':>9} {'tot_loss':>9}")
    sep = '─' * max(len(hdr1), len(hdr2))

    lines = []
    cur_split = None
    for label, split, split_label in rows_data:
        if split != cur_split:
            lines.append(f"\n{'─'*6} {split_label} {'─'*6}")
            lines.append(hdr1)
            lines.append(hdr2)
            lines.append(sep)
            cur_split = split

        m   = results_by_run.get(label, {}).get(split)
        lss = losses.get(label, {})
        bl  = _fmt(lss.get('box_loss')) if split == 'val' else '—'
        cl  = _fmt(lss.get('cls_loss')) if split == 'val' else '—'
        dl  = _fmt(lss.get('dfl_loss')) if split == 'val' else '—'
        tl  = _fmt(lss.get('tot_loss')) if split == 'val' else '—'

        def _rate_cells(key_prefix, fmt_fn, w):
            parts = []
            for iou_t in IOU_THRESHOLDS:
                for conf_t in CONF_THRESHOLDS:
                    v = m.get(f'{key_prefix}{conf_t}_{iou_t}') if m else None
                    parts.append(f"{fmt_fn(v):>{w}}")
            return '  '.join(parts)

        if m is None:
            blank = '  '.join(f"{'—':>8}" for _ in IOU_THRESHOLDS for _ in CONF_THRESHOLDS)
            lines.append(f"  {label:<24} {split:<6} "
                         f"{'—':>10} {'—':>7} {'—':>7}  "
                         f"{blank}  {blank}  {blank}  {blank}  {blank}  {blank}  {'—':>6}  "
                         f"{'—':>8} {'—':>10} {'—':>7}  "
                         f"{'—':>9} {'—':>9} {'—':>9} {'—':>9}")
        else:
            fpr_s = _rate_cells('fpr_',  lambda v: _fmt(v, 3) if v is not None else '—', 8)
            fnr_s = _rate_cells('fnr_',  lambda v: _fmt(v, 3) if v is not None else '—', 8)
            tpr_s = _rate_cells('tpr_',  lambda v: _fmt(v, 3) if v is not None else '—', 8)
            fp_s  = _rate_cells('n_fp_', lambda v: str(int(v)) if v is not None else '—', 8)
            fn_s  = _rate_cells('n_fn_', lambda v: str(int(v)) if v is not None else '—', 8)
            tp_s  = _rate_cells('n_tp_', lambda v: str(int(v)) if v is not None else '—', 8)
            n_gt  = str(m['n_gt']) if m.get('n_gt') is not None else '—'
            lines.append(
                f"  {label:<24} {split:<6} "
                f"{_fmt(m['precision']):>10} {_fmt(m['recall']):>7} {_fmt(m['f1']):>7}  "
                f"{fpr_s}  {fnr_s}  {tpr_s}  {fp_s}  {fn_s}  {tp_s}  {n_gt:>6}  "
                f"{_fmt(m['map50']):>8} {_fmt(m['map5095']):>10} {_fmt(m['per_cls'].get('empty')):>7}  "
                f"{bl:>9} {cl:>9} {dl:>9} {tl:>9}"
            )

    lines.append(sep)
    lines.append("\nNote: val loss = last epoch  |  per-class = mAP@50")
    lines.append("Precision = mp Ultralytics (P@conf=0.25 default)  |  FPR=#FP/#GT  FNR=#FN/#GT  TPR=#TP/#GT")
    lines.append("#FP = unmatched predictions  |  #FN = unmatched GT boxes  |  #TP = matched pred/GT")
    lines.append(f"Conf thresholds: {CONF_THRESHOLDS}  |  IoU thresholds: {IOU_THRESHOLDS}")
    lines.append(f"predict with conf={CONF_PREDICT_MIN}")
    summary_text = '\n'.join(lines)
    print(f"\n{summary_text}")

    txt_path = os.path.join(output_dir, 'summary_table.txt')
    with open(txt_path, 'w') as f:
        f.write(summary_text)
    print(f"\n  Table TXT: {txt_path}")

    png_path1, png_path2a, png_path2b, png_path2c = render_summary_png_tables(results_by_run, losses, run_configs, output_dir)
    print(f"  PNG table 1  (metrics):  {png_path1}")
    print(f"  PNG table 2a (FPR/#FP): {png_path2a}")
    print(f"  PNG table 2b (FNR/#FN): {png_path2b}")
    print(f"  PNG table 2c (TPR/#TP): {png_path2c}")

    save_summary_tsv(results_by_run, losses, run_configs, output_dir)

    return results_by_run, losses


def save_summary_tsv(results_by_run, losses, run_configs, output_dir):
    """
    Produce 4 TSV files mirroring the 4 PNG tables exactly:
      table1_metrics.tsv  — Precision Recall F1 mAP@50 mAP@50-95 empty box_loss cls_loss dfl_loss tot_loss
      table2a_fpr.tsv      — #GT  FPR IoU@x c0.25/c0.5  #FP IoU@x c0.25/c0.5
      table2b_fnr.tsv      — #GT  FNR IoU@x c0.25/c0.5  #FN IoU@x c0.25/c0.5
      table2c_tpr.tsv      — #GT  TPR IoU@x c0.25/c0.5  #TP IoU@x c0.25/c0.5
    Each file has a VALIDATION SET and a TEST SET section separated by an empty row.
    """
    import csv as _csv

    C = CONF_THRESHOLDS   # [0.25, 0.5]
    I = IOU_THRESHOLDS    # [0.0, 0.25, 0.5, 0.75]

    def fmt(val):
        """Format a number with 3 decimals."""
        if val == '' or val is None:
            return ''
        try:
            return f'{float(val):.3f}'
        except (ValueError, TypeError):
            return str(val)

    def v(m, key, default=''):
        if m is None:
            return default
        val = m.get(key)
        return '' if val is None else fmt(val)

    def write_section(writer, split_label, run_configs, results_by_run, losses, row_fn):
        writer.writerow({'Run': f'--- {split_label} ---'})
        for cfg in run_configs:
            writer.writerow(row_fn(cfg['label'], split_label.lower().split()[0], results_by_run, losses))

    # ── Table 1: metrics ──────────────────────────────────────────
    fields1 = ['Run', 'Split', 'Precision', 'Recall', 'F1',
                'mAP@50', 'mAP@50-95',
                'box_loss', 'cls_loss', 'dfl_loss', 'tot_loss']

    def row1(label, split, results_by_run, losses):
        m   = results_by_run.get(label, {}).get(split)
        lss = losses.get(label, {})
        return {
            'Run':        label,
            'Split':      split,
            'Precision':  v(m, 'precision'),
            'Recall':     v(m, 'recall'),
            'F1':         v(m, 'f1'),
            'mAP@50':     v(m, 'map50'),
            'mAP@50-95':  v(m, 'map5095'),
            'box_loss':   fmt(lss.get('box_loss', '')) if split == 'val' else '',
            'cls_loss':   fmt(lss.get('cls_loss', '')) if split == 'val' else '',
            'dfl_loss':   fmt(lss.get('dfl_loss', '')) if split == 'val' else '',
            'tot_loss':   fmt(lss.get('tot_loss', '')) if split == 'val' else '',
        }

    p1 = os.path.join(output_dir, 'table1_metrics.tsv')
    with open(p1, 'w', newline='', encoding='utf-8') as f:
        w = _csv.DictWriter(f, fieldnames=fields1, extrasaction='ignore', delimiter='	')
        w.writeheader()
        for split, label in [('val', 'VALIDATION SET'), ('test', 'TEST SET')]:
            w.writerow({'Run': f'--- {label} ---'})
            for cfg in run_configs:
                w.writerow(row1(cfg['label'], split, results_by_run, losses))
            w.writerow({k: '' for k in fields1})
    print(f"  CSV table 1: {p1}")

    # ── Tabelle 2a/2b/2c: FPR / FNR / TPR ───────────────────────
    def rate_fields(metric_up, count_up):
        cols = ['Run', 'Split', '#GT']
        for iou in I:
            for conf in C:
                cols.append(f'{metric_up} IoU@{iou} c{conf}')
        for iou in I:
            for conf in C:
                cols.append(f'{count_up} IoU@{iou} c{conf}')
        return cols

    def row_rate(label, split, results_by_run, metric, count_prefix):
        m = results_by_run.get(label, {}).get(split)
        row = {'Run': label, 'Split': split,
               '#GT': int(m.get('n_gt', 0)) if m and m.get('n_gt') else ''}
        metric_up = metric.upper()
        count_up  = count_prefix.replace('n_', '#').upper()
        for iou in I:
            for conf in C:
                val = m.get(f'{metric}_{conf}_{iou}') if m else None
                row[f'{metric_up} IoU@{iou} c{conf}'] = fmt(val) if val is not None else ''
        for iou in I:
            for conf in C:
                val = m.get(f'{count_prefix}_{conf}_{iou}') if m else None
                row[f'{count_up} IoU@{iou} c{conf}'] = int(val) if val is not None else ''
        return row

    for metric, count_prefix, filename, metric_up, count_up in [
        ('fpr', 'n_fp', 'table2a_fpr.tsv', 'FPR', '#FP'),
        ('fnr', 'n_fn', 'table2b_fnr.tsv', 'FNR', '#FN'),
        ('tpr', 'n_tp', 'table2c_tpr.tsv', 'TPR', '#TP'),
    ]:
        fields = rate_fields(metric_up, count_up)
        path   = os.path.join(output_dir, filename)
        with open(path, 'w', newline='', encoding='utf-8') as f:
            w = _csv.DictWriter(f, fieldnames=fields, extrasaction='ignore', delimiter='	')
            w.writeheader()
            for split, label in [('val', 'VALIDATION SET'), ('test', 'TEST SET')]:
                w.writerow({'Run': f'--- {label} ---'})
                for cfg in run_configs:
                    w.writerow(row_rate(cfg['label'], split, results_by_run, metric, count_prefix))
                w.writerow({k: '' for k in fields})
        print(f"  CSV {filename}: {path}")


def render_summary_png_tables(results_by_run, losses, run_configs, output_dir):
    """Generate four PNG tables: metrics, FPR, FNR, TPR."""

    # Table 1 (metrics): val + test, separated by group bands
    all_rows_metrics = []
    for split_key, split_label in [('val', 'VALIDATION SET'), ('test', 'TEST SET')]:
        all_rows_metrics.append(('__group__', split_label))
        for cfg in run_configs:
            all_rows_metrics.append((cfg['label'], split_key))

    # Tables 2a/2b/2c (rates): val + test, separated by group bands
    all_rows_rates = []
    for split_key, split_label in [('val', 'VALIDATION SET'), ('test', 'TEST SET')]:
        all_rows_rates.append(('__group__', split_label))
        for cfg in run_configs:
            all_rows_rates.append((cfg['label'], split_key))

    # ── Table 1: main metrics ─────────────────────────────────
    COLS1 = ['Run', 'Split',
             'Precision', 'Recall', 'F1',
             'mAP@50', 'mAP@50-95', 'empty',
             'box_loss', 'cls_loss', 'dfl_loss', 'tot_loss']
    COL_W1 = [170, 50,
              82, 68, 62,
              76, 96, 66,
              82, 82, 82, 82]

    METRIC_COLS1 = ['Precision', 'Recall', 'F1', 'mAP@50', 'mAP@50-95', 'empty']
    LOSS_COLS1   = ['box_loss', 'cls_loss', 'dfl_loss', 'tot_loss']

    def get_metric1(label, split, col):
        m   = results_by_run.get(label, {}).get(split)
        lss = losses.get(label, {})
        if m is None:
            return None
        return {
            'Precision': m.get('precision'),
            'Recall':    m.get('recall'),
            'F1':        m.get('f1'),
            'mAP@50':    m.get('map50'),
            'mAP@50-95': m.get('map5095'),
            'empty':     m['per_cls'].get('empty') if m.get('per_cls') else None,
            'box_loss':  lss.get('box_loss') if split == 'val' else None,
            'cls_loss':  lss.get('cls_loss') if split == 'val' else None,
            'dfl_loss':  lss.get('dfl_loss') if split == 'val' else None,
            'tot_loss':  lss.get('tot_loss') if split == 'val' else None,
        }.get(col)

    path1 = _make_png_table(
        title_text    = "Table 1 — Main metrics",
        COLS          = COLS1,
        COL_W         = COL_W1,
        all_rows      = all_rows_metrics,
        get_metric_fn = get_metric1,
        METRIC_COLS   = METRIC_COLS1,
        LOSS_COLS     = LOSS_COLS1,
        int_cols      = set(),
        footer_text   = ("Green=best value per column  |  Precision=P@conf0.25 (Ultralytics default)  "
                         "|  empty=mAP@50 per class  |  Loss: last val epoch"),
        output_path   = os.path.join(output_dir, 'table1_metrics.png'),
    )

    # ── Rate columns ──────────────────────────────────────────
    fpr_cols = [f'FPR_IoU@{iou}/c{conf}'
                for iou in IOU_THRESHOLDS for conf in CONF_THRESHOLDS]
    fnr_cols = [f'FNR_IoU@{iou}/c{conf}'
                for iou in IOU_THRESHOLDS for conf in CONF_THRESHOLDS]
    tpr_cols = [f'TPR_IoU@{iou}/c{conf}'
                for iou in IOU_THRESHOLDS for conf in CONF_THRESHOLDS]
    fp_cols  = [f'#FP_IoU@{iou}/c{conf}'
                for iou in IOU_THRESHOLDS for conf in CONF_THRESHOLDS]
    fn_cols  = [f'#FN_IoU@{iou}/c{conf}'
                for iou in IOU_THRESHOLDS for conf in CONF_THRESHOLDS]
    tp_cols  = [f'#TP_IoU@{iou}/c{conf}'
                for iou in IOU_THRESHOLDS for conf in CONF_THRESHOLDS]

    COL_W_RATE = 72

    footer_common = (f"conf thresholds: {CONF_THRESHOLDS}  |  IoU thresholds: {IOU_THRESHOLDS}  |  "
                     f"predict with conf={CONF_PREDICT_MIN}")

    def get_metric_rates(label, split, col):
        m = results_by_run.get(label, {}).get(split)
        if m is None:
            return None
        if col == '#GT':
            return m.get('n_gt')
        for iou_t in IOU_THRESHOLDS:
            for conf_t in CONF_THRESHOLDS:
                if col == f'FPR_IoU@{iou_t}/c{conf_t}':
                    return m.get(f'fpr_{conf_t}_{iou_t}')
                if col == f'FNR_IoU@{iou_t}/c{conf_t}':
                    return m.get(f'fnr_{conf_t}_{iou_t}')
                if col == f'TPR_IoU@{iou_t}/c{conf_t}':
                    return m.get(f'tpr_{conf_t}_{iou_t}')
                if col == f'#FP_IoU@{iou_t}/c{conf_t}':
                    return m.get(f'n_fp_{conf_t}_{iou_t}')
                if col == f'#FN_IoU@{iou_t}/c{conf_t}':
                    return m.get(f'n_fn_{conf_t}_{iou_t}')
                if col == f'#TP_IoU@{iou_t}/c{conf_t}':
                    return m.get(f'n_tp_{conf_t}_{iou_t}')
        return None

    # ── Table 2a: FPR ─────────────────────────────────────────
    COLS2a  = ['Run', 'Split', '#GT', *fpr_cols, *fp_cols]
    COL_W2a = [170, 50, 52,
               *[COL_W_RATE]*len(fpr_cols),
               *[COL_W_RATE]*len(fp_cols)]

    path2a = _make_png_table(
        title_text    = ("Table 2a — FPR (unmatched predictions / #GT)\n"
                         f"Grouped by IoU  |  conf: {CONF_THRESHOLDS}"),
        COLS          = COLS2a,
        COL_W         = COL_W2a,
        all_rows      = all_rows_rates,
        get_metric_fn = get_metric_rates,
        METRIC_COLS   = [],
        LOSS_COLS     = fpr_cols,
        int_cols      = {*fp_cols, '#GT'},
        footer_text   = f"Green=lowest value per column  |  FPR = unmatched predictions / #GT  |  {footer_common}",
        output_path   = os.path.join(output_dir, 'table2a_fpr.png'),
    )

    # ── Table 2b: FNR ─────────────────────────────────────────
    COLS2b  = ['Run', 'Split', '#GT', *fnr_cols, *fn_cols]
    COL_W2b = [170, 50, 52,
               *[COL_W_RATE]*len(fnr_cols),
               *[COL_W_RATE]*len(fn_cols)]

    path2b = _make_png_table(
        title_text    = ("Table 2b — FNR (unmatched GT boxes / #GT)\n"
                         f"Grouped by IoU  |  conf: {CONF_THRESHOLDS}"),
        COLS          = COLS2b,
        COL_W         = COL_W2b,
        all_rows      = all_rows_rates,
        get_metric_fn = get_metric_rates,
        METRIC_COLS   = [],
        LOSS_COLS     = fnr_cols,
        int_cols      = {*fn_cols, '#GT'},
        footer_text   = f"Green=lowest value per column  |  FNR = unmatched GT boxes / #GT  |  {footer_common}",
        output_path   = os.path.join(output_dir, 'table2b_fnr.png'),
    )

    # ── Table 2c: TPR ─────────────────────────────────────────
    COLS2c  = ['Run', 'Split', '#GT', *tpr_cols, *tp_cols]
    COL_W2c = [170, 50, 52,
               *[COL_W_RATE]*len(tpr_cols),
               *[COL_W_RATE]*len(tp_cols)]

    path2c = _make_png_table(
        title_text    = ("Table 2c — TPR (matched pred/GT / #GT)\n"
                         f"Grouped by IoU  |  conf: {CONF_THRESHOLDS}"),
        COLS          = COLS2c,
        COL_W         = COL_W2c,
        all_rows      = all_rows_rates,
        get_metric_fn = get_metric_rates,
        METRIC_COLS   = tpr_cols,
        LOSS_COLS     = [],
        int_cols      = {*tp_cols, '#GT'},
        footer_text   = f"Green=highest value per column  |  TPR = matched pred/GT / #GT  |  {footer_common}",
        output_path   = os.path.join(output_dir, 'table2c_tpr.png'),
    )

    return path1, path2a, path2b, path2c


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    global FONT_BOLD, FONT_REG

    args = parse_args()
    FONT_BOLD = args.font_bold
    FONT_REG  = args.font_regular

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    random.seed(args.seed)

    print(f"\n{'='*60}")
    print(f"  OOS Pipeline Visualization")
    print(f"  Device: {device}")
    print(f"  Output: {args.output}")
    print(f"{'='*60}\n")

    os.makedirs(args.output, exist_ok=True)

    # ── Run configs for the summary table ─────────────────────
    # Includes detector_only_aug as the comparison baseline
    run_configs = [
        {
            'label':          'detector_only_aug',
            'run_dir':        str(Path(args.weights_detector).parent.parent),
            'data_yaml':      args.data_yaml,
            'splits_to_eval': [('val', 'VAL'), ('test', 'TEST')],
        },
        {
            'label':          'gan_frozen',
            'run_dir':        str(Path(args.weights_frozen).parent.parent),
            'data_yaml':      args.data_yaml,
            'splits_to_eval': [('val', 'VAL'), ('test', 'TEST')],
        },
        {
            'label':          'joint',
            'run_dir':        str(Path(args.weights_joint).parent.parent),
            'data_yaml':      args.data_yaml,
            'splits_to_eval': [('val', 'VAL'), ('test', 'TEST')],
        },
    ]

    # ── Summary table ──────────────────────────────────────────
    if not args.skip_table:
        generate_summary_table(run_configs, device, args.output, args.imgsz)

    # ── Visual panels ──────────────────────────────────────────
    if args.skip_panels:
        return

    # ── Load GAN models ────────────────────────────────────────
    print("Loading pre-trained GAN (frozen)...")
    gan_frozen = load_gan(args.gan_ckpt, device, blur2blur_root=args.blur2blur_dir)
    print("Loading joint-trained GAN...")
    gan_joint  = load_gan_joint(args.gan_ckpt_joint, device, blur2blur_root=args.blur2blur_dir)
    if gan_joint is None:
        print("  [WARN] Joint GAN not available — joint and frozen images "
              "will be identical until you specify --gan-ckpt-joint")
        gan_joint = gan_frozen  # fallback

    # ── Load YOLO models ───────────────────────────────────────
    from ultralytics import YOLO

    models = {}
    for name, w_path in [
        ('detector_only', args.weights_detector),
        ('gan_frozen',    args.weights_frozen),
        ('joint',         args.weights_joint),
    ]:
        if os.path.exists(w_path):
            models[name] = YOLO(w_path)
            print(f"  {name}: OK")
        else:
            models[name] = None
            print(f"  [WARN] {name}: {w_path} not found")

    img_dir  = Path(args.images)
    lbl_dir  = Path(args.labels)
    ext      = ('.jpg', '.jpeg', '.png')
    all_imgs = [f for f in sorted(img_dir.iterdir()) if f.suffix.lower() in ext]

    if not all_imgs:
        print(f"[ERROR] No images found in {img_dir}")
        return

    # Sampling stratified by source video, for variety
    video_groups = {}
    for img_p in all_imgs:
        stem     = img_p.stem
        video_id = stem.split('_')[0] if '_' in stem else stem
        video_groups.setdefault(video_id, []).append(img_p)

    n_video     = len(video_groups)
    n_per_video = max(1, args.n_samples // n_video)
    sample      = []
    for video_imgs in video_groups.values():
        random.shuffle(video_imgs)
        sample.extend(video_imgs[:n_per_video])
    remaining = [p for p in all_imgs if p not in set(sample)]
    random.shuffle(remaining)
    sample = (sample + remaining)[:args.n_samples]
    random.shuffle(sample)

    print(f"\nGenerating {len(sample)} panels (out of {len(all_imgs)} images, ")
    print(f"  spread across {n_video} source videos, ~{n_per_video} per video)...")

    _gan_stats_log = []
    for i, img_path in enumerate(sample):
        print(f"  [{i+1}/{len(sample)}] {img_path.name}", end=' ', flush=True)

        img_orig = Image.open(img_path).convert('RGB')
        W, H     = img_orig.size
        gt_boxes = read_yolo_labels(str(lbl_dir / (img_path.stem + '.txt')), W, H)

        img_t          = pil_to_tensor(img_orig, device)
        img_gan_fr_pil = tensor_to_pil(apply_gan(gan_frozen, img_t, device))
        img_gan_jt_pil = tensor_to_pil(apply_gan(gan_joint,  img_t, device))

        stats_fr = compute_gan_statistics(img_orig, img_gan_fr_pil, 'frozen')
        stats_jt = compute_gan_statistics(img_orig, img_gan_jt_pil, 'joint')
        _gan_stats_log.append({**{'image': img_path.name}, **stats_fr})
        _gan_stats_log.append({**{'image': img_path.name}, **stats_jt})

        pred_det  = predict_yolo(models['detector_only'], img_orig,       args, device) \
                    if models['detector_only'] else []
        pred_froz = predict_yolo(models['gan_frozen'],    img_gan_fr_pil, args, device) \
                    if models['gan_frozen']    else []
        pred_jnt  = predict_yolo(models['joint'],         img_gan_jt_pil, args, device) \
                    if models['joint']         else []

        panel = create_panel(
            img_orig, img_gan_fr_pil, img_gan_jt_pil,
            gt_boxes, pred_det, pred_froz, pred_jnt,
            img_path.name, args,
        )
        out_path = os.path.join(args.output, f"compare_{img_path.stem}.png")
        save_figure(panel, out_path)
        plt.close(panel)
        print("-> OK")

    # ── Save GAN statistics report ────────────────────────────
    stats_path = os.path.join(args.output, 'gan_stats.csv')
    with open(stats_path, 'w', newline='') as csvf:
        writer = csv.DictWriter(csvf, fieldnames=[
            'image','gan','label','psnr','mean_diff','var_orig','var_gan'],
            extrasaction='ignore')
        writer.writeheader()
        for entry in _gan_stats_log:
            writer.writerow(entry)
    print(f"  GAN statistics: {stats_path}")

    print(f"\n{'='*60}")
    print(f"  Output: {args.output}")
    print(f"  Panels: {len(sample)}")
    print(f"  Legend: Green=GT  Red=Pred(conf>={args.conf})")
    print(f"  GAN statistics saved to: {stats_path}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()