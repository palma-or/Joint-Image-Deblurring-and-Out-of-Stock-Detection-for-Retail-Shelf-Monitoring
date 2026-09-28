# train_joint.py
"""
Joint training of Blur2Blur + YOLO26n via Ultralytics callbacks.

Training modes:
  detector_only → native YOLO26n on original images (baseline)
  gan_frozen    → GAN pre-processes video frames; GAN weights are frozen
  joint         → GAN pre-processes video frames; GAN weights are updated
                  by the detection loss (end-to-end joint training)

Loss names (Ultralytics convention):
  box_loss → bounding box regression
  cls_loss → classification
  dfl_loss → Distribution Focal Loss

Usage:
    python train_joint.py --mode detector_only --data <data.yaml>
    python train_joint.py --mode gan_frozen    --data <data.yaml> --gan_ckpt <ckpt.pth>
    python train_joint.py --mode joint         --data <data.yaml> --gan_ckpt <ckpt.pth>
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
from ultralytics import YOLO


# ── CONFIG ────────────────────────────────────────────────────────────────────
# Defaults — all paths can be overridden via CLI arguments.
OOS_ROOT    = os.environ.get('OOS_ROOT', os.path.expanduser('~/oos_orlando'))
PROJECT_DIR = os.path.join(OOS_ROOT, 'training')
GAN_SIZE    = 256   # spatial resolution used by the GAN generator

# U-Net encoder layer name prefixes — used to selectively freeze the encoder
# in joint training when --freeze_encoder is passed.
ENCODER_PREFIXES = (
    'model.model.0.',
    'model.model.1.model.1.',
    'model.model.1.model.2.',
    'model.model.1.model.3.model.1.',
    'model.model.1.model.3.model.2.',
)
# ─────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.abspath(os.path.join(PROJECT_DIR, '..', 'Blur2Blur')))


def parse_args():
    p = argparse.ArgumentParser(
        description='Joint training of Blur2Blur GAN + YOLO26n detector.'
    )
    p.add_argument(
        '--mode',
        default='detector_only',
        choices=['detector_only', 'gan_frozen', 'joint'],
        help='Training mode (default: detector_only).',
    )
    p.add_argument('--epochs',         type=int,   default=150)
    p.add_argument('--batch',          type=int,   default=32)
    p.add_argument('--imgsz',          type=int,   default=640)
    p.add_argument(
        '--lr0',
        type=float,
        default=None,
        help='Initial learning rate (default: 0.0054 for SGD / 0.001 for AdamW).',
    )
    p.add_argument(
        '--phase1_epochs',
        type=int,
        default=5,
        help='Warm-up epochs with the GAN frozen before joint updates begin (--mode joint only).',
    )
    p.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to the YOLO dataset YAML file.',
    )
    p.add_argument(
        '--project',
        type=str,
        default=os.path.join(OOS_ROOT, 'runs'),
        help='Root directory for saving training runs.',
    )
    p.add_argument(
        '--name',
        type=str,
        default=None,
        help='Run folder name (default: oos_<mode>).',
    )
    p.add_argument(
        '--gan_ckpt',
        type=str,
        default=None,
        help='Path to the corrected GAN checkpoint (latest_net_G_ready.pth). Required for gan_frozen and joint modes.',
    )
    p.add_argument(
        '--freeze_encoder',
        action='store_true',
        default=False,
        help='Freeze the U-Net encoder in joint training; only update the bottleneck and decoder.',
    )
    p.add_argument(
        '--augment',
        action='store_true',
        default=False,
        help='Enable advanced augmentation (detector_only mode only).',
    )
    return p.parse_args()


# ── GAN utilities ─────────────────────────────────────────────────────────────

def load_gan(gan_ckpt_path: str, device: torch.device):
    """
    Load the Blur2Blur generator wrapped for joint training.

    Args:
        gan_ckpt_path: Path to the corrected generator checkpoint
                       (output of fix_checkpoint_final.py).
        device:        Torch device.

    Returns:
        Blur2BlurWrapper with the generator loaded and placed on *device*.
    """
    from training.gan_wrapper import Blur2BlurWrapper

    print(f"Loading GAN from: {gan_ckpt_path}")
    # config_path is a required constructor argument but is never actually
    # read as a file internally -- Blur2BlurWrapper hardcodes its options
    # via sys.argv (see oos_models/gan_wrapper.py). Confirmed by reading
    # the full source: no open(config_path...) / yaml.load anywhere.
    gan = Blur2BlurWrapper(config_path="unused")

    # Unwrap DataParallel if present
    if isinstance(gan.generator, torch.nn.DataParallel):
        gan.generator = gan.generator.module

    if gan_ckpt_path and os.path.exists(gan_ckpt_path):
        state = torch.load(gan_ckpt_path, map_location='cpu', weights_only=False)
        missing, unexpected = gan.generator.load_state_dict(state, strict=False)
        print(f"  GAN weights loaded — missing: {len(missing)}, unexpected: {len(unexpected)}")
        if missing:
            print(f"  [WARN] First missing keys: {missing[:3]}")
    else:
        print(f"  [WARN] Checkpoint not found: {gan_ckpt_path}")

    gan.generator.to(device)
    return gan


def apply_gan(gan, imgs: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Pass images through the Blur2Blur generator.

    Input/output: tensor (B, 3, H, W) in [0, 1] (Ultralytics convention).
    The GAN expects input in [-1, 1]: normalisation is applied internally
    and the output is mapped back to [0, 1].

    In joint mode the computational graph is preserved so that the
    detection loss can back-propagate through the GAN.

    IMPORTANT (fixed after debugging): this function is called from inside
    the patched preprocess_batch(), which runs inside Ultralytics' own
    `with autocast(self.amp):` block during training. That means the GAN
    forward pass was silently running in FP16 mixed precision, even though
    the generator was trained/validated in FP32. FP16's much narrower
    range (~65504 max) is a well-known source of Inf/NaN in networks never
    designed or tested for reduced precision. We now force this block to
    run in FP32 regardless of the ambient autocast context, which is
    consistent with how the GAN was actually trained and validated
    (e.g. via fig06_gan_pipeline.py's standalone FP32 inference).

    Args:
        gan:    Blur2BlurWrapper instance.
        imgs:   Input images (B, 3, H, W) in [0, 1].
        device: Torch device.

    Returns:
        Generator output (B, 3, H, W) in [0, 1], same spatial size as
        input, in the SAME dtype as the input imgs tensor (so it drops
        back into Ultralytics' mixed-precision pipeline transparently).
    """
    input_dtype = imgs.dtype

    with torch.autocast(device_type=device.type, enabled=False):
        imgs_fp32 = imgs.float()
        imgs_norm = imgs_fp32 * 2.0 - 1.0
        imgs_256  = F.interpolate(imgs_norm, size=(GAN_SIZE, GAN_SIZE),
                                  mode='bilinear', align_corners=False)

        out_256 = gan.generator(imgs_256)

        # Some generator architectures return a list/tuple of outputs
        if isinstance(out_256, (list, tuple)):
            out_256 = out_256[-1]

        # Lightweight safety net: forcing FP32 above should prevent this in
        # practice (see docstring). If it still triggers, it's a genuine
        # numerical issue inside gan.generator(), not a precision artifact.
        if not torch.isfinite(out_256).all():
            n_bad = (~torch.isfinite(out_256)).any(dim=(1, 2, 3)).sum().item()
            print(f"  [WARN] gan.generator() produced {n_bad}/{out_256.shape[0]} "
                  f"non-finite outputs even in FP32")

        out = F.interpolate(out_256, size=(imgs.shape[2], imgs.shape[3]),
                            mode='bilinear', align_corners=False)
        result = torch.clamp((out + 1.0) / 2.0, 0.0, 1.0)

    return result.to(input_dtype)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*55}")
    print(f"  Joint Training v2 — mode: {args.mode}")
    print(f"  Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'='*55}\n")

    model = YOLO('yolo26n.pt')

    gan = None
    if args.mode in ('gan_frozen', 'joint'):
        if not args.gan_ckpt:
            raise ValueError("--gan_ckpt is required for gan_frozen and joint modes.")
        gan = load_gan(args.gan_ckpt, device)

    # Shared mutable state accessed by all callbacks
    state = {
        'current_epoch':  0,
        'optimizer_gan':  None,
        'imgs_original':  None,
        'imgs_gan':       None,
        'gan':            gan,
        'mode':           args.mode,
        'phase1_epochs':  args.phase1_epochs,
        'freeze_encoder': args.freeze_encoder,
        'device':         device,
        'debug_example_saved': False,
    }

    _batch_losses: list[tuple[float, float]] = []

    # ── Callback: training start ───────────────────────────────────────────────
    def on_train_start(trainer):
        """
        Set custom loss names AND monkey-patch trainer.preprocess_batch to
        apply the GAN right after Ultralytics' own preprocessing.

        IMPORTANT FIX (found via debugging): earlier versions of this script
        tried to intercept images via `trainer.batch` inside an
        `on_train_batch_start` callback. That attribute does NOT exist on
        DetectionTrainer in this Ultralytics version -- `batch` is a plain
        local variable inside `_do_train()`'s loop, never assigned to
        `self.batch`. As a result, the GAN was silently never applied
        during training (the callback returned early every time, with no
        error). The correct interception point is `trainer.preprocess_batch`,
        whose return value is used directly for the forward pass
        (`batch = self.preprocess_batch(batch); preds = self.model(batch["img"])`),
        so wrapping it here guarantees our GAN-processed tensor -- graph
        intact -- is what the detector actually sees.
        """
        trainer.loss_names = ('box_loss', 'cls_loss', 'dfl_loss')

        if state['mode'] == 'detector_only':
            return

        original_preprocess_batch = trainer.preprocess_batch

        def patched_preprocess_batch(batch):
            batch = original_preprocess_batch(batch)
            return _apply_gan_to_batch(batch)

        trainer.preprocess_batch = patched_preprocess_batch

    def _apply_gan_to_batch(batch):
        """
        Apply the GAN to every image in the training batch.

        By project design, ALL training images -- regardless of source
        (real Pepper video frames, Albumentations-simulated motion blur,
        or plain store images) -- are meant to go through the GAN. There
        is deliberately no per-file classification here: an earlier
        version tried to infer "is this a video frame?" from the filename
        (substring/regex matching), which is exactly what caused a serious
        bug (folder names like '..._senza_video' contain the substring
        'video' but mean the opposite). Routing 100% of images uniformly
        removes that entire class of bug.
        """
        if 'img' not in batch:
            return batch

        # During joint warm-up phase, skip GAN processing
        if state['mode'] == 'joint' and state['current_epoch'] < state['phase1_epochs']:
            return batch

        dev  = state['device']
        imgs = batch['img']  # already normalised + on-device by Ultralytics' own preprocess_batch

        with (torch.no_grad() if state['mode'] == 'gan_frozen' else torch.enable_grad()):
            imgs_gan = apply_gan(state['gan'], imgs, dev)

        # Safety net: apply_gan() now runs in forced FP32 (see its
        # docstring) so this should not trigger in practice. If it ever
        # does, log which file caused it instead of failing silently.
        if not torch.isfinite(imgs_gan).all():
            img_paths = batch.get('im_file', batch.get('img_path', []))
            bad_idx = (~torch.isfinite(imgs_gan)).any(dim=(1, 2, 3)).nonzero(as_tuple=True)[0].tolist()
            bad_files = [img_paths[i] for i in bad_idx if i < len(img_paths)]
            print(f"  [WARN] non-finite GAN output for: {bad_files}")

        state['imgs_gan'] = imgs_gan
        batch['img'] = imgs_gan
        return batch

    # ── Callback: epoch start ──────────────────────────────────────────────────
    def on_train_epoch_start(trainer):
        """
        Configure GAN gradient flow at the start of each epoch.

        - detector_only: nothing to do.
        - gan_frozen:    GAN eval mode, no gradients.
        - joint / phase 1 (warm-up): GAN frozen.
        - joint / phase 2: GAN in train mode; initialise the GAN optimiser
          on first entry and respect the --freeze_encoder setting.
        """
        state['current_epoch'] = trainer.epoch
        epoch = trainer.epoch

        if state['mode'] == 'detector_only':
            return

        g = state['gan'].generator

        if state['mode'] == 'gan_frozen':
            g.eval()
            for param in g.parameters():
                param.requires_grad = False

        elif state['mode'] == 'joint':
            if epoch < state['phase1_epochs']:
                # Phase 1: warm up the detector with the GAN frozen
                g.eval()
                for param in g.parameters():
                    param.requires_grad = False
                print(f"  [Epoch {epoch+1}] Phase 1: warm-up detector, GAN frozen")
            else:
                # Phase 2: joint training — initialise GAN optimiser on first entry
                if state['optimizer_gan'] is None:
                    if state['freeze_encoder']:
                        # Freeze the U-Net encoder; train only bottleneck and decoder
                        trainable_params = []
                        n_frozen = 0
                        for name, param in g.named_parameters():
                            if any(name.startswith(pfx) for pfx in ENCODER_PREFIXES):
                                param.requires_grad = False
                                n_frozen += 1
                            else:
                                param.requires_grad = True
                                trainable_params.append(param)
                        state['optimizer_gan'] = torch.optim.Adam(
                            trainable_params, lr=1e-4, betas=(0.5, 0.999)
                        )
                        print(
                            f"  [Epoch {epoch+1}] Phase 2: encoder frozen "
                            f"({n_frozen} params), bottleneck+decoder trainable"
                        )
                    else:
                        state['optimizer_gan'] = torch.optim.Adam(
                            g.parameters(), lr=1e-4, betas=(0.5, 0.999)
                        )
                        print(f"  [Epoch {epoch+1}] Phase 2: joint training — all GAN layers trainable")

                # Ensure requires_grad is consistent with --freeze_encoder
                g.train()
                for name, param in g.named_parameters():
                    if state['freeze_encoder'] and any(
                        name.startswith(pfx) for pfx in ENCODER_PREFIXES
                    ):
                        param.requires_grad = False
                    else:
                        param.requires_grad = True

    # ── Callback: batch end ────────────────────────────────────────────────────
    def on_train_batch_end(trainer):
        """
        Update GAN weights (joint mode, phase 2 only) and log batch losses.
        """
        if (
            state['mode'] == 'joint'
            and state['current_epoch'] >= state['phase1_epochs']
            and state['optimizer_gan'] is not None
        ):
            torch.nn.utils.clip_grad_norm_(
                state['gan'].generator.parameters(), max_norm=5.0
            )
            state['optimizer_gan'].step()
            state['optimizer_gan'].zero_grad()

        state['imgs_original'] = None
        state['imgs_gan']      = None

        items = getattr(trainer, 'loss_items', None)
        if items is None:
            return

        _batch_losses.append((float(items[0]), float(items[1])))

        batch  = getattr(trainer, 'batch_i', 0) + 1
        n_bat  = getattr(trainer, 'nb', '?')
        ep     = trainer.epoch + 1
        ep_tot = trainer.epochs
        print(
            f"\r  [{ep}/{ep_tot}] batch {batch}/{n_bat} │ "
            f"box_loss={float(items[0]):.4f}  cls_loss={float(items[1]):.4f}   ",
            end='', flush=True,
        )

    # ── Callback: epoch end ────────────────────────────────────────────────────
    def on_train_epoch_end(trainer):
        """
        Log epoch-average losses and save the GAN checkpoint every 5 epochs
        (joint mode only, after the warm-up phase).
        """
        if _batch_losses:
            avg_box = sum(x[0] for x in _batch_losses) / len(_batch_losses)
            avg_cls = sum(x[1] for x in _batch_losses) / len(_batch_losses)
            ep      = trainer.epoch + 1
            ep_tot  = trainer.epochs
            print(
                f"\n  ── Epoch {ep}/{ep_tot} ──  "
                f"box_loss={avg_box:.4f}  cls_loss={avg_cls:.4f}"
            )
            _batch_losses.clear()

        if state['mode'] != 'joint' or state['current_epoch'] < state['phase1_epochs']:
            return

        ep = state['current_epoch'] + 1
        if ep % 5 == 0:
            # Save the GAN checkpoint alongside the YOLO weights
            run_dir  = getattr(trainer, 'save_dir', None)
            ckpt_dir = (
                os.path.join(str(run_dir), 'weights')
                if run_dir
                else os.path.join(args.project, f'oos_{args.mode}', 'weights')
            )
            os.makedirs(ckpt_dir, exist_ok=True)
            gan_path = os.path.join(ckpt_dir, f'blur2blur_joint_epoch{ep}.pt')
            torch.save(state['gan'].generator.state_dict(), gan_path)
            print(f"  GAN checkpoint saved: {gan_path}")

    # ── Register callbacks ─────────────────────────────────────────────────────
    model.add_callback('on_train_start',       on_train_start)
    model.add_callback('on_train_epoch_start', on_train_epoch_start)
    model.add_callback('on_train_batch_end',   on_train_batch_end)
    model.add_callback('on_train_epoch_end',   on_train_epoch_end)

    # ── Training hyperparameters ───────────────────────────────────────────────
    # Official YOLO26n recipe: https://docs.ultralytics.com/guides/yolo26-training-recipe
    #   detector_only → SGD,   lr0=0.0054
    #   gan_frozen    → AdamW, lr0=0.001  (more stable with GAN pre-processing)
    #   joint         → AdamW, lr0=0.001  (co-optimisation of GAN + detector)
    if args.mode == 'detector_only':
        optimizer = 'SGD'
        lr0       = args.lr0 if args.lr0 is not None else 0.0054
        warmup_ep = 0.98
        patience  = 50
    else:
        optimizer = 'AdamW'
        lr0       = args.lr0 if args.lr0 is not None else 0.001
        warmup_ep = 5.0
        patience  = 20

    # Hyperparameters common to all modes (YOLO26n official recipe)
    TRAIN_COMMON = dict(
        data            = args.data,
        epochs          = args.epochs,
        batch           = args.batch,
        imgsz           = args.imgsz,
        device          = 0 if torch.cuda.is_available() else 'cpu',
        project         = args.project,
        name            = args.name if args.name is not None else f'oos_{args.mode}',
        optimizer       = optimizer,
        lr0             = lr0,
        lrf             = 0.0495,
        momentum        = 0.947,
        weight_decay    = 0.00064,
        warmup_epochs   = warmup_ep,
        warmup_momentum = 0.8,
        warmup_bias_lr  = 0.1,
        box             = 5.63,
        cls             = 0.56,
        dfl             = 9.04,
        patience        = patience,
        freeze          = 10,
        save_period     = 10,
        exist_ok        = False,
        workers         = 4,
        verbose         = True,
        cos_lr          = True,
        multi_scale     = False,
    )

    print(
        f"Starting training: {args.epochs} epochs, batch={args.batch}, "
        f"imgsz={args.imgsz}, optimizer={optimizer}, lr0={lr0}"
    )

    if args.mode == 'detector_only':
        if args.augment:
            aug_params = dict(
                degrees      = 1.11,
                scale        = 0.56,
                shear        = 1.46,
                translate    = 0.071,
                mosaic       = 0.70,
                close_mosaic = 10,
                mixup        = 0.012,
                copy_paste   = 0.075,
                erasing      = 0.45,
                hsv_h        = 0.015,
                hsv_s        = 0.7,
                hsv_v        = 0.4,
                bgr          = 0.0,
                flipud       = 0.0,
                fliplr       = 0.5,
            )
            print("  Augmentation ON:")
            for k, v in aug_params.items():
                print(f"    {k:15s} = {v}")
        else:
            aug_params = dict(
                degrees=0.0, scale=0.0, shear=0.0, translate=0.0,
                mosaic=0.0, close_mosaic=0, mixup=0.0, copy_paste=0.0,
                erasing=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
                bgr=0.0, flipud=0.0, fliplr=0.0,
            )
            print("  Augmentation OFF")

        model.train(**TRAIN_COMMON, **aug_params)

    else:
        # gan_frozen / joint: use standard YOLO26n geometric augmentation
        model.train(
            **TRAIN_COMMON,
            degrees      = 1.11,
            scale        = 0.56,
            shear        = 1.46,
            translate    = 0.071,
            mosaic       = 0.70,
            close_mosaic = 10,
            mixup        = 0.012,
            copy_paste   = 0.075,
            erasing      = 0.45,
            hsv_h        = 0.015,
            hsv_s        = 0.7,
            hsv_v        = 0.4,
            bgr          = 0.0,
            flipud       = 0.0,
            fliplr       = 0.5,
        )

    run_name = args.name if args.name is not None else f'oos_{args.mode}'
    print(f"\nTraining complete — results in: {args.project}/{run_name}/")


if __name__ == '__main__':
    main()