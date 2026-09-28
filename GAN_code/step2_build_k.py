"""
step2_build_k.py
================
STEP 2 of 2 — Generate trainB and trainC using the Blur2Blur KernelWizard.

Requires a GPU. Run directly or submit as an HPC job.

Prerequisite: step1_build_dataset.py must have been run first.
              trainA and trainD must already exist in the data directory.

What it does:
  For each sharp frame in trainD (S):
    1. Randomly samples a blurry frame from trainA (B)
    2. Feeds the pair (sharp_frame, blurry_frame) to the KernelWizard
       → the model extracts the blur kernel from the blurry frame
    3. Applies that kernel to the sharp frame via adaptKernel
       → produces K: a sharp frame degraded with realistic video blur
    4. Saves K to trainB and trainC (identical copies, required by the dataloader)

Why KernelWizard instead of Gaussian blur:
  Gaussian blur is isotropic (equal in all directions), whereas real video
  motion blur is anisotropic (directional). Using the KernelWizard with
  GoPro-pretrained weights transfers realistic blur kernels, following the
  guidelines of the Blur2Blur paper (CVPR 2024).

Configuration used:
  - Config: augmentation.yml (GOPRO_woVAE.pth, use_vae=False)
  - use_vae=False: deterministic kernel (no VAE stochasticity)
  - Weights: pre-trained on the GoPro dataset (240fps, real motion blur)

Usage:
    python step2_build_k.py --data_dir <path/to/data> \\
        --blur2blur_dir <path/to/Blur2Blur>
"""

import os
import sys
import shutil
import random
import argparse
import torch
import yaml
from pathlib import Path
from PIL import Image
from torchvision import transforms


# ── CONFIG ────────────────────────────────────────────────────────────────────
SEED     = 42
IMG_SIZE = 256   # spatial resolution used by KernelWizard
# ─────────────────────────────────────────────────────────────────────────────

random.seed(SEED)
torch.manual_seed(SEED)


def parse_args():
    p = argparse.ArgumentParser(
        description='Generate trainB/C (K images) using the Blur2Blur KernelWizard.'
    )
    p.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Root directory of the Blur2Blur dataset (contains trainA and trainD).',
    )
    p.add_argument(
        '--blur2blur_dir',
        type=str,
        required=True,
        help='Path to the Blur2Blur repository root (must contain options/ and models/).',
    )
    p.add_argument(
        '--img_size',
        type=int,
        default=IMG_SIZE,
        help=f'Spatial resolution for KernelWizard input (default: {IMG_SIZE}).',
    )
    return p.parse_args()


def load_kernel_wizard(blur2blur_dir: Path, device: torch.device):
    """
    Load the KernelWizard model pre-trained on the GoPro dataset.

    The KernelWizard consists of:
      - KernelExtractor: extracts the blur kernel from a (sharp, blurry) pair
                         → produces kernel_mean and kernel_sigma
      - KernelAdapter:   applies the extracted kernel to a new sharp image
                         via adaptKernel(sharp, kernel) → blurred image K

    Args:
        blur2blur_dir: Path to the Blur2Blur repository root.
        device:        Torch device to load the model onto.

    Returns:
        KernelWizard model in eval mode.
    """
    config_path = blur2blur_dir / 'options' / 'generate_blur' / 'augmentation.yml'
    ckpt_path   = blur2blur_dir / 'options' / 'generate_blur' / 'GOPRO_woVAE.pth'

    sys.path.insert(0, str(blur2blur_dir))
    from models.explore.kernel_encoding.kernel_wizard import KernelWizard

    with open(config_path) as f:
        opt = yaml.load(f, Loader=yaml.FullLoader)['KernelWizard']

    model = KernelWizard(opt)
    model.load_state_dict(
        torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
    )
    model.eval()
    model.to(device)

    print(f"  KernelWizard loaded")
    print(f"  Config:  {config_path}")
    print(f"  Weights: {ckpt_path}")
    print(f"  use_vae: {opt.get('use_vae', False)}")
    return model


def load_image_tensor(path: Path, size: int, device: torch.device) -> torch.Tensor:
    """
    Load an image as a float tensor of shape (1, 3, size, size) in [0, 1].

    Args:
        path:   Path to the image file.
        size:   Target spatial resolution (height and width).
        device: Torch device.

    Returns:
        Float tensor of shape (1, 3, size, size).
    """
    img = Image.open(path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])
    return transform(img).unsqueeze(0).to(device)


def save_tensor_image(tensor: torch.Tensor, path: Path) -> None:
    """
    Save a tensor of shape (1, 3, H, W) as a JPEG image.

    Args:
        tensor: Float tensor with values in [0, 1].
        path:   Destination file path.
    """
    img = transforms.ToPILImage()(tensor.squeeze(0).clamp(0, 1).cpu())
    img.save(str(path), quality=95)


def check_prerequisites(data_dir: Path) -> None:
    """
    Verify that trainA and trainD exist and are non-empty.

    Args:
        data_dir: Root directory of the Blur2Blur dataset.

    Raises:
        RuntimeError: If either directory is missing or empty.
    """
    for folder in ['trainA', 'trainD']:
        p = data_dir / folder
        if not p.exists() or not any(p.glob('*.jpg')):
            raise RuntimeError(
                f"Directory {p} is missing or empty.\n"
                f"Run step1_build_dataset.py first."
            )


def generate_K(
    kernel_wizard,
    train_a: list[Path],
    train_d: list[Path],
    data_dir: Path,
    img_size: int,
    device: torch.device,
) -> tuple[int, int]:
    """
    Generate K images for every sharp frame in trainD.

    For each sharp frame S_i in trainD:
      1. Randomly sample a blurry frame B_j from trainA
      2. Load both as (1, 3, img_size, img_size) tensors in [0, 1]
      3. KernelWizard.forward(S_i, B_j) → extracts the blur pattern from B_j
         (the model captures the blur difference between S_i and B_j)
      4. KernelWizard.adaptKernel(S_i, kernel) → applies blur to S_i
      5. Save result to both trainB and trainC (identical — required by the dataloader)

    Args:
        kernel_wizard: Loaded KernelWizard model.
        train_a:       List of blurry frame paths (trainA).
        train_d:       List of sharp frame paths  (trainD).
        data_dir:      Root directory of the dataset.
        img_size:      Spatial resolution for KernelWizard.
        device:        Torch device.

    Returns:
        Tuple (n_success, n_failed).
    """
    # Recreate trainB and trainC
    for folder in ['trainB', 'trainC']:
        p = data_dir / folder
        if p.exists():
            shutil.rmtree(p)
        p.mkdir(parents=True)

    n_success, n_failed = 0, 0

    with torch.no_grad():
        for i, sharp_path in enumerate(train_d):
            if i % 50 == 0:
                print(f"    {i}/{len(train_d)}...", end='\r')

            try:
                # Sample a random blurry frame as the blur source
                blur_path = random.choice(train_a)

                # Load as (1, 3, img_size, img_size) tensors in [0, 1]
                sharp_t = load_image_tensor(sharp_path, img_size, device)
                blur_t  = load_image_tensor(blur_path,  img_size, device)

                # Extract kernel: forward(sharp, blurry)
                # Order: sharp first, blurry second
                # The model extracts the blur type present in blur_t
                kernel_mean, kernel_sigma = kernel_wizard(sharp_t, blur_t)

                # With use_vae=False: kernel_sigma ≈ 0 → deterministic kernel
                kernel = kernel_mean

                # Apply the extracted kernel to the sharp frame
                k_image = kernel_wizard.adaptKernel(sharp_t, kernel)
                k_image = k_image.clamp(0, 1)

                # Save to trainB and trainC (identical copies)
                out_name = f"K_{i:05d}_{sharp_path.name}"
                save_tensor_image(k_image, data_dir / 'trainB' / out_name)
                save_tensor_image(k_image, data_dir / 'trainC' / out_name)
                n_success += 1

            except Exception as e:
                print(f"\n    [WARN] Error on {sharp_path.name}: {e}")
                n_failed += 1

    return n_success, n_failed


def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_dir     = Path(args.data_dir).expanduser()
    blur2blur_dir = Path(args.blur2blur_dir).expanduser()

    print(f"\n{'='*60}")
    print(f"  STEP 2 — Generate K images with KernelWizard")
    print(f"  Device:         {device}")
    print(f"  Data dir:       {data_dir}")
    print(f"  Blur2Blur dir:  {blur2blur_dir}")
    print(f"{'='*60}\n")

    check_prerequisites(data_dir)

    # Step 1 — Load KernelWizard
    print("[ 1/3 ] Loading KernelWizard...")
    kw = load_kernel_wizard(blur2blur_dir, device)

    # Step 2 — Collect images
    print("\n[ 2/3 ] Collecting images...")
    train_a = sorted(data_dir.glob('trainA/*.jpg'))
    train_d = sorted(data_dir.glob('trainD/*.jpg'))

    print(f"  trainA (B, blurry frames): {len(train_a)}")
    print(f"  trainD (S, sharp frames):  {len(train_d)}")
    print(f"  Effective B:S ratio:        {len(train_a)/max(len(train_d), 1):.2f}")

    # Step 3 — Generate K images
    print(f"\n[ 3/3 ] Generating K images ({len(train_d)} frames)...")
    n_success, n_failed = generate_K(
        kw, train_a, train_d, data_dir, args.img_size, device
    )

    n_train_b = len(list(data_dir.glob('trainB/*.jpg')))
    n_train_c = len(list(data_dir.glob('trainC/*.jpg')))

    print(f"""
{'='*60}
  STEP 2 complete

  Final dataset for GAN training:
    trainA/ : {len(train_a)} images  ← B: blurry frames (video)
    trainB/ : {n_train_b} images  ← K: sharp frames + real blur (KernelWizard)
    trainC/ : {n_train_c} images  ← copy of trainB (required by dataloader)
    trainD/ : {len(train_d)} images  ← S: sharp frames (video)
    Failed:   {n_failed}

  B:S ratio = {len(train_a)}:{len(train_d)} ≈ {len(train_a)/max(len(train_d),1):.2f}  (target: 6:4 = 1.50)
{'='*60}
""")


if __name__ == '__main__':
    main()