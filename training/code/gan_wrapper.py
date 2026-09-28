# oos_orlando/training/code/gan_wrapper.py
import sys
import os

import torch
import torch.nn as nn


class Blur2BlurWrapper(nn.Module):
    """
    Wrapper that exposes the Blur2Blur Generator for joint training.

    Notes on resizing:
        The generator uses a UNet-256, so it requires 256x256 input.
        The caller (train_joint.py) must resize BEFORE calling
        gan.generator(imgs_for_gan) and then resize the output back to the
        detector's img_size. Same applies to get_gan_losses.
    """

    def __init__(self, checkpoint_path=None,
                 blur2blur_root=None, data_root=None):
        super().__init__()

        current_script_dir = os.path.dirname(os.path.abspath(__file__))

        # blur2blur_root / data_root default to sibling directories two
        # levels up from this file (i.e. <project_root>/Blur2Blur and
        # <project_root>/blur2blur_data). Override via the constructor
        # arguments or the BLUR2BLUR_ROOT / BLUR2BLUR_DATA_ROOT
        # environment variables if your directory layout differs.
        self.blur2blur_root = os.path.abspath(
            blur2blur_root
            or os.environ.get('BLUR2BLUR_ROOT')
            or os.path.join(current_script_dir, '../../Blur2Blur')
        )
        self.data_root = os.path.abspath(
            data_root
            or os.environ.get('BLUR2BLUR_DATA_ROOT')
            or os.path.join(current_script_dir, '../../blur2blur_data')
        )

        if self.blur2blur_root not in sys.path:
            sys.path.insert(0, self.blur2blur_root)

        from options.train_options import TrainOptions
        from models import create_model

        old_cwd = os.getcwd()
        os.chdir(self.blur2blur_root)

        try:
            backup_argv = sys.argv
            # Force CPU if CUDA is not available:
            # base_options.py calls torch.cuda.set_device(gpu_ids[0])
            # even without a GPU, causing a RuntimeError.
            # '--gpu_ids -1' disables CUDA across the whole Blur2Blur stack.
            gpu_args = ['--gpu_ids', '-1'] if not torch.cuda.is_available() else []
            sys.argv = [
                'train.py',
                '--dataroot', self.data_root,
                '--name',     'blur2blur_oos',
                '--model',    'blur2blur',
                '--display_id', '-1',
                '--no_dropout',
            ] + gpu_args

            self.opt   = TrainOptions().parse()
            sys.argv   = backup_argv

            # CPU patch: vgg_loss.py calls .cuda() hardcoded.
            # When CUDA is not available we temporarily replace
            # Module.cuda() and Tensor.cuda() with no-ops that return
            # self / the tensor unchanged.
            if not torch.cuda.is_available():
                import torch.nn as _nn
                _orig_module_cuda = _nn.Module.cuda
                _orig_tensor_cuda = torch.Tensor.cuda
                _nn.Module.cuda  = lambda self, *a, **kw: self
                torch.Tensor.cuda = lambda self, *a, **kw: self

            try:
                self.model = create_model(self.opt)
                self.model.setup(self.opt)
            finally:
                if not torch.cuda.is_available():
                    _nn.Module.cuda  = _orig_module_cuda
                    torch.Tensor.cuda = _orig_tensor_cuda

            if checkpoint_path:
                self.load_checkpoint(checkpoint_path)

            # Expose only the generator for joint training
            # Extract the raw generator, removing any DataParallel wrapper
            netG = self.model.netG
            if isinstance(netG, torch.nn.DataParallel):
                netG = netG.module
            self.generator = netG

        finally:
            os.chdir(old_cwd)

    def forward(self, x):
        """
        Input:  256x256 image (batch, 3, 256, 256)
        Output: transformed 256x256 image (batch, 3, 256, 256)
        Note: resizing to 256 must be done by the caller.
        """
        return self.generator(x)

    def load_generator_weights(self, path):
        """
        Load generator weights with automatic key fixing.
        Handles both keys with a 'model.' prefix and without.
        """
        state = torch.load(path, map_location='cpu', weights_only=False)

        # Get the current generator's keys
        gen = self.generator
        gen_keys = set(gen.state_dict().keys())

        # Try loading directly
        missing = [k for k in gen_keys if k not in state]

        if missing:
            # Try removing the 'model.' prefix
            state_fixed = {}
            for k, v in state.items():
                new_k = k[len('model.'):] if k.startswith('model.') else k
                state_fixed[new_k] = v
            missing_fixed = [k for k in gen_keys if k not in state_fixed]

            if len(missing_fixed) < len(missing):
                state = state_fixed
                print(f"  Key fix applied (removed 'model.' prefix)")

        m, u = gen.load_state_dict(state, strict=False)
        print(f"  Generator weights loaded from: {path}")
        print(f"  Missing: {len(m)}, Unexpected: {len(u)}")
        if m:
            print(f"  [WARN] Missing keys: {m[:3]}")
        return len(m) == 0

    def load_checkpoint(self, path):
        """Load pre-trained GAN weights (backward compatibility)."""
        return self.load_generator_weights(path)
