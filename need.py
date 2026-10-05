"""NEED (SPDiff + DGDiff) trained and evaluated on the ULDCT dataset.

Stage 1 (SPDiff) is a Poisson diffusion model with three-slice context that
runs in the projection domain, as in the paper: full-dose images are
forward-projected to fan-beam transmission sinograms (672x672, see
``img2sino``), where the Poisson degradation is physically meaningful. The
network learns to map Poisson-degraded full-dose sinograms back to the clean
ones. At test time the real LDCT slice is projected, sampled from the
dose-matched step, and reconstructed with FBP to a 512x512 image.

Stage 2 (DGDiff) refines the reconstructed image. It requires the
``denoising_diffusion_pytorch`` package of NEED-main/DGDiff (``--dgdiff-root``)
and a trained ``dgdiff_stage2.pt`` in the output folder; without them only
SPDiff is used.

Model code adapted from NEED-main/SPDiff (diffusion_modules.py,
SPDiff_wrapper.py, SPDiff.py).

Usage:
    python need.py --dose 5pct --data-root /path/to/uldct_5pct/dataset
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

import img2sino
from common.checkpoint import best_psnr_so_far, load_checkpoint
from common.config import (
    DATA_RANGE,
    LOW_DOSE_DIR,
    TEST_SPLIT,
    TRAIN_SPLIT,
    VAL_SPLIT,
    RunConfig,
    build_arg_parser,
    get_device,
    parse_run_config,
)
from common.data import (
    group_by_patient_sorted,
    list_low_dose_inputs,
    list_normal_dose,
    load_input_array,
    normalize,
    to_hu_window,
)
from common.evaluation import (
    ABDOMEN_FIG_NAME,
    NUM_TEST_FIGURES,
    abdomen_fig_index,
    save_fig,
    write_test_metrics,
)
from common.metrics import compute_measure, compute_psnr, compute_rmse, compute_ssim
from common.runtime import optional_split

try:
    import pydicom
except ImportError:
    pydicom = None

MODEL_NAME = "need"

# Hyperparameters (NEED-main/SPDiff: train_SPDiff.sh and SPDiff.py defaults).
NUM_EPOCHS = 10_000  # upper bound; training stops at MAX_ITERS
MAX_ITERS = 100_000  # train_SPDiff.sh --max_iter
BATCH_SIZE = 8  # train_SPDiff.sh --batch_size
# The network runs on the 672x672 fan-beam sinogram (N_ANGLES = DETECT_NUMBER).
SINO_SIZE = 672
LR = 2e-4  # SPDiff.py init_lr
TIMESTEPS = 10
# SPDiff does not sample from the full T: it starts at the step whose photon
# level lambda_ matches the LD dose, ld_I0 = I0 / dose_factor
# (match_t = argmin |lambda_ - ld_I0|, sample(t=match_t + 1)).
I0 = 2.5e5
DOSE_FACTOR = {"5pct": 20, "10pct": 10}
DEFAULT_DOSE_FACTOR = 10
LAMBDA_0, LAMBDA_T = 3e5, 2.5e4  # frac_schedule endpoints
EMA_DECAY = 0.995  # SPDiff.py ema_decay
EMA_UPDATE_EVERY = 10  # SPDiff.py update_ema_iter
EMA_START_ITER = 2000  # SPDiff.py start_ema_iter
USE_EMA = True
PRINT_ITERS = 500
SAVE_ITERS = 2500  # train_SPDiff.sh --save_freq
USE_CONTEXT = True  # three-slice context (above, current, below)
PREFETCH_FACTOR = 4
# Checkpoints are tagged with their domain so an older image-domain checkpoint
# (same architecture, different training domain) is never resumed.
CKPT_DOMAIN = "sino"

# Mayo-2016 protocol of the paper: 8 training and 2 test patients. Validation
# is restricted to the same two patient ids as the test split.
USE_PAPER_PATIENT_SPLIT = True
PAPER_TRAIN_PATIENTS = {"L067", "L096", "L109", "L192", "L286", "L291", "L310", "L333"}
PAPER_TEST_PATIENTS = {"L143", "L506"}
PAPER_VAL_PATIENTS = PAPER_TEST_PATIENTS
VAL_MAX_SLICES = 0  # 0 = full validation at every snapshot

# DGDiff (NEED stage 2). With a stage-2 checkpoint, testing runs
# SPDiff -> FBP -> DGDiff; otherwise SPDiff only.
USE_DGDIFF = True
DGDIFF_REQUIRED = False
DGDIFF_REQUIRE_STAGE1 = False
DGDIFF_STAGE1_CKPT_NAME = "dgdiff_stage1.pt"
DGDIFF_STAGE2_CKPT_NAME = "dgdiff_stage2.pt"
DGDIFF_STAGE1_IMAGE_SIZE = 256
DGDIFF_STAGE2_IMAGE_SIZE = 512
DGDIFF_COND_SIZE = 256
DGDIFF_TIMESTEPS = 1000
DGDIFF_STAGE2_SAMPLING_STEPS = 30
DGDIFF_SAMPLER = "ddim"

DEFAULT_FD_CACHE_DIR = Path("cache") / "fd_target_npy"
# Figures use the evaluation window.
DISPLAY_TRUNC_MIN, DISPLAY_TRUNC_MAX = -160.0, 240.0

logger = logging.getLogger(MODEL_NAME)


# ============================================================================
# Model: SPDiff (diffusion_modules.py, SPDiff_wrapper.py)
# ============================================================================
def extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    """Gather ``a[t]`` per batch element, shaped for broadcasting over ``x``."""
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class Diffusion(nn.Module):
    """Poisson diffusion of SPDiff.

    The forward process degrades a sinogram with Poisson noise at photon level
    ``lambda_[t]``; the network predicts the clean sinogram directly.

    Attributes:
        denoise_fn: Denoising network ``f(x_t, t)``.
        num_timesteps: Number of diffusion steps.
        context: Whether inputs carry three slices (above, current, below).
        lambda_: Photon-count schedule, one value per step (buffer).
    """

    def __init__(
        self,
        denoise_fn: nn.Module | None = None,
        img_size: int = 512,
        channels: int = 1,
        timesteps: int = 10,
        context: bool = True,
        lambda_: torch.Tensor | None = None,
    ) -> None:
        """Store the network and the schedule.

        Args:
            denoise_fn: Denoising network.
            img_size: Spatial size of the (square) inputs.
            channels: Number of output channels.
            timesteps: Number of diffusion steps.
            context: Whether inputs carry three slices.
            lambda_: Photon-count schedule of length ``timesteps``.
        """
        super().__init__()
        self.channels = channels
        self.img_size = img_size
        self.denoise_fn = denoise_fn
        self.num_timesteps = int(timesteps)
        self.context = context
        self.register_buffer("lambda_", lambda_)

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Degrade ``x_start`` with Poisson noise at step ``t`` (none at t=0)."""
        lambda_t = extract(self.lambda_, t, x_start.shape)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_start.shape) - 1)))
        return (
            nonzero_mask * torch.poisson(lambda_t * x_start + 10) / lambda_t
            + (1 - nonzero_mask) * x_start
        )

    @torch.no_grad()
    def sample(
        self,
        batch_size: int = 16,
        img: torch.Tensor | None = None,
        t: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the reverse process from step ``t`` starting at ``img``.

        Args:
            batch_size: Batch size of ``img``.
            img: Degraded input, ``(B, 3, H, W)`` with context.
            t: Starting step; defaults to ``num_timesteps``.

        Returns:
            The final estimate clipped to ``[0, 1]``, the stacked direct ``x0``
            predictions and the stacked intermediate images.
        """
        self.denoise_fn.eval()
        if t is None:
            t = self.num_timesteps

        if self.context:
            up_img = img[:, 0].unsqueeze(1)
            down_img = img[:, 2].unsqueeze(1)
            img = img[:, 1].unsqueeze(1)

        direct_recons = []
        imstep_imgs = []

        while t:
            step = torch.full((batch_size,), t - 1, dtype=torch.long, device=img.device)
            full_img = (
                torch.cat((up_img, img, down_img), dim=1) if self.context else img
            )
            pred_x0 = self.denoise_fn(full_img, step)
            direct_recons.append(pred_x0)

            if t > 2:
                lambda_t = extract(self.lambda_, step, img.shape)
                lambda_t_sub1 = extract(self.lambda_, step - 1, img.shape)
                # pred_x0 is a sinogram (>= 0), but the network may predict
                # negatives (mostly early in training): a negative Poisson rate
                # triggers a CUDA assert.
                rate = ((lambda_t_sub1 - lambda_t) * pred_x0).clamp(min=0.0)
                tau_t = torch.poisson(rate)
                img = (1 / lambda_t_sub1) * (lambda_t * img + tau_t)
            else:
                img = pred_x0
            imstep_imgs.append(img)
            t = t - 1
        return img.clamp(0.0, 1.0), torch.stack(direct_recons), torch.stack(imstep_imgs)

    def forward(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Degrade ``y`` at a random step and predict the clean middle slice.

        Args:
            y: Clean sinograms ``(B, 3, H, W)`` with context.

        Returns:
            The network prediction and the degraded input.

        Raises:
            ValueError: If the spatial size differs from ``img_size``.
        """
        b, _, h, w = y.shape
        if h != self.img_size or w != self.img_size:
            raise ValueError(f"height and width of image must be {self.img_size}")

        t = torch.randint(0, self.num_timesteps, (b,), device=y.device).long()

        if self.context:
            x_mix = self.q_sample(x_start=y[:, 1].unsqueeze(1), t=t)
            x_mix_up = self.q_sample(x_start=y[:, 0].unsqueeze(1), t=t)
            x_mix_down = self.q_sample(x_start=y[:, 2].unsqueeze(1), t=t)
            x_mix = torch.cat((x_mix_up, x_mix, x_mix_down), dim=1)
        else:
            x_mix = self.q_sample(x_start=y, t=t)

        return self.denoise_fn(x_mix, t), x_mix


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal embedding of the diffusion step."""

    def __init__(self, dim: int) -> None:
        """Initialise the module.

        Args:
            dim: Embedding size.
        """
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed a batch of steps ``(B,)`` into ``(B, dim)``."""
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class single_conv(nn.Module):
    """3x3 convolution followed by ReLU."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        """Initialise the module.

        Args:
            in_ch: Input channels.
            out_ch: Output channels.
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the block."""
        return self.conv(x)


class up(nn.Module):
    """2x transposed-convolution upsampling with an additive skip connection."""

    def __init__(self, in_ch: int) -> None:
        """Initialise the module.

        Args:
            in_ch: Input channels (halved by the upsampling).
        """
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Upsample ``x1``, pad it to the size of ``x2`` and add ``x2``."""
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(
            x1, (diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2)
        )
        return x2 + x1


class outconv(nn.Module):
    """1x1 output convolution."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        """Initialise the module.

        Args:
            in_ch: Input channels.
            out_ch: Output channels.
        """
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the convolution."""
        return self.conv(x)


class UNet(nn.Module):
    """Two-level SPDiff UNet with additive step conditioning at every level."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1) -> None:
        """Build the layers.

        Args:
            in_channels: Input channels (3 with slice context).
            out_channels: Output channels.
        """
        super().__init__()

        dim = 32
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

        self.inc = nn.Sequential(single_conv(in_channels, 64), single_conv(64, 64))

        self.down1 = nn.AvgPool2d(2)
        self.mlp1 = nn.Sequential(nn.GELU(), nn.Linear(dim, 64))
        self.conv1 = nn.Sequential(
            single_conv(64, 128), single_conv(128, 128), single_conv(128, 128)
        )

        self.down2 = nn.AvgPool2d(2)
        self.mlp2 = nn.Sequential(nn.GELU(), nn.Linear(dim, 128))
        self.conv2 = nn.Sequential(
            single_conv(128, 256),
            single_conv(256, 256),
            single_conv(256, 256),
            single_conv(256, 256),
            single_conv(256, 256),
            single_conv(256, 256),
        )

        self.up1 = up(256)
        self.mlp3 = nn.Sequential(nn.GELU(), nn.Linear(dim, 128))
        self.conv3 = nn.Sequential(
            single_conv(128, 128), single_conv(128, 128), single_conv(128, 128)
        )

        self.up2 = up(128)
        self.mlp4 = nn.Sequential(nn.GELU(), nn.Linear(dim, 64))
        self.conv4 = nn.Sequential(single_conv(64, 64), single_conv(64, 64))

        self.outc = outconv(64, out_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict from input ``x`` ``(B, C, H, W)`` at steps ``t`` ``(B,)``."""
        inx = self.inc(x)
        time_emb = self.time_mlp(t)

        down1 = self.down1(inx) + self.mlp1(time_emb)[:, :, None, None]
        conv1 = self.conv1(down1)

        down2 = self.down2(conv1) + self.mlp2(time_emb)[:, :, None, None]
        conv2 = self.conv2(down2)

        up1 = self.up1(conv2, conv1) + self.mlp3(time_emb)[:, :, None, None]
        conv3 = self.conv3(up1)

        up2 = self.up2(conv3, inx) + self.mlp4(time_emb)[:, :, None, None]
        conv4 = self.conv4(up2)

        return self.outc(conv4)


class Network(nn.Module):
    """UNet predicting the clean middle slice as a residual of the input."""

    def __init__(
        self, in_channels: int = 1, out_channels: int = 1, context: bool = True
    ) -> None:
        """Initialise the module.

        Args:
            in_channels: Input channels (3 with slice context).
            out_channels: Output channels.
            context: Whether the input carries three slices.
        """
        super().__init__()
        self.context = context
        self.unet = UNet(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Return ``unet(x, t) + x_middle``."""
        x_middle = x[:, 1].unsqueeze(1) if self.context else x
        return self.unet(x, t) + x_middle


def make_lambda_schedule(
    timesteps: int, lambda_0: float = LAMBDA_0, lambda_t: float = LAMBDA_T
) -> torch.Tensor:
    """Photon-count schedule decreasing from ``lambda_0`` to ``lambda_t``.

    Same as SPDiff ``frac_schedule``: ``lambda_0 / linspace(1, lambda_0 / lambda_t)``.
    """
    x = 1 / torch.linspace(1, lambda_0 / lambda_t, timesteps, dtype=torch.float32)
    return x * lambda_0


def build_diffusion(lambda_sched: torch.Tensor, device: torch.device) -> Diffusion:
    """Create the SPDiff network wrapped in its diffusion process."""
    network = Network(
        in_channels=3 if USE_CONTEXT else 1, out_channels=1, context=USE_CONTEXT
    )
    return Diffusion(
        denoise_fn=network,
        img_size=SINO_SIZE,
        channels=1,
        timesteps=TIMESTEPS,
        context=USE_CONTEXT,
        lambda_=lambda_sched,
    ).to(device)


def match_t_for_dose(model: Diffusion, dose: str) -> int:
    """Dose-matched starting step of the sampling (SPDiff calculate_match_t).

    With the default schedule both 5% and 10% doses map to the last step, so
    ``match_t + 1`` equals the full ``T``; the rule matters for other doses.

    Args:
        model: Diffusion model holding ``lambda_``.
        dose: Dose label.

    Returns:
        ``match_t``; sampling uses ``t = match_t + 1``.
    """
    ld_i0 = I0 / DOSE_FACTOR.get(dose, DEFAULT_DOSE_FACTOR)
    return int(torch.argmin(torch.abs(model.lambda_ - ld_i0)).item())


class SimpleEMA:
    """Exponential moving average of model parameters."""

    def __init__(self, beta: float) -> None:
        """Store the decay factor.

        Args:
            beta: Decay factor.
        """
        self.beta = beta

    def update(self, ema_model: nn.Module, model: nn.Module) -> None:
        """Blend the parameters of ``model`` into ``ema_model``."""
        with torch.no_grad():
            for ep, p in zip(ema_model.parameters(), model.parameters()):
                ep.data.mul_(self.beta).add_(p.data, alpha=1.0 - self.beta)


# ============================================================================
# Data
# ============================================================================
def _paper_patients_for_split(split: str) -> set[str] | None:
    """Patients allowed in ``split`` by the paper protocol (``None`` = all)."""
    if not USE_PAPER_PATIENT_SPLIT:
        return None
    return {
        TRAIN_SPLIT: PAPER_TRAIN_PATIENTS,
        VAL_SPLIT: PAPER_VAL_PATIENTS,
        TEST_SPLIT: PAPER_TEST_PATIENTS,
    }.get(split)


def filter_groups_by_paper_split(
    groups: dict[str, list[str]], split: str
) -> dict[str, list[str]]:
    """Drop the patients that the paper protocol does not assign to ``split``."""
    allowed = _paper_patients_for_split(split)
    if not allowed:
        return groups
    kept = {pid: paths for pid, paths in groups.items() if pid in allowed}
    dropped = sorted(set(groups) - set(kept))
    if dropped:
        logger.info(
            "%s: dropping patients outside the paper protocol: %s", split, dropped
        )
    return kept


def fd_npy_cache_path(path: str, cache_dir: Path) -> Path:
    """Cache file of a decoded DICOM target, keyed by its absolute path."""
    # os.path.abspath (not Path.resolve) keeps symlinks, so existing cache keys
    # remain valid.
    digest = hashlib.sha1(os.path.abspath(path).encode("utf-8")).hexdigest()
    return Path(cache_dir) / f"{digest}.npy"


def load_target_array(path: str, cache_dir: Path | None = None) -> np.ndarray:
    """Load a full-dose target normalised to ``[0, 1]``, caching DICOM decodes.

    Decoding with pydicom is slow; each DICOM is decoded once and the
    normalised array saved to ``cache_dir`` (a dedicated folder, never listed
    as targets). Numerically identical to decoding on the fly.

    Args:
        path: ``.IMA``/``.dcm`` DICOM or ``.npy`` (HU) target.
        cache_dir: Cache folder, or ``None`` to disable caching.
    """
    if path.lower().endswith((".ima", ".dcm")):
        cache = fd_npy_cache_path(path, cache_dir) if cache_dir is not None else None
        if cache is not None and cache.exists():
            return np.load(cache)
        if pydicom is None:
            raise RuntimeError("pydicom is required to read DICOM targets")
        ds = pydicom.dcmread(path, force=True)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        hu = ds.pixel_array.astype(np.float32) * slope + intercept
        arr = normalize(hu).astype(np.float32)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            # Atomic write: several DataLoader workers may decode the same file.
            tmp = cache.with_name(f"{cache.name}.{os.getpid()}.tmp")
            with tmp.open("wb") as fh:  # a file handle stops np.save adding .npy
                np.save(fh, arr)
            tmp.replace(cache)
        return arr
    return normalize(np.load(path).astype(np.float32))


class ULDCTDatasetContext(Dataset):
    """Three-slice LD inputs paired with three-slice full-dose targets.

    Slices are paired by sorted position within each patient; patients are
    filtered by the paper protocol.

    Attributes:
        split: Dataset split.
        ld_triples: ``(previous, current, next)`` LDCT paths per item.
        fd_triples: Matching full-dose target paths.
    """

    def __init__(
        self, split: str, data_root: Path, fd_cache_dir: Path | None = None
    ) -> None:
        """Discover and pair the files of ``split``.

        Args:
            split: Dataset split.
            data_root: Dataset root.
            fd_cache_dir: Cache folder for decoded DICOM targets.

        Raises:
            FileNotFoundError: If the split has no inputs or no pairs.
        """
        self.split = split
        self.fd_cache_dir = fd_cache_dir
        raw = list_low_dose_inputs(data_root, split)
        if not raw:
            raise FileNotFoundError(
                f"No .npy inputs in {Path(data_root) / split / LOW_DOSE_DIR}"
            )
        groups = filter_groups_by_paper_split(group_by_patient_sorted(raw), split)
        ld_triples, fd_triples, skipped = [], [], []
        for pid in sorted(groups):
            lst = groups[pid]
            fd_list = list_normal_dose(Path(data_root), pid, split)
            if not fd_list:
                skipped.append(pid)
                continue
            n = min(len(lst), len(fd_list))
            if n < 3:
                continue
            for i in range(1, n - 1):
                ld_triples.append((lst[i - 1], lst[i], lst[i + 1]))
                fd_triples.append((fd_list[i - 1], fd_list[i], fd_list[i + 1]))
        if skipped:
            logger.info("No full-dose data for %s (split=%s); skipping", skipped, split)
        if not ld_triples:
            raise FileNotFoundError(f"No paired LD/FD triples in split {split!r}")
        self.ld_triples = ld_triples
        self.fd_triples = fd_triples

    def __len__(self) -> int:
        """Number of triples."""
        return len(self.ld_triples)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the ``(3, H, W)`` input and target stacks."""
        x = np.stack([load_input_array(p) for p in self.ld_triples[idx]], axis=0)
        y = np.stack(
            [load_target_array(p, self.fd_cache_dir) for p in self.fd_triples[idx]],
            axis=0,
        )
        return x, y


# ============================================================================
# Projection (image <-> sinogram)
# ============================================================================
def img_to_sino(img_batch: torch.Tensor, radon: img2sino.Radon) -> torch.Tensor:
    """Normalised images ``(B, 3, 512, 512)`` -> transmission ``(B, 3, 672, 672)``."""
    return img2sino.image_to_sinogram(img_batch, radon, is_norm=True)


def sino_to_img(sino_batch: torch.Tensor, radon: img2sino.Radon) -> torch.Tensor:
    """Transmission sinograms -> normalised images in ``[0, 1]`` (FBP)."""
    return img2sino.recon(sino_batch, radon).clamp(0.0, 1.0)


# ============================================================================
# DGDiff bridge (NEED image-domain refinement)
# ============================================================================
_DGDIFF_IMPORT_CACHE: tuple[type, type] | None = None


def _load_dgdiff_classes(dgdiff_root: Path | None) -> tuple[type, type]:
    """Import Unet/GaussianDiffusion of NEED-main/DGDiff and patch conditioning.

    The official file bundles training and inference; only the two classes are
    reused here, with methods replaced so that the condition image ``cond`` is
    forwarded through sampling and training.

    Args:
        dgdiff_root: Folder containing the ``denoising_diffusion_pytorch``
            package of DGDiff; prepended to ``sys.path`` when given.

    Returns:
        The ``(Unet, GaussianDiffusion)`` classes.
    """
    global _DGDIFF_IMPORT_CACHE
    if _DGDIFF_IMPORT_CACHE is not None:
        return _DGDIFF_IMPORT_CACHE
    if dgdiff_root is not None and str(dgdiff_root) not in sys.path:
        sys.path.insert(0, str(dgdiff_root))
    from functools import partial

    from denoising_diffusion_pytorch.denoising_diffusion_pytorch import (
        GaussianDiffusion as DGGaussianDiffusion,
    )
    from denoising_diffusion_pytorch.denoising_diffusion_pytorch import (
        ModelPrediction as DGModelPrediction,
    )
    from denoising_diffusion_pytorch.denoising_diffusion_pytorch import (
        Unet as DGUnet,
    )
    from denoising_diffusion_pytorch.denoising_diffusion_pytorch import (
        default as dg_default,
    )
    from denoising_diffusion_pytorch.denoising_diffusion_pytorch import (
        divisible_by as dg_divisible_by,
    )
    from denoising_diffusion_pytorch.denoising_diffusion_pytorch import (
        extract as dg_extract,
    )
    from einops import rearrange, reduce

    if not getattr(DGGaussianDiffusion, "_need_patched", False):

        def normalize_to_neg_one_to_one(img):
            return img * 2 - 1

        def _unet_forward(self, x, time, x_self_cond=None, cond=None):
            if not all(
                dg_divisible_by(d, self.downsample_factor) for d in x.shape[-2:]
            ):
                raise ValueError(
                    f"your input dimensions {x.shape[-2:]} need to be divisible by "
                    f"{self.downsample_factor}, given the unet"
                )
            if self.self_condition:
                x_self_cond = dg_default(x_self_cond, lambda: torch.zeros_like(x))
                x = torch.cat((x_self_cond, x), dim=1)

            x = self.init_conv(x)

            if self.use_cond:
                if cond is None:
                    raise ValueError("DGDiff conditional UNet requires cond")
                if cond.shape[-2:] != x.shape[-2:]:
                    cond = F.interpolate(
                        cond, size=x.shape[-2:], mode="bicubic", align_corners=False
                    )
                x = torch.cat((x, cond), dim=1)

            r = x.clone()
            t = self.time_mlp(time)
            h = []

            for block1, block2, attn, downsample in self.downs:
                x = block1(x, t)
                h.append(x)
                x = block2(x, t)
                x = attn(x) + x
                h.append(x)
                x = downsample(x)

            x = self.mid_block1(x, t)
            x = self.mid_attn(x) + x
            x = self.mid_block2(x, t)

            for block1, block2, attn, upsample in self.ups:
                x = torch.cat((x, h.pop()), dim=1)
                x = block1(x, t)
                x = torch.cat((x, h.pop()), dim=1)
                x = block2(x, t)
                x = attn(x) + x
                x = upsample(x)

            x = torch.cat((x, r), dim=1)
            x = self.final_res_block(x, t)
            return self.final_conv(x)

        def _model_predictions(
            self,
            x,
            t,
            x_self_cond=None,
            cond=None,
            clip_x_start=False,
            rederive_pred_noise=False,
        ):
            model_output = self.model(x, t, x_self_cond, cond=cond)
            maybe_clip = (
                partial(torch.clamp, min=-1.0, max=1.0)
                if clip_x_start
                else (lambda z: z)
            )

            if self.objective == "pred_noise":
                pred_noise = model_output
                x_start = maybe_clip(self.predict_start_from_noise(x, t, pred_noise))
                if clip_x_start and rederive_pred_noise:
                    pred_noise = self.predict_noise_from_start(x, t, x_start)
            elif self.objective == "pred_x0":
                x_start = maybe_clip(model_output)
                pred_noise = self.predict_noise_from_start(x, t, x_start)
            elif self.objective == "pred_v":
                x_start = maybe_clip(self.predict_start_from_v(x, t, model_output))
                pred_noise = self.predict_noise_from_start(x, t, x_start)
            else:
                raise ValueError(f"unknown objective {self.objective}")
            return DGModelPrediction(pred_noise, x_start)

        def _p_mean_variance(
            self, x, t, x_self_cond=None, cond=None, clip_denoised=True
        ):
            preds = self.model_predictions(x, t, x_self_cond=x_self_cond, cond=cond)
            x_start = preds.pred_x_start
            if clip_denoised:
                x_start.clamp_(-1.0, 1.0)
            model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_start, x_t=x, t=t
            )
            return model_mean, posterior_variance, posterior_log_variance, x_start

        def _p_sample_improved(
            self, x, cond, init_img, low_dose, t, s1=1.0, s2=1.0, x_self_cond=None
        ):
            b = x.shape[0]
            batched_times = torch.full((b,), t, device=self.device, dtype=torch.long)
            low_t = low_dose * dg_extract(
                self.sqrt_alphas_cumprod, batched_times, x.shape
            )
            x = s1 * x + (1 - s1) * low_t
            model_mean, _, model_log_variance, x_start = self.p_mean_variance(
                x=x,
                t=batched_times,
                x_self_cond=x_self_cond,
                cond=cond,
                clip_denoised=True,
            )
            noise = torch.randn_like(x) if t > 0 else 0.0
            new_mean = s2 * model_mean + (1 - s2) * init_img
            pred_img = new_mean + (0.5 * model_log_variance).exp() * noise
            return pred_img, x_start

        def _p_sample_loop_improved(
            self, shape, conds, init_imgs, low_doses, return_all_timesteps=False
        ):
            batch = shape[0]
            # Estimate the noise level of the SPDiff output to choose the step
            # the refinement resumes from and the guidance weight s1.
            residuals = init_imgs - low_doses
            est_stds = torch.abs(torch.std(residuals, dim=(1, 2, 3)))
            est_vars = est_stds**2
            diff = torch.abs(
                est_stds[:, None] - self.sqrt_one_minus_alphas_cumprod[None, :]
            )
            match_t = torch.argmin(diff, dim=1).long()
            resume_time = torch.div(match_t, 10, rounding_mode="floor").clamp(min=1)
            exp_term = (5000 * est_vars).exp()
            s1 = 0.7 * torch.div(exp_term, 10 + exp_term)
            s2 = 0.8

            imgs = self.q_sample(init_imgs, resume_time)
            outs = []
            for i in range(batch):
                img = imgs[i].unsqueeze(0)
                init_img = init_imgs[i].unsqueeze(0)
                low_dose = low_doses[i].unsqueeze(0)
                cond = conds[i].unsqueeze(0) if conds is not None else None
                x_start = None
                for t in reversed(range(0, int(resume_time[i].item()))):
                    self_cond = x_start if self.self_condition else None
                    img, x_start = self.p_sample_improved(
                        img, cond, init_img, low_dose, t, s1[i], s2, self_cond
                    )
                outs.append(img)
            return self.unnormalize(torch.cat(outs, dim=0))

        def _p_losses(
            self, x_start, t, cond=None, noise=None, offset_noise_strength=None
        ):
            noise = dg_default(noise, lambda: torch.randn_like(x_start))
            offset_noise_strength = dg_default(
                offset_noise_strength, self.offset_noise_strength
            )
            if offset_noise_strength > 0.0:
                offset_noise = torch.randn(x_start.shape[:2], device=self.device)
                noise += offset_noise_strength * rearrange(
                    offset_noise, "b c -> b c 1 1"
                )

            x = self.q_sample(x_start=x_start, t=t, noise=noise)
            model_out = self.model(x, t, None, cond=cond)

            if self.objective == "pred_noise":
                target = noise
            elif self.objective == "pred_x0":
                target = x_start
            elif self.objective == "pred_v":
                target = self.predict_v(x_start, t, noise)
            else:
                raise ValueError(f"unknown objective {self.objective}")

            loss = F.mse_loss(model_out, target, reduction="none")
            loss = reduce(loss, "b ... -> b", "mean")
            loss = loss * dg_extract(self.loss_weight, t, loss.shape)
            return loss.mean()

        def _forward(self, img, cond=None, *args, **kwargs):
            b, _, h, w = img.shape
            if h != self.image_size[0] or w != self.image_size[1]:
                raise ValueError(f"height and width of image must be {self.image_size}")
            t = torch.randint(0, self.num_timesteps, (b,), device=img.device).long()
            img = self.normalize(img)
            if cond is not None:
                cond = normalize_to_neg_one_to_one(cond) if cond.min() >= 0 else cond
            return self.p_losses(img, t, cond=cond, **kwargs)

        def _sample(
            self,
            batch_size=16,
            cond=None,
            init_img=None,
            low_dose=None,
            return_all_timesteps=False,
        ):
            (h, w), channels = self.image_size, self.channels
            shape = (batch_size, channels, h, w)
            if init_img is not None and low_dose is not None:
                return self.p_sample_loop_improved(
                    shape, cond, init_img, low_dose, return_all_timesteps
                )
            return self.p_sample_loop(shape, return_all_timesteps=return_all_timesteps)

        DGUnet.forward = _unet_forward
        DGGaussianDiffusion.model_predictions = _model_predictions
        DGGaussianDiffusion.p_mean_variance = _p_mean_variance
        DGGaussianDiffusion.p_sample_improved = _p_sample_improved
        DGGaussianDiffusion.p_sample_loop_improved = _p_sample_loop_improved
        DGGaussianDiffusion.p_losses = _p_losses
        DGGaussianDiffusion.forward = _forward
        DGGaussianDiffusion.sample = _sample
        DGGaussianDiffusion._need_patched = True

    _DGDIFF_IMPORT_CACHE = (DGUnet, DGGaussianDiffusion)
    return _DGDIFF_IMPORT_CACHE


def _resolve_dgdiff_ckpt(path: Path, stage: str) -> Path | None:
    """Return ``path`` if it exists, else the last matching ``.pt`` beside it."""
    if path.exists():
        return path
    patterns = (f"*{stage}*.pt", f"model-*{stage}*.pt", "model-*.pt")
    matches = sorted({str(p) for pat in patterns for p in path.parent.glob(pat)})
    return Path(matches[-1]) if matches else None


def _load_dgdiff_state(
    diffusion: nn.Module, ckpt_path: Path, device: torch.device
) -> None:
    """Load a DGDiff checkpoint, trying the layouts used by the official trainer.

    Raises:
        RuntimeError: If no candidate state dict can be loaded.
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    candidates = []
    if isinstance(ckpt, dict):
        candidates.extend(
            ckpt[key]
            for key in ("model", "state_dict")
            if isinstance(ckpt.get(key), dict)
        )
        if isinstance(ckpt.get("ema"), dict):
            ema = ckpt["ema"]
            candidates.extend(v for v in ema.values() if isinstance(v, dict))
            prefix = "ema_model."
            stripped = {
                k[len(prefix) :]: v
                for k, v in ema.items()
                if isinstance(k, str) and k.startswith(prefix)
            }
            if stripped:
                candidates.append(stripped)
    candidates.append(ckpt)

    last_err = None
    for state in candidates:
        if not isinstance(state, dict):
            continue
        for target in (diffusion, diffusion.model):
            try:
                target.load_state_dict(state, strict=False)
                return
            except Exception as exc:
                last_err = exc
    raise RuntimeError(f"could not load DGDiff checkpoint {ckpt_path}: {last_err}")


def _make_dgdiff_model(
    device: torch.device,
    dgdiff_root: Path | None,
    *,
    image_size: int,
    use_cond: bool,
    sampling_steps: int,
) -> nn.Module:
    """Create a DGDiff diffusion model in eval mode."""
    dg_unet, dg_gaussian_diffusion = _load_dgdiff_classes(dgdiff_root)
    model = dg_unet(
        channels=1, dim=64, dim_mults=(1, 2, 4, 8), flash_attn=False, use_cond=use_cond
    )
    diffusion = dg_gaussian_diffusion(
        model,
        image_size=image_size,
        timesteps=DGDIFF_TIMESTEPS,
        sampling_timesteps=sampling_steps,
        ddim_sampling_eta=0.0,
        sampler_=DGDIFF_SAMPLER,
    ).to(device)
    diffusion.eval()
    return diffusion


class DGDiffRefiner:
    """Two-stage DGDiff refinement of the SPDiff reconstruction.

    Stage 1 (optional) produces a 256x256 condition image; stage 2 refines the
    full-resolution image guided by it.
    """

    def __init__(self, stage1: nn.Module | None, stage2: nn.Module) -> None:
        """Store the two stages.

        Args:
            stage1: Low-resolution model, or ``None`` to use a bicubic
                downsampling of the SPDiff output as condition.
            stage2: Full-resolution conditional model.
        """
        self.stage1 = stage1
        self.stage2 = stage2

    @torch.no_grad()
    def refine(self, low_img: torch.Tensor, pre_img: torch.Tensor) -> torch.Tensor:
        """Refine ``pre_img`` (SPDiff output) given the LDCT slice ``low_img``.

        Args:
            low_img: LDCT middle slice ``(B, 1, H, W)`` in ``[0, 1]``.
            pre_img: SPDiff reconstruction ``(B, 1, H, W)`` in ``[0, 1]``.

        Returns:
            The refined image in ``[0, 1]``.
        """
        low_img = low_img.clamp(0.0, 1.0)
        pre_img = pre_img.clamp(0.0, 1.0)
        b = pre_img.shape[0]

        def resize(img: torch.Tensor, size: int | tuple) -> torch.Tensor:
            if isinstance(size, int):
                size = (size, size)
            return F.interpolate(img, size=size, mode="bicubic", align_corners=False)

        if self.stage1 is not None:
            low_256 = resize(low_img, DGDIFF_STAGE1_IMAGE_SIZE).clamp(0.0, 1.0)
            pre_256 = resize(pre_img, DGDIFF_STAGE1_IMAGE_SIZE).clamp(0.0, 1.0)
            cond = self.stage1.sample(
                batch_size=b, init_img=pre_256 * 2 - 1, low_dose=low_256 * 2 - 1
            )
        else:
            cond = resize(pre_img, DGDIFF_COND_SIZE).clamp(0.0, 1.0)
        cond = resize(cond * 2 - 1, pre_img.shape[-2:])
        out = self.stage2.sample(
            batch_size=b, cond=cond, init_img=pre_img * 2 - 1, low_dose=low_img * 2 - 1
        )
        return out.clamp(0.0, 1.0)


def build_dgdiff_refiner(
    cfg: RunConfig, device: torch.device, dgdiff_root: Path | None, enabled: bool
) -> DGDiffRefiner | None:
    """Load the DGDiff refiner when its stage-2 checkpoint exists.

    Args:
        cfg: Run configuration (checkpoints are looked up in ``output_dir``).
        device: Inference device.
        dgdiff_root: Folder of the DGDiff package.
        enabled: Whether to use DGDiff at all.

    Returns:
        The refiner, or ``None`` to use SPDiff only.

    Raises:
        FileNotFoundError: If a required checkpoint is missing.
    """
    if not enabled:
        return None
    stage2_default = cfg.output_dir / DGDIFF_STAGE2_CKPT_NAME
    stage2_path = _resolve_dgdiff_ckpt(stage2_default, "stage2")
    if not stage2_path:
        msg = (
            f"DGDiff stage-2 checkpoint not found ({stage2_default}); using SPDiff only"
        )
        if DGDIFF_REQUIRED:
            raise FileNotFoundError(msg)
        logger.info(msg)
        return None

    stage2 = _make_dgdiff_model(
        device,
        dgdiff_root,
        image_size=DGDIFF_STAGE2_IMAGE_SIZE,
        use_cond=True,
        sampling_steps=DGDIFF_STAGE2_SAMPLING_STEPS,
    )
    _load_dgdiff_state(stage2, stage2_path, device)
    logger.info("DGDiff stage 2 loaded: %s", stage2_path)

    stage1 = None
    stage1_default = cfg.output_dir / DGDIFF_STAGE1_CKPT_NAME
    stage1_path = _resolve_dgdiff_ckpt(stage1_default, "stage1")
    if stage1_path:
        stage1 = _make_dgdiff_model(
            device,
            dgdiff_root,
            image_size=DGDIFF_STAGE1_IMAGE_SIZE,
            use_cond=False,
            sampling_steps=DGDIFF_TIMESTEPS,
        )
        _load_dgdiff_state(stage1, stage1_path, device)
        logger.info("DGDiff stage 1 loaded: %s", stage1_path)
    elif DGDIFF_REQUIRE_STAGE1:
        raise FileNotFoundError(
            f"DGDiff stage-1 checkpoint not found ({stage1_default})"
        )
    else:
        logger.info("DGDiff stage 1 missing; conditioning stage 2 on bicubic pre_img")
    return DGDiffRefiner(stage1, stage2)


# ============================================================================
# Training and testing
# ============================================================================
def predict_image(
    model: Diffusion,
    x: torch.Tensor,
    radon: img2sino.Radon,
    match_t: int,
    dgdiff_refiner: DGDiffRefiner | None = None,
) -> torch.Tensor:
    """Denoise LDCT image triples: project, sample, reconstruct, refine.

    Args:
        model: SPDiff diffusion model.
        x: Normalised LDCT triples ``(B, 3, 512, 512)``.
        radon: Projector.
        match_t: Dose-matched step; sampling starts at ``match_t + 1``.
        dgdiff_refiner: Optional DGDiff refiner.

    Returns:
        The normalised middle-slice prediction ``(B, 1, 512, 512)``.
    """
    x_sino = img_to_sino(x, radon)
    pred_sino, _, _ = model.sample(batch_size=x.shape[0], img=x_sino, t=match_t + 1)
    pred_img = sino_to_img(pred_sino, radon)
    if dgdiff_refiner is not None:
        pred_img = dgdiff_refiner.refine(x[:, 1].unsqueeze(1), pred_img)
    return pred_img


@torch.no_grad()
def validate(
    model: Diffusion,
    loader: DataLoader | None,
    device: torch.device,
    radon: img2sino.Radon,
    dose: str,
    dgdiff_refiner: DGDiffRefiner | None = None,
) -> tuple[float, float, float]:
    """Mean PSNR/SSIM/RMSE (HU) of the prediction on the validation split.

    Args:
        model: Model to evaluate (the EMA model when available).
        loader: Validation loader with batch size 1, or ``None``.
        device: Inference device.
        radon: Projector.
        dose: Dose label (selects the starting step).
        dgdiff_refiner: Optional DGDiff refiner.

    Returns:
        ``(psnr, ssim, rmse)``, or zeros without a loader.
    """
    model.eval()
    match_t = match_t_for_dose(model, dose)
    if loader is None:
        return 0.0, 0.0, 0.0
    psnr_sum, ssim_sum, rmse_sum, n = 0.0, 0.0, 0.0, 0
    for j, (xb, yb) in enumerate(loader):
        if VAL_MAX_SLICES and j >= VAL_MAX_SLICES:
            break
        xb = xb.float().to(device)
        yb = yb.float().to(device)
        pred_img = predict_image(model, xb, radon, match_t, dgdiff_refiner)
        yv = to_hu_window(yb[0, 1])
        pv = to_hu_window(pred_img)
        psnr_sum += compute_psnr(pv, yv, DATA_RANGE)
        ssim_sum += compute_ssim(pv, yv, DATA_RANGE)
        rmse_sum += compute_rmse(pv, yv)
        n += 1
    if n == 0:
        return 0.0, 0.0, 0.0
    return psnr_sum / n, ssim_sum / n, rmse_sum / n


def save_snapshot(
    path: Path,
    diffusion: Diffusion,
    opt: optim.Optimizer,
    ema_model: Diffusion | None,
    **progress: Any,
) -> None:
    """Save a training snapshot tagged with the sinogram domain.

    Args:
        path: Destination file.
        diffusion: Trained model.
        opt: Optimizer.
        ema_model: EMA model, stored under ``"ema"`` when given.
        **progress: ``epoch``, ``step``, ``lr``, ``loss``, ``psnr``, ``ssim``
            and ``rmse_hu``.
    """
    state = {
        "model": diffusion.state_dict(),
        "optimizer": opt.state_dict(),
        "epoch": int(progress["epoch"]),
        "step": int(progress["step"]),
        "lr": float(progress["lr"]),
        "loss": float(progress["loss"]),
        "psnr": float(progress["psnr"]),
        "ssim": float(progress["ssim"]),
        "rmse_hu": float(progress["rmse_hu"]),
        "domain": CKPT_DOMAIN,
    }
    if ema_model is not None:
        state["ema"] = ema_model.state_dict()
    torch.save(state, path)


def train(
    diffusion: Diffusion,
    loader: DataLoader,
    device: torch.device,
    cfg: RunConfig,
    radon: img2sino.Radon,
    ema_model: Diffusion | None = None,
    val_loader: DataLoader | None = None,
    dgdiff_refiner: DGDiffRefiner | None = None,
) -> None:
    """Train SPDiff with L1 loss in the sinogram domain for ``MAX_ITERS`` steps.

    Resumes from the last checkpoint when it is a sinogram-domain checkpoint.
    Every ``SAVE_ITERS`` iterations the (EMA) model is validated and the last
    and best checkpoints are written.

    Args:
        diffusion: SPDiff model.
        loader: Training loader yielding LD/FD image triples.
        device: Training device.
        cfg: Run configuration.
        radon: Projector.
        ema_model: EMA copy of the model, or ``None``.
        val_loader: Validation loader.
        dgdiff_refiner: Optional DGDiff refiner used during validation.
    """
    diffusion.train()
    opt = optim.Adam(diffusion.parameters(), lr=LR)

    ckpt = None
    if cfg.last_ckpt.exists():
        if torch.load(cfg.last_ckpt, map_location="cpu").get("domain") != CKPT_DOMAIN:
            logger.warning(
                "last.pt is not a sinogram-domain checkpoint; ignoring it. "
                "Move or delete the file to silence this warning."
            )
        else:
            extra = {"ema": ema_model} if ema_model is not None else None
            ckpt = load_checkpoint(
                cfg.last_ckpt, model=diffusion, optimizer=opt, extra_models=extra
            )
    start_epoch = (ckpt["epoch"] + 1) if ckpt else 1
    step = ckpt["step"] if ckpt else 0
    best_psnr = best_psnr_so_far(cfg.best_ckpt)
    losses = []
    ema = SimpleEMA(EMA_DECAY) if ema_model is not None else None
    if ema_model is not None and ckpt is None:
        ema_model.load_state_dict(diffusion.state_dict())
    t0 = time.time()

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        for _, y_fd in loader:
            step += 1
            y_fd = y_fd.float().to(device)
            y_sino = img_to_sino(y_fd, radon)
            x_recon, _ = diffusion(y_sino)
            loss = F.l1_loss(x_recon, y_sino[:, 1].unsqueeze(1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

            if ema is not None and step % EMA_UPDATE_EVERY == 0:
                if step < EMA_START_ITER:
                    ema_model.load_state_dict(diffusion.state_dict())
                else:
                    ema.update(ema_model, diffusion)

            if step % PRINT_ITERS == 0:
                # The running mean shows the trend under the per-batch noise of
                # the random diffusion step.
                logger.info(
                    "epoch %d/%d  iter %d  loss %.6f  (mean %.6f)  lr %.2e  (%.0fs)",
                    epoch,
                    NUM_EPOCHS,
                    step,
                    loss.item(),
                    float(np.mean(losses[-PRINT_ITERS:])),
                    opt.param_groups[0]["lr"],
                    time.time() - t0,
                )

            if step % SAVE_ITERS == 0:
                # The test split is reserved for the final test.
                eval_model = ema_model if ema_model is not None else diffusion
                psnr, ssim, rmse = validate(
                    eval_model, val_loader, device, radon, cfg.dose, dgdiff_refiner
                )
                diffusion.train()
                progress = dict(
                    epoch=epoch,
                    step=step,
                    lr=opt.param_groups[0]["lr"],
                    loss=losses[-1] if losses else 0.0,
                    psnr=psnr,
                    ssim=ssim,
                    rmse_hu=rmse,
                )
                save_snapshot(cfg.last_ckpt, diffusion, opt, ema_model, **progress)
                if psnr > best_psnr:
                    best_psnr = psnr
                    save_snapshot(cfg.best_ckpt, diffusion, opt, ema_model, **progress)
                np.save(cfg.losses_path, np.array(losses))
                logger.info(
                    "  snapshot (epoch=%d step=%d psnr=%.3f ssim=%.4f rmse_hu=%.3f) "
                    "best_psnr=%.3f",
                    epoch,
                    step,
                    psnr,
                    ssim,
                    rmse,
                    best_psnr,
                )
            if step >= MAX_ITERS:
                return


def test(
    diffusion: Diffusion,
    loader: DataLoader,
    device: torch.device,
    cfg: RunConfig,
    radon: img2sino.Radon,
    dgdiff_refiner: DGDiffRefiner | None = None,
) -> None:
    """Evaluate on the full test split and write metrics and figures.

    Metrics use the common HU window ``[-160, 240]`` (not the
    ``[-1000, 1000]`` window of the paper) so that NEED is comparable with the
    other models.

    Args:
        diffusion: Model to test (the EMA model when available).
        loader: Test loader with batch size 1.
        device: Inference device.
        cfg: Run configuration.
        radon: Projector.
        dgdiff_refiner: Optional DGDiff refiner.
    """
    diffusion.eval()
    match_t = match_t_for_dose(diffusion, cfg.dose)
    logger.info(
        "[test] dose=%s ld_I0=%.3e -> match_t=%d (sample t=%d/%d)",
        cfg.dose,
        I0 / DOSE_FACTOR.get(cfg.dose, DEFAULT_DOSE_FACTOR),
        match_t,
        match_t + 1,
        TIMESTEPS,
    )
    abdomen_idx = abdomen_fig_index(loader, cfg.abdomen_idx, cfg.abdomen_frac)
    fig_kwargs = dict(
        fig_dir=cfg.fig_dir,
        dose=cfg.dose,
        vmin=DISPLAY_TRUNC_MIN,
        vmax=DISPLAY_TRUNC_MAX,
    )
    input_sum, pred_sum, n = [0.0] * 3, [0.0] * 3, 0
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.float().to(device)
            y = y.float().to(device)
            pred_img = predict_image(diffusion, x, radon, match_t, dgdiff_refiner)
            xv, yv, pv = (
                to_hu_window(x[0, 1]),
                to_hu_window(y[0, 1]),
                to_hu_window(pred_img),
            )
            o, p = compute_measure(xv, yv, pv, DATA_RANGE)
            for k in range(3):
                input_sum[k] += o[k]
                pred_sum[k] += p[k]
            n += 1
            if i < NUM_TEST_FIGURES:
                save_fig(xv, yv, pv, i, o, p, **fig_kwargs)
            if i == abdomen_idx:
                save_fig(xv, yv, pv, i, o, p, name=ABDOMEN_FIG_NAME, **fig_kwargs)
    write_test_metrics(cfg, [v / n for v in input_sum], [v / n for v in pred_sum], n)


def select_test_model(
    cfg: RunConfig,
    diffusion: Diffusion,
    ema_model: Diffusion | None,
    device: torch.device,
    test_only: bool,
) -> Diffusion:
    """Load the best (else last) checkpoint for testing, preferring EMA weights.

    Args:
        cfg: Run configuration.
        diffusion: Model.
        ema_model: EMA model, or ``None``.
        device: Device the checkpoint is loaded onto.
        test_only: Whether training was skipped (weights in memory are random).

    Returns:
        The model to test.

    Raises:
        FileNotFoundError: In test-only mode without any checkpoint.
    """
    ckpt_path = next((p for p in (cfg.best_ckpt, cfg.last_ckpt) if p.exists()), None)
    if ckpt_path is None:
        if test_only:
            raise FileNotFoundError(
                f"--test-only given but no best.pt or last.pt in {cfg.output_dir}"
            )
        logger.info("No best.pt/last.pt; testing the last-iteration weights")
        return ema_model if ema_model is not None else diffusion

    ckpt = torch.load(ckpt_path, map_location=device)
    if ema_model is not None and ckpt.get("ema") is not None:
        ema_model.load_state_dict(ckpt["ema"])
        test_model, tag = ema_model, "EMA"
    else:
        diffusion.load_state_dict(ckpt["model"])
        test_model, tag = diffusion, "model"
    logger.info(
        "Testing %s %s (epoch=%s psnr=%.3f)",
        ckpt_path.name,
        tag,
        ckpt.get("epoch"),
        float(ckpt.get("psnr", float("nan"))),
    )
    return test_model


def build_parser() -> argparse.ArgumentParser:
    """Command-line parser: common options plus NEED-specific ones."""
    parser = build_arg_parser(MODEL_NAME, description=__doc__)
    parser.add_argument(
        "--test-only", action="store_true", help="Skip training; test saved weights."
    )
    parser.add_argument(
        "--fd-cache-dir",
        type=Path,
        default=DEFAULT_FD_CACHE_DIR,
        help="Cache folder for decoded full-dose DICOM targets.",
    )
    parser.add_argument(
        "--no-fd-cache", action="store_true", help="Decode DICOM targets every time."
    )
    parser.add_argument(
        "--dgdiff-root",
        type=Path,
        default=None,
        help="Folder of NEED-main/DGDiff (contains denoising_diffusion_pytorch).",
    )
    parser.add_argument(
        "--no-dgdiff", action="store_true", help="Never use the DGDiff refinement."
    )
    parser.add_argument(
        "--radon-backend",
        choices=img2sino.BACKENDS,
        default="fanbeam",
        help="Native projector used when torch_radon is not installed.",
    )
    parser.add_argument(
        "--require-torch-radon",
        action="store_true",
        help="Fail instead of using a native projector without torch_radon.",
    )
    parser.add_argument(
        "--ray-samples",
        type=int,
        default=img2sino.DEFAULT_RAY_SAMPLES,
        help="Ray samples of the native fan-beam projector.",
    )
    parser.add_argument(
        "--angle-chunk",
        type=int,
        default=img2sino.DEFAULT_ANGLE_CHUNK,
        help="Views per step of the native fan-beam projector.",
    )
    return parser


def make_loaders(
    cfg: RunConfig, device: torch.device, fd_cache_dir: Path | None
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build the training, test and validation loaders.

    Without a validation split, the test loader is used for validation.
    """

    def dataset(split: str) -> ULDCTDatasetContext:
        return ULDCTDatasetContext(split, cfg.data_root, fd_cache_dir)

    train_ds = dataset(TRAIN_SPLIT)
    test_ds = dataset(TEST_SPLIT)
    val_ds = optional_split(lambda: dataset(VAL_SPLIT), VAL_SPLIT)
    logger.info(
        "train files: %d | val files: %d | test files: %d",
        len(train_ds),
        len(val_ds) if val_ds is not None else 0,
        len(test_ds),
    )
    # Persistent workers and prefetching keep the GPU fed across short epochs.
    workers = cfg.num_workers
    common = dict(
        num_workers=workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(workers > 0),
    )
    prefetch = {"prefetch_factor": PREFETCH_FACTOR} if workers > 0 else {}
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, **common, **prefetch
    )
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, **common)
    val_loader = (
        DataLoader(val_ds, batch_size=1, shuffle=False, **common)
        if val_ds is not None
        else test_loader
    )
    return train_loader, test_loader, val_loader


def main() -> None:
    """Train SPDiff, then test the best checkpoint (with DGDiff if available)."""
    cfg, args = parse_run_config(build_parser(), MODEL_NAME)
    torch.backends.cudnn.benchmark = True
    # TF32 on Ampere GPUs: faster matmul/conv at near-fp32 precision.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = get_device()
    logger.info(
        "[%s] dose=%s device=%s out=%s", MODEL_NAME, cfg.dose, device, cfg.output_dir
    )

    fd_cache_dir = None if args.no_fd_cache else args.fd_cache_dir
    train_loader, test_loader, val_loader = make_loaders(cfg, device, fd_cache_dir)
    radon = img2sino.build_radon(
        backend=args.radon_backend,
        require_torch_radon=args.require_torch_radon,
        ray_samples=args.ray_samples,
        angle_chunk=args.angle_chunk,
    )

    lambda_sched = make_lambda_schedule(TIMESTEPS).to(device)
    diffusion = build_diffusion(lambda_sched, device)
    ema_model = build_diffusion(lambda_sched, device) if USE_EMA else None
    dgdiff_refiner = build_dgdiff_refiner(
        cfg, device, args.dgdiff_root, enabled=USE_DGDIFF and not args.no_dgdiff
    )
    if args.test_only:
        logger.info("--test-only: skipping training")
    else:
        train(
            diffusion,
            train_loader,
            device,
            cfg,
            radon,
            ema_model=ema_model,
            val_loader=val_loader,
            dgdiff_refiner=dgdiff_refiner,
        )

    test_model = select_test_model(cfg, diffusion, ema_model, device, args.test_only)
    test(test_model, test_loader, device, cfg, radon, dgdiff_refiner=dgdiff_refiner)


if __name__ == "__main__":
    main()
