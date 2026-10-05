"""CoreDiff (Gao et al., MICCAI 2023) trained and evaluated on the ULDCT dataset.

Contextual error-modulated generalised diffusion: a cold-diffusion model whose
degradation operator mixes the full-dose image with the LDCT image, denoised by
a time-conditioned U-Net with Error-Modulated Modules (EMM) that see the
adjacent slices. Hyperparameters follow the official repository
(``train_mayo2016.sh``, ``corediff.py`` and ``basic_template.py``).

The network and diffusion code builds on
https://github.com/arpitbansal297/Cold-Diffusion-Models.

Usage:
    python corediff.py --dose 5pct --data-root /path/to/uldct_5pct/dataset
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from common.checkpoint import best_psnr_so_far, load_checkpoint, save_checkpoint
from common.config import (
    DATA_RANGE,
    LOW_DOSE_DIR,
    NORM_MAX,
    NORM_MIN,
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
    load_target_array,
    random_patch_pairs,
    to_hu_window,
)
from common.evaluation import (
    ABDOMEN_FIG_NAME,
    NUM_TEST_FIGURES,
    abdomen_fig_index,
    save_fig,
    write_test_metrics,
)
from common.metrics import compute_measure, compute_mse, compute_ssim
from common.runtime import optional_split

try:
    import pydicom
except ImportError:
    pydicom = None

MODEL_NAME = "corediff"

# Hyperparameters of the official CoreDiff repository:
#   train_mayo2016.sh: --batch_size 4, --max_iter 150000, --context,
#                      --only_adjust_two_step, --save_freq 2500
#   corediff.py:       init_lr=2e-4, T=10, update_ema_iter=10,
#                      start_ema_iter=2000, ema_decay=0.995, start_adjust_iter=1
#   basic_template.py: image_size=512 (training uses whole 512x512 slices)
NUM_EPOCHS = 10_000  # upper bound; training stops at MAX_ITERS
MAX_ITERS = 150_000
BATCH_SIZE = 4
IMG_SIZE = 512
PATCH_SIZE = IMG_SIZE
PATCH_N = 1
LR = 2e-4
TIMESTEPS = 10
PRINT_ITERS = 50
SAVE_ITERS = 2500
START_ADJUST_ITER = 1
ONLY_ADJUST_TWO_STEP = True
USE_CONTEXT = True  # three consecutive slices (above, current, below) as input

EMA_DECAY = 0.995
EMA_UPDATE_EVERY = 10
EMA_START_ITER = 2000

TIME_EMB_DIM = 32
TIME_EMB_MAX_PERIOD = 10000
MAX_ALPHA_CUMPROD = 0.999

# CoreDiff-native HU windows (utils/measure.py and basic_template.transfer_*):
# CALC_* for the training/validation metrics (HU -> clip -> [0, 255]),
# DISP_* for figures only (soft-tissue window).
CALC_CUT_MIN, CALC_CUT_MAX = -1000.0, 1000.0
DISP_CUT_MIN, DISP_CUT_MAX = -100.0, 200.0
CALC_SCALE = 255.0
RMSE_UNIT = "HU"

DEFAULT_FD_CACHE_DIR = Path("fd_npy_cache")
FD_CACHE_LOG_EVERY = 200

logger = logging.getLogger(MODEL_NAME)


# ============================================================================
# Denoising network (corediff_wrapper.py)
# ============================================================================
class SinusoidalPosEmb(nn.Module):
    """Sinusoidal timestep embedding."""

    def __init__(self, dim: int) -> None:
        """Store the embedding size.

        Args:
            dim: Embedding dimension (sine and cosine halves).
        """
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed a ``(B,)`` tensor of timesteps into ``(B, dim)``."""
        half_dim = self.dim // 2
        emb = math.log(TIME_EMB_MAX_PERIOD) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class single_conv(nn.Module):
    """3x3 convolution followed by ReLU."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        """Build the layer.

        Args:
            in_ch: Input channels.
            out_ch: Output channels.
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the convolution."""
        return self.conv(x)


class up(nn.Module):
    """2x transposed-conv upsampling with an additive skip connection."""

    def __init__(self, in_ch: int) -> None:
        """Build the layer.

        Args:
            in_ch: Input channels; the output has ``in_ch // 2``.
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
        """Build the layer.

        Args:
            in_ch: Input channels.
            out_ch: Output channels.
        """
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the convolution."""
        return self.conv(x)


class adjust_net(nn.Module):
    """Error-Modulated Module: predicts per-channel scale and shift.

    Takes the previous estimate and the LDCT image (2 channels) and returns
    ``(gamma, beta)`` of shape ``(B, out_channels, 1, 1)`` each.
    """

    def __init__(self, out_channels: int = 64, middle_channels: int = 32) -> None:
        """Build the layers.

        Args:
            out_channels: Channels of the modulated feature map.
            middle_channels: Width of the first hidden layer.
        """
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(2, middle_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
            nn.Conv2d(middle_channels, middle_channels * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
            nn.Conv2d(middle_channels * 2, middle_channels * 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
            nn.Conv2d(middle_channels * 4, out_channels * 2, 1, padding=0),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the ``(gamma, beta)`` modulation of a ``(B, 2, H, W)`` input."""
        out = self.model(x)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        half = out.shape[1] // 2
        return out[:, :half], out[:, half:]


class UNet(nn.Module):
    """Time-conditioned U-Net with EMM modulation at every scale.

    Architecture from CBDNet ("Toward Convolutional Blind Denoising of Real
    Photographs", https://github.com/GuoShi28/CBDNet), with timestep embedding
    and EMM modules added and the noise-estimation network removed.
    """

    def __init__(self, in_channels: int = 2, out_channels: int = 1) -> None:
        """Build the layers.

        Args:
            in_channels: Input channels (3 with slice context).
            out_channels: Output channels.
        """
        super().__init__()
        dim = TIME_EMB_DIM
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

        self.inc = nn.Sequential(single_conv(in_channels, 64), single_conv(64, 64))

        self.down1 = nn.AvgPool2d(2)
        self.mlp1 = nn.Sequential(nn.GELU(), nn.Linear(dim, 64))
        self.adjust1 = adjust_net(64)
        self.conv1 = nn.Sequential(
            single_conv(64, 128), single_conv(128, 128), single_conv(128, 128)
        )

        self.down2 = nn.AvgPool2d(2)
        self.mlp2 = nn.Sequential(nn.GELU(), nn.Linear(dim, 128))
        self.adjust2 = adjust_net(128)
        self.conv2 = nn.Sequential(
            single_conv(128, 256), *(single_conv(256, 256) for _ in range(5))
        )

        self.up1 = up(256)
        self.mlp3 = nn.Sequential(nn.GELU(), nn.Linear(dim, 128))
        self.adjust3 = adjust_net(128)
        self.conv3 = nn.Sequential(
            single_conv(128, 128), single_conv(128, 128), single_conv(128, 128)
        )

        self.up2 = up(128)
        self.mlp4 = nn.Sequential(nn.GELU(), nn.Linear(dim, 64))
        self.adjust4 = adjust_net(64)
        self.conv4 = nn.Sequential(single_conv(64, 64), single_conv(64, 64))

        self.outc = outconv(64, out_channels)

    @staticmethod
    def _modulate(
        h: torch.Tensor,
        condition: torch.Tensor,
        adjust_fn: adjust_net,
        x_adjust: torch.Tensor,
        adjust: bool,
    ) -> torch.Tensor:
        """Add the time condition, scaled and shifted by the EMM if ``adjust``."""
        condition = condition[:, :, None, None]
        if adjust:
            gamma, beta = adjust_fn(x_adjust)
            return h + gamma * condition + beta
        return h + condition

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        x_adjust: torch.Tensor,
        adjust: bool,
    ) -> torch.Tensor:
        """Predict the residual of the current estimate.

        Args:
            x: Network input ``(B, C, H, W)``.
            t: Timesteps ``(B,)``.
            x_adjust: EMM input ``(B, 2, H, W)``.
            adjust: Whether the EMM modulation is applied.
        """
        inx = self.inc(x)
        time_emb = self.time_mlp(t)

        down1 = self._modulate(
            self.down1(inx), self.mlp1(time_emb), self.adjust1, x_adjust, adjust
        )
        conv1 = self.conv1(down1)

        down2 = self._modulate(
            self.down2(conv1), self.mlp2(time_emb), self.adjust2, x_adjust, adjust
        )
        conv2 = self.conv2(down2)

        up1 = self._modulate(
            self.up1(conv2, conv1), self.mlp3(time_emb), self.adjust3, x_adjust, adjust
        )
        conv3 = self.conv3(up1)

        up2 = self._modulate(
            self.up2(conv3, inx), self.mlp4(time_emb), self.adjust4, x_adjust, adjust
        )
        conv4 = self.conv4(up2)

        return self.outc(conv4)


class Network(nn.Module):
    """Denoiser: U-Net residual added to the centre slice of the input."""

    def __init__(
        self, in_channels: int = 3, out_channels: int = 1, context: bool = True
    ) -> None:
        """Build the U-Net.

        Args:
            in_channels: Input channels (3 with slice context).
            out_channels: Output channels.
            context: Whether the input stacks three adjacent slices.
        """
        super().__init__()
        self.unet = UNet(in_channels=in_channels, out_channels=out_channels)
        self.context = context

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        x_end: torch.Tensor,
        adjust: bool = True,
    ) -> torch.Tensor:
        """Estimate the full-dose image.

        Args:
            x: Degraded input (``(B, 3, H, W)`` with context).
            t: Timesteps ``(B,)``.
            y: Previous full-dose estimate, fed to the EMM.
            x_end: LDCT image (end point of the degradation), fed to the EMM.
            adjust: Whether the EMM modulation is applied.
        """
        x_middle = x[:, 1].unsqueeze(1) if self.context else x
        x_adjust = torch.cat((y, x_end), dim=1)
        return self.unet(x, t, x_adjust, adjust=adjust) + x_middle


# ============================================================================
# Generalised diffusion (diffusion_modules.py)
# ============================================================================
def extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    """Gather ``a[t]`` and reshape it to broadcast over ``x_shape``."""
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def linear_alpha_schedule(timesteps: int) -> torch.Tensor:
    """Linearly decreasing cumulative-alpha schedule, clipped to ``[0, 0.999]``."""
    alphas_cumprod = 1 - torch.linspace(0, timesteps, timesteps) / timesteps
    return torch.clip(alphas_cumprod, 0, MAX_ALPHA_CUMPROD)


class Diffusion(nn.Module):
    """Cold diffusion with a mean-preserving LDCT degradation operator."""

    def __init__(
        self,
        denoise_fn: nn.Module | None = None,
        image_size: int = 512,
        channels: int = 1,
        timesteps: int = 10,
        context: bool = True,
    ) -> None:
        """Register the schedule buffers.

        Args:
            denoise_fn: Denoising network (:class:`Network`).
            image_size: Expected training image side.
            channels: Image channels.
            timesteps: Number of diffusion steps ``T``.
            context: Whether inputs stack three adjacent slices.
        """
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.num_timesteps = int(timesteps)
        self.context = context

        alphas_cumprod = linear_alpha_schedule(timesteps)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("one_minus_alphas_cumprod", 1.0 - alphas_cumprod)

    def q_sample(
        self, x_start: torch.Tensor, x_end: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Mean-preserving degradation: mix ``x_start`` towards ``x_end``."""
        return (
            extract(self.alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.one_minus_alphas_cumprod, t, x_start.shape) * x_end
        )

    def get_x2_bar_from_xt(
        self, x1_bar: torch.Tensor, xt: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Recover the degradation end point from ``x_t`` and the estimate."""
        return (xt - extract(self.alphas_cumprod, t, x1_bar.shape) * x1_bar) / extract(
            self.one_minus_alphas_cumprod, t, x1_bar.shape
        )

    @torch.no_grad()
    def sample(
        self,
        batch_size: int = 4,
        img: torch.Tensor | None = None,
        t: int | None = None,
        n_iter: int = 1,
        start_adjust_iter: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Restore an LDCT image with the DDIM-style sampling routine.

        Args:
            batch_size: Batch size of ``img``.
            img: LDCT input (``(B, 3, H, W)`` with context).
            t: Starting timestep; defaults to ``T``.
            n_iter: Training iteration, compared to ``start_adjust_iter``.
            start_adjust_iter: Iteration from which the EMM is enabled.

        Returns:
            The restored image clipped to ``[0, 1]``, and the stacked direct
            reconstructions and intermediate images of every step.
        """
        self.denoise_fn.eval()
        if t is None:
            t = self.num_timesteps

        if self.context:
            up_img = img[:, 0].unsqueeze(1)
            down_img = img[:, 2].unsqueeze(1)
            img = img[:, 1].unsqueeze(1)

        noise = img
        x1_bar = img
        direct_recons = []
        imstep_imgs = []

        while t:
            step = torch.full((batch_size,), t - 1, dtype=torch.long, device=img.device)
            full_img = (
                torch.cat((up_img, img, down_img), dim=1) if self.context else img
            )
            adjust = not (t == self.num_timesteps or n_iter < start_adjust_iter)

            x1_bar = self.denoise_fn(full_img, step, x1_bar, noise, adjust=adjust)
            x2_bar = self.get_x2_bar_from_xt(x1_bar, img, step)

            xt_bar = x1_bar
            if t != 0:
                xt_bar = self.q_sample(x_start=xt_bar, x_end=x2_bar, t=step)

            xt_sub1_bar = x1_bar
            if t - 1 != 0:
                step2 = torch.full(
                    (batch_size,), t - 2, dtype=torch.long, device=img.device
                )
                xt_sub1_bar = self.q_sample(x_start=xt_sub1_bar, x_end=x2_bar, t=step2)

            img = img - xt_bar + xt_sub1_bar

            direct_recons.append(x1_bar)
            imstep_imgs.append(img)
            t = t - 1

        return img.clamp(0.0, 1.0), torch.stack(direct_recons), torch.stack(imstep_imgs)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        n_iter: int,
        only_adjust_two_step: bool = False,
        start_adjust_iter: int = 1,
        t: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Two-stage training forward pass.

        Args:
            x: LDCT input (``(B, 3, H, W)`` with context).
            y: Full-dose target ``(B, 1, H, W)``.
            n_iter: Training iteration.
            only_adjust_two_step: Use the EMM only in the second stage.
            start_adjust_iter: Iteration from which the EMM is trained.
            t: Batch timesteps. When ``None``, one timestep is drawn for the
                whole batch, as in the paper. Drawing it outside makes
                ``DataParallel`` replicas share the same ``t``, so single- and
                multi-GPU training are equivalent.

        Returns:
            ``(x_recon, x_mix, x_recon_sub1, x_mix_sub1)``: the stage-I estimate
            and its input, and the stage-II estimate and its input.
        """
        b, c, h, w = y.shape
        device, img_size = y.device, self.image_size
        assert (
            h == img_size and w == img_size
        ), f"height and width of image must be {img_size}"

        if t is None:
            t_single = torch.randint(0, self.num_timesteps, (1,), device=device).long()
            t = t_single.repeat((b,))
        else:
            t = t.to(device=device).long()
            t_single = t[:1]

        if self.context:
            x_end = x[:, 1].unsqueeze(1)
            x_mix = self.q_sample(x_start=y, x_end=x_end, t=t)
            x_mix = torch.cat(
                (x[:, 0].unsqueeze(1), x_mix, x[:, 2].unsqueeze(1)), dim=1
            )
        else:
            x_end = x
            x_mix = self.q_sample(x_start=y, x_end=x_end, t=t)

        # Stage I
        if only_adjust_two_step or n_iter < start_adjust_iter:
            x_recon = self.denoise_fn(x_mix, t, y, x_end, adjust=False)
        else:
            adjust = bool(t[0] != self.num_timesteps - 1)
            x_recon = self.denoise_fn(x_mix, t, y, x_end, adjust=adjust)

        # Stage II
        if n_iter >= start_adjust_iter and t_single.item() >= 1:
            t_sub1 = t - 1
            t_sub1[t_sub1 < 0] = 0
            x_mix_sub1 = self.q_sample(x_start=x_recon, x_end=x_end, t=t_sub1)
            if self.context:
                x_mix_sub1 = torch.cat(
                    (x[:, 0].unsqueeze(1), x_mix_sub1, x[:, 2].unsqueeze(1)), dim=1
                )
            x_recon_sub1 = self.denoise_fn(
                x_mix_sub1, t_sub1, x_recon, x_end, adjust=True
            )
        else:
            x_recon_sub1, x_mix_sub1 = x_recon, x_mix

        return x_recon, x_mix, x_recon_sub1, x_mix_sub1


def build_model() -> Diffusion:
    """Build the CoreDiff diffusion model with the configured hyperparameters."""
    in_ch = 3 if USE_CONTEXT else 1
    denoise_fn = Network(in_channels=in_ch, out_channels=1, context=USE_CONTEXT)
    return Diffusion(
        denoise_fn=denoise_fn,
        image_size=IMG_SIZE,
        channels=1,
        timesteps=TIMESTEPS,
        context=USE_CONTEXT,
    )


class EMA:
    """Exponential moving average of model weights (CoreDiff ``utils/ema.py``)."""

    def __init__(self, beta: float) -> None:
        """Store the decay.

        Args:
            beta: Weight of the previous average.
        """
        self.beta = beta

    def update_model_average(
        self, ema_model: nn.Module, current_model: nn.Module
    ) -> None:
        """Blend the parameters of ``current_model`` into ``ema_model``."""
        for current_params, ema_params in zip(
            current_model.parameters(), ema_model.parameters()
        ):
            old_weight, up_weight = ema_params.data, current_params.data
            ema_params.data = old_weight * self.beta + (1.0 - self.beta) * up_weight


# ============================================================================
# CoreDiff-native metrics (training log and checkpoint selection)
# ============================================================================
def _clip_window(
    img: torch.Tensor | np.ndarray, cut_min: float, cut_max: float
) -> torch.Tensor | np.ndarray:
    """Convert a normalised image to HU and clip it to ``[cut_min, cut_max]``."""
    img = img * (NORM_MAX - NORM_MIN) + NORM_MIN
    if torch.is_tensor(img):
        img = img.clone()
        img[img < cut_min] = cut_min
        img[img > cut_max] = cut_max
        return img
    return np.clip(img, cut_min, cut_max)


def transfer_calculate_window(
    img: torch.Tensor | np.ndarray,
) -> torch.Tensor | np.ndarray:
    """Map a normalised image to the CoreDiff metric scale ``[0, 255]``.

    Equivalent to ``basic_template.transfer_calculate_window`` (HU window
    ``[CALC_CUT_MIN, CALC_CUT_MAX]``).
    """
    img = _clip_window(img, CALC_CUT_MIN, CALC_CUT_MAX)
    return CALC_SCALE * (img - CALC_CUT_MIN) / (CALC_CUT_MAX - CALC_CUT_MIN)


def transfer_display_window(
    img: torch.Tensor | np.ndarray,
) -> torch.Tensor | np.ndarray:
    """Map a normalised image to ``[0, 1]`` in the soft-tissue display window.

    Equivalent to ``basic_template.transfer_display_window``.
    """
    img = _clip_window(img, DISP_CUT_MIN, DISP_CUT_MAX)
    return (img - DISP_CUT_MIN) / (DISP_CUT_MAX - DISP_CUT_MIN)


def _rmse_hu_from_calc_scale(a: torch.Tensor, b: torch.Tensor) -> float:
    """RMSE in HU of two images on the ``[0, 255]`` metric scale."""
    hu_span = CALC_CUT_MAX - CALC_CUT_MIN
    a = a * hu_span / CALC_SCALE + CALC_CUT_MIN
    b = b * hu_span / CALC_SCALE + CALC_CUT_MIN
    return torch.sqrt(compute_mse(a, b)).item()


def _safe_train_metrics(
    pred: torch.Tensor, target: torch.Tensor
) -> tuple[float, float, float]:
    """CoreDiff-native PSNR/SSIM/RMSE, guarded against degenerate images.

    PSNR and SSIM are computed on the ``[0, 255]`` scale with the target's own
    dynamic range, as in CoreDiff's ``utils/measure.py``; RMSE is in HU.

    Returns:
        ``(psnr, ssim, rmse)``. PSNR is NaN for a constant target or a perfect
        prediction.
    """
    pred_cal = transfer_calculate_window(pred.detach())
    target_cal = transfer_calculate_window(target.detach())
    data_range = float((target_cal.max() - target_cal.min()).item())
    if data_range <= 1e-6:
        return float("nan"), float("nan"), 0.0
    mse = float(torch.mean((pred_cal - target_cal) ** 2).item())
    if mse <= 1e-12:
        return float("nan"), 1.0, 0.0
    psnr = 10.0 * math.log10((data_range * data_range) / mse)
    ssim = max(min(compute_ssim(pred_cal, target_cal, data_range), 1.0), -1.0)
    rmse = _rmse_hu_from_calc_scale(pred_cal, target_cal)
    return psnr, ssim, rmse


# ============================================================================
# Data
# ============================================================================
def _decode_dicom_to_hu(path: str) -> np.ndarray:
    """Decode a DICOM slice to HU."""
    ds = pydicom.dcmread(path, force=True)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    return ds.pixel_array.astype(np.float32) * slope + intercept


def _fd_cache_path(path: str, cache_dir: Path) -> Path:
    """Cache file of a DICOM target: ``<stem>_<md5 of absolute path>.npy``."""
    # os.path.abspath (not Path.resolve) keeps the hashes of existing caches.
    digest = hashlib.md5(os.path.abspath(path).encode()).hexdigest()[:12]
    return cache_dir / f"{Path(path).stem}_{digest}.npy"


def ensure_fd_npy_cache(paths: list[str], cache_dir: Path) -> list[str]:
    """Convert DICOM targets to ``.npy`` (HU) once and return the cached paths.

    Keeps ``pydicom`` decoding out of the per-item data loading, which was the
    I/O bottleneck of training. ``.npy`` targets are returned unchanged; the
    cached values are normalised on load like any other target.

    Args:
        paths: Target paths.
        cache_dir: Cache folder, shared by both doses.

    Returns:
        Paths to load, aligned with ``paths``.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    out, todo = [], []
    for p in paths:
        if p.lower().endswith(".npy"):
            out.append(p)
            continue
        cached = _fd_cache_path(p, cache_dir)
        out.append(str(cached))
        if not cached.exists():
            todo.append((p, cached))
    if todo:
        if pydicom is None:
            raise RuntimeError("pydicom is required to cache DICOM targets")
        logger.info(
            "Converting %d full-dose DICOMs to .npy (once) in %s", len(todo), cache_dir
        )
        for i, (p, cached) in enumerate(todo, 1):
            np.save(cached, _decode_dicom_to_hu(p).astype(np.float32))
            if i % FD_CACHE_LOG_EVERY == 0:
                logger.info("  converted %d/%d", i, len(todo))
        logger.info("FD cache ready (%d converted)", len(todo))
    return out


class CoreDiffDataset(Dataset):
    """ULDCT pairs with three-slice LDCT context and whole 512x512 slices.

    With context, the input is the stack of three consecutive LDCT slices and
    the target the full-dose slice matching the centre one; the first and last
    slice of every patient are therefore dropped.

    Attributes:
        split: Dataset split.
        inputs: Centre LDCT slice of every item.
        targets: Full-dose target paths (cached ``.npy``).
        input_triples: ``(above, centre, below)`` LDCT paths, or ``None``
            without context.
    """

    def __init__(
        self,
        split: str,
        data_root: Path,
        fd_cache_dir: Path,
        patch_size: int | None = None,
        patch_n: int | None = None,
    ) -> None:
        """Discover and pair the files of ``split``.

        Args:
            split: Dataset split.
            data_root: Dataset root.
            fd_cache_dir: Folder of the decoded full-dose cache.
            patch_size: Patch side (only used without context and below
                ``IMG_SIZE``).
            patch_n: Patches per slice.

        Raises:
            FileNotFoundError: If the split has no inputs or no LD/FD pairs.
        """
        self.split = split
        self.data_root = Path(data_root)
        self.patch_size, self.patch_n = patch_size, patch_n
        self.context = USE_CONTEXT
        raw_inputs = list_low_dose_inputs(self.data_root, split)
        if not raw_inputs:
            raise FileNotFoundError(
                f"No .npy inputs in {self.data_root / split / LOW_DOSE_DIR}"
            )
        self._pair(raw_inputs)
        self.targets = ensure_fd_npy_cache(self.targets, Path(fd_cache_dir))

    def _pair(self, raw_inputs: list[str]) -> None:
        """Pair LDCT inputs with full-dose targets by sorted position per patient.

        Patients without full-dose data are skipped.
        """
        inputs, triples, targets, skipped = [], [], [], []
        groups = group_by_patient_sorted(raw_inputs)
        for pid in sorted(groups):
            lst = groups[pid]
            fd_list = list_normal_dose(self.data_root, pid, self.split)
            if not fd_list:
                skipped.append(pid)
                continue
            n = min(len(lst), len(fd_list))
            if self.context:
                if n < 3:
                    continue
                for i in range(1, n - 1):
                    triples.append((lst[i - 1], lst[i], lst[i + 1]))
                    targets.append(fd_list[i])
            else:
                inputs.extend(lst[:n])
                targets.extend(fd_list[:n])
        if skipped:
            logger.info(
                "split=%s: no full-dose data for %s; skipping", self.split, skipped
            )
        if self.context:
            if not triples:
                raise FileNotFoundError(f"No LD/FD triples in split {self.split!r}")
            self.input_triples = triples
            self.inputs = [t[1] for t in triples]
        else:
            if not inputs:
                raise FileNotFoundError(f"No LD/FD pairs in split {self.split!r}")
            self.input_triples = None
            self.inputs = inputs
        self.targets = targets

    def __len__(self) -> int:
        """Number of items."""
        if self.input_triples is not None:
            return len(self.input_triples)
        return len(self.inputs)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(x, y)``: ``(3, H, W)`` and ``(1, H, W)`` with context."""
        if self.input_triples is not None:
            x = np.stack([load_input_array(p) for p in self.input_triples[idx]], axis=0)
            y = load_target_array(self.targets[idx])[np.newaxis, ...]
            return x, y
        x = load_input_array(self.inputs[idx])
        y = load_target_array(self.targets[idx])
        if self.patch_size and self.patch_size != IMG_SIZE:
            return random_patch_pairs(x, y, self.patch_size, self.patch_n)
        return x, y


def _prepare_batch(
    x: torch.Tensor, y: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Move a batch to ``device`` with shapes ``(B, C, H, W)`` / ``(B, 1, H, W)``."""
    x = x.float().to(device)
    y = y.float().to(device)
    if x.dim() == 3:
        x = x.unsqueeze(1)
        y = y.unsqueeze(1) if y.dim() == 3 else y
        return x, y
    if y.dim() == 3:
        y = y.unsqueeze(1)
    return x, y


# ============================================================================
# Training and testing
# ============================================================================
@torch.no_grad()
def quick_eval_corediff(
    diffusion: Diffusion,
    loader: DataLoader,
    device: torch.device,
    max_slices: int | None = None,
) -> tuple[float, float, float]:
    """Mean CoreDiff-native PSNR/SSIM/RMSE over the validation loader.

    Slices with a non-finite PSNR are skipped.

    Args:
        diffusion: Model to sample from.
        loader: Validation loader.
        device: Inference device.
        max_slices: Optional limit on the number of evaluated slices.

    Returns:
        ``(psnr, ssim, rmse)``, or zeros if no slice was evaluated.
    """
    psnr_sum, ssim_sum, rmse_sum, n = 0.0, 0.0, 0.0, 0
    for x, y in tqdm.tqdm(loader, desc="test", leave=False):
        if max_slices is not None and n >= max_slices:
            break
        x, y = _prepare_batch(x, y, device)
        pred, _, _ = diffusion.sample(
            batch_size=x.shape[0], img=x, n_iter=1, start_adjust_iter=START_ADJUST_ITER
        )
        psnr, ssim, rmse = _safe_train_metrics(pred, y)
        if not math.isfinite(psnr):
            continue
        psnr_sum += psnr
        ssim_sum += ssim
        rmse_sum += rmse
        n += 1
    if n == 0:
        return 0.0, 0.0, 0.0
    return psnr_sum / n, ssim_sum / n, rmse_sum / n


def _format_metric(value: float, spec: str, nan_text: str) -> str:
    """Format ``value`` with ``spec``, or return ``nan_text`` if not finite."""
    return format(value, spec) if math.isfinite(value) else nan_text


def train(
    diffusion: Diffusion,
    loader: DataLoader,
    device: torch.device,
    cfg: RunConfig,
    ema_model: Diffusion | None = None,
    val_loader: DataLoader | None = None,
) -> None:
    """Train for ``MAX_ITERS`` iterations, resuming from the last checkpoint.

    Every ``SAVE_ITERS`` iterations (and at the end) the EMA model is
    validated and the last/best checkpoints, the loss history and the
    training-metrics CSV are written.

    Args:
        diffusion: Model to train.
        loader: Training loader.
        device: Training device.
        cfg: Run configuration.
        ema_model: EMA copy of the model; evaluated and checkpointed.
        val_loader: Validation loader.
    """
    diffusion.train()
    opt = optim.Adam(diffusion.parameters(), lr=LR)
    ema = EMA(EMA_DECAY) if ema_model is not None else None
    if ema_model is not None:
        ema_model.load_state_dict(diffusion.state_dict())
        ema_model.train()

    extra_models = {"ema_model": ema_model}
    ckpt = load_checkpoint(
        cfg.last_ckpt,
        model=diffusion,
        optimizer=opt,
        extra_models={k: m for k, m in extra_models.items() if m is not None},
    )
    start_epoch = (ckpt["epoch"] + 1) if ckpt else 1
    step = ckpt["step"] if ckpt else 0
    best_psnr = best_psnr_so_far(cfg.best_ckpt)
    losses: list[float] = []
    log_rows: list[tuple[int, float, float, float, float]] = []
    csv_path = cfg.output_dir / f"train_metrics_{cfg.dose}.csv"
    t0 = time.time()
    logger.info(
        "start_epoch=%d start_iter=%d print_every=%d save_every=%d",
        start_epoch,
        step,
        PRINT_ITERS,
        SAVE_ITERS,
    )

    # The DataParallel wrapper is only used for the training forward pass;
    # evaluation and checkpoints use the bare modules, so the saved state_dicts
    # have no "module." prefix and stay compatible with single-GPU runs.
    net: nn.Module = diffusion
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(diffusion)
        logger.info("DataParallel on %d GPUs", torch.cuda.device_count())

    def eval_and_save(at_epoch: int, at_step: int) -> None:
        """Validate the EMA model and write checkpoints, losses and CSV."""
        nonlocal best_psnr
        eval_model = ema_model if ema_model is not None else diffusion
        diffusion.eval()
        eval_model.eval()
        if val_loader is not None:
            logger.info(
                "[eval] epoch=%d step=%d: evaluating EMA on val_loader (%d batches)",
                at_epoch,
                at_step,
                len(val_loader),
            )
            psnr, ssim, rmse = quick_eval_corediff(eval_model, val_loader, device)
        else:
            psnr, ssim, rmse = (0.0, 0.0, 0.0)
        diffusion.train()
        if ema_model is not None:
            ema_model.train()
        state = dict(
            model=diffusion,
            optimizer=opt,
            extra_models=extra_models,
            epoch=at_epoch,
            step=at_step,
            lr=opt.param_groups[0]["lr"],
            loss=losses[-1] if losses else 0.0,
            psnr=psnr,
            ssim=ssim,
            rmse_hu=rmse,
        )
        save_checkpoint(cfg.last_ckpt, **state)
        if psnr > best_psnr:
            best_psnr = psnr
            save_checkpoint(cfg.best_ckpt, **state)
        np.save(cfg.losses_path, np.array(losses))
        if log_rows:
            lines = [f"iter,loss,psnr,ssim,rmse_{RMSE_UNIT}\n"]
            lines += [
                f"{r[0]},{r[1]:.6f},{r[2]:.4f},{r[3]:.4f},{r[4]:.4f}\n"
                for r in log_rows
            ]
            csv_path.write_text("".join(lines))
        logger.info(
            "  saved last.pt (epoch=%d step=%d psnr=%.3f ssim=%.4f rmse_hu=%.3f) "
            "best_psnr=%.3f",
            at_epoch,
            at_step,
            psnr,
            ssim,
            rmse,
            best_psnr,
        )

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        for x, y in loader:
            step += 1
            x, y = _prepare_batch(x, y, device)
            # One timestep per batch, drawn here so that every DataParallel
            # replica receives the same t.
            t1 = torch.randint(0, diffusion.num_timesteps, (1,), device=device).long()
            t_batch = t1.repeat(y.shape[0])
            x_recon, _, x_recon_sub1, _ = net(
                x,
                y,
                n_iter=step,
                only_adjust_two_step=ONLY_ADJUST_TWO_STEP,
                start_adjust_iter=START_ADJUST_ITER,
                t=t_batch,
            )
            loss = 0.5 * F.mse_loss(x_recon, y) + 0.5 * F.mse_loss(x_recon_sub1, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
            if ema is not None and step % EMA_UPDATE_EVERY == 0:
                if step < EMA_START_ITER:
                    ema_model.load_state_dict(diffusion.state_dict())
                else:
                    ema.update_model_average(ema_model, diffusion)
            if step % PRINT_ITERS == 0:
                with torch.no_grad():
                    psnr, ssim, rmse = _safe_train_metrics(x_recon, y)
                log_rows.append((step, loss.item(), psnr, ssim, rmse))
                logger.info(
                    "epoch %d/%d  iter %d  loss %.6f  PSNR %s  SSIM %s  "
                    "RMSE %6.4f %s  (%.0fs)",
                    epoch,
                    NUM_EPOCHS,
                    step,
                    loss.item(),
                    _format_metric(psnr, "6.2f", "  nan "),
                    _format_metric(ssim, ".4f", " nan "),
                    rmse,
                    RMSE_UNIT,
                    time.time() - t0,
                )
            if step % SAVE_ITERS == 0:
                eval_and_save(epoch, step)
            if step >= MAX_ITERS:
                # Runs again when MAX_ITERS is a multiple of SAVE_ITERS
                # (kept from the original run).
                eval_and_save(epoch, step)
                logger.info("Reached MAX_ITERS=%d", MAX_ITERS)
                return


def test(
    diffusion: Diffusion, loader: DataLoader, device: torch.device, cfg: RunConfig
) -> None:
    """Evaluate on the full test split with the protocol shared by all models.

    Metrics use HU clipped to ``[TRUNC_MIN, TRUNC_MAX]`` with data range 400,
    like the other models; CoreDiff's native ``[0, 255]`` windowing would put
    PSNR about 14 dB off the common scale. Figures use the soft-tissue display
    window.

    Args:
        diffusion: Model to sample from (the EMA model).
        loader: Test loader with batch size 1.
        device: Inference device.
        cfg: Run configuration.
    """
    diffusion.eval()
    input_sum, pred_sum, n = [0.0] * 3, [0.0] * 3, 0
    abdomen_idx = abdomen_fig_index(loader, cfg.abdomen_idx, cfg.abdomen_frac)
    cs = PATCH_SIZE
    with torch.no_grad():
        for i, (x, y) in enumerate(tqdm.tqdm(loader, desc="final test", leave=False)):
            x, y = _prepare_batch(x, y, device)
            off = (x.shape[-1] - cs) // 2
            x_eval = x[..., off : off + cs, off : off + cs]
            yc = y[..., off : off + cs, off : off + cs]
            xc = x_eval[:, 1:2] if (USE_CONTEXT and x_eval.shape[1] == 3) else x_eval
            pred, _, _ = diffusion.sample(
                batch_size=x_eval.shape[0],
                img=x_eval,
                n_iter=1,
                start_adjust_iter=START_ADJUST_ITER,
            )
            xv, yv, pv = to_hu_window(xc), to_hu_window(yc), to_hu_window(pred)
            o, p = compute_measure(xv, yv, pv, DATA_RANGE)
            for k in range(3):
                input_sum[k] += o[k]
                pred_sum[k] += p[k]
            n += 1
            if i < NUM_TEST_FIGURES or i == abdomen_idx:
                xd, yd, pd = (
                    transfer_display_window(t.cpu().detach()).view(cs, cs)
                    for t in (xc, yc, pred)
                )
                name = ABDOMEN_FIG_NAME if i == abdomen_idx else None
                save_fig(
                    xd, yd, pd, i, o, p, cfg.fig_dir, cfg.dose,
                    name=name, vmin=0.0, vmax=1.0, rmse_unit=" HU",
                )  # fmt: skip
    write_test_metrics(
        cfg,
        [v / n for v in input_sum],
        [v / n for v in pred_sum],
        n,
        rmse_unit=" HU",
    )


def _make_loader(ds: Dataset, cfg: RunConfig, **kwargs: object) -> DataLoader:
    """DataLoader with pinned memory and persistent, prefetching workers."""
    if cfg.num_workers > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=4)
    return DataLoader(ds, num_workers=cfg.num_workers, pin_memory=True, **kwargs)


def main() -> None:
    """Train CoreDiff on the training split, then test the final EMA model."""
    parser = build_arg_parser(MODEL_NAME, description=__doc__)
    parser.add_argument(
        "--fd-cache-dir",
        type=Path,
        default=DEFAULT_FD_CACHE_DIR,
        help="Cache of full-dose DICOMs decoded to .npy (shared by both doses).",
    )
    cfg, args = parse_run_config(parser, MODEL_NAME)
    torch.backends.cudnn.benchmark = True
    device = get_device()
    logger.info(
        "[%s] dose=%s device=%s out=%s", MODEL_NAME, cfg.dose, device, cfg.output_dir
    )

    def make_ds(split: str, **kwargs: int) -> CoreDiffDataset:
        return CoreDiffDataset(split, cfg.data_root, args.fd_cache_dir, **kwargs)

    train_ds = make_ds(TRAIN_SPLIT, patch_size=PATCH_SIZE, patch_n=PATCH_N)
    test_ds = make_ds(TEST_SPLIT)
    val_ds = optional_split(lambda: make_ds(VAL_SPLIT), VAL_SPLIT)
    logger.info(
        "train files: %d | val files: %d | test files: %d",
        len(train_ds),
        len(val_ds) if val_ds is not None else 0,
        len(test_ds),
    )

    train_loader = _make_loader(train_ds, cfg, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = _make_loader(test_ds, cfg, batch_size=1, shuffle=False)
    val_loader = (
        _make_loader(val_ds, cfg, batch_size=1, shuffle=False)
        if val_ds is not None
        else test_loader
    )

    diffusion = build_model().to(device)
    ema_model = build_model().to(device)
    train(
        diffusion, train_loader, device, cfg, ema_model=ema_model, val_loader=val_loader
    )
    # The final EMA weights are tested (not the best checkpoint), as in the
    # reported runs.
    test(ema_model, test_loader, device, cfg)


if __name__ == "__main__":
    main()
