"""Image-quality metrics (PSNR, SSIM, RMSE) shared by all models."""

from __future__ import annotations

from math import exp

import numpy as np
import torch
import torch.nn.functional as F

Array = torch.Tensor | np.ndarray

SSIM_WINDOW_SIZE = 11
SSIM_SIGMA = 1.5
SSIM_K1, SSIM_K2 = 0.01, 0.03


def compute_mse(a: Array, b: Array) -> Array:
    """Mean squared error between two images."""
    return ((a - b) ** 2).mean()


def compute_rmse(a: Array, b: Array) -> float:
    """Root mean squared error, in the units of the inputs (HU here)."""
    if torch.is_tensor(a):
        return torch.sqrt(compute_mse(a, b)).item()
    return float(np.sqrt(compute_mse(a, b)))


def compute_psnr(a: Array, b: Array, data_range: float) -> float:
    """Peak signal-to-noise ratio in dB.

    Args:
        a: Test image.
        b: Reference image.
        data_range: Dynamic range of the images (peak value).
    """
    mse = compute_mse(a, b)
    if torch.is_tensor(a):
        return (10 * torch.log10((data_range**2) / mse)).item()
    return float(10 * np.log10((data_range**2) / mse))


def _gaussian(window_size: int, sigma: float) -> torch.Tensor:
    """Normalised 1-D Gaussian kernel."""
    g = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return g / g.sum()


def _window(window_size: int, channels: int) -> torch.Tensor:
    """2-D Gaussian window of shape ``(channels, 1, window_size, window_size)``."""
    w1 = _gaussian(window_size, SSIM_SIGMA).unsqueeze(1)
    w2 = w1.mm(w1.t()).float().unsqueeze(0).unsqueeze(0)
    return w2.expand(channels, 1, window_size, window_size).contiguous()


def compute_ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    data_range: float,
    window_size: int = SSIM_WINDOW_SIZE,
    channels: int = 1,
    offset: float = 0.0,
) -> float:
    """Structural similarity index (Wang et al., 2004) with a Gaussian window.

    Args:
        img1: Test image, ``(H, W)`` with ``H == W`` or ``(N, C, H, W)``.
        img2: Reference image, same shape as ``img1``.
        data_range: Dynamic range of the images.
        window_size: Side of the Gaussian window.
        channels: Number of image channels.
        offset: Value subtracted from both images before the computation.
            Non-zero only for FGDM, which shifts HU images to ``[0, data_range]``
            (``offset=TRUNC_MIN``). Variances are unaffected, but the luminance
            term is not shift-invariant, so the result differs from
            ``offset=0``.

    Returns:
        Mean SSIM over the image.
    """
    if img1.dim() == 2:
        s = img1.shape[-1]
        img1 = img1.view(1, 1, s, s)
        img2 = img2.view(1, 1, s, s)
    if offset:
        img1 = img1 - offset
        img2 = img2 - offset
    w = _window(window_size, channels).type_as(img1)
    pad = window_size // 2
    mu1 = F.conv2d(img1, w, padding=pad)
    mu2 = F.conv2d(img2, w, padding=pad)
    mu1_sq, mu2_sq, mu12 = mu1.pow(2), mu2.pow(2), mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, w, padding=pad) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, w, padding=pad) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, w, padding=pad) - mu12
    c1, c2 = (SSIM_K1 * data_range) ** 2, (SSIM_K2 * data_range) ** 2
    ssim_map = ((2 * mu12 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return ssim_map.mean().item()


Measure = tuple[float, float, float]


def compute_measure(
    x: torch.Tensor,
    y: torch.Tensor,
    pred: torch.Tensor,
    data_range: float,
    ssim_offset: float = 0.0,
) -> tuple[Measure, Measure]:
    """Compare the LDCT input and the prediction against the full-dose target.

    Args:
        x: LDCT input.
        y: Full-dose target.
        pred: Model prediction.
        data_range: Dynamic range of the images.
        ssim_offset: Forwarded to :func:`compute_ssim` as ``offset``.

    Returns:
        ``((psnr, ssim, rmse) of x vs y, (psnr, ssim, rmse) of pred vs y)``.
    """

    def measure(a: torch.Tensor) -> Measure:
        return (
            compute_psnr(a, y, data_range),
            compute_ssim(a, y, data_range, offset=ssim_offset),
            compute_rmse(a, y),
        )

    return measure(x), measure(pred)
