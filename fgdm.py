"""FGDM (frequency-guided diffusion model) trained and evaluated on ULDCT.

Adversarial diffusion with an NCSN++ generator (FGDM-main ``models_noZ``
variant: no latent z, time-conditioned AdaGN), a time-dependent discriminator,
edge (Sobel high-pass) conditioning, R1 regularisation and cosine-annealed
learning rates for both networks.

Training only sees full-dose images and their edge maps. Inference follows the
paper's zero-shot translation (Algorithm 1): the low-dose input is
forward-diffused to step ~T and then denoised in reverse, conditioned at every
step on the Sobel edges of the low-dose input.

The network code is vendored from FGDM-main/score_sde/models (utils,
dense_layer, up_or_down_sampling, layers, layerspp, discriminator and
models_noZ/ncsnpp_generator_adagn). The compiled ``score_sde.op.upfirdn2d``
kernel is used when installed; otherwise a pure-PyTorch port is used.

Usage:
    python fgdm.py --dose 5pct --data-root /path/to/uldct_5pct/dataset
"""

from __future__ import annotations

import functools
import hashlib
import logging
import math
import os
import random
import string
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.nn.init import _calculate_fan_in_and_fan_out
from torch.utils.data import DataLoader

from common.checkpoint import best_psnr_so_far, resolve_checkpoint_path
from common.config import (
    DATA_RANGE,
    TEST_SPLIT,
    TRAIN_SPLIT,
    TRUNC_MIN,
    VAL_SPLIT,
    RunConfig,
    build_arg_parser,
    get_device,
    parse_run_config,
)
from common.data import (
    DICOM_SUFFIXES,
    ULDCTDataset,
    denormalize,
    load_input_array,
    normalize,
    truncate,
)
from common.evaluation import run_test
from common.metrics import compute_psnr, compute_rmse, compute_ssim
from common.runtime import optional_split

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import pydicom
except ImportError:
    pydicom = None

MODEL_NAME = "fgdm"

# Hyperparameters: FGDM-main/main.py defaults, except where the paper differs
# (learning rate, image size, training eta range), in which case the paper is
# followed.
NUM_EPOCHS = 200
BATCH_SIZE = 8
PATCH_SIZE = 192  # paper: 192x192 for AAPM (256 was only for the MR T1 dataset)
PATCH_N = 1
LR = 1e-4  # paper (the repo used 1.5e-4)
LR_D = 1e-4
LR_MIN = 1e-5  # eta_min of the cosine schedules
ADAM_BETA1 = 0.5
ADAM_BETA2 = 0.9
TIMESTEPS = 4
BETA_MIN = 0.1
BETA_MAX = 1.0
NGF = 64
T_EMB_DIM = 256
R1_GAMMA = 0.05
D_LEAKY_SLOPE = 0.2
PRINT_ITERS = 25
SAVE_CKPT_EVERY = 10  # epochs
VAL_MAX_SLICES = 8
PREFETCH_FACTOR = 4

# Zero-shot translation (paper Algorithm 1, Sec. III-E): forward-diffuse the
# low-dose input to ~T (Eq. 22/29) and condition every reverse step on its
# Sobel edges with threshold eta (Eq. 25). Paper: ~T = 4 and eta = 10 at test
# time; eta drawn from [1, 25] during training.
TEST_TILDE_T = TIMESTEPS
TEST_ETA = 10
TEST_BILATERAL = 9  # repo value; the paper does not specify it
EDGE_ETA_MAX = 25  # paper (the repo used 1-10)
EDGE_BILATERAL_MAX = 30
BILATERAL_DIAMETER = 10

DEFAULT_FD_CACHE_DIR = Path("outputs") / "_fd_target_npy_cache"

logger = logging.getLogger(MODEL_NAME)


# ============================================================================
# Vendored: score_sde/models/dense_layer.py
# ============================================================================
def _calculate_correct_fan(tensor: torch.Tensor, mode: str) -> int:
    """Return the fan-in or fan-out of ``tensor``."""
    mode = mode.lower()
    valid_modes = ["fan_in", "fan_out", "fan_avg"]
    if mode not in valid_modes:
        raise ValueError(f"Mode {mode} not supported, please use one of {valid_modes}")
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == "fan_in" else fan_out


def kaiming_uniform_(
    tensor: torch.Tensor, gain: float = 1.0, mode: str = "fan_in"
) -> torch.Tensor:
    """Fill ``tensor`` in place with a variance-scaled uniform distribution."""
    fan = _calculate_correct_fan(tensor, mode)
    var = gain / max(1.0, fan)
    bound = math.sqrt(3.0 * var)
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)


def variance_scaling_init_(tensor: torch.Tensor, scale: float) -> torch.Tensor:
    """Fan-average variance-scaling initialisation (``scale=0`` -> ~0)."""
    return kaiming_uniform_(tensor, gain=1e-10 if scale == 0 else scale, mode="fan_avg")


def dense(in_channels: int, out_channels: int, init_scale: float = 1.0) -> nn.Linear:
    """Linear layer with variance-scaling weights and zero bias."""
    lin = nn.Linear(in_channels, out_channels)
    variance_scaling_init_(lin.weight, scale=init_scale)
    nn.init.zeros_(lin.bias)
    return lin


def conv2d(
    in_planes: int,
    out_planes: int,
    kernel_size: int | tuple[int, int] = (3, 3),
    stride: int = 1,
    dilation: int = 1,
    padding: int = 1,
    bias: bool = True,
    padding_mode: str = "zeros",
    init_scale: float = 1.0,
) -> nn.Conv2d:
    """2-D convolution with variance-scaling weights and zero bias."""
    conv = nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias,
        padding_mode=padding_mode,
    )
    variance_scaling_init_(conv.weight, scale=init_scale)
    if bias:
        nn.init.zeros_(conv.bias)
    return conv


# ============================================================================
# Vendored: score_sde/models/up_or_down_sampling.py
# ============================================================================
try:
    from score_sde.op import upfirdn2d  # type: ignore
except ImportError:

    def _upfirdn2d_native(
        x: torch.Tensor,
        kernel: torch.Tensor,
        up_x: int,
        up_y: int,
        down_x: int,
        down_y: int,
        pad_x0: int,
        pad_x1: int,
        pad_y0: int,
        pad_y1: int,
    ) -> torch.Tensor:
        """Pure-PyTorch port of FGDM-main/score_sde/op/upfirdn2d.py."""
        _, channel, in_h, in_w = x.shape
        x = x.reshape(-1, in_h, in_w, 1)
        _, in_h, in_w, minor = x.shape
        kernel_h, kernel_w = kernel.shape

        out = x.view(-1, in_h, 1, in_w, 1, minor)
        out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
        out = out.view(-1, in_h * up_y, in_w * up_x, minor)

        out = F.pad(
            out,
            [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)],
        )
        out = out[
            :,
            max(-pad_y0, 0) : out.shape[1] - max(-pad_y1, 0),
            max(-pad_x0, 0) : out.shape[2] - max(-pad_x1, 0),
            :,
        ]

        out = out.permute(0, 3, 1, 2)
        out = out.reshape(
            [-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1]
        )
        w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
        out = F.conv2d(out, w)
        out = out.reshape(
            -1,
            minor,
            in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
            in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
        )
        out = out.permute(0, 2, 3, 1)
        out = out[:, ::down_y, ::down_x, :]

        out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
        out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1
        return out.view(-1, channel, out_h, out_w)

    def upfirdn2d(
        input: torch.Tensor,
        kernel: torch.Tensor,
        up: int = 1,
        down: int = 1,
        pad: tuple[int, int] = (0, 0),
    ) -> torch.Tensor:
        """Upsample, FIR-filter and downsample (same API as the compiled op)."""
        return _upfirdn2d_native(
            input, kernel, up, up, down, down, pad[0], pad[1], pad[0], pad[1]
        )


class Conv2d(nn.Module):
    """Conv2d layer with optimal upsampling and downsampling (StyleGAN2)."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int,
        up: bool = False,
        down: bool = False,
        resample_kernel: Sequence[int] = (1, 3, 3, 1),
        use_bias: bool = True,
        kernel_init: Callable | None = None,
    ) -> None:
        """Create the weight (and bias) parameters."""
        super().__init__()
        assert not (up and down)
        assert kernel >= 1 and kernel % 2 == 1
        self.weight = nn.Parameter(torch.zeros(out_ch, in_ch, kernel, kernel))
        if kernel_init is not None:
            self.weight.data = kernel_init(self.weight.data.shape)
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(out_ch))

        self.up = up
        self.down = down
        self.resample_kernel = resample_kernel
        self.kernel = kernel
        self.use_bias = use_bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convolve, upsampling or downsampling by 2 when configured."""
        if self.up:
            x = upsample_conv_2d(x, self.weight, k=self.resample_kernel)
        elif self.down:
            x = conv_downsample_2d(x, self.weight, k=self.resample_kernel)
        else:
            x = F.conv2d(x, self.weight, stride=1, padding=self.kernel // 2)

        if self.use_bias:
            x = x + self.bias.reshape(1, -1, 1, 1)
        return x


def naive_upsample_2d(x: torch.Tensor, factor: int = 2) -> torch.Tensor:
    """Nearest-neighbour upsampling."""
    _N, C, H, W = x.shape
    x = torch.reshape(x, (-1, C, H, 1, W, 1))
    x = x.repeat(1, 1, 1, factor, 1, factor)
    return torch.reshape(x, (-1, C, H * factor, W * factor))


def naive_downsample_2d(x: torch.Tensor, factor: int = 2) -> torch.Tensor:
    """Average-pooling downsampling."""
    _N, C, H, W = x.shape
    x = torch.reshape(x, (-1, C, H // factor, factor, W // factor, factor))
    return torch.mean(x, dim=(3, 5))


def upsample_conv_2d(
    x: torch.Tensor,
    w: torch.Tensor,
    k: Sequence[int] | None = None,
    factor: int = 2,
    gain: float = 1,
) -> torch.Tensor:
    """Fused upsample + convolution (transposed conv followed by FIR filter)."""
    assert isinstance(factor, int) and factor >= 1

    assert len(w.shape) == 4
    convH = w.shape[2]
    convW = w.shape[3]
    inC = w.shape[1]

    assert convW == convH

    if k is None:
        k = [1] * factor
    k = _setup_kernel(k) * (gain * (factor**2))
    p = (k.shape[0] - factor) - (convW - 1)

    stride = [1, 1, factor, factor]
    output_shape = (
        (_shape(x, 2) - 1) * factor + convH,
        (_shape(x, 3) - 1) * factor + convW,
    )
    output_padding = (
        output_shape[0] - (_shape(x, 2) - 1) * stride[0] - convH,
        output_shape[1] - (_shape(x, 3) - 1) * stride[1] - convW,
    )
    assert output_padding[0] >= 0 and output_padding[1] >= 0
    num_groups = _shape(x, 1) // inC

    w = torch.reshape(w, (num_groups, -1, inC, convH, convW))
    w = w[..., ::-1, ::-1].permute(0, 2, 1, 3, 4)
    w = torch.reshape(w, (num_groups * inC, -1, convH, convW))

    x = F.conv_transpose2d(
        x, w, stride=stride, output_padding=output_padding, padding=0
    )

    return upfirdn2d(
        x,
        torch.tensor(k, device=x.device),
        pad=((p + 1) // 2 + factor - 1, p // 2 + 1),
    )


def conv_downsample_2d(
    x: torch.Tensor,
    w: torch.Tensor,
    k: Sequence[int] | None = None,
    factor: int = 2,
    gain: float = 1,
) -> torch.Tensor:
    """Fused FIR filter + strided convolution."""
    assert isinstance(factor, int) and factor >= 1
    _outC, _inC, convH, convW = w.shape
    assert convW == convH
    if k is None:
        k = [1] * factor
    k = _setup_kernel(k) * gain
    p = (k.shape[0] - factor) + (convW - 1)
    s = [factor, factor]
    x = upfirdn2d(x, torch.tensor(k, device=x.device), pad=((p + 1) // 2, p // 2))
    return F.conv2d(x, w, stride=s, padding=0)


def _setup_kernel(k: Sequence[int]) -> np.ndarray:
    """Normalised 2-D FIR kernel (outer product of a 1-D kernel)."""
    k = np.asarray(k, dtype=np.float32)
    if k.ndim == 1:
        k = np.outer(k, k)
    k /= np.sum(k)
    assert k.ndim == 2
    assert k.shape[0] == k.shape[1]
    return k


def _shape(x: torch.Tensor, dim: int) -> int:
    """Size of ``x`` along ``dim``."""
    return x.shape[dim]


def upsample_2d(
    x: torch.Tensor, k: Sequence[int] | None = None, factor: int = 2, gain: float = 1
) -> torch.Tensor:
    """FIR-filtered upsampling."""
    assert isinstance(factor, int) and factor >= 1
    if k is None:
        k = [1] * factor
    k = _setup_kernel(k) * (gain * (factor**2))
    p = k.shape[0] - factor
    return upfirdn2d(
        x,
        torch.tensor(k, device=x.device),
        up=factor,
        pad=((p + 1) // 2 + factor - 1, p // 2),
    )


def downsample_2d(
    x: torch.Tensor, k: Sequence[int] | None = None, factor: int = 2, gain: float = 1
) -> torch.Tensor:
    """FIR-filtered downsampling."""
    assert isinstance(factor, int) and factor >= 1
    if k is None:
        k = [1] * factor
    k = _setup_kernel(k) * gain
    p = k.shape[0] - factor
    return upfirdn2d(
        x, torch.tensor(k, device=x.device), down=factor, pad=((p + 1) // 2, p // 2)
    )


# ============================================================================
# Vendored: score_sde/models/layers.py
# ============================================================================
def variance_scaling(
    scale: float,
    mode: str,
    distribution: str,
    in_axis: int = 1,
    out_axis: int = 0,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> Callable[..., torch.Tensor]:
    """Variance-scaling initialiser ported from JAX."""

    def _compute_fans(
        shape: Sequence[int], in_axis: int = 1, out_axis: int = 0
    ) -> tuple[float, float]:
        receptive_field_size = np.prod(shape) / shape[in_axis] / shape[out_axis]
        fan_in = shape[in_axis] * receptive_field_size
        fan_out = shape[out_axis] * receptive_field_size
        return fan_in, fan_out

    def init(
        shape: Sequence[int], dtype: torch.dtype = dtype, device: str = device
    ) -> torch.Tensor:
        fan_in, fan_out = _compute_fans(shape, in_axis, out_axis)
        if mode == "fan_in":
            denominator = fan_in
        elif mode == "fan_out":
            denominator = fan_out
        elif mode == "fan_avg":
            denominator = (fan_in + fan_out) / 2
        else:
            raise ValueError(f"invalid mode for variance scaling initializer: {mode}")
        variance = scale / denominator
        if distribution == "normal":
            return torch.randn(*shape, dtype=dtype, device=device) * np.sqrt(variance)
        elif distribution == "uniform":
            return (
                torch.rand(*shape, dtype=dtype, device=device) * 2.0 - 1.0
            ) * np.sqrt(3 * variance)
        else:
            raise ValueError("invalid distribution for variance scaling initializer")

    return init


def default_init(scale: float = 1.0) -> Callable[..., torch.Tensor]:
    """The initialisation used in DDPM."""
    scale = 1e-10 if scale == 0 else scale
    return variance_scaling(scale, "fan_avg", "uniform")


def ddpm_conv1x1(
    in_planes: int,
    out_planes: int,
    stride: int = 1,
    bias: bool = True,
    init_scale: float = 1.0,
    padding: int = 0,
) -> nn.Conv2d:
    """1x1 convolution with DDPM initialisation."""
    conv = nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        padding=padding,
        bias=bias,
    )
    conv.weight.data = default_init(init_scale)(conv.weight.data.shape)
    nn.init.zeros_(conv.bias)
    return conv


def ddpm_conv3x3(
    in_planes: int,
    out_planes: int,
    stride: int = 1,
    bias: bool = True,
    dilation: int = 1,
    init_scale: float = 1.0,
    padding: int = 1,
) -> nn.Conv2d:
    """3x3 convolution with DDPM initialisation."""
    conv = nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias,
    )
    conv.weight.data = default_init(init_scale)(conv.weight.data.shape)
    nn.init.zeros_(conv.bias)
    return conv


def get_timestep_embedding(
    timesteps: torch.Tensor, embedding_dim: int, max_positions: int = 10000
) -> torch.Tensor:
    """Sinusoidal timestep embedding of shape ``(N, embedding_dim)``."""
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(
        torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb
    )
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1), mode="constant")
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


def _einsum(
    a: list[str], b: list[str], c: list[str], x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    """Einsum from per-operand index lists."""
    einsum_str = "{},{}->{}".format("".join(a), "".join(b), "".join(c))
    return torch.einsum(einsum_str, x, y)


def contract_inner(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """``tensordot(x, y, 1)``."""
    x_chars = list(string.ascii_lowercase[: len(x.shape)])
    y_chars = list(string.ascii_lowercase[len(x.shape) : len(y.shape) + len(x.shape)])
    y_chars[0] = x_chars[-1]
    out_chars = x_chars[:-1] + y_chars[1:]
    return _einsum(x_chars, y_chars, out_chars, x, y)


class NIN(nn.Module):
    """Network-in-network (per-pixel linear) layer."""

    def __init__(self, in_dim: int, num_units: int, init_scale: float = 0.1) -> None:
        """Create the weight and bias parameters."""
        super().__init__()
        self.W = nn.Parameter(
            default_init(scale=init_scale)((in_dim, num_units)), requires_grad=True
        )
        self.b = nn.Parameter(torch.zeros(num_units), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the layer to ``(N, C, H, W)`` features."""
        x = x.permute(0, 2, 3, 1)
        y = contract_inner(x, self.W) + self.b
        return y.permute(0, 3, 1, 2)


# ============================================================================
# Vendored: score_sde/models/layerspp.py
# ============================================================================
conv1x1 = ddpm_conv1x1
conv3x3 = ddpm_conv3x3


class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for noise levels."""

    def __init__(self, embedding_size: int = 256, scale: float = 1.0) -> None:
        """Draw the fixed random frequencies."""
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed a batch of noise levels."""
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Combine(nn.Module):
    """Combine information from skip connections."""

    def __init__(self, dim1: int, dim2: int, method: str = "cat") -> None:
        """Create the 1x1 projection."""
        super().__init__()
        self.Conv_0 = conv1x1(dim1, dim2)
        self.method = method

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Project ``x`` and concatenate it with, or add it to, ``y``."""
        h = self.Conv_0(x)
        if self.method == "cat":
            return torch.cat([h, y], dim=1)
        elif self.method == "sum":
            return h + y
        else:
            raise ValueError(f"Method {self.method} not recognized.")


class AttnBlockpp(nn.Module):
    """Channel-wise self-attention block. Modified from DDPM."""

    def __init__(
        self, channels: int, skip_rescale: bool = False, init_scale: float = 0.0
    ) -> None:
        """Create the normalisation and the q/k/v/output projections."""
        super().__init__()
        self.GroupNorm_0 = nn.GroupNorm(
            num_groups=min(channels // 4, 32), num_channels=channels, eps=1e-6
        )
        self.NIN_0 = NIN(channels, channels)
        self.NIN_1 = NIN(channels, channels)
        self.NIN_2 = NIN(channels, channels)
        self.NIN_3 = NIN(channels, channels, init_scale=init_scale)
        self.skip_rescale = skip_rescale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply self-attention over all spatial positions."""
        B, C, H, W = x.shape
        h = self.GroupNorm_0(x)
        q = self.NIN_0(h)
        k = self.NIN_1(h)
        v = self.NIN_2(h)

        w = torch.einsum("bchw,bcij->bhwij", q, k) * (int(C) ** (-0.5))
        w = torch.reshape(w, (B, H, W, H * W))
        w = F.softmax(w, dim=-1)
        w = torch.reshape(w, (B, H, W, H, W))
        h = torch.einsum("bhwij,bcij->bchw", w, v)
        h = self.NIN_3(h)
        if not self.skip_rescale:
            return x + h
        else:
            return (x + h) / np.sqrt(2.0)


class Upsample(nn.Module):
    """2x upsampling, optionally FIR-filtered and/or followed by a conv."""

    def __init__(
        self,
        in_ch: int | None = None,
        out_ch: int | None = None,
        with_conv: bool = False,
        fir: bool = False,
        fir_kernel: Sequence[int] = (1, 3, 3, 1),
    ) -> None:
        """Create the optional convolution."""
        super().__init__()
        out_ch = out_ch if out_ch else in_ch
        if not fir:
            if with_conv:
                self.Conv_0 = conv3x3(in_ch, out_ch)
        else:
            if with_conv:
                self.Conv2d_0 = Conv2d(
                    in_ch,
                    out_ch,
                    kernel=3,
                    up=True,
                    resample_kernel=fir_kernel,
                    use_bias=True,
                    kernel_init=default_init(),
                )
        self.fir = fir
        self.with_conv = with_conv
        self.fir_kernel = fir_kernel
        self.out_ch = out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upsample ``x`` by 2."""
        B, C, H, W = x.shape
        if not self.fir:
            h = F.interpolate(x, (H * 2, W * 2), "nearest")
            if self.with_conv:
                h = self.Conv_0(h)
        else:
            if not self.with_conv:
                h = upsample_2d(x, self.fir_kernel, factor=2)
            else:
                h = self.Conv2d_0(x)
        return h


class Downsample(nn.Module):
    """2x downsampling, optionally FIR-filtered and/or with a strided conv."""

    def __init__(
        self,
        in_ch: int | None = None,
        out_ch: int | None = None,
        with_conv: bool = False,
        fir: bool = False,
        fir_kernel: Sequence[int] = (1, 3, 3, 1),
    ) -> None:
        """Create the optional convolution."""
        super().__init__()
        out_ch = out_ch if out_ch else in_ch
        if not fir:
            if with_conv:
                self.Conv_0 = conv3x3(in_ch, out_ch, stride=2, padding=0)
        else:
            if with_conv:
                self.Conv2d_0 = Conv2d(
                    in_ch,
                    out_ch,
                    kernel=3,
                    down=True,
                    resample_kernel=fir_kernel,
                    use_bias=True,
                    kernel_init=default_init(),
                )
        self.fir = fir
        self.fir_kernel = fir_kernel
        self.with_conv = with_conv
        self.out_ch = out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Downsample ``x`` by 2."""
        B, C, H, W = x.shape
        if not self.fir:
            if self.with_conv:
                x = F.pad(x, (0, 1, 0, 1))
                x = self.Conv_0(x)
            else:
                x = F.avg_pool2d(x, 2, stride=2)
        else:
            if not self.with_conv:
                x = downsample_2d(x, self.fir_kernel, factor=2)
            else:
                x = self.Conv2d_0(x)
        return x


# ============================================================================
# Vendored: score_sde/models/discriminator.py
# ============================================================================
# Shared default activation (upstream evaluates nn.LeakyReLU(0.2) once, at
# definition time, so every default user shares one parameter-free instance).
_DEFAULT_ACT = nn.LeakyReLU(0.2)


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding followed by a two-layer MLP."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
        act: nn.Module = _DEFAULT_ACT,
    ) -> None:
        """Create the MLP."""
        super().__init__()

        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.main = nn.Sequential(
            dense(embedding_dim, hidden_dim),
            act,
            dense(hidden_dim, output_dim),
        )

    def forward(self, temp: torch.Tensor) -> torch.Tensor:
        """Embed a batch of integer timesteps."""
        temb = get_timestep_embedding(temp, self.embedding_dim)
        temb = self.main(temb)
        return temb


class DownConvBlock(nn.Module):
    """Time-conditioned residual block with optional FIR downsampling."""

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: int = 3,
        padding: int = 1,
        t_emb_dim: int = 128,
        downsample: bool = False,
        act: nn.Module = _DEFAULT_ACT,
        fir_kernel: Sequence[int] = (1, 3, 3, 1),
    ) -> None:
        """Create the convolutions, the time projection and the skip path."""
        super().__init__()

        self.fir_kernel = fir_kernel
        self.downsample = downsample

        self.conv1 = nn.Sequential(
            conv2d(in_channel, out_channel, kernel_size, padding=padding),
        )

        self.conv2 = nn.Sequential(
            conv2d(
                out_channel, out_channel, kernel_size, padding=padding, init_scale=0.0
            )
        )
        self.dense_t1 = dense(t_emb_dim, out_channel)

        self.act = act

        self.skip = nn.Sequential(
            conv2d(in_channel, out_channel, 1, padding=0, bias=False),
        )

    def forward(self, input: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """Apply the block to ``input`` conditioned on ``t_emb``."""
        out = self.act(input)
        out = self.conv1(out)
        out += self.dense_t1(t_emb)[..., None, None]

        out = self.act(out)

        if self.downsample:
            out = downsample_2d(out, self.fir_kernel, factor=2)
            input = downsample_2d(input, self.fir_kernel, factor=2)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out + skip) / np.sqrt(2)
        return out


class Discriminator_large(nn.Module):
    """A time-dependent discriminator for large images (CelebA, LSUN)."""

    def __init__(
        self,
        nc: int = 1,
        ngf: int = 32,
        t_emb_dim: int = 128,
        act: nn.Module = _DEFAULT_ACT,
    ) -> None:
        """Create the layers.

        Args:
            nc: Input channels (``x_t`` and ``x_{t+1}`` concatenated).
            ngf: Base number of feature maps.
            t_emb_dim: Timestep-embedding size.
            act: Activation shared by all layers.
        """
        super().__init__()
        self.act = act

        self.t_embed = TimestepEmbedding(
            embedding_dim=t_emb_dim,
            hidden_dim=t_emb_dim,
            output_dim=t_emb_dim,
            act=act,
        )

        self.start_conv = conv2d(nc, ngf * 2, 1, padding=0)
        self.conv1 = DownConvBlock(
            ngf * 2, ngf * 4, t_emb_dim=t_emb_dim, downsample=True, act=act
        )
        self.conv2 = DownConvBlock(
            ngf * 4, ngf * 8, t_emb_dim=t_emb_dim, downsample=True, act=act
        )
        self.conv3 = DownConvBlock(
            ngf * 8, ngf * 8, t_emb_dim=t_emb_dim, downsample=True, act=act
        )
        self.conv4 = DownConvBlock(
            ngf * 8, ngf * 8, t_emb_dim=t_emb_dim, downsample=True, act=act
        )
        self.conv5 = DownConvBlock(
            ngf * 8, ngf * 8, t_emb_dim=t_emb_dim, downsample=True, act=act
        )
        self.conv6 = DownConvBlock(
            ngf * 8, ngf * 8, t_emb_dim=t_emb_dim, downsample=True, act=act
        )

        self.final_conv = conv2d(ngf * 8 + 1, ngf * 8, 3, padding=1)
        self.end_linear = dense(ngf * 8, 1)

        self.stddev_group = 4
        self.stddev_feat = 1

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, x_t: torch.Tensor
    ) -> torch.Tensor:
        """Score the pair ``(x, x_t)`` at timestep ``t``; returns ``(N, 1)``."""
        t_embed = self.act(self.t_embed(t))

        input_x = torch.cat((x, x_t), dim=1)

        h = self.start_conv(input_x)
        h = self.conv1(h, t_embed)
        h = self.conv2(h, t_embed)
        h = self.conv3(h, t_embed)
        h = self.conv4(h, t_embed)
        h = self.conv5(h, t_embed)
        out = self.conv6(h, t_embed)

        # Minibatch standard deviation feature (StyleGAN2).
        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)
        out = self.act(out)

        out = out.view(out.shape[0], out.shape[1], -1).sum(2)
        out = self.end_linear(out)
        return out


# ============================================================================
# Vendored: score_sde/models_noZ/ncsnpp_generator_adagn.py
# FGDM-main generator variant: no latent-z mapping, time-conditioned AdaGN.
# ============================================================================
class NoZAdaptiveGroupNorm(nn.Module):
    """Non-affine group normalisation (AdaGN without the style input)."""

    def __init__(self, num_groups: int, in_channel: int) -> None:
        """Create the normalisation layer."""
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, in_channel, affine=False, eps=1e-6)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Normalise ``input``."""
        return self.norm(input)


class NoZResnetBlockDDPMppAdagn(nn.Module):
    """DDPM++ residual block (``resblock_type="ddpm"``)."""

    def __init__(
        self,
        act: nn.Module,
        in_ch: int,
        out_ch: int | None = None,
        temb_dim: int | None = None,
        conv_shortcut: bool = False,
        dropout: float = 0.1,
        skip_rescale: bool = False,
        init_scale: float = 0.0,
    ) -> None:
        """Create the layers."""
        super().__init__()
        out_ch = out_ch if out_ch else in_ch
        self.GroupNorm_0 = NoZAdaptiveGroupNorm(min(in_ch // 4, 32), in_ch)
        self.Conv_0 = conv3x3(in_ch, out_ch)
        if temb_dim is not None:
            self.Dense_0 = nn.Linear(temb_dim, out_ch)
            self.Dense_0.weight.data = default_init()(self.Dense_0.weight.data.shape)
            nn.init.zeros_(self.Dense_0.bias)
        self.GroupNorm_1 = NoZAdaptiveGroupNorm(min(out_ch // 4, 32), out_ch)
        self.Dropout_0 = nn.Dropout(dropout)
        self.Conv_1 = conv3x3(out_ch, out_ch, init_scale=init_scale)
        if in_ch != out_ch:
            if conv_shortcut:
                self.Conv_2 = conv3x3(in_ch, out_ch)
            else:
                self.NIN_0 = NIN(in_ch, out_ch)
        self.skip_rescale = skip_rescale
        self.act = act
        self.out_ch = out_ch
        self.conv_shortcut = conv_shortcut

    def forward(
        self, x: torch.Tensor, temb: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Apply the block, conditioned on the time embedding ``temb``."""
        h = self.act(self.GroupNorm_0(x))
        h = self.Conv_0(h)
        if temb is not None:
            h += self.Dense_0(self.act(temb))[:, :, None, None]
        h = self.act(self.GroupNorm_1(h))
        h = self.Dropout_0(h)
        h = self.Conv_1(h)
        if x.shape[1] != self.out_ch:
            if self.conv_shortcut:
                x = self.Conv_2(x)
            else:
                x = self.NIN_0(x)
        if not self.skip_rescale:
            return x + h
        return (x + h) / np.sqrt(2.0)


class NoZResnetBlockBigGANppAdagn(nn.Module):
    """BigGAN++ residual block with optional up/down sampling (``"biggan"``)."""

    def __init__(
        self,
        act: nn.Module,
        in_ch: int,
        out_ch: int | None = None,
        temb_dim: int | None = None,
        up: bool = False,
        down: bool = False,
        dropout: float = 0.1,
        fir: bool = False,
        fir_kernel: Sequence[int] = (1, 3, 3, 1),
        skip_rescale: bool = True,
        init_scale: float = 0.0,
    ) -> None:
        """Create the layers."""
        super().__init__()
        out_ch = out_ch if out_ch else in_ch
        self.GroupNorm_0 = NoZAdaptiveGroupNorm(min(in_ch // 4, 32), in_ch)
        self.up = up
        self.down = down
        self.fir = fir
        self.fir_kernel = fir_kernel
        self.Conv_0 = conv3x3(in_ch, out_ch)
        if temb_dim is not None:
            self.Dense_0 = nn.Linear(temb_dim, out_ch)
            self.Dense_0.weight.data = default_init()(self.Dense_0.weight.shape)
            nn.init.zeros_(self.Dense_0.bias)
        self.GroupNorm_1 = NoZAdaptiveGroupNorm(min(out_ch // 4, 32), out_ch)
        self.Dropout_0 = nn.Dropout(dropout)
        self.Conv_1 = conv3x3(out_ch, out_ch, init_scale=init_scale)
        if in_ch != out_ch or up or down:
            self.Conv_2 = conv1x1(in_ch, out_ch)
        self.skip_rescale = skip_rescale
        self.act = act
        self.in_ch = in_ch
        self.out_ch = out_ch

    def forward(
        self, x: torch.Tensor, temb: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Apply the block, conditioned on the time embedding ``temb``."""
        h = self.act(self.GroupNorm_0(x))
        if self.up:
            if self.fir:
                h = upsample_2d(h, self.fir_kernel, factor=2)
                x = upsample_2d(x, self.fir_kernel, factor=2)
            else:
                h = naive_upsample_2d(h, factor=2)
                x = naive_upsample_2d(x, factor=2)
        elif self.down:
            if self.fir:
                h = downsample_2d(h, self.fir_kernel, factor=2)
                x = downsample_2d(x, self.fir_kernel, factor=2)
            else:
                h = naive_downsample_2d(h, factor=2)
                x = naive_downsample_2d(x, factor=2)
        h = self.Conv_0(h)
        if temb is not None:
            h += self.Dense_0(self.act(temb))[:, :, None, None]
        h = self.act(self.GroupNorm_1(h))
        h = self.Dropout_0(h)
        h = self.Conv_1(h)
        if self.in_ch != self.out_ch or self.up or self.down:
            x = self.Conv_2(x)
        if not self.skip_rescale:
            return x + h
        return (x + h) / np.sqrt(2.0)


class NoZResnetBlockBigGANppAdagnOne(nn.Module):
    """BigGAN++ block with an affine second norm (``"biggan_oneadagn"``)."""

    def __init__(
        self,
        act: nn.Module,
        in_ch: int,
        out_ch: int | None = None,
        temb_dim: int | None = None,
        up: bool = False,
        down: bool = False,
        dropout: float = 0.1,
        fir: bool = False,
        fir_kernel: Sequence[int] = (1, 3, 3, 1),
        skip_rescale: bool = True,
        init_scale: float = 0.0,
    ) -> None:
        """Create the layers."""
        super().__init__()
        out_ch = out_ch if out_ch else in_ch
        self.GroupNorm_0 = NoZAdaptiveGroupNorm(min(in_ch // 4, 32), in_ch)
        self.up = up
        self.down = down
        self.fir = fir
        self.fir_kernel = fir_kernel
        self.Conv_0 = conv3x3(in_ch, out_ch)
        if temb_dim is not None:
            self.Dense_0 = nn.Linear(temb_dim, out_ch)
            self.Dense_0.weight.data = default_init()(self.Dense_0.weight.shape)
            nn.init.zeros_(self.Dense_0.bias)
        self.GroupNorm_1 = nn.GroupNorm(
            num_groups=min(out_ch // 4, 32), num_channels=out_ch, eps=1e-6
        )
        self.Dropout_0 = nn.Dropout(dropout)
        self.Conv_1 = conv3x3(out_ch, out_ch, init_scale=init_scale)
        if in_ch != out_ch or up or down:
            self.Conv_2 = conv1x1(in_ch, out_ch)
        self.skip_rescale = skip_rescale
        self.act = act
        self.in_ch = in_ch
        self.out_ch = out_ch

    def forward(
        self, x: torch.Tensor, temb: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Apply the block, conditioned on the time embedding ``temb``."""
        h = self.act(self.GroupNorm_0(x))
        if self.up:
            if self.fir:
                h = upsample_2d(h, self.fir_kernel, factor=2)
                x = upsample_2d(x, self.fir_kernel, factor=2)
            else:
                h = naive_upsample_2d(h, factor=2)
                x = naive_upsample_2d(x, factor=2)
        elif self.down:
            if self.fir:
                h = downsample_2d(h, self.fir_kernel, factor=2)
                x = downsample_2d(x, self.fir_kernel, factor=2)
            else:
                h = naive_downsample_2d(h, factor=2)
                x = naive_downsample_2d(x, factor=2)
        h = self.Conv_0(h)
        if temb is not None:
            h += self.Dense_0(self.act(temb))[:, :, None, None]
        h = self.act(self.GroupNorm_1(h))
        h = self.Dropout_0(h)
        h = self.Conv_1(h)
        if self.in_ch != self.out_ch or self.up or self.down:
            x = self.Conv_2(x)
        if not self.skip_rescale:
            return x + h
        return (x + h) / np.sqrt(2.0)


class NCSNppNoZ(nn.Module):
    """NCSN++ generator from FGDM-main/score_sde/models_noZ.

    Predicts ``x_0`` from the concatenation of the noisy image ``x_{t+1}`` and
    its edge map, conditioned on the timestep. All options are read from the
    config built by :func:`build_fgdm_config`; the unused branches are kept so
    the class can be compared line by line with the upstream implementation.
    """

    def __init__(self, config: SimpleNamespace) -> None:
        """Build the U-Net from ``config``."""
        super().__init__()
        self.config = config
        self.not_use_tanh = config.not_use_tanh
        self.act = act = nn.SiLU()
        self.nf = nf = config.num_channels_dae
        ch_mult = config.ch_mult
        self.num_res_blocks = num_res_blocks = config.num_res_blocks
        self.attn_resolutions = attn_resolutions = config.attn_resolutions
        dropout = config.dropout
        resamp_with_conv = config.resamp_with_conv
        self.num_resolutions = num_resolutions = len(ch_mult)
        self.all_resolutions = all_resolutions = [
            config.image_size // (2**i) for i in range(num_resolutions)
        ]
        self.conditional = conditional = config.conditional
        fir = config.fir
        fir_kernel = config.fir_kernel
        self.skip_rescale = skip_rescale = config.skip_rescale
        self.resblock_type = resblock_type = config.resblock_type.lower()
        self.progressive = progressive = config.progressive.lower()
        self.progressive_input = progressive_input = config.progressive_input.lower()
        self.embedding_type = embedding_type = config.embedding_type.lower()
        init_scale = 0.0
        assert progressive in ["none", "output_skip", "residual"]
        assert progressive_input in ["none", "input_skip", "residual"]
        assert embedding_type in ["fourier", "positional"]
        combine_method = config.progressive_combine.lower()
        combiner = functools.partial(Combine, method=combine_method)
        modules = []
        if embedding_type == "fourier":
            modules.append(
                GaussianFourierProjection(embedding_size=nf, scale=config.fourier_scale)
            )
            embed_dim = 2 * nf
        elif embedding_type == "positional":
            embed_dim = nf
        else:
            raise ValueError(f"embedding type {embedding_type} unknown.")
        if conditional:
            modules.append(nn.Linear(embed_dim, nf * 4))
            modules[-1].weight.data = default_init()(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)
            modules.append(nn.Linear(nf * 4, nf * 4))
            modules[-1].weight.data = default_init()(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)
        AttnBlockLocal = functools.partial(
            AttnBlockpp, init_scale=init_scale, skip_rescale=skip_rescale
        )
        UpsampleLocal = functools.partial(
            Upsample, with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel
        )
        if progressive == "output_skip":
            self.pyramid_upsample = Upsample(
                fir=fir, fir_kernel=fir_kernel, with_conv=False
            )
        elif progressive == "residual":
            pyramid_upsample = functools.partial(
                Upsample, fir=fir, fir_kernel=fir_kernel, with_conv=True
            )
        DownsampleLocal = functools.partial(
            Downsample, with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel
        )
        if progressive_input == "input_skip":
            self.pyramid_downsample = Downsample(
                fir=fir, fir_kernel=fir_kernel, with_conv=False
            )
        elif progressive_input == "residual":
            pyramid_downsample = functools.partial(
                Downsample, fir=fir, fir_kernel=fir_kernel, with_conv=True
            )
        if resblock_type == "ddpm":
            ResnetBlock = functools.partial(
                NoZResnetBlockDDPMppAdagn,
                act=act,
                dropout=dropout,
                init_scale=init_scale,
                skip_rescale=skip_rescale,
                temb_dim=nf * 4,
            )
        elif resblock_type == "biggan":
            ResnetBlock = functools.partial(
                NoZResnetBlockBigGANppAdagn,
                act=act,
                dropout=dropout,
                fir=fir,
                fir_kernel=fir_kernel,
                init_scale=init_scale,
                skip_rescale=skip_rescale,
                temb_dim=nf * 4,
            )
        elif resblock_type == "biggan_oneadagn":
            ResnetBlock = functools.partial(
                NoZResnetBlockBigGANppAdagnOne,
                act=act,
                dropout=dropout,
                fir=fir,
                fir_kernel=fir_kernel,
                init_scale=init_scale,
                skip_rescale=skip_rescale,
                temb_dim=nf * 4,
            )
        else:
            raise ValueError(f"resblock type {resblock_type} unrecognized.")
        channels = config.num_channels
        if progressive_input != "none":
            input_pyramid_ch = channels
        modules.append(conv3x3(channels, nf))
        hs_c = [nf]
        in_ch = nf
        for i_level in range(num_resolutions):
            for _ in range(num_res_blocks):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
                in_ch = out_ch
                if all_resolutions[i_level] in attn_resolutions:
                    modules.append(AttnBlockLocal(channels=in_ch))
                hs_c.append(in_ch)
            if i_level != num_resolutions - 1:
                if resblock_type == "ddpm":
                    modules.append(DownsampleLocal(in_ch=in_ch))
                else:
                    modules.append(ResnetBlock(down=True, in_ch=in_ch))
                if progressive_input == "input_skip":
                    modules.append(combiner(dim1=input_pyramid_ch, dim2=in_ch))
                    if combine_method == "cat":
                        in_ch *= 2
                elif progressive_input == "residual":
                    modules.append(
                        pyramid_downsample(in_ch=input_pyramid_ch, out_ch=in_ch)
                    )
                    input_pyramid_ch = in_ch
                hs_c.append(in_ch)
        in_ch = hs_c[-1]
        modules.append(ResnetBlock(in_ch=in_ch))
        modules.append(AttnBlockLocal(channels=in_ch))
        modules.append(ResnetBlock(in_ch=in_ch))
        pyramid_ch = 0
        for i_level in reversed(range(num_resolutions)):
            for _ in range(num_res_blocks + 1):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(), out_ch=out_ch))
                in_ch = out_ch
            if all_resolutions[i_level] in attn_resolutions:
                modules.append(AttnBlockLocal(channels=in_ch))
            if progressive != "none":
                if i_level == num_resolutions - 1:
                    if progressive == "output_skip":
                        modules.append(
                            nn.GroupNorm(
                                num_groups=min(in_ch // 4, 32),
                                num_channels=in_ch,
                                eps=1e-6,
                            )
                        )
                        modules.append(conv3x3(in_ch, channels, init_scale=init_scale))
                        pyramid_ch = channels
                    elif progressive == "residual":
                        modules.append(
                            nn.GroupNorm(
                                num_groups=min(in_ch // 4, 32),
                                num_channels=in_ch,
                                eps=1e-6,
                            )
                        )
                        modules.append(conv3x3(in_ch, in_ch, bias=True))
                        pyramid_ch = in_ch
                    else:
                        raise ValueError(f"{progressive} is not a valid name.")
                else:
                    if progressive == "output_skip":
                        modules.append(
                            nn.GroupNorm(
                                num_groups=min(in_ch // 4, 32),
                                num_channels=in_ch,
                                eps=1e-6,
                            )
                        )
                        modules.append(
                            conv3x3(in_ch, channels, bias=True, init_scale=init_scale)
                        )
                        pyramid_ch = channels
                    elif progressive == "residual":
                        modules.append(pyramid_upsample(in_ch=pyramid_ch, out_ch=in_ch))
                        pyramid_ch = in_ch
                    else:
                        raise ValueError(f"{progressive} is not a valid name")
            if i_level != 0:
                if resblock_type == "ddpm":
                    modules.append(UpsampleLocal(in_ch=in_ch))
                else:
                    modules.append(ResnetBlock(in_ch=in_ch, up=True))
        assert not hs_c
        if progressive != "output_skip":
            modules.append(
                nn.GroupNorm(
                    num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6
                )
            )
            modules.append(conv3x3(in_ch, 1, init_scale=init_scale))
        self.all_modules = nn.ModuleList(modules)

    def forward(self, x: torch.Tensor, time_cond: torch.Tensor) -> torch.Tensor:
        """Predict ``x_0`` from ``x`` (noisy image + edges) at ``time_cond``."""
        modules = self.all_modules
        m_idx = 0
        if self.embedding_type == "fourier":
            used_sigmas = time_cond
            temb = modules[m_idx](torch.log(used_sigmas))
            m_idx += 1
        elif self.embedding_type == "positional":
            timesteps = time_cond
            temb = get_timestep_embedding(timesteps, self.nf)
        else:
            raise ValueError(f"embedding type {self.embedding_type} unknown.")
        if self.conditional:
            temb = modules[m_idx](temb)
            m_idx += 1
            temb = modules[m_idx](self.act(temb))
            m_idx += 1
        else:
            temb = None
        if not self.config.centered:
            x = 2 * x - 1.0
        input_pyramid = None
        if self.progressive_input != "none":
            input_pyramid = x
        hs = [modules[m_idx](x)]
        m_idx += 1
        for i_level in range(self.num_resolutions):
            for _ in range(self.num_res_blocks):
                h = modules[m_idx](hs[-1], temb)
                m_idx += 1
                if h.shape[-1] in self.attn_resolutions:
                    h = modules[m_idx](h)
                    m_idx += 1
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                if self.resblock_type == "ddpm":
                    h = modules[m_idx](hs[-1])
                    m_idx += 1
                else:
                    h = modules[m_idx](hs[-1], temb)
                    m_idx += 1
                if self.progressive_input == "input_skip":
                    input_pyramid = self.pyramid_downsample(input_pyramid)
                    h = modules[m_idx](input_pyramid, h)
                    m_idx += 1
                elif self.progressive_input == "residual":
                    input_pyramid = modules[m_idx](input_pyramid)
                    m_idx += 1
                    if self.skip_rescale:
                        input_pyramid = (input_pyramid + h) / np.sqrt(2.0)
                    else:
                        input_pyramid = input_pyramid + h
                    h = input_pyramid
                hs.append(h)
        h = hs[-1]
        h = modules[m_idx](h, temb)
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1
        h = modules[m_idx](h, temb)
        m_idx += 1
        pyramid = None
        for i_level in reversed(range(self.num_resolutions)):
            for _ in range(self.num_res_blocks + 1):
                h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb)
                m_idx += 1
            if h.shape[-1] in self.attn_resolutions:
                h = modules[m_idx](h)
                m_idx += 1
            if self.progressive != "none":
                if i_level == self.num_resolutions - 1:
                    if self.progressive in ("output_skip", "residual"):
                        pyramid = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                    else:
                        raise ValueError(f"{self.progressive} is not a valid name.")
                else:
                    if self.progressive == "output_skip":
                        pyramid = self.pyramid_upsample(pyramid)
                        pyramid_h = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid_h = modules[m_idx](pyramid_h)
                        m_idx += 1
                        pyramid = pyramid + pyramid_h
                    elif self.progressive == "residual":
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                        if self.skip_rescale:
                            pyramid = (pyramid + h) / np.sqrt(2.0)
                        else:
                            pyramid = pyramid + h
                        h = pyramid
                    else:
                        raise ValueError(f"{self.progressive} is not a valid name")
            if i_level != 0:
                if self.resblock_type == "ddpm":
                    h = modules[m_idx](h)
                    m_idx += 1
                else:
                    h = modules[m_idx](h, temb)
                    m_idx += 1
        assert not hs
        if self.progressive == "output_skip":
            h = pyramid
        else:
            h = self.act(modules[m_idx](h))
            m_idx += 1
            h = modules[m_idx](h)
            m_idx += 1
        assert m_idx == len(modules)
        if not self.not_use_tanh:
            return torch.tanh(h)
        return h


# ============================================================================
# FGDM configuration and diffusion process (FGDM-main/main.py)
# ============================================================================
def build_fgdm_config() -> SimpleNamespace:
    """Generator and diffusion settings (FGDM-main parser defaults)."""
    return SimpleNamespace(
        image_size=PATCH_SIZE,
        num_channels=2,  # generator input: concat(x_{t+1}, edge map)
        centered=True,
        use_geometric=False,
        num_channels_dae=128,
        n_mlp=3,
        ch_mult=(1, 2, 2, 2),
        num_res_blocks=2,
        attn_resolutions=(16,),
        dropout=0.0,
        resamp_with_conv=True,
        conditional=True,
        fir=True,
        fir_kernel=(1, 3, 3, 1),
        skip_rescale=True,
        resblock_type="biggan",
        progressive="none",
        progressive_input="residual",
        progressive_combine="sum",
        embedding_type="positional",
        fourier_scale=16.0,
        not_use_tanh=False,
        num_timesteps=TIMESTEPS,
        beta_min=BETA_MIN,
        beta_max=BETA_MAX,
    )


def _var_func_vp(t: torch.Tensor, beta_min: float, beta_max: float) -> torch.Tensor:
    """Variance of the VP-SDE marginal at continuous time ``t``."""
    log_mean_coeff = -0.25 * t**2 * (beta_max - beta_min) - 0.5 * t * beta_min
    return 1.0 - torch.exp(2.0 * log_mean_coeff)


def _var_func_geometric(
    t: torch.Tensor, beta_min: float, beta_max: float
) -> torch.Tensor:
    """Geometric variance schedule."""
    return beta_min * ((beta_max / beta_min) ** t)


def _extract(input: torch.Tensor, t: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """Gather ``input[t]`` and reshape it to broadcast against ``shape``."""
    out = torch.gather(input, 0, t.long())
    return out.reshape(shape[0], *([1] * (len(shape) - 1)))


def get_sigma_schedule(
    args: SimpleNamespace, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Discrete noise schedule.

    Returns:
        ``(sigmas, a_s, betas)``, each of length ``num_timesteps + 1``.
    """
    t = np.arange(0, args.num_timesteps + 1, dtype=np.float64)
    t = t / args.num_timesteps
    t = torch.from_numpy(t).to(device) * (1.0 - 1e-3) + 1e-3
    if getattr(args, "use_geometric", False):
        var = _var_func_geometric(t, args.beta_min, args.beta_max)
    else:
        var = _var_func_vp(t, args.beta_min, args.beta_max)
    alpha_bars = 1.0 - var
    betas = 1 - alpha_bars[1:] / alpha_bars[:-1]
    betas = torch.cat(
        (torch.tensor([1e-8], dtype=torch.float32, device=device), betas.float())
    )
    sigmas = betas.sqrt()
    a_s = torch.sqrt(1 - betas)
    return sigmas, a_s, betas


class DiffusionCoefficients:
    """Coefficients of the forward process ``q(x_t | x_0)``."""

    def __init__(self, args: SimpleNamespace, device: torch.device) -> None:
        """Precompute the cumulative coefficients on ``device``."""
        self.sigmas, self.a_s, _ = get_sigma_schedule(args, device=device)
        self.a_s_cum = torch.cumprod(self.a_s.cpu(), dim=0).to(device)
        self.sigmas_cum = torch.sqrt(1 - self.a_s_cum**2).to(device)
        self.a_s_prev = self.a_s.clone()
        self.a_s_prev[-1] = 1


def q_sample(
    coeff: DiffusionCoefficients,
    x_start: torch.Tensor,
    t: torch.Tensor,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sample ``x_t ~ q(x_t | x_0)``."""
    if noise is None:
        noise = torch.randn_like(x_start)
    return (
        _extract(coeff.a_s_cum, t, x_start.shape) * x_start
        + _extract(coeff.sigmas_cum, t, x_start.shape) * noise
    )


def q_sample_pairs(
    coeff: DiffusionCoefficients, x_start: torch.Tensor, t: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a pair ``(x_t, x_{t+1})`` from the forward process."""
    noise = torch.randn_like(x_start)
    x_t = q_sample(coeff, x_start, t)
    x_t_plus_one = (
        _extract(coeff.a_s, t + 1, x_start.shape) * x_t
        + _extract(coeff.sigmas, t + 1, x_start.shape) * noise
    )
    return x_t, x_t_plus_one


class PosteriorCoefficients:
    """Coefficients of the posterior ``q(x_{t-1} | x_t, x_0)``."""

    def __init__(self, args: SimpleNamespace, device: torch.device) -> None:
        """Precompute the posterior mean and variance coefficients."""
        _, _, betas = get_sigma_schedule(args, device=device)
        self.betas = betas.float()[1:]
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.alphas_cumprod_prev = torch.cat(
            (
                torch.tensor([1.0], dtype=torch.float32, device=device),
                self.alphas_cumprod[:-1],
            ),
            0,
        )
        self.posterior_variance = (
            self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        )
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.rsqrt(self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod - 1)
        self.posterior_mean_coef1 = (
            self.betas
            * torch.sqrt(self.alphas_cumprod_prev)
            / (1 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1 - self.alphas_cumprod_prev)
            * torch.sqrt(self.alphas)
            / (1 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            self.posterior_variance.clamp(min=1e-20)
        )


def sample_posterior(
    coefficients: PosteriorCoefficients,
    x_0: torch.Tensor,
    x_t: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """Sample ``x_{t-1} ~ q(x_{t-1} | x_t, x_0)`` (no noise at ``t == 0``)."""
    mean = (
        _extract(coefficients.posterior_mean_coef1, t, x_t.shape) * x_0
        + _extract(coefficients.posterior_mean_coef2, t, x_t.shape) * x_t
    )
    log_var = _extract(coefficients.posterior_log_variance_clipped, t, x_t.shape)
    noise = torch.randn_like(x_t)
    nonzero_mask = (1 - (t == 0).type(torch.float32))[:, None, None, None]
    return mean + nonzero_mask * torch.exp(0.5 * log_var) * noise


@torch.no_grad()
def sample_from_model(
    coefficients: PosteriorCoefficients,
    generator: nn.Module,
    n_time: int,
    x_init: torch.Tensor,
    edge_data: torch.Tensor,
) -> torch.Tensor:
    """Run the reverse process from ``x_init`` for ``n_time`` steps.

    Args:
        coefficients: Posterior coefficients.
        generator: Network predicting ``x_0``.
        n_time: Number of reverse steps.
        x_init: Starting image ``x_{n_time}``.
        edge_data: Edge map that conditions every step.

    Returns:
        The denoised image, clipped to ``[0, 1]``.
    """
    x = x_init
    generator.eval()
    for i in reversed(range(n_time)):
        t = torch.full((x.size(0),), i, dtype=torch.int64, device=x.device)
        x_0 = generator(torch.cat((x.detach(), edge_data.detach()), dim=1), t)
        x = sample_posterior(coefficients, x_0, x, t).detach()
    return x.clamp(0.0, 1.0)


# ============================================================================
# Data
# ============================================================================
def _fd_npy_cache_path(path: str, cache_dir: Path) -> Path:
    """Location of the decoded-DICOM cache of ``path``.

    ``os.path.abspath`` (not ``Path.resolve``) keeps the hashes, and therefore
    existing caches, identical to the original implementation.
    """
    digest = hashlib.sha1(os.path.abspath(path).encode("utf-8")).hexdigest()
    return Path(cache_dir) / f"{digest}.npy"


def load_target_array_cached(path: str, cache_dir: Path | None) -> np.ndarray:
    """Load a full-dose target in ``[0, 1]``, caching decoded DICOMs as ``.npy``.

    Decoding with pydicom is CPU-bound and would otherwise run on every
    access. The cache lives in a dedicated folder that the file listers never
    scan, so it cannot pollute the LD/FD pairing. Results are identical to
    decoding on the fly.

    Args:
        path: ``.IMA``/``.dcm`` DICOM file or ``.npy`` in HU.
        cache_dir: Cache folder, or ``None`` to disable caching.
    """
    if not path.lower().endswith(DICOM_SUFFIXES):
        return normalize(np.load(path).astype(np.float32))
    cache = _fd_npy_cache_path(path, cache_dir) if cache_dir is not None else None
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
        with open(tmp, "wb") as fh:  # a file handle stops np.save adding ".npy"
            np.save(fh, arr)
        os.replace(tmp, cache)
    return arr


def _sobel_edges(img_u8: np.ndarray, bilateral: int) -> np.ndarray:
    """Bilateral-filtered Sobel magnitude, with a NumPy fallback without OpenCV.

    The fallback skips the bilateral filter, so its edges differ from the
    OpenCV ones.
    """
    try:
        lab = cv2.bilateralFilter(img_u8, BILATERAL_DIAMETER, bilateral, bilateral)
        sx = cv2.Sobel(lab, cv2.CV_16S, 1, 0)
        sy = cv2.Sobel(lab, cv2.CV_16S, 0, 1)
        return cv2.addWeighted(
            cv2.convertScaleAbs(sx), 0.5, cv2.convertScaleAbs(sy), 0.5, 0
        )
    except Exception:
        lab = img_u8.astype(np.float32)
        pad = np.pad(lab, 1, mode="edge")
        sx = (
            pad[:-2, 2:]
            + 2 * pad[1:-1, 2:]
            + pad[2:, 2:]
            - pad[:-2, :-2]
            - 2 * pad[1:-1, :-2]
            - pad[2:, :-2]
        )
        sy = (
            pad[2:, :-2]
            + 2 * pad[2:, 1:-1]
            + pad[2:, 2:]
            - pad[:-2, :-2]
            - 2 * pad[:-2, 1:-1]
            - pad[:-2, 2:]
        )
        return 0.5 * np.abs(sx) + 0.5 * np.abs(sy)


def edge_map(
    img: np.ndarray,
    training: bool = True,
    threshold: int | None = None,
    bilateral: int | None = None,
) -> np.ndarray:
    """High-pass edge map ``H_eta`` of a normalised image (paper Eq. 25).

    Args:
        img: Image in ``[0, 1]``.
        training: Draw eta and the bilateral strength at random (training) or
            use the fixed test values.
        threshold: Sobel threshold eta; overrides ``training``.
        bilateral: Bilateral-filter strength; overrides ``training``.

    Returns:
        The thresholded edge magnitude rescaled to ``[0, 1]``.
    """
    img01 = np.clip(img, 0.0, 1.0).astype(np.float32)
    img_u8 = np.clip(img01 * 255.0, 0, 255).astype(np.uint8)
    if threshold is None:
        threshold = (
            int(np.random.randint(1, EDGE_ETA_MAX + 1)) if training else TEST_ETA
        )
    if bilateral is None:
        bilateral = (
            int(np.random.randint(1, EDGE_BILATERAL_MAX + 1))
            if training
            else TEST_BILATERAL
        )
    edge = _sobel_edges(img_u8, bilateral).astype(np.float32)
    edge[edge < threshold] = 0.0
    return ((edge - edge.min() + 1e-12) / (edge.max() - edge.min() + 1e-8)).astype(
        np.float32
    )


def source_edge(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Edge map of the low-dose input used at inference (Algorithm 1).

    The edge map returned by the loader belongs to the full-dose target and is
    only valid for training; at inference the edges come from the input.

    Args:
        x: Low-dose batch ``(N, 1, H, W)`` in ``[0, 1]``.
        device: Device of the returned tensor.
    """
    x_np = x.detach().cpu().numpy()
    edges = np.stack(
        [
            edge_map(
                x_np[b, 0],
                training=False,
                threshold=TEST_ETA,
                bilateral=TEST_BILATERAL,
            )
            for b in range(x_np.shape[0])
        ]
    )
    return torch.from_numpy(edges).float().unsqueeze(1).to(device)


def random_patch_triples(
    x: np.ndarray,
    y: np.ndarray,
    e: np.ndarray,
    patch_size: int,
    patch_n: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Crop aligned random patches from an input/target/edge triple.

    Args:
        x: Input image ``(H, W)``.
        y: Target image ``(H, W)``.
        e: Edge map of the target ``(H, W)``.
        patch_size: Patch side; clipped to the image size.
        patch_n: Number of patches (``None`` or 0 means one).

    Returns:
        Stacked patches, ``(patch_n, ps, ps)`` each.
    """
    h = min(x.shape[0], y.shape[0], e.shape[0])
    w = min(x.shape[1], y.shape[1], e.shape[1])
    ps = min(int(patch_size), h, w)
    xs, ys, es = [], [], []
    for _ in range(int(patch_n or 1)):
        top = np.random.randint(0, h - ps + 1) if h > ps else 0
        left = np.random.randint(0, w - ps + 1) if w > ps else 0
        xs.append(x[top : top + ps, left : left + ps])
        ys.append(y[top : top + ps, left : left + ps])
        es.append(e[top : top + ps, left : left + ps])
    return np.stack(xs), np.stack(ys), np.stack(es)


class FGDMDataset(ULDCTDataset):
    """ULDCT dataset returning ``(input, target, target edge map)`` triples.

    Only DICOM files are taken as full-dose targets (``dicom_only``), because
    any ``.npy`` file next to them is a decoded cache that would shift the
    pairing.
    """

    def __init__(
        self,
        split: str,
        data_root: Path,
        patch_size: int | None = None,
        patch_n: int | None = None,
        fd_cache_dir: Path | None = None,
    ) -> None:
        """Discover and pair the files of ``split``.

        Args:
            split: Dataset split.
            data_root: Dataset root.
            patch_size: Training patch side, or ``None`` for full slices.
            patch_n: Patches per slice.
            fd_cache_dir: Decoded-DICOM cache folder, or ``None`` to disable.
        """
        super().__init__(
            split,
            data_root,
            patch_size=patch_size,
            patch_n=patch_n,
            dicom_only=True,
            warn_count_mismatch=True,
        )
        self.fd_cache_dir = fd_cache_dir

    def load_pair(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Load the normalised input and (cached) target slices at ``idx``."""
        x = load_input_array(self.inputs[idx])
        y = load_target_array_cached(self.targets[idx], self.fd_cache_dir)
        return x, y

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the triple, or random patches of it when patching."""
        x, y = self.load_pair(idx)
        e = edge_map(y, training=self.split == TRAIN_SPLIT)
        if self.patch_size:
            return random_patch_triples(x, y, e, self.patch_size, self.patch_n)
        return x, y, e


def _seed_worker(worker_id: int) -> None:
    """Seed NumPy and ``random`` per DataLoader worker.

    PyTorch reseeds ``torch`` in each worker but not NumPy, so without this
    the random crops and edge thresholds would be correlated across workers.
    """
    seed = (torch.initial_seed() + worker_id) % (2**32)
    np.random.seed(seed)
    random.seed(seed)


# ============================================================================
# Evaluation
# ============================================================================
def _to_hu(img: torch.Tensor) -> torch.Tensor:
    """Normalised image -> HU, clipped to the evaluation window (on CPU)."""
    return truncate(denormalize(img.cpu().detach()))


def _measure(pred_hu: torch.Tensor, target_hu: torch.Tensor) -> tuple[float, ...]:
    """PSNR, SSIM (with FGDM's offset) and RMSE of a prediction."""
    return (
        compute_psnr(pred_hu, target_hu, DATA_RANGE),
        compute_ssim(pred_hu, target_hu, DATA_RANGE, offset=TRUNC_MIN),
        compute_rmse(pred_hu, target_hu),
    )


def safe_train_metrics(
    pred_norm: torch.Tensor, target_norm: torch.Tensor
) -> tuple[float, float, float]:
    """PSNR/SSIM/RMSE of a training batch; SSIM is clipped to ``[-1, 1]``."""
    psnr, ssim, rmse = _measure(_to_hu(pred_norm), _to_hu(target_norm))
    return psnr, max(min(ssim, 1.0), -1.0), rmse


def translate(
    netG: nn.Module,
    x: torch.Tensor,
    device: torch.device,
    coeff: DiffusionCoefficients,
    pos_coeff: PosteriorCoefficients,
) -> torch.Tensor:
    """Zero-shot LD -> FD translation (paper Algorithm 1).

    The low-dose input is forward-diffused to step ~T (``L_~T``) and denoised
    in reverse, conditioned on its own Sobel edges (``H_eta``).

    Args:
        netG: Generator.
        x: Low-dose batch ``(N, 1, H, W)`` in ``[0, 1]``.
        device: Inference device.
        coeff: Forward-process coefficients.
        pos_coeff: Posterior coefficients.

    Returns:
        The denoised batch in ``[0, 1]``.
    """
    h_eta = source_edge(x, device)
    t = torch.full((x.size(0),), TEST_TILDE_T - 1, dtype=torch.int64, device=device)
    _, l_tt = q_sample_pairs(coeff, x, t)
    return sample_from_model(pos_coeff, netG, TEST_TILDE_T, l_tt, h_eta)


@torch.no_grad()
def fgdm_eval(
    netG: nn.Module,
    loader: DataLoader,
    device: torch.device,
    coeff: DiffusionCoefficients,
    pos_coeff: PosteriorCoefficients,
    max_slices: int = VAL_MAX_SLICES,
) -> tuple[float, float, float]:
    """Mean PSNR/SSIM/RMSE of the translation on the first validation slices.

    Leaves ``netG`` in eval mode.
    """
    netG.eval()
    psnr_sum, ssim_sum, rmse_sum, n = 0.0, 0.0, 0.0, 0
    for x, y, _ in loader:
        if n >= max_slices:
            break
        x = x.float().to(device).unsqueeze(1)
        y = y.float().to(device).unsqueeze(1)
        pred = translate(netG, x, device, coeff, pos_coeff)
        h, w = pred.shape[-2:]
        psnr, ssim, rmse = _measure(_to_hu(pred[0, 0]), _to_hu(y[0, 0, :h, :w]))
        psnr_sum += psnr
        ssim_sum += ssim
        rmse_sum += rmse
        n += 1
    if n == 0:
        return 0.0, 0.0, 0.0
    return psnr_sum / n, ssim_sum / n, rmse_sum / n


# ============================================================================
# Checkpoints
# ============================================================================
def _uncompiled(module: nn.Module) -> nn.Module:
    """Module underneath ``torch.compile``.

    Saving and loading through it keeps the raw state_dict keys (without the
    ``_orig_mod.`` prefix), so checkpoints work with and without compilation.
    """
    return getattr(module, "_orig_mod", module)


def load_fgdm_checkpoint(
    path: Path,
    *,
    netG: nn.Module,
    netD: nn.Module | None = None,
    optimizerG: optim.Optimizer | None = None,
    optimizerD: optim.Optimizer | None = None,
    schedulerG: Any | None = None,
    schedulerD: Any | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any] | None:
    """Restore generator, discriminator, optimizers and schedulers.

    Returns:
        The checkpoint dictionary, or ``None`` when none was found or it is
        incompatible with the current networks (a warning is logged; modules
        loaded before the failure keep the loaded weights).
    """
    path = resolve_checkpoint_path(path, prefer="last")
    if not path.exists():
        return None
    ckpt = torch.load(path, map_location=map_location)
    try:
        _uncompiled(netG).load_state_dict(ckpt["model"], strict=True)
        if netD is not None and ckpt.get("netD") is not None:
            _uncompiled(netD).load_state_dict(ckpt["netD"], strict=True)
        for obj, key in (
            (optimizerG, "optimizerG"),
            (optimizerD, "optimizerD"),
            (schedulerG, "schedulerG"),
            (schedulerD, "schedulerD"),
        ):
            if obj is not None and ckpt.get(key) is not None:
                obj.load_state_dict(ckpt[key])
        return ckpt
    except RuntimeError as exc:
        logger.warning("Incompatible FGDM checkpoint ignored: %s", exc)
        return None


def save_fgdm_checkpoint(
    path: Path,
    *,
    netG: nn.Module,
    netD: nn.Module,
    optimizerG: optim.Optimizer,
    optimizerD: optim.Optimizer,
    schedulerG: Any,
    schedulerD: Any,
    epoch: int,
    step: int,
    lr: float,
    loss: float,
    psnr: float,
    ssim: float,
    rmse_hu: float,
) -> None:
    """Save both networks, their optimizers/schedulers, progress and metrics."""
    torch.save(
        {
            "model": _uncompiled(netG).state_dict(),
            "netD": _uncompiled(netD).state_dict(),
            "optimizerG": optimizerG.state_dict(),
            "optimizerD": optimizerD.state_dict(),
            "schedulerG": schedulerG.state_dict(),
            "schedulerD": schedulerD.state_dict(),
            "epoch": int(epoch),
            "step": int(step),
            "lr": float(lr),
            "loss": float(loss),
            "psnr": float(psnr),
            "ssim": float(ssim),
            "rmse_hu": float(rmse_hu),
        },
        path,
    )


# ============================================================================
# Training and testing
# ============================================================================
def train(
    netG: nn.Module,
    netD: nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: RunConfig,
    coeff: DiffusionCoefficients,
    pos_coeff: PosteriorCoefficients,
    val_loader: DataLoader | None = None,
) -> None:
    """Adversarial diffusion training (FGDM-main), resuming if possible.

    Every ``SAVE_CKPT_EVERY`` epochs (and at the last one) the generator is
    validated, the last checkpoint is saved and the best-PSNR one is updated.

    Args:
        netG: Generator.
        netD: Time-dependent discriminator.
        loader: Training loader of ``(input, target, edge)`` patches.
        device: Training device.
        cfg: Run configuration.
        coeff: Forward-process coefficients.
        pos_coeff: Posterior coefficients.
        val_loader: Loader for periodic validation.
    """
    netG.train()
    netD.train()
    optimizerD = optim.Adam(netD.parameters(), lr=LR_D, betas=(ADAM_BETA1, ADAM_BETA2))
    optimizerG = optim.Adam(netG.parameters(), lr=LR, betas=(ADAM_BETA1, ADAM_BETA2))
    schedulerG = optim.lr_scheduler.CosineAnnealingLR(
        optimizerG, NUM_EPOCHS, eta_min=LR_MIN
    )
    schedulerD = optim.lr_scheduler.CosineAnnealingLR(
        optimizerD, NUM_EPOCHS, eta_min=LR_MIN
    )

    ckpt = load_fgdm_checkpoint(
        cfg.last_ckpt,
        netG=netG,
        netD=netD,
        optimizerG=optimizerG,
        optimizerD=optimizerD,
        schedulerG=schedulerG,
        schedulerD=schedulerD,
    )
    # Epochs run from 0 to NUM_EPOCHS inclusive, as in the original code.
    start_epoch = (ckpt["epoch"] + 1) if ckpt else 0
    step = ckpt["step"] if ckpt else 0
    best_psnr = best_psnr_so_far(cfg.best_ckpt)
    losses = []
    t0 = time.time()

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        # The low-dose input is not used for training: FGDM learns the
        # full-dose distribution only and translates zero-shot at test time.
        for _, real_data, edge_data in loader:
            step += 1
            real_data = real_data.float().to(device)
            edge_data = edge_data.float().to(device)
            if real_data.dim() == 4:
                real_data = real_data.view(-1, 1, PATCH_SIZE, PATCH_SIZE)
                edge_data = edge_data.view(-1, 1, PATCH_SIZE, PATCH_SIZE)
            else:
                real_data = real_data.unsqueeze(1)
                edge_data = edge_data.unsqueeze(1)

            # Discriminator step with R1 penalty on real samples.
            for param in netD.parameters():
                param.requires_grad = True
            netD.zero_grad(set_to_none=True)
            t = torch.randint(0, TIMESTEPS, (real_data.size(0),), device=device)
            x_t, x_tp1 = q_sample_pairs(coeff, real_data, t)
            x_t.requires_grad = True
            d_real = netD(x_t, t, x_tp1.detach()).view(-1)
            errD_real = F.softplus(-d_real).mean()
            errD_real.backward(retain_graph=True)
            grad_real = torch.autograd.grad(
                outputs=d_real.sum(), inputs=x_t, create_graph=True
            )[0]
            grad_penalty = (
                grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
            ).mean()
            grad_penalty = R1_GAMMA / 2 * grad_penalty
            grad_penalty.backward()
            x_0_predict = netG(
                torch.cat((x_tp1.detach(), edge_data.detach()), dim=1), t
            )
            x_pos_sample = sample_posterior(pos_coeff, x_0_predict, x_tp1, t)
            d_fake = netD(x_pos_sample, t, x_tp1.detach()).view(-1)
            errD_fake = F.softplus(d_fake).mean()
            errD_fake.backward()
            errD = errD_real + errD_fake
            optimizerD.step()

            # Generator step.
            for param in netD.parameters():
                param.requires_grad = False
            netG.zero_grad(set_to_none=True)
            x_t, x_tp1 = q_sample_pairs(coeff, real_data, t)
            x_0_predict = netG(
                torch.cat((x_tp1.detach(), edge_data.detach()), dim=1), t
            )
            x_pos_sample = sample_posterior(pos_coeff, x_0_predict, x_tp1, t)
            output = netD(x_pos_sample, t, x_tp1.detach()).view(-1)
            errG = F.softplus(-output).mean()
            errG.backward()
            optimizerG.step()
            losses.append(errG.item())

            if step % PRINT_ITERS == 0:
                with torch.no_grad():
                    train_psnr, train_ssim, train_rmse = safe_train_metrics(
                        x_0_predict.clamp(0.0, 1.0), real_data
                    )
                logger.info(
                    "epoch %d/%d  iter %d  G %.6f  D %.6f  PSNR %6.2f  SSIM %.4f  "
                    "RMSE %6.2f HU  (%.0fs)",
                    epoch,
                    NUM_EPOCHS,
                    step,
                    errG.item(),
                    errD.item(),
                    train_psnr,
                    train_ssim,
                    train_rmse,
                    time.time() - t0,
                )

        schedulerG.step()
        schedulerD.step()
        if epoch % SAVE_CKPT_EVERY == 0 or epoch == NUM_EPOCHS:
            # Validate on the validation split; the test split is only used by
            # the final test.
            psnr, ssim, rmse = (
                fgdm_eval(netG, val_loader, device, coeff, pos_coeff)
                if val_loader is not None
                else (0.0, 0.0, 0.0)
            )
            netG.train()  # fgdm_eval leaves netG in eval mode
            state = dict(
                netG=netG,
                netD=netD,
                optimizerG=optimizerG,
                optimizerD=optimizerD,
                schedulerG=schedulerG,
                schedulerD=schedulerD,
                epoch=epoch,
                step=step,
                lr=optimizerG.param_groups[0]["lr"],
                loss=losses[-1] if losses else 0.0,
                psnr=psnr,
                ssim=ssim,
                rmse_hu=rmse,
            )
            save_fgdm_checkpoint(cfg.last_ckpt, **state)
            if psnr > best_psnr:
                best_psnr = psnr
                save_fgdm_checkpoint(cfg.best_ckpt, **state)
            np.save(cfg.losses_path, np.array(losses))
            logger.info(
                "  saved (epoch=%d step=%d eval PSNR %6.2f SSIM %.4f RMSE %6.2f HU) "
                "best_psnr=%.3f",
                epoch,
                step,
                psnr,
                ssim,
                rmse,
                best_psnr,
            )


def test(
    netG: nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: RunConfig,
    coeff: DiffusionCoefficients,
    pos_coeff: PosteriorCoefficients,
) -> None:
    """Translate the full test split and write metrics and figures.

    Args:
        netG: Generator.
        loader: Test loader with batch size 1.
        device: Inference device.
        cfg: Run configuration.
        coeff: Forward-process coefficients.
        pos_coeff: Posterior coefficients.
    """
    netG.eval()

    def predict(
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y, _ = batch
        x = x.float().to(device).unsqueeze(1)
        y = y.float().to(device).unsqueeze(1)
        pred = translate(netG, x, device, coeff, pos_coeff)
        h, w = pred.shape[-2:]
        return _to_hu(x[0, 0, :h, :w]), _to_hu(y[0, 0, :h, :w]), _to_hu(pred[0, 0])

    run_test(loader, predict, cfg, ssim_offset=TRUNC_MIN)


def main() -> None:
    """Train FGDM on the training split, then test the final generator."""
    parser = build_arg_parser(MODEL_NAME, description=__doc__)
    parser.add_argument(
        "--fd-cache-dir",
        type=Path,
        default=DEFAULT_FD_CACHE_DIR,
        help=f"Decoded full-dose DICOM cache (default: {DEFAULT_FD_CACHE_DIR}).",
    )
    parser.add_argument(
        "--no-fd-cache", action="store_true", help="Decode DICOM targets every time."
    )
    parser.add_argument(
        "--no-compile", action="store_true", help="Do not torch.compile the generator."
    )
    cfg, args = parse_run_config(parser, MODEL_NAME)
    fd_cache_dir = None if args.no_fd_cache else args.fd_cache_dir

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = get_device()
    logger.info(
        "[%s] dose=%s device=%s out=%s", MODEL_NAME, cfg.dose, device, cfg.output_dir
    )

    def make_dataset(split: str, **kwargs: Any) -> FGDMDataset:
        return FGDMDataset(split, cfg.data_root, fd_cache_dir=fd_cache_dir, **kwargs)

    train_ds = make_dataset(TRAIN_SPLIT, patch_size=PATCH_SIZE, patch_n=PATCH_N)
    test_ds = make_dataset(TEST_SPLIT)
    val_ds = optional_split(lambda: make_dataset(VAL_SPLIT), VAL_SPLIT)
    logger.info(
        "train files: %d | val files: %d | test files: %d",
        len(train_ds),
        len(val_ds) if val_ds is not None else 0,
        len(test_ds),
    )

    workers = cfg.num_workers
    loader_kwargs = dict(
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        worker_init_fn=_seed_worker if workers > 0 else None,
    )
    prefetch = {"prefetch_factor": PREFETCH_FACTOR} if workers > 0 else {}
    # drop_last: a final batch of size 1 would trigger a torch.compile
    # recompilation and break the discriminator's minibatch-stddev feature.
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
        **prefetch,
    )
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, **loader_kwargs)
    val_loader = (
        DataLoader(val_ds, batch_size=1, shuffle=False, **loader_kwargs)
        if val_ds is not None
        else test_loader
    )

    config = build_fgdm_config()
    netG = NCSNppNoZ(config).to(device)
    netD = Discriminator_large(
        nc=2, ngf=NGF, t_emb_dim=T_EMB_DIM, act=nn.LeakyReLU(D_LEAKY_SLOPE)
    ).to(device)
    logger.info("discriminator=%s", type(netD).__name__)

    # Only the generator is compiled: the R1 penalty needs double backward
    # through the discriminator, which torch.compile does not support.
    if not args.no_compile and hasattr(torch, "compile") and device.type == "cuda":
        try:
            netG = torch.compile(netG)
            logger.info("netG compiled with torch.compile")
        except Exception as exc:
            logger.warning("torch.compile unavailable (%s); running eagerly", exc)
    coeff = DiffusionCoefficients(config, device)
    pos_coeff = PosteriorCoefficients(config, device)

    train(
        netG, netD, train_loader, device, cfg, coeff, pos_coeff, val_loader=val_loader
    )
    # FGDM is tested with the in-memory weights of the last epoch, not with the
    # best checkpoint (unlike the other models).
    test(netG, test_loader, device, cfg, coeff, pos_coeff)


if __name__ == "__main__":
    main()
