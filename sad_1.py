"""SAD-1 (Du et al., 2024) trained and evaluated on the ULDCT dataset.

From-scratch reimplementation of "Structure-aware diffusion for low-dose CT
imaging", Phys. Med. Biol. 69 (2024) 155008, doi:10.1088/1361-6560/ad5d47.
No official code is public. Components (paper Sec. 2):

1. Schrödinger bridge between the LDCT (X1) and NDCT (X0) distributions with
   f := 0 (eqs. 4-8), using the analytic posterior q(X_t | X_0, X_1).
2. Structure prior network G: the official PiDiNet (Su et al., 2021),
   pretrained on BSDS500 and frozen. Its multiscale encoder features are the
   prompts P_s (eqs. 9-10) and its fused edge map is S.
3. Guided filter module (GFM, eqs. 11-15) in each U-Net decoder stage: filters
   the encoder feature E_s guided by P_s.
4. Implicit conditional representation (ICR, eq. 16): 2-layer MLP with hidden
   size 128 on (features, S, coordinates).
5. Iterative prompt refinement (Sec. 2.3): during sampling G receives the
   latest prediction.

Training (Sec. 3.2): Adam, lr 5e-4, 300k iterations, T = 1000 with quadratic
discretisation; inference with 1 step (SAD-1) or 5 steps (SAD-5).

Caveats: the beta schedule and the exact sigma_t of eq. 17 are not given in the
paper (sigma_t = sqrt(sigma^2_t) with a quadratic beta schedule is used), and
the data is ULDCT 5%/10% with AAPM full-dose targets instead of AAPM-Mayo.

Usage:
    python sad_1.py --dose 5pct --data-root /path/to/uldct_5pct/dataset \
        --pidinet-weights /path/to/pidinet_table5_pretrained.pth
"""

from __future__ import annotations

import logging
import math
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from common.checkpoint import best_psnr_so_far, load_checkpoint, save_checkpoint
from common.config import (
    DATA_RANGE,
    TEST_SPLIT,
    TRAIN_SPLIT,
    VAL_SPLIT,
    RunConfig,
    build_arg_parser,
    get_device,
    parse_run_config,
)
from common.data import ULDCTDataset, denormalize, to_hu_window, truncate
from common.evaluation import quick_eval, run_test
from common.metrics import compute_psnr, compute_rmse, compute_ssim
from common.runtime import load_best_for_test, optional_split

MODEL_NAME = "sad_1"

# Hyperparameters (paper Sec. 3.2: Adam lr 5e-4, 300k iterations, T = 1000).
NUM_EPOCHS = 10_000  # upper bound; training stops at MAX_ITERS
MAX_ITERS = 300_000
BATCH_SIZE = 4
PATCH_SIZE = 64
PATCH_N = 4
LR = 5e-4
# The eq. 17 loss weights by 1/sigma_t^2 (up to ~4000x at small t); without
# global-norm clipping Adam blows up and training diverges.
GRAD_CLIP = 1.0
T_SCHED = 1000
BASE_CH = 64
PRINT_ITERS = 50
SAVE_EPOCHS = 5
VAL_MAX_SLICES = 8
SAMPLE_STEPS = 1  # 1 = SAD-1, 5 = SAD-5

BETA_MAX = 0.3
SIGMA_FLOOR_FRAC = 0.05
TIME_DIM = 256
ICR_HIDDEN = 128
GFM_RADIUS = 4
GFM_EPS = 1e-4

PIDINET_PRETRAINED = True
PIDINET_WEIGHTS = Path("pretrained") / "pidinet_table5_pretrained.pth"
# The checkpoint is committed to the official repository (not Git LFS).
PIDINET_URL = (
    "https://raw.githubusercontent.com/hellozhuo/pidinet/master/"
    "trained_models/table5_pidinet.pth"
)
# Configuration of table5_pidinet.pth ("carv4"): cd, ad, rd, cv repeated 4x.
CARV4_OPS = ["cd", "ad", "rd", "cv"] * 4

logger = logging.getLogger(MODEL_NAME)


# ============================================================================
# Time embedding
# ============================================================================
class SinusoidalPosEmb(nn.Module):
    """Sinusoidal embedding of the diffusion timestep."""

    def __init__(self, dim: int) -> None:
        """Store the embedding size.

        Args:
            dim: Output embedding dimension.
        """
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embed a ``(B,)`` timestep tensor into ``(B, dim)``."""
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


# ============================================================================
# Structure prior network G: PiDiNet (Su et al., 2021)
#
# Faithful to the official repository (hellozhuo/pidinet, models/{ops,
# pidinet,config}.py), with the same module names so that table5_pidinet.pth
# loads exactly: config "carv4", inplane=60, sa=True, dil=24. G is frozen
# (paper Sec. 2.2.1: parameters fixed, "without any retraining").
# ============================================================================
ConvFunc = Callable[..., torch.Tensor]


def createConvFunc(op_type: str) -> ConvFunc:
    """Return a pixel-difference convolution (``cd``/``ad``/``rd``) or ``cv``.

    Verbatim port of ``pidinet/models/ops.py``.

    Args:
        op_type: ``"cv"`` (vanilla), ``"cd"`` (central), ``"ad"`` (angular) or
            ``"rd"`` (radial) difference.

    Returns:
        A function with the signature of :func:`torch.nn.functional.conv2d`.
    """
    assert op_type in ["cv", "cd", "ad", "rd"], f"unknown op type: {op_type}"
    if op_type == "cv":
        return F.conv2d

    if op_type == "cd":

        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], "dilation for cd_conv should be in 1 or 2"
            assert (
                weights.size(2) == 3 and weights.size(3) == 3
            ), "kernel size for cd_conv should be 3x3"
            assert padding == dilation, "padding for cd_conv set wrong"
            weights_c = weights.sum(dim=[2, 3], keepdim=True)
            yc = F.conv2d(x, weights_c, stride=stride, padding=0, groups=groups)
            y = F.conv2d(
                x,
                weights,
                bias,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
            )
            return y - yc

        return func

    if op_type == "ad":

        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], "dilation for ad_conv should be in 1 or 2"
            assert (
                weights.size(2) == 3 and weights.size(3) == 3
            ), "kernel size for ad_conv should be 3x3"
            assert padding == dilation, "padding for ad_conv set wrong"
            shape = weights.shape
            weights = weights.view(shape[0], shape[1], -1)
            # Clockwise neighbour differences.
            weights_conv = (weights - weights[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]).view(
                shape
            )
            return F.conv2d(
                x,
                weights_conv,
                bias,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
            )

        return func

    def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
        assert dilation in [1, 2], "dilation for rd_conv should be in 1 or 2"
        assert (
            weights.size(2) == 3 and weights.size(3) == 3
        ), "kernel size for rd_conv should be 3x3"
        padding = 2 * dilation
        shape = weights.shape
        buffer = torch.zeros(shape[0], shape[1], 5 * 5, device=weights.device)
        weights = weights.view(shape[0], shape[1], -1)
        buffer[:, :, [0, 2, 4, 10, 14, 20, 22, 24]] = weights[:, :, 1:]
        buffer[:, :, [6, 7, 8, 11, 13, 16, 17, 18]] = -weights[:, :, 1:]
        buffer[:, :, 12] = 0
        buffer = buffer.view(shape[0], shape[1], 5, 5)
        return F.conv2d(
            x,
            buffer,
            bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )

    return func


class _PDCConv2d(nn.Module):
    """Convolution layer applying a pixel-difference function (``ops.Conv2d``)."""

    def __init__(
        self,
        pdc: ConvFunc,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
    ) -> None:
        """Create the weights.

        Args:
            pdc: Convolution function from :func:`createConvFunc`.
            in_channels: Input channels.
            out_channels: Output channels.
            kernel_size: Square kernel side.
            stride: Convolution stride.
            padding: Zero padding.
            dilation: Kernel dilation.
            groups: Number of channel groups.
            bias: Whether to learn an additive bias.
        """
        super().__init__()
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()
        self.pdc = pdc

    def reset_parameters(self) -> None:
        """Initialise like :class:`torch.nn.Conv2d`."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the pixel-difference convolution."""
        return self.pdc(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class _CSAM(nn.Module):
    """Compact spatial attention module (PiDiNet)."""

    def __init__(self, channels: int) -> None:
        """Build the layers.

        Args:
            channels: Input channels.
        """
        super().__init__()
        mid_channels = 4
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(channels, mid_channels, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(mid_channels, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        nn.init.constant_(self.conv1.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reweight ``x`` with a spatial attention map."""
        y = self.relu1(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.sigmoid(y)
        return x * y


class _CDCM(nn.Module):
    """Compact dilation convolution module (PiDiNet)."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Build the layers.

        Args:
            in_channels: Input channels.
            out_channels: Output channels.
        """
        super().__init__()
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

        def dilated(d: int) -> nn.Conv2d:
            return nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                dilation=d,
                padding=d,
                bias=False,
            )

        self.conv2_1 = dilated(5)
        self.conv2_2 = dilated(7)
        self.conv2_3 = dilated(9)
        self.conv2_4 = dilated(11)
        nn.init.constant_(self.conv1.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Sum of four dilated convolutions of the projected input."""
        x = self.relu1(x)
        x = self.conv1(x)
        return self.conv2_1(x) + self.conv2_2(x) + self.conv2_3(x) + self.conv2_4(x)


class _MapReduce(nn.Module):
    """Reduce features to a single-channel edge map (PiDiNet)."""

    def __init__(self, channels: int) -> None:
        """Build the 1x1 projection.

        Args:
            channels: Input channels.
        """
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1, padding=0)
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project ``x`` to one channel."""
        return self.conv(x)


class _PDCBlock(nn.Module):
    """Depthwise PDC + pointwise 1x1 convolution with a residual (PiDiNet)."""

    def __init__(self, pdc: ConvFunc, inplane: int, ouplane: int, stride: int = 1):
        """Build the layers.

        Args:
            pdc: Convolution function from :func:`createConvFunc`.
            inplane: Input channels.
            ouplane: Output channels.
            stride: 2 to downsample (max-pool + 1x1 shortcut), else 1.
        """
        super().__init__()
        self.stride = stride
        if self.stride > 1:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.shortcut = nn.Conv2d(inplane, ouplane, kernel_size=1, padding=0)
        self.conv1 = _PDCConv2d(
            pdc, inplane, inplane, kernel_size=3, padding=1, groups=inplane, bias=False
        )
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(inplane, ouplane, kernel_size=1, padding=0, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the block."""
        if self.stride > 1:
            x = self.pool(x)
        y = self.conv1(x)
        y = self.relu2(y)
        y = self.conv2(y)
        if self.stride > 1:
            x = self.shortcut(x)
        return y + x


class PiDiNetPrior(nn.Module):
    """Frozen PiDiNet (carv4, sa, dil=24, inplane=60) used as structure prior.

    Attributes:
        prior_chs: Channels of the multiscale prompts, ``(60, 120, 240, 240)``.
    """

    def __init__(
        self,
        in_ch: int = 1,
        inplane: int = 60,
        dil: int = 24,
        pretrained: bool = True,
        weights_path: Path = PIDINET_WEIGHTS,
    ) -> None:
        """Build the network and load the BSDS500 weights.

        Args:
            in_ch: Input channels; single-channel CT is replicated to RGB.
            inplane: Channels of the first stage.
            dil: Channels of the dilation (CDCM) heads.
            pretrained: Load the pretrained weights (disable only for tests).
            weights_path: Local weights file; downloaded from ``PIDINET_URL``
                if missing.
        """
        super().__init__()
        self._in_ch = in_ch
        self.sa = True
        self.dil = dil
        pdcs = [createConvFunc(op) for op in CARV4_OPS]

        self.fuseplanes = []
        self.inplane = inplane
        self.init_block = _PDCConv2d(pdcs[0], 3, self.inplane, kernel_size=3, padding=1)

        self.block1_1 = _PDCBlock(pdcs[1], self.inplane, self.inplane)
        self.block1_2 = _PDCBlock(pdcs[2], self.inplane, self.inplane)
        self.block1_3 = _PDCBlock(pdcs[3], self.inplane, self.inplane)
        self.fuseplanes.append(self.inplane)

        inplane = self.inplane
        self.inplane = self.inplane * 2
        self.block2_1 = _PDCBlock(pdcs[4], inplane, self.inplane, stride=2)
        self.block2_2 = _PDCBlock(pdcs[5], self.inplane, self.inplane)
        self.block2_3 = _PDCBlock(pdcs[6], self.inplane, self.inplane)
        self.block2_4 = _PDCBlock(pdcs[7], self.inplane, self.inplane)
        self.fuseplanes.append(self.inplane)

        inplane = self.inplane
        self.inplane = self.inplane * 2
        self.block3_1 = _PDCBlock(pdcs[8], inplane, self.inplane, stride=2)
        self.block3_2 = _PDCBlock(pdcs[9], self.inplane, self.inplane)
        self.block3_3 = _PDCBlock(pdcs[10], self.inplane, self.inplane)
        self.block3_4 = _PDCBlock(pdcs[11], self.inplane, self.inplane)
        self.fuseplanes.append(self.inplane)

        self.block4_1 = _PDCBlock(pdcs[12], self.inplane, self.inplane, stride=2)
        self.block4_2 = _PDCBlock(pdcs[13], self.inplane, self.inplane)
        self.block4_3 = _PDCBlock(pdcs[14], self.inplane, self.inplane)
        self.block4_4 = _PDCBlock(pdcs[15], self.inplane, self.inplane)
        self.fuseplanes.append(self.inplane)

        # sa + dil heads: CDCM -> CSAM -> MapReduce per stage.
        self.conv_reduces = nn.ModuleList()
        self.attentions = nn.ModuleList()
        self.dilations = nn.ModuleList()
        for i in range(4):
            self.dilations.append(_CDCM(self.fuseplanes[i], self.dil))
            self.attentions.append(_CSAM(self.dil))
            self.conv_reduces.append(_MapReduce(self.dil))

        self.classifier = nn.Conv2d(4, 1, kernel_size=1)
        nn.init.constant_(self.classifier.weight, 0.25)
        nn.init.constant_(self.classifier.bias, 0)

        self.prior_chs = tuple(self.fuseplanes)

        if pretrained:
            self._load_pretrained(Path(weights_path))

        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> PiDiNetPrior:
        """Stay in eval mode: the prior is fixed even when SAD is training."""
        return super().train(False)

    def _load_pretrained(self, weights_path: Path) -> None:
        """Download (if needed) and load the official BSDS500 checkpoint.

        Fails loudly on any mismatch: a random frozen prior would violate the
        paper (fixed pretrained parameters) and is worse than training it.

        Args:
            weights_path: Local weights file.

        Raises:
            RuntimeError: If not every parameter matches the checkpoint.
        """
        if not weights_path.exists():
            logger.info("[PiDiNet] downloading pretrained weights -> %s", weights_path)
            weights_path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(PIDINET_URL, weights_path)
        sd = torch.load(weights_path, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        # Strip the DataParallel "module." prefix.
        sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
        own = self.state_dict()
        matched = {k: v for k, v in sd.items() if k in own and v.shape == own[k].shape}
        res = self.load_state_dict(matched, strict=False)
        logger.info(
            "[PiDiNet] loaded %d/%d pretrained params (missing: %d, unexpected: %d)",
            len(matched),
            len(own),
            len(res.missing_keys),
            len(res.unexpected_keys),
        )
        if len(matched) < len(own):
            raise RuntimeError(
                f"[PiDiNet] only {len(matched)}/{len(own)} tensors matched "
                f"{weights_path}; the prior must be the pretrained PiDiNet (paper "
                f"Sec. 2.2.1). Missing e.g.: {res.missing_keys[:5]}"
            )

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Extract the structural prompts.

        Args:
            x: Image batch ``(B, 1, H, W)``.

        Returns:
            ``(Ps, S)``: the four encoder feature maps (60, 120, 240, 240
            channels at scales 1, 1/2, 1/4, 1/8; eq. 10) and the fused
            full-resolution edge map (sigmoid, 1 channel; eq. 9).
        """
        # PiDiNet was pretrained on RGB: replicate the CT channel.
        if self._in_ch == 1 and x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        H, W = x.size()[2:]
        x = self.init_block(x)

        x1 = self.block1_3(self.block1_2(self.block1_1(x)))
        x2 = self.block2_4(self.block2_3(self.block2_2(self.block2_1(x1))))
        x3 = self.block3_4(self.block3_3(self.block3_2(self.block3_1(x2))))
        x4 = self.block4_4(self.block4_3(self.block4_2(self.block4_1(x3))))

        feats = [x1, x2, x3, x4]
        edges = []
        for i, xi in enumerate(feats):
            f = self.attentions[i](self.dilations[i](xi))
            e = self.conv_reduces[i](f)
            edges.append(F.interpolate(e, (H, W), mode="bilinear", align_corners=False))
        S = torch.sigmoid(self.classifier(torch.cat(edges, dim=1)))
        return feats, S


# ============================================================================
# Guided filter module (paper eqs. 11-15; He et al., 2010)
# ============================================================================
def _boxfilter(x: torch.Tensor, r: int) -> torch.Tensor:
    """Mean filter with a ``(2r+1) x (2r+1)`` window (no padding bias)."""
    k = 2 * r + 1
    return F.avg_pool2d(x, kernel_size=k, stride=1, padding=r, count_include_pad=False)


class GuidedFilterModule(nn.Module):
    """Differentiable linear guided filter: ``E_hat = a * P + b`` (eq. 12)."""

    def __init__(
        self,
        encoder_ch: int,
        prompt_ch: int,
        radius: int = GFM_RADIUS,
        eps: float = GFM_EPS,
    ) -> None:
        """Build the prompt projection.

        Args:
            encoder_ch: Channels of the encoder feature E_s.
            prompt_ch: Channels of the prompt P_s.
            radius: Box-filter radius.
            eps: Regulariser of the local linear coefficients.
        """
        super().__init__()
        self.radius = radius
        self.eps = eps
        self.align = (
            nn.Conv2d(prompt_ch, encoder_ch, 1)
            if prompt_ch != encoder_ch
            else nn.Identity()
        )

    def forward(self, E_s: torch.Tensor, P_s: torch.Tensor) -> torch.Tensor:
        """Filter the encoder feature ``E_s`` guided by the prompt ``P_s``."""
        if P_s.shape[-2:] != E_s.shape[-2:]:
            P_s = F.interpolate(
                P_s, size=E_s.shape[-2:], mode="bilinear", align_corners=False
            )
        P_s = self.align(P_s)
        r = self.radius
        mu_p = _boxfilter(P_s, r)
        mu_e = _boxfilter(E_s, r)
        mu_pe = _boxfilter(P_s * E_s, r)
        mu_pp = _boxfilter(P_s * P_s, r)
        var_p = mu_pp - mu_p * mu_p
        cov = mu_pe - mu_p * mu_e
        a = cov / (var_p + self.eps)
        b = mu_e - a * mu_p
        return _boxfilter(a, r) * P_s + _boxfilter(b, r)


# ============================================================================
# Implicit conditional representation (paper Sec. 2.2.3, eq. 16)
# ============================================================================
class ImplicitConditionalRepresentation(nn.Module):
    """Per-pixel MLP on (features, structure map, normalised coordinates)."""

    def __init__(self, feature_ch: int, hidden: int = ICR_HIDDEN, out_ch: int = 1):
        """Build the MLP.

        Args:
            feature_ch: Channels of the decoder features.
            hidden: Hidden size.
            out_ch: Output channels.
        """
        super().__init__()
        # Inputs: features + S (1 channel) + (y, x) coordinates.
        self.mlp = nn.Sequential(
            nn.Linear(feature_ch + 1 + 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_ch),
        )

    def _make_coords(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        """Coordinate grid in ``[-1, 1]``, shape ``(2, h, w)``."""
        yy = torch.linspace(-1, 1, h, device=device)
        xx = torch.linspace(-1, 1, w, device=device)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
        return torch.stack([grid_y, grid_x], dim=0)

    def forward(self, F_feat: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        """Map ``(B, C, H, W)`` features and the edge map to ``(B, out_ch, H, W)``."""
        B, C, H, W = F_feat.shape
        if S.shape[-2:] != (H, W):
            S = F.interpolate(S, size=(H, W), mode="bilinear", align_corners=False)
        coords = (
            self._make_coords(H, W, F_feat.device).unsqueeze(0).expand(B, -1, -1, -1)
        )
        x = torch.cat([F_feat, S, coords], dim=1)
        x = x.permute(0, 2, 3, 1).reshape(B * H * W, -1)
        return self.mlp(x).reshape(B, H, W, -1).permute(0, 3, 1, 2)


# ============================================================================
# U-Net backbone (4 encoder + 4 decoder stages; GFM per decoder stage)
# ============================================================================
class ResBlock(nn.Module):
    """Pre-activation residual block conditioned on the time embedding."""

    def __init__(self, in_ch: int, out_ch: int, time_ch: int) -> None:
        """Build the layers.

        Args:
            in_ch: Input channels.
            out_ch: Output channels.
            time_ch: Size of the time embedding.
        """
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_ch, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """Apply the block to ``x`` given the time embedding ``t_emb``."""
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.time_proj(self.act(t_emb))[:, :, None, None]
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.skip(x)


class UNetDenoiser(nn.Module):
    """U-Net predicting X_0 from X_t with GFM prompts and an ICR head."""

    def __init__(
        self,
        in_ch: int = 1,
        base_ch: int = BASE_CH,
        time_dim: int = TIME_DIM,
        prior_chs: tuple[int, ...] = (60, 120, 240, 240),
    ) -> None:
        """Build the network.

        Args:
            in_ch: Image channels.
            base_ch: Channels of the first stage (doubled per stage).
            time_dim: Size of the time embedding.
            prior_chs: Channels of the four PiDiNet prompts.
        """
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )

        self.inc = nn.Conv2d(in_ch, base_ch, 3, padding=1)
        self.enc1 = ResBlock(base_ch, base_ch, time_dim)
        self.enc2 = ResBlock(base_ch, base_ch * 2, time_dim)
        self.enc3 = ResBlock(base_ch * 2, base_ch * 4, time_dim)
        self.enc4 = ResBlock(base_ch * 4, base_ch * 8, time_dim)
        self.pool = nn.AvgPool2d(2)

        self.mid = ResBlock(base_ch * 8, base_ch * 8, time_dim)

        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.gfm4 = GuidedFilterModule(base_ch * 8, prior_chs[3])
        self.dec4 = ResBlock(base_ch * 8 * 2, base_ch * 4, time_dim)
        self.gfm3 = GuidedFilterModule(base_ch * 4, prior_chs[2])
        self.dec3 = ResBlock(base_ch * 4 * 2, base_ch * 2, time_dim)
        self.gfm2 = GuidedFilterModule(base_ch * 2, prior_chs[1])
        self.dec2 = ResBlock(base_ch * 2 * 2, base_ch, time_dim)
        self.gfm1 = GuidedFilterModule(base_ch, prior_chs[0])
        self.dec1 = ResBlock(base_ch * 2, base_ch, time_dim)

        self.icr = ImplicitConditionalRepresentation(
            feature_ch=base_ch, hidden=ICR_HIDDEN, out_ch=in_ch
        )

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        Ps: list[torch.Tensor],
        S: torch.Tensor,
    ) -> torch.Tensor:
        """Predict X_0.

        Args:
            x_t: Bridge sample ``(B, 1, H, W)``.
            t: Timesteps ``(B,)``.
            Ps: Multiscale prompts from :class:`PiDiNetPrior`.
            S: Fused edge map from :class:`PiDiNetPrior`.

        Returns:
            The X_0 prediction, same shape as ``x_t``.
        """
        t_emb = self.time_mlp(t)
        e0 = self.inc(x_t)
        e1 = self.enc1(e0, t_emb)
        e2 = self.enc2(self.pool(e1), t_emb)
        e3 = self.enc3(self.pool(e2), t_emb)
        e4 = self.enc4(self.pool(e3), t_emb)

        m = self.mid(e4, t_emb)

        # Eqs. 11-12: the GFM filters the ENCODER feature E_s guided by P_s; the
        # result is concatenated with the decoded feature.
        d = self.dec4(torch.cat([self.gfm4(e4, Ps[3]), m], dim=1), t_emb)
        d = self.up(d)
        d = self.dec3(torch.cat([self.gfm3(e3, Ps[2]), d], dim=1), t_emb)
        d = self.up(d)
        d = self.dec2(torch.cat([self.gfm2(e2, Ps[1]), d], dim=1), t_emb)
        d = self.up(d)
        d = self.dec1(torch.cat([self.gfm1(e1, Ps[0]), d], dim=1), t_emb)

        return self.icr(d, S)


# ============================================================================
# Schrödinger bridge (paper eqs. 4-8, f := 0, quadratic beta schedule)
# ============================================================================
def quadratic_beta_schedule(T: int, beta_max: float = BETA_MAX) -> torch.Tensor:
    """``beta_t = beta_max * (t / T)^2`` sampled at ``T`` points in ``[0, 1]``."""
    t = torch.linspace(0, 1, T)
    return beta_max * (t**2)


class SchrodingerBridge(nn.Module):
    """Noise schedule and analytic posterior of the diffusion bridge."""

    def __init__(self, T: int = T_SCHED, beta_max: float = BETA_MAX) -> None:
        """Precompute the schedule buffers.

        Args:
            T: Number of diffusion timesteps.
            beta_max: Final value of the quadratic beta schedule.
        """
        super().__init__()
        self.T = T
        betas = quadratic_beta_schedule(T, beta_max)
        # sigma^2_t = int_0^t beta dtau (eq. 1a: sqrt(beta) multiplies dW, so the
        # variance accumulates as a time integral). Without the step 1/T,
        # sigma^2_T would be 1000x too large (std ~5 noise on [0, 1] images).
        sigma2 = torch.cumsum(betas, dim=0) * (1.0 / T)
        self.register_buffer("betas", betas)
        self.register_buffer("sigma2", sigma2)
        self.register_buffer("sigma", sigma2.sqrt())
        # Floor for sigma_t in the eq. 17 weight (1/sigma_t explodes as t -> 0);
        # 5% of sigma_T caps the weight at ~400x.
        self.sigma_floor = float(SIGMA_FLOOR_FRAC * self.sigma[-1])

    def q_sample(
        self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Sample X_t ~ q(X_t | X_0, X_1) (eq. 7).

        With f = 0 and pinned endpoints the mean interpolates linearly in
        sigma^2: ``mu_t = (1 - a_t) X_0 + a_t X_1`` and
        ``Sigma_t = sigma^2_t (1 - a_t)``, with ``a_t = sigma^2_t / sigma^2_T``.

        Args:
            x0: Full-dose images.
            x1: Low-dose images.
            t: Timesteps ``(B,)``.
        """
        s_T = self.sigma2[-1]
        s_t = self.sigma2[t].view(-1, 1, 1, 1)
        alpha = s_t / s_T
        mean = (1 - alpha) * x0 + alpha * x1
        var = s_t * (1 - alpha)
        noise = torch.randn_like(x0)
        return mean + var.sqrt() * noise


# ============================================================================
# SAD model
# ============================================================================
class SAD(nn.Module):
    """Structure-aware diffusion: frozen PiDiNet prior + U-Net + bridge."""

    def __init__(
        self,
        T: int = T_SCHED,
        base_ch: int = BASE_CH,
        pidinet_pretrained: bool = True,
        pidinet_weights: Path = PIDINET_WEIGHTS,
    ) -> None:
        """Build the model.

        Args:
            T: Number of diffusion timesteps.
            base_ch: Base channels of the U-Net.
            pidinet_pretrained: Load the pretrained PiDiNet (disable only for
                tests).
            pidinet_weights: Local PiDiNet weights file.
        """
        super().__init__()
        self.G = PiDiNetPrior(
            in_ch=1, pretrained=pidinet_pretrained, weights_path=pidinet_weights
        )
        self.denoiser = UNetDenoiser(
            in_ch=1, base_ch=base_ch, prior_chs=self.G.prior_chs
        )
        self.bridge = SchrodingerBridge(T=T)

    def forward(
        self, x0: torch.Tensor, x1: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Training pass: predict X_0 from X_t with prompts computed from X_t.

        Args:
            x0: Full-dose images (target).
            x1: Low-dose images.

        Returns:
            ``(x_pred, x_t, t)``.
        """
        t = torch.randint(0, self.bridge.T, (x0.size(0),), device=x0.device)
        x_t = self.bridge.q_sample(x0, x1, t)
        Ps, S = self.G(x_t)
        x_pred = self.denoiser(x_t, t, Ps, S)
        return x_pred, x_t, t

    @torch.no_grad()
    def sample(self, x1: torch.Tensor, steps: int = 1) -> torch.Tensor:
        """Denoise LDCT images with ``steps`` reverse steps (1 = SAD-1, 5 = SAD-5).

        Timesteps follow a quadratic discretisation (more steps near t = 0).
        With iterative prompt refinement (Sec. 2.3), G receives the latest
        prediction at each step; at the first step that is the LDCT itself,
        which equals the initial X_t.

        Args:
            x1: Low-dose images.
            steps: Number of reverse steps.

        Returns:
            The X_0 prediction.
        """
        T = self.bridge.T
        u = torch.linspace(0.0, 1.0, steps + 1)
        idx = ((u**2) * (T - 1)).round().long().flip(0).tolist()  # T-1 -> 0
        x_t = x1.clone()
        g_input = x1
        x_pred = x1
        for i in range(steps):
            t = torch.full((x_t.size(0),), idx[i], device=x_t.device, dtype=torch.long)
            Ps, S = self.G(g_input)
            x_pred = self.denoiser(x_t, t, Ps, S)
            g_input = x_pred
            if i < steps - 1:
                t_next = torch.full(
                    (x_t.size(0),), idx[i + 1], device=x_t.device, dtype=torch.long
                )
                x_t = self.bridge.q_sample(x_pred, x1, t_next)
        return x_pred


def build_optimizer(model: SAD, lr: float = LR) -> torch.optim.Optimizer:
    """Adam over the trainable parameters (the PiDiNet prior is frozen)."""
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.Adam(params, lr=lr)


# ============================================================================
# Training and testing
# ============================================================================
def _safe_train_metrics(
    pred_norm: torch.Tensor, target_norm: torch.Tensor
) -> tuple[float, float, float]:
    """PSNR, SSIM (clipped to [-1, 1]) and RMSE of a training batch in HU."""
    p_hu = truncate(denormalize(pred_norm.detach().cpu()))
    t_hu = truncate(denormalize(target_norm.detach().cpu()))
    psnr = compute_psnr(p_hu, t_hu, DATA_RANGE)
    ssim = max(min(compute_ssim(p_hu, t_hu, DATA_RANGE), 1.0), -1.0)
    rmse = compute_rmse(p_hu, t_hu)
    return psnr, ssim, rmse


def _checkpoint_save(
    model: SAD,
    opt: torch.optim.Optimizer,
    cfg: RunConfig,
    epoch: int,
    step: int,
    losses: list[float],
    best_psnr: float,
    val_loader: DataLoader | None,
    device: torch.device,
    sample_steps: int,
) -> float:
    """Validate, save the last/best checkpoints and the loss history.

    Args:
        model: SAD model.
        opt: Optimizer.
        cfg: Run configuration.
        epoch: Current epoch.
        step: Global iteration counter.
        losses: Per-iteration training losses.
        best_psnr: Best validation PSNR so far.
        val_loader: Validation loader (first ``VAL_MAX_SLICES`` slices used).
        device: Inference device.
        sample_steps: Reverse steps used for validation sampling.

    Returns:
        The updated best validation PSNR.
    """
    model.eval()
    psnr, ssim, rmse = (
        quick_eval(
            lambda xb: model.sample(xb, steps=sample_steps),
            val_loader,
            device,
            max_slices=VAL_MAX_SLICES,
        )
        if val_loader is not None
        else (0.0, 0.0, 0.0)
    )
    model.train()
    state = dict(
        model=model,
        optimizer=opt,
        epoch=epoch,
        step=step,
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
    logger.info(
        "  saved (epoch=%d step=%d psnr=%.3f ssim=%.4f rmse_hu=%.3f) best_psnr=%.3f",
        epoch,
        step,
        psnr,
        ssim,
        rmse,
        best_psnr,
    )
    return best_psnr


def train(
    model: SAD,
    loader: DataLoader,
    device: torch.device,
    cfg: RunConfig,
    val_loader: DataLoader | None = None,
    sample_steps: int = SAMPLE_STEPS,
) -> None:
    """Train with the eq. 17 loss until ``MAX_ITERS``, resuming if possible.

    The denoising score-matching loss ``||eps - (X_t - X_0) / sigma_t||^2`` is
    implemented in the X_0 parameterisation (``x_pred = X_t - sigma_t * eps``),
    i.e. an L2 loss on X_0 weighted by ``1 / sigma_t^2`` with a floor on
    ``sigma_t``.

    Args:
        model: SAD model.
        loader: Training loader yielding stacks of patches.
        device: Training device.
        cfg: Run configuration.
        val_loader: Loader for validation at checkpoint time.
        sample_steps: Reverse steps used for validation sampling.
    """
    model.train()
    opt = build_optimizer(model, lr=LR)

    ckpt = load_checkpoint(cfg.last_ckpt, model=model, optimizer=opt)
    start_epoch = (ckpt["epoch"] + 1) if ckpt else 1
    step = ckpt["step"] if ckpt else 0
    best_psnr = best_psnr_so_far(cfg.best_ckpt)
    losses = []
    t0 = time.time()

    def save(epoch: int, best: float) -> float:
        return _checkpoint_save(
            model, opt, cfg, epoch, step, losses, best, val_loader, device, sample_steps
        )

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        for x, y in loader:
            step += 1
            x = x.float().to(device)
            y = y.float().to(device)
            if x.dim() == 4:
                x = x.view(-1, 1, PATCH_SIZE, PATCH_SIZE)
                y = y.view(-1, 1, PATCH_SIZE, PATCH_SIZE)
            else:
                x, y = x.unsqueeze(1), y.unsqueeze(1)
            # X_0 = full dose (target), X_1 = low dose.
            x_pred, _, t_b = model(y, x)
            sigma_t = (
                model.bridge.sigma[t_b]
                .view(-1, 1, 1, 1)
                .clamp(min=model.bridge.sigma_floor)
            )
            loss = (((x_pred - y) / sigma_t) ** 2).mean()
            # Skip batches whose loss overflowed (1/sigma_t^2 weight) instead of
            # propagating NaN/Inf into the weights.
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True)
                continue
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            losses.append(loss.item())

            if step % PRINT_ITERS == 0:
                with torch.no_grad():
                    psnr, ssim, rmse = _safe_train_metrics(x_pred.clamp(0.0, 1.0), y)
                logger.info(
                    "epoch %d/%d  iter %d  loss %.6f  PSNR %6.2f  SSIM %.4f  "
                    "RMSE %6.2f HU  (%.0fs)",
                    epoch,
                    NUM_EPOCHS,
                    step,
                    loss.item(),
                    psnr,
                    ssim,
                    rmse,
                    time.time() - t0,
                )

            if step >= MAX_ITERS:
                save(epoch, best_psnr)
                return

        if epoch % SAVE_EPOCHS == 0 or epoch == NUM_EPOCHS:
            best_psnr = save(epoch, best_psnr)


def test(
    model: SAD,
    loader: DataLoader,
    device: torch.device,
    cfg: RunConfig,
    sample_steps: int = SAMPLE_STEPS,
) -> None:
    """Evaluate on the full test split and write metrics and figures.

    Args:
        model: Trained SAD model.
        loader: Test loader with batch size 1.
        device: Inference device.
        cfg: Run configuration.
        sample_steps: Reverse steps (1 = SAD-1, 5 = SAD-5).
    """
    model.eval()

    def predict(
        batch: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y = batch
        x = x.float().to(device).unsqueeze(1)
        y = y.float().to(device).unsqueeze(1)
        pred = model.sample(x, steps=sample_steps).clamp(0.0, 1.0)
        return to_hu_window(x), to_hu_window(y), to_hu_window(pred)

    run_test(loader, predict, cfg)


def main() -> None:
    """Train SAD on the training split, then test the best checkpoint."""
    parser = build_arg_parser(MODEL_NAME, description=__doc__)
    parser.add_argument(
        "--pidinet-weights",
        type=Path,
        default=PIDINET_WEIGHTS,
        help="PiDiNet table5 weights; downloaded from the official repo if missing.",
    )
    parser.add_argument(
        "--sample-steps",
        type=int,
        default=SAMPLE_STEPS,
        help="Reverse steps at inference (1 = SAD-1, 5 = SAD-5).",
    )
    cfg, args = parse_run_config(parser, MODEL_NAME)
    torch.backends.cudnn.benchmark = True
    device = get_device()
    logger.info(
        "[%s] dose=%s device=%s out=%s", MODEL_NAME, cfg.dose, device, cfg.output_dir
    )

    train_ds = ULDCTDataset(
        TRAIN_SPLIT, cfg.data_root, patch_size=PATCH_SIZE, patch_n=PATCH_N
    )
    test_ds = ULDCTDataset(TEST_SPLIT, cfg.data_root)
    val_ds = optional_split(lambda: ULDCTDataset(VAL_SPLIT, cfg.data_root), VAL_SPLIT)
    logger.info(
        "train files: %d | val files: %d | test files: %d",
        len(train_ds),
        len(val_ds) if val_ds is not None else 0,
        len(test_ds),
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=cfg.num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=cfg.num_workers
    )
    val_loader = (
        DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=cfg.num_workers)
        if val_ds is not None
        else test_loader
    )

    model = SAD(
        T=T_SCHED,
        base_ch=BASE_CH,
        pidinet_pretrained=PIDINET_PRETRAINED,
        pidinet_weights=args.pidinet_weights,
    ).to(device)
    train(
        model,
        train_loader,
        device,
        cfg,
        val_loader=val_loader,
        sample_steps=args.sample_steps,
    )
    load_best_for_test(cfg, model)
    test(model, test_loader, device, cfg, sample_steps=args.sample_steps)


if __name__ == "__main__":
    main()
