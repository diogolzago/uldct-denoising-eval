"""CTformer (Wang et al., 2023) trained and evaluated on the ULDCT dataset.

Convolution-free token-to-token dilated vision transformer: a performer-based
tokens-to-token encoder, a transformer block and a mirrored token-back-to-image
decoder that predicts the noise residual. The configuration and training
schedule follow ``main.py`` of the reference CTformer repository. Full slices
are denoised with overlapping 64x64 patches (sliding-window inference).

Usage:
    python ctformer.py --dose 5pct --data-root /path/to/uldct_5pct/dataset
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
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
from common.evaluation import run_test
from common.metrics import Measure, compute_psnr, compute_rmse, compute_ssim
from common.runtime import load_best_for_test, optional_split

try:
    import pydicom
except ImportError:
    pydicom = None

MODEL_NAME = "ctformer"

# Training schedule (reference repo main.py defaults).
NUM_EPOCHS = 4000
BATCH_SIZE = 16
PATCH_SIZE = 64
PATCH_N = 4
LR = 1e-5
LR_DECAY_ITERS = 8000
LR_DECAY_FACTOR = 0.5
PRINT_ITERS = 20
SAVE_ITERS = 1500
# main.py: loss = MSE(pred, y) * 100 + 1e-4
LOSS_SCALE = 100.0
LOSS_OFFSET = 1e-4
QUICK_EVAL_MAX_SLICES = 32

# Model configuration (reference repo main.py).
EMBED_DIM = 64
DEPTH = 1
NUM_HEADS = 8
TOKEN_KERNEL = 4
TOKEN_STRIDE = 4
MLP_RATIO = 2.0
TOKEN_DIM = 64
INIT_STD = 0.02

# Sliding-window inference: 64x64 patches with stride 32 over a 16-px
# zero-padded image; only the central 32x32 of each patch is kept.
INFER_STRIDE = 32
INFER_PAD = 16
INFER_CHUNK = 64

DATALOADER_PREFETCH = 4
FD_CACHE_LOG_EVERY = 200

logger = logging.getLogger(MODEL_NAME)


# ============================================================================
# Model: building blocks
# ============================================================================
def drop_path(
    x: torch.Tensor, drop_prob: float = 0.0, training: bool = False
) -> torch.Tensor:
    """Stochastic depth: randomly drop whole residual branches per sample.

    Args:
        x: Branch output ``(B, ...)``.
        drop_prob: Probability of dropping the branch.
        training: Whether the module is in training mode.

    Returns:
        ``x`` rescaled by ``1 / keep_prob`` with dropped samples zeroed.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    """Module wrapper around :func:`drop_path` (local replacement for timm)."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        """Store the drop probability.

        Args:
            drop_prob: Probability of dropping the branch.
        """
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply stochastic depth in training mode."""
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    """Two-layer feed-forward network with dropout."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: type[nn.Module] = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        """Build the layers.

        Args:
            in_features: Input width.
            hidden_features: Hidden width (defaults to ``in_features``).
            out_features: Output width (defaults to ``in_features``).
            act_layer: Activation class.
            drop: Dropout probability.
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the network to tokens ``(B, N, C)``."""
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


class Attention(nn.Module):
    """Multi-head self-attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        """Build the projections.

        Args:
            dim: Token width.
            num_heads: Number of attention heads.
            qkv_bias: Whether the QKV projection has a bias.
            qk_scale: Attention scale (defaults to ``head_dim ** -0.5``).
            attn_drop: Dropout on the attention weights.
            proj_drop: Dropout on the output projection.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Attend over tokens ``(B, N, C)``."""
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class Block(nn.Module):
    """Pre-norm transformer block (attention + MLP, both residual)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ) -> None:
        """Build the block.

        Args:
            dim: Token width.
            num_heads: Number of attention heads.
            mlp_ratio: Hidden width of the MLP relative to ``dim``.
            qkv_bias: Whether the QKV projection has a bias.
            qk_scale: Attention scale override.
            drop: Dropout on projections and MLP.
            attn_drop: Dropout on the attention weights.
            drop_path: Stochastic-depth probability.
            act_layer: MLP activation class.
            norm_layer: Normalisation class.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the block to tokens ``(B, N, C)``."""
        x = x + self.drop_path(self.attn(self.norm1(x)))
        return x + self.drop_path(self.mlp(self.norm2(x)))


def get_sinusoid_encoding(n_position: int, d_hid: int) -> torch.Tensor:
    """Fixed sinusoidal position encoding of shape ``(1, n_position, d_hid)``."""

    def get_position_angle_vec(position: int) -> list[float]:
        return [
            position / np.power(10000, 2 * (hid_j // 2) / d_hid)
            for hid_j in range(d_hid)
        ]

    table = np.array([get_position_angle_vec(pos) for pos in range(n_position)])
    table[:, 0::2] = np.sin(table[:, 0::2])
    table[:, 1::2] = np.cos(table[:, 1::2])
    return torch.FloatTensor(table).unsqueeze(0)


class AttentionToken(nn.Module):
    """Single-scale token attention that changes the token width to ``in_dim``.

    Named ``Attention`` in the reference ``token_transformer.py``; renamed to
    avoid clashing with :class:`Attention`.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        in_dim: int | None = None,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        """Build the projections.

        Args:
            dim: Input token width.
            num_heads: Number of attention heads.
            in_dim: Output token width.
            qkv_bias: Whether the QKV projection has a bias.
            qk_scale: Attention scale (defaults to ``(dim // num_heads) ** -0.5``).
            attn_drop: Dropout on the attention weights.
            proj_drop: Dropout on the output projection.
        """
        super().__init__()
        self.num_heads = num_heads
        self.in_dim = in_dim
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, in_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(in_dim, in_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Attend over tokens ``(B, N, dim)`` and return ``(B, N, in_dim)``."""
        B, N, _ = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.in_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(B, N, self.in_dim)
        x = self.proj_drop(self.proj(x))
        # Skip connection on the values, since the width changes from dim to in_dim.
        return v.squeeze(1) + x


class Token_transformer(nn.Module):
    """Tokens-to-token transformer layer (attention + MLP)."""

    def __init__(
        self,
        dim: int,
        in_dim: int,
        num_heads: int,
        mlp_ratio: float = 1.0,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ) -> None:
        """Build the layer.

        Args:
            dim: Input token width.
            in_dim: Output token width.
            num_heads: Number of attention heads.
            mlp_ratio: Hidden width of the MLP relative to ``in_dim``.
            qkv_bias: Whether the QKV projection has a bias.
            qk_scale: Attention scale override.
            drop: Dropout on projections and MLP.
            attn_drop: Dropout on the attention weights.
            drop_path: Stochastic-depth probability.
            act_layer: MLP activation class.
            norm_layer: Normalisation class.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = AttentionToken(
            dim,
            in_dim=in_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(in_dim)
        self.mlp = Mlp(
            in_features=in_dim,
            hidden_features=int(in_dim * mlp_ratio),
            out_features=in_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the layer to tokens ``(B, N, dim)``."""
        x = self.attn(self.norm1(x))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class Token_performer(nn.Module):
    """Performer layer: linear-complexity attention with random features."""

    def __init__(
        self,
        dim: int,
        in_dim: int,
        head_cnt: int = 1,
        kernel_ratio: float = 0.5,
        dp1: float = 0.1,
        dp2: float = 0.1,
    ) -> None:
        """Build the layer.

        Args:
            dim: Input token width.
            in_dim: Output width per head.
            head_cnt: Number of heads.
            kernel_ratio: Number of random features relative to the embedding.
            dp1: Dropout on the attention output projection.
            dp2: Dropout in the MLP.
        """
        super().__init__()
        self.emb = in_dim * head_cnt
        self.kqv = nn.Linear(dim, 3 * self.emb)
        self.dp = nn.Dropout(dp1)
        self.proj = nn.Linear(self.emb, self.emb)
        self.head_cnt = head_cnt
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(self.emb)
        self.epsilon = 1e-8

        self.mlp = nn.Sequential(
            nn.Linear(self.emb, 1 * self.emb),
            nn.GELU(),
            nn.Linear(1 * self.emb, self.emb),
            nn.Dropout(dp2),
        )

        # Fixed orthogonal random features (stored as a frozen parameter).
        self.m = int(self.emb * kernel_ratio)
        self.w = torch.randn(self.m, self.emb)
        self.w = nn.Parameter(
            nn.init.orthogonal_(self.w) * math.sqrt(self.m), requires_grad=False
        )

    def prm_exp(self, x: torch.Tensor) -> torch.Tensor:
        """Positive random-feature map approximating the softmax kernel."""
        xd = ((x * x).sum(dim=-1, keepdim=True)).repeat(1, 1, self.m) / 2
        wtx = torch.einsum("bti,mi->btm", x.float(), self.w)
        return torch.exp(wtx - xd) / math.sqrt(self.m)

    def single_attn(self, x: torch.Tensor) -> torch.Tensor:
        """Linear attention with a residual connection on the values."""
        k, q, v = torch.split(self.kqv(x), self.emb, dim=-1)
        kp, qp = self.prm_exp(k), self.prm_exp(q)
        D = torch.einsum("bti,bi->bt", qp, kp.sum(dim=1)).unsqueeze(dim=2)
        kptv = torch.einsum("bin,bim->bnm", v.float(), kp)
        y = torch.einsum("bti,bni->btn", qp, kptv) / (
            D.repeat(1, 1, self.emb) + self.epsilon
        )
        return v + self.dp(self.proj(y))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the layer to tokens ``(B, N, dim)``."""
        x = self.single_attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


# ============================================================================
# Model: CTformer
# ============================================================================
class T2T_module(nn.Module):
    """CTformer encoder: dilated tokens-to-token with cyclic shifts.

    The unfold/fold geometry is hard-coded for 64x64 inputs, as in the
    reference implementation.
    """

    def __init__(
        self,
        img_size: int = 64,
        tokens_type: str = "performer",
        in_chans: int = 1,
        embed_dim: int = 256,
        token_dim: int = 64,
        kernel: int = 32,
        stride: int = 32,
    ) -> None:
        """Build the encoder.

        Args:
            img_size: Input size (unused; the geometry assumes 64).
            tokens_type: ``"performer"`` or ``"transformer"`` token layers.
            in_chans: Input channels.
            embed_dim: Width of the output tokens.
            token_dim: Width of the intermediate tokens.
            kernel: Unused, kept for API compatibility.
            stride: Unused, kept for API compatibility.
        """
        super().__init__()
        if tokens_type == "transformer":
            self.soft_split0 = nn.Unfold(kernel_size=(7, 7), stride=(2, 2))
            self.soft_split1 = nn.Unfold(
                kernel_size=(3, 3), stride=(1, 1), dilation=(2, 2)
            )
            self.soft_split2 = nn.Unfold(kernel_size=(3, 3), stride=(1, 1))
            self.attention1 = Token_transformer(
                dim=in_chans * 7 * 7, in_dim=token_dim, num_heads=1, mlp_ratio=1.0
            )
            self.attention2 = Token_transformer(
                dim=token_dim * 3 * 3, in_dim=token_dim, num_heads=1, mlp_ratio=1.0
            )
            self.project = nn.Linear(token_dim * 3 * 3, embed_dim)
        elif tokens_type == "performer":
            self.soft_split0 = nn.Unfold(kernel_size=(7, 7), stride=(2, 2))
            self.soft_split1 = nn.Unfold(
                kernel_size=(3, 3), stride=(1, 1), dilation=(2, 2)
            )
            self.soft_split2 = nn.Unfold(kernel_size=(3, 3), stride=(1, 1))
            self.attention1 = Token_performer(
                dim=in_chans * 7 * 7, in_dim=token_dim, kernel_ratio=0.5
            )
            self.attention2 = Token_performer(
                dim=token_dim * 3 * 3, in_dim=token_dim, kernel_ratio=0.5
            )
            self.project = nn.Linear(token_dim * 3 * 3, embed_dim)

        # 23 x 23 tokens remain after the three unfolds of a 64x64 input.
        self.num_patches = 529

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode ``(B, 1, 64, 64)`` images.

        Returns:
            The output tokens and the two intermediate token maps used as skip
            connections by the decoder.
        """
        x = self.soft_split0(x)

        x = self.attention1(x.transpose(1, 2))
        res_11 = x
        B, new_HW, C = x.shape
        side = int(np.sqrt(new_HW))
        x = x.transpose(1, 2).reshape(B, C, side, side)
        x = torch.roll(x, shifts=(2, 2), dims=(2, 3))
        x = self.soft_split1(x)

        x = self.attention2(x.transpose(1, 2))
        res_22 = x
        B, new_HW, C = x.shape
        side = int(np.sqrt(new_HW))
        x = x.transpose(1, 2).reshape(B, C, side, side)
        x = torch.roll(x, shifts=(2, 2), dims=(2, 3))
        x = self.soft_split2(x)

        x = self.project(x.transpose(1, 2))
        return x, res_11, res_22


class Token_back_Image(nn.Module):
    """CTformer decoder: inverse tokens-to-token (fold) with reverse shifts."""

    def __init__(
        self,
        img_size: int = 64,
        tokens_type: str = "performer",
        in_chans: int = 1,
        embed_dim: int = 256,
        token_dim: int = 64,
        kernel: int = 32,
        stride: int = 32,
    ) -> None:
        """Build the decoder.

        Args:
            img_size: Output size (fold sizes assume 64).
            tokens_type: ``"performer"`` or ``"transformer"`` token layers.
            in_chans: Output channels.
            embed_dim: Width of the input tokens.
            token_dim: Width of the intermediate tokens.
            kernel: Unused, kept for API compatibility.
            stride: Unused, kept for API compatibility.
        """
        super().__init__()
        if tokens_type == "transformer":
            self.soft_split0 = nn.Fold((64, 64), kernel_size=(7, 7), stride=(2, 2))
            self.soft_split1 = nn.Fold(
                (29, 29), kernel_size=(3, 3), stride=(1, 1), dilation=(2, 2)
            )
            self.soft_split2 = nn.Fold((25, 25), kernel_size=(3, 3), stride=(1, 1))
            self.attention1 = Token_transformer(
                dim=token_dim, in_dim=in_chans * 7 * 7, num_heads=1, mlp_ratio=1.0
            )
            self.attention2 = Token_transformer(
                dim=token_dim, in_dim=token_dim * 3 * 3, num_heads=1, mlp_ratio=1.0
            )
            self.project = nn.Linear(embed_dim, token_dim * 3 * 3)
        elif tokens_type == "performer":
            self.soft_split0 = nn.Fold((64, 64), kernel_size=(7, 7), stride=(2, 2))
            self.soft_split1 = nn.Fold(
                (29, 29), kernel_size=(3, 3), stride=(1, 1), dilation=(2, 2)
            )
            self.soft_split2 = nn.Fold((25, 25), kernel_size=(3, 3), stride=(1, 1))
            self.attention1 = Token_performer(
                dim=token_dim, in_dim=in_chans * 7 * 7, kernel_ratio=0.5
            )
            self.attention2 = Token_performer(
                dim=token_dim, in_dim=token_dim * 3 * 3, kernel_ratio=0.5
            )
            self.project = nn.Linear(embed_dim, token_dim * 3 * 3)

        self.num_patches = (img_size // (1 * 2 * 2)) * (img_size // (1 * 2 * 2))

    def forward(
        self, x: torch.Tensor, res_11: torch.Tensor, res_22: torch.Tensor
    ) -> torch.Tensor:
        """Decode tokens back to a ``(B, 1, 64, 64)`` image.

        Args:
            x: Tokens from the transformer blocks.
            res_11: First encoder token map (skip connection).
            res_22: Second encoder token map (skip connection).
        """
        x = self.project(x).transpose(1, 2)

        x = self.soft_split2(x)
        x = torch.roll(x, shifts=(-2, -2), dims=(-1, -2))
        x = x.flatten(2).transpose(1, 2)
        x = x + res_22
        x = self.attention2(x).transpose(1, 2)

        x = self.soft_split1(x)
        x = torch.roll(x, shifts=(-2, -2), dims=(-1, -2))
        x = x.flatten(2).transpose(1, 2)
        x = x + res_11
        x = self.attention1(x).transpose(1, 2)

        return self.soft_split0(x)


class CTformer(nn.Module):
    """CTformer denoiser: predicts the noise and subtracts it from the input.

    ``cls_token`` and ``head`` are never used in the forward pass but are kept
    so that the parameter set (and the checkpoints) match the reference code.
    """

    def __init__(
        self,
        img_size: int = 512,
        tokens_type: str = "convolution",
        in_chans: int = 1,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        kernel: int = 32,
        stride: int = 32,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        drop_rate: float = 0.1,
        attn_drop_rate: float = 0.1,
        drop_path_rate: float = 0.1,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        token_dim: int = 1024,
    ) -> None:
        """Build the network.

        Args:
            img_size: Input size.
            tokens_type: ``"performer"`` or ``"transformer"`` token layers.
            in_chans: Input channels.
            num_classes: Width of the unused classification head.
            embed_dim: Token width of the transformer blocks.
            depth: Number of transformer blocks.
            num_heads: Attention heads per block.
            kernel: Forwarded to the encoder/decoder (unused there).
            stride: Forwarded to the encoder/decoder (unused there).
            mlp_ratio: MLP hidden width relative to ``embed_dim``.
            qkv_bias: Whether the QKV projections have a bias.
            qk_scale: Attention scale override.
            drop_rate: Dropout on positions, projections and MLPs.
            attn_drop_rate: Dropout on attention weights.
            drop_path_rate: Maximum stochastic-depth probability.
            norm_layer: Normalisation class.
            token_dim: Width of the intermediate tokens.
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        self.tokens_to_token = T2T_module(
            img_size=img_size,
            tokens_type=tokens_type,
            in_chans=in_chans,
            embed_dim=embed_dim,
            token_dim=token_dim,
            kernel=kernel,
            stride=stride,
        )
        num_patches = self.tokens_to_token.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            data=get_sinusoid_encoding(n_position=num_patches, d_hid=embed_dim),
            requires_grad=False,
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        self.dconv1 = Token_back_Image(
            img_size=img_size,
            tokens_type=tokens_type,
            in_chans=in_chans,
            embed_dim=embed_dim,
            token_dim=token_dim,
            kernel=kernel,
            stride=stride,
        )
        self.head = (
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

        nn.init.trunc_normal_(self.cls_token, std=INIT_STD)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        """Truncated-normal init for linear layers, unit/zero init for LayerNorm."""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=INIT_STD)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Denoise ``(B, 1, 64, 64)`` patches."""
        res1 = x
        x, res_11, res_22 = self.tokens_to_token(x)
        x = self.pos_drop(x + self.pos_embed)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return res1 - self.dconv1(x, res_11, res_22)


def build_model() -> CTformer:
    """CTformer with the configuration of the reference ``main.py``."""
    return CTformer(
        img_size=PATCH_SIZE,
        tokens_type="performer",
        embed_dim=EMBED_DIM,
        depth=DEPTH,
        num_heads=NUM_HEADS,
        kernel=TOKEN_KERNEL,
        stride=TOKEN_STRIDE,
        mlp_ratio=MLP_RATIO,
        token_dim=TOKEN_DIM,
    )


# ============================================================================
# Sliding-window inference
# ============================================================================
def split_into_patches(
    arr: torch.Tensor, patch_size: int = PATCH_SIZE, stride: int = INFER_STRIDE
) -> tuple[torch.Tensor, int, int]:
    """Cut a zero-padded image batch into overlapping patches.

    Args:
        arr: Images ``(B, 1, H, W)``.
        patch_size: Patch side.
        stride: Distance between patch origins.

    Returns:
        Patches ``(B * num_h * num_w, 1, patch_size, patch_size)`` and the
        number of patches along each axis.
    """
    arr = F.pad(arr, (INFER_PAD,) * 4, "constant", 0)
    bsz, _, h, w = arr.shape
    num_h = h // stride - 1
    num_w = w // stride - 1
    patches = [
        arr[b : b + 1, :, i * stride : i * stride + patch_size,
            j * stride : j * stride + patch_size]
        for b in range(bsz)
        for i in range(num_h)
        for j in range(num_w)
    ]  # fmt: skip
    return torch.cat(patches, dim=0), num_h, num_w


def merge_patches(
    patches: torch.Tensor,
    out_h: int,
    out_w: int,
    num_h: int,
    num_w: int,
    batch_size: int,
    stride: int = INFER_STRIDE,
) -> torch.Tensor:
    """Reassemble the central ``stride x stride`` region of each patch.

    Args:
        patches: Output of the model on :func:`split_into_patches` patches.
        out_h: Output height.
        out_w: Output width.
        num_h: Patches along the height.
        num_w: Patches along the width.
        batch_size: Number of images.
        stride: Patch stride used when splitting.

    Returns:
        Images ``(batch_size, 1, out_h, out_w)``.
    """
    out = patches.new_zeros(batch_size, 1, out_h, out_w)
    lo, hi = INFER_PAD, INFER_PAD + stride
    k = 0
    for b in range(batch_size):
        for i in range(num_h):
            for j in range(num_w):
                out[
                    b, :, i * stride : (i + 1) * stride, j * stride : (j + 1) * stride
                ] = patches[k, :, lo:hi, lo:hi]
                k += 1
    return out


@torch.no_grad()
def ctformer_infer(
    model: nn.Module, x: torch.Tensor, chunk: int = INFER_CHUNK
) -> torch.Tensor:
    """Denoise images of any size that is a multiple of the inference stride.

    Patch-sized inputs are passed straight through the model.

    Args:
        model: Trained CTformer.
        x: Normalised images ``(B, 1, H, W)``.
        chunk: Number of patches per forward pass.
    """
    if x.shape[-2:] == (PATCH_SIZE, PATCH_SIZE):
        return model(x)
    bsz, _, h, w = x.shape
    patches, num_h, num_w = split_into_patches(x, PATCH_SIZE, stride=INFER_STRIDE)
    outs = [
        model(patches[start : start + chunk])
        for start in range(0, patches.size(0), chunk)
    ]
    return merge_patches(
        torch.cat(outs, dim=0), h, w, num_h, num_w, bsz, stride=INFER_STRIDE
    )


# ============================================================================
# Data
# ============================================================================
def _decode_dicom_to_hu(path: str) -> np.ndarray:
    """Decode a full-dose ``.IMA``/``.dcm`` file to HU (float32)."""
    ds = pydicom.dcmread(path, force=True)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    return ds.pixel_array.astype(np.float32) * slope + intercept


def _fd_cache_path(path: str, cache_dir: Path) -> Path:
    """Cache file for a DICOM target: ``<stem>_<md5(abspath)[:12]>.npy``."""
    # os.path.abspath (not Path.resolve) keeps the hashes of existing caches.
    digest = hashlib.md5(os.path.abspath(path).encode()).hexdigest()[:12]
    return cache_dir / f"{Path(path).stem}_{digest}.npy"


def ensure_fd_npy_cache(paths: list[str], cache_dir: Path) -> list[str]:
    """Convert DICOM targets to ``.npy`` (HU) once and return the cached paths.

    Decoding DICOM on every ``__getitem__`` was the data-loading bottleneck.
    The cache stores raw HU, so targets are normalised exactly as before.
    ``.npy`` targets are returned unchanged.

    Args:
        paths: Target paths (DICOM or ``.npy``).
        cache_dir: Folder for the cached arrays; shared across doses.

    Returns:
        Target paths with every DICOM file replaced by its cache.
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
        logger.info("Caching %d DICOM targets as .npy in %s", len(todo), cache_dir)
        for i, (p, cached) in enumerate(todo, 1):
            np.save(cached, _decode_dicom_to_hu(p).astype(np.float32))
            if i % FD_CACHE_LOG_EVERY == 0:
                logger.info("  cached %d/%d", i, len(todo))
        logger.info("DICOM cache ready (%d files)", len(todo))
    return out


class CTformerDataset(ULDCTDataset):
    """ULDCT dataset whose DICOM targets are read from a one-off ``.npy`` cache."""

    def __init__(
        self,
        split: str,
        data_root: Path,
        fd_cache_dir: Path,
        patch_size: int | None = None,
        patch_n: int | None = None,
    ) -> None:
        """Pair the files of ``split`` and cache the DICOM targets.

        Args:
            split: Dataset split.
            data_root: Dataset root.
            fd_cache_dir: Folder for the cached full-dose targets.
            patch_size: Training patch side, or ``None`` for full slices.
            patch_n: Patches per slice.
        """
        super().__init__(split, data_root, patch_size=patch_size, patch_n=patch_n)
        self.targets = ensure_fd_npy_cache(self.targets, Path(fd_cache_dir))


# ============================================================================
# Training and testing
# ============================================================================
def batch_metrics(pred_norm: torch.Tensor, target_norm: torch.Tensor) -> Measure:
    """PSNR, SSIM (clipped to [-1, 1]) and RMSE in HU of a training batch.

    Args:
        pred_norm: Normalised predictions.
        target_norm: Normalised targets.
    """
    p_hu = truncate(denormalize(pred_norm.detach().cpu()))
    t_hu = truncate(denormalize(target_norm.detach().cpu()))
    psnr = compute_psnr(p_hu, t_hu, DATA_RANGE)
    ssim = max(min(compute_ssim(p_hu, t_hu, DATA_RANGE), 1.0), -1.0)
    return psnr, ssim, compute_rmse(p_hu, t_hu)


@torch.no_grad()
def quick_eval_with_baseline(
    infer_fn: Callable[[torch.Tensor], torch.Tensor],
    loader: DataLoader,
    device: torch.device,
    max_slices: int = QUICK_EVAL_MAX_SLICES,
) -> tuple[Measure, Measure]:
    """Validation metrics of the prediction and of the LDCT input.

    The input-vs-target baseline separates a weak model from a misregistered
    LD/FD pair.

    Args:
        infer_fn: Maps a normalised ``(1, 1, H, W)`` input to the prediction.
        loader: Validation loader with batch size 1.
        device: Device the inputs are moved to.
        max_slices: Number of slices evaluated.

    Returns:
        ``((psnr, ssim, rmse) of pred, (psnr, ssim, rmse) of input)``, averaged.
    """
    pred_sum, input_sum, n = [0.0] * 3, [0.0] * 3, 0
    for x, y in loader:
        if n >= max_slices:
            break
        x = x.float().to(device).unsqueeze(1)
        y = y.float().to(device).unsqueeze(1)
        pred = infer_fn(x).clamp(0.0, 1.0)
        xv, yv, pv = to_hu_window(x), to_hu_window(y), to_hu_window(pred)
        for acc, img in ((input_sum, xv), (pred_sum, pv)):
            acc[0] += compute_psnr(img, yv, DATA_RANGE)
            acc[1] += compute_ssim(img, yv, DATA_RANGE)
            acc[2] += compute_rmse(img, yv)
        n += 1
    if n == 0:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    return tuple(v / n for v in pred_sum), tuple(v / n for v in input_sum)


def train(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: RunConfig,
    val_loader: DataLoader | None = None,
) -> None:
    """Train with scaled MSE/Adam, resuming from the last checkpoint if present.

    Validates and checkpoints every ``SAVE_ITERS`` iterations, keeping the
    checkpoint with the best validation PSNR.

    Args:
        model: Network to train.
        loader: Training loader yielding stacks of patches.
        device: Training device.
        cfg: Run configuration.
        val_loader: Loader for periodic validation.
    """
    model.train()
    opt = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    ckpt = load_checkpoint(cfg.last_ckpt, model=model, optimizer=opt)
    start_epoch = (ckpt["epoch"] + 1) if ckpt else 1
    step = ckpt["step"] if ckpt else 0
    if ckpt:
        logger.info(
            "Resuming from epoch=%d step=%d (psnr=%.3f); start_epoch=%d",
            ckpt["epoch"],
            ckpt["step"],
            ckpt.get("psnr", float("nan")),
            start_epoch,
        )
    else:
        logger.info("No checkpoint at %s; training from scratch", cfg.last_ckpt)
    best_psnr = best_psnr_so_far(cfg.best_ckpt)
    losses = []
    t0 = time.time()

    # Only the training forward pass is wrapped; evaluation and checkpoints use
    # the bare model, so state_dict keys carry no "module." prefix.
    net = model
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(model)
        logger.info("DataParallel on %d GPUs", torch.cuda.device_count())

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
            pred = net(x)
            loss = criterion(pred, y) * LOSS_SCALE + LOSS_OFFSET
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

            # The reference solver sets lr = initial_lr * 0.5 (it never updates
            # its stored lr), so the decay is one-shot, not compounded.
            if LR_DECAY_ITERS and step % LR_DECAY_ITERS == 0:
                for group in opt.param_groups:
                    group["lr"] = LR * LR_DECAY_FACTOR

            if step % PRINT_ITERS == 0:
                with torch.no_grad():
                    psnr, ssim, rmse = batch_metrics(pred.clamp(0.0, 1.0), y)
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

            if step % SAVE_ITERS == 0:
                model.eval()
                if val_loader is not None:
                    (psnr, ssim, rmse), (in_psnr, in_ssim, in_rmse) = (
                        quick_eval_with_baseline(
                            lambda xb: ctformer_infer(model, xb), val_loader, device
                        )
                    )
                else:
                    psnr, ssim, rmse = 0.0, 0.0, 0.0
                    in_psnr, in_ssim, in_rmse = 0.0, 0.0, 0.0
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
                    "  saved last.pt (epoch=%d step=%d psnr=%.3f ssim=%.4f "
                    "rmse_hu=%.3f) best_psnr=%.3f",
                    epoch,
                    step,
                    psnr,
                    ssim,
                    rmse,
                    best_psnr,
                )
                logger.info(
                    "    baseline(input) psnr=%.3f ssim=%.4f rmse_hu=%.3f  "
                    "gain=%+.3f dB",
                    in_psnr,
                    in_ssim,
                    in_rmse,
                    psnr - in_psnr,
                )


def test(
    model: nn.Module, loader: DataLoader, device: torch.device, cfg: RunConfig
) -> None:
    """Evaluate on the full test split with sliding-window inference.

    Args:
        model: Trained network.
        loader: Test loader with batch size 1.
        device: Inference device.
        cfg: Run configuration.
    """
    model.eval()

    def predict(
        batch: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y = batch
        x = x.float().to(device).unsqueeze(1)
        y = y.float().to(device).unsqueeze(1)
        pred = ctformer_infer(model, x).clamp(0.0, 1.0)
        return to_hu_window(x), to_hu_window(y), to_hu_window(pred)

    run_test(loader, predict, cfg)


def main() -> None:
    """Train CTformer on the training split, then test the best checkpoint."""
    parser = build_arg_parser(MODEL_NAME, description=__doc__)
    parser.add_argument(
        "--fd-cache-dir",
        type=Path,
        default=Path("fd_npy_cache"),
        help="Folder for the .npy cache of the DICOM full-dose targets.",
    )
    cfg, args = parse_run_config(parser, MODEL_NAME)
    torch.backends.cudnn.benchmark = True
    device = get_device()
    logger.info(
        "[%s] dose=%s device=%s out=%s", MODEL_NAME, cfg.dose, device, cfg.output_dir
    )

    def make_dataset(split: str, **kwargs: Any) -> CTformerDataset:
        return CTformerDataset(split, cfg.data_root, args.fd_cache_dir, **kwargs)

    train_ds = make_dataset(TRAIN_SPLIT, patch_size=PATCH_SIZE, patch_n=PATCH_N)
    test_ds = make_dataset(TEST_SPLIT)
    val_ds = optional_split(lambda: make_dataset(VAL_SPLIT), VAL_SPLIT)
    logger.info(
        "train files: %d | val files: %d | test files: %d",
        len(train_ds),
        len(val_ds) if val_ds is not None else 0,
        len(test_ds),
    )

    # Worker-only options are skipped with --num-workers 0, where they are invalid.
    loader_kwargs: dict[str, Any] = {"num_workers": cfg.num_workers}
    if cfg.num_workers > 0:
        loader_kwargs.update(
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=DATALOADER_PREFETCH,
        )
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, **loader_kwargs
    )
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, **loader_kwargs)
    val_loader = (
        DataLoader(val_ds, batch_size=1, shuffle=False, **loader_kwargs)
        if val_ds is not None
        else test_loader
    )

    model = build_model().to(device)
    train(model, train_loader, device, cfg, val_loader=val_loader)
    load_best_for_test(cfg, model)
    test(model, test_loader, device, cfg)


if __name__ == "__main__":
    main()
