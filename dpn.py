"""DPN (Yang et al., 2023) trained and evaluated on the ULDCT dataset.

From-scratch reimplementation of "Low-dose CT denoising with a high-level
feature refinement and dynamic convolution network", Med. Phys. 50(6):3597-3611,
2023 (DOI: 10.1002/mp.16175); no official code is public.

Architecture (paper Sec. 2.2), RDB -> DPBnet topology (best in paper Table 1):
    - FRN (Feature Refinement Network, eqs. 3-6): three Residual Dense Blocks
      (eqs. 7-9). Its intermediate features refine the DPN.
    - DPN (Dynamic Perception Network, eqs. 10-13): mirrors the FRN with three
      Dynamic Perception Blocks. Each DPB is a Local Channel Attention
      (eqs. 14-16: per-quadrant squeeze-excitation) followed by four Dynamic
      Dilated Convolutions (eqs. 17-20: 8 softmax-aggregated 3x3 base kernels,
      dilations 2/3/4/5) and a 1x1 fusion.

Training (paper Sec. 2.3): L1 loss, Adam, batch 16, 64x64 patches, lr 1e-4
halved every 100 epochs over 500 epochs (one continuous schedule across both
stages); stage 1 trains the FRN alone, stage 2 the whole network. Augmentation:
rotations by multiples of 90 degrees and horizontal/vertical flips.

Deviations from the paper: the dataset (ULDCT 5%/10% dose with normal-dose
targets instead of AAPM-Mayo quarter dose) and the number of stage-1 epochs
(``FRN_PRETRAIN_EPOCHS``), which the paper does not specify.

Usage:
    python dpn.py --dose 5pct --data-root /path/to/uldct_5pct/dataset
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader

from common.checkpoint import best_psnr_so_far, load_checkpoint, save_checkpoint
from common.config import (
    TEST_SPLIT,
    TRAIN_SPLIT,
    VAL_SPLIT,
    RunConfig,
    build_arg_parser,
    get_device,
    parse_run_config,
)
from common.data import (
    DICOM_SUFFIXES,
    ULDCTDataset,
    load_input_array,
    load_target_array,
    random_patch_pairs,
    to_hu_window,
)
from common.evaluation import quick_eval, run_test
from common.runtime import load_best_for_test, optional_split

MODEL_NAME = "dpn"

# Hyperparameters (paper Sec. 2.3.2 unless noted).
NUM_EPOCHS = 500
FRN_PRETRAIN_EPOCHS = 100  # stage-1 length; not specified by the paper
BATCH_SIZE = 16
PATCH_SIZE = 64
PATCH_N = 1
REPEAT_PER_EPOCH = 20  # each training epoch iterates over the data 20 times
BASE_CH = 64
LR = 1e-4
LR_STEP_EPOCHS = 100  # lr halved every 100 epochs
LR_GAMMA = 0.5
PRINT_ITERS = 50
SAVE_EPOCHS = 5

NUM_DDC_KERNELS = 8
DDC_DILATIONS = (2, 3, 4, 5)
SE_REDUCTION = 16
DDC_REDUCTION = 4
MIN_HIDDEN_CHANNELS = 4

logger = logging.getLogger(MODEL_NAME)


# ============================================================================
# Model
# ============================================================================
class RDB(nn.Module):
    """Residual Dense Block (paper Fig. 2, eqs. 7-9)."""

    def __init__(self, channels: int = 64, growth: int = 64, n_layers: int = 4) -> None:
        """Build the block.

        Args:
            channels: Input/output channels.
            growth: Channels added by each dense layer.
            n_layers: Number of densely connected 3x3 conv + ReLU layers.
        """
        super().__init__()
        self.n_layers = n_layers
        # eq. 8: layer m sees the input plus the outputs of all previous layers.
        self.convs = nn.ModuleList(
            nn.Conv2d(channels + m * growth, growth, 3, padding=1)
            for m in range(n_layers)
        )
        # eq. 9: 1x1 local feature fusion back to `channels`.
        self.lff = nn.Conv2d(channels + n_layers * growth, channels, 1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the dense layers, fuse them and add the local residual."""
        feats = [x]
        for conv in self.convs:
            feats.append(self.act(conv(torch.cat(feats, dim=1))))
        return self.lff(torch.cat(feats, dim=1)) + x


def _spatial_split_4(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split ``(B, C, H, W)`` into its top-left, top-right, bottom-left and
    bottom-right quadrants."""
    h, w = x.shape[-2:]
    h2, w2 = h // 2, w // 2
    return (
        x[:, :, :h2, :w2],
        x[:, :, :h2, w2:],
        x[:, :, h2:, :w2],
        x[:, :, h2:, w2:],
    )


def _spatial_join_4(
    tl: torch.Tensor, tr: torch.Tensor, bl: torch.Tensor, br: torch.Tensor
) -> torch.Tensor:
    """Inverse of :func:`_spatial_split_4`."""
    top = torch.cat([tl, tr], dim=-1)
    bot = torch.cat([bl, br], dim=-1)
    return torch.cat([top, bot], dim=-2)


class _SE(nn.Module):
    """Squeeze-excitation applied to each quadrant by :class:`LCA`."""

    def __init__(self, channels: int, reduction: int = SE_REDUCTION) -> None:
        """Build the gating MLP.

        Args:
            channels: Number of channels.
            reduction: Channel reduction ratio of the hidden layer.
        """
        super().__init__()
        hidden = max(channels // reduction, MIN_HIDDEN_CHANNELS)
        self.fc1 = nn.Conv2d(channels, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, channels, 1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Rescale the channels of ``x`` by their learned gates."""
        s = F.adaptive_avg_pool2d(x, 1)
        s = self.act(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s


class LCA(nn.Module):
    """Local Channel Attention (paper Fig. 3, eqs. 14-16).

    Fuses the previous DPN features with the matching FRN features, then
    applies a shared squeeze-excitation independently to each quadrant.
    """

    def __init__(self, channels: int = 64, reduction: int = SE_REDUCTION) -> None:
        """Build the shared squeeze-excitation.

        Args:
            channels: Number of channels.
            reduction: Channel reduction ratio of the squeeze-excitation.
        """
        super().__init__()
        self.se = _SE(channels, reduction)

    def forward(self, If_prev: torch.Tensor, Ia_n: torch.Tensor) -> torch.Tensor:
        """Attend ``If_prev + Ia_n`` per quadrant.

        Args:
            If_prev: Output of the previous DPN stage.
            Ia_n: Intermediate FRN feature of the same depth.
        """
        x = If_prev + Ia_n  # eq. 14
        tl, tr, bl, br = (self.se(q) for q in _spatial_split_4(x))
        return _spatial_join_4(tl, tr, bl, br)


class DDC(nn.Module):
    """Dynamic Dilated Convolution (paper Fig. 4, eqs. 18-19).

    The per-sample kernel is a softmax-weighted sum of ``n_kernels`` dilated
    3x3 base kernels, applied with a grouped convolution (one group per sample).
    """

    def __init__(
        self,
        channels: int = 16,
        dilation: int = 2,
        n_kernels: int = NUM_DDC_KERNELS,
        reduction: int = DDC_REDUCTION,
    ) -> None:
        """Build the base kernels and the attention head.

        Args:
            channels: Input/output channels.
            dilation: Dilation rate; the padding equals it to keep the size.
            n_kernels: Number of base kernels.
            reduction: Channel reduction ratio of the attention head.
        """
        super().__init__()
        self.channels = channels
        self.dilation = dilation
        self.n_kernels = n_kernels
        self.padding = dilation

        # eq. 19: K base kernels of shape (C_out, C_in, 3, 3).
        self.weight = nn.Parameter(torch.empty(n_kernels, channels, channels, 3, 3))
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")

        # eq. 18: kernel attention.
        hidden = max(channels // reduction, MIN_HIDDEN_CHANNELS)
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, n_kernels, 1),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convolve each sample with its own aggregated kernel."""
        B, C, H, W = x.shape
        a = F.softmax(self.attn(x).flatten(1), dim=1)  # (B, K)
        agg_w = torch.einsum("bk,kocij->bocij", a, self.weight)
        # Fold the batch into channels so that groups=B applies one kernel per sample.
        x_g = x.reshape(1, B * C, H, W)
        w_g = agg_w.reshape(B * C, C, 3, 3)
        out = F.conv2d(x_g, w_g, padding=self.padding, dilation=self.dilation, groups=B)
        return self.act(out.reshape(B, C, H, W))


class DPB(nn.Module):
    """Dynamic Perception Block (paper Fig. 3, eqs. 17-20).

    LCA, channel split into four groups, one DDC per group with dilations
    2/3/4/5, and a 1x1 fusion.
    """

    def __init__(self, channels: int = 64) -> None:
        """Build the block.

        Args:
            channels: Number of channels; must be divisible by 4.
        """
        super().__init__()
        assert channels % 4 == 0, "channels must be divisible by 4 (channel split)"
        group_ch = channels // 4
        self.lca = LCA(channels)
        self.ddc1 = DDC(group_ch, dilation=DDC_DILATIONS[0])
        self.ddc2 = DDC(group_ch, dilation=DDC_DILATIONS[1])
        self.ddc3 = DDC(group_ch, dilation=DDC_DILATIONS[2])
        self.ddc4 = DDC(group_ch, dilation=DDC_DILATIONS[3])
        self.fuse = nn.Conv2d(channels, channels, 1)

    def forward(self, If_prev: torch.Tensor, Ia_n: torch.Tensor) -> torch.Tensor:
        """Compute the next DPN feature map.

        Args:
            If_prev: Output of the previous DPN stage.
            Ia_n: Intermediate FRN feature of the same depth.
        """
        cs = torch.chunk(self.lca(If_prev, Ia_n), 4, dim=1)
        d1 = self.ddc1(cs[0])
        d2 = self.ddc2(cs[1])
        d3 = self.ddc3(cs[2])
        d4 = self.ddc4(cs[3])
        # eq. 20 has no residual at block level; the skips are at network level
        # (eqs. 12-13).
        return self.fuse(torch.cat([d1, d2, d3, d4], dim=1))


class FRN(nn.Module):
    """Feature Refinement Network (paper eqs. 3-6)."""

    def __init__(self, in_ch: int = 1, base_ch: int = 64) -> None:
        """Build the network.

        Args:
            in_ch: Image channels.
            base_ch: Feature channels.
        """
        super().__init__()
        # eq. 3: Ia0 = ReLU(C3x3(C3x3(x)))
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, padding=1),
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.rdb1 = RDB(base_ch)
        self.rdb2 = RDB(base_ch)
        self.rdb3 = RDB(base_ch)
        self.lff = nn.Conv2d(base_ch * 3, base_ch, 1)  # eq. 4
        self.merge = nn.Conv2d(base_ch, base_ch, 3, padding=1)  # eq. 5
        self.tail = nn.Conv2d(base_ch, in_ch, 3, padding=1)  # eq. 6
        self.act = nn.ReLU(inplace=True)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Denoise ``x``.

        Returns:
            The FRN output and the intermediate features ``(Ia1, Ia2, Ia3)``
            passed to the DPN.
        """
        Ia0 = self.head(x)
        Ia1 = self.rdb1(Ia0)
        Ia2 = self.rdb2(Ia1)
        Ia3 = self.rdb3(Ia2)
        Ia4 = self.lff(torch.cat([Ia1, Ia2, Ia3], dim=1))
        Ia5 = self.act(Ia0 + self.merge(Ia4))
        return x + self.tail(Ia5), (Ia1, Ia2, Ia3)


class DPN(nn.Module):
    """Dynamic Perception Network (paper eqs. 10-13).

    Mirrors the FRN with DPBs and takes the FRN's intermediate features as
    cross-network priors.
    """

    def __init__(self, in_ch: int = 1, base_ch: int = 64) -> None:
        """Build the network.

        Args:
            in_ch: Image channels.
            base_ch: Feature channels.
        """
        super().__init__()
        self.head = nn.Sequential(  # eq. 10
            nn.Conv2d(in_ch, base_ch, 3, padding=1),
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.dpb1 = DPB(base_ch)
        self.dpb2 = DPB(base_ch)
        self.dpb3 = DPB(base_ch)
        self.lff = nn.Conv2d(base_ch * 3, base_ch, 1)  # eq. 11
        self.merge = nn.Conv2d(base_ch, base_ch, 3, padding=1)  # eq. 12
        self.tail = nn.Conv2d(base_ch, in_ch, 3, padding=1)  # eq. 13
        self.act = nn.ReLU(inplace=True)

    def forward(
        self,
        x: torch.Tensor,
        frn_feats: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Denoise ``x`` guided by the FRN features ``(Ia1, Ia2, Ia3)``."""
        Ia1, Ia2, Ia3 = frn_feats
        If0 = self.head(x)
        If1 = self.dpb1(If0, Ia1)
        If2 = self.dpb2(If1, Ia2)
        If3 = self.dpb3(If2, Ia3)
        If4 = self.lff(torch.cat([If1, If2, If3], dim=1))
        If5 = self.act(If0 + self.merge(If4))
        return x + self.tail(If5)


class DPNDenoiser(nn.Module):
    """Full dual network: FRN followed by the FRN-guided DPN."""

    def __init__(self, in_ch: int = 1, base_ch: int = BASE_CH) -> None:
        """Build both sub-networks.

        Args:
            in_ch: Image channels.
            base_ch: Feature channels.
        """
        super().__init__()
        self.frn = FRN(in_ch, base_ch)
        self.dpn = DPN(in_ch, base_ch)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(frn_out, dpn_out)``; ``dpn_out`` is the final prediction."""
        frn_out, frn_feats = self.frn(x)
        return frn_out, self.dpn(x, frn_feats)


# ============================================================================
# Data
# ============================================================================
def augment_pair_dpn(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply the paper's augmentation (Sec. 2.3.1) identically to a pair.

    Random rotation by 0/90/180/270 degrees (``rot90``, no interpolation),
    then random horizontal and vertical flips.

    Args:
        x: Input image ``(H, W)``.
        y: Target image ``(H, W)``.

    Returns:
        The augmented, contiguous ``float32`` pair.
    """
    k = int(np.random.randint(0, 4))
    if k:
        x, y = np.rot90(x, k), np.rot90(y, k)
    if np.random.rand() < 0.5:
        x, y = np.fliplr(x), np.fliplr(y)
    if np.random.rand() < 0.5:
        x, y = np.flipud(x), np.flipud(y)
    return (
        np.ascontiguousarray(x, dtype=np.float32),
        np.ascontiguousarray(y, dtype=np.float32),
    )


def load_target_array_cached(path: str) -> np.ndarray:
    """Load a normalised target, caching decoded DICOMs as ``<path>.npy``.

    DICOM decoding is CPU-bound and would otherwise run for every item of
    every epoch. The cache stores the already normalised array, so results
    are unchanged. Writes are atomic because several DataLoader workers may
    decode the same file; read-only file systems simply run without a cache.

    Args:
        path: Target file (``.IMA``/``.dcm`` or ``.npy``).
    """
    if not path.lower().endswith(DICOM_SUFFIXES):
        return load_target_array(path)
    cache = Path(path + ".npy")
    if cache.exists():
        try:
            return np.load(cache).astype(np.float32)
        except Exception:
            pass  # corrupted cache: decode again
    arr = load_target_array(path)
    try:
        tmp = Path(f"{cache}.tmp{os.getpid()}")
        with tmp.open("wb") as fh:
            np.save(fh, arr)
        tmp.replace(cache)
    except Exception:
        pass
    return arr


class DPNDataset(ULDCTDataset):
    """ULDCT dataset with DPN augmentation, epoch repetition and target cache.

    The training split is repeated ``REPEAT_PER_EPOCH`` times per epoch.
    """

    def __init__(
        self,
        split: str,
        data_root: Path,
        patch_size: int | None = None,
        patch_n: int | None = None,
    ) -> None:
        """Discover and pair the files of ``split``.

        Args:
            split: Dataset split.
            data_root: Dataset root.
            patch_size: Training patch side, or ``None`` for full slices.
            patch_n: Patches per slice.
        """
        self.repeat = REPEAT_PER_EPOCH if split == TRAIN_SPLIT else 1
        super().__init__(split, data_root, patch_size=patch_size, patch_n=patch_n)

    def __len__(self) -> int:
        """Number of paired slices times the per-epoch repetition."""
        return len(self.inputs) * self.repeat

    def load_pair(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Load the normalised input and (cached) target slices at ``idx``."""
        return (
            load_input_array(self.inputs[idx]),
            load_target_array_cached(self.targets[idx]),
        )

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the slice pair, or augmented random patches when patching."""
        x, y = self.load_pair(idx % len(self.inputs))
        if self.patch_size:
            x, y = augment_pair_dpn(x, y)
            return random_patch_pairs(x, y, self.patch_size, self.patch_n)
        return x, y


# ============================================================================
# Training and testing
# ============================================================================
def _amp_device() -> str:
    """Autocast device type; mixed precision is only enabled on CUDA."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def train_step_full(
    model: DPNDenoiser,
    ldct: torch.Tensor,
    ndct: torch.Tensor,
    optimizer: optim.Optimizer,
    scaler: torch.amp.GradScaler,
) -> dict[str, float]:
    """One stage-2 step: L1 loss on the final DPN output (paper Sec. 2.3.2).

    Args:
        model: Full network.
        ldct: Input patches ``(N, 1, H, W)``.
        ndct: Target patches ``(N, 1, H, W)``.
        optimizer: Optimizer over all parameters.
        scaler: Mixed-precision gradient scaler.

    Returns:
        The training loss and, for monitoring only, the FRN output's L1 loss.
    """
    model.train()
    optimizer.zero_grad()
    with torch.autocast(_amp_device(), enabled=scaler.is_enabled()):
        frn_out, dpn_out = model(ldct)
        loss = F.l1_loss(dpn_out, ndct)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return {
        "loss": loss.item(),
        "loss_frn": F.l1_loss(frn_out.float(), ndct).detach().item(),
        "loss_dpn": loss.item(),
    }


def train_step_frn_only(
    model: DPNDenoiser,
    ldct: torch.Tensor,
    ndct: torch.Tensor,
    optimizer: optim.Optimizer,
    scaler: torch.amp.GradScaler,
) -> float:
    """One stage-1 step: L1 loss on the FRN output alone.

    Args:
        model: Full network (only ``model.frn`` is used).
        ldct: Input patches ``(N, 1, H, W)``.
        ndct: Target patches ``(N, 1, H, W)``.
        optimizer: Optimizer over the FRN parameters.
        scaler: Mixed-precision gradient scaler.

    Returns:
        The training loss.
    """
    model.train()
    optimizer.zero_grad()
    with torch.autocast(_amp_device(), enabled=scaler.is_enabled()):
        frn_out, _ = model.frn(ldct)
        loss = F.l1_loss(frn_out, ndct)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return loss.item()


def _to_patch_batch(
    x: torch.Tensor, y: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Move a training batch to ``device`` as ``(N, 1, PATCH_SIZE, PATCH_SIZE)``."""
    x = x.float().to(device, non_blocking=True)
    y = y.float().to(device, non_blocking=True)
    if x.dim() == 4:
        return (
            x.view(-1, 1, PATCH_SIZE, PATCH_SIZE),
            y.view(-1, 1, PATCH_SIZE, PATCH_SIZE),
        )
    return x.unsqueeze(1), y.unsqueeze(1)


def train(
    model: DPNDenoiser,
    loader: DataLoader,
    device: torch.device,
    cfg: RunConfig,
    val_loader: DataLoader | None = None,
) -> None:
    """Two-stage training, resuming each stage from its own checkpoint.

    Stage 1 trains the FRN alone for ``FRN_PRETRAIN_EPOCHS``; stage 2 trains
    the whole network up to ``NUM_EPOCHS``. Validation runs once per epoch and
    the stage-2 checkpoint with the best PSNR is kept.

    Args:
        model: Network to train.
        loader: Training loader yielding patches.
        device: Training device.
        cfg: Run configuration.
        val_loader: Loader for per-epoch validation.
    """
    last_s1 = cfg.output_dir / f"{cfg.run_name}_last_stage1.pt"
    last_s2 = cfg.output_dir / f"{cfg.run_name}_last_stage2.pt"
    # One scaler shared by both stages.
    scaler = torch.amp.GradScaler(_amp_device(), enabled=torch.cuda.is_available())

    # Stage 1: FRN only.
    opt_frn = optim.Adam(model.frn.parameters(), lr=LR)
    sched_frn = optim.lr_scheduler.StepLR(
        opt_frn, step_size=LR_STEP_EPOCHS, gamma=LR_GAMMA
    )
    ckpt = load_checkpoint(last_s1, model=model, optimizer=opt_frn, scheduler=sched_frn)
    start_epoch = (ckpt["epoch"] + 1) if ckpt else 1
    step = ckpt["step"] if ckpt else 0
    losses_frn = []
    t0 = time.time()
    logger.info("[stage1] FRN-only pretrain %d..%d", start_epoch, FRN_PRETRAIN_EPOCHS)
    for epoch in range(start_epoch, FRN_PRETRAIN_EPOCHS + 1):
        for x, y in loader:
            step += 1
            x, y = _to_patch_batch(x, y, device)
            loss = train_step_frn_only(model, x, y, opt_frn, scaler)
            losses_frn.append(loss)
            if step % PRINT_ITERS == 0:
                logger.info(
                    "[s1] epoch %d/%d  iter %d  loss_frn %.6f  (%.0fs)",
                    epoch,
                    FRN_PRETRAIN_EPOCHS,
                    step,
                    loss,
                    time.time() - t0,
                )
        sched_frn.step()
        model.eval()
        psnr, ssim, rmse = (
            quick_eval(lambda xb: model.frn(xb)[0], val_loader, device)
            if val_loader is not None
            else (0.0, 0.0, 0.0)
        )
        model.train()
        last_loss = losses_frn[-1] if losses_frn else 0.0
        logger.info(
            "[s1] epoch %d/%d  iter %d  loss_frn %.6f  val PSNR %6.2f  SSIM %.4f  "
            "RMSE %6.2f HU  (%.0fs)",
            epoch,
            FRN_PRETRAIN_EPOCHS,
            step,
            last_loss,
            psnr,
            ssim,
            rmse,
            time.time() - t0,
        )
        if epoch % SAVE_EPOCHS == 0 or epoch == FRN_PRETRAIN_EPOCHS:
            save_checkpoint(
                last_s1,
                model=model,
                optimizer=opt_frn,
                scheduler=sched_frn,
                epoch=epoch,
                step=step,
                lr=opt_frn.param_groups[0]["lr"],
                loss=last_loss,
                psnr=psnr,
                ssim=ssim,
                rmse_hu=rmse,
            )
            np.save(
                cfg.output_dir / f"losses_stage1_{cfg.dose}.npy", np.array(losses_frn)
            )
            logger.info(
                "  [s1] saved last_stage1.pt (epoch=%d psnr=%.3f ssim=%.4f "
                "rmse_hu=%.3f)",
                epoch,
                psnr,
                ssim,
                rmse,
            )

    # Stage 2: joint FRN + DPN. The lr schedule is continuous over all epochs,
    # so stage 2 starts from the lr already decayed during stage 1 and its
    # StepLR keeps halving every LR_STEP_EPOCHS global epochs.
    s2_lr0 = LR * (LR_GAMMA ** (FRN_PRETRAIN_EPOCHS // LR_STEP_EPOCHS))
    opt = optim.Adam(model.parameters(), lr=s2_lr0)
    sched = optim.lr_scheduler.StepLR(opt, step_size=LR_STEP_EPOCHS, gamma=LR_GAMMA)
    ckpt = load_checkpoint(last_s2, model=model, optimizer=opt, scheduler=sched)
    if ckpt is None:
        # Start stage 2 from the trained stage-1 FRN rather than whatever is in
        # memory; an untrained FRN collapses the DPN's PSNR.
        s1_ckpt = load_checkpoint(last_s1, model=model)
        if s1_ckpt is not None:
            logger.info(
                "[s2] FRN inherited from last_stage1.pt (epoch=%s psnr=%s)",
                s1_ckpt.get("epoch"),
                s1_ckpt.get("psnr"),
            )
        else:
            logger.warning(
                "[s2] last_stage1.pt not found; FRN not inherited "
                "(stage 2 trains from scratch)"
            )
    start_epoch_s2 = (ckpt["epoch"] + 1) if ckpt else (FRN_PRETRAIN_EPOCHS + 1)
    step_s2 = ckpt["step"] if ckpt else step
    best_psnr = best_psnr_so_far(cfg.best_ckpt)
    losses = []
    logger.info("[stage2] joint train %d..%d", start_epoch_s2, NUM_EPOCHS)
    for epoch in range(start_epoch_s2, NUM_EPOCHS + 1):
        for x, y in loader:
            step_s2 += 1
            x, y = _to_patch_batch(x, y, device)
            stats = train_step_full(model, x, y, opt, scaler)
            losses.append(stats["loss"])
            if step_s2 % PRINT_ITERS == 0:
                logger.info(
                    "[s2] epoch %d/%d  iter %d  loss %.6f  loss_frn %.6f  (%.0fs)",
                    epoch,
                    NUM_EPOCHS,
                    step_s2,
                    stats["loss"],
                    stats["loss_frn"],
                    time.time() - t0,
                )
        sched.step()
        model.eval()
        psnr, ssim, rmse = (
            quick_eval(lambda xb: model(xb)[1], val_loader, device)
            if val_loader is not None
            else (0.0, 0.0, 0.0)
        )
        model.train()
        last_loss = losses[-1] if losses else 0.0
        logger.info(
            "[s2] epoch %d/%d  iter %d  loss %.6f  val PSNR %6.2f  SSIM %.4f  "
            "RMSE %6.2f HU  (%.0fs)",
            epoch,
            NUM_EPOCHS,
            step_s2,
            last_loss,
            psnr,
            ssim,
            rmse,
            time.time() - t0,
        )
        if epoch % SAVE_EPOCHS == 0 or epoch == NUM_EPOCHS:
            state = dict(
                model=model,
                optimizer=opt,
                scheduler=sched,
                epoch=epoch,
                step=step_s2,
                lr=opt.param_groups[0]["lr"],
                loss=last_loss,
                psnr=psnr,
                ssim=ssim,
                rmse_hu=rmse,
            )
            save_checkpoint(last_s2, **state)
            if psnr > best_psnr:
                best_psnr = psnr
                save_checkpoint(cfg.best_ckpt, **state)
            np.save(cfg.losses_path, np.array(losses))
            logger.info(
                "  [s2] saved last_stage2.pt (epoch=%d psnr=%.3f ssim=%.4f "
                "rmse_hu=%.3f) best_psnr=%.3f",
                epoch,
                psnr,
                ssim,
                rmse,
                best_psnr,
            )


def test(
    model: DPNDenoiser, loader: DataLoader, device: torch.device, cfg: RunConfig
) -> None:
    """Evaluate the DPN output on the full test split.

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
        _, dpn_out = model(x)
        return to_hu_window(x), to_hu_window(y), to_hu_window(dpn_out.clamp(0.0, 1.0))

    run_test(loader, predict, cfg)


def main() -> None:
    """Train DPN in two stages, then test the best checkpoint."""
    parser = build_arg_parser(MODEL_NAME, description=__doc__)
    cfg, _ = parse_run_config(parser, MODEL_NAME)
    torch.backends.cudnn.benchmark = True
    device = get_device()
    logger.info(
        "[%s] dose=%s device=%s out=%s", MODEL_NAME, cfg.dose, device, cfg.output_dir
    )

    train_ds = DPNDataset(
        TRAIN_SPLIT, cfg.data_root, patch_size=PATCH_SIZE, patch_n=PATCH_N
    )
    test_ds = DPNDataset(TEST_SPLIT, cfg.data_root)
    val_ds = optional_split(lambda: DPNDataset(VAL_SPLIT, cfg.data_root), VAL_SPLIT)
    # The training count includes the REPEAT_PER_EPOCH repetition.
    logger.info(
        "train files: %d | val files: %d | test files: %d",
        len(train_ds),
        len(val_ds) if val_ds is not None else 0,
        len(test_ds),
    )

    loader_kwargs = dict(
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
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

    model = DPNDenoiser(in_ch=1, base_ch=BASE_CH).to(device)
    train(model, train_loader, device, cfg, val_loader=val_loader)
    load_best_for_test(cfg, model)
    test(model, test_loader, device, cfg)


if __name__ == "__main__":
    main()
