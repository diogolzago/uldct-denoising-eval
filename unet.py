"""U-Net (Ronneberger et al., 2015) trained and evaluated on the ULDCT dataset.

Faithful to the original segmentation U-Net: unpadded 3x3 convolutions without
batch normalisation, crop-and-copy skip connections, 2x2 up-convolutions, He
initialisation and a pixel-wise soft-max with weighted cross-entropy. To fit
this classification setting, the continuous full-dose target is binarised into
two intensity classes, and each predicted class is mapped back to the midpoint
of its intensity bin for evaluation.

Usage:
    python unet.py --dose 5pct --data-root /path/to/uldct_5pct/dataset
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable
from math import pi
from pathlib import Path

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
from common.data import (
    ULDCTDataset,
    load_target_array,
    random_patch_pairs,
    to_hu_window,
)
from common.evaluation import QUICK_EVAL_MAX_SLICES, run_test
from common.metrics import Measure, compute_psnr, compute_rmse, compute_ssim
from common.runtime import optional_split

MODEL_NAME = "unet"

# Hyperparameters (paper Sec. 3 "Training" unless noted).
NUM_EPOCHS = 100
BATCH_SIZE = 1  # paper: large input tiles, batch size 1
PATCH_SIZE = 572  # paper Fig. 1: a 572x572 input tile ...
OUTPUT_SIZE = 388  # ... yields a 388x388 output
PATCH_N = 1  # one tile per image
LR = 0.01  # not given in the paper; Caffe SGD default of the time
MOMENTUM = 0.99  # paper
N_CLASSES = 2  # paper: two classes
SAVE_EPOCHS = 1
DROPOUT = 0.5  # paper: dropout at the end of the contracting path, rate unstated
# Threshold on the normalised target separating the two classes; 0.26 ~ 40 HU,
# the centre of the [-160, 240] HU evaluation window, for balanced classes.
HU_BINARIZE_THRESHOLD = 0.26

# Elastic deformation (paper Sec. 3): random displacements on a coarse 3x3
# grid with a 10 px standard deviation, upsampled bicubically.
ELASTIC_GRID = 3
ELASTIC_SIGMA_PX = 10.0

CLASS_WEIGHT_SAMPLES = 64
CLASS_WEIGHT_SEED = 0
MIN_CLASS_FREQ = 1e-4

logger = logging.getLogger(MODEL_NAME)


# ============================================================================
# Model
# ============================================================================
class DoubleConv(nn.Module):
    """Two unpadded 3x3 convolutions, each followed by a ReLU (no batch norm)."""

    def __init__(
        self, in_channels: int, out_channels: int, mid_channels: int | None = None
    ) -> None:
        """Build the block.

        Args:
            in_channels: Input feature maps.
            out_channels: Output feature maps.
            mid_channels: Feature maps after the first convolution; defaults to
                ``out_channels``.
        """
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=0, bias=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the two convolutions."""
        return self.double_conv(x)


class Down(nn.Module):
    """2x2 max pooling followed by a :class:`DoubleConv`."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Build the block.

        Args:
            in_channels: Input feature maps.
            out_channels: Output feature maps.
        """
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Downsample and convolve."""
        return self.maxpool_conv(x)


def center_crop(t: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """Crop the centre ``(target_h, target_w)`` region of a ``(B, C, H, W)`` tensor.

    This is the "copy and crop" operation of paper Fig. 1.
    """
    _, _, h, w = t.shape
    top = (h - target_h) // 2
    left = (w - target_w) // 2
    return t[:, :, top : top + target_h, left : left + target_w]


class Up(nn.Module):
    """2x2 up-convolution, crop-and-copy skip connection and :class:`DoubleConv`."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Build the block.

        Args:
            in_channels: Input feature maps (halved by the up-convolution).
            out_channels: Output feature maps.
        """
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Upsample ``x1`` and fuse it with the skip features ``x2``.

        Args:
            x1: Features from the previous decoder level.
            x2: Encoder skip features; larger than the upsampled ``x1`` because
                the convolutions are unpadded, so they are centre-cropped.
        """
        x1 = self.up(x1)
        x2 = center_crop(x2, x1.size(2), x1.size(3))
        return self.conv(torch.cat([x2, x1], dim=1))


class OutConv(nn.Module):
    """1x1 convolution mapping features to class logits."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Build the layer.

        Args:
            in_channels: Input feature maps.
            out_channels: Number of classes.
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the logits."""
        return self.conv(x)


class UNet(nn.Module):
    """Original U-Net with 23 convolutional layers.

    A 572x572 input produces 388x388 class logits.
    """

    def __init__(self, n_channels: int = 1, n_classes: int = 2) -> None:
        """Build the network.

        Args:
            n_channels: Input channels.
            n_classes: Output classes.
        """
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.dropout = nn.Dropout2d(p=DROPOUT)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        """He initialisation: Gaussian with std ``sqrt(2 / fan_in)`` (paper)."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map a ``(N, 1, 572, 572)`` input to ``(N, n_classes, 388, 388)`` logits."""
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.dropout(self.down4(x4))
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


# ============================================================================
# Intensity <-> class quantisation
# ============================================================================
def hu_to_class(y_norm: torch.Tensor) -> torch.Tensor:
    """Quantise a normalised image into class indices.

    With two classes this thresholds at ``HU_BINARIZE_THRESHOLD``; otherwise
    ``[0, 1]`` is split into ``N_CLASSES`` uniform bins.
    """
    if N_CLASSES == 2:
        return (y_norm >= HU_BINARIZE_THRESHOLD).long()
    return torch.clamp((y_norm * N_CLASSES).long(), 0, N_CLASSES - 1)


def class_to_hu(cls_idx: torch.Tensor) -> torch.Tensor:
    """Map class indices back to normalised intensities (bin midpoints)."""
    if N_CLASSES == 2:
        lo = HU_BINARIZE_THRESHOLD / 2.0
        hi = (1.0 + HU_BINARIZE_THRESHOLD) / 2.0
        return torch.where(
            cls_idx == 1,
            torch.full_like(cls_idx, hi, dtype=torch.float32),
            torch.full_like(cls_idx, lo, dtype=torch.float32),
        )
    return (cls_idx.float() + 0.5) / float(N_CLASSES)


def mirror_pad_to(x: torch.Tensor, target: int) -> torch.Tensor:
    """Reflect-pad a ``(B, C, H, W)`` tensor up to ``target x target``.

    Implements the overlap-tile strategy of paper Fig. 2 for inputs smaller
    than the 572x572 tile.
    """
    h, w = x.size(-2), x.size(-1)
    if h >= target and w >= target:
        return x
    ph = max(0, target - h)
    pw = max(0, target - w)
    pad = [pw // 2, pw - pw // 2, ph // 2, ph - ph // 2]
    return F.pad(x, pad, mode="reflect")


def predict_norm(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Predict a normalised ``(1, 1, OUT, OUT)`` image from a ``(1, 1, H, W)`` input."""
    logits = model(mirror_pad_to(x, PATCH_SIZE))
    return class_to_hu(logits.argmax(dim=1)).unsqueeze(1)


# ============================================================================
# Data
# ============================================================================
def _aug_warp(
    t: torch.Tensor, ang: torch.Tensor, disp_coarse: torch.Tensor
) -> torch.Tensor:
    """Rotate and elastically deform a ``(1, 1, H, W)`` tensor.

    Args:
        t: Image to warp.
        ang: Rotation angle in radians.
        disp_coarse: ``(1, 2, g, g)`` displacement grid in pixels, upsampled
            bicubically to the image size (paper Sec. 3).

    Returns:
        The warped image, with reflection at the borders.
    """
    h, w = t.shape[-2], t.shape[-1]
    ys, xs = torch.meshgrid(
        torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij"
    )
    cos, sin = float(torch.cos(ang)), float(torch.sin(ang))
    xr = cos * xs - sin * ys
    yr = sin * xs + cos * ys
    disp = F.interpolate(disp_coarse, size=(h, w), mode="bicubic", align_corners=True)
    # Pixels -> normalised [-1, 1] grid coordinates.
    dx = disp[0, 0] / ((w - 1) / 2.0)
    dy = disp[0, 1] / ((h - 1) / 2.0)
    grid = torch.stack([xr + dx, yr + dy], dim=-1).unsqueeze(0)
    return F.grid_sample(
        t, grid, mode="bilinear", padding_mode="reflection", align_corners=True
    )


def augment_pair(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply the paper's augmentation identically to an input/target pair.

    Random horizontal/vertical flips, a random rotation in ``[-pi, pi]`` and an
    elastic deformation. Only the torch RNG is used because the DataLoader
    re-seeds it per worker, whereas NumPy's RNG would repeat the same
    augmentation in every worker. The paper's grey-value variation is omitted:
    here the target is the full-dose image, so altering the input intensities
    would corrupt the LDCT-to-FD mapping.

    Args:
        x: Input image ``(H, W)``.
        y: Target image ``(H, W)``.

    Returns:
        The augmented pair.
    """
    xt = torch.from_numpy(np.ascontiguousarray(x))[None, None]
    yt = torch.from_numpy(np.ascontiguousarray(y))[None, None]
    if torch.rand(1).item() < 0.5:
        xt, yt = torch.flip(xt, dims=[3]), torch.flip(yt, dims=[3])
    if torch.rand(1).item() < 0.5:
        xt, yt = torch.flip(xt, dims=[2]), torch.flip(yt, dims=[2])
    ang = (torch.rand(1) * 2.0 - 1.0) * pi
    disp_coarse = torch.randn(1, 2, ELASTIC_GRID, ELASTIC_GRID) * ELASTIC_SIGMA_PX
    xt = _aug_warp(xt, ang, disp_coarse)
    yt = _aug_warp(yt, ang, disp_coarse)
    return xt[0, 0].numpy(), yt[0, 0].numpy()


class UNetDataset(ULDCTDataset):
    """ULDCT dataset with optional whole-image augmentation before patching."""

    def __init__(
        self,
        split: str,
        data_root: Path,
        patch_size: int | None = None,
        patch_n: int | None = None,
        augment: bool = False,
    ) -> None:
        """Discover and pair the files of ``split``.

        Args:
            split: Dataset split.
            data_root: Dataset root.
            patch_size: Training tile side, or ``None`` for full slices.
            patch_n: Tiles per slice.
            augment: Apply :func:`augment_pair` (training only).
        """
        super().__init__(split, data_root, patch_size=patch_size, patch_n=patch_n)
        self.augment = augment

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the (optionally augmented) slice pair or random tiles of it."""
        x, y = self.load_pair(idx)
        if self.augment:
            x, y = augment_pair(x, y)
        if self.patch_size:
            return random_patch_pairs(x, y, self.patch_size, self.patch_n)
        return x, y


# ============================================================================
# Training and testing
# ============================================================================
@torch.no_grad()
def quick_eval(
    infer_fn: Callable[[torch.Tensor], torch.Tensor],
    loader: DataLoader,
    device: torch.device,
    max_slices: int = QUICK_EVAL_MAX_SLICES,
) -> Measure:
    """Mean PSNR/SSIM/RMSE on the first validation slices.

    Metrics are computed on the 388x388 output region, with the target
    centre-cropped to match.

    Args:
        infer_fn: Maps a padded ``(1, 1, 572, 572)`` input to class logits.
        loader: Validation loader with batch size 1.
        device: Device the inputs are moved to.
        max_slices: Number of slices evaluated.

    Returns:
        ``(psnr, ssim, rmse)`` averaged over the evaluated slices.
    """
    psnr_sum, ssim_sum, rmse_sum, n = 0.0, 0.0, 0.0, 0
    for x, y in loader:
        if n >= max_slices:
            break
        x = x.float().to(device).unsqueeze(1)
        y = y.float().to(device).unsqueeze(1)
        logits = infer_fn(mirror_pad_to(x, PATCH_SIZE))
        pred_norm = class_to_hu(logits.argmax(dim=1)).unsqueeze(1)
        y_crop = center_crop(y, logits.size(2), logits.size(3))
        yv, pv = to_hu_window(y_crop), to_hu_window(pred_norm)
        psnr_sum += compute_psnr(pv, yv, DATA_RANGE)
        ssim_sum += compute_ssim(pv, yv, DATA_RANGE)
        rmse_sum += compute_rmse(pv, yv)
        n += 1
    if n == 0:
        return 0.0, 0.0, 0.0
    return psnr_sum / n, ssim_sum / n, rmse_sum / n


def estimate_class_weights(
    targets: Iterable[str],
    n_sample: int = CLASS_WEIGHT_SAMPLES,
    seed: int = CLASS_WEIGHT_SEED,
) -> torch.Tensor:
    """Inverse-frequency class weights ``w_c`` (paper Sec. 3).

    Estimated on a fixed random sample of the training targets and normalised
    to a mean of 1. The paper's border term ``w_0 * exp(-(d1 + d2)^2 / 2s^2)``
    is omitted: it separates touching cells in EM segmentation and has no
    counterpart in two-class CT.

    Args:
        targets: Training target paths.
        n_sample: Number of targets sampled.
        seed: Seed of the sampling RNG.

    Returns:
        A ``(N_CLASSES,)`` float tensor.
    """
    rng = np.random.default_rng(seed)
    paths = list(targets)
    if len(paths) > n_sample:
        sel = rng.choice(len(paths), size=n_sample, replace=False)
        paths = [paths[i] for i in sel]
    counts = np.zeros(N_CLASSES, dtype=np.float64)
    for p in paths:
        cls = hu_to_class(torch.from_numpy(load_target_array(p)))
        counts += np.bincount(cls.reshape(-1).numpy(), minlength=N_CLASSES)[:N_CLASSES]
    freq = counts / max(counts.sum(), 1.0)
    w = 1.0 / np.clip(freq, MIN_CLASS_FREQ, None)
    w = w / w.sum() * N_CLASSES
    return torch.tensor(w, dtype=torch.float32)


def train(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: RunConfig,
    val_loader: DataLoader | None = None,
) -> None:
    """Train with SGD and weighted cross-entropy, resuming if possible.

    Validates once per epoch and keeps the checkpoint with the best PSNR.

    Args:
        model: Network to train.
        loader: Training loader yielding stacks of tiles.
        device: Training device.
        cfg: Run configuration.
        val_loader: Loader for per-epoch validation.
    """
    model.train()
    opt = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    class_w = estimate_class_weights(loader.dataset.targets).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_w)
    logger.info(
        "[loss] weighted CE (w_c) class_weights=%s",
        [round(w, 4) for w in class_w.tolist()],
    )

    ckpt = load_checkpoint(cfg.last_ckpt, model=model, optimizer=opt)
    start_epoch = (ckpt["epoch"] + 1) if ckpt else 1
    step = ckpt["step"] if ckpt else 0
    best_psnr = best_psnr_so_far(cfg.best_ckpt)
    losses = []
    t0 = time.time()

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        epoch_loss_sum, epoch_n = 0.0, 0
        for x, y in loader:
            step += 1
            x = x.float().to(device)
            y = y.float().to(device)
            if x.dim() == 4:
                # (B, n_patches, H, W) -> (B * n_patches, 1, H, W). H and W are
                # the actual tile size, smaller than PATCH_SIZE for 512x512
                # slices; mirror padding below restores the 572x572 tile.
                x = x.reshape(-1, 1, x.size(2), x.size(3))
                y = y.reshape(-1, 1, y.size(2), y.size(3))
            else:
                x, y = x.unsqueeze(1), y.unsqueeze(1)
            x = mirror_pad_to(x, PATCH_SIZE)
            y = mirror_pad_to(y, PATCH_SIZE)
            pred = model(x)
            y_crop = center_crop(y, pred.size(2), pred.size(3))
            loss = criterion(pred, hu_to_class(y_crop.squeeze(1)))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
            epoch_loss_sum += loss.item()
            epoch_n += 1

        # Validate on the validation split; the test split is only used once,
        # by the final test.
        model.eval()
        psnr, ssim, rmse = (
            quick_eval(model, val_loader, device)
            if val_loader is not None
            else (0.0, 0.0, 0.0)
        )
        model.train()
        avg_loss = epoch_loss_sum / max(epoch_n, 1)
        logger.info(
            "epoch %d/%d  loss %.6f  val PSNR %6.2f  SSIM %.4f  RMSE %6.2f HU  (%.0fs)",
            epoch,
            NUM_EPOCHS,
            avg_loss,
            psnr,
            ssim,
            rmse,
            time.time() - t0,
        )

        if epoch % SAVE_EPOCHS == 0 or epoch == NUM_EPOCHS:
            state = dict(
                model=model,
                optimizer=opt,
                epoch=epoch,
                step=step,
                lr=opt.param_groups[0]["lr"],
                loss=avg_loss,
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
                "  saved last.pt (epoch=%d psnr=%.3f ssim=%.4f rmse_hu=%.3f) "
                "best_psnr=%.3f",
                epoch,
                psnr,
                ssim,
                rmse,
                best_psnr,
            )


def test(
    model: nn.Module, loader: DataLoader, device: torch.device, cfg: RunConfig
) -> None:
    """Evaluate on the full test split and write metrics and figures.

    Input, target and prediction are compared on the 388x388 output region
    (input and target centre-cropped).

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
        pred_norm = predict_norm(model, x)
        out = pred_norm.size(2)
        return (
            to_hu_window(center_crop(x, out, out)),
            to_hu_window(center_crop(y, out, out)),
            to_hu_window(pred_norm),
        )

    run_test(loader, predict, cfg)


def main() -> None:
    """Train U-Net on the training split, then test it."""
    parser = build_arg_parser(MODEL_NAME, description=__doc__)
    cfg, _ = parse_run_config(parser, MODEL_NAME)
    torch.backends.cudnn.benchmark = True
    device = get_device()
    logger.info(
        "[%s] dose=%s device=%s out=%s", MODEL_NAME, cfg.dose, device, cfg.output_dir
    )

    train_ds = UNetDataset(
        TRAIN_SPLIT, cfg.data_root, patch_size=PATCH_SIZE, patch_n=PATCH_N, augment=True
    )
    test_ds = UNetDataset(TEST_SPLIT, cfg.data_root)
    val_ds = optional_split(lambda: UNetDataset(VAL_SPLIT, cfg.data_root), VAL_SPLIT)
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

    model = UNet(n_channels=1, n_classes=N_CLASSES).to(device)
    train(model, train_loader, device, cfg, val_loader=val_loader)
    # Unlike the other models, the original U-Net run tests the in-memory
    # weights of the last epoch, not the best checkpoint.
    test(model, test_loader, device, cfg)


if __name__ == "__main__":
    main()
