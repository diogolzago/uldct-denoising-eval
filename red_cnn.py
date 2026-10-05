"""RED-CNN (Chen et al., 2017) trained and evaluated on the ULDCT dataset.

Residual encoder-decoder with five convolutional and five deconvolutional
layers and three residual shortcuts, trained with MSE and Adam. Where the paper
and the reference repository disagree, the paper is followed.

Usage:
    python red_cnn.py --dose 5pct --data-root /path/to/uldct_5pct/dataset
"""

from __future__ import annotations

import logging
import time

import numpy as np
import torch
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
from common.data import ULDCTDataset, random_patch_pairs, to_hu_window
from common.evaluation import quick_eval, run_test
from common.runtime import load_best_for_test, optional_split

try:
    from scipy.ndimage import rotate as nd_rotate
    from scipy.ndimage import zoom as nd_zoom

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

MODEL_NAME = "red_cnn"

# Hyperparameters (paper Sec. III.B "Parameter selection" unless noted).
NUM_EPOCHS = 100  # repo default; the paper specifies ~1e6 patches, not epochs
BATCH_SIZE = 16  # repo main.py
PATCH_SIZE = 55  # paper (the repo used 64)
PATCH_N = 10  # patches per image -> 160 patches per iteration
LR = 1e-4  # paper base learning rate (the repo used a fixed 1e-5)
LR_MIN = 1e-5  # paper: lr "slowly decreased down to 1e-5"
LR_DECAY_EVERY = 3000  # repo: halve the lr every 3000 iterations
LR_DECAY_FACTOR = 0.5
SAVE_EPOCHS = 1
AUGMENT = True  # paper: rotation by 45 degrees, vertical/horizontal flips, scaling

NUM_FEATURES = 96
KERNEL_SIZE = 5
INIT_STD = 0.01  # paper: Gaussian weight initialisation N(0, 0.01)

AUG_ANGLES = (0, 45, 90, 135, 180, 225, 270, 315)
AUG_SCALES = (0.5, 1.0, 2.0)

logger = logging.getLogger(MODEL_NAME)


# ============================================================================
# Model
# ============================================================================
def _init_gaussian(module: nn.Module) -> None:
    """Initialise conv weights with N(0, INIT_STD) and biases with zeros."""
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, mean=0.0, std=INIT_STD)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class RED_CNN(nn.Module):
    """Residual encoder-decoder CNN.

    Five 5x5 unpadded convolutions followed by five 5x5 transposed
    convolutions, with residual shortcuts from the input, the 2nd and the 4th
    encoder layers, and a final ReLU.
    """

    def __init__(self, out_ch: int = NUM_FEATURES) -> None:
        """Build the layers.

        Args:
            out_ch: Number of feature maps of the hidden layers.
        """
        super().__init__()
        conv = dict(kernel_size=KERNEL_SIZE, stride=1, padding=0)
        self.conv1 = nn.Conv2d(1, out_ch, **conv)
        self.conv2 = nn.Conv2d(out_ch, out_ch, **conv)
        self.conv3 = nn.Conv2d(out_ch, out_ch, **conv)
        self.conv4 = nn.Conv2d(out_ch, out_ch, **conv)
        self.conv5 = nn.Conv2d(out_ch, out_ch, **conv)

        self.tconv1 = nn.ConvTranspose2d(out_ch, out_ch, **conv)
        self.tconv2 = nn.ConvTranspose2d(out_ch, out_ch, **conv)
        self.tconv3 = nn.ConvTranspose2d(out_ch, out_ch, **conv)
        self.tconv4 = nn.ConvTranspose2d(out_ch, out_ch, **conv)
        self.tconv5 = nn.ConvTranspose2d(out_ch, 1, **conv)

        self.relu = nn.ReLU()
        # Neither the repo nor the PyTorch default use the paper's Gaussian init.
        self.apply(_init_gaussian)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Denoise a batch of ``(N, 1, H, W)`` images."""
        residual_1 = x
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        residual_2 = out
        out = self.relu(self.conv3(out))
        out = self.relu(self.conv4(out))
        residual_3 = out
        out = self.relu(self.conv5(out))

        out = self.tconv1(out)
        out += residual_3
        out = self.tconv2(self.relu(out))
        out = self.tconv3(self.relu(out))
        out += residual_2
        out = self.tconv4(self.relu(out))
        out = self.tconv5(self.relu(out))
        out += residual_1
        return self.relu(out)


# ============================================================================
# Data
# ============================================================================
def augment_pair(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply the paper's augmentation identically to an input/target pair.

    Random horizontal and vertical flips, rotation by a multiple of 45 degrees
    and scaling by 0.5 or 2, applied to the whole image before patching.
    Without SciPy, rotations fall back to multiples of 90 degrees and scaling
    is skipped.

    Args:
        x: Input image ``(H, W)``.
        y: Target image ``(H, W)``.

    Returns:
        The augmented, contiguous ``float32`` pair.
    """
    if np.random.rand() < 0.5:
        x, y = np.fliplr(x), np.fliplr(y)
    if np.random.rand() < 0.5:
        x, y = np.flipud(x), np.flipud(y)
    angle = float(np.random.choice(AUG_ANGLES))
    if angle:
        if HAS_SCIPY:
            x = nd_rotate(x, angle, reshape=False, order=1, mode="reflect")
            y = nd_rotate(y, angle, reshape=False, order=1, mode="reflect")
        else:
            k = int(round(angle / 90)) % 4
            x, y = np.rot90(x, k), np.rot90(y, k)
    scale = float(np.random.choice(AUG_SCALES))
    if scale != 1.0 and HAS_SCIPY:
        x, y = nd_zoom(x, scale, order=1), nd_zoom(y, scale, order=1)
    return (
        np.ascontiguousarray(x, dtype=np.float32),
        np.ascontiguousarray(y, dtype=np.float32),
    )


class REDCNNDataset(ULDCTDataset):
    """ULDCT dataset with the paper's augmentation on training patches."""

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the slice pair, or augmented random patches when patching."""
        x, y = self.load_pair(idx)
        if self.patch_size:
            if AUGMENT:
                x, y = augment_pair(x, y)
            return random_patch_pairs(x, y, self.patch_size, self.patch_n)
        return x, y


# ============================================================================
# Training and testing
# ============================================================================
def train(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: RunConfig,
    val_loader: DataLoader | None = None,
) -> None:
    """Train with MSE/Adam, resuming from the last checkpoint if present.

    Validates once per epoch and keeps the checkpoint with the best PSNR.

    Args:
        model: Network to train.
        loader: Training loader yielding stacks of patches.
        device: Training device.
        cfg: Run configuration.
        val_loader: Loader for per-epoch validation.
    """
    model.train()
    opt = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

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
                x = x.view(-1, 1, PATCH_SIZE, PATCH_SIZE)
                y = y.view(-1, 1, PATCH_SIZE, PATCH_SIZE)
            else:
                x, y = x.unsqueeze(1), y.unsqueeze(1)
            loss = criterion(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
            epoch_loss_sum += loss.item()
            epoch_n += 1

            if LR_DECAY_EVERY and step % LR_DECAY_EVERY == 0:
                for group in opt.param_groups:
                    group["lr"] = max(group["lr"] * LR_DECAY_FACTOR, LR_MIN)

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
            "epoch %d/%d  iter %d  loss %.6f  val PSNR %6.2f  SSIM %.4f  "
            "RMSE %6.2f HU  (%.0fs)",
            epoch,
            NUM_EPOCHS,
            step,
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
                "  saved last.pt (epoch=%d step=%d psnr=%.3f ssim=%.4f "
                "rmse_hu=%.3f) best_psnr=%.3f",
                epoch,
                step,
                psnr,
                ssim,
                rmse,
                best_psnr,
            )


def test(
    model: nn.Module, loader: DataLoader, device: torch.device, cfg: RunConfig
) -> None:
    """Evaluate on the full test split and write metrics and figures.

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
        pred = model(x).clamp(0.0, 1.0)
        return to_hu_window(x), to_hu_window(y), to_hu_window(pred)

    run_test(loader, predict, cfg)


def main() -> None:
    """Train RED-CNN on the training split, then test the best checkpoint."""
    parser = build_arg_parser(MODEL_NAME, description=__doc__)
    cfg, _ = parse_run_config(parser, MODEL_NAME)
    torch.backends.cudnn.benchmark = True
    device = get_device()
    logger.info(
        "[%s] dose=%s device=%s out=%s", MODEL_NAME, cfg.dose, device, cfg.output_dir
    )

    train_ds = REDCNNDataset(
        TRAIN_SPLIT, cfg.data_root, patch_size=PATCH_SIZE, patch_n=PATCH_N
    )
    test_ds = REDCNNDataset(TEST_SPLIT, cfg.data_root)
    val_ds = optional_split(lambda: REDCNNDataset(VAL_SPLIT, cfg.data_root), VAL_SPLIT)
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

    model = RED_CNN().to(device)
    train(model, train_loader, device, cfg, val_loader=val_loader)
    load_best_for_test(cfg, model)
    test(model, test_loader, device, cfg)


if __name__ == "__main__":
    main()
