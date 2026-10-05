"""EDCNN (Liang et al., 2020) trained and evaluated on the ULDCT dataset.

Edge-enhancement-based densely connected network: a trainable Sobel layer
followed by eight dense "conveying path" blocks (1x1 then 3x3 convolutions) and
a global residual. The model, Sobel layer and compound loss are copied from
the official repository (workingcoder/EDCNN: ``edcnn_model.py`` and
``compound_loss.py``). Training follows the paper's experimental setup: AdamW
with lr 1e-3, 200 epochs, 32 images x 4 random 64x64 patches per iteration, and
a compound loss of MSE plus a ResNet-50 perceptual term (weight 0.01, blocks
1-4). Augmentation is random cropping only; testing uses full 512x512 slices.

The ImageNet ResNet-50 weights are downloaded by torchvision into the torch
hub cache (``$TORCH_HOME``, or ``--torch-hub-dir``).

Usage:
    python edcnn.py --dose 5pct --data-root /path/to/uldct_5pct/dataset
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import ResNet50_Weights

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

MODEL_NAME = "edcnn"

# Hyperparameters (paper Sec. III-B, experimental setup).
NUM_EPOCHS = 200
BATCH_SIZE = 32
PATCH_SIZE = 64
PATCH_N = 4  # patches per image -> 128 patches per iteration
LR = 1e-3
PRINT_ITERS = 50
SAVE_EPOCHS = 5  # validation and checkpointing happen every SAVE_EPOCHS epochs

IN_CHANNELS = 1
NUM_FEATURES = 32
SOBEL_CHANNELS = 32

# Compound loss (paper defaults): MSE + 0.01 * ResNet-50 perceptual loss.
PERCEPTUAL_BLOCKS = (1, 2, 3, 4)
MSE_WEIGHT = 1.0
PERCEPTUAL_WEIGHT = 0.01

logger = logging.getLogger(MODEL_NAME)


# ============================================================================
# Model (verbatim from the official repository)
# ============================================================================
class SobelConv2d(nn.Module):
    """Convolution with Sobel-initialised kernels scaled by trainable factors.

    Output channels cycle through four Sobel orientations (vertical,
    horizontal and the two diagonals). The kernels themselves are fixed; only
    a per-channel factor (initialised to 1) and the bias are trained.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        requires_grad: bool = True,
    ) -> None:
        """Build the Sobel kernels.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels (multiple of 4 and of
                ``groups``).
            kernel_size: Odd kernel side.
            stride: Convolution stride.
            padding: Convolution padding.
            dilation: Convolution dilation.
            groups: Convolution groups.
            bias: Whether to add a trainable bias.
            requires_grad: If ``False``, a plain Sobel operator with fixed
                weights and no bias.
        """
        assert kernel_size % 2 == 1, "SobelConv2d's kernel_size must be odd."
        assert (
            out_channels % 4 == 0
        ), "SobelConv2d's out_channels must be a multiple of 4."
        assert (
            out_channels % groups == 0
        ), "SobelConv2d's out_channels must be a multiple of groups."

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.bias = bias if requires_grad else False
        if self.bias:
            self.bias = nn.Parameter(
                torch.zeros(size=(out_channels,), dtype=torch.float32),
                requires_grad=True,
            )
        else:
            self.bias = None

        self.sobel_weight = nn.Parameter(
            torch.zeros(
                size=(out_channels, int(in_channels / groups), kernel_size, kernel_size)
            ),
            requires_grad=False,
        )

        kernel_mid = kernel_size // 2
        for idx in range(out_channels):
            if idx % 4 == 0:
                self.sobel_weight[idx, :, 0, :] = -1
                self.sobel_weight[idx, :, 0, kernel_mid] = -2
                self.sobel_weight[idx, :, -1, :] = 1
                self.sobel_weight[idx, :, -1, kernel_mid] = 2
            elif idx % 4 == 1:
                self.sobel_weight[idx, :, :, 0] = -1
                self.sobel_weight[idx, :, kernel_mid, 0] = -2
                self.sobel_weight[idx, :, :, -1] = 1
                self.sobel_weight[idx, :, kernel_mid, -1] = 2
            elif idx % 4 == 2:
                self.sobel_weight[idx, :, 0, 0] = -2
                for i in range(0, kernel_mid + 1):
                    self.sobel_weight[idx, :, kernel_mid - i, i] = -1
                    self.sobel_weight[idx, :, kernel_size - 1 - i, kernel_mid + i] = 1
                self.sobel_weight[idx, :, -1, -1] = 2
            else:
                self.sobel_weight[idx, :, -1, 0] = -2
                for i in range(0, kernel_mid + 1):
                    self.sobel_weight[idx, :, kernel_mid + i, i] = -1
                    self.sobel_weight[idx, :, i, kernel_mid + i] = 1
                self.sobel_weight[idx, :, 0, -1] = 2

        self.sobel_factor = nn.Parameter(
            torch.ones(size=(out_channels, 1, 1, 1), dtype=torch.float32),
            requires_grad=requires_grad,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convolve ``x`` with the scaled Sobel kernels."""
        if torch.cuda.is_available():
            self.sobel_factor = self.sobel_factor.cuda()
            if isinstance(self.bias, nn.Parameter):
                self.bias = self.bias.cuda()

        sobel_weight = self.sobel_weight * self.sobel_factor

        if torch.cuda.is_available():
            sobel_weight = sobel_weight.cuda()

        return F.conv2d(
            x,
            sobel_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class EDCNN(nn.Module):
    """Edge-enhancement-based densely connected CNN.

    The input and its Sobel edge maps are concatenated and fed to eight
    blocks; each block receives that concatenation plus the previous block's
    output. The last block predicts a residual that is added to the input.
    """

    def __init__(
        self,
        in_ch: int = IN_CHANNELS,
        out_ch: int = NUM_FEATURES,
        sobel_ch: int = SOBEL_CHANNELS,
    ) -> None:
        """Build the layers.

        Args:
            in_ch: Number of image channels.
            out_ch: Number of feature maps per block.
            sobel_ch: Number of Sobel edge maps.
        """
        super().__init__()

        self.conv_sobel = SobelConv2d(
            in_ch, sobel_ch, kernel_size=3, stride=1, padding=1, bias=True
        )

        point = dict(kernel_size=1, stride=1, padding=0)
        full = dict(kernel_size=3, stride=1, padding=1)
        dense_in = in_ch + sobel_ch + out_ch

        self.conv_p1 = nn.Conv2d(in_ch + sobel_ch, out_ch, **point)
        self.conv_f1 = nn.Conv2d(out_ch, out_ch, **full)

        self.conv_p2 = nn.Conv2d(dense_in, out_ch, **point)
        self.conv_f2 = nn.Conv2d(out_ch, out_ch, **full)

        self.conv_p3 = nn.Conv2d(dense_in, out_ch, **point)
        self.conv_f3 = nn.Conv2d(out_ch, out_ch, **full)

        self.conv_p4 = nn.Conv2d(dense_in, out_ch, **point)
        self.conv_f4 = nn.Conv2d(out_ch, out_ch, **full)

        self.conv_p5 = nn.Conv2d(dense_in, out_ch, **point)
        self.conv_f5 = nn.Conv2d(out_ch, out_ch, **full)

        self.conv_p6 = nn.Conv2d(dense_in, out_ch, **point)
        self.conv_f6 = nn.Conv2d(out_ch, out_ch, **full)

        self.conv_p7 = nn.Conv2d(dense_in, out_ch, **point)
        self.conv_f7 = nn.Conv2d(out_ch, out_ch, **full)

        self.conv_p8 = nn.Conv2d(dense_in, out_ch, **point)
        self.conv_f8 = nn.Conv2d(out_ch, in_ch, **full)

        self.relu = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Denoise a batch of ``(N, 1, H, W)`` images."""
        out_0 = self.conv_sobel(x)
        out_0 = torch.cat((x, out_0), dim=-3)

        blocks = [
            (self.conv_p1, self.conv_f1),
            (self.conv_p2, self.conv_f2),
            (self.conv_p3, self.conv_f3),
            (self.conv_p4, self.conv_f4),
            (self.conv_p5, self.conv_f5),
            (self.conv_p6, self.conv_f6),
            (self.conv_p7, self.conv_f7),
        ]
        out = out_0
        for conv_p, conv_f in blocks:
            block_out = self.relu(conv_p(out))
            block_out = self.relu(conv_f(block_out))
            out = torch.cat((out_0, block_out), dim=-3)

        out_8 = self.relu(self.conv_p8(out))
        out_8 = self.conv_f8(out_8)
        return self.relu(x + out_8)


class ResNet50FeatureExtractor(nn.Module):
    """ResNet-50 trunk returning the outputs of the selected residual stages."""

    def __init__(
        self,
        blocks: Sequence[int] = PERCEPTUAL_BLOCKS,
        pretrained: bool = False,
        progress: bool = True,
    ) -> None:
        """Build the trunk.

        Args:
            blocks: Residual stages (1-4) whose outputs are returned.
            pretrained: Load the ImageNet (V1) weights.
            progress: Show a download progress bar.
        """
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        self.model = models.resnet50(weights=weights, progress=progress)
        del self.model.avgpool
        del self.model.fc
        self.blocks = blocks

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return the feature maps of the selected stages, in stage order."""
        feats = []

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        for stage, layer in enumerate(
            (
                self.model.layer1,
                self.model.layer2,
                self.model.layer3,
                self.model.layer4,
            ),
            start=1,
        ):
            x = layer(x)
            if stage in self.blocks:
                feats.append(x)
        return feats


class CompoundLoss(_Loss):
    """MSE plus a ResNet-50 perceptual loss (frozen ImageNet features)."""

    def __init__(
        self,
        blocks: Sequence[int] = PERCEPTUAL_BLOCKS,
        mse_weight: float = MSE_WEIGHT,
        resnet_weight: float = PERCEPTUAL_WEIGHT,
    ) -> None:
        """Load the pretrained feature extractor.

        Args:
            blocks: Stages averaged in the perceptual term. As in the official
                code, the extractor itself always returns stages 1-4.
            mse_weight: Weight of the pixel-wise MSE.
            resnet_weight: Weight of the perceptual term.
        """
        super().__init__()

        self.mse_weight = mse_weight
        self.resnet_weight = resnet_weight

        self.blocks = blocks
        self.model = ResNet50FeatureExtractor(pretrained=True)

        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()

        self.criterion = nn.MSELoss()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the weighted loss of single-channel ``input`` vs ``target``."""
        input_feats = self.model(torch.cat([input, input, input], dim=1))
        target_feats = self.model(torch.cat([target, target, target], dim=1))

        feats_num = len(self.blocks)
        loss_value = 0
        for idx in range(feats_num):
            loss_value += self.criterion(input_feats[idx], target_feats[idx])
        loss_value /= feats_num

        return (
            self.mse_weight * self.criterion(input, target)
            + self.resnet_weight * loss_value
        )


# ============================================================================
# Training and testing
# ============================================================================
def train_batch_metrics(
    pred_norm: torch.Tensor, target_norm: torch.Tensor
) -> tuple[float, float, float]:
    """PSNR, SSIM (clipped to [-1, 1]) and RMSE of a training batch, in HU."""
    pred_hu = truncate(denormalize(pred_norm.detach().cpu()))
    target_hu = truncate(denormalize(target_norm.detach().cpu()))
    psnr = compute_psnr(pred_hu, target_hu, DATA_RANGE)
    ssim = max(min(compute_ssim(pred_hu, target_hu, DATA_RANGE), 1.0), -1.0)
    rmse = compute_rmse(pred_hu, target_hu)
    return psnr, ssim, rmse


def train(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: RunConfig,
    val_loader: DataLoader | None = None,
) -> None:
    """Train with AdamW and the compound loss, resuming from the last checkpoint.

    Every ``SAVE_EPOCHS`` epochs the model is validated, the last checkpoint is
    written and the best-PSNR checkpoint is updated.

    Args:
        model: Network to train.
        loader: Training loader yielding stacks of patches.
        device: Training device.
        cfg: Run configuration.
        val_loader: Loader for periodic validation.
    """
    model.train()
    opt = optim.AdamW(model.parameters(), lr=LR)
    criterion = CompoundLoss(
        blocks=PERCEPTUAL_BLOCKS, mse_weight=MSE_WEIGHT, resnet_weight=PERCEPTUAL_WEIGHT
    )

    ckpt = load_checkpoint(cfg.last_ckpt, model=model, optimizer=opt)
    start_epoch = (ckpt["epoch"] + 1) if ckpt else 1
    step = ckpt["step"] if ckpt else 0
    best_psnr = best_psnr_so_far(cfg.best_ckpt)
    losses = []
    t0 = time.time()

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
            pred = model(x)
            loss = criterion(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

            if step % PRINT_ITERS == 0:
                with torch.no_grad():
                    psnr, ssim, rmse = train_batch_metrics(pred.clamp(0.0, 1.0), y)
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

        if epoch % SAVE_EPOCHS == 0 or epoch == NUM_EPOCHS:
            model.eval()
            psnr, ssim, rmse = (
                quick_eval(model, val_loader, device)
                if val_loader is not None
                else (0.0, 0.0, 0.0)
            )
            model.train()
            # The stored loss is that of the last iteration, not an epoch mean.
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
    """Train EDCNN on the training split, then test the best checkpoint."""
    parser = build_arg_parser(MODEL_NAME, description=__doc__)
    parser.add_argument(
        "--torch-hub-dir",
        type=Path,
        default=None,
        help="Cache folder for the pretrained ResNet-50 (default: $TORCH_HOME/hub).",
    )
    cfg, args = parse_run_config(parser, MODEL_NAME)
    if args.torch_hub_dir is not None:
        torch.hub.set_dir(str(args.torch_hub_dir))
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

    model = EDCNN(in_ch=IN_CHANNELS, out_ch=NUM_FEATURES, sobel_ch=SOBEL_CHANNELS)
    model = model.to(device)
    train(model, train_loader, device, cfg, val_loader=val_loader)
    load_best_for_test(cfg, model)
    test(model, test_loader, device, cfg)


if __name__ == "__main__":
    main()
