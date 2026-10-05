"""WGAN-VGG (Yang et al., 2018) trained and evaluated on the ULDCT dataset.

Wasserstein GAN with gradient penalty whose generator is additionally trained
with a VGG19 perceptual loss. Where the paper and the reference repository
disagree, the paper is followed unless noted next to the hyperparameter.

The ImageNet-pretrained VGG19 weights are downloaded by torchvision on the
first run (into ``~/.cache/torch/hub``); pre-populate that cache on machines
without internet access.

Usage:
    python wgan_vgg.py --dose 5pct --data-root /path/to/uldct_5pct/dataset
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.models import VGG19_Weights, vgg19

from common.checkpoint import best_psnr_so_far, resolve_checkpoint_path
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
from common.data import ULDCTDataset, to_hu_window
from common.evaluation import QUICK_EVAL_MAX_SLICES, run_test
from common.metrics import compute_psnr, compute_rmse, compute_ssim
from common.runtime import load_best_for_test, optional_split

MODEL_NAME = "wgan_vgg"

# Hyperparameters (paper Fig. 3 "Require" and Sec. III.B unless noted).
NUM_EPOCHS = 100  # paper N_epoch (the repo used 200)
BATCH_SIZE = 16  # repo; effective batch = BATCH_SIZE * PATCH_N patches
PATCH_SIZE = 80  # paper and repo
PATCH_N = 10  # repo (160 patches per iteration; the paper cites m=128)
N_D_TRAIN = 4  # paper N_D: critic updates per generator update
LR = 1e-6  # reference-folder value; the paper's 1e-5 diverged on this data
ADAM_BETAS = (0.5, 0.9)  # paper; helps WGAN-GP stability
LAMBDA_GP = 10.0  # paper: gradient-penalty weight
LAMBDA_VGG = 0.1  # paper Eq. (7): perceptual-loss weight lambda_1
LR_DECAY_ITERS = 3000  # repo: the lr is set to 0.5 * LR every 3000 iterations
LR_DECAY_FACTOR = 0.5

# Paper Sec. III.A excludes patches that are mostly air; as in the repo, a patch
# is rejected when its mean (normalised to [0, 1]) is below this threshold.
DROP_BACKGROUND = 0.1
# Rejection attempts per requested patch before any patch is accepted, so an
# all-air image cannot loop forever.
MAX_TRIES_PER_PATCH = 50

GEN_FEATURES = 32
GEN_NUM_CONVS = 8
DISC_BLOCKS = (
    (1, 64, 1),
    (64, 64, 2),
    (64, 128, 1),
    (128, 128, 2),
    (128, 256, 1),
    (256, 256, 2),
)  # (in_channels, out_channels, stride) of each critic conv
DISC_FC_FEATURES = 1024
VGG_NUM_LAYERS = 35  # VGG19 features[:35], i.e. up to conv5_4 + ReLU

logger = logging.getLogger(MODEL_NAME)

Measure4 = tuple[float, float, float, float]


# ============================================================================
# Model
# ============================================================================
class WGAN_VGG_generator(nn.Module):
    """Generator: eight 3x3 convolutions with ReLU between them."""

    def __init__(self) -> None:
        """Build the layers."""
        super().__init__()
        layers = [nn.Conv2d(1, GEN_FEATURES, 3, 1, 1), nn.ReLU()]
        for _ in range(GEN_NUM_CONVS - 2):
            layers.append(nn.Conv2d(GEN_FEATURES, GEN_FEATURES, 3, 1, 1))
            layers.append(nn.ReLU())
        # No ReLU on the output, unlike the paper and the repo: the final ReLU
        # died (output stuck at 0), producing constant images and frozen
        # validation metrics.
        layers.append(nn.Conv2d(GEN_FEATURES, 1, 3, 1, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Denoise a batch of ``(N, 1, H, W)`` images."""
        return self.net(x)


class WGAN_VGG_discriminator(nn.Module):
    """Critic: six unpadded 3x3 convolutions followed by two linear layers."""

    def __init__(self, input_size: int) -> None:
        """Build the layers.

        Args:
            input_size: Side of the square input patches.
        """
        super().__init__()
        layers: list[nn.Module] = []
        for ch_in, ch_out, stride in DISC_BLOCKS:
            layers.append(nn.Conv2d(ch_in, ch_out, 3, stride, 0))
            layers.append(nn.LeakyReLU())
        strides = [stride for _, _, stride in DISC_BLOCKS]
        self.output_size = _conv_output_size(input_size, [3] * len(strides), strides)
        self.net = nn.Sequential(*layers)
        self.fc1 = nn.Linear(
            256 * self.output_size * self.output_size, DISC_FC_FEATURES
        )
        self.fc2 = nn.Linear(DISC_FC_FEATURES, 1)
        self.lrelu = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return one critic score per image, shape ``(N, 1)``."""
        out = self.net(x)
        out = out.view(-1, 256 * self.output_size * self.output_size)
        out = self.lrelu(self.fc1(out))
        return self.fc2(out)


def _conv_output_size(
    input_size: int, kernel_sizes: list[int], strides: list[int]
) -> int:
    """Spatial output size of a stack of unpadded convolutions."""
    n = input_size
    for k, s in zip(kernel_sizes, strides):
        n = (n - k) // s + 1
    return n


class WGAN_VGG_FeatureExtractor(nn.Module):
    """Frozen, ImageNet-pretrained VGG19 feature extractor."""

    def __init__(self) -> None:
        """Load VGG19 (downloaded on first use) and freeze it."""
        super().__init__()
        vgg19_model = vgg19(weights=VGG19_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(
            *list(vgg19_model.features.children())[:VGG_NUM_LAYERS]
        )
        # Paper Sec. II.D keeps the VGG parameters intact. Freezing avoids
        # storing their gradients; the perceptual gradient still flows through
        # the network to the generator.
        for p in self.feature_extractor.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the VGG feature maps of a 3-channel batch."""
        return self.feature_extractor(x)


class WGAN_VGG(nn.Module):
    """Container for the generator, the critic and the VGG feature extractor."""

    def __init__(self, input_size: int = 64) -> None:
        """Build the three networks.

        Args:
            input_size: Side of the training patches seen by the critic.
        """
        super().__init__()
        self.generator = WGAN_VGG_generator()
        self.discriminator = WGAN_VGG_discriminator(input_size)
        self.feature_extractor = WGAN_VGG_FeatureExtractor()
        # Perceptual loss as in the reference folder: L1 between VGG features
        # (lambda_1 = 0.1 was tuned for it). The paper's squared Frobenius norm
        # (MSE) unbalanced the losses.
        self.p_criterion = nn.L1Loss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Denoise with the generator."""
        return self.generator(x)

    def d_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        gp: bool = True,
        return_gp: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        """Critic loss ``-E[D(y)] + E[D(G(x))]`` plus optional gradient penalty.

        Args:
            x: LDCT input batch.
            y: Full-dose target batch.
            gp: Add the gradient penalty.
            return_gp: Also return the gradient-penalty term.

        Returns:
            The loss, or ``(loss, gp_loss)`` when ``return_gp`` is set.
        """
        fake = self.generator(x)
        d_real = self.discriminator(y)
        d_fake = self.discriminator(fake)
        d_loss = -torch.mean(d_real) + torch.mean(d_fake)
        if gp:
            gp_loss = self.gp(y, fake)
            loss = d_loss + gp_loss
        else:
            gp_loss = None
            loss = d_loss
        return (loss, gp_loss) if return_gp else loss

    def g_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        perceptual: bool = True,
        return_p: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        """Generator loss ``-E[D(G(x))]`` plus optional weighted perceptual loss.

        Args:
            x: LDCT input batch.
            y: Full-dose target batch.
            perceptual: Add ``LAMBDA_VGG`` times the perceptual loss.
            return_p: Also return the (unweighted) perceptual term.

        Returns:
            The loss, or ``(loss, p_loss)`` when ``return_p`` is set.
        """
        fake = self.generator(x)
        d_fake = self.discriminator(fake)
        g_loss = -torch.mean(d_fake)
        if perceptual:
            p_loss = self.p_loss(x, y)
            loss = g_loss + (LAMBDA_VGG * p_loss)
        else:
            p_loss = None
            loss = g_loss
        return (loss, p_loss) if return_p else loss

    def p_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """L1 distance between VGG features of ``G(x)`` and ``y``."""
        fake = self.generator(x).repeat(1, 3, 1, 1)
        real = y.repeat(1, 3, 1, 1)
        return self.p_criterion(
            self.feature_extractor(fake), self.feature_extractor(real)
        )

    def gp(
        self, y: torch.Tensor, fake: torch.Tensor, lambda_: float = LAMBDA_GP
    ) -> torch.Tensor:
        """WGAN-GP gradient penalty on random real/fake interpolations.

        Args:
            y: Real batch.
            fake: Generated batch, same shape as ``y``.
            lambda_: Penalty weight.

        Returns:
            ``lambda_ * E[(||grad D(interp)||_2 - 1)^2]``.
        """
        if y.size() != fake.size():
            raise ValueError(f"shape mismatch: {y.size()} vs {fake.size()}")
        device = y.device
        a = torch.rand((y.size(0), 1, 1, 1), device=device)
        interp = (a * y + ((1 - a) * fake)).requires_grad_(True)
        d_interp = self.discriminator(interp)
        grad_outputs = torch.ones((y.shape[0], 1), device=device, requires_grad=False)
        gradients = torch.autograd.grad(
            outputs=d_interp,
            inputs=interp,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        return ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_


# ============================================================================
# Data
# ============================================================================
def random_patch_pairs_no_air(
    x: np.ndarray, y: np.ndarray, patch_size: int, patch_n: int | None
) -> tuple[np.ndarray, np.ndarray]:
    """Crop aligned random patches, rejecting patches that are mostly air.

    A patch is rejected when the mean of its input or target is below
    ``DROP_BACKGROUND``. After ``MAX_TRIES_PER_PATCH * patch_n`` attempts any
    patch is accepted.

    Args:
        x: Input image ``(H, W)``.
        y: Target image ``(H, W)``.
        patch_size: Patch side; clipped to the image size.
        patch_n: Number of patches (``None`` or 0 means one).

    Returns:
        Stacked input and target patches, ``(patch_n, ps, ps)`` each.
    """
    h = min(x.shape[0], y.shape[0])
    w = min(x.shape[1], y.shape[1])
    ps = min(int(patch_size), h, w)
    n = int(patch_n or 1)
    xs, ys = [], []
    tries = 0
    max_tries = n * MAX_TRIES_PER_PATCH
    while len(xs) < n:
        top = np.random.randint(0, h - ps + 1) if h > ps else 0
        left = np.random.randint(0, w - ps + 1) if w > ps else 0
        xp = x[top : top + ps, left : left + ps]
        yp = y[top : top + ps, left : left + ps]
        tries += 1
        if tries < max_tries and (
            xp.mean() < DROP_BACKGROUND or yp.mean() < DROP_BACKGROUND
        ):
            continue
        xs.append(xp)
        ys.append(yp)
    return np.stack(xs), np.stack(ys)


class WGANVGGDataset(ULDCTDataset):
    """ULDCT dataset whose training patches exclude mostly-air regions."""

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the slice pair, or non-air random patches when patching."""
        x, y = self.load_pair(idx)
        if self.patch_size:
            return random_patch_pairs_no_air(x, y, self.patch_size, self.patch_n)
        return x, y


# ============================================================================
# Checkpoints (generator + critic share one state_dict; two optimizers)
# ============================================================================
def save_gan_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    opt_g: optim.Optimizer | None = None,
    opt_d: optim.Optimizer | None = None,
    epoch: int = 0,
    step: int = 0,
    lr: float = 0.0,
    loss: float = 0.0,
    psnr: float = 0.0,
    ssim: float = 0.0,
    rmse_hu: float = 0.0,
) -> None:
    """Save the model, both optimizers, training progress and metrics.

    Args:
        path: Destination file.
        model: Full :class:`WGAN_VGG` model.
        opt_g: Generator optimizer.
        opt_d: Critic optimizer.
        epoch: Last completed epoch.
        step: Global iteration counter.
        lr: Current generator learning rate.
        loss: Mean generator loss of the epoch.
        psnr: Validation PSNR (dB).
        ssim: Validation SSIM.
        rmse_hu: Validation RMSE (HU).
    """
    torch.save(
        {
            "model": model.state_dict(),
            "opt_g": opt_g.state_dict() if opt_g is not None else None,
            "opt_d": opt_d.state_dict() if opt_d is not None else None,
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


def load_gan_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    opt_g: optim.Optimizer | None = None,
    opt_d: optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any] | None:
    """Restore a checkpoint written by :func:`save_gan_checkpoint`.

    Args:
        path: Requested checkpoint; resolved with ``resolve_checkpoint_path``.
        model: Model to load the weights into.
        opt_g: Optional generator optimizer to restore.
        opt_d: Optional critic optimizer to restore.
        map_location: Device the tensors are loaded onto.

    Returns:
        The checkpoint dictionary, or ``None`` when no checkpoint was found.
    """
    path = resolve_checkpoint_path(path, prefer="last")
    if not path.exists():
        return None
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"])
    if opt_g is not None and ckpt.get("opt_g") is not None:
        opt_g.load_state_dict(ckpt["opt_g"])
    if opt_d is not None and ckpt.get("opt_d") is not None:
        opt_d.load_state_dict(ckpt["opt_d"])
    return ckpt


# ============================================================================
# Training and testing
# ============================================================================
@torch.no_grad()
def quick_eval_gan(
    infer_fn: Callable[[torch.Tensor], torch.Tensor],
    loader: DataLoader,
    device: torch.device,
    max_slices: int = QUICK_EVAL_MAX_SLICES,
    diag: bool = False,
) -> Measure4:
    """Validation PSNR/SSIM/RMSE plus the reconstruction MSE.

    Args:
        infer_fn: Maps a normalised ``(1, 1, H, W)`` input to the prediction.
        loader: Validation loader with batch size 1.
        device: Device the inputs are moved to.
        max_slices: Number of slices evaluated.
        diag: Also log the LDCT-vs-target PSNR and the range of the raw
            (unclamped) generator output, to detect a collapsed generator.

    Returns:
        ``(psnr, ssim, rmse, mse)`` averaged over the evaluated slices, where
        ``mse`` is computed in the normalised ``[0, 1]`` space; zeros if the
        loader is empty.
    """
    psnr_sum, ssim_sum, rmse_sum, mse_sum, n = 0.0, 0.0, 0.0, 0.0, 0
    input_psnr_sum = 0.0
    raw_min, raw_max, raw_mean_sum = float("inf"), -float("inf"), 0.0
    for x, y in loader:
        if n >= max_slices:
            break
        x = x.float().to(device).unsqueeze(1)
        y = y.float().to(device).unsqueeze(1)
        raw = infer_fn(x)
        pred = raw.clamp(0.0, 1.0)
        mse_sum += float(((pred - y) ** 2).mean())
        yv, pv = to_hu_window(y), to_hu_window(pred)
        psnr_sum += compute_psnr(pv, yv, DATA_RANGE)
        ssim_sum += compute_ssim(pv, yv, DATA_RANGE)
        rmse_sum += compute_rmse(pv, yv)
        if diag:
            input_psnr_sum += compute_psnr(to_hu_window(x), yv, DATA_RANGE)
            raw_min = min(raw_min, float(raw.min()))
            raw_max = max(raw_max, float(raw.max()))
            raw_mean_sum += float(raw.mean())
        n += 1
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    if diag:
        logger.info(
            "    [diag] LDCT vs target PSNR %5.2f dB | raw G output: min %.3f "
            "max %.3f mean %.3f (target ~0.2-0.3; all <0 -> constant after clamp)",
            input_psnr_sum / n,
            raw_min,
            raw_max,
            raw_mean_sum / n,
        )
    return psnr_sum / n, ssim_sum / n, rmse_sum / n, mse_sum / n


def train(
    model: WGAN_VGG,
    loader: DataLoader,
    device: torch.device,
    cfg: RunConfig,
    val_loader: DataLoader | None = None,
) -> None:
    """Train WGAN-VGG, resuming from the last checkpoint if present.

    Each iteration performs ``N_D_TRAIN`` critic updates followed by one
    generator update (paper Fig. 3). Validates once per epoch, saves the last
    checkpoint every epoch and keeps the one with the best PSNR.

    Args:
        model: Network to train.
        loader: Training loader yielding stacks of patches.
        device: Training device.
        cfg: Run configuration.
        val_loader: Loader for per-epoch validation.
    """
    model.train()
    opt_g = optim.Adam(model.generator.parameters(), lr=LR, betas=ADAM_BETAS)
    opt_d = optim.Adam(model.discriminator.parameters(), lr=LR, betas=ADAM_BETAS)

    ckpt = load_gan_checkpoint(cfg.last_ckpt, model=model, opt_g=opt_g, opt_d=opt_d)
    start_epoch = (ckpt["epoch"] + 1) if ckpt else 1
    step = ckpt["step"] if ckpt else 0
    best_psnr = best_psnr_so_far(cfg.best_ckpt)
    losses = []
    t0 = time.time()

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        epoch_g_sum, epoch_d_sum, epoch_n = 0.0, 0.0, 0
        for x, y in loader:
            step += 1
            x = x.float().to(device)
            y = y.float().to(device)
            if x.dim() == 4:
                x = x.view(-1, 1, PATCH_SIZE, PATCH_SIZE)
                y = y.view(-1, 1, PATCH_SIZE, PATCH_SIZE)
            else:
                x, y = x.unsqueeze(1), y.unsqueeze(1)

            for _ in range(N_D_TRAIN):
                opt_d.zero_grad()
                d_loss, gp_loss = model.d_loss(x, y, gp=True, return_gp=True)
                d_loss.backward()
                opt_d.step()

            opt_g.zero_grad()
            g_loss, p_loss = model.g_loss(x, y, perceptual=True, return_p=True)
            g_loss.backward()
            opt_g.step()

            # The four curves of paper Fig. 4, logged as in the repo:
            # adversarial G loss, VGG loss, Wasserstein distance, gradient penalty.
            losses.append(
                [
                    g_loss.item() - LAMBDA_VGG * p_loss.item(),
                    p_loss.item(),
                    d_loss.item() - gp_loss.item(),
                    gp_loss.item(),
                ]
            )
            epoch_g_sum += float(g_loss.item())
            epoch_d_sum += float(d_loss.item())
            epoch_n += 1

            # As in the repo, the lr is reset to LR_DECAY_FACTOR * LR (it does
            # not compound), so in practice it decays once and then stays.
            if LR_DECAY_ITERS and step % LR_DECAY_ITERS == 0:
                decayed = LR * LR_DECAY_FACTOR
                for group in opt_g.param_groups:
                    group["lr"] = decayed
                for group in opt_d.param_groups:
                    group["lr"] = decayed

        # Validate on the validation split; the test split is only used once,
        # by the final test.
        model.eval()
        psnr, ssim, rmse, rec_mse = (
            quick_eval_gan(model.generator, val_loader, device, diag=True)
            if val_loader is not None
            else (0.0, 0.0, 0.0, 0.0)
        )
        model.train()
        avg_g = epoch_g_sum / max(epoch_n, 1)
        avg_d = epoch_d_sum / max(epoch_n, 1)
        # rec_mse (lower is better) tracks reconstruction quality; the WGAN
        # d_loss/g_loss oscillate and can be negative.
        logger.info(
            "epoch %d/%d  iter %d  MSE %.5f  |  d_loss %.4f  g_loss %.4f  "
            "val PSNR %6.2f  SSIM %.4f  RMSE %6.2f HU  (%.0fs)",
            epoch,
            NUM_EPOCHS,
            step,
            rec_mse,
            avg_d,
            avg_g,
            psnr,
            ssim,
            rmse,
            time.time() - t0,
        )

        state = dict(
            model=model,
            opt_g=opt_g,
            opt_d=opt_d,
            epoch=epoch,
            step=step,
            lr=opt_g.param_groups[0]["lr"],
            loss=avg_g,
            psnr=psnr,
            ssim=ssim,
            rmse_hu=rmse,
        )
        save_gan_checkpoint(cfg.last_ckpt, **state)
        if psnr > best_psnr:
            best_psnr = psnr
            save_gan_checkpoint(cfg.best_ckpt, **state)
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
    model: WGAN_VGG, loader: DataLoader, device: torch.device, cfg: RunConfig
) -> None:
    """Evaluate the generator on the full test split.

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
        pred = model.generator(x).clamp(0.0, 1.0)
        return to_hu_window(x), to_hu_window(y), to_hu_window(pred)

    run_test(loader, predict, cfg)


def main() -> None:
    """Train WGAN-VGG on the training split, then test the best checkpoint."""
    parser = build_arg_parser(MODEL_NAME, description=__doc__)
    cfg, _ = parse_run_config(parser, MODEL_NAME)
    torch.backends.cudnn.benchmark = True
    device = get_device()
    logger.info(
        "[%s] dose=%s device=%s out=%s", MODEL_NAME, cfg.dose, device, cfg.output_dir
    )

    train_ds = WGANVGGDataset(
        TRAIN_SPLIT, cfg.data_root, patch_size=PATCH_SIZE, patch_n=PATCH_N
    )
    test_ds = WGANVGGDataset(TEST_SPLIT, cfg.data_root)
    val_ds = optional_split(lambda: WGANVGGDataset(VAL_SPLIT, cfg.data_root), VAL_SPLIT)
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

    model = WGAN_VGG(input_size=PATCH_SIZE).to(device)
    train(model, train_loader, device, cfg, val_loader=val_loader)
    # As in the reference folder, testing uses the best saved model.
    load_best_for_test(cfg, model)
    test(model, test_loader, device, cfg)


if __name__ == "__main__":
    main()
