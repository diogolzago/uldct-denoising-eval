"""Checkpoint saving, loading and resume-path resolution."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch import nn

logger = logging.getLogger(__name__)

LAST_SUFFIX = "_last.pt"
BEST_SUFFIX = "_best.pt"


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    epoch: int = 0,
    step: int = 0,
    lr: float = 0.0,
    loss: float = 0.0,
    psnr: float = 0.0,
    ssim: float = 0.0,
    rmse_hu: float = 0.0,
    extra_models: dict[str, nn.Module | None] | None = None,
) -> None:
    """Save model/optimizer state together with training progress and metrics.

    Args:
        path: Destination file.
        model: Model whose ``state_dict`` is stored under ``"model"``.
        optimizer: Optional optimizer.
        scheduler: Optional learning-rate scheduler.
        epoch: Last completed epoch.
        step: Global iteration counter.
        lr: Current learning rate.
        loss: Mean training loss of the epoch.
        psnr: Validation PSNR (dB).
        ssim: Validation SSIM.
        rmse_hu: Validation RMSE (HU).
        extra_models: Additional modules (e.g. ``{"ema_model": ema}``) stored
            under their key; ``None`` values are stored as ``None``.
    """
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": int(epoch),
        "step": int(step),
        "lr": float(lr),
        "loss": float(loss),
        "psnr": float(psnr),
        "ssim": float(ssim),
        "rmse_hu": float(rmse_hu),
    }
    for key, module in (extra_models or {}).items():
        state[key] = module.state_dict() if module is not None else None
    torch.save(state, path)


def _swap_suffix(name: str, old: str, new: str) -> str:
    """Replace the ``old`` suffix of ``name`` by ``new``."""
    return name[: -len(old)] + new


def resolve_checkpoint_path(path: Path, prefer: str = "last") -> Path:
    """Find the checkpoint to load when ``path`` itself may not exist.

    Looks for other ``.pt`` files in the same folder, in this order: the exact
    name, the ``_last``/``_best`` sibling, any file with the preferred suffix
    and finally (``prefer="last"`` only) any checkpoint. Staged checkpoints
    (``*_stage<k>.pt``) only match by exact name, because the optimizers of
    different stages have different parameter groups.

    Args:
        path: Requested checkpoint.
        prefer: ``"last"`` or ``"best"``.

    Returns:
        The chosen checkpoint, or ``path`` unchanged when nothing matches.
    """
    path = Path(path)
    if path.exists():
        return path
    candidates = sorted({str(p) for p in path.parent.glob("*.pt")})
    if not candidates:
        return path
    basename = path.name

    def named(name: str) -> list[str]:
        return [p for p in candidates if Path(p).name == name]

    def ending(suffix: str) -> list[str]:
        return [p for p in candidates if Path(p).name.endswith(suffix)]

    preferred = named(basename)
    if "_stage" in basename:
        return Path(preferred[-1]) if preferred else path
    if not preferred and basename.endswith(LAST_SUFFIX):
        preferred = named(_swap_suffix(basename, LAST_SUFFIX, BEST_SUFFIX))
    if not preferred and basename.endswith(BEST_SUFFIX):
        preferred = named(_swap_suffix(basename, BEST_SUFFIX, LAST_SUFFIX))
    if not preferred and prefer == "last":
        preferred = ending(LAST_SUFFIX)
    if not preferred and prefer == "best":
        preferred = ending(BEST_SUFFIX)
    if not preferred and prefer == "last":
        preferred = candidates
    if not preferred:
        return path
    chosen = Path(preferred[-1])
    logger.info("Resuming from local checkpoint: %s", chosen)
    return chosen


def load_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    extra_models: dict[str, nn.Module] | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any] | None:
    """Restore a checkpoint written by :func:`save_checkpoint`.

    Args:
        path: Requested checkpoint; resolved with :func:`resolve_checkpoint_path`.
        model: Model to load the weights into.
        optimizer: Optional optimizer to restore.
        scheduler: Optional scheduler to restore.
        extra_models: Additional modules restored from their key, when present.
        map_location: Device the tensors are loaded onto.

    Returns:
        The checkpoint dictionary, or ``None`` when no checkpoint was found.
    """
    path = resolve_checkpoint_path(path, prefer="last")
    if not path.exists():
        return None
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"])
    for key, module in (extra_models or {}).items():
        if ckpt.get(key) is not None:
            module.load_state_dict(ckpt[key])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt


def best_psnr_so_far(path: Path) -> float:
    """Validation PSNR stored in the best checkpoint, or ``-inf`` if none."""
    path = resolve_checkpoint_path(path, prefer="best")
    if not path.exists():
        return -float("inf")
    try:
        return float(torch.load(path, map_location="cpu").get("psnr", -float("inf")))
    except Exception:
        return -float("inf")
