"""Helpers shared by the ``main`` entry points of the model scripts."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from torch import nn
from torch.utils.data import Dataset

from .checkpoint import load_checkpoint
from .config import RunConfig

logger = logging.getLogger(__name__)


def optional_split(factory: Callable[[], Dataset], split: str) -> Dataset | None:
    """Build the dataset of an optional split, or return ``None`` if missing.

    Used for the validation split: without it, per-epoch evaluation falls back
    to the test split instead of aborting the run.

    Args:
        factory: Zero-argument callable that builds the dataset.
        split: Split name, used in the log message.
    """
    try:
        return factory()
    except FileNotFoundError as exc:
        logger.warning(
            "Split %r unavailable (%s); using TEST for per-epoch evaluation", split, exc
        )
        return None


def load_best_for_test(cfg: RunConfig, model: nn.Module, **kwargs: Any) -> None:
    """Load the best checkpoint (falling back to the last one) before testing.

    Keeps the in-memory weights when no checkpoint exists.

    Args:
        cfg: Run configuration.
        model: Model to load the weights into.
        **kwargs: Forwarded to :func:`~common.checkpoint.load_checkpoint`.
    """
    ckpt = load_checkpoint(cfg.best_ckpt, model=model, **kwargs)
    if ckpt is not None:
        logger.info(
            "Testing saved checkpoint (epoch=%s psnr=%.3f)",
            ckpt.get("epoch"),
            float(ckpt.get("psnr", float("nan"))),
        )
    else:
        logger.info("No saved checkpoint; testing the last-epoch weights")
