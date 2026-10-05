"""Validation, test loop, result figures and metric reports."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from .config import DATA_RANGE, TRUNC_MAX, TRUNC_MIN, RunConfig  # noqa: E402
from .data import patient_of, to_hu_window  # noqa: E402
from .metrics import (  # noqa: E402
    Measure,
    compute_measure,
    compute_psnr,
    compute_rmse,
    compute_ssim,
)

logger = logging.getLogger(__name__)

QUICK_EVAL_MAX_SLICES = 32
NUM_TEST_FIGURES = 8
ABDOMEN_FIG_NAME = "result_abdomen.png"

# Maps a test batch to (input, target, prediction), each a (H, W) HU tensor
# clipped to the evaluation window.
PredictFn = Callable[[Any], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]


@torch.no_grad()
def quick_eval(
    infer_fn: Callable[[torch.Tensor], torch.Tensor],
    loader: DataLoader,
    device: torch.device,
    max_slices: int = QUICK_EVAL_MAX_SLICES,
) -> Measure:
    """Mean PSNR/SSIM/RMSE of the prediction on the first validation slices.

    Args:
        infer_fn: Maps a normalised ``(1, 1, H, W)`` input to the prediction.
        loader: Validation loader with batch size 1.
        device: Device the inputs are moved to.
        max_slices: Number of slices evaluated.

    Returns:
        ``(psnr, ssim, rmse)`` averaged over the evaluated slices, or zeros if
        the loader is empty.
    """
    psnr_sum, ssim_sum, rmse_sum, n = 0.0, 0.0, 0.0, 0
    for x, y in loader:
        if n >= max_slices:
            break
        x = x.float().to(device).unsqueeze(1)
        y = y.float().to(device).unsqueeze(1)
        pred = infer_fn(x).clamp(0.0, 1.0)
        yv, pv = to_hu_window(y), to_hu_window(pred)
        psnr_sum += compute_psnr(pv, yv, DATA_RANGE)
        ssim_sum += compute_ssim(pv, yv, DATA_RANGE)
        rmse_sum += compute_rmse(pv, yv)
        n += 1
    if n == 0:
        return 0.0, 0.0, 0.0
    return psnr_sum / n, ssim_sum / n, rmse_sum / n


def abdomen_fig_index(
    loader: DataLoader, abdomen_idx: int | None = None, abdomen_frac: float = 0.5
) -> int:
    """Pick the test slice saved as ``result_abdomen.png``.

    By default this is the mid-volume slice of the test patient with the most
    slices, which lands in the abdomen; the last slice of the loader would be
    in the pelvis.

    Args:
        loader: Test loader.
        abdomen_idx: Absolute loader index; takes precedence when given.
        abdomen_frac: Relative position inside the patient (0 = superior,
            1 = inferior).

    Returns:
        The loader index of the chosen slice.
    """
    ds = loader.dataset
    n = len(ds)
    if n <= 1:
        return 0
    if abdomen_idx is not None:
        return max(0, min(n - 1, abdomen_idx))
    frac = min(max(abdomen_frac, 0.0), 1.0)
    if hasattr(ds, "ld_triples"):
        paths = [t[1] for t in ds.ld_triples]
    elif hasattr(ds, "fd_triples"):
        paths = [t[1] for t in ds.fd_triples]
    elif hasattr(ds, "inputs"):
        paths = list(ds.inputs)
    else:
        return int(round(frac * (n - 1)))
    runs: dict[str | None, list[int]] = {}
    for i, p in enumerate(paths[:n]):
        runs.setdefault(patient_of(p), []).append(i)
    largest = max(runs.values(), key=len)
    idx = largest[int(round(frac * (len(largest) - 1)))]
    logger.info(
        "Abdomen figure -> loader idx %d/%d (%s) frac=%s",
        idx,
        n - 1,
        Path(str(paths[idx])).name,
        frac,
    )
    return idx


def save_fig(
    x: torch.Tensor,
    y: torch.Tensor,
    pred: torch.Tensor,
    idx: int,
    input_measure: Measure | None,
    pred_measure: Measure | None,
    fig_dir: Path,
    dose: str,
    name: str | None = None,
    vmin: float = TRUNC_MIN,
    vmax: float = TRUNC_MAX,
    rmse_unit: str = "",
) -> None:
    """Save an input / prediction / target comparison figure.

    Args:
        x: LDCT input image ``(H, W)``.
        y: Full-dose target image ``(H, W)``.
        pred: Prediction ``(H, W)``.
        idx: Slice index, used in the default file name.
        input_measure: ``(psnr, ssim, rmse)`` of the input, shown below it.
        pred_measure: ``(psnr, ssim, rmse)`` of the prediction.
        fig_dir: Output folder.
        dose: Dose label shown in the input title.
        name: File name; defaults to ``result_<idx>.png``.
        vmin: Lower display bound.
        vmax: Upper display bound.
        rmse_unit: Unit appended to the RMSE label (e.g. ``" HU"``).
    """
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    panels = (
        (axes[0], x, f"LDCT {dose}", input_measure),
        (axes[1], pred, "Pred", pred_measure),
        (axes[2], y, "Target", None),
    )
    for ax, img, title, measure in panels:
        ax.imshow(img.numpy(), cmap=plt.cm.gray, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=22)
        if measure is not None:
            ax.set_xlabel(
                f"PSNR {measure[0]:.3f}\nSSIM {measure[1]:.4f}\n"
                f"RMSE {measure[2]:.3f}{rmse_unit}",
                fontsize=14,
            )
    fig.savefig(Path(fig_dir) / (name or f"result_{idx:04d}.png"))
    plt.close(fig)


def run_test(
    loader: DataLoader,
    predict: PredictFn,
    cfg: RunConfig,
    ssim_offset: float = 0.0,
) -> tuple[Measure, Measure]:
    """Evaluate on the whole test split, save figures and write the metrics.

    Args:
        loader: Test loader with batch size 1.
        predict: Maps a batch to ``(input, target, prediction)`` HU images.
        cfg: Run configuration.
        ssim_offset: Forwarded to :func:`~common.metrics.compute_measure`.

    Returns:
        Mean ``(psnr, ssim, rmse)`` of the input and of the prediction.
    """
    input_sum, pred_sum, n = [0.0] * 3, [0.0] * 3, 0
    abdomen_idx = abdomen_fig_index(loader, cfg.abdomen_idx, cfg.abdomen_frac)
    with torch.no_grad():
        for i, batch in enumerate(loader):
            xv, yv, pv = predict(batch)
            o, p = compute_measure(xv, yv, pv, DATA_RANGE, ssim_offset=ssim_offset)
            for k in range(3):
                input_sum[k] += o[k]
                pred_sum[k] += p[k]
            n += 1
            if i < NUM_TEST_FIGURES:
                save_fig(xv, yv, pv, i, o, p, cfg.fig_dir, cfg.dose)
            if i == abdomen_idx:
                save_fig(
                    xv, yv, pv, i, o, p, cfg.fig_dir, cfg.dose, name=ABDOMEN_FIG_NAME
                )
    input_mean = tuple(v / n for v in input_sum)
    pred_mean = tuple(v / n for v in pred_sum)
    write_test_metrics(cfg, input_mean, pred_mean, n)
    return input_mean, pred_mean


def write_test_metrics(
    cfg: RunConfig,
    input_mean: Sequence[float],
    pred_mean: Sequence[float],
    n: int,
    rmse_unit: str = "",
) -> None:
    """Log the final test metrics and write them to ``cfg.metrics_path``.

    Args:
        cfg: Run configuration.
        input_mean: Mean ``(psnr, ssim, rmse)`` of the LDCT input.
        pred_mean: Mean ``(psnr, ssim, rmse)`` of the prediction.
        n: Number of test slices.
        rmse_unit: Unit appended to the RMSE values (e.g. ``" HU"``).
    """

    def fmt(m: Sequence[float]) -> str:
        return f"PSNR {m[0]:.4f} SSIM {m[1]:.4f} RMSE {m[2]:.4f}{rmse_unit}"

    logger.info("=== %s @ %s - original ===", cfg.model_name, cfg.dose)
    logger.info(fmt(input_mean))
    logger.info("=== %s @ %s - predicted ===", cfg.model_name, cfg.dose)
    logger.info(fmt(pred_mean))
    cfg.metrics_path.write_text(
        f"model={cfg.model_name} dose={cfg.dose} n={n}\n"
        f"original {fmt(input_mean)}\n"
        f"pred     {fmt(pred_mean)}\n"
    )
