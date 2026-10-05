"""Shared constants, command-line interface and logging setup."""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

DOSES = ("5pct", "10pct")
TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT = "train", "validation", "test"

# Dataset layout:
#   {data_root}/{split}/images_low_dose/**/*.npy                (LDCT inputs)
#   {data_root}/{split}/images_normal_dose/{pid}/full_1mm/*.IMA (full-dose targets)
LOW_DOSE_DIR = "images_low_dose"
NORMAL_DOSE_DIR = "images_normal_dose"
NORMAL_DOSE_LEAF = "full_1mm"

# HU range mapped linearly onto [0, 1] by the input normalisation.
NORM_MIN, NORM_MAX = -1024.0, 3072.0
# Soft-tissue window (HU) used for every reported metric and figure.
TRUNC_MIN, TRUNC_MAX = -160.0, 240.0
DATA_RANGE = TRUNC_MAX - TRUNC_MIN

DEFAULT_NUM_WORKERS = 8
DEFAULT_ABDOMEN_FRAC = 0.5

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunConfig:
    """Run-level settings shared by every model script.

    Attributes:
        model_name: Short model identifier, used in output file names.
        dose: Dose level of the LDCT inputs (``"5pct"`` or ``"10pct"``).
        data_root: Root of the ULDCT dataset (contains one folder per split).
        output_dir: Folder for checkpoints, losses, metrics and figures.
        num_workers: Number of DataLoader worker processes.
        abdomen_idx: Absolute test-loader index of the slice saved as
            ``result_abdomen.png``; overrides ``abdomen_frac`` when set.
        abdomen_frac: Relative position (0 = superior, 1 = inferior) of the
            ``result_abdomen.png`` slice within the largest test patient.
    """

    model_name: str
    dose: str
    data_root: Path
    output_dir: Path
    num_workers: int = DEFAULT_NUM_WORKERS
    abdomen_idx: int | None = None
    abdomen_frac: float = DEFAULT_ABDOMEN_FRAC

    @property
    def fig_dir(self) -> Path:
        """Folder where test figures are written."""
        return self.output_dir / "fig"

    @property
    def run_name(self) -> str:
        """Prefix shared by checkpoint file names, e.g. ``red_cnn_5pct``."""
        return f"{self.model_name}_{self.dose}"

    @property
    def last_ckpt(self) -> Path:
        """Checkpoint written at the end of every epoch (used to resume)."""
        return self.output_dir / f"{self.run_name}_last.pt"

    @property
    def best_ckpt(self) -> Path:
        """Checkpoint with the best validation PSNR so far."""
        return self.output_dir / f"{self.run_name}_best.pt"

    @property
    def losses_path(self) -> Path:
        """Per-iteration training losses (``.npy``)."""
        return self.output_dir / f"losses_{self.dose}.npy"

    @property
    def metrics_path(self) -> Path:
        """Final test metrics (plain text)."""
        return self.output_dir / f"metrics_{self.dose}.txt"


def build_arg_parser(
    model_name: str, description: str | None = None
) -> argparse.ArgumentParser:
    """Create the command-line parser with the options common to all models.

    Model scripts may add their own options to the returned parser before
    calling :func:`parse_run_config`.

    Args:
        model_name: Model identifier, used to derive the default output folder.
        description: Help text shown by ``--help``.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dose",
        choices=DOSES,
        default="5pct",
        help="LDCT dose level (default: %(default)s).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Dataset root (default: ./data/uldct_<dose>/dataset).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Output folder (default: ./outputs/<dose>/{model_name}_<dose>).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help="DataLoader worker processes (default: %(default)s).",
    )
    parser.add_argument(
        "--abdomen-idx",
        type=int,
        default=None,
        help="Test-loader index of the slice saved as result_abdomen.png.",
    )
    parser.add_argument(
        "--abdomen-frac",
        type=float,
        default=DEFAULT_ABDOMEN_FRAC,
        help="Relative slice position without --abdomen-idx (default: %(default)s).",
    )
    return parser


def parse_run_config(
    parser: argparse.ArgumentParser, model_name: str, argv: list[str] | None = None
) -> tuple[RunConfig, argparse.Namespace]:
    """Parse the command line and build the run configuration.

    Also creates the output and figure folders and configures logging.

    Args:
        parser: Parser returned by :func:`build_arg_parser`, optionally extended.
        model_name: Model identifier.
        argv: Arguments to parse; defaults to ``sys.argv[1:]``.

    Returns:
        The run configuration and the raw parsed namespace (for model-specific
        options).
    """
    args = parser.parse_args(argv)
    data_root = args.data_root or Path("data") / f"uldct_{args.dose}" / "dataset"
    output_dir = (
        args.output_dir or Path("outputs") / args.dose / f"{model_name}_{args.dose}"
    )
    cfg = RunConfig(
        model_name=model_name,
        dose=args.dose,
        data_root=data_root,
        output_dir=output_dir,
        num_workers=args.num_workers,
        abdomen_idx=args.abdomen_idx,
        abdomen_frac=args.abdomen_frac,
    )
    cfg.fig_dir.mkdir(parents=True, exist_ok=True)
    setup_logging()
    return cfg, args


def setup_logging(level: int = logging.INFO) -> None:
    """Send log records to stdout with timestamps.

    Args:
        level: Minimum level of the records that are emitted.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,
    )


def get_device() -> torch.device:
    """Return the CUDA device when available, otherwise the CPU.

    Select a specific GPU with the ``CUDA_VISIBLE_DEVICES`` environment
    variable.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
