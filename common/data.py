"""ULDCT dataset: file discovery, LD/FD pairing, normalisation and patching."""

from __future__ import annotations

import functools
import glob
import logging
import re
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import (
    LOW_DOSE_DIR,
    NORM_MAX,
    NORM_MIN,
    NORMAL_DOSE_DIR,
    NORMAL_DOSE_LEAF,
    TRUNC_MAX,
    TRUNC_MIN,
)

try:
    import pydicom
except ImportError:
    pydicom = None

logger = logging.getLogger(__name__)

DICOM_SUFFIXES = (".ima", ".dcm")
TARGET_SUFFIXES = (*DICOM_SUFFIXES, ".npy")
# Decoded-DICOM caches (``slice.IMA.npy``) that may sit next to the targets.
DICOM_CACHE_SUFFIXES = (".ima.npy", ".dcm.npy")

_PATIENT_RE = re.compile(r"/(L\d{3,4})(?:/|_)")
_SLICE_FILE_RE = re.compile(r"^slice_(\d+)$")


# ----------------------------------------------------------------------------
# Intensity transforms
# ----------------------------------------------------------------------------
def normalize(img: np.ndarray) -> np.ndarray:
    """Map HU values onto ``[0, 1]`` using ``[NORM_MIN, NORM_MAX]``."""
    return (img - NORM_MIN) / (NORM_MAX - NORM_MIN)


def denormalize(img: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`normalize`: map ``[0, 1]`` back to HU."""
    return img * (NORM_MAX - NORM_MIN) + NORM_MIN


def truncate(img: torch.Tensor) -> torch.Tensor:
    """Clip ``img`` to the evaluation window ``[TRUNC_MIN, TRUNC_MAX]`` in place.

    Returns:
        The same (modified) tensor, for chaining.
    """
    img[img <= TRUNC_MIN] = TRUNC_MIN
    img[img >= TRUNC_MAX] = TRUNC_MAX
    return img


def to_hu_window(img: torch.Tensor) -> torch.Tensor:
    """Convert a normalised square image to HU, clipped to the evaluation window.

    Args:
        img: Normalised image holding ``s * s`` values (any leading dims of 1).

    Returns:
        A ``(s, s)`` CPU tensor in HU.
    """
    s = img.size(-1)
    return truncate(denormalize(img.view(s, s).cpu().detach()))


# ----------------------------------------------------------------------------
# File discovery and LD/FD pairing
# ----------------------------------------------------------------------------
def patient_of(path: str) -> str | None:
    """Extract the Mayo patient id (e.g. ``L067``) from a file path."""
    match = _PATIENT_RE.search(path.replace("\\", "/"))
    return match.group(1) if match else None


def slice_index(path: str) -> int | None:
    """Extract the slice index from a file name.

    Supports ``slice_NNNN.npy``, Mayo ``*.CT.<series>.<idx>.*`` names and, as a
    fallback, all digits of the stem.
    """
    stem = Path(path).stem
    match = _SLICE_FILE_RE.match(stem)
    if match:
        return int(match.group(1))
    parts = stem.split(".")
    for i, part in enumerate(parts):
        if part == "CT" and i + 2 < len(parts):
            try:
                return int(parts[i + 2])
            except ValueError:
                pass
    digits = "".join(c for c in stem if c.isdigit())
    return int(digits) if digits else None


def list_low_dose_inputs(data_root: Path, split: str) -> list[str]:
    """List the LDCT ``.npy`` slices of a split, sorted by path.

    ``glob.glob`` is used instead of ``Path.rglob`` to keep its handling of
    hidden files and directory symlinks, which determines the file order.
    """
    pattern = Path(data_root) / split / LOW_DOSE_DIR / "**" / "*.npy"
    return sorted(glob.glob(str(pattern), recursive=True))


def normal_dose_dir(data_root: Path, patient_id: str, split: str) -> Path:
    """Folder holding the full-dose slices of one patient."""
    return Path(data_root) / split / NORMAL_DOSE_DIR / patient_id / NORMAL_DOSE_LEAF


@functools.cache
def list_normal_dose(
    data_root: Path, patient_id: str, split: str, dicom_only: bool = False
) -> list[str]:
    """List the full-dose target files of one patient, sorted by slice index.

    Args:
        data_root: Dataset root.
        patient_id: Mayo patient id.
        split: Dataset split.
        dicom_only: Keep only DICOM files, falling back to ``.npy`` only when
            the patient has no DICOM at all. Used by FGDM; see
            :func:`_select_target_files`.

    Returns:
        Target paths (empty when the folder does not exist). The result is
        cached; callers must not modify it.
    """
    folder = normal_dose_dir(data_root, patient_id, split)
    if not folder.exists():
        return []
    files = _select_target_files(list(folder.iterdir()), dicom_only)
    return sorted((str(f) for f in files), key=lambda s: (slice_index(s) or 0, s))


def _select_target_files(entries: list[Path], dicom_only: bool) -> list[Path]:
    """Filter the files of a full-dose folder down to the actual targets.

    Decoded-DICOM caches must be skipped: they would duplicate the target list
    and shift the LD/FD pairing. The default rule only recognises caches named
    ``<name>.IMA.npy``; ``dicom_only`` also rejects ``<stem>.npy`` caches.
    """

    def is_npy_target(f: Path) -> bool:
        return f.suffix.lower() == ".npy" and not f.name.lower().endswith(
            DICOM_CACHE_SUFFIXES
        )

    if not dicom_only:
        return [
            f
            for f in entries
            if f.suffix.lower() in TARGET_SUFFIXES
            and not f.name.lower().endswith(DICOM_CACHE_SUFFIXES)
        ]
    dicom = [f for f in entries if f.suffix.lower() in DICOM_SUFFIXES]
    return dicom or [f for f in entries if is_npy_target(f)]


def group_by_patient_sorted(paths: Iterable[str]) -> dict[str, list[str]]:
    """Group paths by patient id; each group is sorted by slice index.

    Paths without a recognisable patient id are dropped.
    """
    groups: dict[str, list[str]] = {}
    for p in paths:
        pid = patient_of(p)
        if pid is not None:
            groups.setdefault(pid, []).append(p)
    for pid in groups:
        groups[pid].sort(key=lambda p: slice_index(p) or 0)
    return groups


def pair_inputs_with_targets(
    input_paths: list[str],
    data_root: Path,
    split: str,
    dicom_only: bool = False,
    warn_count_mismatch: bool = False,
) -> tuple[list[str], list[str]]:
    """Pair LDCT inputs with full-dose targets by sorted position per patient.

    Each patient contributes ``min(#LD, #FD)`` pairs; patients without
    full-dose data in ``split`` are skipped.

    Args:
        input_paths: LDCT slice paths.
        data_root: Dataset root.
        split: Dataset split.
        dicom_only: Forwarded to :func:`list_normal_dose`.
        warn_count_mismatch: Log a warning for patients whose LD and FD slice
            counts differ by more than 5% (a symptom of a polluted FD folder).

    Returns:
        The kept input paths and their matching target paths.
    """
    groups = group_by_patient_sorted(input_paths)
    kept: list[str] = []
    targets: list[str] = []
    skipped: list[str] = []
    mismatched: list[str] = []
    for pid in sorted(groups):
        inputs = groups[pid]
        fd_list = list_normal_dose(Path(data_root), pid, split, dicom_only)
        if not fd_list:
            skipped.append(pid)
            continue
        if warn_count_mismatch and abs(len(inputs) - len(fd_list)) > max(
            2, int(0.05 * len(inputs))
        ):
            mismatched.append(f"{pid}(LD={len(inputs)},FD={len(fd_list)})")
        n = min(len(inputs), len(fd_list))
        kept.extend(inputs[:n])
        targets.extend(fd_list[:n])
    if skipped:
        logger.info("No full-dose data for %s (split=%s); skipping", skipped, split)
    if mismatched:
        logger.warning(
            "LD/FD slice count mismatch (split=%s): %s; check for .npy caches in "
            "the normal-dose folder",
            split,
            mismatched,
        )
    return kept, targets


def load_target_array(path: str) -> np.ndarray:
    """Load a full-dose target, normalised to ``[0, 1]``.

    Args:
        path: ``.IMA``/``.dcm`` DICOM file (rescaled to HU) or ``.npy`` in HU.
    """
    if path.lower().endswith(DICOM_SUFFIXES):
        if pydicom is None:
            raise RuntimeError("pydicom is required to read DICOM targets")
        ds = pydicom.dcmread(path, force=True)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        hu = ds.pixel_array.astype(np.float32) * slope + intercept
        return normalize(hu).astype(np.float32)
    return normalize(np.load(path).astype(np.float32))


def load_input_array(path: str) -> np.ndarray:
    """Load an LDCT ``.npy`` slice (HU), normalised to ``[0, 1]``."""
    return normalize(np.load(path).astype(np.float32))


# ----------------------------------------------------------------------------
# Patching and dataset
# ----------------------------------------------------------------------------
def random_patch_pairs(
    x: np.ndarray, y: np.ndarray, patch_size: int, patch_n: int | None
) -> tuple[np.ndarray, np.ndarray]:
    """Crop ``patch_n`` aligned random patches from an input/target pair.

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
    xs, ys = [], []
    for _ in range(int(patch_n or 1)):
        top = np.random.randint(0, h - ps + 1) if h > ps else 0
        left = np.random.randint(0, w - ps + 1) if w > ps else 0
        xs.append(x[top : top + ps, left : left + ps])
        ys.append(y[top : top + ps, left : left + ps])
    return np.stack(xs), np.stack(ys)


class ULDCTDataset(Dataset):
    """Paired ULDCT inputs and full-dose targets of one split.

    Inputs are ``.npy`` LDCT slices; targets are the matching full-dose slices.
    With ``patch_size`` set, each item is a stack of random aligned patches
    (training); otherwise it is the full slice pair (evaluation).

    Subclasses customise items by overriding :meth:`__getitem__` and calling
    :meth:`load_pair`.

    Attributes:
        split: Dataset split.
        inputs: LDCT input paths.
        targets: Full-dose target paths, aligned with ``inputs``.
    """

    def __init__(
        self,
        split: str,
        data_root: Path,
        patch_size: int | None = None,
        patch_n: int | None = None,
        dicom_only: bool = False,
        warn_count_mismatch: bool = False,
    ) -> None:
        """Discover and pair the files of ``split``.

        Args:
            split: Dataset split.
            data_root: Dataset root.
            patch_size: Training patch side, or ``None`` for full slices.
            patch_n: Patches per slice.
            dicom_only: Forwarded to :func:`pair_inputs_with_targets`.
            warn_count_mismatch: Forwarded to :func:`pair_inputs_with_targets`.

        Raises:
            FileNotFoundError: If the split has no inputs or no LD/FD pairs.
        """
        self.split = split
        self.data_root = Path(data_root)
        self.patch_size, self.patch_n = patch_size, patch_n
        inputs = list_low_dose_inputs(self.data_root, split)
        if not inputs:
            raise FileNotFoundError(
                f"No .npy inputs in {self.data_root / split / LOW_DOSE_DIR}"
            )
        self.inputs, self.targets = pair_inputs_with_targets(
            inputs, self.data_root, split, dicom_only, warn_count_mismatch
        )
        if not self.inputs:
            raise FileNotFoundError(f"No LD/FD pairs in split {split!r}")

    def __len__(self) -> int:
        """Number of paired slices."""
        return len(self.inputs)

    def load_pair(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Load the normalised input and target slices at ``idx``."""
        return load_input_array(self.inputs[idx]), load_target_array(self.targets[idx])

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the slice pair, or random patches of it when patching."""
        x, y = self.load_pair(idx)
        if self.patch_size:
            return random_patch_pairs(x, y, self.patch_size, self.patch_n)
        return x, y
