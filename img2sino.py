"""Image <-> transmission-sinogram projection with the SPDiff fan-beam geometry.

Used by NEED (SPDiff), whose diffusion runs in the projection domain: images are
forward-projected to transmission sinograms ``y = I / I0`` in ``(0, 1]``, where
the Poisson degradation of SPDiff is physically meaningful, and reconstructed
back with FBP.

Backends:
    * ``torch_radon.RadonFanbeam`` (the library used by the paper), when
      installed.
    * A native PyTorch ray-driven fan-beam projector with an approximate FBP,
      using the same geometry (default fallback; slower).
    * A "pseudo" sinogram (bilinear resize), kept only as a legacy option.

Pipeline (exact inverse of the SPDiff reconstruction)::

    norm [0, 1] -> HU -> mu (512x512) -> P(mu) / kappa -> exp(-.) -> y (672x672)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

try:
    from torch_radon import RadonFanbeam

    HAS_TORCH_RADON = True
    _TORCH_RADON_ERR: Exception | None = None
except Exception as exc:  # torch_radon also fails with CUDA/ABI errors
    HAS_TORCH_RADON = False
    _TORCH_RADON_ERR = exc

logger = logging.getLogger(__name__)

# Fan-beam geometry copied verbatim from SPDiff (SPDiff.py:79-95). The forward
# projection and recon() must share it exactly for the round trip to close.
IMAGE_SIZE = 512
DETECT_NUMBER = 672
N_ANGLES = 672
SRC_DIST = 1361.2 - 615.18
DET_DIST = 615.18
DET_SPACING = 1.85 / 0.68
ANGLES = np.linspace(-0.5 * np.pi, 1.5 * np.pi, N_ANGLES).astype(np.float32)
U_WATER = 0.0192
KAPPA = 1.5
CLIP_TO_CIRCLE = False

# HU <-> [0, 1] normalisation, identical to the dataset and SPDiff recon.
NORM_MIN, NORM_MAX = -1024.0, 3072.0

_EPS = 1e-6
# Native FBP scale: the empirical recon factor inflated intensities ~2.75x;
# 0.364 puts water back at 0 HU (measured with recon(forward(phantom)), stable
# for 96/192/256 ray samples).
_NATIVE_FBP_CAL = 0.364
# Transmission floor for the native FBP: a predicted ~0 transmission gives
# -log(1e-6) * KAPPA ~ 20.7, i.e. a bright radial streak. 1e-3 (~10.4) removes
# it without clipping real tissue (physical floor ~4e-3).
_NATIVE_FBP_TRANS_FLOOR = 1e-3

DEFAULT_RAY_SAMPLES = 256
DEFAULT_ANGLE_CHUNK = 4
BACKENDS = ("fanbeam", "pseudo")

# Evaluation window (HU) used by the round-trip check.
_TRUNC_MIN, _TRUNC_MAX = -160.0, 240.0


class _NativePseudoRadon:
    """Marker for the legacy pseudo-sinogram backend (bilinear resize)."""

    is_native_pseudo = True


class _NativeFanbeamRadon:
    """Native PyTorch fan-beam projector with the SPDiff geometry.

    Ray-driven forward projection and approximate FBP. Slower than
    ``torch_radon`` but dependency-free, keeping the fan-beam physics,
    detector and angles of the paper.

    Attributes:
        ray_samples: Samples along each ray in the forward projection.
        angle_chunk: Number of views processed per vectorised step.
    """

    is_native_fanbeam = True

    def __init__(
        self,
        ray_samples: int = DEFAULT_RAY_SAMPLES,
        angle_chunk: int = DEFAULT_ANGLE_CHUNK,
    ) -> None:
        """Store the sampling parameters.

        Args:
            ray_samples: Samples along each ray.
            angle_chunk: Views per vectorised step (memory/speed trade-off).
        """
        self.ray_samples = int(ray_samples)
        self.angle_chunk = int(angle_chunk)


# RadonFanbeam | _NativeFanbeamRadon | _NativePseudoRadon
Radon = Any


def build_radon(
    backend: str = "fanbeam",
    require_torch_radon: bool = False,
    ray_samples: int = DEFAULT_RAY_SAMPLES,
    angle_chunk: int = DEFAULT_ANGLE_CHUNK,
) -> Radon:
    """Create the projection backend.

    ``torch_radon`` is always preferred when installed; otherwise ``backend``
    selects the native fallback.

    Args:
        backend: Native fallback, ``"fanbeam"`` or ``"pseudo"``.
        require_torch_radon: Raise instead of falling back when ``torch_radon``
            is missing.
        ray_samples: Ray samples of the native fan-beam projector.
        angle_chunk: Views per step of the native fan-beam projector.

    Returns:
        The projector object passed to :func:`image_to_sinogram` and
        :func:`recon`.

    Raises:
        RuntimeError: If ``require_torch_radon`` is set and it is unavailable.
    """
    if HAS_TORCH_RADON:
        return RadonFanbeam(
            IMAGE_SIZE,
            ANGLES,
            SRC_DIST,
            DET_DIST,
            det_count=DETECT_NUMBER,
            det_spacing=DET_SPACING,
            clip_to_circle=CLIP_TO_CIRCLE,
        )
    if require_torch_radon:
        raise RuntimeError(
            f"torch_radon is required but could not be imported: {_TORCH_RADON_ERR}"
        )
    if backend.lower() == "pseudo":
        logger.info("torch_radon unavailable; using the pseudo-sinogram backend")
        return _NativePseudoRadon()
    projector = _NativeFanbeamRadon(ray_samples, angle_chunk)
    logger.info(
        "torch_radon unavailable; using the native fan-beam projector "
        "(%dx%d, samples=%d)",
        N_ANGLES,
        DETECT_NUMBER,
        projector.ray_samples,
    )
    return projector


# ----------------------------------------------------------------------------
# Point-wise conversions HU <-> norm <-> mu
# ----------------------------------------------------------------------------
def denorm_to_hu(norm: torch.Tensor) -> torch.Tensor:
    """Map normalised ``[0, 1]`` values to HU."""
    return norm * (NORM_MAX - NORM_MIN) + NORM_MIN


def hu_to_norm(hu: torch.Tensor) -> torch.Tensor:
    """Map HU values to ``[0, 1]``."""
    return (hu - NORM_MIN) / (NORM_MAX - NORM_MIN)


def hu_to_mu(hu: torch.Tensor) -> torch.Tensor:
    """Convert HU to linear attenuation (inverse of the recon HU formula)."""
    return (hu / 1000.0) * U_WATER + U_WATER


def _native_detector_positions(
    device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Centred detector-element positions (mm)."""
    dc = (DETECT_NUMBER - 1) / 2.0
    return (torch.arange(DETECT_NUMBER, device=device, dtype=dtype) - dc) * DET_SPACING


def _native_angles(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Projection angles (rad) as a tensor."""
    return torch.as_tensor(ANGLES, device=device, dtype=dtype)


def _native_fanbeam_forward(
    mu: torch.Tensor, radon: _NativeFanbeamRadon
) -> torch.Tensor:
    """Ray-driven fan-beam projection of attenuation maps.

    Args:
        mu: Attenuation maps ``(..., H, W)``.
        radon: Native projector holding the sampling parameters.

    Returns:
        Transmission sinograms ``(..., N_ANGLES, DETECT_NUMBER)``.
    """
    *lead, H, W = mu.shape
    flat = mu.reshape(-1, 1, H, W).contiguous()
    n = flat.shape[0]
    device, dtype = flat.device, flat.dtype
    half = (IMAGE_SIZE - 1) / 2.0
    samples = max(8, int(radon.ray_samples))
    angle_chunk = max(1, int(radon.angle_chunk))
    dsd = SRC_DIST + DET_DIST

    a = torch.linspace(-half, half, samples, device=device, dtype=dtype)
    da = (2.0 * half) / max(samples - 1, 1)
    det = _native_detector_positions(device, dtype)
    path_scale = torch.sqrt(1.0 + (det / dsd) ** 2)
    angles = _native_angles(device, dtype)

    outs = []
    for start in range(0, N_ANGLES, angle_chunk):
        beta = angles[start : start + angle_chunk]
        cb = torch.cos(beta)[:, None, None]
        sb = torch.sin(beta)[:, None, None]
        aa = a[None, :, None]
        tt = det[None, None, :]
        b = tt * (aa + SRC_DIST) / dsd

        # Axial basis v = (-sin, cos); detector axis e = (cos, sin).
        x = -aa * sb + b * cb
        y = aa * cb + b * sb
        gx = (2.0 * x / max(W - 1, 1)).clamp(-2.0, 2.0)
        gy = (2.0 * y / max(H - 1, 1)).clamp(-2.0, 2.0)
        grid = torch.stack((gx, gy), dim=-1)  # (views, samples, det, 2)
        c = grid.shape[0]
        grid = grid.unsqueeze(0).expand(n, c, samples, DETECT_NUMBER, 2)
        grid = grid.reshape(n, c * samples, DETECT_NUMBER, 2)
        vals = F.grid_sample(
            flat, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        vals = vals.reshape(n, c, samples, DETECT_NUMBER)
        outs.append(vals.sum(dim=2) * da * path_scale[None, None, :])

    li = torch.cat(outs, dim=1)
    yi = torch.exp(-(li / KAPPA)).clamp(min=_EPS, max=1.0)
    return yi.reshape(*lead, N_ANGLES, DETECT_NUMBER)


def _native_ramp_filter(sino: torch.Tensor) -> torch.Tensor:
    """Apply a ramp filter along the detector axis."""
    n = sino.shape[-1]
    freqs = torch.fft.rfftfreq(n, d=DET_SPACING, device=sino.device).to(sino.dtype)
    spec = torch.fft.rfft(sino, dim=-1)
    return torch.fft.irfft(spec * torch.abs(freqs), n=n, dim=-1)


def _native_fanbeam_recon(yi: torch.Tensor, radon: _NativeFanbeamRadon) -> torch.Tensor:
    """Approximate fan-beam FBP for the native backend.

    Args:
        yi: Transmission sinograms ``(..., A, D)``.
        radon: Native projector holding the sampling parameters.

    Returns:
        Normalised images ``(..., IMAGE_SIZE, IMAGE_SIZE)`` (not clipped).
    """
    *lead, A, D = yi.shape
    flat = yi.reshape(-1, A, D).contiguous()
    flat = flat.clamp(min=_NATIVE_FBP_TRANS_FLOOR, max=1.0)
    n = flat.shape[0]
    device, dtype = flat.device, flat.dtype
    dsd = SRC_DIST + DET_DIST
    dc = (DETECT_NUMBER - 1) / 2.0
    angle_chunk = max(1, int(radon.angle_chunk))

    filt = _native_ramp_filter(-torch.log(flat) * KAPPA)

    pix = torch.linspace(
        -(IMAGE_SIZE - 1) / 2.0,
        (IMAGE_SIZE - 1) / 2.0,
        IMAGE_SIZE,
        device=device,
        dtype=dtype,
    )
    yy, xx = torch.meshgrid(pix, pix, indexing="ij")
    angles = _native_angles(device, dtype)
    out = torch.zeros((n, IMAGE_SIZE, IMAGE_SIZE), device=device, dtype=dtype)

    # grid_sample over the sinogram: x = detector index, y = angle index.
    inp = filt.unsqueeze(1)
    for start in range(0, N_ANGLES, angle_chunk):
        beta = angles[start : start + angle_chunk]
        cb = torch.cos(beta)[:, None, None]
        sb = torch.sin(beta)[:, None, None]
        a = yy[None] * cb - xx[None] * sb
        lat = xx[None] * cb + yy[None] * sb
        L = (SRC_DIST + a).clamp(min=1.0)
        idx = (dsd * lat / L) / DET_SPACING + dc
        gx = 2.0 * idx / max(D - 1, 1) - 1.0
        if A > 1:
            angle_idx = torch.arange(
                start, start + beta.numel(), device=device, dtype=dtype
            )
            gy = (2.0 * angle_idx[:, None, None] / (A - 1) - 1.0).expand_as(gx)
        else:
            gy = torch.zeros_like(gx)
        grid = torch.stack((gx, gy), dim=-1)  # (views, H, W, 2)
        c = grid.shape[0]
        grid = grid.unsqueeze(0).expand(n, c, IMAGE_SIZE, IMAGE_SIZE, 2)
        grid = grid.reshape(n, c * IMAGE_SIZE, IMAGE_SIZE, 2)
        vals = F.grid_sample(
            inp, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        vals = vals.reshape(n, c, IMAGE_SIZE, IMAGE_SIZE)
        out += (vals * ((SRC_DIST / L) ** 2)[None]).sum(dim=1)

    # Empirical scale, analogous to the torch_radon ramp/backprojection pair.
    out = out * (2.0 * np.pi / max(N_ANGLES, 1)) * DET_SPACING * _NATIVE_FBP_CAL
    hu = (out - U_WATER) / U_WATER * 1000.0
    norm = (hu + 1024.0) / (3072.0 + 1024.0)
    return norm.reshape(*lead, IMAGE_SIZE, IMAGE_SIZE)


# ----------------------------------------------------------------------------
# Forward projection and reconstruction
# ----------------------------------------------------------------------------
def _as_torch(img: torch.Tensor | np.ndarray) -> torch.Tensor:
    """Convert to a float tensor; NumPy input goes to the GPU when available."""
    if torch.is_tensor(img):
        return img.float()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.as_tensor(np.asarray(img), dtype=torch.float32, device=device)


def image_to_sinogram(
    img: torch.Tensor | np.ndarray, radon: Radon, *, is_norm: bool = True
) -> torch.Tensor:
    """Forward-project images to transmission sinograms.

    Args:
        img: Images ``(..., H, W)``, normalised to ``[0, 1]`` if ``is_norm``,
            otherwise in HU.
        radon: Projector from :func:`build_radon`.
        is_norm: Whether ``img`` is normalised.

    Returns:
        Transmission sinograms ``(..., N_ANGLES, DETECT_NUMBER)`` in
        ``[_EPS, 1]``.
    """
    x = _as_torch(img)
    hu = denorm_to_hu(x) if is_norm else x

    if getattr(radon, "is_native_pseudo", False):
        norm = hu_to_norm(hu).clamp(0.0, 1.0)
        *lead, H, W = norm.shape
        flat = norm.reshape(-1, 1, H, W).contiguous()
        yi = F.interpolate(
            flat, size=(N_ANGLES, DETECT_NUMBER), mode="bilinear", align_corners=False
        )
        return yi.reshape(*lead, N_ANGLES, DETECT_NUMBER).clamp(min=_EPS, max=1.0)

    if getattr(radon, "is_native_fanbeam", False):
        return _native_fanbeam_forward(hu_to_mu(hu).clamp(min=0.0), radon)

    if not x.is_cuda:
        hu = hu.cuda()
    mu = hu_to_mu(hu).clamp(min=0.0)
    *lead, H, W = mu.shape
    flat = mu.reshape(-1, H, W).contiguous()
    # recon() computes FBP(li_hat * kappa), so the forward model is P(mu) / kappa.
    li_hat = radon.forward(flat) / KAPPA
    yi = torch.exp(-li_hat).clamp(min=_EPS, max=1.0)
    return yi.reshape(*lead, *yi.shape[-2:])


def recon(yi: torch.Tensor | np.ndarray, radon: Radon) -> torch.Tensor:
    """Reconstruct normalised images from transmission sinograms.

    Mirrors ``recon()`` of SPDiff (basic_template.py:178).

    Args:
        yi: Transmission sinograms ``(..., A, D)``.
        radon: Projector from :func:`build_radon`.

    Returns:
        Normalised images ``(..., IMAGE_SIZE, IMAGE_SIZE)``; clipped to
        ``[0, 1]`` for the native backends only.
    """
    yi = _as_torch(yi)
    *lead, A, D = yi.shape

    if getattr(radon, "is_native_pseudo", False):
        flat = yi.reshape(-1, 1, A, D).contiguous()
        img = F.interpolate(
            flat, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False
        )
        return img.reshape(*lead, IMAGE_SIZE, IMAGE_SIZE).clamp(0.0, 1.0)

    if getattr(radon, "is_native_fanbeam", False):
        return _native_fanbeam_recon(yi, radon).clamp(0.0, 1.0)

    if not yi.is_cuda:
        yi = yi.cuda()
    flat = yi.reshape(-1, A, D).clone()
    flat[flat <= 0] = 1.0
    sino = -torch.log(flat) * KAPPA
    bp = radon.backprojection(radon.filter_sinogram(sino, "ramp"))
    img = (bp - U_WATER) / U_WATER * 1000.0
    img = (img + 1024.0) / (3072.0 + 1024.0)
    return img.reshape(*lead, *img.shape[-2:])


# ----------------------------------------------------------------------------
# Round-trip check (used to calibrate _NATIVE_FBP_CAL)
# ----------------------------------------------------------------------------
def read_image_hu(path: str) -> np.ndarray:
    """Read a ``.IMA``/``.dcm`` (rescaled to HU) or ``.npy`` (HU) slice."""
    if path.lower().endswith((".ima", ".dcm")):
        import pydicom

        ds = pydicom.dcmread(path, force=True)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        return ds.pixel_array.astype(np.float32) * slope + intercept
    return np.load(path).astype(np.float32)


def roundtrip_metrics(path: str, radon: Radon) -> tuple[float, float, tuple]:
    """Measure the ``recon(forward(img))`` error of one slice.

    Args:
        path: Image file readable by :func:`read_image_hu`.
        radon: Projector from :func:`build_radon`.

    Returns:
        PSNR (dB) and RMSE (HU) in the evaluation window, and the sinogram
        shape.
    """
    norm0 = hu_to_norm(_as_torch(read_image_hu(path))).clamp(0, 1)
    yi = image_to_sinogram(norm0, radon, is_norm=True)
    norm1 = recon(yi, radon).clamp(0, 1)
    a = denorm_to_hu(norm0).clamp(_TRUNC_MIN, _TRUNC_MAX)
    b = denorm_to_hu(norm1).clamp(_TRUNC_MIN, _TRUNC_MAX)
    mse = torch.mean((a - b) ** 2).item()
    dr = _TRUNC_MAX - _TRUNC_MIN
    psnr = float(10.0 * np.log10((dr**2) / mse)) if mse > 0 else float("inf")
    return psnr, float(np.sqrt(mse)), tuple(yi.shape)
