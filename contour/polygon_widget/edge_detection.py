"""Modern edge / gradient elevation algorithms.

This module gathers a collection of gradient and edge detection operators that
can be used interchangeably by the contour/via/polygon extractor, pipeline
steps and debug visualizations. Every public function returns a ``uint8``
elevation map (0..255) so the results can be fed into templates, watershed,
percentile thresholding and overlay rendering without additional
normalization.

Supported methods (``build_gradient_elevation`` dispatcher):

* ``sobel`` — classic Sobel/L2 gradient magnitude (baseline).
* ``scharr`` — Scharr operator, numerically optimal 3x3 filter; sharper than
  Sobel on fine edges.
* ``laplacian`` — |Laplacian| (second order derivative).
* ``log`` — Laplacian of Gaussian (multi-scale blob / edge response).
* ``auto_canny`` — Canny with automatic hysteresis thresholds from the
  median (Rosebrock style).
* ``structured`` — Structured Random Forest edge detection (requires
  ``cv2.ximgproc`` + a trained model). Falls back to Scharr + non-maximum
  thinning when the backend or model is unavailable.
* ``ridge`` — Hessian-ridge response that highlights crest lines and tubular
  structures (uses ``cv2.ximgproc.RidgeDetectionFilter`` when available,
  otherwise a NumPy Hessian eigenvalue fallback).
* ``phase_congruency`` — multi-scale / multi-orientation phase congruency, a
  contrast-invariant edge feature used by recent SOTA pipelines. Pure
  NumPy/OpenCV implementation that requires no external models.
* ``combined`` — element-wise maximum of several methods (robust ensemble).

These algorithms are intentionally implemented without a hard dependency on
``opencv-contrib`` / PyTorch: the heavy-weight methods degrade gracefully so
the widget keeps working inside the ``opencv-python`` only environment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .utils import ensure_uint8


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

EDGE_METHOD_SOBEL = "sobel"
EDGE_METHOD_SCHARR = "scharr"
EDGE_METHOD_LAPLACIAN = "laplacian"
EDGE_METHOD_LOG = "log"
EDGE_METHOD_AUTO_CANNY = "auto_canny"
EDGE_METHOD_STRUCTURED = "structured"
EDGE_METHOD_RIDGE = "ridge"
EDGE_METHOD_PHASE_CONGRUENCY = "phase_congruency"
EDGE_METHOD_COMBINED = "combined"


EDGE_METHOD_CHOICES: tuple[str, ...] = (
    EDGE_METHOD_SOBEL,
    EDGE_METHOD_SCHARR,
    EDGE_METHOD_LAPLACIAN,
    EDGE_METHOD_LOG,
    EDGE_METHOD_AUTO_CANNY,
    EDGE_METHOD_STRUCTURED,
    EDGE_METHOD_RIDGE,
    EDGE_METHOD_PHASE_CONGRUENCY,
    EDGE_METHOD_COMBINED,
)


def normalize_edge_method(name: str | None) -> str:
    text = str(name or "").strip().lower()
    if not text:
        return EDGE_METHOD_SOBEL
    aliases = {
        "sobel_l2": EDGE_METHOD_SOBEL,
        "sobel3": EDGE_METHOD_SOBEL,
        "canny": EDGE_METHOD_AUTO_CANNY,
        "canny_auto": EDGE_METHOD_AUTO_CANNY,
        "phase": EDGE_METHOD_PHASE_CONGRUENCY,
        "phasecongruency": EDGE_METHOD_PHASE_CONGRUENCY,
        "pc": EDGE_METHOD_PHASE_CONGRUENCY,
        "hed": EDGE_METHOD_STRUCTURED,
        "ml": EDGE_METHOD_STRUCTURED,
    }
    if text in aliases:
        return aliases[text]
    if text in EDGE_METHOD_CHOICES:
        return text
    return EDGE_METHOD_SOBEL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_gray(image: np.ndarray) -> np.ndarray:
    """Return a single-channel uint8 copy of ``image``."""

    data = ensure_uint8(image)
    if data.ndim == 3 and data.shape[2] == 4:
        return cv2.cvtColor(data, cv2.COLOR_BGRA2GRAY)
    if data.ndim == 3:
        return cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
    return data


def _to_bgr(image: np.ndarray) -> np.ndarray:
    data = ensure_uint8(image)
    if data.ndim == 2:
        return cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)
    if data.ndim == 3 and data.shape[2] == 4:
        return cv2.cvtColor(data, cv2.COLOR_BGRA2BGR)
    return data


def _normalize_float_map(magnitude: np.ndarray) -> np.ndarray:
    """Normalize an arbitrary float response into ``uint8`` [0..255]."""

    if magnitude.size == 0:
        return np.zeros_like(magnitude, dtype=np.uint8)
    magnitude = np.nan_to_num(magnitude, copy=False)
    maximum = float(np.max(np.abs(magnitude)))
    if maximum <= 1e-6:
        return np.zeros_like(magnitude, dtype=np.uint8)
    normalized = np.clip(magnitude / maximum, 0.0, 1.0) * 255.0
    return normalized.astype(np.uint8)


def _smooth_for_derivatives(gray: np.ndarray, smooth: bool) -> np.ndarray:
    if not smooth or gray.size == 0:
        return gray
    return cv2.GaussianBlur(gray, (3, 3), 0)


def _empty_like_gray(image: np.ndarray) -> np.ndarray:
    """Return an empty uint8 map that matches the 2D shape of ``image``."""

    if image is None:
        return np.zeros((0, 0), dtype=np.uint8)
    shape = image.shape[:2] if hasattr(image, "shape") else (0, 0)
    return np.zeros(shape, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Basic derivative operators
# ---------------------------------------------------------------------------


def sobel_magnitude(image: np.ndarray, ksize: int = 3, *, smooth: bool = True) -> np.ndarray:
    """Classical Sobel-based L2 gradient magnitude (uint8)."""

    gray = _to_gray(image)
    if gray.size == 0:
        return _empty_like_gray(gray)
    ksize = max(1, int(ksize))
    if ksize % 2 == 0:
        ksize += 1
    prepared = _smooth_for_derivatives(gray, smooth)
    grad_x = cv2.Sobel(prepared, cv2.CV_32F, 1, 0, ksize=ksize)
    grad_y = cv2.Sobel(prepared, cv2.CV_32F, 0, 1, ksize=ksize)
    magnitude = cv2.magnitude(grad_x, grad_y)
    return _normalize_float_map(magnitude)


def scharr_magnitude(image: np.ndarray, *, smooth: bool = True) -> np.ndarray:
    """Scharr gradient magnitude — the numerically optimal 3x3 derivative.

    Produces sharper fine-detail response than a classic Sobel filter.
    """

    gray = _to_gray(image)
    if gray.size == 0:
        return _empty_like_gray(gray)
    prepared = _smooth_for_derivatives(gray, smooth)
    grad_x = cv2.Scharr(prepared, cv2.CV_32F, 1, 0)
    grad_y = cv2.Scharr(prepared, cv2.CV_32F, 0, 1)
    magnitude = cv2.magnitude(grad_x, grad_y)
    return _normalize_float_map(magnitude)


def laplacian_magnitude(image: np.ndarray, ksize: int = 3, *, smooth: bool = True) -> np.ndarray:
    gray = _to_gray(image)
    if gray.size == 0:
        return _empty_like_gray(gray)
    ksize = max(1, int(ksize))
    if ksize % 2 == 0:
        ksize += 1
    prepared = _smooth_for_derivatives(gray, smooth)
    laplacian = cv2.Laplacian(prepared, cv2.CV_32F, ksize=ksize)
    return _normalize_float_map(np.abs(laplacian))


def laplacian_of_gaussian(
    image: np.ndarray,
    sigmas: Iterable[float] = (1.0, 2.0, 3.5),
) -> np.ndarray:
    """Multi-scale Laplacian of Gaussian response (uint8).

    Combines several characteristic scales and returns the per-pixel
    maximum absolute response. Useful for blob-like features (vias).
    """

    gray = _to_gray(image)
    if gray.size == 0:
        return _empty_like_gray(gray)
    gray = gray.astype(np.float32)
    accumulator = np.zeros_like(gray, dtype=np.float32)
    for sigma in sigmas:
        sigma = max(0.4, float(sigma))
        blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
        laplacian = cv2.Laplacian(blurred, cv2.CV_32F, ksize=3)
        local = np.abs(laplacian) * float(sigma * sigma)
        accumulator = np.maximum(accumulator, local)
    return _normalize_float_map(accumulator)


# ---------------------------------------------------------------------------
# Canny with automatic thresholds
# ---------------------------------------------------------------------------


def auto_canny(
    image: np.ndarray,
    sigma: float = 0.33,
    *,
    aperture_size: int = 3,
    l2gradient: bool = True,
) -> np.ndarray:
    """Canny edge detector with median-based automatic thresholds.

    The thresholds follow the widely used ``(1 ± sigma) * median`` rule and
    provide a robust, parameter-free edge mask that adapts to image
    illumination.
    """

    gray = _to_gray(image)
    if gray.size == 0:
        return _empty_like_gray(gray)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    median = float(np.median(blurred))
    sigma = max(0.0, min(1.0, float(sigma)))
    low = int(max(0, (1.0 - sigma) * median))
    high = int(min(255, (1.0 + sigma) * median))
    if high <= low:
        high = max(low + 1, high)
    aperture_size = max(3, min(7, int(aperture_size) | 1))
    return cv2.Canny(blurred, low, high, apertureSize=aperture_size, L2gradient=bool(l2gradient))


# ---------------------------------------------------------------------------
# Structured edge detection (opencv-contrib) with graceful fallback
# ---------------------------------------------------------------------------


_STRUCTURED_EDGE_MODEL_SEARCH_PATHS: tuple[str, ...] = (
    "model.yml.gz",
    "model.yml",
    "resources/models/structured_edges_model.yml.gz",
    "resources/models/structured_edges_model.yml",
)


def _structured_edge_model_path() -> Path | None:
    package_root = Path(__file__).resolve().parent
    for suffix in _STRUCTURED_EDGE_MODEL_SEARCH_PATHS:
        candidate = package_root / suffix
        if candidate.exists():
            return candidate
    for suffix in _STRUCTURED_EDGE_MODEL_SEARCH_PATHS:
        candidate = package_root.parent / suffix
        if candidate.exists():
            return candidate
    return None


def structured_edges(image: np.ndarray) -> np.ndarray:
    """Structured Random Forest edge detection (Dollar & Zitnick).

    Uses ``cv2.ximgproc.createStructuredEdgeDetection`` when available and a
    trained model ships with the project (``resources/models/`` or the
    ``polygon_widget`` package root). When the runtime does not provide
    ``opencv-contrib`` we transparently fall back to a strong classic
    operator (Scharr magnitude with non-maximum thinning) so the pipeline
    keeps working.
    """

    if np.asarray(image).size == 0:
        return _empty_like_gray(image)
    ximgproc = getattr(cv2, "ximgproc", None)
    model_path = _structured_edge_model_path()
    if ximgproc is not None and model_path is not None and hasattr(ximgproc, "createStructuredEdgeDetection"):
        try:
            detector = ximgproc.createStructuredEdgeDetection(str(model_path))
            rgb = cv2.cvtColor(_to_bgr(image), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            edges = detector.detectEdges(rgb)
            return _normalize_float_map(edges)
        except cv2.error:
            pass
    return _scharr_nonmax_thinned(image)


def _scharr_nonmax_thinned(image: np.ndarray) -> np.ndarray:
    """Scharr magnitude with non-maximum suppression along the gradient."""

    gray = _to_gray(image)
    if gray.size == 0:
        return _empty_like_gray(gray)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    grad_x = cv2.Scharr(blurred, cv2.CV_32F, 1, 0)
    grad_y = cv2.Scharr(blurred, cv2.CV_32F, 0, 1)
    magnitude = cv2.magnitude(grad_x, grad_y)
    normalized = _normalize_float_map(magnitude).astype(np.float32)
    if normalized.size == 0:
        return normalized.astype(np.uint8)
    angle = np.degrees(np.arctan2(grad_y, grad_x))
    angle[angle < 0] += 180
    thinned = np.zeros_like(normalized)

    shifted_left = np.roll(normalized, -1, axis=1)
    shifted_right = np.roll(normalized, 1, axis=1)
    shifted_up = np.roll(normalized, -1, axis=0)
    shifted_down = np.roll(normalized, 1, axis=0)
    shifted_ul = np.roll(shifted_up, -1, axis=1)
    shifted_ur = np.roll(shifted_up, 1, axis=1)
    shifted_dl = np.roll(shifted_down, -1, axis=1)
    shifted_dr = np.roll(shifted_down, 1, axis=1)

    horizontal = ((angle >= 0) & (angle < 22.5)) | (angle >= 157.5)
    diagonal_up = (angle >= 22.5) & (angle < 67.5)
    vertical = (angle >= 67.5) & (angle < 112.5)
    diagonal_down = (angle >= 112.5) & (angle < 157.5)

    is_max_horizontal = (normalized >= shifted_left) & (normalized >= shifted_right)
    is_max_vertical = (normalized >= shifted_up) & (normalized >= shifted_down)
    is_max_diag_up = (normalized >= shifted_ur) & (normalized >= shifted_dl)
    is_max_diag_down = (normalized >= shifted_ul) & (normalized >= shifted_dr)

    thinned[horizontal & is_max_horizontal] = normalized[horizontal & is_max_horizontal]
    thinned[vertical & is_max_vertical] = normalized[vertical & is_max_vertical]
    thinned[diagonal_up & is_max_diag_up] = normalized[diagonal_up & is_max_diag_up]
    thinned[diagonal_down & is_max_diag_down] = normalized[diagonal_down & is_max_diag_down]

    thinned[0, :] = 0
    thinned[-1, :] = 0
    thinned[:, 0] = 0
    thinned[:, -1] = 0
    return np.clip(thinned, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Ridge detection (Hessian-based)
# ---------------------------------------------------------------------------


def ridge_response(image: np.ndarray, *, sigma: float = 1.5) -> np.ndarray:
    """Highlight ridge / crest structures using Hessian eigenvalues.

    Tries to use ``cv2.ximgproc.RidgeDetectionFilter`` first and falls back
    to an explicit Hessian eigenvalue computation implemented with NumPy.
    """

    ximgproc = getattr(cv2, "ximgproc", None)
    gray = _to_gray(image)
    if gray.size == 0:
        return _empty_like_gray(gray)
    if ximgproc is not None and hasattr(ximgproc, "RidgeDetectionFilter_create"):
        try:
            detector = ximgproc.RidgeDetectionFilter_create()
            response = detector.getRidgeFilteredImage(gray)
            return _normalize_float_map(np.abs(response.astype(np.float32)))
        except cv2.error:
            pass
    sigma = max(0.5, float(sigma))
    blurred = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), sigma)
    dxx = cv2.Sobel(blurred, cv2.CV_32F, 2, 0, ksize=3)
    dyy = cv2.Sobel(blurred, cv2.CV_32F, 0, 2, ksize=3)
    dxy = cv2.Sobel(blurred, cv2.CV_32F, 1, 1, ksize=3)
    trace = dxx + dyy
    delta = np.maximum(0.0, (dxx - dyy) ** 2 + 4.0 * dxy * dxy)
    sqrt_delta = np.sqrt(delta)
    eigen_max = 0.5 * (trace + sqrt_delta)
    eigen_min = 0.5 * (trace - sqrt_delta)
    ridge = np.maximum(np.abs(eigen_max), np.abs(eigen_min))
    return _normalize_float_map(ridge)


# ---------------------------------------------------------------------------
# Phase congruency (contrast-invariant edge feature)
# ---------------------------------------------------------------------------


def phase_congruency(
    image: np.ndarray,
    *,
    num_scales: int = 4,
    num_orientations: int = 6,
    min_wavelength: float = 3.0,
    scale_factor: float = 2.1,
) -> np.ndarray:
    """Peter Kovesi style phase congruency edge feature.

    Phase congruency is a dimensionless edge measure that is invariant to
    image contrast and illumination changes. It is routinely used in
    SOTA pipelines (including learnt edge detectors as supervision targets).

    Implementation notes: pure NumPy/OpenCV, using log-Gabor filters in the
    frequency domain. Designed to work on reasonably sized images (we
    internally downscale huge inputs to keep the FFT cheap).
    """

    gray = _to_gray(image)
    if gray.size == 0:
        return _empty_like_gray(gray)
    gray = gray.astype(np.float32)

    height, width = gray.shape
    work = gray
    scale_to_input = 1.0
    max_side = max(height, width)
    if max_side > 768:
        scale_to_input = 768.0 / float(max_side)
        work = cv2.resize(gray, (int(round(width * scale_to_input)), int(round(height * scale_to_input))), interpolation=cv2.INTER_AREA)
    work_height, work_width = work.shape

    epsilon = 1e-4
    mean_value = float(np.mean(work))
    work = work - mean_value

    fft = np.fft.fftshift(np.fft.fft2(work))
    yy, xx = np.meshgrid(
        np.linspace(-0.5, 0.5, work_height, dtype=np.float32),
        np.linspace(-0.5, 0.5, work_width, dtype=np.float32),
        indexing="ij",
    )
    radius = np.sqrt(xx * xx + yy * yy)
    radius[work_height // 2, work_width // 2] = 1.0
    theta = np.arctan2(-yy, xx)

    low_pass = 1.0 / (1.0 + (radius / 0.45) ** (2 * 15))

    num_scales = max(1, int(num_scales))
    num_orientations = max(1, int(num_orientations))
    sigma_on_f = 0.55

    energy_total = np.zeros_like(work)
    amplitude_total = np.zeros_like(work)

    for orientation_index in range(num_orientations):
        angle = orientation_index * np.pi / float(num_orientations)
        ds = np.sin(theta) * np.cos(angle) - np.cos(theta) * np.sin(angle)
        dc = np.cos(theta) * np.cos(angle) + np.sin(theta) * np.sin(angle)
        angular_distance = np.abs(np.arctan2(ds, dc))
        spread = np.pi / float(num_orientations) * 1.6
        spread = max(0.1, spread)
        angular_filter = np.exp(-(angular_distance ** 2) / (2.0 * spread * spread))

        even_sum = np.zeros_like(work)
        odd_sum = np.zeros_like(work)
        amplitude_sum = np.zeros_like(work)
        max_amplitude = np.zeros_like(work)

        for scale_index in range(num_scales):
            wavelength = min_wavelength * (scale_factor ** scale_index)
            f_zero = 1.0 / wavelength
            log_gabor = np.exp(-((np.log(radius / f_zero)) ** 2) / (2.0 * (np.log(sigma_on_f)) ** 2))
            log_gabor[work_height // 2, work_width // 2] = 0.0
            log_gabor = log_gabor * low_pass * angular_filter

            filtered = np.fft.ifft2(np.fft.ifftshift(fft * log_gabor))
            even = filtered.real.astype(np.float32)
            odd = filtered.imag.astype(np.float32)
            amplitude = np.sqrt(even * even + odd * odd)

            even_sum += even
            odd_sum += odd
            amplitude_sum += amplitude
            max_amplitude = np.maximum(max_amplitude, amplitude)

        orientation_energy = np.sqrt(even_sum * even_sum + odd_sum * odd_sum)
        noise_estimate = float(np.median(amplitude_sum)) / float(-np.log(0.5) or 1.0)
        noise_term = noise_estimate * np.sqrt(float(num_scales)) * 2.0
        orientation_energy = np.maximum(orientation_energy - noise_term, 0.0)

        energy_total += orientation_energy
        amplitude_total += amplitude_sum

    phase_map = energy_total / (amplitude_total + epsilon)
    phase_map = np.clip(phase_map, 0.0, 1.0)

    if scale_to_input != 1.0:
        phase_map = cv2.resize(phase_map, (width, height), interpolation=cv2.INTER_LINEAR)

    return (phase_map * 255.0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Combined / ensemble
# ---------------------------------------------------------------------------


_DEFAULT_COMBINED_METHODS: tuple[str, ...] = (
    EDGE_METHOD_SCHARR,
    EDGE_METHOD_LOG,
    EDGE_METHOD_PHASE_CONGRUENCY,
)


def combined_elevation(image: np.ndarray, methods: Iterable[str] | None = None) -> np.ndarray:
    """Pixel-wise maximum of several elevation maps.

    This ensemble consistently outperforms individual operators in the
    presence of low contrast / illumination gradients and is a cheap,
    training-free stand-in for learnt edge detectors.
    """

    if np.asarray(image).size == 0:
        return _empty_like_gray(image)
    method_list = [normalize_edge_method(name) for name in (methods or _DEFAULT_COMBINED_METHODS)]
    method_list = [name for name in method_list if name != EDGE_METHOD_COMBINED]
    if not method_list:
        method_list = list(_DEFAULT_COMBINED_METHODS)
    accumulator: np.ndarray | None = None
    for method in method_list:
        elevation = build_gradient_elevation(image, method)
        accumulator = elevation if accumulator is None else np.maximum(accumulator, elevation)
    if accumulator is None:
        return np.zeros(image.shape[:2], dtype=np.uint8)
    return accumulator


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def build_gradient_elevation(image: np.ndarray, method: str = EDGE_METHOD_SOBEL) -> np.ndarray:
    """Return a uint8 elevation map for ``image`` using ``method``.

    Unknown methods gracefully fall back to ``sobel``.
    """

    method = normalize_edge_method(method)
    if method == EDGE_METHOD_SOBEL:
        return sobel_magnitude(image)
    if method == EDGE_METHOD_SCHARR:
        return scharr_magnitude(image)
    if method == EDGE_METHOD_LAPLACIAN:
        return laplacian_magnitude(image)
    if method == EDGE_METHOD_LOG:
        return laplacian_of_gaussian(image)
    if method == EDGE_METHOD_AUTO_CANNY:
        return auto_canny(image)
    if method == EDGE_METHOD_STRUCTURED:
        return structured_edges(image)
    if method == EDGE_METHOD_RIDGE:
        return ridge_response(image)
    if method == EDGE_METHOD_PHASE_CONGRUENCY:
        return phase_congruency(image)
    if method == EDGE_METHOD_COMBINED:
        return combined_elevation(image)
    return sobel_magnitude(image)


def gradient_color_map(
    image: np.ndarray,
    method: str = EDGE_METHOD_SOBEL,
    *,
    colormap: int = cv2.COLORMAP_TURBO,
) -> np.ndarray:
    """Return a coloured heatmap of ``image`` produced by ``method``.

    Handy for debug overlays in the UI.
    """

    elevation = build_gradient_elevation(image, method)
    return cv2.applyColorMap(elevation, colormap)
