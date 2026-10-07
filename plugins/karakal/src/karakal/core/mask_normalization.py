"""One way to turn any network or operator mask image into a binary mask.

The method depends on what the image is, and the result says which method and
threshold were used, so a manifest can record them:

- exactly two gray values (a true binary PNG): no threshold, the brighter value is foreground;
- a black-and-white mask saved as JPEG (values near 0 and 255, a thin gray fringe): fixed 128;
- a soft or probability mask: the configured threshold, else Otsu when the histogram
  really has two modes, else the fixed 128.

``mask > 0`` is never used: on a JPEG mask it turns compression noise into hundreds of
specks. Binarization never cleans the mask (no morphology, no small-component removal);
cleaning is a separate, optional step. Inversion happens only when asked for.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - OpenCV is a hard dependency of the app
    cv2 = None

MASK_NORMALIZATION_VERSION = "mask_normalization.v1"

KIND_BINARY = "binary"
KIND_BINARY_LIKE = "binary_like"
KIND_SOFT = "soft"

METHOD_EXACT = "exact"
METHOD_FIXED = "fixed"
METHOD_CONFIG = "config"
METHOD_OTSU = "otsu"
METHOD_FIXED_FALLBACK = "fixed_fallback"

STATUS_OK = "ok"
STATUS_EMPTY = "empty"
STATUS_NEAR_EMPTY = "near_empty"
STATUS_FULL = "full"

FIXED_THRESHOLD = 128
# Gray levels below this are background-dark; a mask's foreground sits near 255.
_DARK_LEVEL = 48
_BRIGHT_LEVEL = 208
# A JPEG of a black-and-white mask: at most this share of its non-dark pixels is mid-gray
# (measured 1-7% on real JPEG masks; grainy soft network outputs reach 30%).
BINARY_LIKE_MAX_GRAY_SHARE = 0.15
# Otsu's separability (between-class / total variance) needed to trust its threshold.
OTSU_MIN_SEPARABILITY = 0.80
# Foreground below this share of the frame is reported as near empty (still kept).
NEAR_EMPTY_FRACTION = 1.0e-5
FULL_FRACTION = 0.999


@dataclass(frozen=True, slots=True)
class MaskNormalizationConfig:
    # Explicit threshold from metadata or settings, gray levels 0..255 or a fraction 0..1. Wins over the rules.
    threshold: float | None = None
    # Foreground is dark on light. Never guessed.
    invert: bool = False

    def threshold_level(self) -> float | None:
        if self.threshold is None:
            return None
        value = float(self.threshold)
        if 0.0 < value <= 1.0:
            value *= 255.0
        return float(np.clip(value, 0.0, 255.0))


@dataclass(frozen=True, slots=True)
class NormalizedMask:
    binary: np.ndarray
    kind: str
    method: str
    # Gray level used; a pixel at or above it is foreground (exact masks: the midpoint of the two values).
    threshold: float
    inverted: bool
    status: str
    foreground_fraction: float

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.binary.shape[0]), int(self.binary.shape[1]))

    def as_uint8(self) -> np.ndarray:
        return np.where(self.binary, np.uint8(255), np.uint8(0))

    def metadata(self) -> dict[str, Any]:
        return {
            "version": MASK_NORMALIZATION_VERSION,
            "kind": self.kind,
            "method": self.method,
            "threshold": round(float(self.threshold), 3),
            "inverted": bool(self.inverted),
            "status": self.status,
            "foreground_fraction": round(float(self.foreground_fraction), 8),
        }


def grayscale_levels(image: np.ndarray) -> np.ndarray:
    """uint8 gray levels of any mask array: color is averaged, a 0..1 float map is scaled to 0..255."""

    values = np.asarray(image)
    if values.ndim == 3:
        values = values[..., :3].mean(axis=2) if values.shape[2] >= 3 else values[..., 0]
    if values.ndim != 2:
        raise ValueError(f"mask must be a 2D image, got shape {np.asarray(image).shape}")
    if values.dtype == np.uint8:
        return np.ascontiguousarray(values)
    if values.dtype == np.bool_:
        return np.where(values, np.uint8(255), np.uint8(0))
    floats = np.nan_to_num(values.astype(np.float64, copy=False), nan=0.0, posinf=255.0, neginf=0.0)
    if floats.size and float(floats.max()) <= 1.0:
        floats = floats * 255.0
    return np.clip(np.rint(floats), 0.0, 255.0).astype(np.uint8)


def _otsu(gray: np.ndarray, histogram: np.ndarray) -> tuple[float, float]:
    """(threshold, separability). Separability is 0 when the image has one level."""

    total = float(histogram.sum())
    levels = np.arange(256, dtype=np.float64)
    probability = histogram / max(total, 1.0)
    mean = float((probability * levels).sum())
    variance = float((probability * (levels - mean) ** 2).sum())
    if variance <= 1.0e-12:
        return float(FIXED_THRESHOLD), 0.0
    if cv2 is not None:
        level, _ignored = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        level = int(level)
    else:  # pragma: no cover - OpenCV is a hard dependency of the app
        weights = np.cumsum(probability)
        means = np.cumsum(probability * levels)
        between = (mean * weights - means) ** 2 / np.maximum(weights * (1.0 - weights), 1.0e-12)
        level = int(np.argmax(between))
    low = probability[: level + 1]
    high = probability[level + 1 :]
    weight_low, weight_high = float(low.sum()), float(high.sum())
    if weight_low <= 0.0 or weight_high <= 0.0:
        return float(level + 1), 0.0
    mean_low = float((low * levels[: level + 1]).sum()) / weight_low
    mean_high = float((high * levels[level + 1 :]).sum()) / weight_high
    separability = weight_low * weight_high * (mean_high - mean_low) ** 2 / variance
    # cv2 marks values above the level; the mask keeps values at or above the returned threshold.
    return float(level + 1), float(separability)


def _status(foreground: int, total: int) -> str:
    if foreground <= 0:
        return STATUS_EMPTY
    fraction = float(foreground) / float(max(1, total))
    if fraction >= FULL_FRACTION:
        return STATUS_FULL
    if fraction < NEAR_EMPTY_FRACTION:
        return STATUS_NEAR_EMPTY
    return STATUS_OK


def normalize_mask(image: np.ndarray, config: MaskNormalizationConfig | None = None) -> NormalizedMask:
    """Binary mask of one image, with the kind, method and threshold that produced it."""

    cfg = config or MaskNormalizationConfig()
    gray = grayscale_levels(image)
    histogram = np.bincount(gray.reshape(-1), minlength=256).astype(np.float64)
    present = np.flatnonzero(histogram)
    explicit = cfg.threshold_level()

    if present.size <= 2:
        kind = KIND_BINARY
    else:
        not_dark = float(histogram[_DARK_LEVEL:].sum())
        gray_share = float(histogram[_DARK_LEVEL:_BRIGHT_LEVEL].sum()) / not_dark if not_dark > 0.0 else 1.0
        kind = KIND_BINARY_LIKE if gray_share <= BINARY_LIKE_MAX_GRAY_SHARE else KIND_SOFT

    if explicit is not None:
        method, threshold = METHOD_CONFIG, float(explicit)
    elif kind == KIND_BINARY:
        if present.size == 2:
            method, threshold = METHOD_EXACT, (float(present[0]) + float(present[1])) / 2.0
        else:
            # One gray level: a black frame is empty, a white one is full.
            method, threshold = METHOD_EXACT, float(FIXED_THRESHOLD)
    elif kind == KIND_BINARY_LIKE or not histogram[FIXED_THRESHOLD:].any():
        # No pixel reaches mid-gray: the frame is empty whatever Otsu would split in its noise.
        method, threshold = METHOD_FIXED, float(FIXED_THRESHOLD)
    else:
        level, separability = _otsu(gray, histogram)
        if separability >= OTSU_MIN_SEPARABILITY:
            method, threshold = METHOD_OTSU, level
        else:
            method, threshold = METHOD_FIXED_FALLBACK, float(FIXED_THRESHOLD)

    binary = gray >= threshold if threshold > 0.0 else np.ones(gray.shape, dtype=bool)
    if cfg.invert:
        binary = ~binary
    foreground = int(np.count_nonzero(binary))
    return NormalizedMask(
        binary=np.ascontiguousarray(binary),
        kind=kind,
        method=method,
        threshold=float(threshold),
        inverted=bool(cfg.invert),
        status=_status(foreground, int(binary.size)),
        foreground_fraction=float(foreground) / float(max(1, binary.size)),
    )


def binarize_mask(image: np.ndarray, config: MaskNormalizationConfig | None = None) -> np.ndarray:
    """Only the boolean mask of :func:`normalize_mask`."""

    return normalize_mask(image, config).binary
