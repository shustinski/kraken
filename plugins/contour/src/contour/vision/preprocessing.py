"""Preprocessing presets for SEM: illumination flattening, mild denoising, CLAHE."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import cv2
import numpy as np

from . import io_normalize
from .schemas import SemPolarity

try:
    from ..utils import ensure_uint8
except ImportError:  # pragma: no cover

    def ensure_uint8(image: Any) -> np.ndarray:  # type: ignore[misc]
        return io_normalize.ensure_uint8_local(image)


class NoiseLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True, slots=True)
class PreprocessConfig:
    """SEM preprocessing parameters shared by presets and typed UI settings."""

    clahe_clip: float = 2.0
    clahe_grid: int = 8
    subtract_background: bool = True
    background_sigma_fraction: float = 0.04
    denoise: NoiseLevel = NoiseLevel.MEDIUM


def metal_preprocess_config_from_settings(settings: Any) -> PreprocessConfig:
    """Build the conductor-recognition preprocessing used by the real benchmark."""

    raw_noise = str(getattr(settings, "metal_preprocess_denoise", "low") or "low")
    noise = NoiseLevel(raw_noise) if raw_noise in {level.value for level in NoiseLevel} else NoiseLevel.LOW
    return PreprocessConfig(
        clahe_clip=max(0.1, min(20.0, float(getattr(settings, "metal_preprocess_clahe_clip", 2.0) or 2.0))),
        clahe_grid=max(2, min(64, int(getattr(settings, "metal_preprocess_clahe_grid", 8) or 8))),
        subtract_background=bool(getattr(settings, "metal_preprocess_subtract_background", True)),
        background_sigma_fraction=max(
            0.005,
            min(
                0.25,
                float(getattr(settings, "metal_preprocess_background_sigma_fraction", 0.05) or 0.05),
            ),
        ),
        denoise=noise,
    )


def _auto_odd_kernel(gray: np.ndarray, sigma_fraction: float) -> int:
    side = round(min(gray.shape[0], gray.shape[1]) * float(sigma_fraction))
    side = max(15, min(side, 201))
    if side % 2 == 0:
        side += 1
    return side


def flatten_illumination(gray: np.ndarray, sigma_fraction: float) -> np.ndarray:
    g = ensure_uint8(gray)
    k = _auto_odd_kernel(g, sigma_fraction)
    bg = cv2.GaussianBlur(g, (k, k), 0)
    bg = np.clip(bg.astype(np.float32), 1.0, 255.0)
    out = np.clip(g.astype(np.float32) * (128.0 / bg), 0, 255).astype(np.uint8)
    return out


def apply_clahe(gray: np.ndarray, *, clip: float, grid: int) -> np.ndarray:
    g = ensure_uint8(gray)
    tile = max(2, int(grid))
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(tile, tile))
    return clahe.apply(g)


def denoise(gray: np.ndarray, level: NoiseLevel) -> np.ndarray:
    g = ensure_uint8(gray)
    if level == NoiseLevel.LOW:
        return cv2.GaussianBlur(g, (3, 3), 0)
    if level == NoiseLevel.MEDIUM:
        if g.shape[0] >= 8 and g.shape[1] >= 8:
            return cv2.fastNlMeansDenoising(g, h=6, templateWindowSize=7, searchWindowSize=15)
        return cv2.GaussianBlur(g, (3, 3), 0)
    if g.shape[0] >= 8 and g.shape[1] >= 8:
        return cv2.fastNlMeansDenoising(g, h=9, templateWindowSize=7, searchWindowSize=21)
    return cv2.GaussianBlur(g, (5, 5), 0)


def preprocess_for_sem(gray: np.ndarray, config: PreprocessConfig) -> np.ndarray:
    g = ensure_uint8(gray)
    if config.subtract_background:
        g = flatten_illumination(g, config.background_sigma_fraction)
    g = apply_clahe(g, clip=config.clahe_clip, grid=config.clahe_grid)
    g = denoise(g, config.denoise)
    return g


def guess_polarity(gray: np.ndarray) -> SemPolarity:
    """Heuristic: compare tails of the histogram; SEM conductors often differ by dataset."""

    g = ensure_uint8(gray).astype(np.float32)
    p95 = float(np.percentile(g, 95))
    p5 = float(np.percentile(g, 5))
    if p95 - p5 < 12:
        return SemPolarity.DARK_FOREGROUND
    m = float(g.mean())
    if m > 135:
        return SemPolarity.DARK_FOREGROUND
    if m < 120:
        return SemPolarity.BRIGHT_FOREGROUND
    return SemPolarity.DARK_FOREGROUND
