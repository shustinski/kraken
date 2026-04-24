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
    """User-facing only via presets; these fields are set internally."""

    clahe_clip: float = 2.0
    clahe_grid: int = 8
    subtract_background: bool = True
    background_sigma_fraction: float = 0.04
    denoise: NoiseLevel = NoiseLevel.MEDIUM


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
