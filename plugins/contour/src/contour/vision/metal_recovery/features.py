"""Cached structural evidence shared by the new segmentation strategies."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from time import perf_counter

import cv2
import numpy as np

from ...utils import ensure_uint8
from .gradient_watershed import GradientWatershedConfig, build_conductor_seeds
from .structural_watershed import (
    _extract_structural_features,
    clamped_structural_watershed_config,
)


@dataclass(frozen=True, slots=True)
class MetalStructuralFeatures:
    gray: np.ndarray
    denoised: np.ndarray
    gradient_x: np.ndarray
    gradient_y: np.ndarray
    gradient_magnitude: np.ndarray
    boundary_strength: np.ndarray
    orientation: np.ndarray
    orientation_coherence: np.ndarray
    orientation_persistence: np.ndarray
    local_contrast: np.ndarray
    core_evidence: np.ndarray
    substrate_evidence: np.ndarray
    rim_evidence: np.ndarray
    build_time_ms: float


_FEATURE_CACHE: dict[tuple[str, float, float], MetalStructuralFeatures] = {}
_FEATURE_CACHE_ITEMS = 3


def _signature(gray: np.ndarray) -> str:
    digest = hashlib.sha1(np.ascontiguousarray(gray).tobytes()).hexdigest()
    return f"{gray.shape[0]}x{gray.shape[1]}:{digest}"


def _normalized(values: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    source = values.astype(np.float32, copy=False)
    scale = float(np.percentile(source, percentile)) if source.size else 0.0
    if scale <= 1e-6:
        return np.zeros(source.shape, dtype=np.float32)
    return np.clip(source / scale, 0.0, 1.0).astype(np.float32)


def clear_metal_feature_cache() -> None:
    _FEATURE_CACHE.clear()


def build_metal_structural_features(
    gray: np.ndarray,
    *,
    smoothing_sigma: float = 1.0,
    orientation_smoothing_sigma: float = 2.0,
) -> MetalStructuralFeatures:
    source = ensure_uint8(gray)
    smooth = max(0.1, min(8.0, float(smoothing_sigma)))
    orientation_smooth = max(0.1, min(16.0, float(orientation_smoothing_sigma)))
    key = (_signature(source), smooth, orientation_smooth)
    cached = _FEATURE_CACHE.get(key)
    if cached is not None:
        return cached

    started = perf_counter()
    structural_config = clamped_structural_watershed_config(
        smoothing_sigma=smooth,
        orientation_smoothing_scale=orientation_smooth,
    )
    structural = _extract_structural_features(source, structural_config)
    magnitude = _normalized(structural.magnitude)
    oriented = _normalized(structural.oriented_gradient)
    persistent = _normalized(structural.persistent_edge)
    rim = _normalized(structural.rim_response)
    boundary = np.clip(
        0.35 * magnitude + 0.35 * oriented + 0.2 * persistent + 0.1 * rim,
        0.0,
        1.0,
    ).astype(np.float32)

    background = cv2.GaussianBlur(structural.denoised, (0, 0), max(3.0, 6.0 * smooth))
    local_contrast = _normalized(cv2.absdiff(structural.denoised, background))
    seeds = build_conductor_seeds(
        source,
        GradientWatershedConfig(smoothing_sigma=smooth),
        check_presence=False,
    )
    core: np.ndarray
    substrate: np.ndarray
    if seeds is None:
        core = np.zeros(source.shape, dtype=np.float32)
        substrate = np.zeros(source.shape, dtype=np.float32)
    else:
        core = cv2.GaussianBlur((seeds.core_seeds > 0).astype(np.float32), (0, 0), 1.5)
        substrate = cv2.GaussianBlur((seeds.groove_seeds > 0).astype(np.float32), (0, 0), 1.5)
        core = np.clip(core, 0.0, 1.0)
        substrate = np.clip(substrate, 0.0, 1.0)
    features = MetalStructuralFeatures(
        gray=source,
        denoised=structural.denoised,
        gradient_x=structural.gradient_x,
        gradient_y=structural.gradient_y,
        gradient_magnitude=magnitude,
        boundary_strength=boundary,
        orientation=structural.structure_orientation,
        orientation_coherence=structural.coherence.astype(np.float32, copy=False),
        orientation_persistence=np.clip(persistent * structural.coherence, 0.0, 1.0).astype(np.float32),
        local_contrast=local_contrast,
        core_evidence=core.astype(np.float32),
        substrate_evidence=substrate.astype(np.float32),
        rim_evidence=rim,
        build_time_ms=(perf_counter() - started) * 1000.0,
    )
    while len(_FEATURE_CACHE) >= _FEATURE_CACHE_ITEMS:
        _FEATURE_CACHE.pop(next(iter(_FEATURE_CACHE)))
    _FEATURE_CACHE[key] = features
    return features


def float_map_to_u8(values: np.ndarray) -> np.ndarray:
    return np.clip(values.astype(np.float32, copy=False) * 255.0, 0.0, 255.0).astype(np.uint8)


__all__ = [
    "MetalStructuralFeatures",
    "build_metal_structural_features",
    "clear_metal_feature_cache",
    "float_map_to_u8",
]
