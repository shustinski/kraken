"""Otsu-based metal segmentation and morphology stages for conductor recovery."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from ...utils import ensure_binary_mask, ensure_uint8
from ..schemas import SemPolarity


def normalize_metal_segmentation_strategy(value: Any) -> str:
    """All strategies normalize to Otsu (legacy UI values are migrated here)."""
    _ = value
    return "legacy_otsu"


def contrast_bias_to_otsu_offset(contrast_bias: float) -> float:
    """Map UI contrast bias (-50..+50) to Otsu threshold offset."""
    bias = max(-50.0, min(50.0, float(contrast_bias)))
    t = (bias + 50.0) / 100.0
    return float((t - 0.5) * 18.0)


def contrast_bias_to_threshold_params(contrast_bias: float) -> dict[str, float]:
    """Backward-compatible helper; only ``otsu_offset`` is used by the pipeline."""
    offset = contrast_bias_to_otsu_offset(contrast_bias)
    return {"c_adaptive": 0.0, "sauvola_k": 0.0, "otsu_offset": offset}


def migrate_legacy_metal_settings(payload: dict[str, Any]) -> dict[str, Any]:
    """Convert deprecated sensitivity/segmentation fields to the Otsu pipeline."""
    out = dict(payload)
    if "metal_contrast_bias" not in payload and "metal_sensitivity_0_100" in payload:
        sens = max(0, min(100, int(payload.get("metal_sensitivity_0_100", 50))))
        tok = str(payload.get("metal_sensitivity", "medium") or "medium").lower()
        mid = {"low": 35, "medium": 50, "high": 65}.get(tok, 50)
        blend = 0.35 * mid + 0.65 * sens
        out["metal_contrast_bias"] = int(round((blend - 50.0) * 0.6))
    if "metal_gap_bridge_px" not in payload and "metal_morph_close_radius" in payload:
        out["metal_gap_bridge_px"] = int(payload.get("metal_morph_close_radius", 2) or 2)
    if "metal_speckle_removal_px" not in payload and "metal_morph_open_radius" in payload:
        out["metal_speckle_removal_px"] = int(payload.get("metal_morph_open_radius", 0) or 0)
    out["metal_segmentation_strategy"] = "legacy_otsu"
    if "metal_hierarchy_mode" not in payload:
        out["metal_hierarchy_mode"] = "full"
    return out


@dataclass(slots=True)
class MetalSegmentationConfig:
    contrast_bias: float = 0.0
    gap_bridge_px: int = 2
    speckle_removal_px: int = 0
    min_component_area: int = 20
    max_hole_fill_area: int = 200
    # Legacy fields kept for deserialization compatibility; ignored at runtime.
    noise_suppression: int = 0
    segmentation_strategy: str = "legacy_otsu"
    min_width_px: float = 8.0


@dataclass(slots=True)
class MetalSegmentationResult:
    mask: np.ndarray
    preprocessed: np.ndarray
    raw_segmentation: np.ndarray
    after_topology: np.ndarray
    strategy: str
    polarity: SemPolarity
    debug_images: dict[str, np.ndarray] = field(default_factory=dict)


def otsu_segmentation_mask(gray: np.ndarray, *, otsu_offset: float, dark_foreground: bool) -> np.ndarray:
    mode = cv2.THRESH_BINARY_INV if dark_foreground else cv2.THRESH_BINARY
    otsu_t, mask = cv2.threshold(gray, 0, 255, mode + cv2.THRESH_OTSU)
    if abs(otsu_offset) > 0.05:
        t = int(max(1, min(254, round(float(otsu_t) + otsu_offset))))
        _, mask = cv2.threshold(gray, t, 255, mode)
    return ensure_binary_mask(mask)


def _fill_small_holes(mask: np.ndarray, *, max_area: int) -> np.ndarray:
    m = (mask > 0).astype(np.uint8) * 255
    if cv2.countNonZero(m) == 0 or cv2.countNonZero(m) == m.size:
        return m
    inv = cv2.bitwise_not(m)
    h, s = m.shape
    border = np.zeros((h + 2, s + 2), dtype=np.uint8)
    inv_copy = inv.copy()
    cv2.floodFill(inv_copy, border, (0, 0), 255)
    holes = cv2.subtract(inv, inv_copy)
    if cv2.countNonZero(holes) == 0:
        return m
    n, labels, stats, _ = cv2.connectedComponentsWithStats(holes, connectivity=8)
    for i in range(1, n):
        if int(stats[i, cv2.CC_STAT_AREA]) <= max_area:
            holes[labels == i] = 0
    return cv2.subtract(m, holes)


def apply_topology_repair(raw_mask: np.ndarray, config: MetalSegmentationConfig) -> np.ndarray:
    """Gap bridge (close) → fill small holes → speckle removal (open)."""
    m = ensure_uint8(raw_mask)
    close_r = max(0, int(config.gap_bridge_px))
    if close_r > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_r + 1, 2 * close_r + 1))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)
    m = _fill_small_holes(m, max_area=int(config.max_hole_fill_area))
    open_r = max(0, int(config.speckle_removal_px))
    if open_r > 0:
        ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * open_r + 1, 2 * open_r + 1))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ko, iterations=1)
    return ensure_binary_mask(m)


def filter_mask_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    m = (ensure_uint8(mask) > 0).astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = np.zeros_like(m)
    for i in range(1, n):
        if int(stats[i, cv2.CC_STAT_AREA]) >= min_area:
            out[labels == i] = 255
    return out


def build_metal_segmentation_mask(gray: np.ndarray, config: MetalSegmentationConfig) -> MetalSegmentationResult:
    """Non-cached entry point (tests/batch); interactive preview uses ``pipeline_stages``."""
    from .pipeline_stages import build_metal_segmentation_mask_staged

    return build_metal_segmentation_mask_staged(gray, config)
