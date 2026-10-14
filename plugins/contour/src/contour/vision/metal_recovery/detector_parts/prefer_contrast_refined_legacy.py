"""detector: prefer contrast refined legacy."""
from __future__ import annotations
import cv2
import numpy as np
from contour.utils import ensure_binary_mask
from contour.vision.metal_recovery.gradient_watershed import (
    build_conductor_seeds,
    gradient_watershed_config_from_object,
)

def _non_hole_count(result: MetalDetectionResult) -> int:
    return sum(not polygon.is_hole for polygon in result.accepted)

def _mask_iou(first: np.ndarray | None, second: np.ndarray | None) -> float:
    if first is None or second is None or first.shape[:2] != second.shape[:2]:
        return 0.0
    first_active = ensure_binary_mask(first) > 0
    second_active = ensure_binary_mask(second) > 0
    union = int(np.logical_or(first_active, second_active).sum())
    if union <= 0:
        return 0.0
    return float(np.logical_and(first_active, second_active).sum() / union)

def _prefer_contrast_refined_legacy(
    *,
    legacy: MetalDetectionResult,
    refined: MetalDetectionResult,
    watershed: MetalDetectionResult,
) -> bool:
    """Accept a stricter Otsu mask only when it removes stable low-contrast bridges."""
    legacy_count = _non_hole_count(legacy)
    refined_count = _non_hole_count(refined)
    watershed_count = _non_hole_count(watershed)
    if legacy_count <= 0 or watershed_count < 1.15 * legacy_count:
        return False
    if refined_count < legacy_count or refined_count > 1.05 * legacy_count:
        return False

    watershed_mask = watershed.debug_images.get("metal_filtered_mask")
    legacy_agreement = _mask_iou(
        legacy.debug_images.get("metal_filtered_mask"),
        watershed_mask,
    )
    refined_agreement = _mask_iou(
        refined.debug_images.get("metal_filtered_mask"),
        watershed_mask,
    )
    return refined_agreement > legacy_agreement

def _extended_separator_mask(gray: np.ndarray, config: MetalRecoveryConfig) -> np.ndarray | None:
    watershed = gradient_watershed_config_from_object(config)
    seeds = build_conductor_seeds(gray, watershed)
    if seeds is None or not np.any(seeds.groove_seeds):
        return None
    extension = max(5, 4 * int(watershed.valley_span_px) - 3) | 1
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (extension, 3))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, extension))
    horizontal_lines = cv2.morphologyEx(
        seeds.groove_seeds,
        cv2.MORPH_OPEN,
        horizontal_kernel,
    )
    vertical_lines = cv2.morphologyEx(
        seeds.groove_seeds,
        cv2.MORPH_OPEN,
        vertical_kernel,
    )
    kernel = (
        horizontal_kernel
        if cv2.countNonZero(horizontal_lines) >= cv2.countNonZero(vertical_lines)
        else vertical_kernel
    )
    return cv2.morphologyEx(seeds.groove_seeds, cv2.MORPH_CLOSE, kernel)
