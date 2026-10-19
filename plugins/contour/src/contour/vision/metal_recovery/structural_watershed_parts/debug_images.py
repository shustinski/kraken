"""structural_watershed: debug images."""
from __future__ import annotations
from typing import TYPE_CHECKING
import cv2
import numpy as np
from contour.utils import ensure_binary_mask, ensure_uint8

from contour.vision.metal_recovery.structural_watershed_parts.geodesic_label_competition import (
    _label_boundary_map,
    _to_u8,
)

def _orientation_overlay(orientation: np.ndarray, coherence: np.ndarray) -> np.ndarray:
    hsv = np.empty((*orientation.shape, 3), dtype=np.uint8)
    angle = np.mod(orientation, np.pi)
    hsv[:, :, 0] = np.clip(angle * (179.0 / np.pi), 0, 179).astype(np.uint8)
    hsv[:, :, 1] = 255
    hsv[:, :, 2] = np.clip(coherence * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def _label_overlay(labels: np.ndarray) -> np.ndarray:
    color = np.zeros((*labels.shape, 3), dtype=np.uint8)
    positive = labels > 0
    if not np.any(positive):
        return color
    hue = ((labels.astype(np.int32) * 37) % 180).astype(np.uint8)
    hsv = np.zeros((*labels.shape, 3), dtype=np.uint8)
    hsv[:, :, 0] = hue
    hsv[:, :, 1] = 255
    hsv[:, :, 2] = np.where(positive, 220, 0).astype(np.uint8)
    hsv[labels < 0] = (0, 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def _empty_debug(empty: np.ndarray) -> dict[str, np.ndarray]:
    empty_f32 = empty.astype(np.float32)
    empty_bgr = np.zeros((*empty.shape, 3), dtype=np.uint8)
    return {
        "metal_structural_denoised": empty,
        "metal_structural_gx": empty,
        "metal_structural_gy": empty,
        "metal_structural_gradient_magnitude": empty,
        "metal_structural_orientation": empty_bgr,
        "metal_structural_coherence": empty,
        "metal_structural_ridge_response": empty,
        "metal_structural_ridge_markers_raw": empty,
        "metal_structural_ridge_markers": empty,
        "metal_structural_wide_interior_markers": empty,
        "metal_structural_foreground_markers": empty,
        "metal_structural_background_markers": empty,
        "metal_structural_boundary_cost": empty,
        "metal_structural_watershed_labels": empty_bgr,
        "metal_structural_instance_labels": empty_bgr,
        "metal_structural_label_boundary": empty,
        "metal_structural_final_mask": empty,
        "metal_structural_gx_f32": empty_f32,
        "metal_structural_gy_f32": empty_f32,
        "metal_structural_ridge_fragments": empty_bgr,
        "metal_structural_wide_fragments": empty_bgr,
        "metal_structural_ridge_links_accepted": empty_bgr,
        "metal_structural_ridge_links_rejected": empty_bgr,
        "metal_structural_ridge_links_boundary_veto": empty_bgr,
        "metal_structural_logical_ridge": empty_bgr,
        "metal_structural_logical_wide": empty_bgr,
        "metal_structural_logical_markers": empty_bgr,
        "metal_structural_conductor_bands": empty_bgr,
        "metal_structural_transverse_samples": empty_bgr,
        "metal_structural_band_groups_accepted": empty_bgr,
        "metal_structural_band_groups_rejected": empty_bgr,
    }

def _debug_images(
    features: _StructuralFeatures,
    *,
    ridge_raw: np.ndarray,
    ridge_markers: np.ndarray,
    wide_markers: np.ndarray,
    fg_markers: np.ndarray,
    bg_markers: np.ndarray,
    cost: np.ndarray,
    watershed_labels: np.ndarray,
    instance_labels: np.ndarray,
    mask: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "metal_structural_denoised": ensure_uint8(features.denoised),
        "metal_structural_gx": cv2.convertScaleAbs(features.gradient_x),
        "metal_structural_gy": cv2.convertScaleAbs(features.gradient_y),
        "metal_structural_gradient_magnitude": _to_u8(features.magnitude),
        "metal_structural_orientation": _orientation_overlay(
            features.structure_orientation,
            features.coherence,
        ),
        "metal_structural_coherence": _to_u8(features.coherence),
        "metal_structural_ridge_response": _to_u8(features.ridge_response),
        "metal_structural_ridge_markers_raw": ensure_binary_mask(ridge_raw),
        "metal_structural_ridge_markers": ensure_binary_mask(ridge_markers),
        "metal_structural_wide_interior_markers": ensure_binary_mask(wide_markers),
        "metal_structural_foreground_markers": ensure_binary_mask(fg_markers),
        "metal_structural_background_markers": ensure_binary_mask(bg_markers),
        "metal_structural_boundary_cost": _to_u8(cost),
        "metal_structural_watershed_labels": _label_overlay(watershed_labels),
        "metal_structural_instance_labels": _label_overlay(instance_labels),
        "metal_structural_label_boundary": _label_boundary_map(instance_labels),
        "metal_structural_final_mask": ensure_binary_mask(mask),
        "metal_structural_gx_f32": features.gradient_x,
        "metal_structural_gy_f32": features.gradient_y,
        "metal_structural_instance_labels_i32": instance_labels.astype(np.int32, copy=False),
    }

if TYPE_CHECKING:
    from contour.vision.metal_recovery.structural_watershed_parts.extract_structural_features import _StructuralFeatures
