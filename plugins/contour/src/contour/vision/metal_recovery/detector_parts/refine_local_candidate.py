"""detector: refine local candidate."""
from __future__ import annotations
import cv2
import numpy as np
from contour.utils import ensure_uint8

from contour.vision.metal_recovery.detector_parts.detect_metalization_explicit import _detect_metalization_explicit

from contour.vision.metal_recovery.detector_parts.directionally_bridge_mask import _directionally_bridge_mask

from contour.vision.metal_recovery.detector_parts.prefer_contrast_refined_legacy import _non_hole_count

def _refine_local_candidate(
    gray: np.ndarray,
    config: MetalRecoveryConfig,
    *,
    source_image: np.ndarray | None,
    candidate: MetalDetectionResult,
    reference: MetalDetectionResult,
) -> MetalDetectionResult:
    candidate_count = _non_hole_count(candidate)
    if source_image is None or candidate_count <= 0:
        return candidate
    pixels_per_object = gray.size / candidate_count
    if pixels_per_object < 10_000:
        return candidate
    candidate_mask = candidate.debug_images.get("metal_filtered_mask")
    reference_mask = reference.debug_images.get("metal_filtered_mask")
    if candidate_mask is None or reference_mask is None:
        return candidate
    if source_image.ndim == 3:
        source_gray = cv2.cvtColor(ensure_uint8(source_image), cv2.COLOR_BGR2GRAY)
    else:
        source_gray = ensure_uint8(source_image)

    refined_mask = candidate_mask.copy()
    gradient_x = np.abs(cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3))
    gradient_y = np.abs(cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3))
    x_energy = float(np.mean(np.minimum(gradient_x, 255.0)))
    y_energy = float(np.mean(np.minimum(gradient_y, 255.0)))
    if y_energy >= 1.5 * max(0.1, x_energy):
        refined_mask = _directionally_bridge_mask(
            refined_mask,
            source_gray,
            bridge_px=config.auto_directional_gap_bridge_px,
            min_source_intensity=config.auto_directional_gap_min_source_intensity,
        )

    _candidate_components, candidate_labels = cv2.connectedComponents(
        refined_mask,
        connectivity=8,
    )
    _reference_components, reference_labels = cv2.connectedComponents(
        ensure_uint8(reference_mask),
        connectivity=8,
    )
    for reference_id in np.unique(reference_labels[reference_labels > 0]):
        reference_region = reference_labels == reference_id
        if float(np.mean(candidate_labels[reference_region] > 0)) < 0.10:
            refined_mask[reference_region] = 255

    return _detect_metalization_explicit(
        gray,
        config,
        source_image=source_image,
        mask_override=refined_mask,
        include_axis_debug=False,
    )
