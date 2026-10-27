"""detector: stable local refinement."""
from __future__ import annotations
from typing import TYPE_CHECKING
from dataclasses import replace
from math import ceil
import numpy as np

from contour.vision.metal_recovery.detector_parts.prefer_contrast_refined_legacy import (
    _mask_iou,
    _non_hole_count,
)

def _stable_local_refinement(
    gray: np.ndarray,
    config: MetalRecoveryConfig,
    *,
    source_image: np.ndarray | None,
    reference: MetalDetectionResult,
) -> tuple[MetalDetectionResult | None, float | None, list[int]]:
    """Find a source-filter-stable local mask that still agrees with Auto."""
    # Keep the facade's documented injection point after splitting this module.
    from contour.vision.metal_recovery.detector import _detect_metalization_explicit
    reference_count = _non_hole_count(reference)
    step = max(0.0, float(config.auto_source_contrast_step))
    if reference_count < 100 or step <= 0.0:
        return None, None, []
    if gray.size / max(1, reference_count) >= 50_000:
        return None, None, []

    local_config = replace(
        config,
        segmentation_strategy="local_adaptive",
        use_wide_conductor_gradient=False,
    )
    variants: list[tuple[float, MetalDetectionResult, int]] = []
    for step_index in range(6):
        source_contrast = min(
            255.0,
            float(config.min_object_source_contrast) + step_index * step,
        )
        candidate = _detect_metalization_explicit(
            gray,
            replace(local_config, min_object_source_contrast=source_contrast),
            source_image=source_image,
            include_axis_debug=False,
        )
        candidate_count = _non_hole_count(candidate)
        variants.append((source_contrast, candidate, candidate_count))
        if len(variants) < 3:
            continue
        previous = variants[-2]
        if abs(previous[2] - candidate_count) > max(
            1,
            round(0.01 * max(previous[2], candidate_count)),
        ):
            continue

        base_count = variants[0][2]
        required_drop = max(1, ceil(0.01 * max(1, base_count)))
        if base_count - previous[2] < required_drop:
            return None, None, [item[2] for item in variants]
        if abs(previous[2] - reference_count) > 0.10 * max(previous[2], reference_count):
            return None, None, [item[2] for item in variants]
        agreement = _mask_iou(
            previous[1].debug_images.get("metal_filtered_mask"),
            reference.debug_images.get("metal_filtered_mask"),
        )
        moderate_agreement_with_more_objects = (
            agreement >= 0.80
            and previous[2] > reference_count
            and previous[2] <= 1.05 * reference_count
        )
        if agreement < 0.90 and not moderate_agreement_with_more_objects:
            return None, None, [item[2] for item in variants]
        return previous[1], previous[0], [item[2] for item in variants]
    return None, None, [item[2] for item in variants]

if TYPE_CHECKING:
    from contour.vision.metal_recovery.detector_parts.contour_extraction_settings_metal import MetalDetectionResult
    from contour.vision.metal_recovery.detector_parts.metal_recovery_config import MetalRecoveryConfig
