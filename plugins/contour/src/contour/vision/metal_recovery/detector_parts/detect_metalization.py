"""detector: detect metalization."""
from __future__ import annotations
from dataclasses import replace
import cv2
import numpy as np
from contour.utils import ensure_uint8
from contour.vision.metal_recovery.gradient_watershed import (
    analyze_metal_presence,
)
from contour.vision.metal_recovery.pipeline_stages import axis_gradient_debug_images

from contour.vision.metal_recovery.detector_parts.contour_extraction_settings_metal import (
    MetalDetectionResult,
    _requested_segmentation_strategy,
)

from contour.vision.metal_recovery.detector_parts.detect_metalization_explicit import _detect_metalization_explicit

from contour.vision.metal_recovery.detector_parts.prefer_contrast_refined_legacy import (
    _extended_separator_mask,
    _non_hole_count,
    _prefer_contrast_refined_legacy,
)

from contour.vision.metal_recovery.detector_parts.refine_local_candidate import _refine_local_candidate

from contour.vision.metal_recovery.detector_parts.stable_local_refinement import _stable_local_refinement

def detect_metalization(
    image: np.ndarray,
    config: MetalRecoveryConfig,
    *,
    source_image: np.ndarray | None = None,
) -> MetalDetectionResult:
    """Recognize conductors and preserve topology with an automatic dual-path mode."""
    if _requested_segmentation_strategy(config) != "auto":
        return _detect_metalization_explicit(image, config, source_image=source_image)

    if image.ndim == 3:
        gray = cv2.cvtColor(ensure_uint8(image), cv2.COLOR_BGR2GRAY)
    else:
        gray = ensure_uint8(image)
    if gray.size == 0:
        return MetalDetectionResult(params_snapshot=config.to_snapshot())

    legacy_config = replace(
        config,
        segmentation_strategy="legacy_otsu",
        use_wide_conductor_gradient=False,
    )
    watershed_config = replace(
        config,
        segmentation_strategy="gradient_watershed",
        use_wide_conductor_gradient=True,
    )
    legacy = _detect_metalization_explicit(gray, legacy_config, source_image=source_image)
    watershed = _detect_metalization_explicit(gray, watershed_config, source_image=source_image)
    legacy_count = _non_hole_count(legacy)
    watershed_count = _non_hole_count(watershed)
    presence = analyze_metal_presence(
        gray,
        smoothing_sigma=float(config.watershed_smoothing_sigma),
    )
    coherence = presence.coherent_contrast_fraction

    selected = legacy
    selected_strategy = "legacy_otsu"
    selected_min_contrast = float(config.min_contrast)
    contrast_refined_count: int | None = None
    if not presence.has_metal:
        selected = watershed
        selected_strategy = "gradient_watershed"
    elif watershed_count == 0:
        selected = legacy
        selected_strategy = "legacy_otsu"
    elif (
        coherence < 0.30
        or legacy_count == 0
        or watershed_count <= 0.80 * legacy_count
    ):
        selected = watershed
        selected_strategy = "gradient_watershed"
    else:
        pixels_per_legacy_object = gray.size / max(1, legacy_count)
        counts_are_close = abs(watershed_count - legacy_count) <= 0.05 * max(
            legacy_count,
            watershed_count,
        )
        if pixels_per_legacy_object < 10_000 and counts_are_close:
            selected = watershed
            selected_strategy = "gradient_watershed"
        elif legacy_count <= 0.90 * watershed_count and pixels_per_legacy_object < 10_000:
            separators = _extended_separator_mask(gray, config)
            if separators is not None:
                legacy_mask = legacy.debug_images.get("metal_binary_mask")
                if legacy_mask is not None:
                    hybrid_mask = cv2.bitwise_and(
                        legacy_mask,
                        cv2.bitwise_not(separators),
                    )
                    hybrid = _detect_metalization_explicit(
                        gray,
                        legacy_config,
                        source_image=source_image,
                        mask_override=hybrid_mask,
                    )
                    hybrid_count = _non_hole_count(hybrid)
                    if legacy_count < hybrid_count <= round(1.05 * watershed_count):
                        selected = hybrid
                        selected_strategy = "legacy_otsu_extended_separators"

    if selected is legacy and watershed_count >= 1.15 * legacy_count:
        refined_min_contrast = min(
            255.0,
            float(config.min_contrast) + max(0.0, float(config.auto_contrast_step)),
        )
        if refined_min_contrast > float(config.min_contrast):
            refined_config = replace(legacy_config, min_contrast=refined_min_contrast)
            refined = _detect_metalization_explicit(
                gray,
                refined_config,
                source_image=source_image,
            )
            contrast_refined_count = _non_hole_count(refined)
            if _prefer_contrast_refined_legacy(
                legacy=legacy,
                refined=refined,
                watershed=watershed,
            ):
                selected = refined
                selected_strategy = "legacy_otsu_contrast_refined"
                selected_min_contrast = refined_min_contrast

    local_refined, local_source_contrast, local_counts = _stable_local_refinement(
        gray,
        config,
        source_image=source_image,
        reference=selected,
    )
    if local_refined is not None:
        assert local_source_contrast is not None
        local_config = replace(
            config,
            segmentation_strategy="local_adaptive",
            use_wide_conductor_gradient=False,
            min_object_source_contrast=float(local_source_contrast),
        )
        local_refined = _refine_local_candidate(
            gray,
            local_config,
            source_image=source_image,
            candidate=local_refined,
            reference=selected,
        )
        selected = local_refined
        selected_strategy = "local_adaptive_source_refined"
        selected.debug_images.update(axis_gradient_debug_images(gray))

    selected.params_snapshot = {
        **config.to_snapshot(),
        "auto_selected_strategy": selected_strategy,
        "auto_selected_min_contrast": selected_min_contrast,
        "auto_legacy_objects": legacy_count,
        "auto_watershed_objects": watershed_count,
        "auto_contrast_refined_objects": contrast_refined_count,
        "auto_local_source_contrast": local_source_contrast,
        "auto_local_objects": local_counts,
    }
    return selected
