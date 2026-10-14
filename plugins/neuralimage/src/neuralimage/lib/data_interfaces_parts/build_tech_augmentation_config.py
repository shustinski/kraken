"""data_interfaces: build tech augmentation config."""
from __future__ import annotations
from collections.abc import Mapping
from typing import Any

from neuralimage.lib.data_interfaces_parts.coerce_float_range import (
    _coerce_float_range,
    _coerce_int_range,
)

from neuralimage.lib.data_interfaces_parts.ic_defect_parameters import (
    _coerce_positive_int,
    _coerce_probability,
)

from neuralimage.lib.data_interfaces_parts.pcb_defect_parameters import (
    TechAugmentationParameters,
    TechBlurThresholdParameters,
    TechBoundaryAwareVariationParameters,
    TechGapVariationParameters,
    TechGlobalWidthVariationParameters,
    TechLocalMorphologyParameters,
    TechScaleRethresholdParameters,
)

def build_tech_augmentation_config(raw: Any | None) -> TechAugmentationParameters:
    defaults = TechAugmentationParameters()
    if raw is None:
        return defaults
    if isinstance(raw, TechAugmentationParameters):
        return raw
    if isinstance(raw, Mapping):
        payload = dict(raw)
    elif hasattr(raw, '__dict__'):
        payload = dict(vars(raw))
    else:
        return defaults

    def _nested_mapping(key: str) -> dict[str, Any]:
        nested = payload.get(key, {})
        if isinstance(nested, Mapping):
            return dict(nested)
        if hasattr(nested, '__dict__'):
            return dict(vars(nested))
        return {}

    global_width_payload = _nested_mapping('global_width')
    scale_payload = _nested_mapping('scale_rethreshold')
    blur_payload = _nested_mapping('blur_threshold')
    boundary_payload = _nested_mapping('boundary_aware')
    local_payload = _nested_mapping('local_morphology')
    gap_payload = _nested_mapping('gap_variation')

    def _operation_probability(nested: Mapping[str, Any], default: float) -> float:
        probability = _coerce_probability(nested.get('probability', default), default)
        return probability if bool(nested.get('enabled', True)) else 0.0

    min_operations = _coerce_positive_int(
        payload.get('min_operations', defaults.min_operations),
        defaults.min_operations,
    )
    max_operations = _coerce_positive_int(
        payload.get('max_operations', defaults.max_operations),
        defaults.max_operations,
    )
    if min_operations > max_operations:
        min_operations, max_operations = max_operations, min_operations

    return TechAugmentationParameters(
        enabled=bool(payload.get('enabled', defaults.enabled)),
        min_operations=min_operations,
        max_operations=max_operations,
        debug_return_pair=bool(payload.get('debug_return_pair', defaults.debug_return_pair)),
        binarization_threshold=_coerce_probability(
            payload.get('binarization_threshold', defaults.binarization_threshold),
            defaults.binarization_threshold,
        ),
        binary_tolerance=_coerce_probability(
            payload.get('binary_tolerance', defaults.binary_tolerance),
            defaults.binary_tolerance,
        ),
        max_changed_pixels_ratio=_coerce_probability(
            payload.get('max_changed_pixels_ratio', defaults.max_changed_pixels_ratio),
            defaults.max_changed_pixels_ratio,
        ),
        max_foreground_ratio_delta=_coerce_probability(
            payload.get('max_foreground_ratio_delta', defaults.max_foreground_ratio_delta),
            defaults.max_foreground_ratio_delta,
        ),
        global_width=TechGlobalWidthVariationParameters(
            probability=_operation_probability(global_width_payload, defaults.global_width.probability),
            kernel_size_range=_coerce_int_range(
                global_width_payload.get('kernel_size_range', defaults.global_width.kernel_size_range),
                defaults.global_width.kernel_size_range,
                minimum=1,
            ),
            erosion_probability=_coerce_probability(
                global_width_payload.get('erosion_probability', defaults.global_width.erosion_probability),
                defaults.global_width.erosion_probability,
            ),
        ),
        scale_rethreshold=TechScaleRethresholdParameters(
            probability=_operation_probability(scale_payload, defaults.scale_rethreshold.probability),
            scale_range=_coerce_float_range(
                scale_payload.get('scale_range', defaults.scale_rethreshold.scale_range),
                defaults.scale_rethreshold.scale_range,
                minimum=0.1,
            ),
            threshold=_coerce_probability(
                scale_payload.get('threshold', defaults.scale_rethreshold.threshold),
                defaults.scale_rethreshold.threshold,
            ),
        ),
        blur_threshold=TechBlurThresholdParameters(
            probability=_operation_probability(blur_payload, defaults.blur_threshold.probability),
            blur_radius_range=_coerce_float_range(
                blur_payload.get('blur_radius_range', defaults.blur_threshold.blur_radius_range),
                defaults.blur_threshold.blur_radius_range,
                minimum=0.0,
            ),
            threshold=_coerce_probability(
                blur_payload.get('threshold', defaults.blur_threshold.threshold),
                defaults.blur_threshold.threshold,
            ),
        ),
        boundary_aware=TechBoundaryAwareVariationParameters(
            probability=_operation_probability(boundary_payload, defaults.boundary_aware.probability),
            band_width_range=_coerce_int_range(
                boundary_payload.get('band_width_range', defaults.boundary_aware.band_width_range),
                defaults.boundary_aware.band_width_range,
                minimum=1,
            ),
            noise_cell_size_range=_coerce_int_range(
                boundary_payload.get('noise_cell_size_range', defaults.boundary_aware.noise_cell_size_range),
                defaults.boundary_aware.noise_cell_size_range,
                minimum=1,
            ),
            add_probability=_coerce_probability(
                boundary_payload.get('add_probability', defaults.boundary_aware.add_probability),
                defaults.boundary_aware.add_probability,
            ),
            remove_probability=_coerce_probability(
                boundary_payload.get('remove_probability', defaults.boundary_aware.remove_probability),
                defaults.boundary_aware.remove_probability,
            ),
            min_addition_support=_coerce_positive_int(
                boundary_payload.get('min_addition_support', defaults.boundary_aware.min_addition_support),
                defaults.boundary_aware.min_addition_support,
            ),
            min_removal_support=_coerce_positive_int(
                boundary_payload.get('min_removal_support', defaults.boundary_aware.min_removal_support),
                defaults.boundary_aware.min_removal_support,
            ),
            smoothing_kernel_size=_coerce_positive_int(
                boundary_payload.get('smoothing_kernel_size', defaults.boundary_aware.smoothing_kernel_size),
                defaults.boundary_aware.smoothing_kernel_size,
                minimum=0,
            ),
        ),
        local_morphology=TechLocalMorphologyParameters(
            probability=_operation_probability(local_payload, defaults.local_morphology.probability),
            roi_count_range=_coerce_int_range(
                local_payload.get('roi_count_range', defaults.local_morphology.roi_count_range),
                defaults.local_morphology.roi_count_range,
                minimum=1,
            ),
            roi_size_ratio_range=_coerce_float_range(
                local_payload.get('roi_size_ratio_range', defaults.local_morphology.roi_size_ratio_range),
                defaults.local_morphology.roi_size_ratio_range,
                minimum=0.01,
                maximum=1.0,
            ),
            kernel_size_range=_coerce_int_range(
                local_payload.get('kernel_size_range', defaults.local_morphology.kernel_size_range),
                defaults.local_morphology.kernel_size_range,
                minimum=1,
            ),
            erosion_probability=_coerce_probability(
                local_payload.get('erosion_probability', defaults.local_morphology.erosion_probability),
                defaults.local_morphology.erosion_probability,
            ),
        ),
        gap_variation=TechGapVariationParameters(
            probability=_operation_probability(gap_payload, defaults.gap_variation.probability),
            kernel_size_range=_coerce_int_range(
                gap_payload.get('kernel_size_range', defaults.gap_variation.kernel_size_range),
                defaults.gap_variation.kernel_size_range,
                minimum=1,
            ),
            opening_probability=_coerce_probability(
                gap_payload.get('opening_probability', defaults.gap_variation.opening_probability),
                defaults.gap_variation.opening_probability,
            ),
            max_bridge_neighbor_count=_coerce_positive_int(
                gap_payload.get('max_bridge_neighbor_count', defaults.gap_variation.max_bridge_neighbor_count),
                defaults.gap_variation.max_bridge_neighbor_count,
                minimum=0,
            ),
            min_gap_neighbor_count=_coerce_positive_int(
                gap_payload.get('min_gap_neighbor_count', defaults.gap_variation.min_gap_neighbor_count),
                defaults.gap_variation.min_gap_neighbor_count,
                minimum=0,
            ),
        ),
    )
