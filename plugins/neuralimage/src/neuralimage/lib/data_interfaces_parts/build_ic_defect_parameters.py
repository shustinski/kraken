"""data_interfaces: build ic defect parameters."""
from __future__ import annotations
from collections.abc import Mapping
from typing import Any

from neuralimage.lib.data_interfaces_parts.module_state import _IC_DEFECT_TYPE_ALIASES

from neuralimage.lib.data_interfaces_parts.coerce_float_range import _coerce_float_range

from neuralimage.lib.data_interfaces_parts.ic_defect_parameters import (
    ICDefectParameters,
    _coerce_positive_int,
    _coerce_probability,
    _coerce_range,
)

def build_ic_defect_parameters(raw: Any | None) -> ICDefectParameters:
    defaults = ICDefectParameters()
    if raw is None:
        return defaults
    if isinstance(raw, ICDefectParameters):
        return raw
    if isinstance(raw, Mapping):
        payload = dict(raw)
    elif hasattr(raw, '__dict__'):
        payload = dict(vars(raw))
    else:
        return defaults

    probability = _coerce_probability(
        payload.get('defect_probability', payload.get('probability', defaults.defect_probability)),
        defaults.defect_probability,
    )
    min_defects = _coerce_positive_int(payload.get('min_defects', defaults.min_defects), defaults.min_defects)
    max_defects = _coerce_positive_int(payload.get('max_defects', defaults.max_defects), defaults.max_defects)
    if min_defects > max_defects:
        min_defects, max_defects = max_defects, min_defects

    raw_probabilities = payload.get('defect_probabilities', {})
    normalized_probabilities = dict(defaults.defect_probabilities)
    if isinstance(raw_probabilities, Mapping):
        for raw_key, raw_value in raw_probabilities.items():
            canonical_key = _IC_DEFECT_TYPE_ALIASES.get(str(raw_key).strip().lower())
            if canonical_key is None:
                continue
            try:
                normalized_probabilities[canonical_key] = max(0.0, float(raw_value))
            except (TypeError, ValueError):
                continue

    raw_severities = payload.get('defect_severities', {})
    normalized_severities = dict(defaults.defect_severities)
    if isinstance(raw_severities, Mapping):
        for raw_key, raw_value in raw_severities.items():
            canonical_key = _IC_DEFECT_TYPE_ALIASES.get(str(raw_key).strip().lower())
            if canonical_key is None:
                continue
            normalized_severities[canonical_key] = _coerce_probability(
                raw_value,
                defaults.defect_severities.get(canonical_key, 0.5),
            )

    return ICDefectParameters(
        enabled=bool(payload.get('enabled', defaults.enabled)),
        defect_probability=probability,
        min_defects=min_defects,
        max_defects=max_defects,
        max_attempts_per_defect=_coerce_positive_int(
            payload.get('max_attempts_per_defect', defaults.max_attempts_per_defect),
            defaults.max_attempts_per_defect,
        ),
        min_component_area=_coerce_positive_int(
            payload.get('min_component_area', defaults.min_component_area),
            defaults.min_component_area,
        ),
        line_break_width_range=_coerce_range(
            payload.get('line_break_width_range', defaults.line_break_width_range),
            defaults.line_break_width_range,
            cast_type=int,
        ),
        bridge_distance_range=_coerce_range(
            payload.get('bridge_distance_range', defaults.bridge_distance_range),
            defaults.bridge_distance_range,
            cast_type=int,
        ),
        necking_radius_range=_coerce_range(
            payload.get('necking_radius_range', defaults.necking_radius_range),
            defaults.necking_radius_range,
            cast_type=int,
        ),
        missing_metal_radius_range=_coerce_range(
            payload.get('missing_metal_radius_range', defaults.missing_metal_radius_range),
            defaults.missing_metal_radius_range,
            cast_type=int,
        ),
        spur_radius_range=_coerce_range(
            payload.get('spur_radius_range', defaults.spur_radius_range),
            defaults.spur_radius_range,
            cast_type=int,
        ),
        pinhole_radius_range=_coerce_range(
            payload.get('pinhole_radius_range', defaults.pinhole_radius_range),
            defaults.pinhole_radius_range,
            cast_type=int,
        ),
        via_hole_area_range=_coerce_range(
            payload.get('via_hole_area_range', defaults.via_hole_area_range),
            defaults.via_hole_area_range,
            cast_type=int,
        ),
        via_shift_range=_coerce_range(
            payload.get('via_shift_range', defaults.via_shift_range),
            defaults.via_shift_range,
            cast_type=int,
        ),
        via_size_delta_range=_coerce_range(
            payload.get('via_size_delta_range', defaults.via_size_delta_range),
            defaults.via_size_delta_range,
            cast_type=int,
        ),
        line_shift_range=_coerce_range(
            payload.get('line_shift_range', defaults.line_shift_range),
            defaults.line_shift_range,
            cast_type=int,
        ),
        line_shift_roi_scale_range=_coerce_float_range(
            payload.get('line_shift_roi_scale_range', defaults.line_shift_roi_scale_range),
            defaults.line_shift_roi_scale_range,
            minimum=0.05,
            maximum=1.0,
        ),
        use_input_mask=True,
        use_defect_mask_as_label=False,
        defect_probabilities=normalized_probabilities,
        defect_severities=normalized_severities,
    )
