"""data_interfaces: build pcb defect parameters."""
from __future__ import annotations
from collections.abc import Mapping
from typing import Any

from neuralimage.lib.data_interfaces_parts.module_state import _PCB_DEFECT_TYPE_ALIASES

from neuralimage.lib.data_interfaces_parts.ic_defect_parameters import (
    _coerce_positive_int,
    _coerce_probability,
    _coerce_range,
)

from neuralimage.lib.data_interfaces_parts.pcb_defect_parameters import PCBDefectParameters

def build_pcb_defect_parameters(raw: Any | None) -> PCBDefectParameters:
    defaults = PCBDefectParameters()
    if raw is None:
        return defaults
    if isinstance(raw, PCBDefectParameters):
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
            canonical_key = _PCB_DEFECT_TYPE_ALIASES.get(str(raw_key).strip().lower())
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
            canonical_key = _PCB_DEFECT_TYPE_ALIASES.get(str(raw_key).strip().lower())
            if canonical_key is None:
                continue
            normalized_severities[canonical_key] = _coerce_probability(
                raw_value,
                defaults.defect_severities.get(canonical_key, 0.5),
            )
    else:
        for canonical_key in tuple(normalized_severities.keys()):
            if float(normalized_probabilities.get(canonical_key, 0.0)) > 0.0:
                normalized_severities[canonical_key] = 0.5

    return PCBDefectParameters(
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
        break_width_range=_coerce_range(
            payload.get('break_width_range', defaults.break_width_range),
            defaults.break_width_range,
            cast_type=int,
        ),
        short_bridge_distance_range=_coerce_range(
            payload.get('short_bridge_distance_range', defaults.short_bridge_distance_range),
            defaults.short_bridge_distance_range,
            cast_type=int,
        ),
        missing_copper_radius_range=_coerce_range(
            payload.get('missing_copper_radius_range', defaults.missing_copper_radius_range),
            defaults.missing_copper_radius_range,
            cast_type=int,
        ),
        excess_copper_radius_range=_coerce_range(
            payload.get('excess_copper_radius_range', defaults.excess_copper_radius_range),
            defaults.excess_copper_radius_range,
            cast_type=int,
        ),
        pinhole_radius_range=_coerce_range(
            payload.get('pinhole_radius_range', defaults.pinhole_radius_range),
            defaults.pinhole_radius_range,
            cast_type=int,
        ),
        spurious_copper_radius_range=_coerce_range(
            payload.get('spurious_copper_radius_range', defaults.spurious_copper_radius_range),
            defaults.spurious_copper_radius_range,
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
        misalignment_shift_range=_coerce_range(
            payload.get('misalignment_shift_range', defaults.misalignment_shift_range),
            defaults.misalignment_shift_range,
            cast_type=int,
        ),
        misalignment_roi_scale_range=_coerce_range(
            payload.get('misalignment_roi_scale_range', defaults.misalignment_roi_scale_range),
            defaults.misalignment_roi_scale_range,
            cast_type=float,
        ),
        use_input_mask=bool(payload.get('use_input_mask', defaults.use_input_mask)),
        use_defect_mask_as_label=bool(
            payload.get('use_defect_mask_as_label', defaults.use_defect_mask_as_label)
        ),
        defect_probabilities=normalized_probabilities,
        defect_severities=normalized_severities,
    )
