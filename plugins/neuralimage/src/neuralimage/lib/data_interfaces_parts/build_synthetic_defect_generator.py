"""data_interfaces: build synthetic defect generator."""
from __future__ import annotations
from collections.abc import Mapping
from typing import Any

from neuralimage.lib.data_interfaces_parts.module_state import (
    IC_TOPOLOGY_FAMILIES,
    PCB_TOPOLOGY_FAMILIES,
    SYNTHETIC_TOPOLOGY_DOMAINS,
)

from neuralimage.lib.data_interfaces_parts.build_ic_defect_parameters import build_ic_defect_parameters

from neuralimage.lib.data_interfaces_parts.build_pcb_defect_parameters import build_pcb_defect_parameters

from neuralimage.lib.data_interfaces_parts.coerce_float_range import (
    _coerce_float_range,
    _coerce_int_range,
)

from neuralimage.lib.data_interfaces_parts.ic_defect_parameters import (
    SyntheticDefectGeneratorParameters,
    _coerce_positive_float,
    _coerce_positive_int,
    _coerce_size_pair,
)

def build_synthetic_defect_generator_parameters(raw: Any | None) -> SyntheticDefectGeneratorParameters:
    defaults = SyntheticDefectGeneratorParameters()
    if raw is None:
        defaults.defects = defaults.pcb_defects
        return defaults
    if isinstance(raw, SyntheticDefectGeneratorParameters):
        return raw
    if isinstance(raw, Mapping):
        payload = dict(raw)
    elif hasattr(raw, '__dict__'):
        payload = dict(vars(raw))
    else:
        return defaults

    topology_domain = str(payload.get('topology_domain', payload.get('domain', defaults.topology_domain)) or defaults.topology_domain).strip().lower()
    if topology_domain not in SYNTHETIC_TOPOLOGY_DOMAINS:
        topology_domain = defaults.topology_domain
    topology_family = str(payload.get('topology_family', '') or '').strip().lower()
    if not topology_family:
        family_key = f'{topology_domain}_topology_family'
        topology_family = str(
            payload.get(
                family_key,
                IC_TOPOLOGY_FAMILIES[0] if topology_domain == 'ic' else PCB_TOPOLOGY_FAMILIES[0],
            )
            or (IC_TOPOLOGY_FAMILIES[0] if topology_domain == 'ic' else PCB_TOPOLOGY_FAMILIES[0])
        ).strip().lower()

    defects_payload = payload.get('defects')
    if defects_payload is None:
        defects_payload = {
            'enabled': bool(payload.get('enabled', defaults.enabled)),
            'defect_probability': payload.get(
                'defect_probability',
                defaults.defects.defect_probability,
            ),
            'min_defects': payload.get('min_defects', defaults.defects.min_defects),
            'max_defects': payload.get('max_defects', defaults.defects.max_defects),
            'defect_probabilities': payload.get(
                'defect_probabilities',
                defaults.defects.defect_probabilities,
            ),
            'max_attempts_per_defect': payload.get(
                'max_attempts_per_defect',
                defaults.defects.max_attempts_per_defect,
            ),
            'min_component_area': payload.get(
                'min_component_area',
                defaults.defects.min_component_area,
            ),
        }
    pcb_payload = payload.get('pcb_defects', defects_payload if topology_domain == 'pcb' else {})
    ic_payload = payload.get('ic_defects', defects_payload if topology_domain == 'ic' else {})
    pcb_defects = build_pcb_defect_parameters(pcb_payload)
    pcb_enabled_flag = (
        bool(pcb_payload.get('enabled'))
        if isinstance(pcb_payload, Mapping) and 'enabled' in pcb_payload
        else bool(payload.get('enabled', defaults.enabled))
    )
    pcb_defects.enabled = bool(pcb_enabled_flag)
    pcb_defects.use_input_mask = True
    pcb_defects.use_defect_mask_as_label = False
    ic_defects = build_ic_defect_parameters(ic_payload)
    ic_enabled_flag = (
        bool(ic_payload.get('enabled'))
        if isinstance(ic_payload, Mapping) and 'enabled' in ic_payload
        else bool(payload.get('enabled', defaults.enabled))
    )
    ic_defects.enabled = bool(ic_enabled_flag)
    ic_defects.use_input_mask = True
    ic_defects.use_defect_mask_as_label = False
    active_defects: Any = ic_defects if topology_domain == 'ic' else pcb_defects

    trace_count_value = payload.get('trace_count')
    segment_count_value = payload.get('segment_count')
    trace_half_width_value = payload.get('trace_half_width')
    background_noise_sigma_value = payload.get('background_noise_sigma')
    trace_noise_sigma_value = payload.get('trace_noise_sigma')

    trace_count_range = _coerce_int_range(
        payload.get(
            'trace_count_range',
            (
                payload.get('trace_count_min', trace_count_value if trace_count_value is not None else defaults.trace_count_range[0]),
                payload.get('trace_count_max', trace_count_value if trace_count_value is not None else defaults.trace_count_range[1]),
            ),
        ),
        defaults.trace_count_range,
        minimum=1,
    )
    segment_count_range = _coerce_int_range(
        payload.get(
            'segment_count_range',
            (
                payload.get('segment_count_min', segment_count_value if segment_count_value is not None else defaults.segment_count_range[0]),
                payload.get('segment_count_max', segment_count_value if segment_count_value is not None else defaults.segment_count_range[1]),
            ),
        ),
        defaults.segment_count_range,
        minimum=1,
    )
    trace_half_width_range = _coerce_int_range(
        payload.get(
            'trace_half_width_range',
            (
                payload.get('trace_half_width_min', trace_half_width_value if trace_half_width_value is not None else defaults.trace_half_width_range[0]),
                payload.get('trace_half_width_max', trace_half_width_value if trace_half_width_value is not None else defaults.trace_half_width_range[1]),
            ),
        ),
        defaults.trace_half_width_range,
        minimum=1,
    )
    background_noise_sigma_range = _coerce_float_range(
        payload.get(
            'background_noise_sigma_range',
            (
                payload.get('background_noise_sigma_min', background_noise_sigma_value if background_noise_sigma_value is not None else defaults.background_noise_sigma_range[0]),
                payload.get('background_noise_sigma_max', background_noise_sigma_value if background_noise_sigma_value is not None else defaults.background_noise_sigma_range[1]),
            ),
        ),
        defaults.background_noise_sigma_range,
        minimum=0.0,
    )
    trace_noise_sigma_range = _coerce_float_range(
        payload.get(
            'trace_noise_sigma_range',
            (
                payload.get('trace_noise_sigma_min', trace_noise_sigma_value if trace_noise_sigma_value is not None else defaults.trace_noise_sigma_range[0]),
                payload.get('trace_noise_sigma_max', trace_noise_sigma_value if trace_noise_sigma_value is not None else defaults.trace_noise_sigma_range[1]),
            ),
        ),
        defaults.trace_noise_sigma_range,
        minimum=0.0,
    )
    image_size_xy = _coerce_size_pair(
        payload.get(
            'image_size_xy',
            payload.get(
                'frame_size_xy',
                (
                    payload.get(
                        'image_width',
                        payload.get(
                            'frame_width',
                            defaults.image_size_xy[0],
                        ),
                    ),
                    payload.get(
                        'image_height',
                        payload.get(
                            'frame_height',
                            defaults.image_size_xy[1],
                        ),
                    ),
                ),
            ),
        ),
        defaults.image_size_xy,
        minimum=1,
    )

    return SyntheticDefectGeneratorParameters(
        enabled=bool(payload.get('enabled', defaults.enabled)),
        epoch_size_factor=_coerce_positive_float(
            payload.get('epoch_size_factor', defaults.epoch_size_factor),
            defaults.epoch_size_factor,
            minimum=0.0,
        ),
        topology_domain=topology_domain,
        topology_family=topology_family,
        image_size_xy=image_size_xy,
        trace_count_range=trace_count_range,
        segment_count_range=segment_count_range,
        trace_half_width_range=trace_half_width_range,
        margin=_coerce_positive_int(
            payload.get('margin', defaults.margin),
            defaults.margin,
            minimum=1,
        ),
        background_noise_sigma_range=background_noise_sigma_range,
        trace_noise_sigma_range=trace_noise_sigma_range,
        pcb_defects=pcb_defects,
        ic_defects=ic_defects,
        defects=active_defects,
    )
