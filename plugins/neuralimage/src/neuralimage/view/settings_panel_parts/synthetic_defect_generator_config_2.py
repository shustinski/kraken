"""settings_panel: synthetic defect generator config 2."""
from __future__ import annotations
from typing import Any
from neuralimage.lib.data_interfaces import (
    IC_TOPOLOGY_FAMILIES,
    PCB_TOPOLOGY_FAMILIES,
    build_synthetic_defect_generator_parameters,
)
from neuralimage.view.settings_panel_widgets import (
    SlidingPanel,  # noqa: F401 - compatibility re-export from neuralimage.view
    )

from neuralimage.view.settings_panel_parts.module_state import (
    IC_DEFECT_WEIGHT_FIELDS,
    PCB_DEFECT_WEIGHT_FIELDS,
)


class SyntheticDefectGeneratorConfig2Mixin:
    def get_synthetic_defect_generator_config(self) -> dict[str, Any]:
        defaults = build_synthetic_defect_generator_parameters(None)
        payload = self._clone_plain_payload(self._synthetic_defect_generator_payload)
        if not isinstance(payload, dict):
            payload = {}

        self._set_optional_config_value(
            payload,
            ('enabled',),
            bool(self.synthetic_defect_generator_check_box.isChecked()),
            bool(defaults.enabled),
        )
        self._set_optional_config_value(
            payload,
            ('epoch_size_factor',),
            float(self.synthetic_dataset_factor_spinbox.value()),
            float(defaults.epoch_size_factor),
        )
        self._set_optional_config_value(
            payload,
            ('image_size_xy',),
            [
                int(self.synthetic_image_width_spinbox.value()),
                int(self.synthetic_image_height_spinbox.value()),
            ],
            list(defaults.image_size_xy),
        )
        topology_domain = self.get_synthetic_topology_domain_value()
        pcb_topology_family = self.get_combo_value(self.pcb_topology_family_combo)
        ic_topology_family = self.get_combo_value(self.ic_topology_family_combo)
        active_defaults = defaults.ic_defects if topology_domain == 'ic' else defaults.pcb_defects
        active_fields = IC_DEFECT_WEIGHT_FIELDS if topology_domain == 'ic' else PCB_DEFECT_WEIGHT_FIELDS
        active_sliders = self.ic_defect_type_spinboxes if topology_domain == 'ic' else self.pcb_defect_type_spinboxes
        active_checkboxes = self.ic_defect_type_checkboxes if topology_domain == 'ic' else self.pcb_defect_type_checkboxes

        self._set_optional_config_value(
            payload,
            ('topology_domain',),
            topology_domain,
            str(defaults.topology_domain),
        )
        self._set_optional_config_value(
            payload,
            ('pcb_topology_family',),
            pcb_topology_family,
            PCB_TOPOLOGY_FAMILIES[0],
        )
        self._set_optional_config_value(
            payload,
            ('ic_topology_family',),
            ic_topology_family,
            IC_TOPOLOGY_FAMILIES[0],
        )
        self._set_optional_config_value(
            payload,
            ('topology_family',),
            ic_topology_family if topology_domain == 'ic' else pcb_topology_family,
            str(defaults.topology_family),
        )
        trace_count_min = int(self.synthetic_trace_count_min_spinbox.value())
        trace_count_max = int(self.synthetic_trace_count_max_spinbox.value())
        if trace_count_min > trace_count_max:
            trace_count_min, trace_count_max = trace_count_max, trace_count_min
        segment_count_min = int(self.synthetic_segment_count_min_spinbox.value())
        segment_count_max = int(self.synthetic_segment_count_max_spinbox.value())
        if segment_count_min > segment_count_max:
            segment_count_min, segment_count_max = segment_count_max, segment_count_min
        trace_half_width_min = int(self.synthetic_trace_half_width_min_spinbox.value())
        trace_half_width_max = int(self.synthetic_trace_half_width_max_spinbox.value())
        if trace_half_width_min > trace_half_width_max:
            trace_half_width_min, trace_half_width_max = trace_half_width_max, trace_half_width_min
        background_noise_sigma_min = float(self.synthetic_background_noise_sigma_min_spinbox.value())
        background_noise_sigma_max = float(self.synthetic_background_noise_sigma_max_spinbox.value())
        if background_noise_sigma_min > background_noise_sigma_max:
            background_noise_sigma_min, background_noise_sigma_max = background_noise_sigma_max, background_noise_sigma_min
        trace_noise_sigma_min = float(self.synthetic_trace_noise_sigma_min_spinbox.value())
        trace_noise_sigma_max = float(self.synthetic_trace_noise_sigma_max_spinbox.value())
        if trace_noise_sigma_min > trace_noise_sigma_max:
            trace_noise_sigma_min, trace_noise_sigma_max = trace_noise_sigma_max, trace_noise_sigma_min
        self._set_optional_config_value(
            payload,
            ('trace_count_range',),
            [trace_count_min, trace_count_max],
            list(defaults.trace_count_range),
        )
        self._set_optional_config_value(
            payload,
            ('trace_count',),
            trace_count_max,
            int(defaults.trace_count),
        )
        self._set_optional_config_value(
            payload,
            ('segment_count_range',),
            [segment_count_min, segment_count_max],
            list(defaults.segment_count_range),
        )
        self._set_optional_config_value(
            payload,
            ('segment_count',),
            segment_count_max,
            int(defaults.segment_count),
        )
        self._set_optional_config_value(
            payload,
            ('trace_half_width_range',),
            [trace_half_width_min, trace_half_width_max],
            list(defaults.trace_half_width_range),
        )
        self._set_optional_config_value(
            payload,
            ('trace_half_width',),
            trace_half_width_max,
            int(defaults.trace_half_width),
        )
        self._set_optional_config_value(
            payload,
            ('background_noise_sigma_range',),
            [background_noise_sigma_min, background_noise_sigma_max],
            list(defaults.background_noise_sigma_range),
        )
        self._set_optional_config_value(
            payload,
            ('background_noise_sigma',),
            background_noise_sigma_max,
            float(defaults.background_noise_sigma),
        )
        self._set_optional_config_value(
            payload,
            ('trace_noise_sigma_range',),
            [trace_noise_sigma_min, trace_noise_sigma_max],
            list(defaults.trace_noise_sigma_range),
        )
        self._set_optional_config_value(
            payload,
            ('trace_noise_sigma',),
            trace_noise_sigma_max,
            float(defaults.trace_noise_sigma),
        )
        min_defects = int(self.pcb_defects_min_count_spinbox.value())
        max_defects = int(self.pcb_defects_max_count_spinbox.value())
        if min_defects > max_defects:
            min_defects, max_defects = max_defects, min_defects
        for group_name, group_defaults, field_defs, slider_map, checkbox_map in (
            ('pcb_defects', defaults.pcb_defects, PCB_DEFECT_WEIGHT_FIELDS, self.pcb_defect_type_spinboxes, self.pcb_defect_type_checkboxes),
            ('ic_defects', defaults.ic_defects, IC_DEFECT_WEIGHT_FIELDS, self.ic_defect_type_spinboxes, self.ic_defect_type_checkboxes),
        ):
            existing_group_payload = payload.get(group_name, {}) if isinstance(payload.get(group_name), dict) else {}
            existing_active_payload = payload.get('defects', {}) if isinstance(payload.get('defects'), dict) else {}
            self._set_optional_config_value(
                payload,
                (group_name, 'enabled'),
                bool(self.synthetic_defect_generator_check_box.isChecked()),
                bool(group_defaults.enabled),
            )
            self._set_optional_config_value(
                payload,
                (group_name, 'defect_probability'),
                float(self.pcb_defects_probability_spinbox.value()),
                float(group_defaults.defect_probability),
            )
            self._set_optional_config_value(
                payload,
                (group_name, 'min_defects'),
                min_defects,
                int(group_defaults.min_defects),
            )
            self._set_optional_config_value(
                payload,
                (group_name, 'max_defects'),
                max_defects,
                int(group_defaults.max_defects),
            )
            self._set_optional_config_value(
                payload,
                (group_name, 'use_input_mask'),
                True,
                bool(group_defaults.use_input_mask),
            )
            self._set_optional_config_value(
                payload,
                (group_name, 'use_defect_mask_as_label'),
                False,
                bool(group_defaults.use_defect_mask_as_label),
            )
            self._set_optional_config_value(
                payload,
                (group_name, 'max_attempts_per_defect'),
                int(existing_group_payload.get('max_attempts_per_defect', existing_active_payload.get('max_attempts_per_defect', group_defaults.max_attempts_per_defect))),
                int(group_defaults.max_attempts_per_defect),
            )
            self._set_optional_config_value(
                payload,
                (group_name, 'min_component_area'),
                int(existing_group_payload.get('min_component_area', existing_active_payload.get('min_component_area', group_defaults.min_component_area))),
                int(group_defaults.min_component_area),
            )
            for defect_name, _label_key in field_defs:
                self._set_optional_config_value(
                    payload,
                    (group_name, 'defect_probabilities', defect_name),
                    1.0 if bool(checkbox_map[defect_name].isChecked()) else 0.0,
                    float(group_defaults.defect_probabilities.get(defect_name, 1.0)),
                )
                self._set_optional_config_value(
                    payload,
                    (group_name, 'defect_severities', defect_name),
                    float(slider_map[defect_name].value()) / 100.0,
                    float(group_defaults.defect_severities.get(defect_name, 0.5)),
                )
        self._set_optional_config_value(
            payload,
            ('defects', 'enabled'),
            bool(self.synthetic_defect_generator_check_box.isChecked()),
            bool(active_defaults.enabled),
        )
        self._set_optional_config_value(
            payload,
            ('defects', 'defect_probability'),
            float(self.pcb_defects_probability_spinbox.value()),
            float(active_defaults.defect_probability),
        )
        self._set_optional_config_value(
            payload,
            ('defects', 'min_defects'),
            min_defects,
            int(active_defaults.min_defects),
        )
        self._set_optional_config_value(
            payload,
            ('defects', 'max_defects'),
            max_defects,
            int(active_defaults.max_defects),
        )
        self._set_optional_config_value(
            payload,
            ('defects', 'use_input_mask'),
            True,
            bool(active_defaults.use_input_mask),
        )
        self._set_optional_config_value(
            payload,
            ('defects', 'use_defect_mask_as_label'),
            False,
            bool(active_defaults.use_defect_mask_as_label),
        )
        existing_active_payload = payload.get('defects', {}) if isinstance(payload.get('defects'), dict) else {}
        self._set_optional_config_value(
            payload,
            ('defects', 'max_attempts_per_defect'),
            int(existing_active_payload.get('max_attempts_per_defect', active_defaults.max_attempts_per_defect)),
            int(active_defaults.max_attempts_per_defect),
        )
        self._set_optional_config_value(
            payload,
            ('defects', 'min_component_area'),
            int(existing_active_payload.get('min_component_area', active_defaults.min_component_area)),
            int(active_defaults.min_component_area),
        )
        for defect_name, _label_key in active_fields:
            self._set_optional_config_value(
                payload,
                ('defects', 'defect_probabilities', defect_name),
                1.0 if bool(active_checkboxes[defect_name].isChecked()) else 0.0,
                float(active_defaults.defect_probabilities.get(defect_name, 1.0)),
            )
            self._set_optional_config_value(
                payload,
                ('defects', 'defect_severities', defect_name),
                float(active_sliders[defect_name].value()) / 100.0,
                float(active_defaults.defect_severities.get(defect_name, 0.5)),
            )
        self._synthetic_defect_generator_payload = self._clone_plain_payload(payload)
        return payload
