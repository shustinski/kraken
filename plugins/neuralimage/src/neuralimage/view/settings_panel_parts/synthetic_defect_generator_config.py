"""settings_panel: synthetic defect generator config."""
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


class SyntheticDefectGeneratorConfigMixin:
    def set_synthetic_defect_generator_config(self, config: Any) -> None:
        self._synthetic_defect_generator_payload = (
            self._clone_plain_payload(config) if config is not None else {}
        )
        resolved = build_synthetic_defect_generator_parameters(config)
        self.synthetic_defect_generator_check_box.setChecked(bool(resolved.enabled))
        self.synthetic_dataset_factor_spinbox.setValue(float(resolved.epoch_size_factor))
        self.synthetic_image_size_widget.set_target_size(
            int(resolved.image_size_xy[0]),
            int(resolved.image_size_xy[1]),
        )
        self.synthetic_trace_count_min_spinbox.setValue(int(resolved.trace_count_range[0]))
        self.synthetic_trace_count_max_spinbox.setValue(int(resolved.trace_count_range[1]))
        self.synthetic_segment_count_min_spinbox.setValue(int(resolved.segment_count_range[0]))
        self.synthetic_segment_count_max_spinbox.setValue(int(resolved.segment_count_range[1]))
        self.synthetic_trace_half_width_min_spinbox.setValue(int(resolved.trace_half_width_range[0]))
        self.synthetic_trace_half_width_max_spinbox.setValue(int(resolved.trace_half_width_range[1]))
        self.synthetic_background_noise_sigma_min_spinbox.setValue(float(resolved.background_noise_sigma_range[0]))
        self.synthetic_background_noise_sigma_max_spinbox.setValue(float(resolved.background_noise_sigma_range[1]))
        self.synthetic_trace_noise_sigma_min_spinbox.setValue(float(resolved.trace_noise_sigma_range[0]))
        self.synthetic_trace_noise_sigma_max_spinbox.setValue(float(resolved.trace_noise_sigma_range[1]))
        self.set_synthetic_topology_domain_value(str(resolved.topology_domain))
        payload = self._synthetic_defect_generator_payload if isinstance(self._synthetic_defect_generator_payload, dict) else {}
        self.set_combo_value(
            self.pcb_topology_family_combo,
            str(
                payload.get(
                    'pcb_topology_family',
                    resolved.topology_family if resolved.topology_domain == 'pcb' else PCB_TOPOLOGY_FAMILIES[0],
                )
            ),
        )
        self.set_combo_value(
            self.ic_topology_family_combo,
            str(
                payload.get(
                    'ic_topology_family',
                    resolved.topology_family if resolved.topology_domain == 'ic' else IC_TOPOLOGY_FAMILIES[0],
                )
            ),
        )
        active_defects = resolved.ic_defects if resolved.topology_domain == 'ic' else resolved.pcb_defects
        self.pcb_defects_probability_spinbox.setValue(float(active_defects.defect_probability))
        self.pcb_defects_min_count_spinbox.setValue(int(active_defects.min_defects))
        self.pcb_defects_max_count_spinbox.setValue(int(active_defects.max_defects))
        for defect_name, _label_key in PCB_DEFECT_WEIGHT_FIELDS:
            self.pcb_defect_type_checkboxes[defect_name].setChecked(
                float(resolved.pcb_defects.defect_probabilities.get(defect_name, 1.0)) > 0.0
            )
            self.pcb_defect_type_spinboxes[defect_name].setValue(
                int(round(float(resolved.pcb_defects.defect_severities.get(defect_name, 0.5)) * 100.0))
            )
        for defect_name, _label_key in IC_DEFECT_WEIGHT_FIELDS:
            self.ic_defect_type_checkboxes[defect_name].setChecked(
                float(resolved.ic_defects.defect_probabilities.get(defect_name, 1.0)) > 0.0
            )
            self.ic_defect_type_spinboxes[defect_name].setValue(
                int(round(float(resolved.ic_defects.defect_severities.get(defect_name, 0.5)) * 100.0))
            )
        self._sync_synthetic_domain_controls()
        self._sync_synthetic_defect_generator_controls(self.synthetic_defect_generator_check_box.isChecked())
