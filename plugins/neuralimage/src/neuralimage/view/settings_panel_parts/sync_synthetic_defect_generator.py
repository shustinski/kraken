"""settings_panel: sync synthetic defect generator."""
from __future__ import annotations
from neuralimage.view.settings_panel_widgets import (
    SlidingPanel,  # noqa: F401 - compatibility re-export from neuralimage.view
    )

from neuralimage.view.settings_panel_parts.module_state import (
    IC_DEFECT_WEIGHT_FIELDS,
    PCB_DEFECT_WEIGHT_FIELDS,
)


class SyncSyntheticDefectGeneratorMixin:
    def _sync_synthetic_defect_generator_controls(self, _enabled: bool | None = None) -> None:
        training_enabled = self._training_controls_applicable
        synthetic_enabled = training_enabled and bool(self.synthetic_defect_generator_check_box.isChecked())
        current_domain = self.get_synthetic_topology_domain_value()
        self.synthetic_defect_generator_groupbox.setEnabled(training_enabled)
        self.synthetic_defect_generator_check_box.setEnabled(training_enabled)
        synthetic_fields = (
            self.synthetic_topology_domain_combo,
            self.pcb_topology_family_combo,
            self.ic_topology_family_combo,
            self.synthetic_dataset_factor_spinbox,
            self.synthetic_image_size_widget,
            self.synthetic_trace_count_range_widget,
            self.synthetic_segment_count_range_widget,
            self.synthetic_trace_half_width_range_widget,
            self.synthetic_background_noise_sigma_range_widget,
            self.synthetic_trace_noise_sigma_range_widget,
            self.pcb_defects_probability_spinbox,
            self.pcb_defects_min_count_spinbox,
            self.pcb_defects_max_count_spinbox,
            *self.pcb_defect_type_checkboxes.values(),
            *self.pcb_defect_type_spinboxes.values(),
            *self.ic_defect_type_checkboxes.values(),
            *self.ic_defect_type_spinboxes.values(),
        )
        for field in synthetic_fields:
            self._set_field_visible(field, synthetic_enabled)
        self._set_field_enabled(self.synthetic_topology_domain_combo, synthetic_enabled)
        self._set_field_enabled(
            self.pcb_topology_family_combo,
            synthetic_enabled and current_domain == 'pcb',
        )
        self._set_field_enabled(
            self.ic_topology_family_combo,
            synthetic_enabled and current_domain == 'ic',
        )
        self._set_field_enabled(self.synthetic_dataset_factor_spinbox, synthetic_enabled)
        self._set_field_enabled(self.synthetic_image_size_widget, synthetic_enabled)
        self._set_field_enabled(self.synthetic_trace_count_range_widget, synthetic_enabled)
        self._set_field_enabled(self.synthetic_segment_count_range_widget, synthetic_enabled)
        self._set_field_enabled(self.synthetic_trace_half_width_range_widget, synthetic_enabled)
        self._set_field_enabled(self.synthetic_background_noise_sigma_range_widget, synthetic_enabled)
        self._set_field_enabled(self.synthetic_trace_noise_sigma_range_widget, synthetic_enabled)
        self._set_field_enabled(self.pcb_defects_probability_spinbox, synthetic_enabled)
        self._set_field_enabled(self.pcb_defects_min_count_spinbox, synthetic_enabled)
        self._set_field_enabled(self.pcb_defects_max_count_spinbox, synthetic_enabled)
        for defect_name, _label_key in PCB_DEFECT_WEIGHT_FIELDS:
            self._set_field_enabled(
                self.pcb_defect_type_checkboxes[defect_name],
                synthetic_enabled and current_domain == 'pcb',
            )
            self._set_field_enabled(
                self.pcb_defect_type_spinboxes[defect_name],
                synthetic_enabled and current_domain == 'pcb' and bool(self.pcb_defect_type_checkboxes[defect_name].isChecked()),
            )
        for defect_name, _label_key in IC_DEFECT_WEIGHT_FIELDS:
            self._set_field_enabled(
                self.ic_defect_type_checkboxes[defect_name],
                synthetic_enabled and current_domain == 'ic',
            )
            self._set_field_enabled(
                self.ic_defect_type_spinboxes[defect_name],
                synthetic_enabled and current_domain == 'ic' and bool(self.ic_defect_type_checkboxes[defect_name].isChecked()),
            )
        self._sync_synthetic_domain_controls()

    def _sync_synthetic_domain_controls(self, _enabled: bool | None = None) -> None:
        is_ic = self.get_synthetic_topology_domain_value() == 'ic'
        synthetic_enabled = self._training_controls_applicable and self.synthetic_defect_generator_check_box.isChecked()
        self._set_field_visible(self.pcb_topology_family_combo, synthetic_enabled and not is_ic)
        self._set_field_visible(self.ic_topology_family_combo, synthetic_enabled and is_ic)
        for defect_name, _label_key in PCB_DEFECT_WEIGHT_FIELDS:
            self._set_field_visible(self.pcb_defect_type_checkboxes[defect_name], synthetic_enabled and not is_ic)
            self._set_field_visible(self.pcb_defect_type_spinboxes[defect_name], synthetic_enabled and not is_ic)
        for defect_name, _label_key in IC_DEFECT_WEIGHT_FIELDS:
            self._set_field_visible(self.ic_defect_type_checkboxes[defect_name], synthetic_enabled and is_ic)
            self._set_field_visible(self.ic_defect_type_spinboxes[defect_name], synthetic_enabled and is_ic)
