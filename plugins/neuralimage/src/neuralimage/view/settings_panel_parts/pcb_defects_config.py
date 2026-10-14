"""settings_panel: pcb defects config."""
from __future__ import annotations
from typing import Any
from neuralimage.lib.data_interfaces import (
    build_pcb_defect_parameters,
)
from neuralimage.view.settings_panel_widgets import (
    SlidingPanel,  # noqa: F401 - compatibility re-export from neuralimage.view
    )

from neuralimage.view.settings_panel_parts.module_state import PCB_DEFECT_WEIGHT_FIELDS


class PcbDefectsConfigMixin:
    def set_pcb_defects_config(self, config: Any) -> None:
        self._pcb_defects_config_payload = self._clone_plain_payload(config) if config is not None else {}
        resolved = build_pcb_defect_parameters(config)
        self.pcb_defects_check_box.setChecked(bool(resolved.enabled))
        self.pcb_defects_probability_spinbox.setValue(float(resolved.defect_probability))
        self.pcb_defects_min_count_spinbox.setValue(int(resolved.min_defects))
        self.pcb_defects_max_count_spinbox.setValue(int(resolved.max_defects))
        self.pcb_defects_use_input_mask_check_box.setChecked(bool(resolved.use_input_mask))
        self.pcb_defects_use_defect_mask_as_label_check_box.setChecked(bool(resolved.use_defect_mask_as_label))
        for defect_name, _label_key in PCB_DEFECT_WEIGHT_FIELDS:
            self.pcb_defect_type_spinboxes[defect_name].setValue(
                int(round(float(resolved.defect_severities.get(defect_name, 0.5)) * 100.0))
            )
        self._sync_training_augmentation_controls()

    def get_pcb_defects_config(self) -> dict[str, Any]:
        defaults = build_pcb_defect_parameters(None)
        payload = self._clone_plain_payload(self._pcb_defects_config_payload)
        if not isinstance(payload, dict):
            payload = {}

        self._set_optional_config_value(
            payload,
            ('enabled',),
            bool(self.pcb_defects_check_box.isChecked()),
            bool(defaults.enabled),
        )
        min_defects = int(self.pcb_defects_min_count_spinbox.value())
        max_defects = int(self.pcb_defects_max_count_spinbox.value())
        if min_defects > max_defects:
            min_defects, max_defects = max_defects, min_defects
        self._set_optional_config_value(payload, ('defect_probability',), float(self.pcb_defects_probability_spinbox.value()), float(defaults.defect_probability))
        self._set_optional_config_value(payload, ('min_defects',), min_defects, int(defaults.min_defects))
        self._set_optional_config_value(payload, ('max_defects',), max_defects, int(defaults.max_defects))
        self._set_optional_config_value(
            payload,
            ('use_input_mask',),
            bool(self.pcb_defects_use_input_mask_check_box.isChecked()),
            bool(defaults.use_input_mask),
        )
        self._set_optional_config_value(
            payload,
            ('use_defect_mask_as_label',),
            bool(self.pcb_defects_use_defect_mask_as_label_check_box.isChecked()),
            bool(defaults.use_defect_mask_as_label),
        )
        for defect_name, _label_key in PCB_DEFECT_WEIGHT_FIELDS:
            self._set_optional_config_value(
                payload,
                ('defect_probabilities', defect_name),
                1.0 if bool(self.pcb_defects_check_box.isChecked()) else 0.0,
                float(defaults.defect_probabilities.get(defect_name, 1.0)),
            )
            self._set_optional_config_value(
                payload,
                ('defect_severities', defect_name),
                float(self.pcb_defect_type_spinboxes[defect_name].value()) / 100.0,
                float(defaults.defect_severities.get(defect_name, 0.5)),
            )
        self._pcb_defects_config_payload = self._clone_plain_payload(payload)
        return payload
