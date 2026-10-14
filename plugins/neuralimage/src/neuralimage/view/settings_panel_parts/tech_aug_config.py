"""settings_panel: tech aug config."""
from __future__ import annotations
from typing import Any
from collections.abc import Mapping
from neuralimage.lib.data_interfaces import (
    build_tech_augmentation_config,
)
from neuralimage.view.settings_panel_widgets import (
    SlidingPanel,  # noqa: F401 - compatibility re-export from neuralimage.view
    )


class TechAugConfigMixin:
    def set_tech_aug_config(self, config: Any) -> None:
        self._tech_aug_config_payload = self._clone_plain_payload(config) if config is not None else {}
        resolved = build_tech_augmentation_config(config)
        raw = config if isinstance(config, Mapping) else {}
        operation_values = {
            'global_width': resolved.global_width.probability,
            'scale_rethreshold': resolved.scale_rethreshold.probability,
            'blur_threshold': resolved.blur_threshold.probability,
            'boundary_aware': resolved.boundary_aware.probability,
            'local_morphology': resolved.local_morphology.probability,
            'gap_variation': resolved.gap_variation.probability,
        }

        def configured_probability(key: str) -> float:
            nested = raw.get(key, {}) if isinstance(raw, Mapping) else {}
            if isinstance(nested, Mapping) and 'probability' in nested:
                return float(nested['probability'])
            return float(operation_values[key])
        for key, checkbox in self.tech_augmentation_operation_checkboxes.items():
            nested = raw.get(key, {}) if isinstance(raw, Mapping) else {}
            explicitly_enabled = nested.get('enabled') if isinstance(nested, Mapping) else None
            checkbox.setChecked(
                bool(resolved.enabled)
                and (
                    bool(explicitly_enabled)
                    if explicitly_enabled is not None
                    else float(operation_values[key]) > 0.0
                )
            )
        self.tech_augmentation_check_box.setChecked(
            bool(resolved.enabled) and any(
                checkbox.isChecked() for checkbox in self.tech_augmentation_operation_checkboxes.values()
            )
        )
        self.tech_aug_min_operations_spinbox.setValue(int(resolved.min_operations))
        self.tech_aug_max_operations_spinbox.setValue(int(resolved.max_operations))
        self.tech_augmentation_debug_pair_check_box.setChecked(bool(resolved.debug_return_pair))
        self.tech_aug_max_changed_pixels_ratio_spinbox.setValue(float(resolved.max_changed_pixels_ratio))
        self.tech_aug_max_foreground_ratio_delta_spinbox.setValue(float(resolved.max_foreground_ratio_delta))
        self.tech_aug_global_width_probability_spinbox.setValue(configured_probability('global_width'))
        self.tech_aug_scale_rethreshold_probability_spinbox.setValue(
            configured_probability('scale_rethreshold')
        )
        self.tech_aug_blur_threshold_probability_spinbox.setValue(configured_probability('blur_threshold'))
        self.tech_aug_boundary_aware_probability_spinbox.setValue(configured_probability('boundary_aware'))
        self.tech_aug_local_morphology_probability_spinbox.setValue(configured_probability('local_morphology'))
        self.tech_aug_gap_variation_probability_spinbox.setValue(configured_probability('gap_variation'))
        self._sync_tech_augmentation_controls(self.tech_augmentation_check_box.isChecked())
        if hasattr(self, 'training_augmentation_editor'):
            self.training_augmentation_editor.sync_from_controls()

    def get_tech_aug_config(self) -> dict[str, Any]:
        defaults = build_tech_augmentation_config(None)
        payload = self._clone_plain_payload(self._tech_aug_config_payload)
        if not isinstance(payload, dict):
            payload = {}

        self._set_optional_config_value(
            payload,
            ('enabled',),
            bool(self.tech_augmentation_check_box.isChecked()),
            bool(defaults.enabled),
        )
        min_operations = int(self.tech_aug_min_operations_spinbox.value())
        max_operations = int(self.tech_aug_max_operations_spinbox.value())
        if min_operations > max_operations:
            min_operations, max_operations = max_operations, min_operations
        self._set_optional_config_value(payload, ('min_operations',), min_operations, int(defaults.min_operations))
        self._set_optional_config_value(payload, ('max_operations',), max_operations, int(defaults.max_operations))
        self._set_optional_config_value(
            payload,
            ('debug_return_pair',),
            bool(self.tech_augmentation_debug_pair_check_box.isChecked()),
            bool(defaults.debug_return_pair),
        )
        self._set_optional_config_value(
            payload,
            ('max_changed_pixels_ratio',),
            float(self.tech_aug_max_changed_pixels_ratio_spinbox.value()),
            float(defaults.max_changed_pixels_ratio),
        )
        self._set_optional_config_value(
            payload,
            ('max_foreground_ratio_delta',),
            float(self.tech_aug_max_foreground_ratio_delta_spinbox.value()),
            float(defaults.max_foreground_ratio_delta),
        )
        self._set_optional_config_value(
            payload,
            ('global_width', 'probability'),
            float(self.tech_aug_global_width_probability_spinbox.value()),
            float(defaults.global_width.probability),
        )
        self._set_optional_config_value(
            payload,
            ('scale_rethreshold', 'probability'),
            float(self.tech_aug_scale_rethreshold_probability_spinbox.value()),
            float(defaults.scale_rethreshold.probability),
        )
        self._set_optional_config_value(
            payload,
            ('blur_threshold', 'probability'),
            float(self.tech_aug_blur_threshold_probability_spinbox.value()),
            float(defaults.blur_threshold.probability),
        )
        self._set_optional_config_value(
            payload,
            ('boundary_aware', 'probability'),
            float(self.tech_aug_boundary_aware_probability_spinbox.value()),
            float(defaults.boundary_aware.probability),
        )
        self._set_optional_config_value(
            payload,
            ('local_morphology', 'probability'),
            float(self.tech_aug_local_morphology_probability_spinbox.value()),
            float(defaults.local_morphology.probability),
        )
        self._set_optional_config_value(
            payload,
            ('gap_variation', 'probability'),
            float(self.tech_aug_gap_variation_probability_spinbox.value()),
            float(defaults.gap_variation.probability),
        )
        for key, checkbox in self.tech_augmentation_operation_checkboxes.items():
            self._set_optional_config_value(
                payload,
                (key, 'enabled'),
                bool(checkbox.isChecked()),
                False,
            )
        payload['enabled'] = any(
            checkbox.isChecked() for checkbox in self.tech_augmentation_operation_checkboxes.values()
        )
        self._tech_aug_config_payload = self._clone_plain_payload(payload)
        return payload
