"""settings_panel: training augmentation config."""
from __future__ import annotations
from typing import Any
from collections.abc import Mapping
from neuralimage.view.settings_panel_widgets import (
    SlidingPanel,  # noqa: F401 - compatibility re-export from neuralimage.view
    )


class TrainingAugmentationConfigMixin:
    def get_training_augmentation_config(self) -> dict[str, dict[str, float | bool]]:
        rows = {
            'rotate_90': (self.horizontal_rotation, self.horizontal_rotation_probability_spinbox),
            'rotate_180': (self.vertical_rotation, self.vertical_rotation_probability_spinbox),
            'flip_x': (self.flip_x, self.flip_x_probability_spinbox),
            'flip_y': (self.flip_y, self.flip_y_probability_spinbox),
            'scale': (self.scale_augmentation_check_box, self.scale_augmentation_probability_spinbox),
            'brightness': (self.brightness_augmentation_check_box, self.augmentation_brightness_probability_spinbox),
            'contrast': (self.contrast_augmentation_check_box, self.augmentation_contrast_probability_spinbox),
            'gamma': (self.gamma_augmentation_check_box, self.augmentation_gamma_probability_spinbox),
            'noise': (self.noise_augmentation_check_box, self.augmentation_noise_probability_spinbox),
            'blur': (self.blur_augmentation_check_box, self.augmentation_blur_probability_spinbox),
        }
        return {
            key: {'enabled': bool(enabled.isChecked()), 'probability': float(probability.value())}
            for key, (enabled, probability) in rows.items()
        }

    def set_training_augmentation_config(self, payload: Mapping[str, Any] | None) -> None:
        if not payload:
            legacy_enabled = self.additional_augmentation_check_box.isChecked()
            for checkbox in (
                self.brightness_augmentation_check_box,
                self.contrast_augmentation_check_box,
                self.gamma_augmentation_check_box,
                self.noise_augmentation_check_box,
                self.blur_augmentation_check_box,
            ):
                checkbox.setChecked(legacy_enabled)
            self.training_augmentation_editor.sync_from_controls()
            return
        rows = {
            'rotate_90': (self.horizontal_rotation, self.horizontal_rotation_probability_spinbox),
            'rotate_180': (self.vertical_rotation, self.vertical_rotation_probability_spinbox),
            'flip_x': (self.flip_x, self.flip_x_probability_spinbox),
            'flip_y': (self.flip_y, self.flip_y_probability_spinbox),
            'scale': (self.scale_augmentation_check_box, self.scale_augmentation_probability_spinbox),
            'brightness': (self.brightness_augmentation_check_box, self.augmentation_brightness_probability_spinbox),
            'contrast': (self.contrast_augmentation_check_box, self.augmentation_contrast_probability_spinbox),
            'gamma': (self.gamma_augmentation_check_box, self.augmentation_gamma_probability_spinbox),
            'noise': (self.noise_augmentation_check_box, self.augmentation_noise_probability_spinbox),
            'blur': (self.blur_augmentation_check_box, self.augmentation_blur_probability_spinbox),
        }
        for key, (enabled, probability) in rows.items():
            item = payload.get(key)
            if not isinstance(item, Mapping):
                continue
            if 'enabled' in item:
                enabled.setChecked(bool(item['enabled']))
            if 'probability' in item:
                probability.setValue(min(1.0, max(0.0, float(item['probability']))))
        self.training_augmentation_editor.sync_from_controls()
