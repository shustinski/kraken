"""settings_panel: reset training augmentation defaults."""
from __future__ import annotations
from neuralimage.view.settings_panel_widgets import (
    SlidingPanel,  # noqa: F401 - compatibility re-export from neuralimage.view
    )
from neuralimage.configuration import (
    fields_for_section,
)


class ResetTrainingAugmentationDefaultsMixin:
    def _reset_training_augmentation_defaults(self) -> None:
        defaults = {
            self.horizontal_rotation: True,
            self.vertical_rotation: True,
            self.flip_x: False,
            self.flip_y: False,
            self.scale_augmentation_check_box: False,
            self.photometric_groupbox: False,
            self.brightness_augmentation_check_box: False,
            self.contrast_augmentation_check_box: False,
            self.gamma_augmentation_check_box: False,
            self.noise_augmentation_check_box: False,
            self.blur_augmentation_check_box: False,
            self.cutout_groupbox: False,
            self.random_artifacts_groupbox: False,
            self.mixup_groupbox: False,
        }
        for control, checked in defaults.items():
            control.setChecked(checked)
        values = {
            self.horizontal_rotation_probability_spinbox: 1.0,
            self.vertical_rotation_probability_spinbox: 1.0,
            self.flip_x_probability_spinbox: 1.0,
            self.flip_y_probability_spinbox: 1.0,
            self.scale_augmentation_probability_spinbox: 1.0,
            self.scale_augmentation_strength_spinbox: 0.2,
            self.augmentation_brightness_probability_spinbox: 1.0,
            self.augmentation_brightness_spinbox: 0.1,
            self.augmentation_contrast_probability_spinbox: 1.0,
            self.augmentation_contrast_spinbox: 0.1,
            self.augmentation_gamma_probability_spinbox: 1.0,
            self.augmentation_gamma_spinbox: 0.15,
            self.augmentation_noise_probability_spinbox: 0.5,
            self.augmentation_noise_sigma_spinbox: 0.01,
            self.augmentation_blur_probability_spinbox: 0.25,
            self.augmentation_blur_radius_spinbox: 1.0,
            self.cutout_probability_spinbox: 1.0,
            self.cutout_holes_spinbox: 1,
            self.cutout_size_ratio_spinbox: 0.25,
            self.random_artifacts_probability_spinbox: 1.0,
            self.random_artifacts_count_spinbox: 1,
            self.random_artifacts_size_ratio_spinbox: 0.25,
            self.mixup_probability_spinbox: 1.0,
            self.mixup_alpha_spinbox: 0.2,
        }
        for control, value in values.items():
            control.setValue(value)
        for field in fields_for_section('augmentation'):
            self._set_sem_segmentation_control_value(
                field, self.sem_segmentation_controls[field.key], field.default
            )
        self._sync_sem_segmentation_controls()
        self.set_tech_aug_config(None)
