"""settings_panel: sync augmentation controls."""
from __future__ import annotations
from neuralimage.view.settings_panel_widgets import (
    SlidingPanel,  # noqa: F401 - compatibility re-export from neuralimage.view
    )


class SyncAugmentationControlsMixin:
    def _sync_augmentation_controls(self, enabled: bool) -> None:
        control_enabled = self._training_controls_applicable and bool(enabled)
        online_mode_enabled = self._training_controls_applicable
        random_crop_enabled = online_mode_enabled and bool(self.random_crop_check_box.isChecked())
        self.spatial_groupbox.setEnabled(self._training_controls_applicable)
        self.photometric_groupbox.setEnabled(self._training_controls_applicable)
        photometric_fields = (
            self.augmentation_brightness_spinbox,
            self.augmentation_contrast_spinbox,
            self.augmentation_gamma_spinbox,
            self.augmentation_noise_probability_spinbox,
            self.augmentation_noise_sigma_spinbox,
            self.augmentation_blur_probability_spinbox,
            self.augmentation_blur_radius_spinbox,
        )
        if hasattr(self, 'training_augmentation_editor'):
            for field in photometric_fields:
                field.setVisible(True)
        else:
            for field in photometric_fields:
                self._set_field_visible(field, control_enabled)
            if hasattr(self, '_photometric_noise_row'):
                self._photometric_noise_row.setVisible(control_enabled)
            if hasattr(self, '_photometric_blur_row'):
                self._photometric_blur_row.setVisible(control_enabled)
        self._set_field_enabled(self.augmentation_brightness_spinbox, control_enabled)
        self._set_field_enabled(self.augmentation_contrast_spinbox, control_enabled)
        self._set_field_enabled(self.augmentation_gamma_spinbox, control_enabled)
        self._set_field_enabled(self.augmentation_noise_probability_spinbox, control_enabled)
        self._set_field_enabled(self.augmentation_noise_sigma_spinbox, control_enabled)
        self._set_field_enabled(self.augmentation_blur_probability_spinbox, control_enabled)
        self._set_field_enabled(
            self.augmentation_blur_radius_spinbox,
            control_enabled and float(self.augmentation_blur_probability_spinbox.value()) > 0.0,
        )
        self.random_crop_check_box.setEnabled(online_mode_enabled)
        self.scale_augmentation_check_box.setEnabled(online_mode_enabled)
        self._set_field_enabled(self.crops_per_image_spinbox, random_crop_enabled)
        self._set_field_enabled(
            self.shift_spinbox,
            self._training_controls_applicable and not random_crop_enabled,
        )
        if hasattr(self, 'grid_sampling_radio'):
            self._sync_patch_sampling_mode()
        self._set_field_enabled(
            self.scale_augmentation_strength_spinbox,
            online_mode_enabled and bool(self.scale_augmentation_check_box.isChecked()),
        )
        if hasattr(self, 'training_augmentation_editor'):
            self.training_augmentation_editor.sync_from_controls()
        self._sync_tech_augmentation_controls(self.tech_augmentation_check_box.isChecked())
        self._sync_synthetic_defect_generator_controls(self.synthetic_defect_generator_check_box.isChecked())
        self._sync_training_augmentation_controls()
