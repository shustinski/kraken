"""settings_panel: sync basic photometric master."""
from __future__ import annotations
from neuralimage.view.settings_panel_widgets import (
    SlidingPanel,  # noqa: F401 - compatibility re-export from neuralimage.view
    )


class SyncBasicPhotometricMasterMixin:
    def _sync_basic_photometric_master(self, *_args: object) -> None:
        enabled = any(
            checkbox.isChecked()
            for checkbox in (
                self.brightness_augmentation_check_box,
                self.contrast_augmentation_check_box,
                self.gamma_augmentation_check_box,
                self.noise_augmentation_check_box,
                self.blur_augmentation_check_box,
            )
        )
        if self.additional_augmentation_check_box.isChecked() != enabled:
            self.additional_augmentation_check_box.setChecked(enabled)
        if hasattr(self, 'training_augmentation_editor'):
            self.training_augmentation_editor.sync_from_controls()

    def _sync_preprocess_controls(self, _enabled: bool | None = None) -> None:
        self._set_field_enabled(
            self.cut_corner_spinbox,
            self._training_controls_applicable and self.enable_crop_processing.isChecked(),
        )
        self._set_field_enabled(
            self.compression_factor_spinbox,
            self._training_controls_applicable and self.enable_resize_processing.isChecked(),
        )
