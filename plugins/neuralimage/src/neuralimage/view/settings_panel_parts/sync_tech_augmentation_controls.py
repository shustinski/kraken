"""settings_panel: sync tech augmentation controls."""
from __future__ import annotations
from neuralimage.view.settings_panel_widgets import (
    SlidingPanel,  # noqa: F401 - compatibility re-export from neuralimage.view
    )


class SyncTechAugmentationControlsMixin:
    def _sync_tech_augmentation_controls(self, _enabled: bool | None = None) -> None:
        training_enabled = self._training_controls_applicable
        tech_enabled = training_enabled and bool(self.tech_augmentation_check_box.isChecked())
        self.tech_augmentation_check_box.setEnabled(training_enabled)
        self.tech_augmentation_debug_pair_check_box.setEnabled(tech_enabled)
        self._set_field_enabled(self.tech_aug_min_operations_spinbox, tech_enabled)
        self._set_field_enabled(self.tech_aug_max_operations_spinbox, tech_enabled)
        self._set_field_enabled(self.tech_aug_max_changed_pixels_ratio_spinbox, tech_enabled)
        self._set_field_enabled(self.tech_aug_max_foreground_ratio_delta_spinbox, tech_enabled)
        self._set_field_enabled(self.tech_aug_global_width_probability_spinbox, tech_enabled)
        self._set_field_enabled(self.tech_aug_scale_rethreshold_probability_spinbox, tech_enabled)
        self._set_field_enabled(self.tech_aug_blur_threshold_probability_spinbox, tech_enabled)
        self._set_field_enabled(self.tech_aug_boundary_aware_probability_spinbox, tech_enabled)
        self._set_field_enabled(self.tech_aug_local_morphology_probability_spinbox, tech_enabled)
        self._set_field_enabled(self.tech_aug_gap_variation_probability_spinbox, tech_enabled)

    def _sync_tech_operation_controls(self, *_args: object) -> None:
        enabled = any(
            checkbox.isChecked() for checkbox in self.tech_augmentation_operation_checkboxes.values()
        )
        if self.tech_augmentation_check_box.isChecked() != enabled:
            self.tech_augmentation_check_box.setChecked(enabled)
        self._sync_tech_augmentation_controls(enabled)
        if hasattr(self, 'training_augmentation_editor'):
            self.training_augmentation_editor.sync_from_controls()
