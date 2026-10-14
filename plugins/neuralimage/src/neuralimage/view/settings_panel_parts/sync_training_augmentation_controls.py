"""settings_panel: sync training augmentation controls."""
from __future__ import annotations
from neuralimage.view.settings_panel_widgets import (
    SlidingPanel,  # noqa: F401 - compatibility re-export from neuralimage.view
    )

from neuralimage.view.settings_panel_parts.module_state import PCB_DEFECT_WEIGHT_FIELDS


class SyncTrainingAugmentationControlsMixin:
    def _sync_training_augmentation_controls(self, _enabled: bool | None = None) -> None:
        training_enabled = self._training_controls_applicable
        self.cutout_check_box.setEnabled(training_enabled)
        self.random_artifacts_check_box.setEnabled(training_enabled)
        self.cutout_groupbox.setEnabled(training_enabled)
        random_artifacts_enabled = training_enabled and bool(self.random_artifacts_check_box.isChecked())
        self.random_artifacts_groupbox.setEnabled(training_enabled)
        for checkbox in self.random_artifact_type_checkboxes.values():
            checkbox.setEnabled(random_artifacts_enabled)
            checkbox.setVisible(random_artifacts_enabled)
        self.mixup_check_box.setEnabled(training_enabled)
        self.mixup_groupbox.setEnabled(training_enabled)
        pcb_defects_enabled = training_enabled and bool(self.pcb_defects_check_box.isChecked())
        self.pcb_defects_groupbox.setEnabled(training_enabled)
        self.pcb_defects_check_box.setEnabled(training_enabled)
        self.pcb_defects_use_input_mask_check_box.setEnabled(pcb_defects_enabled)
        self.pcb_defects_use_defect_mask_as_label_check_box.setEnabled(pcb_defects_enabled)
        self._set_field_enabled(
            self.cutout_probability_spinbox,
            training_enabled and bool(self.cutout_check_box.isChecked()),
        )
        cutout_enabled = training_enabled and bool(self.cutout_check_box.isChecked())
        for field in (self.cutout_probability_spinbox, self.cutout_holes_spinbox, self.cutout_size_ratio_spinbox):
            self._set_field_visible(field, cutout_enabled)
        for field in (
            self.random_artifacts_probability_spinbox,
            self.random_artifacts_count_spinbox,
            self.random_artifacts_size_ratio_spinbox,
        ):
            self._set_field_visible(field, random_artifacts_enabled)
        self._set_field_enabled(
            self.cutout_holes_spinbox,
            training_enabled and bool(self.cutout_check_box.isChecked()),
        )
        self._set_field_enabled(
            self.cutout_size_ratio_spinbox,
            training_enabled and bool(self.cutout_check_box.isChecked()),
        )
        self._set_field_enabled(
            self.random_artifacts_probability_spinbox,
            random_artifacts_enabled,
        )
        self._set_field_enabled(
            self.random_artifacts_count_spinbox,
            random_artifacts_enabled,
        )
        self._set_field_enabled(
            self.random_artifacts_size_ratio_spinbox,
            random_artifacts_enabled,
        )
        self._set_field_enabled(
            self.mixup_probability_spinbox,
            training_enabled and bool(self.mixup_check_box.isChecked()),
        )
        self._set_field_enabled(
            self.mixup_alpha_spinbox,
            training_enabled and bool(self.mixup_check_box.isChecked()),
        )
        self._set_field_enabled(self.pcb_defects_probability_spinbox, pcb_defects_enabled)
        self._set_field_enabled(self.pcb_defects_min_count_spinbox, pcb_defects_enabled)
        self._set_field_enabled(self.pcb_defects_max_count_spinbox, pcb_defects_enabled)
        for defect_name, _label_key in PCB_DEFECT_WEIGHT_FIELDS:
            self._set_field_enabled(self.pcb_defect_type_spinboxes[defect_name], pcb_defects_enabled)
        self._sync_synthetic_defect_generator_controls(self.synthetic_defect_generator_check_box.isChecked())
