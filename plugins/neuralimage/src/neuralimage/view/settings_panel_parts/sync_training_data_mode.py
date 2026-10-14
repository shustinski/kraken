"""settings_panel: sync training data mode."""
from __future__ import annotations
from typing import Iterable
from PyQt6.QtWidgets import (
    QWidget,
)
from neuralimage.lib.loss_config import (
    MAX_LOSS_TERM_WEIGHT_SUM,
    sanitize_loss_term_weights,
)
from neuralimage.view.settings_panel_policy import (
    resolve_work_mode_applicability as resolve_work_mode_applicability_policy,
)
from neuralimage.view.settings_panel_widgets import (
    SlidingPanel,  # noqa: F401 - compatibility re-export from neuralimage.view
    )

from neuralimage.view.settings_panel_parts.module_state import (
    LOSS_FUNCTIONS,
    LOSS_PRESET_WEIGHTS,
)


class SyncTrainingDataModeMixin:
    def _loss_term_total(self, *, exclude: str | None = None) -> float:
        total = 0.0
        for loss_name in LOSS_FUNCTIONS:
            if exclude is not None and loss_name == exclude:
                continue
            if not self.loss_term_checkboxes[loss_name].isChecked():
                continue
            total += float(self.loss_term_spinboxes[loss_name].value())
        return total

    def _max_loss_weight_for(self, loss_name: str) -> float:
        current_value = float(self.loss_term_spinboxes[loss_name].value())
        return max(0.0, MAX_LOSS_TERM_WEIGHT_SUM - (self._loss_term_total() - current_value))

    def _on_loss_term_toggled(self, loss_name: str, checked: bool) -> None:
        if self._loss_terms_guard:
            return
        spinbox = self.loss_term_spinboxes[loss_name]
        self._loss_terms_guard = True
        try:
            if checked:
                remaining = max(0.0, MAX_LOSS_TERM_WEIGHT_SUM - self._loss_term_total(exclude=loss_name))
                initial_value = float(spinbox.value())
                if initial_value <= 0.0 and remaining > 0.0:
                    initial_value = remaining if self._loss_term_total(exclude=loss_name) <= 0.0 else min(0.1, remaining)
                spinbox.setValue(min(initial_value, remaining))
            else:
                spinbox.setValue(0.0)
        finally:
            self._loss_terms_guard = False
        self._sync_loss_controls()

    def _on_loss_term_value_changed(self, loss_name: str) -> None:
        if self._loss_terms_guard:
            return
        if not self.loss_term_checkboxes[loss_name].isChecked():
            return
        spinbox = self.loss_term_spinboxes[loss_name]
        max_value = max(0.0, MAX_LOSS_TERM_WEIGHT_SUM - self._loss_term_total(exclude=loss_name))
        self._loss_terms_guard = True
        try:
            if float(spinbox.value()) > max_value:
                spinbox.setValue(max_value)
        finally:
            self._loss_terms_guard = False
        self._sync_loss_controls()

    def _on_topograph_enabled_toggled(self, checked: bool) -> None:
        self.topograph_loss_weight_spinbox.setEnabled(bool(checked))
        self.topograph_debug_viz_check_box.setEnabled(bool(checked))
        if checked and float(self.topograph_loss_weight_spinbox.value()) <= 0.0:
            self.topograph_loss_weight_spinbox.setValue(0.1)
        self._sync_loss_controls()

    def get_loss_term_weights(self) -> dict[str, float]:
        weights: dict[str, float] = {}
        for loss_name in LOSS_FUNCTIONS:
            if not self.loss_term_checkboxes[loss_name].isChecked():
                continue
            value = float(self.loss_term_spinboxes[loss_name].value())
            if value > 0.0:
                weights[loss_name] = value
        return sanitize_loss_term_weights(weights)

    def set_loss_term_weights(self, weights: dict[str, float] | None) -> None:
        sanitized = sanitize_loss_term_weights(weights)
        self._loss_terms_guard = True
        try:
            for loss_name in LOSS_FUNCTIONS:
                checkbox = self.loss_term_checkboxes[loss_name]
                spinbox = self.loss_term_spinboxes[loss_name]
                spinbox.setMaximum(MAX_LOSS_TERM_WEIGHT_SUM)
                checkbox.setChecked(loss_name in sanitized)
                spinbox.setValue(float(sanitized.get(loss_name, 0.0)))
        finally:
            self._loss_terms_guard = False
        self._sync_loss_controls()

    def _apply_loss_preset(self, preset_name: str) -> None:
        weights = LOSS_PRESET_WEIGHTS.get(str(preset_name))
        if weights is None:
            return
        self.set_loss_term_weights(dict(weights))
        self.optimizer_settings_changed.emit()

    def _set_field_enabled(self, field: QWidget, enabled: bool) -> None:
        row_widget = self._field_rows.get(field)
        if row_widget is not None:
            row_widget.setEnabled(enabled)
        field.setEnabled(enabled)

    def _set_field_visible(self, field: QWidget, visible: bool) -> None:
        row_widget = self._field_rows.get(field)
        if row_widget is not None:
            row_widget.setVisible(visible)
        field.setVisible(visible)

    def _set_fields_enabled(self, fields: Iterable[QWidget], enabled: bool) -> None:
        for field in fields:
            if isinstance(field, QWidget):
                self._set_field_enabled(field, enabled)

    @staticmethod
    def _set_widgets_enabled(widgets: Iterable[QWidget], enabled: bool) -> None:
        for widget in widgets:
            if isinstance(widget, QWidget):
                widget.setEnabled(enabled)

    @staticmethod
    def _resolve_work_mode_applicability(work_mode: str | None) -> tuple[bool, bool, bool]:
        applicability = resolve_work_mode_applicability_policy(work_mode)
        return applicability.training, applicability.recognition, applicability.model_selector

    def _set_work_mode_flags(
        self,
        training_applicable: bool,
        recognition_applicable: bool,
        model_selector_applicable: bool,
    ) -> None:
        self._training_controls_applicable = training_applicable
        self._recognition_controls_applicable = recognition_applicable
        self._model_selector_applicable = model_selector_applicable

    def _sync_general_mode_controls(
        self,
        training_applicable: bool,
        recognition_applicable: bool,
        model_selector_applicable: bool,
        batch_related: bool,
    ) -> None:
        self._set_widgets_enabled((self.samples_number,), training_applicable)
        self._set_widgets_enabled(
            (self.shuffle_frames_check_box, self.shuffle_patches_in_frame_check_box),
            training_applicable,
        )
        self._set_widgets_enabled((self.model_variants_groupbox,), model_selector_applicable)
        self._set_fields_enabled((self.epochs_spinbox,), training_applicable)
        self._set_fields_enabled(
            (
                self.nn_model_type,
                self.deprecated_model_type,
                self.experimental_model_type,
            ),
            model_selector_applicable,
        )
        self._set_fields_enabled((self.color_type,), training_applicable)
        self._set_fields_enabled((self.train_patch_size_widget,), batch_related)
        self._set_fields_enabled((self.recursive_file_search_check_box,), training_applicable or recognition_applicable)
        self._set_fields_enabled((self.sync_patch_sizes_check_box,), recognition_applicable)
        self._set_fields_enabled((self.recognition_patch_size_widget,), recognition_applicable)

    def _sync_training_data_mode_controls(self, training_applicable: bool) -> None:
        self._set_widgets_enabled(
            (
                self.shuffle_groupbox,
                self.horizontal_rotation,
                self.vertical_rotation,
                self.flip_x,
                self.flip_y,
                self.additional_augmentation_check_box,
                self.random_crop_check_box,
                self.scale_augmentation_check_box,
                self.synthetic_defect_generator_groupbox,
                self.synthetic_defect_generator_check_box,
                self.tech_augmentation_check_box,
                self.tech_augmentation_debug_pair_check_box,
                self.cutout_check_box,
                self.random_artifacts_check_box,
                *self.random_artifact_type_checkboxes.values(),
                self.mixup_check_box,
                self.pcb_defects_groupbox,
                self.pcb_defects_check_box,
                self.pcb_defects_use_input_mask_check_box,
                self.pcb_defects_use_defect_mask_as_label_check_box,
                self.validation_groupbox,
                self.validation_check_box,
                self.augmentation_preview_button,
                self.prepare_samples_groupbox,
                self.preprocessing_groupbox,
                self.enable_crop_processing,
                self.enable_resize_processing,
                self.compression_factor_spinbox,
            ),
            training_applicable,
        )
        self._set_fields_enabled((self.shift_spinbox,), training_applicable)
        self._sync_augmentation_controls(self.additional_augmentation_check_box.isChecked())
        self._sync_tech_augmentation_controls(self.tech_augmentation_check_box.isChecked())
        self._sync_synthetic_defect_generator_controls(self.synthetic_defect_generator_check_box.isChecked())
        self._sync_validation_controls(self.validation_check_box.isChecked())
        self._sync_preprocess_controls()
