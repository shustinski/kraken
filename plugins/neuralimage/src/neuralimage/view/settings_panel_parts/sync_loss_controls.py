"""settings_panel: sync loss controls."""
from __future__ import annotations
from typing import Any
from collections.abc import Mapping
from neuralimage.lib.loss_config import (
    MAX_LOSS_TERM_WEIGHT_SUM,
    format_loss_formula_html,
    loss_term_weight_sum,
)
from neuralimage.view.settings_panel_widgets import (
    SlidingPanel,  # noqa: F401 - compatibility re-export from neuralimage.view
    )
from neuralimage.configuration import (
    SEM_UI_FIELDS,
    build_sem_segmentation_config,
    sem_config_from_form_values,
    sem_config_to_form_values,
)

from neuralimage.view.settings_panel_parts.module_state import LOSS_FUNCTIONS


class SyncLossControlsMixin:
    def _sync_legacy_uncertainty_adapters_from_sem(self) -> None:
        if self._legacy_uncertainty_sync_guard:
            return
        controls = self.sem_segmentation_controls
        if 'uncertainty_enabled' not in controls:
            return
        enabled = bool(controls['uncertainty_enabled'].isChecked())
        method = str(controls['uncertainty_method'].currentData() or 'confidence_head')
        export = enabled and bool(controls['uncertainty_export'].isChecked())
        uses_tta = enabled and method in {'tta_variance', 'combined', 'auto'}
        export_mode = 'off'
        if export:
            export_mode = 'tta' if uses_tta else 'model_output'
        self._legacy_uncertainty_sync_guard = True
        try:
            index = self.confidence_save_mode_combo.findData(export_mode)
            if index >= 0:
                self.confidence_save_mode_combo.setCurrentIndex(index)
        finally:
            self._legacy_uncertainty_sync_guard = False

    def _sync_inference_uncertainty_from_legacy(self, *_args: Any) -> None:
        if self._legacy_uncertainty_sync_guard or self._sem_segmentation_update_guard:
            return
        controls = self.sem_segmentation_controls
        if 'uncertainty_enabled' not in controls:
            return
        export_mode = str(self.confidence_save_mode_combo.currentData() or 'off')
        uses_tta = export_mode == 'tta'
        enabled = uses_tta or export_mode != 'off'
        method = 'tta_variance' if uses_tta else 'confidence_head'
        self._legacy_uncertainty_pending = True
        self._legacy_uncertainty_sync_guard = True
        try:
            controls['uncertainty_enabled'].setChecked(enabled)
            method_index = controls['uncertainty_method'].findData(method)
            if method_index >= 0:
                controls['uncertainty_method'].setCurrentIndex(method_index)
            controls['uncertainty_export'].setChecked(export_mode != 'off')
        finally:
            self._legacy_uncertainty_sync_guard = False
        self._sync_sem_segmentation_controls()
        if hasattr(self, 'configuration_status_button'):
            self._validate_configuration_live()

    def set_sem_segmentation_config(self, payload: Mapping[str, Any] | None) -> None:
        preserve_legacy_uncertainty = not bool(payload) and self._legacy_uncertainty_pending
        canonical = build_sem_segmentation_config(payload)
        values = sem_config_to_form_values(canonical.to_dict())
        self._sem_segmentation_update_guard = True
        try:
            for field in SEM_UI_FIELDS:
                self._set_sem_segmentation_control_value(
                    field,
                    self.sem_segmentation_controls[field.key],
                    values[field.form_name],
                )
            self.preprocessing_groupbox.setChecked(canonical.preprocessing.mode != 'none')
        finally:
            self._sem_segmentation_update_guard = False
        if preserve_legacy_uncertainty:
            self._sync_inference_uncertainty_from_legacy()
        else:
            self._sync_legacy_uncertainty_adapters_from_sem()
        self._legacy_uncertainty_pending = False
        self._sync_sem_segmentation_controls()
        if hasattr(self, 'configuration_status_button'):
            self._validate_configuration_live()

    def get_sem_segmentation_config(self) -> dict[str, Any]:
        values = {
            field.form_name: self._sem_segmentation_control_value(
                field,
                self.sem_segmentation_controls[field.key],
            )
            for field in SEM_UI_FIELDS
        }
        if not self.preprocessing_groupbox.isChecked():
            values['sem__pre_mode'] = 'none'
        if not self.hard_mining_check_box.isChecked():
            values['sem__hard_mode'] = 'off'
        elif values.get('sem__hard_mode') == 'off':
            values['sem__hard_mode'] = 'online'
        return sem_config_from_form_values(values, preset='custom')

    def _validate_sem_segmentation_settings(self) -> None:
        self._validate_configuration_live()

    def _sync_hard_mining_controls(self, enabled: bool) -> None:
        pixel_control_enabled = self._training_controls_applicable and bool(self.hard_pixel_mining_check_box.isChecked())
        self._set_field_enabled(self.hard_pixel_mining_ratio_spinbox, pixel_control_enabled)
        sem_groups = getattr(self, 'sem_segmentation_section_groupboxes', {})
        sem_group = sem_groups.get('hard_mining')
        if sem_group is not None:
            sem_group.setEnabled(self._training_controls_applicable and bool(enabled))
        hard_mode = getattr(self, 'sem_segmentation_controls', {}).get('hard_mode')
        if hard_mode is not None:
            desired_mode = None
            if enabled and hard_mode.currentData() == 'off':
                desired_mode = 'online'
            elif not enabled:
                desired_mode = 'off'
            if desired_mode is not None:
                index = hard_mode.findData(desired_mode)
                if index >= 0 and index != hard_mode.currentIndex():
                    hard_mode.setCurrentIndex(index)

    def _sync_loss_controls(self, _index: int | None = None) -> None:
        total = loss_term_weight_sum(self.get_loss_term_weights())
        for loss_name in LOSS_FUNCTIONS:
            checkbox = self.loss_term_checkboxes[loss_name]
            spinbox = self.loss_term_spinboxes[loss_name]
            label = self.loss_term_labels[loss_name]
            spinbox.setMaximum(self._max_loss_weight_for(loss_name))
            checkbox.setEnabled(self._training_controls_applicable)
            label.setEnabled(self._training_controls_applicable)
            spinbox.setEnabled(self._training_controls_applicable and checkbox.isChecked())
        topograph_enabled = bool(getattr(self, 'topograph_enabled_check_box', None) and self.topograph_enabled_check_box.isChecked())
        if hasattr(self, 'topograph_enabled_check_box'):
            self.topograph_enabled_check_box.setEnabled(self._training_controls_applicable)
            self.topograph_term_label.setEnabled(self._training_controls_applicable)
            self.topograph_loss_weight_spinbox.setEnabled(
                self._training_controls_applicable and topograph_enabled
            )
            self.topograph_debug_viz_check_box.setEnabled(
                self._training_controls_applicable and topograph_enabled
            )
            self.topograph_debug_viz_label.setEnabled(
                self._training_controls_applicable and topograph_enabled
            )
        self.loss_formula_label.setEnabled(self._training_controls_applicable)
        formula_html = format_loss_formula_html(self.get_loss_term_weights())
        total_html = (
            "<span style=\"color:#666; font-size:11px;\">"
            f"&Sigma;&nbsp;n<sub>i</sub> = {total:.2f}/{MAX_LOSS_TERM_WEIGHT_SUM:.2f}"
            "</span>"
        )
        topograph_html = ''
        if topograph_enabled:
            topograph_weight = float(self.topograph_loss_weight_spinbox.value())
            if topograph_weight > 0.0:
                topograph_html = (
                    '<br><span style="font-family:\'Cambria Math\',\'Times New Roman\',serif; font-size:16px;">'
                    f'+ {topograph_weight:.2f}&middot;<i>L</i><sub>Topograph</sub>'
                    '</span>'
                )
        self.loss_formula_label.setText(f'{formula_html}<br>{total_html}{topograph_html}')
        self._sync_active_loss_preset()

    def _sync_early_stopping_controls(self, enabled: bool) -> None:
        control_enabled = self._training_controls_applicable and bool(enabled)
        self._set_field_visible(self.epochs_spinbox, not control_enabled)
        self._set_field_enabled(self.epochs_spinbox, self._training_controls_applicable and not control_enabled)
        show_warning = control_enabled and not self.validation_check_box.isChecked()
        warning_text = (
            'Без validation используется фиксированный train-control set. '
            'Этот режим не позволяет объективно определить переобучение.'
        )
        self.early_stopping_control_warning.setText(warning_text)
        self.early_stopping_control_warning.setVisible(show_warning)

    def _sync_random_patch_size_controls(self, _enabled: bool | None = None) -> None:
        online_mode = self._training_controls_applicable
        enabled = online_mode and bool(self.random_patch_size_check_box.isChecked())
        self.random_patch_size_check_box.setEnabled(online_mode)
        if enabled and self.sync_patch_sizes_check_box.isChecked():
            self.sync_patch_sizes_check_box.setChecked(False)
        if enabled and self.torch_compile_check_box.isChecked():
            self.torch_compile_check_box.setChecked(False)
        self.sync_patch_sizes_check_box.setEnabled(self._recognition_controls_applicable and not enabled)
        self.torch_compile_check_box.setEnabled(self._training_controls_applicable and not enabled)
        self._set_field_visible(self.random_patch_min_size_widget, enabled)
        self._set_field_visible(self.random_patch_max_size_widget, enabled)
        self._set_field_enabled(self.random_patch_min_size_widget, enabled)
        self._set_field_enabled(self.random_patch_max_size_widget, enabled)
        self._sync_patch_size_controls()
