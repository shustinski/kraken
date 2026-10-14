"""settings_panel: confidence export mode value."""
from __future__ import annotations
from neuralimage.view.settings_panel_widgets import (
    NoWheelComboBox,
    SlidingPanel,  # noqa: F401 - compatibility re-export from neuralimage.view
    )

from neuralimage.view.settings_panel_parts.module_state import CONFIDENCE_SAVE_MODES


class ConfidenceExportModeValueMixin:
    def _sync_recognition_output_controls(self, _enabled: bool | None = None) -> None:
        recognition_enabled = self._recognition_controls_applicable
        binarize_output = recognition_enabled and bool(self.recognition_binarize_output_check_box.isChecked())
        auto_threshold = binarize_output and bool(self.recognition_use_auto_threshold_check_box.isChecked())
        postprocess_enabled = binarize_output and bool(self.recognition_postprocess_check_box.isChecked())

        self.recognition_multiprocessing_check_box.setEnabled(recognition_enabled)
        self.recognition_binarize_output_check_box.setEnabled(recognition_enabled)
        self.recognition_use_auto_threshold_check_box.setEnabled(binarize_output)
        self._set_field_enabled(self.recognition_threshold_spinbox, binarize_output and not auto_threshold)
        self.recognition_tta_check_box.setEnabled(recognition_enabled)
        self._set_field_enabled(self.confidence_save_mode_combo, recognition_enabled)
        self.recognition_postprocess_check_box.setEnabled(binarize_output)
        self._set_field_enabled(self.recognition_postprocess_kernel_size_spinbox, postprocess_enabled)

    def get_confidence_export_mode_value(self) -> str:
        controls = getattr(self, 'sem_segmentation_controls', {})
        if 'uncertainty_enabled' in controls:
            enabled = bool(controls['uncertainty_enabled'].isChecked())
            export = bool(controls['uncertainty_export'].isChecked())
            if not enabled or not export:
                return 'off'
            method = str(controls['uncertainty_method'].currentData() or 'confidence_head')
            return 'tta' if method in {'tta_variance', 'combined', 'auto'} else 'model_output'
        normalized = str(self.confidence_save_mode_combo.currentData() or self.confidence_save_mode_combo.currentText() or 'off')
        normalized = normalized.strip().lower()
        if normalized in CONFIDENCE_SAVE_MODES:
            return normalized
        if normalized == 'separate_grayscale':
            return 'model_output'
        return 'off'

    def get_confidence_save_mode_value(self) -> str:
        export_mode = self.get_confidence_export_mode_value()
        if export_mode == 'off':
            return 'off'
        return 'separate_grayscale'

    def is_confidence_tta_enabled(self) -> bool:
        controls = getattr(self, 'sem_segmentation_controls', {})
        if 'uncertainty_enabled' in controls:
            enabled = bool(controls['uncertainty_enabled'].isChecked())
            method = str(controls['uncertainty_method'].currentData() or 'confidence_head')
            return enabled and method in {'tta_variance', 'combined', 'auto'}
        return self.get_confidence_export_mode_value() == 'tta'

    def set_confidence_output_mode(self, confidence_save_mode: str | None, confidence_tta_enabled: bool = False) -> None:
        normalized_save_mode = str(confidence_save_mode or 'off').strip().lower()
        if normalized_save_mode in CONFIDENCE_SAVE_MODES:
            self.set_confidence_export_mode_value(normalized_save_mode)
            return
        if normalized_save_mode == 'off':
            self.set_confidence_export_mode_value('off')
            return
        if confidence_tta_enabled:
            self.set_confidence_export_mode_value('tta')
            return
        self.set_confidence_export_mode_value('model_output')

    def set_confidence_save_mode_value(self, value: str) -> None:
        normalized = str(value or 'off').strip().lower()
        if normalized in CONFIDENCE_SAVE_MODES:
            self.set_confidence_export_mode_value(normalized)
            return
        if normalized == 'off':
            self.set_confidence_export_mode_value('off')
            return
        self.set_confidence_export_mode_value('model_output')

    def set_confidence_export_mode_value(self, value: str) -> None:
        normalized = str(value or 'off').strip().lower()
        index = self.confidence_save_mode_combo.findData(normalized)
        if index < 0:
            index = self.confidence_save_mode_combo.findText(normalized)
        if index < 0:
            index = 0
        self.confidence_save_mode_combo.setCurrentIndex(index)

    @staticmethod
    def get_combo_value(combo: NoWheelComboBox) -> str:
        return str(combo.currentData() or combo.currentText() or '')

    @staticmethod
    def set_combo_value(combo: NoWheelComboBox, value: str) -> None:
        normalized = str(value or '').strip().lower()
        index = combo.findData(normalized)
        if index < 0:
            index = combo.findText(normalized)
        if index < 0:
            index = 0
        combo.setCurrentIndex(index)

    def get_synthetic_topology_domain_value(self) -> str:
        return self.get_combo_value(self.synthetic_topology_domain_combo) or 'pcb'

    def set_synthetic_topology_domain_value(self, value: str) -> None:
        self.set_combo_value(self.synthetic_topology_domain_combo, value)

    def _sync_pcb_defect_count_bounds(self, _value: int | None = None) -> None:
        min_defects = int(self.pcb_defects_min_count_spinbox.value())
        max_defects = int(self.pcb_defects_max_count_spinbox.value())
        if min_defects <= max_defects:
            return
        if self.sender() is self.pcb_defects_min_count_spinbox:
            self.pcb_defects_max_count_spinbox.setValue(min_defects)
        else:
            self.pcb_defects_min_count_spinbox.setValue(max_defects)
