"""settings_panel: preprocessing group toggled."""
from __future__ import annotations
from typing import Any
from PyQt6.QtWidgets import (
    QWidget,
)
from neuralimage.view.settings_panel_widgets import (
    SlidingPanel,  # noqa: F401 - compatibility re-export from neuralimage.view
    )
from neuralimage.configuration import (
    SemUiField,
)


class PreprocessingGroupToggledMixin:
    @staticmethod
    def _sem_segmentation_control_value(field: SemUiField, control: QWidget) -> Any:
        if field.kind == 'bool':
            return bool(control.isChecked())
        if field.kind in {'int', 'float'}:
            return control.value()
        if field.kind == 'choice':
            return control.currentData()
        return str(control.text()).strip()

    @staticmethod
    def _set_sem_segmentation_control_value(field: SemUiField, control: QWidget, value: Any) -> None:
        if field.kind == 'bool':
            control.setChecked(bool(value))
        elif field.kind == 'int':
            control.setValue(int(value))
        elif field.kind == 'float':
            control.setValue(float(value))
        elif field.kind == 'choice':
            index = control.findData(str(value))
            control.setCurrentIndex(index if index >= 0 else 0)
        else:
            control.setText(str(value or ''))

    def _on_sem_segmentation_control_changed(self, *_args: Any) -> None:
        if self._sem_segmentation_update_guard:
            return
        mode_control = self.sem_segmentation_controls.get('pre_mode')
        if mode_control is not None:
            preprocessing_enabled = str(mode_control.currentData() or 'none') != 'none'
            if self.preprocessing_groupbox.isChecked() != preprocessing_enabled:
                self.preprocessing_groupbox.setChecked(preprocessing_enabled)
        self._sync_sem_segmentation_controls()
        self._validate_configuration_live()

    def _on_preprocessing_group_toggled(self, enabled: bool) -> None:
        if self._sem_segmentation_update_guard:
            return
        mode_control = self.sem_segmentation_controls.get('pre_mode')
        if mode_control is None:
            return
        desired_mode = str(mode_control.currentData() or 'none')
        if not enabled:
            desired_mode = 'none'
        elif desired_mode == 'none':
            desired_mode = 'per_image_percentile'
        index = mode_control.findData(desired_mode)
        if index >= 0 and index != mode_control.currentIndex():
            mode_control.setCurrentIndex(index)
