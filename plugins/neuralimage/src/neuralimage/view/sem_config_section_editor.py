from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QWidget,
)

from neuralimage.configuration import (
    SemUiField,
    fields_for_section,
    sem_ui_field_help,
    sem_ui_field_label,
)
from neuralimage.view.settings_panel_widgets import (
    NoWheelComboBox,
    create_double_spinbox,
    create_spinbox,
)


class SemConfigSectionEditor(QWidget):
    """Registry-driven editor shared by SettingsPanel and augmentation preview."""

    changed = pyqtSignal()

    def __init__(
        self,
        section: str,
        parent: QWidget | None = None,
        *,
        control_factory: Callable[[SemUiField], QWidget] | None = None,
    ) -> None:
        super().__init__(parent)
        self.section = str(section)
        self.fields = fields_for_section(self.section)
        self.controls: dict[str, QWidget] = {}
        self.labels: dict[str, QLabel] = {}
        self.error_labels: dict[str, QLabel] = {}
        self._row_widgets: dict[str, QWidget] = {}
        self._update_guard = False
        self._control_factory = control_factory

        self.form = QFormLayout(self)
        self.form.setContentsMargins(0, 0, 0, 0)
        self.form.setSpacing(6)
        self.form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        for field in self.fields:
            self._add_field(field)

    def _add_field(self, field: SemUiField) -> None:
        factory = self._control_factory or self.create_control
        control = factory(field)
        label = QLabel(sem_ui_field_label(field, 'en'))
        label.setWordWrap(True)
        tooltip = sem_ui_field_help(field, 'en')
        label.setToolTip(tooltip)
        control.setToolTip(tooltip)
        control.setProperty('baseToolTip', tooltip)

        row_widget = QWidget(self)
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(4)
        row_layout.addWidget(control, 1)
        error_label = QLabel('')
        error_label.setStyleSheet('color: #ef5350; font-weight: 600;')
        error_label.setVisible(False)
        row_layout.addWidget(error_label)
        self.form.addRow(label, row_widget)

        self.controls[field.key] = control
        self.labels[field.key] = label
        self.error_labels[field.key] = error_label
        self._row_widgets[field.key] = row_widget
        self._connect_change_signal(control, field)

    def _connect_change_signal(self, control: QWidget, field: SemUiField) -> None:
        del field
        if isinstance(control, QCheckBox):
            control.toggled.connect(self._emit_changed)
            return
        if isinstance(control, (QSpinBox, QDoubleSpinBox)):
            control.valueChanged.connect(self._emit_changed)
            return
        if isinstance(control, QComboBox):
            control.currentIndexChanged.connect(self._emit_changed)
            return
        if isinstance(control, QLineEdit):
            control.textChanged.connect(self._emit_changed)
            return
        raise TypeError(f'Unsupported SEM config control type: {type(control).__name__}')

    def _emit_changed(self, *_args: Any) -> None:
        if not self._update_guard:
            self.changed.emit()

    def create_control(self, field: SemUiField) -> QWidget:
        return self.build_control(field, self)

    @staticmethod
    def build_control(field: SemUiField, parent: QWidget | None = None) -> QWidget:
        if field.kind == 'bool':
            control: QWidget = QCheckBox('', parent)
        elif field.kind == 'int':
            control = create_spinbox(
                (int(field.minimum), int(field.maximum)),
                step=int(field.step or 1),
                default_value=int(field.default),
            )
            if parent is not None:
                control.setParent(parent)
        elif field.kind == 'float':
            control = create_double_spinbox(
                (float(field.minimum), float(field.maximum)),
                step=float(field.step or 0.01),
                default_value=float(field.default),
                decimals=int(field.decimals),
            )
            if parent is not None:
                control.setParent(parent)
        elif field.kind == 'choice':
            combo = NoWheelComboBox(parent)
            for value, label in field.choices:
                combo.addItem(label, value)
            control = combo
        else:
            control = QLineEdit(parent)
        control.setProperty('semFieldKey', field.key)
        return control

    def set_form_values(self, values: Mapping[str, Any]) -> None:
        self._update_guard = True
        try:
            for field in self.fields:
                self.set_control_value(
                    field,
                    self.controls[field.key],
                    values.get(field.form_name, field.default),
                )
        finally:
            self._update_guard = False

    def form_values(self) -> dict[str, Any]:
        return {
            field.form_name: self.control_value(field, self.controls[field.key])
            for field in self.fields
        }

    def set_field_visible(self, key: str, visible: bool) -> None:
        if key in self.labels:
            self.labels[key].setVisible(bool(visible))
        if key in self._row_widgets:
            self._row_widgets[key].setVisible(bool(visible))

    def set_language(self, language: str) -> None:
        for field in self.fields:
            label = self.labels[field.key]
            control = self.controls[field.key]
            label.setText(sem_ui_field_label(field, language))
            tooltip = sem_ui_field_help(field, language)
            label.setToolTip(tooltip)
            control.setToolTip(tooltip)

    @staticmethod
    def control_value(field: SemUiField, control: QWidget) -> Any:
        del field
        if isinstance(control, QCheckBox):
            return bool(control.isChecked())
        if isinstance(control, (QSpinBox, QDoubleSpinBox)):
            return control.value()
        if isinstance(control, QComboBox):
            return control.currentData()
        if isinstance(control, QLineEdit):
            return str(control.text()).strip()
        raise TypeError(f'Unsupported SEM config control type: {type(control).__name__}')

    @staticmethod
    def set_control_value(field: SemUiField, control: QWidget, value: Any) -> None:
        if isinstance(control, QCheckBox):
            control.setChecked(bool(value))
        elif isinstance(control, (QSpinBox, QDoubleSpinBox)):
            if field.kind == 'int':
                control.setValue(int(value))
            else:
                control.setValue(float(value))
        elif isinstance(control, QComboBox):
            index = control.findData(str(value))
            control.setCurrentIndex(index if index >= 0 else 0)
        elif isinstance(control, QLineEdit):
            control.setText(str(value or ''))
        else:
            raise TypeError(f'Unsupported SEM config control type: {type(control).__name__}')

