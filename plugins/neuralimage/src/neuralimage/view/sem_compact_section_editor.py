from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from neuralimage.configuration import (
    SEM_UI_FIELDS_BY_KEY,
    fields_for_section,
    sem_ui_choice_label,
    sem_ui_field_help,
    sem_ui_field_label,
    sem_ui_section_help,
    sem_ui_section_label,
)
from neuralimage.configuration.sem_ui_compact_layout import SEM_UI_COMPACT_LAYOUTS, CompactRow
from neuralimage.view.sem_config_section_editor import SemConfigSectionEditor


class CompactSemSectionEditor(QGroupBox):
    """Registry-driven compact SEM section editor with optional master toggle in title."""

    changed = pyqtSignal()

    def __init__(self, section: str, parent: QWidget | None = None, *, language: str = 'en') -> None:
        super().__init__(parent)
        self.section = str(section)
        self.layout_spec = SEM_UI_COMPACT_LAYOUTS[self.section]
        self.fields = fields_for_section(self.section)
        self.controls: dict[str, QWidget] = {}
        self.labels: dict[str, QLabel] = {}
        self.error_labels: dict[str, QLabel] = {}
        self._row_widgets: dict[str, QWidget] = {}
        self._effect_rows: dict[str, QWidget] = {}
        self._update_guard = False
        self._language = str(language)

        if self.layout_spec.master_key is not None:
            self.setCheckable(True)
            self.setChecked(False)
            self._apply_master_title()
            self.toggled.connect(self._on_master_toggled)
        elif self.layout_spec.checkable:
            self.setCheckable(True)
            self.setChecked(False)
            self._apply_section_title()
        else:
            self._apply_section_title()

        content_layout = QVBoxLayout(self)
        content_layout.setContentsMargins(6, 6, 6, 6)
        content_layout.setSpacing(6)

        for row_spec in self.layout_spec.rows:
            content_layout.addWidget(self._build_row(row_spec))

        if self.layout_spec.master_key is not None:
            self.controls[self.layout_spec.master_key] = self
            self._sync_dependent_rows()

        for field in self.fields:
            if field.key in self.controls:
                continue
            control = SemConfigSectionEditor.build_control(field, self)
            control.setVisible(False)
            self.controls[field.key] = control
            self._connect_value_signal(control)

    def _apply_section_title(self) -> None:
        section_titles = {
            'preprocessing': 'Preprocessing',
            'augmentation': 'SEM augmentation',
            'basic_targets': 'Basic supervision',
            'geometry_targets': 'Geometry supervision',
            'losses': 'Loss weighting',
            'hard_mining': 'Hard example mining',
            'context': 'Context branch',
            'confidence_training': 'Confidence head training',
            'inference_uncertainty': 'Inference uncertainty',
            'active_learning': 'Active Learning export',
            'validation': 'Validation',
            'experiment': 'Experiment',
        }
        english_label = section_titles.get(self.section, self.section)
        self.setTitle(sem_ui_section_label(self.section, english_label, self._language))
        self.setToolTip(sem_ui_section_help(self.section, self._language))

    def _apply_master_title(self) -> None:
        master_key = self.layout_spec.master_key
        if master_key is None:
            return
        master_field = SEM_UI_FIELDS_BY_KEY[master_key]
        self.setTitle(sem_ui_field_label(master_field, self._language))
        self.setToolTip(sem_ui_field_help(master_field, self._language))

    def _build_row(self, row_spec: CompactRow) -> QWidget:
        if row_spec.kind == 'effect':
            return self._build_effect_row(row_spec)
        if row_spec.kind == 'bool_weight':
            return self._build_bool_weight_row(row_spec)
        if row_spec.kind == 'labeled':
            return self._build_labeled_row(row_spec.field_keys[0])
        return self._build_inline_row(row_spec.field_keys)

    def _build_effect_row(self, row_spec: CompactRow) -> QWidget:
        enable_key = str(row_spec.enable_key)
        probability_key = str(row_spec.probability_key)
        strength_keys = tuple(str(key) for key in row_spec.strength_keys)

        enable_field = SEM_UI_FIELDS_BY_KEY[enable_key]
        probability_field = SEM_UI_FIELDS_BY_KEY[probability_key]

        row = QWidget(self)
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        enable_checkbox = QCheckBox('', row)
        enable_checkbox.setToolTip(sem_ui_field_help(enable_field, self._language))

        label = QLabel(f'{sem_ui_field_label(enable_field, self._language)} (P)', row)
        label.setWordWrap(True)
        label.setToolTip(sem_ui_field_help(enable_field, self._language))

        probability_control = SemConfigSectionEditor.build_control(probability_field, row)
        probability_control.setToolTip(sem_ui_field_help(probability_field, self._language))
        probability_control.setProperty('baseToolTip', probability_control.toolTip())

        layout.addWidget(enable_checkbox, 0, Qt.AlignmentFlag.AlignTop)
        layout.addWidget(label, 1)
        layout.addWidget(probability_control, 0)

        self._register_control(enable_key, enable_checkbox, label=label, row_widget=row)
        self._register_control(probability_key, probability_control, row_widget=row)
        enable_checkbox.toggled.connect(self._on_dependent_toggled)
        self._connect_value_signal(probability_control)

        for strength_key in strength_keys:
            strength_field = SEM_UI_FIELDS_BY_KEY[strength_key]
            strength_control = SemConfigSectionEditor.build_control(strength_field, row)
            strength_control.setToolTip(sem_ui_field_help(strength_field, self._language))
            strength_control.setProperty('baseToolTip', strength_control.toolTip())
            layout.addWidget(strength_control, 0)
            self._register_control(strength_key, strength_control, row_widget=row)
            self._connect_value_signal(strength_control)

        self._effect_rows[enable_key] = row
        return row

    def _build_bool_weight_row(self, row_spec: CompactRow) -> QWidget:
        enable_key = str(row_spec.enable_key)
        weight_key = str(row_spec.weight_key)
        enable_field = SEM_UI_FIELDS_BY_KEY[enable_key]
        weight_field = SEM_UI_FIELDS_BY_KEY[weight_key]

        row = QWidget(self)
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        enable_checkbox = QCheckBox('', row)
        enable_checkbox.setToolTip(sem_ui_field_help(enable_field, self._language))

        label = QLabel(sem_ui_field_label(enable_field, self._language), row)
        label.setWordWrap(True)
        label.setToolTip(sem_ui_field_help(enable_field, self._language))

        weight_control = SemConfigSectionEditor.build_control(weight_field, row)
        weight_control.setToolTip(sem_ui_field_help(weight_field, self._language))
        weight_control.setProperty('baseToolTip', weight_control.toolTip())

        layout.addWidget(enable_checkbox, 0, Qt.AlignmentFlag.AlignTop)
        layout.addWidget(label, 1)
        layout.addWidget(weight_control, 0)

        self._register_control(enable_key, enable_checkbox, label=label, row_widget=row)
        self._register_control(weight_key, weight_control, row_widget=row)
        enable_checkbox.toggled.connect(self._on_dependent_toggled)
        self._connect_value_signal(weight_control)
        return row

    def _build_labeled_row(self, field_key: str) -> QWidget:
        field = SEM_UI_FIELDS_BY_KEY[field_key]
        row = QWidget(self)
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        label = QLabel(sem_ui_field_label(field, self._language), row)
        label.setWordWrap(True)
        tooltip = sem_ui_field_help(field, self._language)
        label.setToolTip(tooltip)

        control = SemConfigSectionEditor.build_control(field, row)
        control.setToolTip(tooltip)
        control.setProperty('baseToolTip', tooltip)

        layout.addWidget(label, 1)
        layout.addWidget(control, 0)
        self._register_control(field_key, control, label=label, row_widget=row)
        self._connect_value_signal(control)
        return row

    def _build_inline_row(self, field_keys: tuple[str, ...]) -> QWidget:
        row = QWidget(self)
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        for field_key in field_keys:
            field = SEM_UI_FIELDS_BY_KEY[field_key]
            label = QLabel(sem_ui_field_label(field, self._language), row)
            label.setWordWrap(True)
            tooltip = sem_ui_field_help(field, self._language)
            label.setToolTip(tooltip)

            control = SemConfigSectionEditor.build_control(field, row)
            control.setToolTip(tooltip)
            control.setProperty('baseToolTip', tooltip)

            layout.addWidget(label, 1)
            layout.addWidget(control, 0)
            self._register_control(field_key, control, label=label, row_widget=row)
            self._connect_value_signal(control)

        layout.addStretch(1)
        return row

    def _register_control(
        self,
        key: str,
        control: QWidget,
        *,
        label: QLabel | None = None,
        row_widget: QWidget | None = None,
    ) -> None:
        self.controls[key] = control
        if label is not None:
            self.labels[key] = label
        if row_widget is not None:
            self._row_widgets.setdefault(key, row_widget)

    def _connect_value_signal(self, control: QWidget) -> None:
        if isinstance(control, QCheckBox):
            control.toggled.connect(self._emit_changed)
        elif isinstance(control, (QSpinBox, QDoubleSpinBox)):
            control.valueChanged.connect(self._emit_changed)
        elif isinstance(control, QComboBox):
            control.currentIndexChanged.connect(self._emit_changed)
        elif isinstance(control, QLineEdit):
            control.textChanged.connect(self._emit_changed)

    def _on_master_toggled(self, _checked: bool) -> None:
        self._sync_dependent_rows()
        self._emit_changed()

    def _on_dependent_toggled(self, _checked: bool) -> None:
        self._sync_dependent_rows()
        self._emit_changed()

    def _sync_dependent_rows(self) -> None:
        master_enabled = True
        if self.layout_spec.master_key is not None:
            master_enabled = self.isChecked()

        for row_spec in self.layout_spec.rows:
            if row_spec.kind == 'effect':
                enable_key = str(row_spec.enable_key)
                probability_key = str(row_spec.probability_key)
                strength_keys = tuple(str(key) for key in row_spec.strength_keys)
                enable_control = self.controls[enable_key]
                effect_enabled = master_enabled and bool(enable_control.isChecked())
                self.controls[probability_key].setEnabled(effect_enabled)
                for strength_key in strength_keys:
                    self.controls[strength_key].setEnabled(effect_enabled)
            elif row_spec.kind == 'bool_weight':
                enable_key = str(row_spec.enable_key)
                weight_key = str(row_spec.weight_key)
                effect_enabled = master_enabled and bool(self.controls[enable_key].isChecked())
                self.controls[weight_key].setEnabled(effect_enabled)

        if self.layout_spec.master_key is not None:
            for row_spec in self.layout_spec.rows:
                if not row_spec.field_keys:
                    continue
                row_widget = self._row_widgets.get(str(row_spec.field_keys[0]))
                if row_widget is not None and row_spec.kind in {'labeled', 'inline'}:
                    row_widget.setEnabled(master_enabled)

    def _emit_changed(self, *_args: Any) -> None:
        if not self._update_guard:
            self.changed.emit()

    def set_form_values(self, values: Mapping[str, Any]) -> None:
        self._update_guard = True
        try:
            for field in self.fields:
                value = values.get(field.form_name, field.default)
                if field.key == self.layout_spec.master_key:
                    self.setChecked(bool(value))
                    continue
                control = self.controls.get(field.key)
                if control is not None:
                    SemConfigSectionEditor.set_control_value(field, control, value)
        finally:
            self._update_guard = False
        self._sync_dependent_rows()

    def form_values(self) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for field in self.fields:
            if field.key == self.layout_spec.master_key:
                values[field.form_name] = bool(self.isChecked())
                continue
            control = self.controls.get(field.key)
            if control is None:
                continue
            values[field.form_name] = SemConfigSectionEditor.control_value(field, control)
        return values

    def set_field_visible(self, key: str, visible: bool) -> None:
        if key == self.layout_spec.master_key:
            return
        row = self._row_widgets.get(key)
        if row is not None:
            row.setVisible(bool(visible))
        if key in self.labels:
            self.labels[key].setVisible(bool(visible))
        if key in self.controls:
            self.controls[key].setVisible(bool(visible))

    def set_language(self, language: str) -> None:
        self._language = str(language)
        if self.layout_spec.master_key is not None:
            self._apply_master_title()
        else:
            self._apply_section_title()

        for field in self.fields:
            if field.key == self.layout_spec.master_key:
                continue
            label = self.labels.get(field.key)
            control = self.controls.get(field.key)
            if label is not None:
                if any(row.kind == 'effect' and field.key == row.enable_key for row in self.layout_spec.rows):
                    label.setText(f'{sem_ui_field_label(field, self._language)} (P)')
                else:
                    label.setText(sem_ui_field_label(field, self._language))
                label.setToolTip(sem_ui_field_help(field, self._language))
            if control is not None:
                tooltip = sem_ui_field_help(field, self._language)
                control.setToolTip(tooltip)
                control.setProperty('baseToolTip', tooltip)
                if field.kind == 'choice' and isinstance(control, QComboBox):
                    for item_index, (value, english_label) in enumerate(field.choices):
                        control.setItemText(
                            item_index,
                            sem_ui_choice_label(value, english_label, self._language),
                        )

    @property
    def _plan_row(self) -> QWidget | None:
        if self.section != 'augmentation':
            return None
        return self._row_widgets.get('aug_plan')

    def _sync_effect_rows(self) -> None:
        self._sync_dependent_rows()


class SemAugmentationSectionEditor(CompactSemSectionEditor):
    """Backward-compatible alias for the augmentation section editor."""

    def __init__(self, parent: QWidget | None = None, *, language: str = 'en') -> None:
        super().__init__('augmentation', parent, language=language)
