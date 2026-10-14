"""builders: build strategy parameter page."""
from __future__ import annotations
from PyQt6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QVBoxLayout,
    QWidget,
)
from ...vision.metal_recovery.strategy_registry import (
    ParameterSpec,
    StrategySpec,
)
from contour.ui.metal_strategy_i18n import (
    choice_label,
    parameter_label,
    parameter_tooltip,
)
from contour.ui.no_wheel_controls import NoWheelComboBox as QComboBox
from contour.ui.no_wheel_controls import NoWheelDoubleSpinBox as QDoubleSpinBox
from contour.ui.no_wheel_controls import NoWheelSpinBox as QSpinBox

def _strategy_parameter_widget(parameter: ParameterSpec, language: str) -> QWidget:
    if parameter.kind == "bool":
        widget = QCheckBox()
        widget.setChecked(bool(parameter.default))
    elif parameter.kind == "choice":
        combo = QComboBox()
        for value, label in parameter.choices:
            combo.addItem(choice_label(label, language), value)
        index = combo.findData(parameter.default)
        combo.setCurrentIndex(max(0, index))
        widget = combo
    elif parameter.kind == "int":
        spin = QSpinBox()
        minimum = -2_147_483_648 if parameter.minimum is None else int(parameter.minimum)
        maximum = 2_147_483_647 if parameter.maximum is None else int(parameter.maximum)
        spin.setRange(minimum, maximum)
        spin.setSingleStep(max(1, int(parameter.step or 1)))
        spin.setValue(int(parameter.default))
        if parameter.units:
            spin.setSuffix(f" {parameter.units}")
        widget = spin
    else:
        spin = QDoubleSpinBox()
        minimum = -1e12 if parameter.minimum is None else float(parameter.minimum)
        maximum = 1e12 if parameter.maximum is None else float(parameter.maximum)
        spin.setRange(minimum, maximum)
        spin.setSingleStep(float(parameter.step or 0.1))
        spin.setDecimals(4 if float(parameter.step or 0.1) < 0.01 else 3)
        spin.setValue(float(parameter.default))
        if parameter.units:
            spin.setSuffix(f" {parameter.units}")
        widget = spin
    widget.setToolTip(parameter_tooltip(parameter, language))
    widget.setObjectName(f"metal_strategy_parameter_{parameter.key}")
    return widget

def _build_strategy_parameter_page(self, spec: StrategySpec) -> QWidget:
    page = QWidget()
    page_layout = QVBoxLayout(page)
    page_layout.setContentsMargins(0, 0, 0, 0)
    page_layout.setSpacing(6)
    controls: dict[str, QWidget] = {}
    language = str(getattr(self, "_ui_language", "ru") or "ru")
    group_titles = (
        ((False, "Основные"), (True, "Дополнительные")) if language == "ru" else ((False, "Basic"), (True, "Advanced"))
    )
    for advanced, title in group_titles:
        parameters = [parameter for parameter in spec.parameters if parameter.advanced is advanced]
        if not parameters:
            continue
        group = QGroupBox(title)
        group.setObjectName(f"metal_strategy_{spec.strategy_id}_{'advanced' if advanced else 'basic'}_group")
        form = QFormLayout(group)
        self._configure_compact_form(form)
        for parameter in parameters:
            widget = _strategy_parameter_widget(parameter, language)
            label = QLabel(parameter_label(parameter, language))
            label.setToolTip(parameter_tooltip(parameter, language))
            form.addRow(label, widget)
            controls[parameter.key] = widget
            if isinstance(widget, QCheckBox):
                widget.toggled.connect(self._on_extraction_settings_changed)
            elif isinstance(widget, QComboBox):
                widget.currentIndexChanged.connect(self._on_extraction_settings_changed)
            else:
                widget.valueChanged.connect(self._on_extraction_settings_changed)
        page_layout.addWidget(group)
    if "minimum_lifted_distance" in controls and "maximum_lifted_distance" in controls:
        minimum_distance = controls["minimum_lifted_distance"]
        maximum_distance = controls["maximum_lifted_distance"]
        minimum_distance.valueChanged.connect(maximum_distance.setMinimum)
        maximum_distance.setMinimum(minimum_distance.value())
    if "minimum_background_confidence" in controls and "minimum_metal_confidence" in controls:
        background_confidence = controls["minimum_background_confidence"]
        metal_confidence = controls["minimum_metal_confidence"]
        metal_confidence.valueChanged.connect(background_confidence.setMaximum)
        background_confidence.setMaximum(metal_confidence.value())
    page_layout.addStretch(1)
    self.metal_strategy_parameter_widgets[spec.strategy_id] = controls
    return page
