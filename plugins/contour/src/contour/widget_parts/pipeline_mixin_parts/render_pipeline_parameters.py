"""pipeline_mixin: render pipeline parameters."""
from __future__ import annotations
from contour.widget_parts._imports import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QLabel,
    QSpinBox,
    QWidget,
    get_choice_display_label,
    get_operation_descriptor,
    get_parameter_display_label,
)


class RenderPipelineParametersMixin:
    def _render_pipeline_parameters(self, row: int) -> None:
        self._clear_parameters_form()
        if row < 0 or row >= len(self._pipeline.steps):
            self._set_color_pick_active(None)
            return
        step = self._pipeline.steps[row]
        descriptor = get_operation_descriptor(step.operation)
        widget: QWidget
        for spec in descriptor.parameters:
            value = step.parameters.get(spec.name, spec.default)
            if spec.kind == "bool":
                widget = QCheckBox()
                widget.setChecked(bool(value))
                widget.stateChanged.connect(
                    lambda _state, name=spec.name, row_index=row, w=widget: self._update_step_parameter(
                        row_index, name, w.isChecked()
                    )
                )
            elif spec.kind == "choice":
                widget = QComboBox()
                self._register_no_wheel_value_widget(widget)
                for option in spec.options:
                    widget.addItem(get_choice_display_label(spec.name, str(option), self._ui_language), option)
                selected_index = widget.findData(value)
                if selected_index >= 0:
                    widget.setCurrentIndex(selected_index)
                widget.currentIndexChanged.connect(
                    lambda _index, name=spec.name, row_index=row, w=widget: self._update_step_parameter(
                        row_index,
                        name,
                        w.currentData(),
                    )
                )
            elif spec.kind == "int":
                widget = QSpinBox()
                self._register_spinbox(widget)
                widget.setRange(int(spec.minimum or -1_000_000), int(spec.maximum or 1_000_000))
                widget.setSingleStep(int(spec.step or 1))
                widget.setValue(int(value))
                widget.valueChanged.connect(
                    lambda new_value, name=spec.name, row_index=row: self._update_step_parameter(
                        row_index, name, int(new_value)
                    )
                )
            else:
                widget = QDoubleSpinBox()
                self._register_spinbox(widget)
                widget.setDecimals(spec.decimals)
                widget.setRange(float(spec.minimum or -1_000_000), float(spec.maximum or 1_000_000))
                widget.setSingleStep(float(spec.step or 0.1))
                widget.setValue(float(value))
                widget.valueChanged.connect(
                    lambda new_value, name=spec.name, row_index=row: self._update_step_parameter(
                        row_index, name, float(new_value)
                    )
                )
            tooltip = spec.tooltip or self._pipeline_parameter_tooltip(step.operation, spec.name)
            widget.setToolTip(tooltip)
            self._parameter_widgets[spec.name] = widget
            label_widget = QLabel(get_parameter_display_label(spec, self._ui_language))
            label_widget.setToolTip(tooltip)
            self.parameters_form.addRow(label_widget, widget)
        if step.operation == "color_binarize":
            self._render_color_binarize_parameters(row)
        else:
            self._set_color_pick_active(None)
