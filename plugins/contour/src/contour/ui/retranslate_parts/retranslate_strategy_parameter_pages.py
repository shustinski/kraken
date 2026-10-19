"""retranslate: retranslate strategy parameter pages."""
from __future__ import annotations
from typing import TYPE_CHECKING
from PyQt6.QtWidgets import QComboBox, QFormLayout, QGroupBox
from ...vision.metal_recovery.strategy_registry import strategy_spec
from contour.ui.metal_strategy_i18n import choice_label, parameter_label, parameter_tooltip

def _set_form_label(group: QGroupBox, field: object, text: str) -> None:
    form = group.layout()
    if not isinstance(form, QFormLayout):
        return
    label = form.labelForField(field)
    if label is not None:
        label.setText(text)

def _retranslate_strategy_parameter_pages(self: PolygonExtractionWidget) -> None:
    language = str(self._ui_language)
    for strategy_id, controls in getattr(self, "metal_strategy_parameter_widgets", {}).items():
        spec = strategy_spec(strategy_id)
        parameters = {parameter.key: parameter for parameter in spec.parameters}
        for key, control in controls.items():
            parameter = parameters[key]
            tooltip = parameter_tooltip(parameter, language)
            control.setToolTip(tooltip)
            parent = control.parentWidget()
            form = parent.layout() if parent is not None else None
            if isinstance(form, QFormLayout):
                label = form.labelForField(control)
                if label is not None:
                    label.setText(parameter_label(parameter, language))
                    label.setToolTip(tooltip)
            if isinstance(control, QComboBox):
                labels_by_value = dict(parameter.choices)
                for index in range(control.count()):
                    value = str(control.itemData(index))
                    source_label = labels_by_value.get(value)
                    if source_label is not None:
                        control.setItemText(index, choice_label(source_label, language))

        page_index = self.metal_strategy_parameter_pages.get(strategy_id)
        if page_index is None:
            continue
        page = self.metal_strategy_parameter_stack.widget(page_index)
        for group in page.findChildren(QGroupBox):
            advanced = group.objectName().endswith("_advanced_group")
            group.setTitle(
                ("Дополнительные" if advanced else "Основные")
                if language == "ru"
                else ("Advanced" if advanced else "Basic")
            )

if TYPE_CHECKING:
    from contour.widget import PolygonExtractionWidget
