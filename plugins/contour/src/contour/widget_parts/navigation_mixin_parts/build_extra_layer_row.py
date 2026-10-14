"""navigation_mixin: build extra layer row."""
from __future__ import annotations
from typing import Any
from contour.value_conversion import to_int
from contour.widget_parts._imports import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QWidget,
)


class BuildExtraLayerRowMixin:
    def _build_extra_layer_row_widget(self: Any, layer: dict[str, object]) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        layer_id = to_int(layer.get("id", 0))

        visible_checkbox = QCheckBox("")
        visible_checkbox.setFixedWidth(22)
        visible_checkbox.setChecked(bool(layer.get("visible", True)))
        visible_checkbox.setToolTip("Показать/скрыть слой")
        visible_checkbox.stateChanged.connect(
            lambda _state, lid=layer_id: self._set_extra_layer_field(lid, "visible", visible_checkbox.isChecked())
        )

        dx_spin = QSpinBox()
        dx_spin.setRange(-1_000_000, 1_000_000)
        dx_spin.setValue(to_int(layer.get("dx", 0) or 0))
        self._configure_extra_layer_spinbox(dx_spin, width=48)
        dx_spin.setToolTip("Смещение слоя по X")
        dx_spin.valueChanged.connect(lambda value, lid=layer_id: self._set_extra_layer_field(lid, "dx", int(value)))

        dy_spin = QSpinBox()
        dy_spin.setRange(-1_000_000, 1_000_000)
        dy_spin.setValue(to_int(layer.get("dy", 0) or 0))
        self._configure_extra_layer_spinbox(dy_spin, width=48)
        dy_spin.setToolTip("Смещение слоя по Y")
        dy_spin.valueChanged.connect(lambda value, lid=layer_id: self._set_extra_layer_field(lid, "dy", int(value)))

        opacity_spin = QSpinBox()
        opacity_spin.setRange(0, 100)
        try:
            opacity_value = to_int(layer.get("opacity", 100))
        except (TypeError, ValueError):
            opacity_value = 100
        opacity_spin.setValue(max(0, min(100, opacity_value)))
        opacity_spin.setSuffix("%")
        self._configure_extra_layer_spinbox(opacity_spin, width=46)
        opacity_spin.setToolTip("Прозрачность слоя")
        opacity_spin.valueChanged.connect(
            lambda value, lid=layer_id: self._set_extra_layer_field(lid, "opacity", int(value))
        )

        remove_button = QPushButton("X")
        remove_button.setFixedSize(28, 24)
        remove_button.setStyleSheet(
            "QPushButton { background-color: #DC2626; color: white; font-weight: 700; "
            "border-radius: 4px; padding: 0; }"
        )
        remove_button.setToolTip("Удалить слой")
        remove_button.clicked.connect(lambda _checked=False, lid=layer_id: self._remove_extra_layer_by_id(lid))

        folder_name = str(layer.get("name", "Layer"))
        folder_label = QLabel(folder_name)
        folder_label.setMinimumWidth(0)
        folder_label.setToolTip(str(layer.get("folder_path", "")))

        layout.addWidget(visible_checkbox)
        layout.addWidget(folder_label, 1)
        layout.addWidget(dx_spin)
        layout.addWidget(dy_spin)
        layout.addWidget(opacity_spin)
        layout.addWidget(remove_button)
        return row
