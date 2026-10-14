"""builders: build paths tab."""
from __future__ import annotations
from PyQt6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QGroupBox,
    QVBoxLayout,
    QWidget,
)
from ...application.vector_geometry_postprocess import VectorGeometrySettings
from contour.ui.no_wheel_controls import NoWheelDoubleSpinBox as QDoubleSpinBox

def build_paths_tab(self) -> QWidget:
    tab = QWidget()
    layout = QVBoxLayout(tab)
    self.path_panel = self._build_path_panel()
    self.path_panel.setParent(tab)
    self.path_panel.hide()

    self.vector_geom_group = QGroupBox("Постобработка ручных инструментов")
    self.vector_geom_group.setToolTip(
        "Post-process manual vectors when opening frames and after edits.\n"
        "Triangle artifact removal drops unparented 3-vertex outers unless disabled or marked as a contact/box.",
    )
    vg_form = QFormLayout(self.vector_geom_group)
    self._configure_compact_form(vg_form)
    self.vector_geom_clip_checkbox = QCheckBox("Обрезать по границе кадра и удалить внешние объекты")
    self.vector_geom_clip_checkbox.setChecked(True)
    _vector_geom_defaults = VectorGeometrySettings()
    self.vector_geom_min_outer_spin = QDoubleSpinBox()
    self.vector_geom_min_outer_spin.setRange(0.0, 1_000_000.0)
    self.vector_geom_min_outer_spin.setDecimals(3)
    self.vector_geom_min_outer_spin.setSingleStep(1.0)
    self.vector_geom_min_outer_spin.setValue(_vector_geom_defaults.min_outer_area_px2)
    self.vector_geom_min_hole_spin = QDoubleSpinBox()
    self.vector_geom_min_hole_spin.setRange(0.0, 1_000_000.0)
    self.vector_geom_min_hole_spin.setDecimals(3)
    self.vector_geom_min_hole_spin.setSingleStep(1.0)
    self.vector_geom_min_hole_spin.setValue(_vector_geom_defaults.min_hole_area_to_remove_px2)
    self.vector_geom_merge_checkbox = QCheckBox("Объединять пересекающиеся полигоны после перемещения")
    self.vector_geom_merge_checkbox.setChecked(True)
    self.vector_geom_spike_angle_spin = QDoubleSpinBox()
    self.vector_geom_spike_angle_spin.setRange(0.0, 179.0)
    self.vector_geom_spike_angle_spin.setDecimals(1)
    self.vector_geom_spike_angle_spin.setSingleStep(1.0)
    self.vector_geom_spike_angle_spin.setValue(30.0)
    self.vector_geom_spike_angle_spin.setToolTip(
        "Interior angle threshold in degrees below which a vertex spike is flattened. Set to 0 to disable spike removal.",
    )
    self.vector_geom_drop_triangle_checkbox = QCheckBox("Удалять внешние треугольники из 3 вершин как артефакты")
    self.vector_geom_drop_triangle_checkbox.setChecked(True)
    vg_form.addRow(self.vector_geom_clip_checkbox)
    vg_form.addRow("Минимальная площадь внешнего объекта, px²", self.vector_geom_min_outer_spin)
    self.vector_geom_min_outer_label_widget = vg_form.labelForField(self.vector_geom_min_outer_spin)
    self.vector_geom_min_hole_label_widget = None
    vg_form.addRow(self.vector_geom_merge_checkbox)
    vg_form.addRow("Минимальный угол острого выброса, °", self.vector_geom_spike_angle_spin)
    self.vector_geom_spike_angle_label_widget = vg_form.labelForField(self.vector_geom_spike_angle_spin)
    vg_form.addRow(self.vector_geom_drop_triangle_checkbox)
    for _w in (
        self.vector_geom_clip_checkbox,
        self.vector_geom_min_outer_spin,
        self.vector_geom_min_hole_spin,
        self.vector_geom_merge_checkbox,
        self.vector_geom_spike_angle_spin,
        self.vector_geom_drop_triangle_checkbox,
    ):
        if isinstance(_w, QCheckBox):
            _w.stateChanged.connect(self._on_vector_geom_control_changed)
        else:
            _w.valueChanged.connect(self._on_vector_geom_control_changed)
    self.vector_geom_group.setVisible(False)

    layout.addStretch(1)
    return tab
