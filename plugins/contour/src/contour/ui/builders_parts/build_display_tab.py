"""builders: build display tab."""
from __future__ import annotations
from PyQt6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QWidget,
)
from ...application.processing import (
    VIA_DISPLAY_MODE_CIRCLE,
    VIA_DISPLAY_MODE_RECTANGLE,
)
from contour.ui.no_wheel_controls import NoWheelComboBox as QComboBox
from contour.ui.no_wheel_controls import NoWheelDoubleSpinBox as QDoubleSpinBox
from contour.ui.no_wheel_controls import NoWheelSpinBox as QSpinBox

def build_display_tab(self) -> QWidget:
    tab = QWidget()
    self.display_form = QFormLayout(tab)

    self.external_color_button = self._build_color_button(
        self._display_settings.external_color, self._choose_external_color
    )
    self.hole_color_button = self._build_color_button(self._display_settings.hole_color, self._choose_hole_color)
    self.selected_color_button = self._build_color_button(
        self._display_settings.selected_color, self._choose_selected_color
    )
    self.via_selection_color_button = self._build_color_button(
        self._display_settings.via_selection_color, self._choose_via_selection_color
    )
    self.conductor_hover_highlight_color_button = self._build_color_button(
        self._display_settings.conductor_hover_highlight_color,
        self._choose_conductor_hover_highlight_color,
    )
    self.vertex_color_button = self._build_color_button(self._display_settings.vertex_color, self._choose_vertex_color)
    self.line_width_spin = QDoubleSpinBox()
    self.line_width_spin.setRange(1.0, 20.0)
    self.line_width_spin.setValue(self._display_settings.line_width)
    self.vertex_size_spin = QDoubleSpinBox()
    self.vertex_size_spin.setRange(2.0, 30.0)
    self.vertex_size_spin.setValue(self._display_settings.vertex_size)
    self.fill_opacity_spin = QDoubleSpinBox()
    self.fill_opacity_spin.setRange(0.0, 1.0)
    self.fill_opacity_spin.setSingleStep(0.05)
    self.fill_opacity_spin.setValue(self._display_settings.fill_opacity)
    self.show_vertices_checkbox = QCheckBox("Show vertices")
    self.show_vertices_checkbox.setChecked(self._display_settings.show_vertices)
    self.show_labels_checkbox = QCheckBox("Show polygon IDs")
    self.show_labels_checkbox.setChecked(self._display_settings.show_labels)
    self.random_object_colors_checkbox = QCheckBox("Contrasting colors for nearby objects")
    self.via_display_mode_combo = QComboBox()
    self.via_display_mode_combo.addItem("Circle", VIA_DISPLAY_MODE_CIRCLE)
    self.via_display_mode_combo.addItem("Rectangle", VIA_DISPLAY_MODE_RECTANGLE)
    via_display_index = self.via_display_mode_combo.findData(self._display_settings.via_display_mode)
    self.via_display_mode_combo.setCurrentIndex(max(0, via_display_index))
    self.autosave_on_frame_transition_checkbox = QCheckBox("Autosave when changing frame")
    self.autosave_on_frame_transition_checkbox.setChecked(False)
    self.show_frame_matrix_checkbox = QCheckBox("Show frame matrix")
    self.show_frame_matrix_checkbox.setChecked(True)
    self.show_frame_matrix_thumbnails_checkbox = QCheckBox("Load frame matrix thumbnails")
    self.show_frame_matrix_thumbnails_checkbox.setChecked(True)
    self.show_neighbor_frames_checkbox = QCheckBox("Show neighboring frames")
    self.show_neighbor_vectors_checkbox = QCheckBox("Show vectors on neighboring frames")
    self.neighbor_columns_spin = QSpinBox()
    self.neighbor_columns_spin.setRange(1, 1000)
    self.neighbor_columns_spin.setValue(3)
    self.neighbor_max_grid_spin = QSpinBox()
    self.neighbor_max_grid_spin.setRange(3, 7)
    self.neighbor_max_grid_spin.setSingleStep(2)
    self.neighbor_max_grid_spin.setValue(7)
    self.neighbor_opacity_spin = QDoubleSpinBox()
    self.neighbor_opacity_spin.setRange(0.05, 1.0)
    self.neighbor_opacity_spin.setSingleStep(0.05)
    self.neighbor_opacity_spin.setValue(0.35)
    self.neighbor_overlap_spin = QSpinBox()
    self.neighbor_overlap_spin.setRange(0, 100_000)
    self.neighbor_overlap_spin.setValue(0)

    for widget in [
        self.line_width_spin,
        self.vertex_size_spin,
        self.fill_opacity_spin,
        self.show_vertices_checkbox,
        self.show_labels_checkbox,
        self.random_object_colors_checkbox,
        self.via_display_mode_combo,
    ]:
        if isinstance(widget, QCheckBox):
            widget.stateChanged.connect(self._apply_display_settings)
        elif isinstance(widget, QComboBox):
            widget.currentIndexChanged.connect(self._apply_display_settings)
        else:
            widget.valueChanged.connect(self._apply_display_settings)
    self.show_neighbor_frames_checkbox.stateChanged.connect(self._on_neighbor_display_settings_changed)
    self.show_neighbor_vectors_checkbox.stateChanged.connect(self._on_neighbor_display_settings_changed)
    self.show_frame_matrix_checkbox.stateChanged.connect(self._on_frame_matrix_display_settings_changed)
    self.show_frame_matrix_thumbnails_checkbox.stateChanged.connect(self._on_frame_matrix_thumbnail_settings_changed)
    self.neighbor_columns_spin.valueChanged.connect(self._on_neighbor_display_settings_changed)
    self.neighbor_max_grid_spin.valueChanged.connect(self._on_neighbor_display_settings_changed)
    self.neighbor_opacity_spin.valueChanged.connect(self._on_neighbor_display_settings_changed)
    self.neighbor_overlap_spin.valueChanged.connect(self._on_neighbor_display_settings_changed)

    self.display_form.addRow("External contour", self.external_color_button)
    self.external_color_label_widget = self.display_form.labelForField(self.external_color_button)
    self.display_form.addRow("Hole contour", self.hole_color_button)
    self.hole_color_label_widget = self.display_form.labelForField(self.hole_color_button)
    self.display_form.addRow("Selected contour", self.selected_color_button)
    self.selected_color_label_widget = self.display_form.labelForField(self.selected_color_button)
    self.display_form.addRow("Selected contact", self.via_selection_color_button)
    self.via_selection_color_label_widget = self.display_form.labelForField(self.via_selection_color_button)
    self.display_form.addRow("Conductor hover", self.conductor_hover_highlight_color_button)
    self.conductor_hover_highlight_label_widget = self.display_form.labelForField(
        self.conductor_hover_highlight_color_button
    )
    self.display_form.addRow("Vertex color", self.vertex_color_button)
    self.vertex_color_label_widget = self.display_form.labelForField(self.vertex_color_button)
    self.display_form.addRow("Line width", self.line_width_spin)
    self.line_width_label_widget = self.display_form.labelForField(self.line_width_spin)
    self.display_form.addRow("Vertex size", self.vertex_size_spin)
    self.vertex_size_label_widget = self.display_form.labelForField(self.vertex_size_spin)
    self.display_form.addRow("Fill opacity", self.fill_opacity_spin)
    self.fill_opacity_label_widget = self.display_form.labelForField(self.fill_opacity_spin)
    self.display_form.addRow(self.show_vertices_checkbox)
    self.display_form.addRow(self.show_labels_checkbox)
    self.display_form.addRow(self.random_object_colors_checkbox)
    self.display_form.addRow("Contact display", self.via_display_mode_combo)
    self.via_display_mode_label_widget = self.display_form.labelForField(self.via_display_mode_combo)
    self.display_form.addRow(self.autosave_on_frame_transition_checkbox)
    self.display_form.addRow(self.show_frame_matrix_checkbox)
    self.display_form.addRow(self.show_frame_matrix_thumbnails_checkbox)
    self.display_form.addRow(self.show_neighbor_frames_checkbox)
    self.display_form.addRow(self.show_neighbor_vectors_checkbox)
    self.display_form.addRow("Frames per row", self.neighbor_columns_spin)
    self.neighbor_columns_label_widget = self.display_form.labelForField(self.neighbor_columns_spin)
    self.display_form.addRow("Neighbor grid size", self.neighbor_max_grid_spin)
    self.neighbor_max_grid_label_widget = self.display_form.labelForField(self.neighbor_max_grid_spin)
    self.display_form.addRow("Neighbor opacity", self.neighbor_opacity_spin)
    self.neighbor_opacity_label_widget = self.display_form.labelForField(self.neighbor_opacity_spin)
    self.display_form.addRow("Frame overlap", self.neighbor_overlap_spin)
    self.neighbor_overlap_label_widget = self.display_form.labelForField(self.neighbor_overlap_spin)
    return tab
