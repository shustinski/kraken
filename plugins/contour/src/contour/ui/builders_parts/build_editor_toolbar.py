"""builders: build editor toolbar."""
from __future__ import annotations
from PyQt6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QSizePolicy,
    QToolButton,
    QWidget,
)
from ...graphics.tools import MIN_MANUAL_STROKE_WIDTH_PX
from contour.graphics_view import BrushMode, DeleteVertexMode, EditorTool, PolygonCreateMode
from contour.ui.no_wheel_controls import NoWheelComboBox as QComboBox
from contour.ui.no_wheel_controls import NoWheelSpinBox as QSpinBox

from contour.ui.builders_parts.build_ui import _apply_size_hint_geometry

def build_editor_toolbar(self) -> QWidget:
    toolbar = QWidget()
    layout = QHBoxLayout(toolbar)
    self._editor_toolbar_layout = layout
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(6)

    self._tool_button_group = QButtonGroup(self)
    self._tool_button_group.setExclusive(True)
    self._tool_buttons = {}
    for text, tool in [
        ("Select", EditorTool.SELECT),
        ("Pan", EditorTool.PAN),
        ("Ruler", EditorTool.RULER),
        ("Add Polygon", EditorTool.ADD_POLYGON),
        ("Brush", EditorTool.BRUSH),
        ("Trace Pen", EditorTool.TRACE_PEN),
        ("Contact", EditorTool.ADD_VIA),
        ("Add Vertex", EditorTool.ADD_VERTEX),
        ("Delete Vertex", EditorTool.DELETE_VERTEX),
        ("Move Vertex", EditorTool.MOVE_VERTEX),
        ("Antialias", EditorTool.ANTIALIAS),
    ]:
        button = QToolButton()
        self._configure_toolbar_button(button, self._create_editor_tool_icon(tool), text, checkable=True)
        button.clicked.connect(lambda checked=False, tool_value=tool: self._on_editor_tool_button_clicked(tool_value))
        self._tool_button_group.addButton(button)
        self._tool_buttons[tool] = button
        layout.addWidget(button)
        if tool == EditorTool.SELECT:
            button.setChecked(True)

    self._polygon_toolbar_block = QWidget()
    _polygon_blk = QHBoxLayout(self._polygon_toolbar_block)
    _polygon_blk.setContentsMargins(0, 0, 0, 0)
    _polygon_blk.setSpacing(6)
    self.polygon_mode_label = QLabel("Polygon")
    self.polygon_mode_combo = QComboBox()
    self.polygon_mode_combo.addItem(self._mode_text("polygon_points"), PolygonCreateMode.POINTS)
    self.polygon_mode_combo.addItem(self._mode_text("polygon_rectangle"), PolygonCreateMode.RECTANGLE)
    self.polygon_mode_combo.setCurrentIndex(self.polygon_mode_combo.findData(PolygonCreateMode.RECTANGLE))
    self.polygon_mode_combo.currentIndexChanged.connect(
        lambda _index: self.polygon_editor.set_polygon_create_mode(self.polygon_mode_combo.currentData())
    )
    self.polygon_draw_mode_indicator = QLabel("")
    self.polygon_draw_mode_indicator.setVisible(False)
    _polygon_blk.addWidget(self.polygon_mode_label)
    _polygon_blk.addWidget(self.polygon_mode_combo)
    _polygon_blk.addWidget(self.polygon_draw_mode_indicator)

    self._brush_toolbar_block = QWidget()
    _brush_blk = QHBoxLayout(self._brush_toolbar_block)
    _brush_blk.setContentsMargins(0, 0, 0, 0)
    _brush_blk.setSpacing(6)
    self.brush_mode_label = QLabel("Brush")
    self.brush_mode_combo = QComboBox()
    self.brush_mode_combo.addItem(self._mode_text("brush_freeform"), BrushMode.FREEFORM)
    self.brush_mode_combo.addItem(self._mode_text("brush_45deg"), BrushMode.ANGLED)
    self.brush_mode_combo.setCurrentIndex(self.brush_mode_combo.findData(BrushMode.ANGLED))
    self.brush_mode_combo.currentIndexChanged.connect(
        lambda _index: self.polygon_editor.set_brush_mode(self.brush_mode_combo.currentData())
    )
    self.brush_size_label = QLabel("Толщина" if self._ui_language == "ru" else "Width")
    self.brush_size_spin = QSpinBox()
    self.brush_size_spin.setRange(int(MIN_MANUAL_STROKE_WIDTH_PX), 256)
    self.brush_size_spin.setValue(12)
    self.brush_size_spin.valueChanged.connect(lambda value: self.polygon_editor.set_brush_thickness(float(value)))
    _brush_blk.addWidget(self.brush_mode_label)
    _brush_blk.addWidget(self.brush_mode_combo)
    _brush_blk.addWidget(self.brush_size_label)
    _brush_blk.addWidget(self.brush_size_spin)

    self._trace_toolbar_block = QWidget()
    _trace_blk = QHBoxLayout(self._trace_toolbar_block)
    _trace_blk.setContentsMargins(0, 0, 0, 0)
    _trace_blk.setSpacing(6)
    self.trace_mode_label = QLabel("Trace")
    self.trace_mode_combo = QComboBox()
    self.trace_mode_combo.addItem(self._mode_text("brush_freeform"), BrushMode.FREEFORM)
    self.trace_mode_combo.addItem(self._mode_text("brush_45deg"), BrushMode.ANGLED)
    self.trace_mode_combo.setCurrentIndex(self.trace_mode_combo.findData(BrushMode.ANGLED))
    self.trace_mode_combo.currentIndexChanged.connect(
        lambda _index: self.polygon_editor.set_trace_mode(self.trace_mode_combo.currentData())
    )
    self.trace_width_label = QLabel("Ширина" if self._ui_language == "ru" else "Width")
    self.trace_width_spin = QSpinBox()
    self.trace_width_spin.setRange(int(MIN_MANUAL_STROKE_WIDTH_PX), 256)
    self.trace_width_spin.setValue(12)
    self.trace_width_spin.valueChanged.connect(lambda value: self.polygon_editor.set_trace_width(float(value)))
    _trace_blk.addWidget(self.trace_mode_label)
    _trace_blk.addWidget(self.trace_mode_combo)
    _trace_blk.addWidget(self.trace_width_label)
    _trace_blk.addWidget(self.trace_width_spin)

    self._via_toolbar_block = QWidget()
    _via_blk = QHBoxLayout(self._via_toolbar_block)
    _via_blk.setContentsMargins(0, 0, 0, 0)
    _via_blk.setSpacing(6)
    self.via_width_label = QLabel("Width")
    self.via_width_spin = QSpinBox()
    self.via_width_spin.setRange(1, 100_000)
    self.via_width_spin.setValue(12)
    self.via_height_label = QLabel("Height")
    self.via_height_spin = QSpinBox()
    self.via_height_spin.setRange(1, 100_000)
    self.via_height_spin.setValue(12)
    self.via_width_spin.valueChanged.connect(lambda _value: self._sync_editor_via_size())
    self.via_height_spin.valueChanged.connect(lambda _value: self._sync_editor_via_size())
    _via_blk.addWidget(self.via_width_label)
    _via_blk.addWidget(self.via_width_spin)
    _via_blk.addWidget(self.via_height_label)
    _via_blk.addWidget(self.via_height_spin)

    self._delete_vertex_toolbar_block = QWidget()
    _dv_blk = QHBoxLayout(self._delete_vertex_toolbar_block)
    _dv_blk.setContentsMargins(0, 0, 0, 0)
    _dv_blk.setSpacing(6)
    self.delete_vertex_mode_label = QLabel("Delete")
    self.delete_vertex_mode_combo = QComboBox()
    self.delete_vertex_mode_combo.addItem(self._mode_text("delete_single"), DeleteVertexMode.SINGLE)
    self.delete_vertex_mode_combo.addItem(self._mode_text("delete_area"), DeleteVertexMode.AREA)
    self.delete_vertex_mode_combo.setCurrentIndex(self.delete_vertex_mode_combo.findData(DeleteVertexMode.AREA))
    self.delete_vertex_mode_combo.currentIndexChanged.connect(
        lambda _index: self.polygon_editor.set_delete_vertex_mode(self.delete_vertex_mode_combo.currentData())
    )
    _dv_blk.addWidget(self.delete_vertex_mode_label)
    _dv_blk.addWidget(self.delete_vertex_mode_combo)

    self._antialias_toolbar_block = QWidget()
    _aa_blk = QHBoxLayout(self._antialias_toolbar_block)
    _aa_blk.setContentsMargins(0, 0, 0, 0)
    _aa_blk.setSpacing(6)
    self.antialias_grade_label = QLabel("AA")
    self.antialias_grade_spin = QSpinBox()
    self.antialias_grade_spin.setRange(1, 5)
    self.antialias_grade_spin.setValue(1)
    self.antialias_grade_spin.valueChanged.connect(lambda value: self.polygon_editor.set_antialias_grade(value))
    _aa_blk.addWidget(self.antialias_grade_label)
    _aa_blk.addWidget(self.antialias_grade_spin)

    self.tool_settings_strip = QWidget()
    self.tool_settings_strip.setObjectName("toolSettingsStrip")
    self.tool_settings_layout = QHBoxLayout(self.tool_settings_strip)
    self.tool_settings_layout.setContentsMargins(0, 0, 0, 0)
    self.tool_settings_layout.setSpacing(6)
    for block in (
        self._polygon_toolbar_block,
        self._brush_toolbar_block,
        self._trace_toolbar_block,
        self._via_toolbar_block,
        self._delete_vertex_toolbar_block,
        self._antialias_toolbar_block,
    ):
        self.tool_settings_layout.addWidget(block)
    layout.addWidget(self.tool_settings_strip)
    self._tool_parameter_blocks = {
        EditorTool.ADD_POLYGON: self._polygon_toolbar_block,
        EditorTool.BRUSH: self._brush_toolbar_block,
        EditorTool.TRACE_PEN: self._trace_toolbar_block,
        EditorTool.ADD_VIA: self._via_toolbar_block,
        EditorTool.DELETE_VERTEX: self._delete_vertex_toolbar_block,
        EditorTool.ANTIALIAS: self._antialias_toolbar_block,
    }

    self.ruler_status_label = QLabel("")
    self.ruler_status_label.setVisible(False)
    layout.addWidget(self.ruler_status_label)

    self.undo_button = QToolButton()
    self._configure_toolbar_button(self.undo_button, self._create_editor_action_icon("undo"), "Undo")
    self.undo_button.clicked.connect(self.polygon_editor.undo)
    self.redo_button = QToolButton()
    self._configure_toolbar_button(self.redo_button, self._create_editor_action_icon("redo"), "Redo")
    self.redo_button.clicked.connect(self.polygon_editor.redo)
    self.zoom_in_button = QToolButton()
    self._configure_toolbar_button(self.zoom_in_button, self._create_editor_action_icon("zoom_in"), "Zoom +")
    self.zoom_in_button.clicked.connect(self.polygon_editor.zoom_in)
    self.zoom_out_button = QToolButton()
    self._configure_toolbar_button(self.zoom_out_button, self._create_editor_action_icon("zoom_out"), "Zoom -")
    self.zoom_out_button.clicked.connect(self.polygon_editor.zoom_out)
    self.fit_button = QToolButton()
    self._configure_toolbar_button(self.fit_button, self._create_editor_action_icon("fit"), "Fit")
    self.fit_button.clicked.connect(self.polygon_editor.fit_to_view)

    for button in [
        self.undo_button,
        self.redo_button,
        self.zoom_in_button,
        self.zoom_out_button,
        self.fit_button,
    ]:
        layout.addWidget(button)

    self.vector_edit_status_label = QLabel("")
    self.vector_edit_status_label.setMinimumWidth(88)
    layout.addWidget(self.vector_edit_status_label)

    self.preview_busy_label = QLabel(self._busy_indicator_text())
    self.preview_busy_label.setMinimumWidth(0)
    self.preview_busy_label.setMaximumWidth(150)
    self.preview_busy_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
    self.preview_busy_progress = QProgressBar()
    self.preview_busy_progress.setRange(0, 100)
    self.preview_busy_progress.setValue(0)
    self.preview_busy_progress.setFormat("%p%")
    self.preview_busy_progress.setTextVisible(True)
    self.preview_busy_progress.setMinimumWidth(72)
    self.preview_busy_progress.setMaximumWidth(120)
    self.preview_busy_progress.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
    self.preview_busy_label.setVisible(False)
    self.preview_busy_progress.setVisible(False)
    layout.addWidget(self.preview_busy_label)
    layout.addWidget(self.preview_busy_progress)
    layout.addStretch(1)
    for widget in (
        self._polygon_toolbar_block,
        self._brush_toolbar_block,
        self._via_toolbar_block,
        self._delete_vertex_toolbar_block,
        self._antialias_toolbar_block,
        self.tool_settings_strip,
        self.polygon_mode_combo,
        self.polygon_draw_mode_indicator,
        self.brush_mode_combo,
        self.brush_size_spin,
        self.trace_mode_combo,
        self.trace_width_spin,
        self.via_width_spin,
        self.via_height_spin,
        self.delete_vertex_mode_combo,
        self.ruler_status_label,
        self.antialias_grade_label,
        self.antialias_grade_spin,
    ):
        _apply_size_hint_geometry(widget, width=True, height=True)
        widget.setMinimumHeight(max(30, widget.minimumHeight()))
        widget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
    self.polygon_editor.set_polygon_create_mode(self.polygon_mode_combo.currentData())
    self.polygon_editor.set_brush_mode(self.brush_mode_combo.currentData())
    self.polygon_editor.set_brush_thickness(float(self.brush_size_spin.value()))
    self.polygon_editor.set_trace_mode(self.trace_mode_combo.currentData())
    self.polygon_editor.set_trace_width(float(self.trace_width_spin.value()))
    self._restore_persisted_tool_modes()
    self._sync_editor_via_size()
    self.polygon_editor.set_delete_vertex_mode(self.delete_vertex_mode_combo.currentData())
    self.polygon_editor.set_antialias_grade(int(self.antialias_grade_spin.value()))
    self.tool_settings_strip.setMinimumWidth(
        max(
            self._polygon_toolbar_block.sizeHint().width(),
            self._brush_toolbar_block.sizeHint().width(),
            self._trace_toolbar_block.sizeHint().width(),
            self._via_toolbar_block.sizeHint().width(),
            self._delete_vertex_toolbar_block.sizeHint().width(),
            self._antialias_toolbar_block.sizeHint().width(),
        )
    )
    self._on_editor_tool_changed(self.polygon_editor.current_tool)
    return toolbar
