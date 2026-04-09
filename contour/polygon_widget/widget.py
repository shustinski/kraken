from __future__ import annotations

import json
from pathlib import Path

from PyQt6.QtCore import QPointF, QRectF, QSize, Qt, QThreadPool, QTimer, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap, QPolygonF
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .adapters.qt.preview import PreparedImageRunnable, PreviewImageView, PreviewProcessingRunnable
from .application.dto import PersistedPaths
from .application.processing import ContourExtractionSettings, DisplaySettings, SaveOptions
from .application.services import WorkspaceSession
from .application.use_cases import (
    PreparedImageRequest,
    PreviewProcessingRequest,
    build_prepared_image_signature,
    build_preview_request_signature,
    index_cif_directory,
    load_input_directory,
)
from .batch_processor import BatchProcessor
from .contour_extractor import APPROXIMATION_MODE_MAP, RETRIEVAL_MODE_MAP
from .domain import PolygonData
from .graphics_view import BrushMode, DeleteVertexMode, EditorTool, PolygonCreateMode, PolygonEditorView
from .infrastructure import WidgetPathSettingsStore
from .i18n import active_language, tr
from .pipeline import (
    PreprocessingPipeline,
    available_operations,
    get_choice_display_label,
    get_operation_descriptor,
    get_operation_display_name,
    get_parameter_display_label,
)
from .serializers import load_polygons_cif, save_result_bundle
from .utils import is_image_path, load_image_grayscale, scan_image_files


class PolygonExtractionWidget(QWidget):
    imageProcessed = pyqtSignal(str, list)
    batchProgress = pyqtSignal(int, int)
    batchFinished = pyqtSignal()
    polygonsEdited = pyqtSignal()
    logMessage = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("polygonExtractionWidget")
        self._ui_language = active_language()
        self._path_settings_store = WidgetPathSettingsStore()
        self._workspace = WorkspaceSession()
        self._pipeline = PreprocessingPipeline()
        self._display_settings = DisplaySettings()
        self._ignore_pipeline_item_change = False
        self._parameter_widgets: dict[str, QWidget] = {}
        self._updating_views = False
        self._batch_progress_enabled = False
        self._progress_status_key = "idle_status"
        self._progress_status_kwargs: dict[str, object] = {}
        self._preview_thread_pool = QThreadPool(self)
        self._preview_thread_pool.setMaxThreadCount(1)
        self._preview_thread_pool.setExpiryTimeout(-1)
        self._preview_update_timer = QTimer(self)
        self._preview_update_timer.setSingleShot(True)
        self._preview_update_timer.setInterval(180)
        self._preview_update_timer.timeout.connect(self._start_pending_preview_processing)
        self._preview_request_serial = 0
        self._preview_running_request_id: int | None = None
        self._preview_pending_request: PreviewProcessingRequest | None = None
        self._preview_running_signature: tuple[str, str, str] | None = None
        self._preview_pending_signature: tuple[str, str, str] | None = None
        self._prepared_image_thread_pool = QThreadPool(self)
        self._prepared_image_thread_pool.setMaxThreadCount(1)
        self._prepared_image_thread_pool.setExpiryTimeout(-1)
        self._prepared_image_request_serial = 0
        self._prepared_image_running_request_id: int | None = None
        self._prepared_image_pending_request: PreparedImageRequest | None = None
        self._prepared_image_running_signature: tuple[str, str] | None = None
        self._prepared_image_pending_signature: tuple[str, str] | None = None

        self._batch_processor = BatchProcessor(self)
        self._batch_processor.set_ui_language(self._ui_language)
        self._batch_processor.resultReady.connect(self._on_batch_result)
        self._batch_processor.progressChanged.connect(self._on_batch_progress)
        self._batch_processor.finished.connect(self._on_batch_finished)
        self._batch_processor.errorOccurred.connect(self._on_batch_error)
        self._batch_processor.logMessage.connect(self._append_log)

        self._build_ui()
        self._apply_compact_ui_style()
        self._restore_persisted_paths()
        self._populate_pipeline_operations()
        self._populate_pipeline_list()
        self._apply_display_settings()
        self.set_ui_language(self._ui_language)

    def _build_ui(self) -> None:
        root_layout = QVBoxLayout(self)

        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.setChildrenCollapsible(False)
        root_layout.addWidget(main_splitter, 1)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setMinimumWidth(360)
        left_scroll.setMaximumWidth(560)
        controls_container = QWidget()
        left_scroll.setWidget(controls_container)
        controls_layout = QVBoxLayout(controls_container)
        self.control_tabs = self._build_tabs()
        controls_layout.addWidget(self.control_tabs, 1)
        main_splitter.addWidget(left_scroll)
        self.visual_panel = self._build_visual_panel()
        main_splitter.addWidget(self.visual_panel)
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)
        main_splitter.setSizes([380, 1280])

    def _apply_compact_ui_style(self) -> None:
        self.setStyleSheet(
            """
            #polygonExtractionWidget {
                font-size: 12px;
            }
            #polygonExtractionWidget QLabel,
            #polygonExtractionWidget QCheckBox,
            #polygonExtractionWidget QGroupBox {
                font-size: 12px;
            }
            #polygonExtractionWidget QPushButton {
                min-height: 28px;
                padding: 4px 10px;
                font-size: 12px;
            }
            #polygonExtractionWidget QToolButton {
                padding: 2px;
            }
            #polygonExtractionWidget QLineEdit,
            #polygonExtractionWidget QComboBox,
            #polygonExtractionWidget QSpinBox,
            #polygonExtractionWidget QDoubleSpinBox {
                min-height: 26px;
                padding: 2px 6px;
                font-size: 12px;
            }
            #polygonExtractionWidget QTabBar::tab {
                min-height: 24px;
                padding: 4px 10px;
                font-size: 12px;
            }
            #polygonExtractionWidget QListWidget {
                font-size: 12px;
            }
            #polygonExtractionWidget QProgressBar {
                min-height: 18px;
                max-height: 18px;
            }
            """
        )

    def _build_path_panel(self) -> QWidget:
        self.path_group = QGroupBox("Input / Output")
        layout = QVBoxLayout(self.path_group)

        self.input_dir_edit = QLineEdit()
        self.cif_dir_edit = QLineEdit()
        self.output_dir_edit = QLineEdit()
        self.input_dir_label = QLabel("Input directory")
        self.cif_dir_label = QLabel("CIF overlay directory")
        self.output_dir_label = QLabel("Output directory")
        self.browse_input_button = QPushButton("Browse input")
        self.browse_cif_button = QPushButton("Browse CIF")
        self.browse_output_button = QPushButton("Browse output")
        self.refresh_button = QPushButton("Refresh files")

        self.browse_input_button.clicked.connect(self._select_input_directory)
        self.browse_cif_button.clicked.connect(self._select_cif_directory)
        self.browse_output_button.clicked.connect(self._select_output_directory)
        self.refresh_button.clicked.connect(self.refresh_image_list)
        self.input_dir_edit.editingFinished.connect(self._apply_input_directory_edit)
        self.cif_dir_edit.editingFinished.connect(self._apply_cif_directory_edit)
        self.output_dir_edit.editingFinished.connect(self._apply_output_directory_edit)

        for label, edit, button in [
            (self.input_dir_label, self.input_dir_edit, self.browse_input_button),
            (self.cif_dir_label, self.cif_dir_edit, self.browse_cif_button),
            (self.output_dir_label, self.output_dir_edit, self.browse_output_button),
        ]:
            layout.addWidget(label)
            layout.addWidget(edit)
            layout.addWidget(button)
        layout.addWidget(self.refresh_button)
        return self.path_group

    def _build_paths_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.path_panel = self._build_path_panel()
        layout.addWidget(self.path_panel)
        layout.addStretch(1)
        return tab

    def _build_tabs(self) -> QWidget:
        tabs = QTabWidget()
        tabs.setUsesScrollButtons(True)
        self.paths_tab = self._build_paths_tab()
        self.files_tab = self._build_files_tab()
        self.pipeline_tab = self._build_pipeline_tab()
        self.extraction_tab = self._build_extraction_tab()
        self.display_tab = self._build_display_tab()
        tabs.addTab(self.paths_tab, "Paths")
        tabs.addTab(self.files_tab, "Files")
        tabs.addTab(self.pipeline_tab, "Pipeline")
        tabs.addTab(self.extraction_tab, "Extraction")
        tabs.addTab(self.display_tab, "Display")
        return tabs

    def _restore_persisted_paths(self) -> None:
        paths = self._path_settings_store.load()

        if paths.output_directory:
            self.set_output_directory(paths.output_directory)
        if paths.cif_directory:
            self.set_cif_directory(paths.cif_directory)
        if paths.input_directory:
            self.set_input_directory(paths.input_directory)

    def _save_persisted_paths(self) -> None:
        self._path_settings_store.save(
            PersistedPaths(
                input_directory=self.input_dir_edit.text().strip(),
                cif_directory=self.cif_dir_edit.text().strip(),
                output_directory=self.output_dir_edit.text().strip(),
            )
        )

    def _build_files_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.image_list = QListWidget()
        self.image_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.image_list.currentItemChanged.connect(self._on_image_item_changed)
        self.images_label = QLabel("Images")
        layout.addWidget(self.images_label)
        layout.addWidget(self.image_list, 1)

        self.run_group = QGroupBox("Run")
        run_layout = QGridLayout(self.run_group)
        self.process_current_button = QPushButton("Process current")
        self.process_current_button.clicked.connect(self.process_current_image)
        self.batch_button = QPushButton("Start batch")
        self.batch_button.clicked.connect(self.start_batch_processing)
        self.stop_batch_button = QPushButton("Stop batch")
        self.stop_batch_button.clicked.connect(self.stop_batch_processing)
        self.save_current_button = QPushButton("Save current result")
        self.save_current_button.clicked.connect(self.save_current_result)
        self.max_workers_spin = QSpinBox()
        self.max_workers_spin.setRange(1, 32)
        self.max_workers_spin.setValue(4)
        self.max_workers_label = QLabel("Max workers")
        run_layout.addWidget(self.process_current_button, 0, 0, 1, 2)
        run_layout.addWidget(self.batch_button, 1, 0, 1, 2)
        run_layout.addWidget(self.stop_batch_button, 2, 0, 1, 2)
        run_layout.addWidget(self.max_workers_label, 3, 0)
        run_layout.addWidget(self.max_workers_spin, 3, 1)
        run_layout.addWidget(self.save_current_button, 4, 0, 1, 2)
        layout.addWidget(self.run_group)
        self.batch_progress_bar = QProgressBar()
        self.batch_progress_bar.setRange(0, 100)
        self.batch_progress_bar.setValue(0)
        self.batch_progress_bar.setFormat("%p% (%v/%m)")
        self.batch_progress_bar.setTextVisible(True)
        self.batch_progress_bar.setVisible(False)
        layout.addWidget(self.batch_progress_bar)
        return tab

    def _build_pipeline_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        top_panel = QWidget()
        top_layout = QVBoxLayout(top_panel)

        header_layout = QHBoxLayout()
        self.operation_selector = QComboBox()
        self.add_step_button = QPushButton("Add step")
        self.add_step_button.clicked.connect(self._add_pipeline_step)
        header_layout.addWidget(self.operation_selector, 1)
        header_layout.addWidget(self.add_step_button)
        top_layout.addLayout(header_layout)

        self.pipeline_list = QListWidget()
        self.pipeline_list.currentRowChanged.connect(self._on_pipeline_step_selected)
        self.pipeline_list.itemChanged.connect(self._on_pipeline_item_changed)
        self.pipeline_list.setMinimumHeight(120)
        self.pipeline_list.setMaximumHeight(220)
        top_layout.addWidget(self.pipeline_list)

        buttons_layout = QVBoxLayout()
        self.remove_step_button = QPushButton("Remove")
        self.remove_step_button.clicked.connect(self._remove_pipeline_step)
        self.move_up_step_button = QPushButton("Up")
        self.move_up_step_button.clicked.connect(self._move_pipeline_step_up)
        self.move_down_step_button = QPushButton("Down")
        self.move_down_step_button.clicked.connect(self._move_pipeline_step_down)
        buttons_layout.addWidget(self.remove_step_button)
        buttons_layout.addWidget(self.move_up_step_button)
        buttons_layout.addWidget(self.move_down_step_button)
        top_layout.addLayout(buttons_layout)

        apply_layout = QVBoxLayout()
        self.auto_apply_checkbox = QCheckBox("Auto apply")
        self.auto_apply_checkbox.setChecked(True)
        self.apply_pipeline_button = QPushButton("Apply to current image")
        self.apply_pipeline_button.clicked.connect(self.process_current_image)
        self.save_pipeline_button = QPushButton("Save JSON")
        self.save_pipeline_button.clicked.connect(self._save_pipeline_json)
        self.load_pipeline_button = QPushButton("Load JSON")
        self.load_pipeline_button.clicked.connect(self._load_pipeline_json)
        apply_layout.addWidget(self.auto_apply_checkbox)
        apply_layout.addWidget(self.apply_pipeline_button)
        apply_layout.addWidget(self.save_pipeline_button)
        apply_layout.addWidget(self.load_pipeline_button)
        top_layout.addLayout(apply_layout)

        self.parameters_group = QGroupBox("Step parameters")
        parameters_scroll = QScrollArea()
        parameters_scroll.setWidgetResizable(True)
        parameters_widget = QWidget()
        self.parameters_form = QFormLayout(parameters_widget)
        parameters_scroll.setWidget(parameters_widget)
        group_layout = QVBoxLayout(self.parameters_group)
        group_layout.addWidget(parameters_scroll)
        pipeline_splitter = QSplitter(Qt.Orientation.Vertical)
        pipeline_splitter.setChildrenCollapsible(False)
        pipeline_splitter.addWidget(top_panel)
        pipeline_splitter.addWidget(self.parameters_group)
        pipeline_splitter.setStretchFactor(0, 0)
        pipeline_splitter.setStretchFactor(1, 1)
        pipeline_splitter.setSizes([260, 520])
        layout.addWidget(pipeline_splitter, 1)
        return tab

    def _build_extraction_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.contour_group = QGroupBox("Contour extraction")
        contour_form = QFormLayout(self.contour_group)
        self.retrieval_mode_combo = QComboBox()
        for mode_name in RETRIEVAL_MODE_MAP:
            self.retrieval_mode_combo.addItem(mode_name, mode_name)
        self.retrieval_mode_combo.setCurrentIndex(self.retrieval_mode_combo.findData("RETR_EXTERNAL"))
        self.approximation_mode_combo = QComboBox()
        for mode_name in APPROXIMATION_MODE_MAP:
            self.approximation_mode_combo.addItem(mode_name, mode_name)
        self.approximation_mode_combo.setCurrentIndex(self.approximation_mode_combo.findData("CHAIN_APPROX_SIMPLE"))
        self.epsilon_spin = QDoubleSpinBox()
        self.epsilon_spin.setRange(0.0, 1000.0)
        self.epsilon_spin.setDecimals(3)
        self.epsilon_spin.setValue(2.0)
        self.epsilon_relative_checkbox = QCheckBox("Relative to contour perimeter")
        self.min_area_spin = QDoubleSpinBox()
        self.min_area_spin.setRange(0.0, 1_000_000_000.0)
        self.min_area_spin.setValue(10.0)
        self.max_area_spin = QDoubleSpinBox()
        self.max_area_spin.setRange(0.0, 1_000_000_000.0)
        self.max_area_spin.setValue(0.0)
        self.min_perimeter_spin = QDoubleSpinBox()
        self.min_perimeter_spin.setRange(0.0, 1_000_000_000.0)
        self.min_perimeter_spin.setValue(10.0)
        self.min_points_spin = QSpinBox()
        self.min_points_spin.setRange(3, 10_000)
        self.min_points_spin.setValue(3)
        self.retrieval_mode_combo.currentIndexChanged.connect(self._on_extraction_settings_changed)
        self.approximation_mode_combo.currentIndexChanged.connect(self._on_extraction_settings_changed)
        self.epsilon_spin.valueChanged.connect(self._on_extraction_settings_changed)
        self.epsilon_relative_checkbox.stateChanged.connect(self._on_extraction_settings_changed)
        self.min_area_spin.valueChanged.connect(self._on_extraction_settings_changed)
        self.max_area_spin.valueChanged.connect(self._on_extraction_settings_changed)
        self.min_perimeter_spin.valueChanged.connect(self._on_extraction_settings_changed)
        self.min_points_spin.valueChanged.connect(self._on_extraction_settings_changed)

        contour_form.addRow("Retrieval mode", self.retrieval_mode_combo)
        self.retrieval_mode_label_widget = contour_form.labelForField(self.retrieval_mode_combo)
        contour_form.addRow("Approximation mode", self.approximation_mode_combo)
        self.approximation_mode_label_widget = contour_form.labelForField(self.approximation_mode_combo)
        contour_form.addRow("Epsilon", self.epsilon_spin)
        self.epsilon_label_widget = contour_form.labelForField(self.epsilon_spin)
        contour_form.addRow("Epsilon mode", self.epsilon_relative_checkbox)
        self.epsilon_mode_label_widget = contour_form.labelForField(self.epsilon_relative_checkbox)
        contour_form.addRow("Min area", self.min_area_spin)
        self.min_area_label_widget = contour_form.labelForField(self.min_area_spin)
        contour_form.addRow("Max area (0 = unlimited)", self.max_area_spin)
        self.max_area_label_widget = contour_form.labelForField(self.max_area_spin)
        contour_form.addRow("Min perimeter", self.min_perimeter_spin)
        self.min_perimeter_label_widget = contour_form.labelForField(self.min_perimeter_spin)
        contour_form.addRow("Min point count", self.min_points_spin)
        self.min_point_count_label_widget = contour_form.labelForField(self.min_points_spin)
        layout.addWidget(self.contour_group)

        self.save_group = QGroupBox("Save options")
        save_layout = QVBoxLayout(self.save_group)
        self.save_cif_checkbox = QCheckBox("CIF")
        self.save_cif_checkbox.setChecked(True)
        self.save_csv_checkbox = QCheckBox("CSV")
        self.save_txt_checkbox = QCheckBox("TXT")
        self.save_svg_checkbox = QCheckBox("SVG preview")
        self.save_preview_checkbox = QCheckBox("Overlay preview image")
        self.save_preview_checkbox.setChecked(True)
        for checkbox in [
            self.save_cif_checkbox,
            self.save_csv_checkbox,
            self.save_txt_checkbox,
            self.save_svg_checkbox,
            self.save_preview_checkbox,
        ]:
            save_layout.addWidget(checkbox)
        layout.addWidget(self.save_group)
        layout.addStretch(1)
        return tab

    def _build_display_tab(self) -> QWidget:
        tab = QWidget()
        self.display_form = QFormLayout(tab)

        self.external_color_button = self._build_color_button(self._display_settings.external_color, self._choose_external_color)
        self.hole_color_button = self._build_color_button(self._display_settings.hole_color, self._choose_hole_color)
        self.selected_color_button = self._build_color_button(self._display_settings.selected_color, self._choose_selected_color)
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

        for widget in [
            self.line_width_spin,
            self.vertex_size_spin,
            self.fill_opacity_spin,
            self.show_vertices_checkbox,
            self.show_labels_checkbox,
        ]:
            if isinstance(widget, QCheckBox):
                widget.stateChanged.connect(self._apply_display_settings)
            else:
                widget.valueChanged.connect(self._apply_display_settings)

        self.display_form.addRow("External contour", self.external_color_button)
        self.external_color_label_widget = self.display_form.labelForField(self.external_color_button)
        self.display_form.addRow("Hole contour", self.hole_color_button)
        self.hole_color_label_widget = self.display_form.labelForField(self.hole_color_button)
        self.display_form.addRow("Selected contour", self.selected_color_button)
        self.selected_color_label_widget = self.display_form.labelForField(self.selected_color_button)
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
        return tab

    def _build_visual_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        self.editor_group = QGroupBox("Image / polygon editor")
        editor_layout = QVBoxLayout(self.editor_group)
        self.polygon_editor = PolygonEditorView()
        self.polygon_editor.polygonsEdited.connect(self._on_polygons_edited)
        self.polygon_editor.logRequested.connect(self._append_log)
        self.editor_toolbar = self._build_editor_toolbar()
        editor_layout.addWidget(self.editor_toolbar)
        editor_layout.addWidget(self.polygon_editor, 1)

        layout.addWidget(self.editor_group, 1)
        return panel

    def _build_editor_toolbar(self) -> QWidget:
        toolbar = QWidget()
        layout = QHBoxLayout(toolbar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self._tool_button_group = QButtonGroup(self)
        self._tool_button_group.setExclusive(True)
        self._tool_buttons: dict[EditorTool, QToolButton] = {}
        for text, tool in [
            ("Select", EditorTool.SELECT),
            ("Pan", EditorTool.PAN),
            ("Add Polygon", EditorTool.ADD_POLYGON),
            ("Brush", EditorTool.BRUSH),
            ("Add Vertex", EditorTool.ADD_VERTEX),
            ("Delete Vertex", EditorTool.DELETE_VERTEX),
            ("Move Vertex", EditorTool.MOVE_VERTEX),
            ("Delete Polygon", EditorTool.DELETE_POLYGON),
        ]:
            button = QToolButton()
            self._configure_toolbar_button(button, self._create_editor_tool_icon(tool), text, checkable=True)
            button.clicked.connect(lambda checked=False, tool_value=tool: self.polygon_editor.set_tool(tool_value))
            self._tool_button_group.addButton(button)
            self._tool_buttons[tool] = button
            layout.addWidget(button)
            if tool == EditorTool.SELECT:
                button.setChecked(True)

        self.polygon_mode_label = QLabel("Polygon")
        self.polygon_mode_combo = QComboBox()
        self.polygon_mode_combo.addItem(self._mode_text("polygon_points"), PolygonCreateMode.POINTS)
        self.polygon_mode_combo.addItem(self._mode_text("polygon_rectangle"), PolygonCreateMode.RECTANGLE)
        self.polygon_mode_combo.currentIndexChanged.connect(
            lambda _index: self.polygon_editor.set_polygon_create_mode(self.polygon_mode_combo.currentData())
        )
        layout.addWidget(self.polygon_mode_label)
        layout.addWidget(self.polygon_mode_combo)

        self.brush_mode_label = QLabel("Brush")
        self.brush_mode_combo = QComboBox()
        self.brush_mode_combo.addItem(self._mode_text("brush_freeform"), BrushMode.FREEFORM)
        self.brush_mode_combo.addItem(self._mode_text("brush_45deg"), BrushMode.ANGLED)
        self.brush_mode_combo.currentIndexChanged.connect(
            lambda _index: self.polygon_editor.set_brush_mode(self.brush_mode_combo.currentData())
        )
        layout.addWidget(self.brush_mode_label)
        layout.addWidget(self.brush_mode_combo)

        self.brush_size_label = QLabel("Толщина" if self._ui_language == "ru" else "Width")
        self.brush_size_spin = QSpinBox()
        self.brush_size_spin.setRange(1, 256)
        self.brush_size_spin.setValue(12)
        self.brush_size_spin.setFixedWidth(68)
        self.brush_size_spin.valueChanged.connect(
            lambda value: self.polygon_editor.set_brush_thickness(float(value))
        )
        layout.addWidget(self.brush_size_label)
        layout.addWidget(self.brush_size_spin)

        self.delete_vertex_mode_label = QLabel("Delete")
        self.delete_vertex_mode_combo = QComboBox()
        self.delete_vertex_mode_combo.addItem(self._mode_text("delete_single"), DeleteVertexMode.SINGLE)
        self.delete_vertex_mode_combo.addItem(self._mode_text("delete_area"), DeleteVertexMode.AREA)
        self.delete_vertex_mode_combo.currentIndexChanged.connect(
            lambda _index: self.polygon_editor.set_delete_vertex_mode(self.delete_vertex_mode_combo.currentData())
        )
        layout.addWidget(self.delete_vertex_mode_label)
        layout.addWidget(self.delete_vertex_mode_combo)

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

        self.preview_busy_label = QLabel(self._busy_indicator_text())
        self.preview_busy_progress = QProgressBar()
        self.preview_busy_progress.setRange(0, 0)
        self.preview_busy_progress.setTextVisible(False)
        self.preview_busy_progress.setFixedWidth(88)
        self.preview_busy_label.setVisible(False)
        self.preview_busy_progress.setVisible(False)
        layout.addWidget(self.preview_busy_label)
        layout.addWidget(self.preview_busy_progress)
        layout.addStretch(1)
        self.polygon_editor.set_polygon_create_mode(self.polygon_mode_combo.currentData())
        self.polygon_editor.set_brush_mode(self.brush_mode_combo.currentData())
        self.polygon_editor.set_brush_thickness(float(self.brush_size_spin.value()))
        self.polygon_editor.set_delete_vertex_mode(self.delete_vertex_mode_combo.currentData())
        return toolbar

    def _configure_toolbar_button(
        self,
        button: QToolButton,
        icon: QIcon,
        text: str,
        *,
        checkable: bool = False,
    ) -> None:
        button.setIcon(icon)
        button.setIconSize(QSize(self._toolbar_icon_size_px(), self._toolbar_icon_size_px()))
        button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        button.setToolTip(text)
        button.setStatusTip(text)
        button.setAccessibleName(text)
        button.setAutoRaise(False)
        button.setFixedSize(self._toolbar_button_size_px(), self._toolbar_button_size_px())
        button.setCheckable(checkable)

    def _create_editor_tool_icon(self, tool: EditorTool) -> QIcon:
        canvas_size = self._toolbar_icon_canvas_size_px()
        pixmap = QPixmap(canvas_size, canvas_size)
        pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        scale_factor = canvas_size / 28.0
        painter.scale(scale_factor, scale_factor)

        stroke = QColor("#FFFFFF")
        neutral = QColor("#E2E8F0")
        accent = QColor("#38BDF8")
        success = QColor("#4ADE80")
        warning = QColor("#FDBA74")
        danger = QColor("#FB7185")

        if tool == EditorTool.SELECT:
            self._paint_select_icon(painter, stroke)
        elif tool == EditorTool.PAN:
            self._paint_pan_icon(painter, stroke, accent)
        elif tool == EditorTool.ADD_POLYGON:
            self._paint_polygon_badge_icon(painter, stroke, accent, "+")
        elif tool == EditorTool.BRUSH:
            self._paint_brush_icon(painter, stroke, success)
        elif tool == EditorTool.ADD_VERTEX:
            self._paint_vertex_edit_icon(painter, stroke, neutral, success, "+")
        elif tool == EditorTool.DELETE_VERTEX:
            self._paint_vertex_edit_icon(painter, stroke, neutral, danger, "-")
        elif tool == EditorTool.MOVE_VERTEX:
            self._paint_move_vertex_icon(painter, stroke, warning)
        elif tool == EditorTool.DELETE_POLYGON:
            self._paint_polygon_badge_icon(painter, stroke, danger, "x")
        painter.end()
        return QIcon(pixmap)

    def _create_editor_action_icon(self, action: str) -> QIcon:
        canvas_size = self._toolbar_icon_canvas_size_px()
        pixmap = QPixmap(canvas_size, canvas_size)
        pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        scale_factor = canvas_size / 28.0
        painter.scale(scale_factor, scale_factor)
        stroke = QColor("#FFFFFF")
        accent = QColor("#38BDF8")

        if action == "undo":
            self._paint_history_icon(painter, stroke, mirrored=False)
        elif action == "redo":
            self._paint_history_icon(painter, stroke, mirrored=True)
        elif action == "zoom_in":
            self._paint_zoom_icon(painter, stroke, accent, add=True)
        elif action == "zoom_out":
            self._paint_zoom_icon(painter, stroke, accent, add=False)
        else:
            self._paint_fit_icon(painter, stroke, accent)
        painter.end()
        return QIcon(pixmap)

    @staticmethod
    def _toolbar_icon_size_px() -> int:
        return 28

    @staticmethod
    def _toolbar_button_size_px() -> int:
        return 34

    @staticmethod
    def _toolbar_icon_canvas_size_px() -> int:
        return 72

    def _paint_select_icon(self, painter: QPainter, stroke: QColor) -> None:
        path = QPainterPath()
        path.moveTo(5.5, 4.0)
        path.lineTo(5.5, 21.0)
        path.lineTo(10.0, 16.5)
        path.lineTo(12.8, 23.0)
        path.lineTo(16.0, 21.8)
        path.lineTo(13.2, 15.6)
        path.lineTo(20.8, 15.6)
        path.closeSubpath()
        painter.setPen(QPen(stroke, 1.9, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        painter.fillPath(path, QBrush(QColor("#FFFFFF")))
        painter.drawPath(path)

    def _paint_pan_icon(self, painter: QPainter, stroke: QColor, accent: QColor) -> None:
        center = QPointF(14.0, 14.0)
        painter.setPen(QPen(stroke, 2.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(QPointF(14.0, 5.0), QPointF(14.0, 23.0))
        painter.drawLine(QPointF(5.0, 14.0), QPointF(23.0, 14.0))
        self._draw_arrow_head(painter, QPointF(14.0, 5.0), QPointF(14.0, 2.0))
        self._draw_arrow_head(painter, QPointF(14.0, 23.0), QPointF(14.0, 26.0))
        self._draw_arrow_head(painter, QPointF(5.0, 14.0), QPointF(2.0, 14.0))
        self._draw_arrow_head(painter, QPointF(23.0, 14.0), QPointF(26.0, 14.0))
        painter.setPen(QPen(accent, 1.8))
        painter.setBrush(QBrush(accent))
        painter.drawEllipse(QRectF(center.x() - 2.2, center.y() - 2.2, 4.4, 4.4))

    def _paint_polygon_badge_icon(
        self,
        painter: QPainter,
        stroke: QColor,
        badge_color: QColor,
        badge_symbol: str,
    ) -> None:
        polygon = QPolygonF(
            [
                QPointF(4.5, 18.5),
                QPointF(8.6, 7.0),
                QPointF(18.0, 8.6),
                QPointF(20.0, 18.0),
                QPointF(12.0, 22.0),
            ]
        )
        painter.setPen(QPen(stroke, 2.1, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPolygon(polygon)
        for point in polygon:
            self._draw_vertex_marker(painter, point, stroke, QColor("#FFFFFF"), radius=1.9)
        self._draw_badge(painter, QPointF(20.5, 6.5), badge_color, badge_symbol)

    def _paint_vertex_edit_icon(
        self,
        painter: QPainter,
        stroke: QColor,
        neutral: QColor,
        badge_color: QColor,
        badge_symbol: str,
    ) -> None:
        polyline = [QPointF(4.5, 18.0), QPointF(11.0, 8.0), QPointF(19.0, 18.2)]
        painter.setPen(QPen(stroke, 2.1, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        painter.drawPolyline(QPolygonF(polyline))
        self._draw_vertex_marker(painter, polyline[0], stroke, QColor("#FFFFFF"), radius=1.8)
        self._draw_vertex_marker(painter, polyline[2], stroke, QColor("#FFFFFF"), radius=1.8)
        self._draw_vertex_marker(painter, polyline[1], stroke, neutral, radius=2.4)
        self._draw_badge(painter, QPointF(20.0, 6.5), badge_color, badge_symbol)

    def _paint_move_vertex_icon(self, painter: QPainter, stroke: QColor, accent: QColor) -> None:
        polygon = QPolygonF(
            [
                QPointF(4.5, 18.4),
                QPointF(8.8, 8.0),
                QPointF(17.0, 9.0),
                QPointF(19.6, 17.5),
                QPointF(11.4, 21.2),
            ]
        )
        painter.setPen(QPen(stroke, 2.1, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPolygon(polygon)
        target = QPointF(17.0, 9.0)
        self._draw_vertex_marker(painter, target, stroke, accent, radius=2.5)
        painter.setPen(QPen(accent, 1.9, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(QPointF(17.0, 4.2), QPointF(17.0, 13.8))
        painter.drawLine(QPointF(12.2, 9.0), QPointF(21.8, 9.0))
        self._draw_arrow_head(painter, QPointF(17.0, 4.2), QPointF(17.0, 1.6))
        self._draw_arrow_head(painter, QPointF(17.0, 13.8), QPointF(17.0, 16.4))
        self._draw_arrow_head(painter, QPointF(12.2, 9.0), QPointF(9.6, 9.0))
        self._draw_arrow_head(painter, QPointF(21.8, 9.0), QPointF(24.4, 9.0))

    def _paint_brush_icon(self, painter: QPainter, stroke: QColor, accent: QColor) -> None:
        painter.setPen(QPen(stroke, 2.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        path = QPainterPath()
        path.moveTo(7.0, 20.5)
        path.cubicTo(9.0, 15.0, 13.0, 10.0, 18.0, 6.5)
        path.lineTo(21.0, 9.5)
        path.cubicTo(17.5, 14.5, 12.5, 18.5, 7.0, 20.5)
        painter.drawPath(path)
        painter.setBrush(QBrush(accent))
        painter.setPen(QPen(accent, 1.0))
        painter.drawEllipse(QRectF(18.8, 5.2, 4.6, 4.6))

    def _paint_history_icon(self, painter: QPainter, stroke: QColor, mirrored: bool) -> None:
        painter.setPen(QPen(stroke, 2.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        path = QPainterPath()
        if mirrored:
            path.moveTo(7.0, 8.0)
            path.cubicTo(13.0, 3.5, 22.5, 6.0, 22.0, 14.0)
            path.cubicTo(21.5, 21.0, 13.5, 23.5, 8.5, 20.0)
            painter.drawPath(path)
            self._draw_arrow_head(painter, QPointF(7.2, 8.0), QPointF(3.8, 9.0))
        else:
            path.moveTo(21.0, 8.0)
            path.cubicTo(15.0, 3.5, 5.5, 6.0, 6.0, 14.0)
            path.cubicTo(6.5, 21.0, 14.5, 23.5, 19.5, 20.0)
            painter.drawPath(path)
            self._draw_arrow_head(painter, QPointF(20.8, 8.0), QPointF(24.2, 9.0))

    def _paint_zoom_icon(self, painter: QPainter, stroke: QColor, accent: QColor, *, add: bool) -> None:
        painter.setPen(QPen(stroke, 2.1, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(QRectF(5.0, 5.0, 12.0, 12.0))
        painter.drawLine(QPointF(15.2, 15.2), QPointF(22.8, 22.8))
        painter.setPen(QPen(accent, 2.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(QPointF(8.5, 11.0), QPointF(13.5, 11.0))
        if add:
            painter.drawLine(QPointF(11.0, 8.5), QPointF(11.0, 13.5))

    def _paint_fit_icon(self, painter: QPainter, stroke: QColor, accent: QColor) -> None:
        painter.setPen(QPen(stroke, 2.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawRect(QRectF(7.0, 7.0, 14.0, 14.0))
        painter.setPen(QPen(accent, 2.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(QPointF(5.0, 10.0), QPointF(9.0, 10.0))
        painter.drawLine(QPointF(10.0, 5.0), QPointF(10.0, 9.0))
        painter.drawLine(QPointF(19.0, 5.0), QPointF(19.0, 9.0))
        painter.drawLine(QPointF(19.0, 19.0), QPointF(19.0, 23.0))
        painter.drawLine(QPointF(5.0, 19.0), QPointF(9.0, 19.0))
        painter.drawLine(QPointF(19.0, 19.0), QPointF(23.0, 19.0))

    def _draw_vertex_marker(
        self,
        painter: QPainter,
        point: QPointF,
        stroke: QColor,
        fill: QColor,
        radius: float,
    ) -> None:
        painter.setPen(QPen(stroke, 1.2))
        painter.setBrush(QBrush(fill))
        painter.drawEllipse(QRectF(point.x() - radius, point.y() - radius, radius * 2.0, radius * 2.0))

    def _draw_badge(self, painter: QPainter, center: QPointF, color: QColor, symbol: str) -> None:
        badge_rect = QRectF(center.x() - 4.3, center.y() - 4.3, 8.6, 8.6)
        painter.setPen(QPen(color.darker(120), 1.0))
        painter.setBrush(QBrush(color))
        painter.drawEllipse(badge_rect)
        painter.setPen(QPen(QColor("#FFFFFF"), 1.7, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        if symbol == "+":
            painter.drawLine(QPointF(center.x() - 2.2, center.y()), QPointF(center.x() + 2.2, center.y()))
            painter.drawLine(QPointF(center.x(), center.y() - 2.2), QPointF(center.x(), center.y() + 2.2))
        elif symbol == "-":
            painter.drawLine(QPointF(center.x() - 2.2, center.y()), QPointF(center.x() + 2.2, center.y()))
        else:
            painter.drawLine(
                QPointF(center.x() - 1.9, center.y() - 1.9),
                QPointF(center.x() + 1.9, center.y() + 1.9),
            )
            painter.drawLine(
                QPointF(center.x() - 1.9, center.y() + 1.9),
                QPointF(center.x() + 1.9, center.y() - 1.9),
            )

    def _draw_arrow_head(self, painter: QPainter, base: QPointF, tip: QPointF) -> None:
        vector_x = tip.x() - base.x()
        vector_y = tip.y() - base.y()
        if abs(vector_x) >= abs(vector_y):
            direction = 1.0 if vector_x >= 0 else -1.0
            left = QPointF(base.x() + 1.6 * direction, base.y() - 1.4)
            right = QPointF(base.x() + 1.6 * direction, base.y() + 1.4)
        else:
            direction = 1.0 if vector_y >= 0 else -1.0
            left = QPointF(base.x() - 1.4, base.y() + 1.6 * direction)
            right = QPointF(base.x() + 1.4, base.y() + 1.6 * direction)
        painter.drawLine(base, left)
        painter.drawLine(base, right)

    def _tr(self, key: str, default: str = "", **kwargs) -> str:
        return tr(key, default=default, language=self._ui_language, **kwargs)

    def _mode_text(self, key: str) -> str:
        if self._ui_language == "ru":
            mapping = {
                "polygon_points": "По точкам",
                "polygon_rectangle": "Прямоугольник",
                "brush_freeform": "Произвольная",
                "brush_45deg": "45° шаг",
                "delete_single": "Вершина",
                "delete_area": "Область",
            }
        else:
            mapping = {
                "polygon_points": "By points",
                "polygon_rectangle": "Rectangle",
                "brush_freeform": "Freeform",
                "brush_45deg": "45° constrained",
                "delete_single": "Single vertex",
                "delete_area": "Area",
            }
        return mapping[key]

    def _busy_indicator_text(self) -> str:
        return "Обработка..." if self._ui_language == "ru" else "Processing..."

    def _set_progress_status(self, key: str, **kwargs) -> None:
        self._progress_status_key = key
        self._progress_status_kwargs = dict(kwargs)

    def set_ui_language(self, language: str | None) -> None:
        self._ui_language = active_language(language)
        self._batch_processor.set_ui_language(self._ui_language)
        if hasattr(self, "polygon_editor"):
            self.polygon_editor.set_ui_language(self._ui_language)
        self._retranslate_ui()

    def _retranslate_ui(self) -> None:
        if not hasattr(self, "control_tabs"):
            return
        selected_operation = self.operation_selector.currentData() if hasattr(self, "operation_selector") else None
        selected_pipeline_row = self.pipeline_list.currentRow() if hasattr(self, "pipeline_list") else -1

        self.path_group.setTitle(self._tr("path_panel_title"))
        self.input_dir_label.setText(self._tr("input_directory_label"))
        self.cif_dir_label.setText(self._tr("cif_overlay_directory_label"))
        self.output_dir_label.setText(self._tr("output_directory_label"))
        self.browse_input_button.setText(self._tr("browse_input_button"))
        self.browse_cif_button.setText(self._tr("browse_cif_button"))
        self.browse_output_button.setText(self._tr("browse_output_button"))
        self.refresh_button.setText(self._tr("refresh_files_button"))

        self.control_tabs.setTabText(0, self._tr("tab_paths"))
        self.control_tabs.setTabText(1, self._tr("tab_files"))
        self.control_tabs.setTabText(2, self._tr("tab_pipeline"))
        self.control_tabs.setTabText(3, self._tr("tab_extraction"))
        self.control_tabs.setTabText(4, self._tr("tab_display"))

        self.images_label.setText(self._tr("images_label"))
        self.run_group.setTitle(self._tr("run_group_title"))
        self.process_current_button.setText(self._tr("process_current_button"))
        self.batch_button.setText(self._tr("start_batch_button"))
        self.stop_batch_button.setText(self._tr("stop_batch_button"))
        self.save_current_button.setText(self._tr("save_current_button"))
        self.max_workers_label.setText(self._tr("max_workers_label"))

        self.add_step_button.setText(self._tr("add_step_button"))
        self.remove_step_button.setText(self._tr("remove_step_button"))
        self.move_up_step_button.setText(self._tr("move_up_button"))
        self.move_down_step_button.setText(self._tr("move_down_button"))
        self.auto_apply_checkbox.setText(self._tr("auto_apply_checkbox"))
        self.apply_pipeline_button.setText(self._tr("apply_current_button"))
        self.save_pipeline_button.setText(self._tr("save_json_button"))
        self.load_pipeline_button.setText(self._tr("load_json_button"))
        self.parameters_group.setTitle(self._tr("step_parameters_group"))

        self.contour_group.setTitle(self._tr("contour_extraction_group"))
        if self.retrieval_mode_label_widget is not None:
            self.retrieval_mode_label_widget.setText(self._tr("retrieval_mode_label"))
        if self.approximation_mode_label_widget is not None:
            self.approximation_mode_label_widget.setText(self._tr("approximation_mode_label"))
        if self.epsilon_label_widget is not None:
            self.epsilon_label_widget.setText(self._tr("epsilon_label"))
        if self.epsilon_mode_label_widget is not None:
            self.epsilon_mode_label_widget.setText(self._tr("epsilon_mode_label"))
        self.epsilon_relative_checkbox.setText(self._tr("epsilon_relative_checkbox"))
        if self.min_area_label_widget is not None:
            self.min_area_label_widget.setText(self._tr("min_area_label"))
        if self.max_area_label_widget is not None:
            self.max_area_label_widget.setText(self._tr("max_area_label"))
        if self.min_perimeter_label_widget is not None:
            self.min_perimeter_label_widget.setText(self._tr("min_perimeter_label"))
        if self.min_point_count_label_widget is not None:
            self.min_point_count_label_widget.setText(self._tr("min_point_count_label"))
        self.save_group.setTitle(self._tr("save_options_group"))
        self.save_svg_checkbox.setText(self._tr("save_svg_checkbox"))
        self.save_preview_checkbox.setText(self._tr("save_preview_checkbox"))

        if self.external_color_label_widget is not None:
            self.external_color_label_widget.setText(self._tr("external_contour_label"))
        if self.hole_color_label_widget is not None:
            self.hole_color_label_widget.setText(self._tr("hole_contour_label"))
        if self.selected_color_label_widget is not None:
            self.selected_color_label_widget.setText(self._tr("selected_contour_label"))
        if self.vertex_color_label_widget is not None:
            self.vertex_color_label_widget.setText(self._tr("vertex_color_label"))
        if self.line_width_label_widget is not None:
            self.line_width_label_widget.setText(self._tr("line_width_label"))
        if self.vertex_size_label_widget is not None:
            self.vertex_size_label_widget.setText(self._tr("vertex_size_label"))
        if self.fill_opacity_label_widget is not None:
            self.fill_opacity_label_widget.setText(self._tr("fill_opacity_label"))
        self.show_vertices_checkbox.setText(self._tr("show_vertices_checkbox"))
        self.show_labels_checkbox.setText(self._tr("show_labels_checkbox"))

        self.editor_group.setTitle(self._tr("editor_group_title"))
        self._update_tool_button_texts()
        self._update_action_button_texts()
        self.polygon_mode_label.setText("Полигон" if self._ui_language == "ru" else "Polygon")
        self.brush_mode_label.setText("Кисть" if self._ui_language == "ru" else "Brush")
        self.brush_size_label.setText("Толщина" if self._ui_language == "ru" else "Width")
        self.delete_vertex_mode_label.setText("Удаление" if self._ui_language == "ru" else "Delete")
        self._retranslate_editor_mode_combos()
        self.preview_busy_label.setText(self._busy_indicator_text())
        self._set_progress_status(self._progress_status_key, **self._progress_status_kwargs)

        self._populate_pipeline_operations()
        if selected_operation is not None:
            operation_index = self.operation_selector.findData(selected_operation)
            if operation_index >= 0:
                self.operation_selector.setCurrentIndex(operation_index)
        self._populate_pipeline_list()
        if selected_pipeline_row >= 0 and selected_pipeline_row < self.pipeline_list.count():
            self.pipeline_list.setCurrentRow(selected_pipeline_row)
        self._retranslate_contour_mode_combos()

    def _update_tool_button_texts(self) -> None:
        texts = {
            EditorTool.SELECT: self._tr("tool_select", "Выбор" if self._ui_language == "ru" else "Select"),
            EditorTool.PAN: self._tr("tool_pan", "Панорамирование" if self._ui_language == "ru" else "Pan"),
            EditorTool.ADD_POLYGON: self._tr("tool_add_polygon", "Полигон" if self._ui_language == "ru" else "Add polygon"),
            EditorTool.BRUSH: self._tr("tool_brush", "Кисть" if self._ui_language == "ru" else "Brush"),
            EditorTool.ADD_VERTEX: self._tr("tool_add_vertex", "Добавить вершину" if self._ui_language == "ru" else "Add vertex"),
            EditorTool.DELETE_VERTEX: self._tr("tool_delete_vertex", "Удалить вершину" if self._ui_language == "ru" else "Delete vertex"),
            EditorTool.MOVE_VERTEX: self._tr("tool_move_vertex", "Переместить вершину" if self._ui_language == "ru" else "Move vertex"),
            EditorTool.DELETE_POLYGON: self._tr("tool_delete_polygon", "Удалить полигон" if self._ui_language == "ru" else "Delete polygon"),
        }
        for tool, button in self._tool_buttons.items():
            label = texts.get(tool, tool.value)
            button.setToolTip(label)
            button.setStatusTip(label)
            button.setAccessibleName(label)

    def _update_action_button_texts(self) -> None:
        for button, label in [
            (self.undo_button, self._tr("undo_button", "Отменить" if self._ui_language == "ru" else "Undo")),
            (self.redo_button, self._tr("redo_button", "Повторить" if self._ui_language == "ru" else "Redo")),
            (self.zoom_in_button, self._tr("zoom_in_button", "Увеличить" if self._ui_language == "ru" else "Zoom in")),
            (self.zoom_out_button, self._tr("zoom_out_button", "Уменьшить" if self._ui_language == "ru" else "Zoom out")),
            (self.fit_button, self._tr("fit_button", "Подогнать" if self._ui_language == "ru" else "Fit")),
        ]:
            button.setToolTip(label)
            button.setStatusTip(label)
            button.setAccessibleName(label)

    def _retranslate_editor_mode_combos(self) -> None:
        polygon_mode = self.polygon_mode_combo.currentData()
        brush_mode = self.brush_mode_combo.currentData()
        delete_mode = self.delete_vertex_mode_combo.currentData()

        self.polygon_mode_combo.setItemText(0, self._mode_text("polygon_points"))
        self.polygon_mode_combo.setItemText(1, self._mode_text("polygon_rectangle"))
        self.brush_mode_combo.setItemText(0, self._mode_text("brush_freeform"))
        self.brush_mode_combo.setItemText(1, self._mode_text("brush_45deg"))
        self.delete_vertex_mode_combo.setItemText(0, self._mode_text("delete_single"))
        self.delete_vertex_mode_combo.setItemText(1, self._mode_text("delete_area"))

        polygon_index = self.polygon_mode_combo.findData(polygon_mode)
        brush_index = self.brush_mode_combo.findData(brush_mode)
        delete_index = self.delete_vertex_mode_combo.findData(delete_mode)
        if polygon_index >= 0:
            self.polygon_mode_combo.setCurrentIndex(polygon_index)
        if brush_index >= 0:
            self.brush_mode_combo.setCurrentIndex(brush_index)
        if delete_index >= 0:
            self.delete_vertex_mode_combo.setCurrentIndex(delete_index)

    def _retranslate_contour_mode_combos(self) -> None:
        current_retrieval = self.retrieval_mode_combo.currentData()
        for index in range(self.retrieval_mode_combo.count()):
            mode_name = str(self.retrieval_mode_combo.itemData(index))
            self.retrieval_mode_combo.setItemText(index, self._tr(f"retrieval_mode.{mode_name}", default=mode_name))
        if current_retrieval is not None:
            self.retrieval_mode_combo.setCurrentIndex(self.retrieval_mode_combo.findData(current_retrieval))

        current_approximation = self.approximation_mode_combo.currentData()
        for index in range(self.approximation_mode_combo.count()):
            mode_name = str(self.approximation_mode_combo.itemData(index))
            self.approximation_mode_combo.setItemText(
                index,
                self._tr(f"approximation_mode.{mode_name}", default=mode_name),
            )
        if current_approximation is not None:
            self.approximation_mode_combo.setCurrentIndex(self.approximation_mode_combo.findData(current_approximation))

    def _wrap_group(self, title: str, widget: QWidget) -> QWidget:
        group = QGroupBox(title)
        layout = QVBoxLayout(group)
        layout.addWidget(widget)
        return group

    def _build_color_button(self, color: str, handler) -> QPushButton:
        button = QPushButton(color)
        button.clicked.connect(handler)
        self._update_color_button(button, color)
        return button

    def _update_color_button(self, button: QPushButton, color_value: str) -> None:
        button.setText(color_value)
        button.setStyleSheet(f"background-color: {color_value}; color: #111111;")

    def _populate_pipeline_operations(self) -> None:
        self.operation_selector.clear()
        for descriptor in available_operations():
            self.operation_selector.addItem(
                get_operation_display_name(descriptor.type_name, self._ui_language),
                descriptor.type_name,
            )

    def _populate_pipeline_list(self) -> None:
        self._ignore_pipeline_item_change = True
        self.pipeline_list.clear()
        for step in self._pipeline.steps:
            label = get_operation_display_name(step.operation, self._ui_language)
            item = QListWidgetItem(label)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
            item.setData(Qt.ItemDataRole.UserRole, step.operation)
            item.setCheckState(Qt.CheckState.Checked if step.enabled else Qt.CheckState.Unchecked)
            self.pipeline_list.addItem(item)
        self._ignore_pipeline_item_change = False
        if self.pipeline_list.count():
            self.pipeline_list.setCurrentRow(0)
            self._render_pipeline_parameters(0)
        else:
            self._clear_parameters_form()

    def _clear_parameters_form(self) -> None:
        while self.parameters_form.count():
            item = self.parameters_form.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._parameter_widgets.clear()

    def _on_pipeline_step_selected(self, row: int) -> None:
        self._render_pipeline_parameters(row)

    def _render_pipeline_parameters(self, row: int) -> None:
        self._clear_parameters_form()
        if row < 0 or row >= len(self._pipeline.steps):
            return
        step = self._pipeline.steps[row]
        descriptor = get_operation_descriptor(step.operation)
        for spec in descriptor.parameters:
            value = step.parameters.get(spec.name, spec.default)
            if spec.kind == "bool":
                widget = QCheckBox()
                widget.setChecked(bool(value))
                widget.stateChanged.connect(
                    lambda _state, name=spec.name, row_index=row, w=widget: self._update_step_parameter(row_index, name, w.isChecked())
                )
            elif spec.kind == "choice":
                widget = QComboBox()
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
                widget.setRange(int(spec.minimum or -1_000_000), int(spec.maximum or 1_000_000))
                widget.setSingleStep(int(spec.step or 1))
                widget.setValue(int(value))
                widget.valueChanged.connect(
                    lambda new_value, name=spec.name, row_index=row: self._update_step_parameter(row_index, name, int(new_value))
                )
            else:
                widget = QDoubleSpinBox()
                widget.setDecimals(spec.decimals)
                widget.setRange(float(spec.minimum or -1_000_000), float(spec.maximum or 1_000_000))
                widget.setSingleStep(float(spec.step or 0.1))
                widget.setValue(float(value))
                widget.valueChanged.connect(
                    lambda new_value, name=spec.name, row_index=row: self._update_step_parameter(row_index, name, float(new_value))
                )
            self._parameter_widgets[spec.name] = widget
            self.parameters_form.addRow(get_parameter_display_label(spec, self._ui_language), widget)

    def _update_step_parameter(self, row: int, parameter_name: str, value) -> None:
        if row < 0 or row >= len(self._pipeline.steps):
            return
        self._pipeline.steps[row].parameters[parameter_name] = value
        if self.auto_apply_checkbox.isChecked() and self._workspace.current_image_path:
            self.process_current_image(debounced=True)

    def _add_pipeline_step(self) -> None:
        operation_name = str(self.operation_selector.currentData())
        self._pipeline.steps.append(PreprocessingPipeline.create_step(operation_name))
        self._populate_pipeline_list()
        self.pipeline_list.setCurrentRow(len(self._pipeline.steps) - 1)
        self._auto_apply_pipeline()

    def _remove_pipeline_step(self) -> None:
        row = self.pipeline_list.currentRow()
        if row < 0:
            return
        self._pipeline.steps.pop(row)
        self._populate_pipeline_list()
        self._auto_apply_pipeline()

    def _move_pipeline_step_up(self) -> None:
        row = self.pipeline_list.currentRow()
        if row <= 0:
            return
        self._pipeline.steps[row - 1], self._pipeline.steps[row] = self._pipeline.steps[row], self._pipeline.steps[row - 1]
        self._populate_pipeline_list()
        self.pipeline_list.setCurrentRow(row - 1)
        self._auto_apply_pipeline()

    def _move_pipeline_step_down(self) -> None:
        row = self.pipeline_list.currentRow()
        if row < 0 or row >= len(self._pipeline.steps) - 1:
            return
        self._pipeline.steps[row + 1], self._pipeline.steps[row] = self._pipeline.steps[row], self._pipeline.steps[row + 1]
        self._populate_pipeline_list()
        self.pipeline_list.setCurrentRow(row + 1)
        self._auto_apply_pipeline()

    def _on_pipeline_item_changed(self, item: QListWidgetItem) -> None:
        if self._ignore_pipeline_item_change:
            return
        row = self.pipeline_list.row(item)
        if row < 0 or row >= len(self._pipeline.steps):
            return
        self._pipeline.steps[row].enabled = item.checkState() == Qt.CheckState.Checked
        self._auto_apply_pipeline()

    def _on_extraction_settings_changed(self, *_args) -> None:
        if self._workspace.current_image_path:
            self.process_current_image(debounced=True)

    def _save_pipeline_json(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            self._tr("save_pipeline_dialog_title"),
            "",
            self._tr("json_file_filter"),
        )
        if not path:
            return
        Path(path).write_text(json.dumps(self.get_pipeline(), indent=2, ensure_ascii=False), encoding="utf-8")
        self._append_log(self._tr("pipeline_saved_log", path=path))

    def _load_pipeline_json(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            self._tr("load_pipeline_dialog_title"),
            "",
            self._tr("json_file_filter"),
        )
        if not path:
            return
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        self.set_pipeline(payload)
        self._append_log(self._tr("pipeline_loaded_log", path=path))

    def _on_image_item_changed(self, current: QListWidgetItem | None, previous: QListWidgetItem | None) -> None:
        del previous
        if current is None:
            return
        image_path = current.data(Qt.ItemDataRole.UserRole)
        if image_path:
            try:
                self.load_image(str(image_path))
            except Exception as exc:
                self._append_log(self._tr("failed_to_load_image_log", image_path=image_path, error=exc))
                QMessageBox.warning(self, self._tr("image_load_error_title"), str(exc))

    def _select_input_directory(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            self._tr("select_input_directory_dialog"),
            self.input_dir_edit.text(),
        )
        if path:
            self.set_input_directory(path)

    def _select_cif_directory(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            self._tr("select_cif_directory_dialog"),
            self.cif_dir_edit.text(),
        )
        if path:
            self.set_cif_directory(path)

    def _select_output_directory(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            self._tr("select_output_directory_dialog"),
            self.output_dir_edit.text(),
        )
        if path:
            self.set_output_directory(path)

    def _apply_input_directory_edit(self) -> None:
        path = self.input_dir_edit.text().strip()
        if path:
            self.set_input_directory(path)
        else:
            self._workspace.replace_image_selection([], is_supported_image=is_image_path)
            self.image_list.clear()
            self._sync_current_state_views()
            self._save_persisted_paths()

    def _apply_cif_directory_edit(self) -> None:
        path = self.cif_dir_edit.text().strip()
        if path:
            self.set_cif_directory(path)
        else:
            self._workspace.clear_cif_index()
            self._save_persisted_paths()
            if self._workspace.current_image_path:
                try:
                    self.load_image(self._workspace.current_image_path)
                except Exception as exc:
                    self._append_log(self._tr("reload_with_cif_failed_log", error=exc))

    def _apply_output_directory_edit(self) -> None:
        self.set_output_directory(self.output_dir_edit.text().strip())

    def _choose_external_color(self) -> None:
        self._choose_color("external_color", self.external_color_button)

    def _choose_hole_color(self) -> None:
        self._choose_color("hole_color", self.hole_color_button)

    def _choose_selected_color(self) -> None:
        self._choose_color("selected_color", self.selected_color_button)

    def _choose_vertex_color(self) -> None:
        self._choose_color("vertex_color", self.vertex_color_button)

    def _choose_color(self, attribute_name: str, button: QPushButton) -> None:
        initial = QColor(getattr(self._display_settings, attribute_name))
        color = QColorDialog.getColor(initial, self, self._tr("select_color_dialog_title"))
        if not color.isValid():
            return
        value = color.name(QColor.NameFormat.HexRgb)
        setattr(self._display_settings, attribute_name, value)
        self._update_color_button(button, value)
        self._apply_display_settings()

    def _apply_display_settings(self) -> None:
        if hasattr(self, "line_width_spin"):
            self._display_settings.line_width = float(self.line_width_spin.value())
            self._display_settings.vertex_size = float(self.vertex_size_spin.value())
            self._display_settings.fill_opacity = float(self.fill_opacity_spin.value())
            self._display_settings.show_vertices = bool(self.show_vertices_checkbox.isChecked())
            self._display_settings.show_labels = bool(self.show_labels_checkbox.isChecked())
        if hasattr(self, "polygon_editor"):
            self.polygon_editor.set_display_settings(self._display_settings)

    def _auto_apply_pipeline(self) -> None:
        if self.auto_apply_checkbox.isChecked() and self._workspace.current_image_path:
            self.process_current_image(debounced=True)

    def _current_contour_settings(self) -> ContourExtractionSettings:
        max_area = self.max_area_spin.value()
        return ContourExtractionSettings(
            retrieval_mode=str(self.retrieval_mode_combo.currentData() or self.retrieval_mode_combo.currentText()),
            approximation_mode=str(self.approximation_mode_combo.currentData() or self.approximation_mode_combo.currentText()),
            epsilon=self.epsilon_spin.value(),
            epsilon_relative=self.epsilon_relative_checkbox.isChecked(),
            min_area=self.min_area_spin.value(),
            max_area=None if max_area <= 0 else max_area,
            min_perimeter=self.min_perimeter_spin.value(),
            min_points=self.min_points_spin.value(),
        )

    def _current_save_options(self) -> SaveOptions:
        return SaveOptions(
            save_cif=self.save_cif_checkbox.isChecked(),
            save_csv=self.save_csv_checkbox.isChecked(),
            save_txt=self.save_txt_checkbox.isChecked(),
            save_svg=self.save_svg_checkbox.isChecked(),
            save_preview=self.save_preview_checkbox.isChecked(),
        )

    def _sync_current_state_views(self) -> None:
        self._updating_views = True
        try:
            display_image = self._display_image_for_current_state()
            current_state = self._workspace.current_state
            polygons = current_state.polygons if current_state else []
            self.polygon_editor.set_image(display_image)
            self.polygon_editor.set_polygons(polygons)
        finally:
            self._updating_views = False

    def _display_image_for_current_state(self):
        return self._workspace.current_display_image()

    def _queue_prepared_image_update(self, image_path: str, source_image) -> None:
        request = PreparedImageRequest(
            image_path=image_path,
            source_image=source_image,
            pipeline_config=self.get_pipeline(),
        )
        signature = self._prepared_image_request_signature(request)
        if signature == self._prepared_image_running_signature or signature == self._prepared_image_pending_signature:
            self._refresh_busy_indicator()
            return
        self._prepared_image_pending_request = request
        self._prepared_image_pending_signature = signature
        self._refresh_busy_indicator()
        self._start_pending_prepared_image_update()

    def _start_pending_prepared_image_update(self) -> None:
        if self._prepared_image_running_request_id is not None or self._prepared_image_pending_request is None:
            return
        request = self._prepared_image_pending_request
        self._prepared_image_pending_request = None
        request_signature = self._prepared_image_pending_signature
        self._prepared_image_pending_signature = None
        self._prepared_image_request_serial += 1
        request_id = self._prepared_image_request_serial
        self._prepared_image_running_request_id = request_id
        self._prepared_image_running_signature = request_signature

        worker = PreparedImageRunnable(request_id=request_id, request=request)
        worker.signals.result.connect(self._on_prepared_image_result)
        worker.signals.error.connect(self._on_prepared_image_error)
        worker.signals.finished.connect(self._on_prepared_image_finished)
        self._prepared_image_thread_pool.start(worker)
        self._refresh_busy_indicator()

    def _build_preview_request(self) -> PreviewProcessingRequest | None:
        if not self._workspace.current_image_path:
            return None
        return PreviewProcessingRequest(
            image_path=self._workspace.current_image_path,
            pipeline_config=self.get_pipeline(),
            contour_settings=self._current_contour_settings(),
        )

    def _preview_request_signature(self, request: PreviewProcessingRequest) -> tuple[str, str, str]:
        return build_preview_request_signature(request)

    def _prepared_image_request_signature(self, request: PreparedImageRequest) -> tuple[str, str]:
        return build_prepared_image_signature(request)

    def _queue_preview_processing(self, *, debounced: bool) -> None:
        request = self._build_preview_request()
        if request is None:
            self._append_log(self._tr("no_image_selected_log"))
            return
        signature = self._preview_request_signature(request)
        if signature == self._preview_running_signature or signature == self._preview_pending_signature:
            self._refresh_busy_indicator()
            return
        self._preview_pending_request = request
        self._preview_pending_signature = signature
        self._refresh_busy_indicator()
        if debounced:
            self._preview_update_timer.start()
            return
        self._preview_update_timer.stop()
        self._start_pending_preview_processing()

    def _start_pending_preview_processing(self) -> None:
        if self._preview_running_request_id is not None or self._preview_pending_request is None:
            return
        request = self._preview_pending_request
        self._preview_pending_request = None
        request_signature = self._preview_pending_signature
        self._preview_pending_signature = None
        self._preview_request_serial += 1
        request_id = self._preview_request_serial
        self._preview_running_request_id = request_id
        self._preview_running_signature = request_signature

        worker = PreviewProcessingRunnable(request_id=request_id, request=request)
        worker.signals.result.connect(self._on_preview_processing_result)
        worker.signals.error.connect(self._on_preview_processing_error)
        worker.signals.finished.connect(self._on_preview_processing_finished)
        self._preview_thread_pool.start(worker)
        self._refresh_busy_indicator()

    def _append_log(self, message: str) -> None:
        self.logMessage.emit(message)

    def _refresh_busy_indicator(self) -> None:
        active = any(
            (
                self._preview_running_request_id is not None,
                self._preview_pending_request is not None,
                self._preview_update_timer.isActive(),
                self._prepared_image_running_request_id is not None,
                self._prepared_image_pending_request is not None,
            )
        )
        if hasattr(self, "preview_busy_label"):
            self.preview_busy_label.setText(self._busy_indicator_text())
            self.preview_busy_label.setVisible(active)
        if hasattr(self, "preview_busy_progress"):
            self.preview_busy_progress.setVisible(active)

    def _on_prepared_image_result(self, request_id: int, image_path: str, preprocessed_image, pipeline_config: dict) -> None:
        if request_id != self._prepared_image_running_request_id:
            return
        if pipeline_config != self.get_pipeline():
            return
        if self._workspace.store_preprocessed_image(image_path, preprocessed_image):
            self._sync_current_state_views()

    def _on_prepared_image_error(self, request_id: int, message: str) -> None:
        if request_id != self._prepared_image_running_request_id:
            return
        self._append_log(self._tr("processing_failed_log", error=message))

    def _on_prepared_image_finished(self, request_id: int) -> None:
        if request_id == self._prepared_image_running_request_id:
            self._prepared_image_running_request_id = None
            self._prepared_image_running_signature = None
        if self._prepared_image_pending_request is not None:
            self._start_pending_prepared_image_update()
        self._refresh_busy_indicator()

    def _on_preview_processing_result(self, request_id: int, result) -> None:
        if request_id != self._preview_running_request_id:
            return
        if self._workspace.current_image_path != result.image_path:
            return

        if self._workspace.apply_processing_result(result):
            self._sync_current_state_views()
        self._set_progress_status("current_image_processed_status")
        self._append_log(
            self._tr(
                "current_image_processed_log",
                image_name=Path(result.image_path).name,
                count=len(result.polygons),
            )
        )
        self.imageProcessed.emit(result.image_path, result.polygons)

    def _on_preview_processing_error(self, request_id: int, message: str) -> None:
        if request_id != self._preview_running_request_id:
            return
        self._append_log(self._tr("processing_failed_log", error=message))

    def _on_preview_processing_finished(self, request_id: int) -> None:
        if request_id == self._preview_running_request_id:
            self._preview_running_request_id = None
            self._preview_running_signature = None
        if self._preview_pending_request is not None and not self._preview_update_timer.isActive():
            self._start_pending_preview_processing()
        self._refresh_busy_indicator()

    def _show_batch_progress(self, total: int) -> None:
        if not self._batch_progress_enabled:
            self._hide_batch_progress()
            return
        self.batch_progress_bar.setRange(0, max(1, total))
        self.batch_progress_bar.setValue(0)
        self.batch_progress_bar.setVisible(True)

    def _hide_batch_progress(self) -> None:
        self.batch_progress_bar.setVisible(False)
        self.batch_progress_bar.setRange(0, 100)
        self.batch_progress_bar.setValue(0)

    def _on_polygons_edited(self) -> None:
        if self._updating_views:
            return
        if self._workspace.update_current_polygons(self.get_polygons()):
            self.polygonsEdited.emit()

    def _on_batch_result(self, result) -> None:
        self.imageProcessed.emit(result.image_path, result.polygons)
        self._append_log(
            self._tr(
                "batch_result_log",
                image_name=Path(result.image_path).name,
                count=len(result.polygons),
            )
        )

    def _on_batch_progress(self, current: int, total: int) -> None:
        if self._batch_progress_enabled:
            self.batch_progress_bar.setRange(0, max(1, total))
            self.batch_progress_bar.setValue(current)
        self._set_progress_status("batch_progress_status", current=current, total=total)
        self.batchProgress.emit(current, total)

    def _on_batch_finished(self) -> None:
        self._batch_progress_enabled = False
        self._hide_batch_progress()
        self._set_progress_status("batch_finished_status")
        self.batchFinished.emit()

    def _on_batch_error(self, image_path: str, message: str) -> None:
        self._append_log(self._tr("batch_error_log", image_name=Path(image_path).name, message=message))

    def refresh_image_list(self) -> None:
        directory = self.input_dir_edit.text().strip()
        if not directory:
            self._append_log(self._tr("input_directory_empty_log"))
            return
        self.load_images(scan_image_files(directory))

    def set_input_directory(self, path: str) -> None:
        directory_state = load_input_directory(path, scan_images=scan_image_files)
        self.input_dir_edit.setText(directory_state.directory)
        self._save_persisted_paths()
        self.load_images(list(directory_state.image_paths))

    def set_cif_directory(self, path: str) -> None:
        directory_state = index_cif_directory(path)
        self.cif_dir_edit.setText(directory_state.directory)
        self._save_persisted_paths()
        self._workspace.set_cif_index(directory_state.indexed_paths)
        if directory_state.available:
            self._append_log(self._tr("cif_indexed_log", count=len(directory_state.indexed_paths)))
        else:
            self._append_log(self._tr("cif_directory_unavailable_log"))

        if self._workspace.current_image_path:
            try:
                self.load_image(self._workspace.current_image_path)
            except Exception as exc:
                self._append_log(self._tr("reload_with_cif_failed_log", error=exc))

    def set_output_directory(self, path: str) -> None:
        self.output_dir_edit.setText(path)
        self._save_persisted_paths()

    def load_images(self, paths: list[str]) -> None:
        normalized_paths = self._workspace.replace_image_selection(paths, is_supported_image=is_image_path)
        self._preview_update_timer.stop()
        self._preview_pending_request = None
        self._preview_pending_signature = None
        self._prepared_image_pending_request = None
        self._prepared_image_pending_signature = None
        self._refresh_busy_indicator()
        self.image_list.clear()
        for path in normalized_paths:
            item = QListWidgetItem(Path(path).name)
            item.setToolTip(path)
            item.setData(Qt.ItemDataRole.UserRole, path)
            self.image_list.addItem(item)
        if normalized_paths:
            self.image_list.setCurrentRow(0)
        else:
            self._sync_current_state_views()

    def _find_matching_cif_path(self, image_path: str) -> str | None:
        return self._workspace.resolve_cif_path(image_path)

    def _load_cif_overlay_polygons(self, image_path: str) -> list[PolygonData]:
        cif_path = self._find_matching_cif_path(image_path)
        if not cif_path:
            return []
        try:
            referenced_image, image_size, polygons = load_polygons_cif(cif_path)
        except Exception as exc:
            self._append_log(self._tr("cif_load_failed_log", file_name=Path(cif_path).name, error=exc))
            return []
        if referenced_image and Path(referenced_image).stem.lower() != Path(image_path).stem.lower():
            self._append_log(
                self._tr(
                    "cif_reference_name_diff_log",
                    file_name=Path(cif_path).name,
                    referenced_image=referenced_image,
                )
            )
        if image_size is not None:
            self._append_log(
                self._tr(
                    "cif_overlay_loaded_with_size_log",
                    file_name=Path(cif_path).name,
                    width=image_size[0],
                    height=image_size[1],
                    count=len(polygons),
                )
            )
        else:
            self._append_log(self._tr("cif_overlay_loaded_log", file_name=Path(cif_path).name, count=len(polygons)))
        return polygons

    def load_image(self, path: str) -> None:
        self._preview_update_timer.stop()
        self._preview_pending_request = None
        self._preview_pending_signature = None
        self._prepared_image_pending_request = None
        self._prepared_image_pending_signature = None
        self._refresh_busy_indicator()
        image_result = self._workspace.load_image(
            path,
            load_source_image=load_image_grayscale,
            load_cif_overlay=self._load_cif_overlay_polygons,
        )
        if image_result.reused_current_state:
            return
        self._sync_current_state_views()
        if image_result.prepared_image_required and image_result.state is not None and image_result.state.source_image is not None:
            self._queue_prepared_image_update(image_result.image_path, image_result.state.source_image)
        if image_result.cache_hit:
            self._append_log(self._tr("loaded_cached_state_log", image_path=image_result.image_path))
        else:
            self._append_log(self._tr("loaded_image_log", image_path=image_result.image_path))

    def get_polygons(self) -> list[PolygonData]:
        return self.polygon_editor.get_polygons()

    def set_pipeline(self, config: dict) -> None:
        self._pipeline = PreprocessingPipeline.from_dict(config)
        self._populate_pipeline_list()
        self._auto_apply_pipeline()

    def get_pipeline(self) -> dict:
        return self._pipeline.to_dict()

    def process_current_image(self, *_args, debounced: bool = False) -> None:
        self._queue_preview_processing(debounced=debounced)

    def save_current_result(
        self,
        output_directory: str | None = None,
        save_options: SaveOptions | None = None,
    ) -> dict[str, str]:
        current_state = self._workspace.current_state
        current_image_path = self._workspace.current_image_path
        if current_state is None or current_image_path is None:
            self._append_log(self._tr("nothing_to_save_log"))
            return {}
        target_directory = output_directory or self.output_dir_edit.text().strip()
        if not target_directory:
            self._append_log(self._tr("output_directory_not_set_log"))
            return {}
        saved_files = save_result_bundle(
            output_directory=target_directory,
            image_path=current_image_path,
            polygons=self.get_polygons(),
            source_image=current_state.source_image,
            display_settings=self._display_settings,
            save_options=save_options or self._current_save_options(),
            metadata={
                "contour_settings": self._current_contour_settings().to_dict(),
                "pipeline": self.get_pipeline(),
            },
        )
        if saved_files:
            self._append_log(self._tr("saved_result_log", saved_files=saved_files))
        return saved_files

    def start_batch_processing(
        self,
        image_paths: list[str] | None = None,
        max_workers: int | None = None,
    ) -> None:
        if self._batch_processor.is_running:
            self._append_log(self._tr("batch_already_running_log"))
            return
        paths = image_paths or list(self._workspace.image_paths)
        if not paths:
            self._append_log(self._tr("batch_no_images_log"))
            return
        output_directory = self.output_dir_edit.text().strip() or None
        save_options = self._current_save_options()
        self._batch_progress_enabled = bool(output_directory and save_options.save_cif)
        self._show_batch_progress(len(paths))
        self._set_progress_status("batch_started_status")
        self._batch_processor.start(
            image_paths=paths,
            pipeline_config=self.get_pipeline(),
            contour_settings=self._current_contour_settings(),
            output_directory=output_directory,
            save_options=save_options,
            display_settings=self._display_settings,
            max_workers=max_workers or self.max_workers_spin.value(),
        )

    def stop_batch_processing(self) -> None:
        self._batch_processor.stop()
