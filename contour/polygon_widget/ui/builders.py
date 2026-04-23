"""UI builder functions for :class:`PolygonExtractionWidget`.

These functions were extracted from the widget god-class during the
production-ready refactor. They are attached back onto the widget class as
bound methods via attribute assignment (see ``polygon_widget.widget``), which
preserves all ``self.*`` attribute access without behavioral changes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QStyle,
    QTabWidget,
    QToolButton,
    QTreeWidget,
    QVBoxLayout,
    QWidget,
)

from ..application.processing import (
    VIA_SEARCH_MODE_BLOB,
    VIA_SEARCH_MODE_HYBRID,
    VIA_SEARCH_MODE_TEMPLATE,
    VIA_SIZE_MODE_FIXED,
    VIA_SIZE_MODE_RANGE,
)
from ..contour_extractor import APPROXIMATION_MODE_MAP, RETRIEVAL_MODE_MAP
from ..graphics_view import BrushMode, DeleteVertexMode, EditorTool, PolygonCreateMode, PolygonEditorView
from .pipeline_list import PipelineListWidget

if TYPE_CHECKING:
    pass


def build_ui(self) -> None:
    root_layout = QVBoxLayout(self)

    self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
    self.main_splitter.setChildrenCollapsible(False)
    self.main_splitter.splitterMoved.connect(self._on_main_splitter_moved)
    root_layout.addWidget(self.main_splitter, 1)

    left_scroll = QScrollArea()
    left_scroll.setWidgetResizable(True)
    left_scroll.setMinimumWidth(360)
    left_scroll.setMaximumWidth(560)
    controls_container = QWidget()
    left_scroll.setWidget(controls_container)
    controls_layout = QVBoxLayout(controls_container)
    self.control_tabs = self._build_tabs()
    self.control_tabs.currentChanged.connect(self._on_control_tab_changed)
    controls_layout.addWidget(self.control_tabs, 1)
    self.main_splitter.addWidget(left_scroll)
    self.visual_panel = self._build_visual_panel()
    self.main_splitter.addWidget(self.visual_panel)
    self.right_tabs = QTabWidget()
    self.right_tabs.setUsesScrollButtons(True)
    self.right_tabs.setMinimumWidth(280)
    self.right_tabs.setMaximumWidth(440)
    self.files_tab = self._build_files_tab()
    self.right_tabs.addTab(self.files_tab, "Files")
    self.main_splitter.addWidget(self.right_tabs)
    self.main_splitter.setStretchFactor(0, 0)
    self.main_splitter.setStretchFactor(1, 1)
    self.main_splitter.setStretchFactor(2, 0)
    self.main_splitter.setSizes([380, 1000, 320])


def build_path_panel(self) -> QWidget:
    self.path_group = QGroupBox("Input / Output")
    layout = QVBoxLayout(self.path_group)

    self.input_dir_edit = QLineEdit()
    self.cif_dir_edit = QLineEdit()
    self.output_dir_edit = QLineEdit()
    self.dataset_dir_edit = QLineEdit()
    self.input_dir_label = QLabel("Input directory")
    self.cif_dir_label = QLabel("CIF overlay directory")
    self.output_dir_label = QLabel("Output directory")
    self.dataset_dir_label = QLabel("Dataset directory")
    self.browse_input_button = QPushButton()
    self.browse_cif_button = QPushButton()
    self.browse_output_button = QPushButton()
    self.browse_dataset_button = QPushButton()
    self.refresh_button = QPushButton()
    folder_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon)
    for button in (
        self.browse_input_button,
        self.browse_cif_button,
        self.browse_output_button,
        self.browse_dataset_button,
    ):
        self._configure_icon_only_button(button, folder_icon)
    self._configure_icon_only_button(self.refresh_button, self._refresh_files_icon())

    self.browse_input_button.clicked.connect(self._select_input_directory)
    self.browse_cif_button.clicked.connect(self._select_cif_directory)
    self.browse_output_button.clicked.connect(self._select_output_directory)
    self.browse_dataset_button.clicked.connect(self._select_dataset_directory)
    self.refresh_button.clicked.connect(self.refresh_image_list)
    self.input_dir_edit.editingFinished.connect(self._apply_input_directory_edit)
    self.cif_dir_edit.editingFinished.connect(self._apply_cif_directory_edit)
    self.output_dir_edit.editingFinished.connect(self._apply_output_directory_edit)
    self.dataset_dir_edit.editingFinished.connect(self._apply_dataset_directory_edit)

    for label, edit, button in [
        (self.input_dir_label, self.input_dir_edit, self.browse_input_button),
        (self.cif_dir_label, self.cif_dir_edit, self.browse_cif_button),
        (self.output_dir_label, self.output_dir_edit, self.browse_output_button),
        (self.dataset_dir_label, self.dataset_dir_edit, self.browse_dataset_button),
    ]:
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(6)
        row_layout.addWidget(edit, 1)
        row_layout.addWidget(button)
        layout.addWidget(label)
        layout.addWidget(row)
    layout.addWidget(self.refresh_button)
    return self.path_group


def build_paths_tab(self) -> QWidget:
    tab = QWidget()
    layout = QVBoxLayout(tab)
    self.path_panel = self._build_path_panel()
    layout.addWidget(self.path_panel)

    self.extra_layers_group = QGroupBox("Additional layers")
    self.extra_layers_form = QFormLayout(self.extra_layers_group)
    self._configure_compact_form(self.extra_layers_form)
    self.extra_layers_widget = QWidget()
    extra_layers_layout = QVBoxLayout(self.extra_layers_widget)
    extra_layers_layout.setContentsMargins(0, 0, 0, 0)
    extra_layers_layout.setSpacing(6)
    self.extra_layers_list = QListWidget()
    self.extra_layers_list.setMaximumHeight(100)
    extra_layers_layout.addWidget(self.extra_layers_list)
    self.extra_layer_path_widget = QWidget()
    extra_layer_path_layout = QHBoxLayout(self.extra_layer_path_widget)
    extra_layer_path_layout.setContentsMargins(0, 0, 0, 0)
    extra_layer_path_layout.setSpacing(6)
    self.extra_layer_path_edit = QLineEdit()
    self.extra_layer_path_browse_button = QPushButton("...")
    self.extra_layer_path_browse_button.setFixedWidth(34)
    extra_layer_path_layout.addWidget(self.extra_layer_path_edit, 1)
    extra_layer_path_layout.addWidget(self.extra_layer_path_browse_button)
    extra_layer_buttons = QWidget()
    extra_layer_buttons_layout = QHBoxLayout(extra_layer_buttons)
    extra_layer_buttons_layout.setContentsMargins(0, 0, 0, 0)
    self.add_extra_layers_button = QPushButton("Add images")
    self.remove_extra_layer_button = QPushButton("Remove")
    extra_layer_buttons_layout.addWidget(self.add_extra_layers_button)
    extra_layer_buttons_layout.addWidget(self.remove_extra_layer_button)
    extra_layers_layout.addWidget(extra_layer_buttons)
    self.extra_layer_visible_checkbox = QCheckBox("Layer visible")
    self.extra_layer_opacity_spin = QDoubleSpinBox()
    self.extra_layer_opacity_spin.setRange(0.0, 1.0)
    self.extra_layer_opacity_spin.setSingleStep(0.05)
    self.extra_layer_opacity_spin.setValue(0.35)
    self.extra_layer_dx_spin = QDoubleSpinBox()
    self.extra_layer_dx_spin.setRange(-1_000_000.0, 1_000_000.0)
    self.extra_layer_dx_spin.setDecimals(2)
    self.extra_layer_dy_spin = QDoubleSpinBox()
    self.extra_layer_dy_spin.setRange(-1_000_000.0, 1_000_000.0)
    self.extra_layer_dy_spin.setDecimals(2)

    self.extra_layers_list.currentRowChanged.connect(self._on_extra_layer_selected)
    self.add_extra_layers_button.clicked.connect(self._load_extra_layers)
    self.remove_extra_layer_button.clicked.connect(self._remove_selected_extra_layer)
    self.extra_layer_path_browse_button.clicked.connect(self._browse_selected_extra_layer_path)
    self.extra_layer_path_edit.editingFinished.connect(self._on_extra_layer_path_changed)
    self.extra_layer_visible_checkbox.stateChanged.connect(self._on_extra_layer_controls_changed)
    self.extra_layer_opacity_spin.valueChanged.connect(self._on_extra_layer_controls_changed)
    self.extra_layer_dx_spin.valueChanged.connect(self._on_extra_layer_controls_changed)
    self.extra_layer_dy_spin.valueChanged.connect(self._on_extra_layer_controls_changed)

    self.extra_layers_form.addRow("Additional layers", self.extra_layers_widget)
    self.extra_layers_label_widget = self.extra_layers_form.labelForField(self.extra_layers_widget)
    self.extra_layers_form.addRow("Layer path", self.extra_layer_path_widget)
    self.extra_layer_path_label_widget = self.extra_layers_form.labelForField(self.extra_layer_path_widget)
    self.extra_layers_form.addRow(self.extra_layer_visible_checkbox)
    self.extra_layers_form.addRow("Layer opacity", self.extra_layer_opacity_spin)
    self.extra_layer_opacity_label_widget = self.extra_layers_form.labelForField(self.extra_layer_opacity_spin)
    self.extra_layers_form.addRow("Layer dX", self.extra_layer_dx_spin)
    self.extra_layer_dx_label_widget = self.extra_layers_form.labelForField(self.extra_layer_dx_spin)
    self.extra_layers_form.addRow("Layer dY", self.extra_layer_dy_spin)
    self.extra_layer_dy_label_widget = self.extra_layers_form.labelForField(self.extra_layer_dy_spin)
    layout.addWidget(self.extra_layers_group)

    layout.addStretch(1)
    return tab


def build_tabs(self) -> QWidget:
    tabs = QTabWidget()
    tabs.setUsesScrollButtons(True)
    self.paths_tab = self._build_paths_tab()
    self.pipeline_tab = self._build_pipeline_tab()
    self.extraction_tab = self._build_extraction_tab()
    self.display_tab = self._build_display_tab()
    tabs.addTab(self.paths_tab, "Paths")
    tabs.addTab(self.pipeline_tab, "Pipeline")
    tabs.addTab(self.extraction_tab, "Extraction")
    tabs.addTab(self.display_tab, "Display")
    return tabs


def build_files_tab(self) -> QWidget:
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
    self.process_current_button = QPushButton()
    self.process_current_button.clicked.connect(self.process_current_image)
    self.batch_button = QPushButton()
    self.batch_button.clicked.connect(self.start_batch_processing)
    self.stop_batch_button = QPushButton()
    self.stop_batch_button.clicked.connect(self.stop_batch_processing)
    self._configure_icon_only_button(
        self.process_current_button,
        self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay),
    )
    self._configure_icon_only_button(
        self.batch_button,
        self.style().standardIcon(QStyle.StandardPixmap.SP_MediaSeekForward),
    )
    self._configure_icon_only_button(
        self.stop_batch_button,
        self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop),
    )
    self.save_current_button = QPushButton("Save current result")
    self.save_current_button.clicked.connect(self.save_current_result)
    self.export_dataset_button = QPushButton("Export frame to dataset")
    self.export_dataset_button.clicked.connect(self.export_current_frame_to_dataset)
    self.dataset_mode_checkbox = QCheckBox("Dataset mode")
    self.max_workers_spin = QSpinBox()
    self.max_workers_spin.setRange(1, 32)
    self.max_workers_spin.setValue(4)
    self.max_workers_label = QLabel("Max workers")
    run_buttons_row = QWidget()
    run_buttons_layout = QHBoxLayout(run_buttons_row)
    run_buttons_layout.setContentsMargins(0, 0, 0, 0)
    run_buttons_layout.setSpacing(6)
    run_buttons_layout.addWidget(self.process_current_button)
    run_buttons_layout.addWidget(self.batch_button)
    run_buttons_layout.addWidget(self.stop_batch_button)
    run_buttons_layout.addStretch(1)
    run_layout.addWidget(run_buttons_row, 0, 0, 1, 2)
    run_layout.addWidget(self.max_workers_label, 1, 0)
    run_layout.addWidget(self.max_workers_spin, 1, 1)
    run_layout.addWidget(self.save_current_button, 2, 0, 1, 2)
    run_layout.addWidget(self.export_dataset_button, 3, 0, 1, 2)
    run_layout.addWidget(self.dataset_mode_checkbox, 4, 0, 1, 2)
    layout.addWidget(self.run_group)
    self.batch_progress_bar = QProgressBar()
    self.batch_progress_bar.setRange(0, 100)
    self.batch_progress_bar.setValue(0)
    self.batch_progress_bar.setFormat("%p% (%v/%m)")
    self.batch_progress_bar.setTextVisible(True)
    self.batch_progress_bar.setVisible(False)
    layout.addWidget(self.batch_progress_bar)
    return tab


def build_pipeline_tab(self) -> QWidget:
    tab = QWidget()
    layout = QVBoxLayout(tab)

    self.available_filters_group = QGroupBox("Available filters")
    available_layout = QVBoxLayout(self.available_filters_group)
    self.operation_tree = QTreeWidget()
    self.operation_tree.setHeaderHidden(True)
    self.operation_tree.setRootIsDecorated(True)
    self.operation_tree.setUniformRowHeights(True)
    self.operation_tree.currentItemChanged.connect(self._on_available_operation_selected)
    self.operation_tree.itemDoubleClicked.connect(self._on_available_operation_activated)
    self.operation_tree.setMinimumHeight(180)
    available_layout.addWidget(self.operation_tree, 1)

    self.parameters_group = QGroupBox("Step parameters")
    parameters_scroll = QScrollArea()
    parameters_scroll.setWidgetResizable(True)
    parameters_scroll.setMinimumHeight(170)
    parameters_widget = QWidget()
    self.parameters_form = QFormLayout(parameters_widget)
    self._configure_compact_form(self.parameters_form)
    parameters_scroll.setWidget(parameters_widget)
    group_layout = QVBoxLayout(self.parameters_group)
    group_layout.addWidget(parameters_scroll)

    self.pipeline_steps_group = QGroupBox("Applied filters")
    steps_layout = QVBoxLayout(self.pipeline_steps_group)

    self.pipeline_list = PipelineListWidget()
    self.pipeline_list.currentRowChanged.connect(self._on_pipeline_step_selected)
    self.pipeline_list.itemChanged.connect(self._on_pipeline_item_changed)
    self.pipeline_list.deletePressed.connect(self._remove_pipeline_step)
    self.pipeline_list.orderChanged.connect(self._sync_pipeline_order_from_list)
    self.pipeline_list.setMinimumHeight(180)
    steps_layout.addWidget(self.pipeline_list, 1)

    apply_row = QWidget()
    apply_layout = QGridLayout(apply_row)
    apply_layout.setContentsMargins(0, 0, 0, 0)
    self.auto_apply_checkbox = QCheckBox("Auto apply")
    self.auto_apply_checkbox.setChecked(True)
    self.auto_apply_checkbox.hide()
    self.save_pipeline_button = QPushButton("Save JSON")
    self.save_pipeline_button.clicked.connect(self._save_pipeline_json)
    self.load_pipeline_button = QPushButton("Load JSON")
    self.load_pipeline_button.clicked.connect(self._load_pipeline_json)
    self.pipeline_preset_combo = QComboBox()
    self.apply_pipeline_preset_button = QPushButton("Apply filter preset")
    self.apply_pipeline_preset_button.clicked.connect(self._apply_selected_pipeline_preset)
    self.auto_tune_button = QPushButton("Auto-fit from drawing")
    self.auto_tune_button.clicked.connect(self._start_auto_tune_from_reference)
    self.auto_tune_button.setToolTip("Tunes filter parameters using the drawn polygons as the target result")
    apply_layout.addWidget(self.save_pipeline_button, 0, 0)
    apply_layout.addWidget(self.load_pipeline_button, 0, 1)
    apply_layout.addWidget(self.pipeline_preset_combo, 1, 0)
    apply_layout.addWidget(self.apply_pipeline_preset_button, 1, 1)
    apply_layout.addWidget(self.auto_tune_button, 2, 0, 1, 2)
    apply_layout.setColumnStretch(0, 1)
    apply_layout.setColumnStretch(1, 1)
    steps_layout.addWidget(apply_row)

    self.pipeline_help_group = QGroupBox("Filter help")
    help_layout = QVBoxLayout(self.pipeline_help_group)
    self.pipeline_help_title = QLabel()
    self.pipeline_help_title.setWordWrap(True)
    self.pipeline_help_title.setFixedHeight(28)
    self.pipeline_help_summary = QLabel()
    self.pipeline_help_summary.setWordWrap(True)
    self.pipeline_help_summary.setFixedHeight(58)
    self.pipeline_help_use = QLabel()
    self.pipeline_help_use.setWordWrap(True)
    self.pipeline_help_use.setFixedHeight(74)
    preview_row = QWidget()
    preview_layout = QHBoxLayout(preview_row)
    preview_layout.setContentsMargins(0, 0, 0, 0)
    before_column = QVBoxLayout()
    self.pipeline_help_before_title = QLabel("Before")
    self.pipeline_help_before_image = QLabel()
    self.pipeline_help_before_image.setFixedSize(190, 132)
    self.pipeline_help_before_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
    before_column.addWidget(self.pipeline_help_before_title)
    before_column.addWidget(self.pipeline_help_before_image)
    after_column = QVBoxLayout()
    self.pipeline_help_after_title = QLabel("After")
    self.pipeline_help_after_image = QLabel()
    self.pipeline_help_after_image.setFixedSize(190, 132)
    self.pipeline_help_after_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
    after_column.addWidget(self.pipeline_help_after_title)
    after_column.addWidget(self.pipeline_help_after_image)
    preview_layout.addLayout(before_column)
    preview_layout.addLayout(after_column)
    help_layout.addWidget(self.pipeline_help_title)
    help_layout.addWidget(self.pipeline_help_summary)
    help_layout.addWidget(self.pipeline_help_use)
    help_layout.addWidget(preview_row)

    layout.addWidget(self.available_filters_group)
    layout.addWidget(self.parameters_group)
    layout.addWidget(self.pipeline_steps_group)
    layout.addWidget(self.pipeline_help_group)
    layout.addStretch(1)
    return tab


def build_extraction_tab(self) -> QWidget:
    tab = QWidget()
    layout = QVBoxLayout(tab)
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    container = QWidget()
    container_layout = QVBoxLayout(container)
    container_layout.setContentsMargins(0, 0, 0, 0)

    self.contour_group = QGroupBox("Contour extraction")
    contour_layout = QVBoxLayout(self.contour_group)
    self.profile_group = QGroupBox("Profile")
    self.profile_form = QFormLayout(self.profile_group)
    self._configure_compact_form(self.profile_form)
    self.extraction_profile_combo = QComboBox()
    self.extraction_profile_combo.addItem("Conductors", "conductors")
    self.extraction_profile_combo.addItem("Vias", "vias")
    self.profile_form.addRow("Extraction profile", self.extraction_profile_combo)
    self.extraction_profile_label_widget = self.profile_form.labelForField(self.extraction_profile_combo)

    self.basic_filters_group = QGroupBox("Basic filters")
    self.basic_filters_form = QFormLayout(self.basic_filters_group)
    self._configure_compact_form(self.basic_filters_form)
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
    self.max_perimeter_spin = QDoubleSpinBox()
    self.max_perimeter_spin.setRange(0.0, 1_000_000_000.0)
    self.max_perimeter_spin.setValue(0.0)
    self.area_range_widget = self._build_range_row(self.min_area_spin, self.max_area_spin)
    self.perimeter_range_widget = self._build_range_row(self.min_perimeter_spin, self.max_perimeter_spin)
    self.min_points_spin = QSpinBox()
    self.min_points_spin.setRange(3, 10_000)
    self.min_points_spin.setValue(3)

    self.geometry_filters_group = QGroupBox("Geometry filters")
    self.geometry_filters_form = QFormLayout(self.geometry_filters_group)
    self._configure_compact_form(self.geometry_filters_form)
    self.min_bbox_width_spin = QSpinBox()
    self.min_bbox_width_spin.setRange(0, 100_000)
    self.min_bbox_width_spin.setValue(0)
    self.max_bbox_width_spin = QSpinBox()
    self.max_bbox_width_spin.setRange(0, 100_000)
    self.max_bbox_width_spin.setValue(0)
    self.min_bbox_height_spin = QSpinBox()
    self.min_bbox_height_spin.setRange(0, 100_000)
    self.min_bbox_height_spin.setValue(0)
    self.max_bbox_height_spin = QSpinBox()
    self.max_bbox_height_spin.setRange(0, 100_000)
    self.max_bbox_height_spin.setValue(0)
    self.min_aspect_ratio_spin = QDoubleSpinBox()
    self.min_aspect_ratio_spin.setRange(0.0, 1_000.0)
    self.min_aspect_ratio_spin.setDecimals(3)
    self.min_aspect_ratio_spin.setSingleStep(0.05)
    self.min_aspect_ratio_spin.setValue(0.0)
    self.max_aspect_ratio_spin = QDoubleSpinBox()
    self.max_aspect_ratio_spin.setRange(0.0, 1_000.0)
    self.max_aspect_ratio_spin.setDecimals(3)
    self.max_aspect_ratio_spin.setSingleStep(0.05)
    self.max_aspect_ratio_spin.setValue(0.0)
    self.bbox_width_range_widget = self._build_range_row(self.min_bbox_width_spin, self.max_bbox_width_spin)
    self.bbox_height_range_widget = self._build_range_row(self.min_bbox_height_spin, self.max_bbox_height_spin)
    self.aspect_ratio_range_widget = self._build_range_row(self.min_aspect_ratio_spin, self.max_aspect_ratio_spin)
    self.exclude_border_touching_checkbox = QCheckBox("Exclude")
    self.min_solidity_spin = QDoubleSpinBox()
    self.min_solidity_spin.setRange(0.0, 1.0)
    self.min_solidity_spin.setDecimals(3)
    self.min_solidity_spin.setSingleStep(0.05)
    self.min_solidity_spin.setValue(0.0)
    self.min_extent_spin = QDoubleSpinBox()
    self.min_extent_spin.setRange(0.0, 1.0)
    self.min_extent_spin.setDecimals(3)
    self.min_extent_spin.setSingleStep(0.05)
    self.min_extent_spin.setValue(0.0)
    self.min_polygon_angle_spin = QDoubleSpinBox()
    self.min_polygon_angle_spin.setRange(0.0, 180.0)
    self.min_polygon_angle_spin.setDecimals(1)
    self.min_polygon_angle_spin.setSingleStep(5.0)
    self.min_polygon_angle_spin.setValue(90.0)

    self.conductor_group = QGroupBox("Conductor gradient")
    self.conductor_form = QFormLayout(self.conductor_group)
    self._configure_compact_form(self.conductor_form)
    self.conductor_gradient_checkbox = QCheckBox("Enabled")
    self.conductor_gradient_min_strength_spin = QDoubleSpinBox()
    self.conductor_gradient_min_strength_spin.setRange(0.0, 255.0)
    self.conductor_gradient_min_strength_spin.setDecimals(1)
    self.conductor_gradient_min_strength_spin.setSingleStep(1.0)
    self.conductor_gradient_min_strength_spin.setValue(18.0)
    self.conductor_gradient_band_radius_spin = QSpinBox()
    self.conductor_gradient_band_radius_spin.setRange(0, 25)
    self.conductor_gradient_band_radius_spin.setValue(3)

    self.via_group = QGroupBox("Via constraints")
    self.via_form = QFormLayout(self.via_group)
    self._configure_compact_form(self.via_form)
    self.via_size_mode_combo = QComboBox()
    self.via_size_mode_combo.addItem("Range", VIA_SIZE_MODE_RANGE)
    self.via_size_mode_combo.addItem("Fixed values", VIA_SIZE_MODE_FIXED)
    self.via_search_mode_combo = QComboBox()
    self.via_search_mode_combo.addItem("Hybrid", VIA_SEARCH_MODE_HYBRID)
    self.via_search_mode_combo.addItem("Blob only", VIA_SEARCH_MODE_BLOB)
    self.via_search_mode_combo.addItem("Template only", VIA_SEARCH_MODE_TEMPLATE)
    self.via_white_range_checkbox = QCheckBox("White range")
    self.via_white_range_checkbox.setChecked(True)
    self.via_white_range_min_spin = QSpinBox()
    self.via_white_range_min_spin.setRange(0, 255)
    self.via_white_range_min_spin.setValue(200)
    self.via_white_range_max_spin = QSpinBox()
    self.via_white_range_max_spin.setRange(0, 255)
    self.via_white_range_max_spin.setValue(255)
    self.via_white_range_widget = self._build_range_row(self.via_white_range_min_spin, self.via_white_range_max_spin)
    self.via_black_range_checkbox = QCheckBox("Black range")
    self.via_black_range_min_spin = QSpinBox()
    self.via_black_range_min_spin.setRange(0, 255)
    self.via_black_range_min_spin.setValue(0)
    self.via_black_range_max_spin = QSpinBox()
    self.via_black_range_max_spin.setRange(0, 255)
    self.via_black_range_max_spin.setValue(30)
    self.via_black_range_widget = self._build_range_row(self.via_black_range_min_spin, self.via_black_range_max_spin)
    self.via_range_checkboxes_widget = QWidget()
    via_range_checkboxes_layout = QHBoxLayout(self.via_range_checkboxes_widget)
    via_range_checkboxes_layout.setContentsMargins(0, 0, 0, 0)
    via_range_checkboxes_layout.setSpacing(16)
    via_range_checkboxes_layout.addWidget(self.via_white_range_checkbox)
    via_range_checkboxes_layout.addWidget(self.via_black_range_checkbox)
    via_range_checkboxes_layout.addStretch(1)
    self.via_min_score_spin = QDoubleSpinBox()
    self.via_min_score_spin.setRange(0.0, 1.0)
    self.via_min_score_spin.setDecimals(3)
    self.via_min_score_spin.setSingleStep(0.01)
    self.via_min_score_spin.setValue(0.35)
    self.via_min_contrast_spin = QDoubleSpinBox()
    self.via_min_contrast_spin.setRange(0.0, 255.0)
    self.via_min_contrast_spin.setDecimals(1)
    self.via_min_contrast_spin.setSingleStep(1.0)
    self.via_min_contrast_spin.setValue(14.0)
    self.via_min_edge_coverage_spin = QDoubleSpinBox()
    self.via_min_edge_coverage_spin.setRange(0.0, 1.0)
    self.via_min_edge_coverage_spin.setDecimals(3)
    self.via_min_edge_coverage_spin.setSingleStep(0.05)
    self.via_min_edge_coverage_spin.setValue(0.45)
    self.via_spot_line_suppression_spin = QDoubleSpinBox()
    self.via_spot_line_suppression_spin.setRange(0.0, 1.0)
    self.via_spot_line_suppression_spin.setDecimals(2)
    self.via_spot_line_suppression_spin.setSingleStep(0.05)
    self.via_spot_line_suppression_spin.setValue(0.65)
    self.via_template_min_score_spin = QDoubleSpinBox()
    self.via_template_min_score_spin.setRange(0.0, 1.0)
    self.via_template_min_score_spin.setDecimals(3)
    self.via_template_min_score_spin.setSingleStep(0.01)
    self.via_template_min_score_spin.setValue(0.35)
    self.via_templates_widget = QWidget()
    self.via_templates_layout = QVBoxLayout(self.via_templates_widget)
    self.via_templates_layout.setContentsMargins(0, 0, 0, 0)
    self.via_templates_layout.setSpacing(6)
    self.via_template_list = QListWidget()
    self.via_template_list.setMaximumHeight(96)
    self.via_template_list.setIconSize(QSize(56, 56))
    self.via_templates_layout.addWidget(self.via_template_list)
    via_template_buttons = QWidget()
    via_template_buttons_layout = QHBoxLayout(via_template_buttons)
    via_template_buttons_layout.setContentsMargins(0, 0, 0, 0)
    self.add_via_template_button = QPushButton("Pick template")
    self.add_via_template_button.setCheckable(True)
    self.remove_via_template_button = QPushButton("Remove selected")
    self.clear_via_templates_button = QPushButton("Clear templates")
    via_template_buttons_layout.addWidget(self.add_via_template_button)
    via_template_buttons_layout.addWidget(self.remove_via_template_button)
    via_template_buttons_layout.addWidget(self.clear_via_templates_button)
    self.via_templates_layout.addWidget(via_template_buttons)
    self.via_preset_combo = QComboBox()
    self.apply_via_preset_button = QPushButton("Apply preset")
    self.save_via_preset_button = QPushButton("Save preset")
    self.delete_via_preset_button = QPushButton("Delete preset")
    self.via_preset_widget = QWidget()
    via_preset_layout = QGridLayout(self.via_preset_widget)
    via_preset_layout.setContentsMargins(0, 0, 0, 0)
    via_preset_layout.setHorizontalSpacing(6)
    via_preset_layout.setVerticalSpacing(6)
    via_preset_layout.addWidget(self.via_preset_combo, 0, 0, 1, 3)
    via_preset_layout.addWidget(self.apply_via_preset_button, 1, 0)
    via_preset_layout.addWidget(self.save_via_preset_button, 1, 1)
    via_preset_layout.addWidget(self.delete_via_preset_button, 1, 2)
    self._refresh_via_preset_combo()
    self.noisy_traces_via_preset_button = QPushButton("Noisy traces preset")
    self.blurred_via_preset_button = QPushButton("Blurred vias preset")
    self.reset_via_search_button = QPushButton("Reset via search")
    self.debug_candidates_checkbox = QCheckBox("Debug recognition")
    self.show_gradient_debug_button = QPushButton("Show gradient map")
    self.gradient_overlay_checkbox = QCheckBox("Overlay on image")
    self.gradient_overlay_opacity_spin = QDoubleSpinBox()
    self.gradient_overlay_opacity_spin.setRange(0.05, 1.0)
    self.gradient_overlay_opacity_spin.setDecimals(2)
    self.gradient_overlay_opacity_spin.setSingleStep(0.05)
    self.gradient_overlay_opacity_spin.setValue(0.45)
    self.gradient_overlay_mode_combo = QComboBox()
    self.gradient_overlay_mode_combo.addItem("Heatmap", "heatmap")
    self.gradient_overlay_mode_combo.addItem("Threshold mask", "threshold")
    self.gradient_overlay_mode_combo.addItem("Raw elevation", "elevation")
    self.via_roundness_spin = QDoubleSpinBox()
    self.via_roundness_spin.setRange(0.0, 100.0)
    self.via_roundness_spin.setDecimals(1)
    self.via_roundness_spin.setSingleStep(1.0)
    self.via_roundness_spin.setValue(5.0)
    self.min_via_width_spin = QSpinBox()
    self.min_via_width_spin.setRange(0, 100_000)
    self.min_via_width_spin.setValue(0)
    self.max_via_width_spin = QSpinBox()
    self.max_via_width_spin.setRange(0, 100_000)
    self.max_via_width_spin.setValue(0)
    self.min_via_height_spin = QSpinBox()
    self.min_via_height_spin.setRange(0, 100_000)
    self.min_via_height_spin.setValue(0)
    self.max_via_height_spin = QSpinBox()
    self.max_via_height_spin.setRange(0, 100_000)
    self.max_via_height_spin.setValue(0)
    self.via_width_range_widget = self._build_range_row(self.min_via_width_spin, self.max_via_width_spin)
    self.via_height_range_widget = self._build_range_row(self.min_via_height_spin, self.max_via_height_spin)
    self.fixed_vias_widget = QWidget()
    self.fixed_vias_widget.setObjectName("fixedViaArea")
    self.fixed_vias_layout = QVBoxLayout(self.fixed_vias_widget)
    self.fixed_vias_layout.setContentsMargins(10, 10, 10, 10)
    self.fixed_vias_layout.setSpacing(8)
    self.fixed_vias_widget.setStyleSheet(
        "#fixedViaArea { background-color: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.10); border-radius: 8px; }"
        "#fixedViaArea QLabel, #fixedViaArea QSpinBox, #fixedViaArea QPushButton { border: none; background: transparent; }"
    )
    self.fixed_via_rows_widget = QWidget()
    self.fixed_via_rows_layout = QVBoxLayout(self.fixed_via_rows_widget)
    self.fixed_via_rows_layout.setContentsMargins(0, 0, 0, 0)
    self.fixed_via_rows_layout.setSpacing(6)
    self.fixed_vias_layout.addWidget(self.fixed_via_rows_widget)
    self.fixed_via_add_button = QPushButton("+")
    self.fixed_via_add_button.setMinimumHeight(38)
    self.fixed_via_add_button.setStyleSheet(
        "QPushButton { background-color: #2fbf71; color: white; font-size: 22px; font-weight: 700; border-radius: 8px; }"
        "QPushButton:hover { background-color: #28a764; }"
        "QPushButton:pressed { background-color: #229157; }"
    )
    self.fixed_via_add_button.clicked.connect(self._add_fixed_via_row)
    self.fixed_vias_layout.addWidget(self.fixed_via_add_button)

    self.topology_group = QGroupBox("Hierarchy and holes")
    self.topology_form = QFormLayout(self.topology_group)
    self._configure_compact_form(self.topology_form)
    self.min_hierarchy_depth_spin = QSpinBox()
    self.min_hierarchy_depth_spin.setRange(0, 100)
    self.min_hierarchy_depth_spin.setValue(0)
    self.max_hierarchy_depth_spin = QSpinBox()
    self.max_hierarchy_depth_spin.setRange(0, 100)
    self.max_hierarchy_depth_spin.setValue(0)
    self.max_hole_area_ratio_spin = QDoubleSpinBox()
    self.max_hole_area_ratio_spin.setRange(0.0, 10.0)
    self.max_hole_area_ratio_spin.setDecimals(3)
    self.max_hole_area_ratio_spin.setSingleStep(0.05)
    self.max_hole_area_ratio_spin.setValue(0.0)
    self.extraction_profile_combo.currentIndexChanged.connect(self._on_extraction_profile_changed)
    self.retrieval_mode_combo.currentIndexChanged.connect(self._on_extraction_settings_changed)
    self.approximation_mode_combo.currentIndexChanged.connect(self._on_extraction_settings_changed)
    self.epsilon_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.epsilon_relative_checkbox.stateChanged.connect(self._on_extraction_settings_changed)
    self.min_area_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.max_area_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.min_perimeter_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.min_points_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.max_perimeter_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.min_bbox_width_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.max_bbox_width_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.min_bbox_height_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.max_bbox_height_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.min_aspect_ratio_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.max_aspect_ratio_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.exclude_border_touching_checkbox.stateChanged.connect(self._on_extraction_settings_changed)
    self.min_solidity_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.min_extent_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.min_polygon_angle_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.conductor_gradient_checkbox.stateChanged.connect(self._on_extraction_settings_changed)
    self.conductor_gradient_min_strength_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.conductor_gradient_band_radius_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.via_size_mode_combo.currentIndexChanged.connect(self._on_via_size_mode_changed)
    self.via_search_mode_combo.currentIndexChanged.connect(self._on_extraction_settings_changed)
    self.via_white_range_checkbox.stateChanged.connect(self._on_extraction_settings_changed)
    self.via_white_range_min_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.via_white_range_max_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.via_black_range_checkbox.stateChanged.connect(self._on_extraction_settings_changed)
    self.via_black_range_min_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.via_black_range_max_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.via_min_score_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.via_min_contrast_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.via_min_edge_coverage_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.via_spot_line_suppression_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.via_template_min_score_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.add_via_template_button.toggled.connect(self._set_via_template_pick_active)
    self.remove_via_template_button.clicked.connect(self._remove_selected_via_template)
    self.clear_via_templates_button.clicked.connect(self._clear_via_templates)
    self.apply_via_preset_button.clicked.connect(self._apply_selected_via_preset)
    self.save_via_preset_button.clicked.connect(self._save_current_via_preset)
    self.delete_via_preset_button.clicked.connect(self._delete_selected_via_preset)
    self.noisy_traces_via_preset_button.clicked.connect(self._apply_noisy_traces_via_preset)
    self.blurred_via_preset_button.clicked.connect(self._apply_blurred_via_preset)
    self.reset_via_search_button.clicked.connect(self._reset_via_search_parameters)
    self.debug_candidates_checkbox.stateChanged.connect(self._on_extraction_settings_changed)
    self.via_roundness_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.min_via_width_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.max_via_width_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.min_via_height_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.max_via_height_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.min_hierarchy_depth_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.max_hierarchy_depth_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.max_hole_area_ratio_spin.valueChanged.connect(self._on_extraction_settings_changed)

    self.basic_filters_form.addRow("Retrieval mode", self.retrieval_mode_combo)
    self.retrieval_mode_label_widget = self.basic_filters_form.labelForField(self.retrieval_mode_combo)
    self.basic_filters_form.addRow("Approximation mode", self.approximation_mode_combo)
    self.approximation_mode_label_widget = self.basic_filters_form.labelForField(self.approximation_mode_combo)
    self.basic_filters_form.addRow("Epsilon", self.epsilon_spin)
    self.epsilon_label_widget = self.basic_filters_form.labelForField(self.epsilon_spin)
    self.basic_filters_form.addRow("Epsilon mode", self.epsilon_relative_checkbox)
    self.epsilon_mode_label_widget = self.basic_filters_form.labelForField(self.epsilon_relative_checkbox)
    self.basic_filters_form.addRow("Area range", self.area_range_widget)
    self.min_area_label_widget = self.basic_filters_form.labelForField(self.area_range_widget)
    self.max_area_label_widget = None
    self.basic_filters_form.addRow("Perimeter range", self.perimeter_range_widget)
    self.min_perimeter_label_widget = self.basic_filters_form.labelForField(self.perimeter_range_widget)
    self.max_perimeter_label_widget = None
    self.basic_filters_form.addRow("Min point count", self.min_points_spin)
    self.min_point_count_label_widget = self.basic_filters_form.labelForField(self.min_points_spin)

    self.geometry_filters_form.addRow("BBox width range", self.bbox_width_range_widget)
    self.min_bbox_width_label_widget = self.geometry_filters_form.labelForField(self.bbox_width_range_widget)
    self.max_bbox_width_label_widget = None
    self.geometry_filters_form.addRow("BBox height range", self.bbox_height_range_widget)
    self.min_bbox_height_label_widget = self.geometry_filters_form.labelForField(self.bbox_height_range_widget)
    self.max_bbox_height_label_widget = None
    self.geometry_filters_form.addRow("Aspect ratio range", self.aspect_ratio_range_widget)
    self.min_aspect_ratio_label_widget = self.geometry_filters_form.labelForField(self.aspect_ratio_range_widget)
    self.max_aspect_ratio_label_widget = None
    self.geometry_filters_form.addRow("Border handling", self.exclude_border_touching_checkbox)
    self.border_handling_label_widget = self.geometry_filters_form.labelForField(self.exclude_border_touching_checkbox)
    self.geometry_filters_form.addRow("Min solidity", self.min_solidity_spin)
    self.min_solidity_label_widget = self.geometry_filters_form.labelForField(self.min_solidity_spin)
    self.geometry_filters_form.addRow("Min extent", self.min_extent_spin)
    self.min_extent_label_widget = self.geometry_filters_form.labelForField(self.min_extent_spin)
    self.geometry_filters_form.addRow("Min polygon angle", self.min_polygon_angle_spin)
    self.min_polygon_angle_label_widget = self.geometry_filters_form.labelForField(self.min_polygon_angle_spin)

    self.conductor_form.addRow("Gradient boundaries", self.conductor_gradient_checkbox)
    self.conductor_gradient_enabled_label_widget = self.conductor_form.labelForField(self.conductor_gradient_checkbox)
    self.conductor_form.addRow("Min edge", self.conductor_gradient_min_strength_spin)
    self.conductor_gradient_min_strength_label_widget = self.conductor_form.labelForField(
        self.conductor_gradient_min_strength_spin
    )
    self.conductor_form.addRow("Boundary band", self.conductor_gradient_band_radius_spin)
    self.conductor_gradient_band_radius_label_widget = self.conductor_form.labelForField(
        self.conductor_gradient_band_radius_spin
    )

    self.via_form.addRow("Via size mode", self.via_size_mode_combo)
    self.via_size_mode_label_widget = self.via_form.labelForField(self.via_size_mode_combo)
    self.via_form.addRow("Via search mode", self.via_search_mode_combo)
    self.via_search_mode_label_widget = self.via_form.labelForField(self.via_search_mode_combo)
    self.via_form.addRow("Polarity", self.via_range_checkboxes_widget)
    self.via_range_checkboxes_label_widget = self.via_form.labelForField(self.via_range_checkboxes_widget)
    self.via_form.addRow("White range", self.via_white_range_widget)
    self.via_white_range_label_widget = self.via_form.labelForField(self.via_white_range_widget)
    self.via_form.addRow("Black range", self.via_black_range_widget)
    self.via_black_range_label_widget = self.via_form.labelForField(self.via_black_range_widget)
    self.via_form.addRow("Min score", self.via_min_score_spin)
    self.via_min_score_label_widget = self.via_form.labelForField(self.via_min_score_spin)
    self.via_form.addRow("Min contrast", self.via_min_contrast_spin)
    self.via_min_contrast_label_widget = self.via_form.labelForField(self.via_min_contrast_spin)
    self.via_form.addRow("Min edge coverage", self.via_min_edge_coverage_spin)
    self.via_min_edge_coverage_label_widget = self.via_form.labelForField(self.via_min_edge_coverage_spin)
    self.via_form.addRow("Spot trace suppression", self.via_spot_line_suppression_spin)
    self.via_spot_line_suppression_label_widget = self.via_form.labelForField(self.via_spot_line_suppression_spin)
    self.via_form.addRow("Template score", self.via_template_min_score_spin)
    self.via_template_min_score_label_widget = self.via_form.labelForField(self.via_template_min_score_spin)
    self.via_form.addRow("Templates", self.via_templates_widget)
    self.via_templates_label_widget = self.via_form.labelForField(self.via_templates_widget)
    self.via_form.addRow("Saved presets", self.via_preset_widget)
    self.via_preset_label_widget = self.via_form.labelForField(self.via_preset_widget)
    self.via_form.addRow("Preset", self.noisy_traces_via_preset_button)
    self.noisy_traces_via_preset_label_widget = self.via_form.labelForField(self.noisy_traces_via_preset_button)
    self.via_form.addRow("Preset", self.blurred_via_preset_button)
    self.blurred_via_preset_label_widget = self.via_form.labelForField(self.blurred_via_preset_button)
    self.via_form.addRow("Reset", self.reset_via_search_button)
    self.reset_via_search_label_widget = self.via_form.labelForField(self.reset_via_search_button)
    self.via_form.addRow("Debug", self.debug_candidates_checkbox)
    self.debug_candidates_label_widget = self.via_form.labelForField(self.debug_candidates_checkbox)
    self.via_form.addRow("Gradient debug", self.show_gradient_debug_button)
    self.show_gradient_debug_label_widget = self.via_form.labelForField(self.show_gradient_debug_button)
    self.show_gradient_debug_button.clicked.connect(self._show_gradient_debug_window)
    gradient_overlay_row = QWidget()
    gradient_overlay_row_layout = QHBoxLayout(gradient_overlay_row)
    gradient_overlay_row_layout.setContentsMargins(0, 0, 0, 0)
    gradient_overlay_row_layout.addWidget(self.gradient_overlay_checkbox)
    gradient_overlay_row_layout.addWidget(self.gradient_overlay_mode_combo, 1)
    gradient_overlay_row_layout.addWidget(self.gradient_overlay_opacity_spin)
    self.via_form.addRow("Gradient overlay", gradient_overlay_row)
    self.gradient_overlay_label_widget = self.via_form.labelForField(gradient_overlay_row)
    self.gradient_overlay_checkbox.toggled.connect(self._on_gradient_overlay_toggled)
    self.gradient_overlay_opacity_spin.valueChanged.connect(self._on_gradient_overlay_opacity_changed)
    self.gradient_overlay_mode_combo.currentIndexChanged.connect(self._refresh_gradient_overlay)
    self.via_form.addRow("Roundness", self.via_roundness_spin)
    self.via_roundness_label_widget = self.via_form.labelForField(self.via_roundness_spin)
    self.via_form.addRow("Via width range", self.via_width_range_widget)
    self.min_via_width_label_widget = self.via_form.labelForField(self.via_width_range_widget)
    self.max_via_width_label_widget = None
    self.via_form.addRow("Via height range", self.via_height_range_widget)
    self.min_via_height_label_widget = self.via_form.labelForField(self.via_height_range_widget)
    self.max_via_height_label_widget = None
    self.via_form.addRow("Fixed vias", self.fixed_vias_widget)
    self.fixed_vias_label_widget = self.via_form.labelForField(self.fixed_vias_widget)
    self._update_via_size_controls_state()

    self.topology_form.addRow("Min hierarchy depth", self.min_hierarchy_depth_spin)
    self.min_hierarchy_depth_label_widget = self.topology_form.labelForField(self.min_hierarchy_depth_spin)
    self.topology_form.addRow("Max hierarchy depth (0 = unlimited)", self.max_hierarchy_depth_spin)
    self.max_hierarchy_depth_label_widget = self.topology_form.labelForField(self.max_hierarchy_depth_spin)
    self.topology_form.addRow("Max hole area ratio (0 = unlimited)", self.max_hole_area_ratio_spin)
    self.max_hole_area_ratio_label_widget = self.topology_form.labelForField(self.max_hole_area_ratio_spin)

    for group in [
        self.profile_group,
        self.basic_filters_group,
        self.geometry_filters_group,
        self.conductor_group,
        self.via_group,
        self.topology_group,
    ]:
        contour_layout.addWidget(group)
    container_layout.addWidget(self.contour_group)

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
    container_layout.addWidget(self.save_group)
    container_layout.addStretch(1)
    scroll.setWidget(container)
    layout.addWidget(scroll, 1)
    return tab


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
    self.random_object_colors_checkbox = QCheckBox("Random object colors")
    self.show_neighbor_frames_checkbox = QCheckBox("Show neighboring frames")
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
    ]:
        if isinstance(widget, QCheckBox):
            widget.stateChanged.connect(self._apply_display_settings)
        else:
            widget.valueChanged.connect(self._apply_display_settings)
    self.show_neighbor_frames_checkbox.stateChanged.connect(self._on_neighbor_display_settings_changed)
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
    self.display_form.addRow(self.show_neighbor_frames_checkbox)
    self.display_form.addRow("Frames per row", self.neighbor_columns_spin)
    self.neighbor_columns_label_widget = self.display_form.labelForField(self.neighbor_columns_spin)
    self.display_form.addRow("Max neighbor grid", self.neighbor_max_grid_spin)
    self.neighbor_max_grid_label_widget = self.display_form.labelForField(self.neighbor_max_grid_spin)
    self.display_form.addRow("Neighbor opacity", self.neighbor_opacity_spin)
    self.neighbor_opacity_label_widget = self.display_form.labelForField(self.neighbor_opacity_spin)
    self.display_form.addRow("Frame overlap", self.neighbor_overlap_spin)
    self.neighbor_overlap_label_widget = self.display_form.labelForField(self.neighbor_overlap_spin)
    return tab


def build_help_tab(self) -> QWidget:
    tab = QWidget()
    layout = QVBoxLayout(tab)
    self.help_scroll = QScrollArea()
    self.help_scroll.setWidgetResizable(True)
    self.help_container = QWidget()
    self.help_layout = QVBoxLayout(self.help_container)
    self.help_layout.setContentsMargins(0, 0, 0, 0)
    self.help_scroll.setWidget(self.help_container)
    layout.addWidget(self.help_scroll, 1)
    self._rebuild_help_cards()
    return tab


def build_visual_panel(self) -> QWidget:
    panel = QWidget()
    layout = QVBoxLayout(panel)

    self.editor_group = QGroupBox("Image / polygon editor")
    editor_layout = QVBoxLayout(self.editor_group)
    self.polygon_editor = PolygonEditorView()
    self.polygon_editor.polygonsEdited.connect(self._on_polygons_edited)
    self.polygon_editor.logRequested.connect(self._append_log)
    self.polygon_editor.imageClicked.connect(self._on_editor_image_clicked)
    self.polygon_editor.imageRegionSelected.connect(self._on_editor_image_region_selected)
    self.polygon_editor.rulerMeasurementChanged.connect(self._update_ruler_status)
    self.polygon_editor.toolChanged.connect(self._on_editor_tool_changed)
    self.polygon_editor.zoomChanged.connect(lambda _zoom: self._sync_neighbor_frames())
    self.polygon_editor.neighborFrameActivated.connect(self._on_neighbor_frame_activated)
    self.polygon_editor.viaDebugRequested.connect(self._on_via_debug_requested)
    self.polygon_editor.middlePreviewHoldChanged.connect(self._on_middle_preview_hold_changed)
    self.editor_toolbar = self._build_editor_toolbar()
    editor_layout.addWidget(self.editor_toolbar)
    editor_layout.addWidget(self.polygon_editor, 1)

    layout.addWidget(self.editor_group, 1)
    return panel


def build_editor_toolbar(self) -> QWidget:
    toolbar = QWidget()
    layout = QHBoxLayout(toolbar)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(6)

    self._tool_button_group = QButtonGroup(self)
    self._tool_button_group.setExclusive(True)
    self._tool_buttons: dict[EditorTool, QToolButton] = {}
    for text, tool in [
        ("Select", EditorTool.SELECT),
        ("Select Area", EditorTool.SELECT_AREA),
        ("Pan", EditorTool.PAN),
        ("Ruler", EditorTool.RULER),
        ("Add Polygon", EditorTool.ADD_POLYGON),
        ("Brush", EditorTool.BRUSH),
        ("Via", EditorTool.ADD_VIA),
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
    self.brush_size_spin.valueChanged.connect(lambda value: self.polygon_editor.set_brush_thickness(float(value)))
    layout.addWidget(self.brush_size_label)
    layout.addWidget(self.brush_size_spin)

    self.via_width_label = QLabel("Via W")
    self.via_width_spin = QSpinBox()
    self.via_width_spin.setRange(1, 100_000)
    self.via_width_spin.setValue(12)
    self.via_width_spin.setFixedWidth(74)
    self.via_height_label = QLabel("Via H")
    self.via_height_spin = QSpinBox()
    self.via_height_spin.setRange(1, 100_000)
    self.via_height_spin.setValue(12)
    self.via_height_spin.setFixedWidth(74)
    self.via_width_spin.valueChanged.connect(lambda _value: self._sync_editor_via_size())
    self.via_height_spin.valueChanged.connect(lambda _value: self._sync_editor_via_size())
    layout.addWidget(self.via_width_label)
    layout.addWidget(self.via_width_spin)
    layout.addWidget(self.via_height_label)
    layout.addWidget(self.via_height_spin)

    self.delete_vertex_mode_label = QLabel("Delete")
    self.delete_vertex_mode_combo = QComboBox()
    self.delete_vertex_mode_combo.addItem(self._mode_text("delete_single"), DeleteVertexMode.SINGLE)
    self.delete_vertex_mode_combo.addItem(self._mode_text("delete_area"), DeleteVertexMode.AREA)
    self.delete_vertex_mode_combo.currentIndexChanged.connect(
        lambda _index: self.polygon_editor.set_delete_vertex_mode(self.delete_vertex_mode_combo.currentData())
    )
    layout.addWidget(self.delete_vertex_mode_label)
    layout.addWidget(self.delete_vertex_mode_combo)

    self.ruler_status_label = QLabel("")
    self.ruler_status_label.setMinimumWidth(180)
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
    self._sync_editor_via_size()
    self.polygon_editor.set_delete_vertex_mode(self.delete_vertex_mode_combo.currentData())
    return toolbar
