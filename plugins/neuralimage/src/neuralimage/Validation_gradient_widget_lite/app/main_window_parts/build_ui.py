"""main_window: build ui."""
from __future__ import annotations
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMenuBar,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStyle,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from ...ui.ui_constants import (
    BOUNDARY_RADIUS_RANGE,
    CONTROL_PANEL_SPLITTER_SIZES,
    DEFAULT_BOUNDARY_RADIUS,
    DEFAULT_CELL_SIZE,
    DEFAULT_CONFIDENCE_UNCERTAINTY_PROFILE,
    DEFAULT_ANALYSIS_MODE,
    DEFAULT_FRAMES_PER_ROW,
    DEFAULT_GEOMETRY_MODE,
    DEFAULT_MASK_THRESHOLD,
    DEFAULT_POLYGON_COMPARE_PROFILE,
    DEFAULT_POINT_CONFIDENCE_RADIUS,
    DEFAULT_POINT_EXTRACTION_MODE,
    DEFAULT_POINT_MATCH_RADIUS,
    DEFAULT_POLYGON_CONFIDENCE_SUMMARY,
    DEFAULT_MATRIX_COLUMNS,
    DEFAULT_MATRIX_LAYOUT_MODE,
    DEFAULT_MATRIX_SCORE_VIEW_MODE,
    DEFAULT_MATRIX_METRIC_KEY,
    DEFAULT_METRIC_SCOPE,
    DEFAULT_MATRIX_ROWS,
    DEFAULT_SUBPIXEL_AGGREGATION,
    DEFAULT_SUBPIXEL_COLUMNS,
    DEFAULT_SUBPIXEL_ROWS,
    DEFAULT_SUBPIXEL_VIEW_MODE,
    DEFAULT_TILE_HEIGHT,
    DEFAULT_TILE_MODE,
    DEFAULT_TILE_OVERLAP,
    DEFAULT_TILE_OVERLAP_MODE,
    DEFAULT_TILE_WIDTH,
    DEFAULT_TOTAL_FRAMES,
    EXTEND_LANGUAGE_BUTTON_OBJECT_NAME,
    FRAMES_PER_ROW_RANGE,
    MASK_THRESHOLD_RANGE,
    MATRIX_COLUMNS_RANGE,
    MATRIX_METRIC_GROUP_OPTIONS,
    MATRIX_ROWS_RANGE,
    SUBPIXEL_COLUMNS_RANGE,
    SUBPIXEL_ROWS_RANGE,
    POINT_CONFIDENCE_RADIUS_RANGE,
    POINT_MATCH_RADIUS_RANGE,
    TILE_OVERLAP_RANGE,
    TILE_SIZE_RANGE,
    TOTAL_FRAMES_RANGE,
)

from neuralimage.Validation_gradient_widget_lite.app.main_window_parts.expandable_score_card import (
    _NoWheelComboBox,
    _NoWheelDoubleSpinBox,
    _NoWheelSpinBox,
)


class BuildUiMixin:
    def _build_ui(self) -> None:
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self._menu_bar = QMenuBar(self)
        root_layout.addWidget(self._menu_bar)

        content = QWidget(self)
        content_layout = QHBoxLayout(content)
        content_layout.setContentsMargins(8, 6, 8, 8)
        root_layout.addWidget(content, stretch=1)

        splitter = QSplitter(Qt.Orientation.Horizontal, content)
        content_layout.addWidget(splitter)

        control_scroll = QScrollArea(splitter)
        control_scroll.setWidgetResizable(True)
        control_host = QWidget(control_scroll)
        control_layout = QVBoxLayout(control_host)
        control_layout.setContentsMargins(0, 0, 0, 0)
        control_layout.setSpacing(10)
        control_scroll.setWidget(control_host)

        style = self.style()

        self.thumbnail_size_spin = _NoWheelSpinBox(self)
        self.thumbnail_size_spin.setRange(DEFAULT_CELL_SIZE, DEFAULT_CELL_SIZE)
        self.thumbnail_size_spin.setValue(DEFAULT_CELL_SIZE)
        self.thumbnail_size_spin.setEnabled(False)

        self.layout_mode_combo = _NoWheelComboBox(self)
        self._populate_layout_mode_combo(DEFAULT_MATRIX_LAYOUT_MODE)
        self.matrix_score_view_combo = _NoWheelComboBox(self)
        self._populate_matrix_score_view_combo(DEFAULT_MATRIX_SCORE_VIEW_MODE)

        self.total_frames_spin = _NoWheelSpinBox(self)
        self.total_frames_spin.setRange(*TOTAL_FRAMES_RANGE)
        self.total_frames_spin.setValue(DEFAULT_TOTAL_FRAMES)
        self.frames_per_row_spin = _NoWheelSpinBox(self)
        self.frames_per_row_spin.setRange(*FRAMES_PER_ROW_RANGE)
        self.frames_per_row_spin.setValue(DEFAULT_FRAMES_PER_ROW)
        self.matrix_rows_spin = _NoWheelSpinBox(self)
        self.matrix_rows_spin.setRange(*MATRIX_ROWS_RANGE)
        self.matrix_rows_spin.setValue(DEFAULT_MATRIX_ROWS)
        self.matrix_columns_spin = _NoWheelSpinBox(self)
        self.matrix_columns_spin.setRange(*MATRIX_COLUMNS_RANGE)
        self.matrix_columns_spin.setValue(DEFAULT_MATRIX_COLUMNS)
        self.analysis_mode_combo = _NoWheelComboBox(self)
        self._populate_analysis_mode_combo(DEFAULT_ANALYSIS_MODE)
        self.geometry_mode_combo = _NoWheelComboBox(self)
        self._populate_geometry_mode_combo(DEFAULT_GEOMETRY_MODE)
        self.mask_threshold_spin = _NoWheelDoubleSpinBox(self)
        self.mask_threshold_spin.setRange(*MASK_THRESHOLD_RANGE)
        self.mask_threshold_spin.setSingleStep(0.05)
        self.mask_threshold_spin.setDecimals(2)
        self.mask_threshold_spin.setValue(DEFAULT_MASK_THRESHOLD)
        self.mask_threshold_spin.setEnabled(False)
        self.mask_threshold_spin.hide()
        self.boundary_radius_spin = _NoWheelSpinBox(self)
        self.boundary_radius_spin.setRange(*BOUNDARY_RADIUS_RANGE)
        self.boundary_radius_spin.setValue(DEFAULT_BOUNDARY_RADIUS)
        self.boundary_radius_spin.setEnabled(False)
        self.boundary_radius_spin.hide()
        self.polygon_compare_profile_combo = _NoWheelComboBox(self)
        self._populate_polygon_compare_profile_combo(DEFAULT_POLYGON_COMPARE_PROFILE)
        self.confidence_uncertainty_profile_combo = _NoWheelComboBox(self)
        self._populate_confidence_uncertainty_profile_combo(DEFAULT_CONFIDENCE_UNCERTAINTY_PROFILE)
        self.polygon_confidence_summary_combo = _NoWheelComboBox(self)
        self._populate_polygon_confidence_summary_combo(DEFAULT_POLYGON_CONFIDENCE_SUMMARY)
        self.point_match_radius_spin = _NoWheelDoubleSpinBox(self)
        self.point_match_radius_spin.setRange(*POINT_MATCH_RADIUS_RANGE)
        self.point_match_radius_spin.setSingleStep(0.5)
        self.point_match_radius_spin.setDecimals(1)
        self.point_match_radius_spin.setValue(DEFAULT_POINT_MATCH_RADIUS)
        self.point_confidence_radius_spin = _NoWheelSpinBox(self)
        self.point_confidence_radius_spin.setRange(*POINT_CONFIDENCE_RADIUS_RANGE)
        self.point_confidence_radius_spin.setValue(DEFAULT_POINT_CONFIDENCE_RADIUS)
        self.point_extraction_mode_combo = _NoWheelComboBox(self)
        self._populate_point_extraction_mode_combo(DEFAULT_POINT_EXTRACTION_MODE)
        self.tile_mode_combo = _NoWheelComboBox(self)
        self._populate_tile_mode_combo(DEFAULT_TILE_MODE)
        self.tile_mode_combo.hide()
        self.subpixel_view_mode_combo = _NoWheelComboBox(self)
        self._populate_subpixel_view_mode_combo(DEFAULT_SUBPIXEL_VIEW_MODE)
        self.subpixel_rows_spin = _NoWheelSpinBox(self)
        self.subpixel_rows_spin.setRange(*SUBPIXEL_ROWS_RANGE)
        self.subpixel_rows_spin.setValue(DEFAULT_SUBPIXEL_ROWS)
        self.subpixel_columns_spin = _NoWheelSpinBox(self)
        self.subpixel_columns_spin.setRange(*SUBPIXEL_COLUMNS_RANGE)
        self.subpixel_columns_spin.setValue(DEFAULT_SUBPIXEL_COLUMNS)
        self.subpixel_aggregation_combo = _NoWheelComboBox(self)
        self._populate_subpixel_aggregation_combo(DEFAULT_SUBPIXEL_AGGREGATION)
        self.tile_width_spin = _NoWheelSpinBox(self)
        self.tile_width_spin.setRange(*TILE_SIZE_RANGE)
        self.tile_width_spin.setValue(DEFAULT_TILE_WIDTH)
        self.tile_height_spin = _NoWheelSpinBox(self)
        self.tile_height_spin.setRange(*TILE_SIZE_RANGE)
        self.tile_height_spin.setValue(DEFAULT_TILE_HEIGHT)
        self.tile_overlap_mode_combo = _NoWheelComboBox(self)
        self._populate_tile_overlap_mode_combo(DEFAULT_TILE_OVERLAP_MODE)
        self.tile_overlap_spin = _NoWheelSpinBox(self)
        self.tile_overlap_spin.setRange(*TILE_OVERLAP_RANGE)
        self.tile_overlap_spin.setValue(DEFAULT_TILE_OVERLAP)
        self.tile_plan_label = QLabel(self._t("matrix.subpixel_plan.waiting"), self)
        self.tile_plan_label.setWordWrap(True)

        self.metric_group_combo = _NoWheelComboBox(self)
        for label, key in MATRIX_METRIC_GROUP_OPTIONS:
            self.metric_group_combo.addItem(self._t(label), key)
        self.metric_group_combo.hide()
        self.metric_scope_combo = _NoWheelComboBox(self)
        self._populate_metric_scope_combo(None, DEFAULT_METRIC_SCOPE)
        self.metric_combo = _NoWheelComboBox(self)
        self._populate_metric_combo(DEFAULT_MATRIX_METRIC_KEY)
        self.frame_type_filter_combo = _NoWheelComboBox(self)
        self._populate_frame_type_filter_combo('all')

        folders_group = QGroupBox(self._t("folders.group"), control_host)
        self.folders_group = folders_group
        folders_layout = QVBoxLayout(folders_group)
        folders_info = QLabel(self._t("folders.info"), folders_group)
        self.folders_info_label = folders_info
        folders_info.setWordWrap(True)
        folders_info.hide()
        folders_layout.addWidget(folders_info)

        toolbar_layout = QHBoxLayout()
        toolbar_layout.setContentsMargins(0, 0, 0, 0)
        toolbar_layout.setSpacing(4)

        self.btn_add_folder = QToolButton(folders_group)
        self.btn_add_folder.setAutoRaise(True)
        self.btn_add_folder.setProperty("toolbarButton", True)
        self.btn_add_folder.setProperty("liteToolbarButton", True)
        self.btn_add_folder.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_FileDialogNewFolder))
        self.btn_add_folder.setToolTip(self._t("folders.add_model"))
        toolbar_layout.addWidget(self.btn_add_folder)

        self.btn_clear_folders = QToolButton(folders_group)
        self.btn_clear_folders.setAutoRaise(True)
        self.btn_clear_folders.setProperty("toolbarButton", True)
        self.btn_clear_folders.setText("x")
        self.btn_clear_folders.setToolTip(self._t("folders.clear_models"))
        toolbar_layout.addWidget(self.btn_clear_folders)

        self.btn_build = QToolButton(folders_group)
        self.btn_build.setAutoRaise(True)
        self.btn_build.setProperty("toolbarButton", True)
        self.btn_build.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.btn_build.setToolTip(self._t("folders.build"))
        toolbar_layout.addWidget(self.btn_build)

        self.btn_compute = QToolButton(folders_group)
        self.btn_compute.setAutoRaise(True)
        self.btn_compute.setProperty("toolbarButton", True)
        self.btn_compute.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_BrowserReload))
        self.btn_compute.setToolTip(self._t("folders.compute_mismatch"))
        toolbar_layout.addWidget(self.btn_compute)

        self.btn_cancel = QToolButton(folders_group)
        self.btn_cancel.setAutoRaise(True)
        self.btn_cancel.setProperty("toolbarButton", True)
        self.btn_cancel.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_BrowserStop))
        self.btn_cancel.setToolTip(self._t("folders.cancel"))
        toolbar_layout.addWidget(self.btn_cancel)
        toolbar_layout.addStretch(1)
        folders_layout.addLayout(toolbar_layout)

        self.build_progress = QProgressBar(folders_group)
        self.build_progress.setTextVisible(True)
        self.build_progress.setRange(0, 1)
        self.build_progress.setValue(0)
        self.build_progress.hide()
        folders_layout.addWidget(self.build_progress)

        self.folder_list = QListWidget(folders_group)
        self.folder_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.folder_list.setSpacing(2)
        folders_layout.addWidget(self.folder_list, stretch=1)
        control_layout.addWidget(folders_group)

        source_group = QGroupBox(self._t("sources.group"), control_host)
        self.source_group = source_group
        source_layout = QFormLayout(source_group)
        source_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        original_row = QWidget(source_group)
        original_row_layout = QHBoxLayout(original_row)
        original_row_layout.setContentsMargins(0, 0, 0, 0)
        original_row_layout.setSpacing(6)
        self.original_folder_value = QLabel(self._t("sources.not_set"), original_row)
        self.original_folder_value.setWordWrap(True)
        self.original_folder_value.setMinimumWidth(0)
        self.original_folder_value.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.btn_set_original = QToolButton(original_row)
        self.btn_set_original.setText(self._t("common.set"))
        self.btn_clear_original = QToolButton(original_row)
        self.btn_clear_original.setText(self._t("common.clear"))
        original_row_layout.addWidget(self.original_folder_value, stretch=1)
        original_row_layout.addWidget(self.btn_set_original)
        original_row_layout.addWidget(self.btn_clear_original)
        self.original_source_label = QLabel(self._t("sources.original"), source_group)
        source_layout.addRow(self.original_source_label, original_row)

        gt_row = QWidget(source_group)
        gt_row_layout = QHBoxLayout(gt_row)
        gt_row_layout.setContentsMargins(0, 0, 0, 0)
        gt_row_layout.setSpacing(6)
        self.gt_folder_value = QLabel(self._t("sources.not_set"), gt_row)
        self.gt_folder_value.setWordWrap(True)
        self.gt_folder_value.setMinimumWidth(0)
        self.gt_folder_value.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.btn_set_gt = QToolButton(gt_row)
        self.btn_set_gt.setText(self._t("common.set"))
        self.btn_clear_gt = QToolButton(gt_row)
        self.btn_clear_gt.setText(self._t("common.clear"))
        gt_row_layout.addWidget(self.gt_folder_value, stretch=1)
        gt_row_layout.addWidget(self.btn_set_gt)
        gt_row_layout.addWidget(self.btn_clear_gt)
        self.gt_source_label = QLabel(self._t("sources.ground_truth"), source_group)
        source_layout.addRow(self.gt_source_label, gt_row)
        control_layout.addWidget(source_group)

        self.matrix_tabs = QTabWidget(splitter)
        self.matrix_tabs.setTabsClosable(True)
        self.matrix_tabs.setMovable(True)
        self.matrix_tabs.setDocumentMode(True)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes(list(CONTROL_PANEL_SPLITTER_SIZES))

        self.metric_settings_group = QGroupBox(self._t("ui.metric_focus"), control_host)
        metric_settings_layout = QVBoxLayout(self.metric_settings_group)
        metric_settings_layout.setContentsMargins(4, 4, 4, 4)
        metric_settings_layout.setSpacing(0)
        metric_settings_layout.addWidget(self._build_metric_settings_widget())
        control_layout.addWidget(self.metric_settings_group)

        self.analysis_settings_group = QGroupBox(self._t("ui.analysis_setup"), control_host)
        analysis_settings_layout = QVBoxLayout(self.analysis_settings_group)
        analysis_settings_layout.setContentsMargins(6, 6, 6, 6)
        analysis_settings_layout.setSpacing(0)
        analysis_settings_layout.addWidget(self._build_matrix_settings_widget())
        control_layout.addWidget(self.analysis_settings_group)

        self.language_toggle_button = QToolButton(self._menu_bar)
        self.language_toggle_button.setAutoRaise(True)
        self.language_toggle_button.setObjectName(EXTEND_LANGUAGE_BUTTON_OBJECT_NAME)
        self.language_toggle_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._update_language_toggle_button()

        control_layout.addStretch(1)
