"""main_window: build matrix settings widget."""
from __future__ import annotations
from PyQt6.QtWidgets import (
    QGroupBox,
    QVBoxLayout,
    QWidget,
)
from ...ui.ui_constants import (
    DEFAULT_MATRIX_LAYOUT_MODE,
)


class BuildMatrixSettingsWidgetMixin:
    def _build_matrix_settings_widget(self) -> QWidget:
        widget = QWidget(self)
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)
        self._matrix_pixel_size_row = self._build_setting_row(self._t("matrix.pixel_size"), self.thumbnail_size_spin)
        self._matrix_analysis_mode_row = self._build_setting_row(self._t("analysis.mode"), self.analysis_mode_combo)
        self._matrix_geometry_row = self._build_setting_row(self._t("analysis.object_type"), self.geometry_mode_combo)
        self._matrix_polygon_compare_profile_row = self._build_setting_row(self._t("matrix.polygon_compare_profile"), self.polygon_compare_profile_combo)
        self._matrix_confidence_delta_row = self._build_setting_row(self._t('matrix.confidence_delta'), self.confidence_uncertainty_profile_combo)
        self._matrix_polygon_confidence_summary_row = self._build_setting_row(self._t('matrix.polygon_confidence_summary'), self.polygon_confidence_summary_combo)
        self._matrix_point_radius_row = self._build_setting_row(self._t('matrix.point_match_radius'), self.point_match_radius_spin)
        self._matrix_point_confidence_radius_row = self._build_setting_row(self._t('matrix.point_confidence_radius'), self.point_confidence_radius_spin)
        self._matrix_point_mode_row = self._build_setting_row(self._t('matrix.point_extraction_mode'), self.point_extraction_mode_combo)
        self._matrix_layout_row = self._build_setting_row(self._t("matrix.layout"), self.layout_mode_combo)
        self._matrix_score_view_row = self._build_setting_row(self._t("matrix.score_view"), self.matrix_score_view_combo)
        self._matrix_frame_type_filter_row = self._build_setting_row(self._t('matrix.frame_type_filter'), self.frame_type_filter_combo)
        self._matrix_total_frames_row = self._build_setting_row(self._t("matrix.total_frames"), self.total_frames_spin)
        self._matrix_frames_per_row_row = self._build_setting_row(self._t("matrix.frames_per_row"), self.frames_per_row_spin)
        self._matrix_rows_row = self._build_setting_row(self._t("matrix.rows"), self.matrix_rows_spin)
        self._matrix_columns_row = self._build_setting_row(self._t("matrix.columns"), self.matrix_columns_spin)
        self._analysis_task_group = QGroupBox(self._t("ui.analysis_task"), widget)
        analysis_layout = QVBoxLayout(self._analysis_task_group)
        analysis_layout.setContentsMargins(8, 8, 8, 8)
        analysis_layout.setSpacing(8)
        for row in (
            self._matrix_pixel_size_row,
            self._matrix_analysis_mode_row,
            self._matrix_geometry_row,
            self._matrix_polygon_compare_profile_row,
            self._matrix_confidence_delta_row,
            self._matrix_polygon_confidence_summary_row,
            self._matrix_point_radius_row,
            self._matrix_point_confidence_radius_row,
            self._matrix_point_mode_row,
        ):
            analysis_layout.addWidget(row)
        layout.addWidget(self._analysis_task_group)

        self._matrix_view_group = QGroupBox(self._t("ui.matrix_view"), widget)
        matrix_layout = QVBoxLayout(self._matrix_view_group)
        matrix_layout.setContentsMargins(8, 8, 8, 8)
        matrix_layout.setSpacing(8)
        for row in (
            self._matrix_layout_row,
            self._matrix_score_view_row,
            self._matrix_frame_type_filter_row,
            self._matrix_total_frames_row,
            self._matrix_frames_per_row_row,
            self._matrix_rows_row,
            self._matrix_columns_row,
        ):
            matrix_layout.addWidget(row)
        layout.addWidget(self._matrix_view_group)
        self._tile_grid_group = QGroupBox(self._t("matrix.subpixel_grid.group"), widget)
        tile_layout = QVBoxLayout(self._tile_grid_group)
        tile_layout.setContentsMargins(8, 8, 8, 8)
        tile_layout.setSpacing(8)
        self._subpixel_view_mode_row = self._build_setting_row(self._t("matrix.subpixel_view_mode"), self.subpixel_view_mode_combo)
        self._subpixel_rows_row = self._build_setting_row(self._t("matrix.subpixel_rows"), self.subpixel_rows_spin)
        self._subpixel_columns_row = self._build_setting_row(self._t("matrix.subpixel_columns"), self.subpixel_columns_spin)
        self._tile_width_row = self._build_setting_row(self._t("matrix.tile_width"), self.tile_width_spin)
        self._tile_height_row = self._build_setting_row(self._t("matrix.tile_height"), self.tile_height_spin)
        self._tile_overlap_row = self._build_setting_row(self._t("matrix.tile_overlap"), self.tile_overlap_spin)
        self._subpixel_aggregation_row = self._build_setting_row(self._t("matrix.subpixel_aggregation"), self.subpixel_aggregation_combo)
        self._subpixel_plan_row = self._build_setting_row(self._t("matrix.subpixel_plan"), self.tile_plan_label)
        for row in (
            self._subpixel_view_mode_row,
            self._subpixel_rows_row,
            self._subpixel_columns_row,
            self._tile_width_row,
            self._tile_height_row,
            self._tile_overlap_row,
            self._subpixel_aggregation_row,
            self._subpixel_plan_row,
        ):
            tile_layout.addWidget(row)
        layout.addWidget(self._tile_grid_group)
        self._matrix_pixel_size_row.setVisible(False)
        is_indexed_layout = str(self.layout_mode_combo.currentData() or DEFAULT_MATRIX_LAYOUT_MODE) == "indexed_grid"
        self._matrix_total_frames_row.setVisible(is_indexed_layout)
        self._matrix_frames_per_row_row.setVisible(is_indexed_layout)
        self._matrix_rows_row.setVisible(not is_indexed_layout)
        self._matrix_columns_row.setVisible(not is_indexed_layout)
        self.tile_overlap_mode_combo.hide()
        layout.addStretch(1)
        return widget
