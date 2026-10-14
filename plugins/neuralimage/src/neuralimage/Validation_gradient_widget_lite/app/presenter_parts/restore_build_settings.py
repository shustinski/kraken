"""presenter: restore build settings."""
from __future__ import annotations
from PyQt6.QtCore import QSignalBlocker
from ...core.analysis_modes import (
    geometry_mode_for_object_type,
)
from ...ui.ui_constants import (
    DEFAULT_CELL_SIZE,
    DEFAULT_FRAMES_PER_ROW,
    DEFAULT_GEOMETRY_MODE,
    DEFAULT_MATRIX_COLUMNS,
    DEFAULT_MATRIX_LAYOUT_MODE,
    DEFAULT_MATRIX_SCORE_VIEW_MODE,
    DEFAULT_MATRIX_METRIC_KEY,
    DEFAULT_MATRIX_ROWS,
    DEFAULT_SUBPIXEL_AGGREGATION,
    DEFAULT_SUBPIXEL_COLUMNS,
    DEFAULT_SUBPIXEL_ROWS,
    DEFAULT_SUBPIXEL_VIEW_MODE,
    DEFAULT_POINT_CONFIDENCE_RADIUS,
    DEFAULT_POINT_EXTRACTION_MODE,
    DEFAULT_POLYGON_CONFIDENCE_SUMMARY,
    DEFAULT_TILE_HEIGHT,
    DEFAULT_TILE_MODE,
    DEFAULT_TILE_OVERLAP,
    DEFAULT_TILE_OVERLAP_MODE,
    DEFAULT_TILE_WIDTH,
    DEFAULT_TOTAL_FRAMES,
)


class RestoreBuildSettingsMixin:
    def _restore_build_settings(self) -> None:
        payload = self._settings_service.load_build_settings_payload() or {}
        blockers = [
            QSignalBlocker(self.thumbnail_size_spin),
            QSignalBlocker(self.matrix_score_view_combo),
            QSignalBlocker(self.analysis_mode_combo),
            QSignalBlocker(self.geometry_mode_combo),
            QSignalBlocker(self.polygon_compare_profile_combo),
            QSignalBlocker(self.mask_threshold_spin),
            QSignalBlocker(self.boundary_radius_spin),
            QSignalBlocker(self.confidence_uncertainty_profile_combo),
            QSignalBlocker(self.point_match_radius_spin),
            QSignalBlocker(self.point_confidence_radius_spin),
            QSignalBlocker(self.point_extraction_mode_combo),
            QSignalBlocker(self.tile_mode_combo),
            QSignalBlocker(self.subpixel_view_mode_combo),
            QSignalBlocker(self.subpixel_rows_spin),
            QSignalBlocker(self.subpixel_columns_spin),
            QSignalBlocker(self.subpixel_aggregation_combo),
            QSignalBlocker(self.tile_width_spin),
            QSignalBlocker(self.tile_height_spin),
            QSignalBlocker(self.tile_overlap_mode_combo),
            QSignalBlocker(self.tile_overlap_spin),
            QSignalBlocker(self.polygon_confidence_summary_combo),
            QSignalBlocker(self.layout_mode_combo),
            QSignalBlocker(self.total_frames_spin),
            QSignalBlocker(self.frames_per_row_spin),
            QSignalBlocker(self.matrix_rows_spin),
            QSignalBlocker(self.matrix_columns_spin),
            QSignalBlocker(self.metric_group_combo),
            QSignalBlocker(self.metric_scope_combo),
            QSignalBlocker(self.metric_combo),
            QSignalBlocker(self.frame_type_filter_combo),
        ]
        _ = blockers
        self.thumbnail_size_spin.setValue(int(DEFAULT_CELL_SIZE))
        score_view_mode = str(payload.get("matrix_score_view_mode") or DEFAULT_MATRIX_SCORE_VIEW_MODE)
        score_view_index = self.matrix_score_view_combo.findData(score_view_mode)
        self.matrix_score_view_combo.setCurrentIndex(score_view_index if score_view_index >= 0 else 0)
        analysis_mode = str(payload.get("analysis_mode") or self._selected_analysis_mode())
        analysis_index = self.analysis_mode_combo.findData(analysis_mode)
        self.analysis_mode_combo.setCurrentIndex(analysis_index if analysis_index >= 0 else 0)
        geometry_mode = str(payload.get("geometry_mode") or geometry_mode_for_object_type(payload.get("object_type")).value or DEFAULT_GEOMETRY_MODE)
        geometry_index = self.geometry_mode_combo.findData(geometry_mode)
        self.geometry_mode_combo.setCurrentIndex(geometry_index if geometry_index >= 0 else 0)
        tile_mode = str(payload.get("tile_mode") or DEFAULT_TILE_MODE)
        if tile_mode == "subpixel":
            tile_mode = "tile"
        tile_mode_index = self.tile_mode_combo.findData(tile_mode)
        self.tile_mode_combo.setCurrentIndex(tile_mode_index if tile_mode_index >= 0 else 0)
        subpixel_view_mode = str(payload.get("subpixel_view_mode") or DEFAULT_SUBPIXEL_VIEW_MODE)
        if subpixel_view_mode == "subpixel":
            subpixel_view_mode = "tile"
        subpixel_view_mode_index = self.subpixel_view_mode_combo.findData(subpixel_view_mode)
        self.subpixel_view_mode_combo.setCurrentIndex(subpixel_view_mode_index if subpixel_view_mode_index >= 0 else 0)
        self.subpixel_rows_spin.setValue(int(payload.get("subpixel_rows", DEFAULT_SUBPIXEL_ROWS)))
        self.subpixel_columns_spin.setValue(int(payload.get("subpixel_columns", DEFAULT_SUBPIXEL_COLUMNS)))
        subpixel_aggregation = str(payload.get("subpixel_aggregation") or DEFAULT_SUBPIXEL_AGGREGATION)
        subpixel_aggregation_index = self.subpixel_aggregation_combo.findData(subpixel_aggregation)
        self.subpixel_aggregation_combo.setCurrentIndex(subpixel_aggregation_index if subpixel_aggregation_index >= 0 else 0)
        self.tile_width_spin.setValue(int(payload.get("tile_width", DEFAULT_TILE_WIDTH)))
        self.tile_height_spin.setValue(int(payload.get("tile_height", DEFAULT_TILE_HEIGHT)))
        tile_overlap_mode = str(payload.get("tile_overlap_mode") or DEFAULT_TILE_OVERLAP_MODE)
        tile_overlap_mode_index = self.tile_overlap_mode_combo.findData(tile_overlap_mode)
        self.tile_overlap_mode_combo.setCurrentIndex(tile_overlap_mode_index if tile_overlap_mode_index >= 0 else 0)
        self.tile_overlap_spin.setValue(int(payload.get("tile_overlap", DEFAULT_TILE_OVERLAP)))
        compare_profile = str(payload.get("polygon_compare_profile") or "")
        self.mask_threshold_spin.setValue(float(payload.get("mask_threshold", self.mask_threshold_spin.value())))
        self.boundary_radius_spin.setValue(int(payload.get("boundary_radius", self.boundary_radius_spin.value())))
        if not compare_profile:
            compare_profile = self._polygon_compare_profile_for_values(self.mask_threshold_spin.value(), self.boundary_radius_spin.value())
        compare_index = self.polygon_compare_profile_combo.findData(compare_profile)
        self.polygon_compare_profile_combo.setCurrentIndex(compare_index if compare_index >= 0 else 0)
        uncertainty_profile = str(payload.get("confidence_uncertainty_profile") or "")
        if not uncertainty_profile:
            uncertainty_profile = self._confidence_uncertainty_profile_for_value(payload.get("confidence_uncertainty_delta"))
        uncertainty_index = self.confidence_uncertainty_profile_combo.findData(uncertainty_profile)
        self.confidence_uncertainty_profile_combo.setCurrentIndex(uncertainty_index if uncertainty_index >= 0 else 0)
        self.point_match_radius_spin.setValue(float(payload.get("point_match_radius", self.point_match_radius_spin.value())))
        self.point_confidence_radius_spin.setValue(int(payload.get("point_confidence_radius", DEFAULT_POINT_CONFIDENCE_RADIUS)))
        point_extraction_mode = str(payload.get("point_extraction_mode") or DEFAULT_POINT_EXTRACTION_MODE)
        point_mode_index = self.point_extraction_mode_combo.findData(point_extraction_mode)
        self.point_extraction_mode_combo.setCurrentIndex(point_mode_index if point_mode_index >= 0 else 0)
        polygon_confidence_summary = str(payload.get("polygon_confidence_summary") or DEFAULT_POLYGON_CONFIDENCE_SUMMARY)
        polygon_summary_index = self.polygon_confidence_summary_combo.findData(polygon_confidence_summary)
        self.polygon_confidence_summary_combo.setCurrentIndex(polygon_summary_index if polygon_summary_index >= 0 else 0)
        layout_mode = str(payload.get("layout_mode") or DEFAULT_MATRIX_LAYOUT_MODE)
        layout_index = self.layout_mode_combo.findData(layout_mode)
        self.layout_mode_combo.setCurrentIndex(layout_index if layout_index >= 0 else 0)
        self.total_frames_spin.setValue(int(payload.get("total_frames", DEFAULT_TOTAL_FRAMES)))
        self.frames_per_row_spin.setValue(int(payload.get("frames_per_row", DEFAULT_FRAMES_PER_ROW)))
        self.matrix_rows_spin.setValue(int(payload.get("rows", DEFAULT_MATRIX_ROWS)))
        self.matrix_columns_spin.setValue(int(payload.get("columns", DEFAULT_MATRIX_COLUMNS)))
        metric_key = str(payload.get("metric_key") or DEFAULT_MATRIX_METRIC_KEY)
        metric_scope = str(payload.get("confidence_model_id") or payload.get("metric_scope") or self._metric_scope_for_metric_key(metric_key) or "")
        frame_type_filter = str(payload.get('frame_type_filter') or self._selected_object_type())
        self._sync_metric_controls(None, preferred_metric_key=metric_key, preferred_scope_key=metric_scope)
        index = self.frame_type_filter_combo.findData(frame_type_filter)
        self.frame_type_filter_combo.setCurrentIndex(index if index >= 0 else 0)
        self._sync_tile_overlap_bounds()
        self._sync_mode_controls(None, None)
