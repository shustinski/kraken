"""main_window: retranslate ui."""
from __future__ import annotations
from PyQt6.QtWidgets import (
    QMainWindow,
)
from ...ui.ui_constants import (
    DEFAULT_CONFIDENCE_UNCERTAINTY_PROFILE,
    DEFAULT_ANALYSIS_MODE,
    DEFAULT_GEOMETRY_MODE,
    DEFAULT_POLYGON_COMPARE_PROFILE,
    DEFAULT_POINT_EXTRACTION_MODE,
    DEFAULT_POLYGON_CONFIDENCE_SUMMARY,
    DEFAULT_MATRIX_LAYOUT_MODE,
    DEFAULT_MATRIX_SCORE_VIEW_MODE,
    DEFAULT_MATRIX_METRIC_KEY,
    DEFAULT_METRIC_SCOPE,
    DEFAULT_SUBPIXEL_AGGREGATION,
    DEFAULT_SUBPIXEL_VIEW_MODE,
)


class RetranslateUiMixin:
    def retranslate_ui(self) -> None:
        self._t = self._i18n.tr
        self._setup_menu_bar()
        self._update_language_toggle_button()
        self.analysis_settings_group.setTitle(self._t("ui.analysis_setup"))
        self.metric_settings_group.setTitle(self._t("ui.metric_focus"))
        self._analysis_task_group.setTitle(self._t("ui.analysis_task"))
        self._matrix_view_group.setTitle(self._t("ui.matrix_view"))
        self._tile_grid_group.setTitle(self._t("matrix.subpixel_grid.group"))
        self._populate_matrix_score_view_combo(self.matrix_score_view_combo.currentData() or DEFAULT_MATRIX_SCORE_VIEW_MODE)
        self._populate_confidence_uncertainty_profile_combo(self.confidence_uncertainty_profile_combo.currentData() or DEFAULT_CONFIDENCE_UNCERTAINTY_PROFILE)
        self.folders_group.setTitle(self._t("folders.group"))
        self.folders_info_label.setText(self._t("folders.info"))
        self.btn_add_folder.setToolTip(self._t("folders.add_model"))
        self.btn_clear_folders.setToolTip(self._t("folders.clear_models"))
        self.btn_build.setToolTip(self._t("folders.build"))
        self.btn_compute.setToolTip(self._t("folders.compute_mismatch"))
        self.btn_cancel.setToolTip(self._t("folders.cancel"))
        self.source_group.setTitle(self._t("sources.group"))
        self.original_source_label.setText(self._t("sources.original"))
        self.gt_source_label.setText(self._t("sources.ground_truth"))
        self.btn_set_original.setText(self._t("common.set"))
        self.btn_clear_original.setText(self._t("common.clear"))
        self.btn_set_gt.setText(self._t("common.set"))
        self.btn_clear_gt.setText(self._t("common.clear"))
        current_layout = str(self.layout_mode_combo.currentData() or DEFAULT_MATRIX_LAYOUT_MODE)
        current_analysis_mode = str(self.analysis_mode_combo.currentData() or DEFAULT_ANALYSIS_MODE)
        current_geometry = str(self.geometry_mode_combo.currentData() or DEFAULT_GEOMETRY_MODE)
        current_metric_scope = str(self.metric_scope_combo.currentData() or DEFAULT_METRIC_SCOPE)
        current_metric = str(self.metric_combo.currentData() or DEFAULT_MATRIX_METRIC_KEY)
        current_frame_type_filter = str(self.frame_type_filter_combo.currentData() or 'all')
        current_point_extraction_mode = str(self.point_extraction_mode_combo.currentData() or DEFAULT_POINT_EXTRACTION_MODE)
        current_polygon_confidence_summary = str(self.polygon_confidence_summary_combo.currentData() or DEFAULT_POLYGON_CONFIDENCE_SUMMARY)
        self._populate_layout_mode_combo(current_layout)
        self._populate_analysis_mode_combo(current_analysis_mode)
        self._populate_geometry_mode_combo(current_geometry)
        self._populate_polygon_compare_profile_combo(self.polygon_compare_profile_combo.currentData() or DEFAULT_POLYGON_COMPARE_PROFILE)
        self._populate_point_extraction_mode_combo(current_point_extraction_mode)
        self._populate_subpixel_view_mode_combo(self.subpixel_view_mode_combo.currentData() or DEFAULT_SUBPIXEL_VIEW_MODE)
        self._populate_subpixel_aggregation_combo(self.subpixel_aggregation_combo.currentData() or DEFAULT_SUBPIXEL_AGGREGATION)
        self._populate_polygon_confidence_summary_combo(current_polygon_confidence_summary)
        current_state = self._presenter._current_tab_state() if hasattr(self, "_presenter") else None
        if current_state is not None:
            self._presenter._sync_metric_controls(
                current_state.build_result,
                preferred_metric_key=current_metric,
                preferred_scope_key=current_metric_scope,
                context_state=current_state,
            )
        else:
            self._populate_metric_scope_combo(None, current_metric_scope)
            self._populate_metric_combo(current_metric)
        self._populate_frame_type_filter_combo(current_frame_type_filter)
        for row, key in (
            (getattr(self, "_matrix_pixel_size_row", None), "matrix.pixel_size"),
            (getattr(self, "_matrix_analysis_mode_row", None), "analysis.mode"),
            (getattr(self, "_matrix_geometry_row", None), "analysis.object_type"),
            (getattr(self, "_matrix_polygon_compare_profile_row", None), "matrix.polygon_compare_profile"),
            (getattr(self, "_matrix_confidence_delta_row", None), "matrix.confidence_delta"),
            (getattr(self, "_matrix_polygon_confidence_summary_row", None), "matrix.polygon_confidence_summary"),
            (getattr(self, "_matrix_point_radius_row", None), "matrix.point_match_radius"),
            (getattr(self, "_matrix_point_confidence_radius_row", None), "matrix.point_confidence_radius"),
            (getattr(self, "_matrix_point_mode_row", None), "matrix.point_extraction_mode"),
            (getattr(self, "_matrix_layout_row", None), "matrix.layout"),
            (getattr(self, "_matrix_frame_type_filter_row", None), "matrix.frame_type_filter"),
            (getattr(self, "_subpixel_view_mode_row", None), "matrix.subpixel_view_mode"),
            (getattr(self, "_subpixel_rows_row", None), "matrix.subpixel_rows"),
            (getattr(self, "_subpixel_columns_row", None), "matrix.subpixel_columns"),
            (getattr(self, "_tile_width_row", None), "matrix.tile_width"),
            (getattr(self, "_tile_height_row", None), "matrix.tile_height"),
            (getattr(self, "_tile_overlap_row", None), "matrix.tile_overlap"),
            (getattr(self, "_subpixel_aggregation_row", None), "matrix.subpixel_aggregation"),
            (getattr(self, "_subpixel_plan_row", None), "matrix.subpixel_plan"),
            (getattr(self, "_matrix_total_frames_row", None), "matrix.total_frames"),
            (getattr(self, "_matrix_frames_per_row_row", None), "matrix.frames_per_row"),
            (getattr(self, "_matrix_rows_row", None), "matrix.rows"),
            (getattr(self, "_matrix_columns_row", None), "matrix.columns"),
            (getattr(self, "_metric_scope_row", None), "analysis.confidence_model"),
            (getattr(self, "_metric_select_row", None), "menu.metric.select"),
        ):
            label = getattr(row, "_title_label", None)
            if label is not None:
                label.setText(self._t(key))
        if hasattr(self, "_presenter"):
            current_state = self._presenter._current_tab_state()
            self._presenter._sync_mode_controls(current_state, None if current_state is None else current_state.build_result)
        for state in getattr(self, "_presenter", None)._tab_states.values() if hasattr(getattr(self, "_presenter", None), "_tab_states") else ():
            state.preview.group.setTitle(self._t("matrix.preview.group"))
            state.preview.frame_title.setText(self._t("matrix.preview.frame"))
            if state.preview.overall_group is not None:
                state.preview.overall_group.setTitle(self._t("matrix.preview.score_overall"))
            if state.preview.component_group is not None:
                state.preview.component_group.setTitle(self._t("matrix.preview.score_components"))
            for metric_key, card in state.preview.score_cards.items():
                if hasattr(card, "title_label"):
                    card.title_label.setText(self._metric_text_for_key(metric_key, state.build_result))
            for metric_key, card in state.preview.histogram_cards.items():
                if hasattr(card, "title_label"):
                    card.title_label.setText(self._metric_text_for_key(metric_key, state.build_result))
            if state.content_tabs is not None:
                state.content_tabs.setTabText(0, self._t("tab.matrix"))
                state.content_tabs.setTabText(1, self._t("tab.percentiles"))
            if state.repeated_bad_column is not None:
                state.repeated_bad_column.header_button.setText(self._t("correlation.bad"))
            if state.repeated_good_column is not None:
                state.repeated_good_column.header_button.setText(self._t("correlation.good"))
        window = self.window()
        if isinstance(window, QMainWindow):
            window.setWindowTitle(self._t("window.title"))
