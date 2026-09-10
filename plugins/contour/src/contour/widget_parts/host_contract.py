"""Shared typing contract for fields built on the assembled Contour widget.

No runtime state or methods are introduced here. The UI builders and existing
mixins remain responsible for initialization and implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    from PyQt6.QtCore import QModelIndex, QThreadPool, pyqtBoundSignal
    from PyQt6.QtWidgets import (
        QAbstractSpinBox,
        QCheckBox,
        QFormLayout,
        QGroupBox,
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QPushButton,
        QSlider,
        QToolButton,
        QTreeWidget,
        QTreeWidgetItem,
        QVBoxLayout,
        QWidget,
    )

    from contour.application.processing import ContourExtractionSettings
    from contour.application.services import DirectoryScanController, VectorIndexController, WorkspaceSession
    from contour.application.use_cases import AutoTuneResult
    from contour.batch_processor import BatchProcessor
    from contour.domain import PolygonData
    from contour.graphics_view import EditorTool, PolygonEditorView
    from contour.infrastructure import WidgetMetalPresetSettingsStore, WidgetViaPresetSettingsStore
    from contour.infrastructure.frame_switch_profiler import FrameSwitchProfile
    from contour.pipeline import PreprocessingPipeline
    from contour.ui.frame_path_list_model import FramePathListView
    from contour.ui.no_wheel_controls import NoWheelComboBox as QComboBox
    from contour.ui.no_wheel_controls import NoWheelDoubleSpinBox as QDoubleSpinBox
    from contour.ui.no_wheel_controls import NoWheelSpinBox as QSpinBox
    from contour.ui.pipeline_list import PipelineListWidget


if TYPE_CHECKING:
    _WidgetBase = QWidget
else:
    _WidgetBase = object


class WidgetMixinHost(_WidgetBase):
    if TYPE_CHECKING:
        _active_extraction_profile: str
        _asset_list_build_generation: int
        _auto_tune_request_serial: int
        _auto_tune_running_request_id: int | None
        _auto_tune_thread_pool: QThreadPool
        _batch_processor: BatchProcessor
        _cif_load_failure_stems: set[str]
        _contour_settings_profiles: dict[str, ContourExtractionSettings]
        _directory_scanner: DirectoryScanController
        _fixed_via_rows: list[dict[str, QWidget]]
        _ignore_pipeline_item_change: bool
        _metal_preset_settings_store: WidgetMetalPresetSettingsStore
        _parameter_widgets: dict[str, QWidget]
        _persisted_highlight_paths: set[str]
        _pipeline: PreprocessingPipeline
        _preview_running_request_id: int | None
        _suspend_fixed_via_updates: bool
        _tool_buttons: dict[EditorTool, QToolButton]
        _ui_language: str
        _user_metal_presets: dict[str, dict[str, object]]
        _user_via_presets: dict[str, dict[str, object]]
        _vector_indexer: VectorIndexController
        _via_preset_settings_store: WidgetViaPresetSettingsStore
        _via_template_diameters: list[int]
        _via_template_images: list[np.ndarray]
        _via_template_min_scores: list[float]
        _viewed_image_paths: set[str]
        _workspace: WorkspaceSession
        advanced_extraction_checkbox: QCheckBox
        approximation_mode_combo: QComboBox
        approximation_mode_label_widget: QWidget | None
        basic_filters_group: QGroupBox
        border_handling_label_widget: QWidget | None
        bright_via_advanced_outer: QGroupBox
        bright_via_black_range_label_widget: QWidget | None
        bright_via_bright_center_score_spin: QDoubleSpinBox
        bright_via_clahe_clip_spin: QDoubleSpinBox
        bright_via_clahe_tile_spin: QSpinBox
        bright_via_diameter_max_spin: QSpinBox
        bright_via_diameter_min_spin: QSpinBox
        bright_via_diameter_range_label_widget: QWidget | None
        bright_via_diameter_range_widget: QWidget
        bright_via_display_group: QGroupBox
        bright_via_dog_large_spin: QDoubleSpinBox
        bright_via_dog_small_spin: QDoubleSpinBox
        bright_via_expert_candidates_header: QLabel
        bright_via_expert_geometry_header: QLabel
        bright_via_expert_score_header: QLabel
        bright_via_hard_asym_checkbox: QCheckBox
        bright_via_hard_edge_checkbox: QCheckBox
        bright_via_hard_line_checkbox: QCheckBox
        bright_via_mask_combine_combo: QComboBox
        bright_via_max_area_factor_spin: QDoubleSpinBox
        bright_via_max_aspect_spin: QDoubleSpinBox
        bright_via_max_edge_likeness_spin: QDoubleSpinBox
        bright_via_max_line_likeness_spin: QDoubleSpinBox
        bright_via_max_radial_asymmetry_spin: QDoubleSpinBox
        bright_via_median_kernel_spin: QSpinBox
        bright_via_metal_constraint_combo: QComboBox
        bright_via_metal_fraction_spin: QDoubleSpinBox
        bright_via_min_area_factor_spin: QDoubleSpinBox
        bright_via_min_aspect_spin: QDoubleSpinBox
        bright_via_min_circularity_spin: QDoubleSpinBox
        bright_via_min_final_score_spin: QDoubleSpinBox
        bright_via_min_isolation_spin: QDoubleSpinBox
        bright_via_mode_hint: QLabel
        bright_via_mode_stack_label_widget: QWidget | None
        bright_via_nms_distance_spin: QSpinBox
        bright_via_polarity_label_widget: QWidget | None
        bright_via_quality_group: QGroupBox
        bright_via_show_rejected_checkbox: QCheckBox
        bright_via_threshold_percentile_spin: QDoubleSpinBox
        bright_via_tophat_kernel_spin: QSpinBox
        bright_via_white_range_label_widget: QWidget | None
        brush_mode_combo: QComboBox
        cif_dir_edit: QLineEdit
        conductor_gradient_band_radius_spin: QSpinBox
        conductor_gradient_checkbox: QCheckBox
        conductor_gradient_min_strength_spin: QDoubleSpinBox
        conductor_group: QGroupBox
        debug_candidates_checkbox: QCheckBox
        debug_candidates_label_widget: QWidget | None
        delete_vertex_mode_combo: QComboBox
        epsilon_label_widget: QWidget | None
        epsilon_mode_label_widget: QWidget | None
        epsilon_relative_checkbox: QCheckBox
        epsilon_slider: QSlider
        epsilon_spin: QDoubleSpinBox
        exclude_border_touching_checkbox: QCheckBox
        fit_button: QToolButton
        fixed_via_add_button: QPushButton
        fixed_via_rows_layout: QVBoxLayout
        fixed_vias_label_widget: QWidget | None
        fixed_vias_widget: QWidget
        geometry_filters_group: QGroupBox
        heuristic_analysis_window_scale_spin: QDoubleSpinBox
        heuristic_background_sigma_spin: QDoubleSpinBox
        heuristic_border_balance_scale_spin: QDoubleSpinBox
        heuristic_border_penalty_spin: QDoubleSpinBox
        heuristic_contrast_score_max_spin: QDoubleSpinBox
        heuristic_contrast_score_min_spin: QDoubleSpinBox
        heuristic_edge_quality_floor_spin: QDoubleSpinBox
        heuristic_edge_snr_score_max_spin: QDoubleSpinBox
        heuristic_edge_snr_score_min_spin: QDoubleSpinBox
        heuristic_line_penalty_spin: QDoubleSpinBox
        heuristic_local_binarize_percentile_spin: QDoubleSpinBox
        heuristic_max_center_drift_ratio_spin: QDoubleSpinBox
        heuristic_max_elongation_spin: QDoubleSpinBox
        heuristic_max_line_coherence_spin: QDoubleSpinBox
        heuristic_min_abs_peak_spin: QDoubleSpinBox
        heuristic_min_center_brightness_spin: QDoubleSpinBox
        heuristic_min_center_contrast_spin: QDoubleSpinBox
        heuristic_min_circularity_spin: QDoubleSpinBox
        heuristic_min_compactness_spin: QDoubleSpinBox
        heuristic_min_edge_sharpness_spin: QDoubleSpinBox
        heuristic_min_peak_prominence_spin: QDoubleSpinBox
        heuristic_prominence_score_max_spin: QDoubleSpinBox
        heuristic_prominence_score_min_spin: QDoubleSpinBox
        heuristic_seed_percentile_spin: QDoubleSpinBox
        heuristic_size_tolerance_fixed_spin: QDoubleSpinBox
        heuristic_size_tolerance_range_spin: QDoubleSpinBox
        heuristic_use_bilateral_checkbox: QCheckBox
        heuristic_w_balance_spin: QDoubleSpinBox
        heuristic_w_border_spin: QDoubleSpinBox
        heuristic_w_compact_spin: QDoubleSpinBox
        heuristic_w_contrast_spin: QDoubleSpinBox
        heuristic_w_line_spin: QDoubleSpinBox
        heuristic_w_prominence_spin: QDoubleSpinBox
        heuristic_w_round_spin: QDoubleSpinBox
        heuristic_w_size_spin: QDoubleSpinBox
        image_list: FramePathListView
        image_only_list: QListWidget
        input_dir_edit: QLineEdit
        max_area_label_widget: QWidget | None
        max_area_spin: QDoubleSpinBox
        max_aspect_ratio_label_widget: QWidget | None
        max_aspect_ratio_spin: QDoubleSpinBox
        max_bbox_height_label_widget: QWidget | None
        max_bbox_height_spin: QSpinBox
        max_bbox_width_label_widget: QWidget | None
        max_bbox_width_spin: QSpinBox
        max_hierarchy_depth_label_widget: QWidget | None
        max_hierarchy_depth_spin: QSpinBox
        max_hole_area_ratio_label_widget: QWidget | None
        max_hole_area_ratio_spin: QDoubleSpinBox
        max_perimeter_label_widget: QWidget | None
        max_perimeter_spin: QDoubleSpinBox
        max_via_height_label_widget: QWidget | None
        max_via_height_spin: QSpinBox
        max_via_width_label_widget: QWidget | None
        max_via_width_spin: QSpinBox
        metal_adaptive_block_spin: QSpinBox
        metal_adaptive_c_spin: QDoubleSpinBox
        metal_adaptive_method_combo: QComboBox
        metal_allowed_angles_combo: QComboBox
        metal_angle_tolerance_spin: QDoubleSpinBox | QSpinBox
        metal_approximation_checkbox: QCheckBox
        metal_auto_contrast_step_label_widget: QWidget | None
        metal_auto_contrast_step_spin: QDoubleSpinBox
        metal_auto_directional_gap_bridge_label_widget: QWidget | None
        metal_auto_directional_gap_bridge_spin: QSpinBox
        metal_auto_directional_gap_min_source_label_widget: QWidget | None
        metal_auto_directional_gap_min_source_spin: QDoubleSpinBox
        metal_auto_source_contrast_step_label_widget: QWidget | None
        metal_auto_source_contrast_step_spin: QDoubleSpinBox
        metal_border_handling_combo: QComboBox
        metal_boundary_background_label_widget: QWidget | None
        metal_boundary_background_spin: QDoubleSpinBox
        metal_boundary_relief_label_widget: QWidget | None
        metal_boundary_relief_spin: QDoubleSpinBox
        metal_contour_smooth_spin: QDoubleSpinBox | QSpinBox
        metal_debug_visual_combo: QComboBox
        metal_epsilon_spin: QDoubleSpinBox
        metal_gap_bridge_spin: QSpinBox
        metal_gc_iterations_label_widget: QWidget | None
        metal_gc_iterations_spin: QSpinBox
        metal_gradient_3d_button: QPushButton
        metal_hierarchy_combo: QComboBox
        metal_max_area_spin: QDoubleSpinBox
        metal_max_perimeter_spin: QDoubleSpinBox
        metal_max_width_caption: QLabel
        metal_max_width_spin: QDoubleSpinBox
        metal_min_angle_spin: QDoubleSpinBox
        metal_min_area_spin: QDoubleSpinBox
        metal_min_contrast_slider: QSlider
        metal_min_hole_source_contrast_fraction_spin: QDoubleSpinBox
        metal_min_hole_source_contrast_spin: QDoubleSpinBox
        metal_min_length_spin: QDoubleSpinBox
        metal_min_object_rim_area_fraction_spin: QDoubleSpinBox
        metal_min_object_rim_contrast_spin: QDoubleSpinBox
        metal_min_object_source_contrast_spin: QDoubleSpinBox
        metal_min_perimeter_spin: QDoubleSpinBox
        metal_min_points_spin: QSpinBox
        metal_min_width_caption: QLabel
        metal_min_width_spin: QDoubleSpinBox
        metal_morph_close_spin: QDoubleSpinBox | QSpinBox
        metal_morph_open_spin: QDoubleSpinBox | QSpinBox
        metal_noise_suppression_slider: QSlider
        metal_overlay_opacity_spin: QDoubleSpinBox
        metal_preset_combo: QComboBox
        metal_preview_mask_button: QPushButton
        metal_recon_erode_label_widget: QWidget | None
        metal_recon_erode_spin: QSpinBox
        metal_reset_params_button: QPushButton
        metal_rw_beta_label_widget: QWidget | None
        metal_rw_beta_spin: QDoubleSpinBox
        metal_rw_iterations_label_widget: QWidget | None
        metal_rw_iterations_spin: QSpinBox
        metal_segmentation_strategy_combo: QComboBox
        metal_segmentation_strategy_label_widget: QWidget | None
        metal_show_border_checkbox: QCheckBox
        metal_show_conductors_checkbox: QCheckBox
        metal_show_mask_checkbox: QCheckBox
        metal_show_rejected_checkbox: QCheckBox
        metal_show_suspicious_checkbox: QCheckBox
        metal_speckle_removal_spin: QSpinBox
        metal_straightness_spin: QDoubleSpinBox | QSpinBox
        metal_t_junction_checkbox: QCheckBox
        metal_validity_checkbox: QCheckBox
        metal_ws_core_margin_spin: QDoubleSpinBox
        metal_ws_groove_margin_spin: QDoubleSpinBox
        metal_ws_rim_probe_spin: QSpinBox
        metal_ws_seed_speckle_spin: QSpinBox
        metal_ws_smoothing_spin: QDoubleSpinBox
        metal_ws_valley_depth_spin: QDoubleSpinBox
        metal_ws_valley_span_spin: QSpinBox
        min_area_label_widget: QWidget | None
        min_area_spin: QDoubleSpinBox
        min_aspect_ratio_label_widget: QWidget | None
        min_aspect_ratio_spin: QDoubleSpinBox
        min_bbox_height_label_widget: QWidget | None
        min_bbox_height_spin: QSpinBox
        min_bbox_width_label_widget: QWidget | None
        min_bbox_width_spin: QSpinBox
        min_extent_label_widget: QWidget | None
        min_extent_spin: QDoubleSpinBox
        min_hierarchy_depth_label_widget: QWidget | None
        min_hierarchy_depth_spin: QSpinBox
        min_inner_hole_area_label_widget: QWidget | None
        min_inner_hole_area_spin: QDoubleSpinBox
        min_perimeter_label_widget: QWidget | None
        min_perimeter_spin: QDoubleSpinBox
        min_point_count_label_widget: QWidget | None
        min_points_spin: QSpinBox
        min_polygon_angle_spin: QDoubleSpinBox
        min_polygon_width_label_widget: QWidget | None
        min_polygon_width_spin: QDoubleSpinBox
        min_solidity_label_widget: QWidget | None
        min_solidity_spin: QDoubleSpinBox
        min_via_height_label_widget: QWidget | None
        min_via_height_spin: QSpinBox
        min_via_width_label_widget: QWidget | None
        min_via_width_spin: QSpinBox
        operation_tree: QTreeWidget
        output_dir_edit: QLineEdit
        parameters_form: QFormLayout
        pipeline_help_after_image: QLabel
        pipeline_help_before_image: QLabel
        pipeline_help_summary: QLabel
        pipeline_help_use: QLabel
        pipeline_list: PipelineListWidget
        polygon_editor: PolygonEditorView
        polygon_mode_combo: QComboBox
        redo_button: QToolButton
        reset_bright_via_button: QPushButton
        reset_via_search_button: QPushButton
        reset_via_search_label_widget: QWidget | None
        retrieval_mode_combo: QComboBox
        retrieval_mode_label_widget: QWidget | None
        ruler_status_label: QLabel
        selectionStatusChanged: pyqtBoundSignal
        topology_group: QGroupBox
        trace_mode_combo: QComboBox
        undo_button: QToolButton
        vector_only_list: QListWidget
        via_black_range_checkbox: QCheckBox
        via_black_range_label_widget: QWidget | None
        via_black_range_max_spin: QSpinBox
        via_black_range_min_spin: QSpinBox
        via_black_range_widget: QWidget
        via_group: QGroupBox
        via_height_range_widget: QWidget
        via_height_spin: QSpinBox
        via_heuristic_polarity_combo: QComboBox
        via_min_contrast_label_widget: QWidget | None
        via_min_contrast_spin: QDoubleSpinBox
        via_min_edge_coverage_label_widget: QWidget | None
        via_min_edge_coverage_spin: QDoubleSpinBox
        via_min_score_label_widget: QWidget | None
        via_min_score_spin: QDoubleSpinBox
        via_output_diameter_label_widget: QWidget | None
        via_output_diameter_spin: QSpinBox
        via_preset_combo: QComboBox
        via_preset_label_widget: QWidget | None
        via_preset_widget: QWidget
        via_range_checkboxes_label_widget: QWidget | None
        via_range_checkboxes_widget: QWidget
        via_roundness_label_widget: QWidget | None
        via_roundness_spin: QDoubleSpinBox
        via_search_mode_combo: QComboBox
        via_search_mode_label_widget: QWidget | None
        via_spot_line_suppression_label_widget: QWidget | None
        via_spot_line_suppression_spin: QDoubleSpinBox
        via_template_min_score_label_widget: QWidget | None
        via_template_min_score_spin: QDoubleSpinBox
        via_template_nms_distance_spin: QSpinBox
        via_templates_label_widget: QWidget | None
        via_templates_widget: QWidget
        via_white_range_checkbox: QCheckBox
        via_white_range_label_widget: QWidget | None
        via_white_range_max_spin: QSpinBox
        via_white_range_min_spin: QSpinBox
        via_white_range_widget: QWidget
        via_width_range_widget: QWidget
        via_width_spin: QSpinBox
        zoom_in_button: QToolButton
        zoom_out_button: QToolButton

        def _abort_in_flight_interactive_processing(self, *, preview: bool, prepared: bool) -> None: ...

        def _add_fixed_via_row(self, *_args, width: int = 1, height: int = 1) -> None: ...

        def _add_pipeline_step(self) -> None: ...

        def _all_operation_names(self) -> list[str]: ...

        def _append_log(self, message: str) -> None: ...

        def _apply_conductor_display_visibility(self, state=None) -> None: ...

        def _apply_metal_preset_payload(self, payload: dict[str, object]) -> None: ...

        def _apply_via_preset_payload(self, payload: dict[str, object]) -> None: ...

        def _auto_apply_pipeline(self) -> None: ...

        def _auto_apply_recognition_settings(self) -> None: ...

        def _built_in_metal_presets(self) -> dict[str, dict[str, object]]: ...

        def _clear_fixed_via_rows(self) -> None: ...

        def _current_contour_settings(self) -> ContourExtractionSettings: ...

        def _current_via_preset_payload(self) -> dict[str, object]: ...

        def _dialog_start_directory_from_line_edit(self, line_edit, fallback: str | Path | None = None) -> str: ...

        def _find_matching_cif_path(self, image_path: str) -> str | None: ...

        def _find_operation_tree_item(self, operation_name: str) -> QTreeWidgetItem | None: ...

        def _fixed_via_pairs(self) -> list[tuple[int, int]]: ...

        def _image_list_path_from_proxy_index(self, proxy_index: QModelIndex) -> str | None: ...

        def _is_extraction_mode_enabled(self) -> bool: ...

        def _load_cif_overlay_polygons(self, image_path: str) -> list[PolygonData]: ...

        def _mark_thumbnail_grid_rebuild_pending(self) -> None: ...

        def _normalize_via_template_images(self, payload: list[object]) -> list[np.ndarray]: ...

        def _on_auto_tune_error(self, request_id: int, message: str) -> None: ...

        def _on_auto_tune_finished(self, request_id: int) -> None: ...

        def _on_auto_tune_result(self, request_id: int, result: AutoTuneResult) -> None: ...

        def _on_extraction_settings_changed(self, *_args) -> None: ...

        def _operation_help_entry(self, operation_name: str) -> tuple[str, str]: ...

        def _paint_image_row_item(self, item: QListWidgetItem, image_path: str, *, show_text: bool = True) -> None: ...

        def _pipeline_parameter_tooltip(self, operation_name: str, parameter_name: str) -> str: ...

        def _populate_pipeline_list(self) -> None: ...

        def _queue_prepared_image_update(self, image_path: str, source_image) -> None: ...

        def _rebuild_thumbnail_grid(self) -> None: ...

        def _rebuild_vector_list(self) -> None: ...

        def _refresh_busy_indicator(self) -> None: ...

        def _refresh_gradient_overlay(self) -> None: ...

        def _refresh_metal_preset_combo(self) -> None: ...

        def _refresh_via_preset_combo(self) -> None: ...

        def _refresh_via_template_list(self) -> None: ...

        def _register_no_wheel_value_widget(self, widget) -> None: ...

        def _register_spinbox(self, spinbox: QAbstractSpinBox) -> None: ...

        def _render_pipeline_parameters(self, row: int) -> None: ...

        def _save_persisted_paths(self) -> None: ...

        def _save_user_metal_presets(self) -> None: ...

        def _save_user_via_presets(self) -> None: ...

        def _schedule_frame_switch_profile_until_interactive(
            self,
            image_path: str,
            *,
            cif_path: str | None,
            polygon_count: int,
            vectors_only: bool = False,
            failed: bool = False,
        ) -> None: ...

        def _schedule_thumbnail_grid_rebuild(self, *, force: bool = False) -> None: ...

        def _selected_available_operation_name(self) -> str | None: ...

        def _set_extraction_settings(self, settings: ContourExtractionSettings) -> None: ...

        def _set_field_tooltip(self, label_widget: QWidget | None, field_widget: QWidget, help_key: str) -> None: ...

        def _show_manual_tool_postprocess_dialog(self) -> None: ...

        def _start_frame_switch_profile(self, image_path: str) -> FrameSwitchProfile | None: ...

        def _state_has_recognition_result(self, state) -> bool: ...

        def _store_active_extraction_profile_settings(self) -> None: ...

        def _sync_after_cif_index_changed(self) -> None: ...

        def _sync_editor_contact_minimum_distance(self) -> None: ...

        def _sync_recognition_stack_visibility(self) -> None: ...

        def _tr(self, key: str, default: str = "", **kwargs: object) -> str: ...

        def _try_leave_current_frame(self) -> bool: ...

        def _update_bright_via_diameter_controls_state(self) -> None: ...

        def _update_extraction_profile_controls_state(self) -> None: ...

        def _update_pipeline_help_preview(self, operation_name: str | None) -> None: ...

        def _update_thumbnail_grid_selection(self, *, scroll_to_selection: bool | None = None) -> None: ...

        def _update_via_size_controls_state(self) -> None: ...

        def _update_via_threshold_controls_state(self) -> None: ...

        def _via_debug_inspection_enabled(self) -> bool: ...

        def append_images(self, paths: list[str], *, select_first_new: bool = True) -> None: ...

        def get_pipeline(self) -> dict: ...

        def get_polygons(self) -> list[PolygonData]: ...

        def load_image(
            self, path: str, *, load_vectors: bool | None = None, preserve_editor_view_position: bool = False
        ) -> None: ...

        def load_images(
            self,
            paths: list[str],
            *,
            preferred_current_image_path: str | None = None,
            image_list_mode: str | None = None,
        ) -> None: ...

        def process_current_image(self, *_args, debounced: bool = False) -> None: ...

        def set_pipeline(self, config: dict) -> None: ...
