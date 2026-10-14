"""retranslate_ui section 1."""
from __future__ import annotations
from PyQt6.QtCore import QSignalBlocker
from .....vision.metal_recovery.strategy_registry import strategy_spec
from contour.ui.bright_via_i18n import retranslate_bright_via_panel
from contour.ui.i18n_content import PIPELINE_CONTROL_TOOLTIPS, _localized_text
from contour.ui.metal_debug_i18n import retranslate_metal_debug_visual_combo
from contour.ui.metal_strategy_i18n import strategy_name
from contour.ui.retranslate_parts.retranslate_strategy_parameter_pages import (
    _retranslate_strategy_parameter_pages,
    _set_form_label,
)

def _retranslate_ui_section_1(self):
    selected_operation = self._selected_available_operation_name()
    selected_pipeline_row = self.pipeline_list.currentRow() if hasattr(self, "pipeline_list") else -1

    if hasattr(self.path_group, "setTitle"):
        self.path_group.setTitle(self._tr("path_panel_title"))
    self.input_dir_label.setText(self._tr("input_directory_label"))
    self.cif_dir_label.setText(self._tr("cif_overlay_directory_label"))
    self.output_dir_label.setText(self._tr("output_directory_label"))
    self.dataset_dir_label.setText(self._tr("dataset_directory_label"))
    for button, accessible_name in (
        (self.browse_input_button, self._tr("browse_input_button")),
        (self.browse_cif_button, self._tr("browse_cif_button")),
        (self.browse_output_button, self._tr("browse_output_button")),
        (self.browse_dataset_button, self._tr("browse_dataset_button")),
        (self.refresh_button, self._tr("refresh_files_button")),
        (self.pick_input_files_button, self._tr("pick_input_files_button")),
        (self.merge_cif_files_button, self._tr("merge_cif_files_button")),
    ):
        button.setText("")
        button.setAccessibleName(accessible_name)
    for widget, tooltip_key in (
        (self.input_dir_label, "input_dir"),
        (self.input_dir_edit, "input_dir"),
        (self.cif_dir_label, "cif_dir"),
        (self.cif_dir_edit, "cif_dir"),
        (self.output_dir_label, "output_dir"),
        (self.output_dir_edit, "output_dir"),
        (self.dataset_dir_label, "dataset_dir"),
        (self.dataset_dir_edit, "dataset_dir"),
        (self.browse_input_button, "browse_input"),
        (self.browse_cif_button, "browse_cif"),
        (self.browse_output_button, "browse_output"),
        (self.browse_dataset_button, "browse_dataset"),
        (self.refresh_button, "refresh_files"),
        (self.pick_input_files_button, "pick_input_images"),
        (self.merge_cif_files_button, "merge_cif_files"),
    ):
        self._set_common_tooltip(widget, tooltip_key)
    for tab, key in (
        (getattr(self, "paths_tab", None), "tab_paths"),
        (getattr(self, "pipeline_tab", None), "tab_pipeline"),
        (getattr(self, "extraction_tab", None), "tab_extraction"),
        (getattr(self, "display_tab", None), "tab_display"),
    ):
        index = self.control_tabs.indexOf(tab) if tab is not None else -1
        if index >= 0:
            self.control_tabs.setTabText(index, self._tr(key))
    if hasattr(self, "right_tabs"):
        if self.right_tabs.count() > 0:
            self.right_tabs.setTabText(0, self._tr("tab_files"))
        if self.right_tabs.count() > 1:
            self.right_tabs.setTabText(1, "Питомец" if self._ui_language == "ru" else "Pet")

    if hasattr(self, "thumbnail_grid_label"):
        self.thumbnail_grid_label.setText("Матрица кадров" if self._ui_language == "ru" else "Frame thumbnails")
    if hasattr(self, "asset_view_tabs"):
        self.asset_view_tabs.setTabText(0, self._tr("asset_tab_all", "Все" if self._ui_language == "ru" else "All"))
        self.asset_view_tabs.setTabText(
            1,
            self._tr("asset_tab_image_vector", "Изображение+вектор" if self._ui_language == "ru" else "Image+Vector"),
        )
        self.asset_view_tabs.setTabText(
            2,
            self._tr("asset_tab_image_only", "Только изображения" if self._ui_language == "ru" else "Image only"),
        )
        self.asset_view_tabs.setTabText(
            3,
            self._tr("asset_tab_vector_only", "Только векторы" if self._ui_language == "ru" else "Vector only"),
        )
    if hasattr(self, "sidebar_list_mode_combo"):
        with QSignalBlocker(self.sidebar_list_mode_combo):
            self.sidebar_list_mode_combo.setItemText(0, self._tr("images_label"))
            self.sidebar_list_mode_combo.setItemText(1, self._tr("vectors_tab_label"))
        self._set_common_tooltip(self.sidebar_list_mode_combo, "sidebar_list_mode")
    if hasattr(self, "reload_cif_selected_button"):
        self.reload_cif_selected_button.setText(self._tr("reload_selected_cifs_button"))
        self.reload_cif_for_frames_button.setText(self._tr("reload_cifs_for_frames_button"))
        self._set_common_tooltip(self.reload_cif_selected_button, "reload_selected_cif_overlays")
        self._set_common_tooltip(self.reload_cif_for_frames_button, "reload_cif_for_selected_frames")

    if hasattr(self.run_group, "setTitle"):
        self.run_group.setTitle(self._tr("run_group_title"))
    if hasattr(self, "extra_layers_group"):
        self.extra_layers_group.setTitle(
            self._tr(
                "extra_layers_group_title",
                "Дополнительные слои" if self._ui_language == "ru" else "Additional layers",
            )
        )
    if hasattr(self, "vector_geom_group"):
        self.vector_geom_group.setTitle(
            "Геометрия векторов при переходе между кадрами"
            if self._ui_language == "ru"
            else "Vector geometry on frame transitions"
        )
        self.vector_geom_clip_checkbox.setText(
            "Обрезать по границе кадра и удалить внешние объекты"
            if self._ui_language == "ru"
            else "Clip to frame and remove outside objects"
        )
        if getattr(self, "vector_geom_min_outer_label_widget", None) is not None:
            self.vector_geom_min_outer_label_widget.setText(
                "Минимальная площадь внешнего объекта, px²"
                if self._ui_language == "ru"
                else "Minimum outer object area, px²"
            )
        if getattr(self, "vector_geom_min_hole_label_widget", None) is not None:
            self.vector_geom_min_hole_label_widget.setText(
                "Минимальная площадь отверстия для заливки, px²"
                if self._ui_language == "ru"
                else "Minimum hole area to fill, px²"
            )
        self.vector_geom_merge_checkbox.setText(
            "Объединять пересекающиеся полигоны после перемещения"
            if self._ui_language == "ru"
            else "Merge overlapping polygons after moves"
        )
        if getattr(self, "vector_geom_spike_angle_label_widget", None) is not None:
            self.vector_geom_spike_angle_label_widget.setText(
                "Минимальный угол острого выброса, °" if self._ui_language == "ru" else "Minimum spike angle, °"
            )
        self.vector_geom_drop_triangle_checkbox.setText(
            "Удалять внешние треугольники из 3 вершин как артефакты"
            if self._ui_language == "ru"
            else "Drop 3-vertex outer triangles as artifacts"
        )
    for button, accessible_name in (
        (self.process_current_button, self._tr("process_current_button")),
        (self.batch_button, self._tr("start_batch_button")),
        (self.stop_batch_button, self._tr("stop_batch_button")),
    ):
        button.setText("")
        button.setAccessibleName(accessible_name)
    self.antialias_opened_cif_button.setText(self._tr("antialias_opened_cif_button"))
    self.antialias_opened_cif_button.setAccessibleName(self._tr("antialias_opened_cif_button"))
    if hasattr(self, "fix_internal_contours_button"):
        self.fix_internal_contours_button.setText(self._tr("fix_internal_contours_button"))
        self.fix_internal_contours_button.setAccessibleName(self._tr("fix_internal_contours_button"))
    self.save_current_button.setText(self._tr("save_current_button"))
    self.export_dataset_button.setText(self._tr("export_dataset_button"))
    self.dataset_mode_checkbox.setText(self._tr("dataset_mode_checkbox"))
    for widget, tooltip_key in (
        (self.image_list, "image_list"),
        (self.vector_list, "vector_list_sidebar"),
        (self.process_current_button, "process_current"),
        (self.batch_button, "start_batch"),
        (self.stop_batch_button, "stop_batch"),
        (self.antialias_opened_cif_button, "antialias_all_vectors"),
        (self.save_current_button, "save_current"),
        (self.export_dataset_button, "export_dataset"),
        (self.dataset_mode_checkbox, "dataset_mode"),
    ):
        self._set_common_tooltip(widget, tooltip_key)

    self.available_filters_group.setTitle(
        self._tr(
            "available_filters_group_title",
            "Доступные фильтры" if self._ui_language == "ru" else "Available filters",
        )
    )
    self.pipeline_steps_group.setTitle(
        self._tr(
            "applied_filters_group_title", "Примененные фильтры" if self._ui_language == "ru" else "Applied filters"
        )
    )
    self.pipeline_help_group.setTitle(
        self._tr("pipeline_help_group_title", "Справка по фильтру" if self._ui_language == "ru" else "Filter help")
    )
    self.pipeline_help_before_title.setText("До" if self._ui_language == "ru" else "Before")
    self.pipeline_help_after_title.setText("После" if self._ui_language == "ru" else "After")
    self.save_pipeline_button.setText(self._tr("save_json_button"))
    self.load_pipeline_button.setText(self._tr("load_json_button"))
    self.show_applied_filters_checkbox.setText(
        self._tr(
            "show_applied_filters_checkbox",
            "Показывать примененные фильтры" if self._ui_language == "ru" else "Show applied filters",
        )
    )
    for widget, tooltip_key in (
        (self.save_pipeline_button, "save_json_button"),
        (self.load_pipeline_button, "load_json_button"),
    ):
        tooltip = _localized_text(PIPELINE_CONTROL_TOOLTIPS, tooltip_key, self._ui_language)
        widget.setToolTip(tooltip)
        widget.setStatusTip(tooltip)
    self.parameters_group.setTitle(self._tr("step_parameters_group"))

    self.contour_group.setTitle(self._tr("contour_extraction_group"))
    self.basic_filters_group.setTitle(
        self._tr("basic_filters_group_title", "Базовые фильтры" if self._ui_language == "ru" else "Basic filters")
    )
    self.geometry_filters_group.setTitle(
        self._tr("geometry_filters_group_title", "Геометрия" if self._ui_language == "ru" else "Geometry")
    )
    self.via_group.setTitle(
        self._tr(
            "via_constraints_group_title",
            "Ограничения контактов" if self._ui_language == "ru" else "Contact constraints",
        )
    )
    self.topology_group.setTitle(
        self._tr("topology_group_title", "Иерархия и отверстия" if self._ui_language == "ru" else "Hierarchy and holes")
    )
    if hasattr(self, "recognition_mode_combo"):
        self.recognition_mode_combo.setItemText(0, "Отключено" if self._ui_language == "ru" else "Disabled")
        self.recognition_mode_combo.setItemText(1, "Проводники" if self._ui_language == "ru" else "Conductors")
        self.recognition_mode_combo.setItemText(2, "Контакты" if self._ui_language == "ru" else "Contacts")
    if hasattr(self, "recognition_mode_label"):
        self.recognition_mode_label.setText("Распознавание" if self._ui_language == "ru" else "Recognition")
    if hasattr(self, "metal_segmentation_strategy_combo"):
        combo = self.metal_segmentation_strategy_combo
        for index in range(combo.count()):
            data = str(combo.itemData(index) or "")
            combo.setItemText(index, strategy_name(strategy_spec(data), self._ui_language))
    if getattr(self, "metal_segmentation_strategy_label_widget", None) is not None:
        self.metal_segmentation_strategy_label_widget.setText(
            "Алгоритм распознавания" if self._ui_language == "ru" else "Recognition algorithm"
        )
    if getattr(self, "metal_auto_contrast_step_label_widget", None) is not None:
        self.metal_auto_contrast_step_label_widget.setText(
            "Шаг контраста Auto" if self._ui_language == "ru" else "Auto contrast step"
        )
    if getattr(self, "metal_auto_contrast_step_spin", None) is not None:
        self.metal_auto_contrast_step_spin.setSpecialValueText("Выкл." if self._ui_language == "ru" else "Off")
    if getattr(self, "metal_auto_source_contrast_step_label_widget", None) is not None:
        self.metal_auto_source_contrast_step_label_widget.setText(
            "Шаг фильтра объектов Auto" if self._ui_language == "ru" else "Auto object-filter step"
        )
    if getattr(self, "metal_auto_source_contrast_step_spin", None) is not None:
        self.metal_auto_source_contrast_step_spin.setSpecialValueText("Выкл." if self._ui_language == "ru" else "Off")
    if getattr(self, "metal_auto_directional_gap_bridge_label_widget", None) is not None:
        self.metal_auto_directional_gap_bridge_label_widget.setText(
            "Направленная сшивка Auto, пикс." if self._ui_language == "ru" else "Auto directional gap, px"
        )
    if getattr(self, "metal_auto_directional_gap_bridge_spin", None) is not None:
        self.metal_auto_directional_gap_bridge_spin.setSpecialValueText("Выкл." if self._ui_language == "ru" else "Off")
    if getattr(self, "metal_auto_directional_gap_min_source_label_widget", None) is not None:
        self.metal_auto_directional_gap_min_source_label_widget.setText(
            "Мин. яркость направленной сшивки" if self._ui_language == "ru" else "Directional gap source minimum"
        )
    if getattr(self, "metal_min_object_rim_contrast_label_widget", None) is not None:
        self.metal_min_object_rim_contrast_label_widget.setText(
            "\u041c\u0438\u043d. \u043a\u043e\u043d\u0442\u0440\u0430\u0441\u0442 \u044f\u0440\u043a\u043e\u0439 \u043a\u0440\u043e\u043c\u043a\u0438"
            if self._ui_language == "ru"
            else "Bright rim minimum contrast"
        )
    if getattr(self, "metal_min_object_rim_area_fraction_label_widget", None) is not None:
        self.metal_min_object_rim_area_fraction_label_widget.setText(
            "\u041c\u0438\u043d. \u0434\u043e\u043b\u044f \u043f\u043b\u043e\u0449\u0430\u0434\u0438 \u0434\u043b\u044f \u043a\u0440\u043e\u043c\u043e\u0447\u043d\u043e\u0433\u043e \u0444\u0438\u043b\u044c\u0442\u0440\u0430"
            if self._ui_language == "ru"
            else "Rim-filter minimum area fraction"
        )
    if getattr(self, "metal_ws_smoothing_label_widget", None) is not None:
        self.metal_ws_smoothing_label_widget.setText(
            "Сглаживание водораздела, σ" if self._ui_language == "ru" else "Watershed smoothing, σ"
        )
    if getattr(self, "metal_ws_core_margin_label_widget", None) is not None:
        self.metal_ws_core_margin_label_widget.setText(
            "Отступ ядер металла" if self._ui_language == "ru" else "Metal core margin"
        )
    if getattr(self, "metal_ws_groove_margin_label_widget", None) is not None:
        self.metal_ws_groove_margin_label_widget.setText(
            "Отступ затравок зазора" if self._ui_language == "ru" else "Gap seed margin"
        )
    if getattr(self, "metal_ws_rim_probe_label_widget", None) is not None:
        self.metal_ws_rim_probe_label_widget.setText(
            "Радиус кольца кромки, px" if self._ui_language == "ru" else "Rim ring radius, px"
        )
    if getattr(self, "metal_ws_seed_speckle_label_widget", None) is not None:
        self.metal_ws_seed_speckle_label_widget.setText(
            "Очистка затравок, px" if self._ui_language == "ru" else "Seed speckle cleanup, px"
        )
    if getattr(self, "metal_ws_valley_span_label_widget", None) is not None:
        self.metal_ws_valley_span_label_widget.setText(
            "Ширина узкого зазора, px" if self._ui_language == "ru" else "Narrow gap width, px"
        )
    if getattr(self, "metal_ws_valley_depth_label_widget", None) is not None:
        self.metal_ws_valley_depth_label_widget.setText(
            "Глубина узкого зазора" if self._ui_language == "ru" else "Narrow gap depth"
        )
    if getattr(self, "metal_rw_beta_label_widget", None) is not None:
        self.metal_rw_beta_label_widget.setText(
            "Жёсткость границ, β" if self._ui_language == "ru" else "Boundary stiffness, β"
        )
    if getattr(self, "metal_rw_iterations_label_widget", None) is not None:
        self.metal_rw_iterations_label_widget.setText("Итерации" if self._ui_language == "ru" else "Iterations")
    if getattr(self, "metal_gc_iterations_label_widget", None) is not None:
        self.metal_gc_iterations_label_widget.setText(
            "Итерации GrabCut" if self._ui_language == "ru" else "GrabCut iterations"
        )
    if getattr(self, "metal_recon_erode_label_widget", None) is not None:
        self.metal_recon_erode_label_widget.setText(
            "Эрозия ядер, px" if self._ui_language == "ru" else "Core erosion, px"
        )
    if getattr(self, "metal_boundary_relief_label_widget", None) is not None:
        self.metal_boundary_relief_label_widget.setText(
            "Высота рельефа границы" if self._ui_language == "ru" else "Edge relief height"
        )
    if getattr(self, "metal_boundary_background_label_widget", None) is not None:
        self.metal_boundary_background_label_widget.setText(
            "Масштаб фона, σ" if self._ui_language == "ru" else "Background scale, σ"
        )
    if getattr(self, "metal_advanced_group", None) is not None:
        self.metal_advanced_group.setTitle("Дополнительно" if self._ui_language == "ru" else "Advanced")
    if getattr(self, "metal_filter_group", None) is not None:
        self.metal_filter_group.setTitle(
            "Фильтрация распознанных" if self._ui_language == "ru" else "Recognized-object filters"
        )
    if getattr(self, "metal_recognition_params_group", None) is not None:
        self.metal_recognition_params_group.setTitle(
            "Параметры распознавания" if self._ui_language == "ru" else "Recognition parameters"
        )
    if getattr(self, "metal_watershed_group", None) is not None:
        self.metal_watershed_group.setTitle("Watershed")
    if getattr(self, "metal_random_walker_group", None) is not None:
        self.metal_random_walker_group.setTitle("Random Walker")
    if getattr(self, "metal_graph_cut_group", None) is not None:
        self.metal_graph_cut_group.setTitle("Graph Cut")
    if getattr(self, "metal_reconstruction_group", None) is not None:
        self.metal_reconstruction_group.setTitle("Reconstruction")
    if getattr(self, "metal_closed_boundary_group", None) is not None:
        self.metal_closed_boundary_group.setTitle(
            "Замкнутые границы" if self._ui_language == "ru" else "Closed boundary"
        )
    return index, selected_operation, selected_pipeline_row, tooltip, tooltip_key, widget
