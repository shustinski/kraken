"""Retranslate implementation for :class:`PolygonExtractionWidget`.

Split out of ``contour.widget`` during the production-ready refactor.
The function is bound as a method of the widget via attribute assignment, so
``self`` refers to the original widget and all attribute access is unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import QSignalBlocker

from .i18n_content import PIPELINE_CONTROL_TOOLTIPS, _localized_text

if TYPE_CHECKING:
    from contour.widget import PolygonExtractionWidget


def retranslate_ui(self: PolygonExtractionWidget) -> None:
    if not hasattr(self, "control_tabs"):
        return
    selected_operation = self._selected_available_operation_name()
    selected_pipeline_row = self.pipeline_list.currentRow() if hasattr(self, "pipeline_list") else -1

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
    self.pick_input_files_button.setToolTip("Выбрать кадры базового слоя")
    self.pick_input_files_button.setStatusTip(self.pick_input_files_button.toolTip())
    self.browse_input_button.setToolTip("Загрузить папку базового слоя")
    self.browse_input_button.setStatusTip(self.browse_input_button.toolTip())

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
                "Минимальный угол острого выброса, °"
                if self._ui_language == "ru"
                else "Minimum spike angle, °"
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
    self.save_current_button.setText(self._tr("save_current_button"))
    self.export_dataset_button.setText(self._tr("export_dataset_button"))
    self.dataset_mode_checkbox.setText(self._tr("dataset_mode_checkbox"))
    for widget, tooltip_key in (
        (self.image_list, "image_list"),
        (self.vector_list, "vector_list_sidebar"),
        (self.process_current_button, "process_current"),
        (self.batch_button, "start_batch"),
        (self.stop_batch_button, "stop_batch"),
        (self.save_current_button, "save_current"),
        (self.export_dataset_button, "export_dataset"),
        (self.dataset_mode_checkbox, "dataset_mode"),
    ):
        self._set_common_tooltip(widget, tooltip_key)

    self.available_filters_group.setTitle(
        self._tr(
            "available_filters_group_title", "Фильтры pipeline" if self._ui_language == "ru" else "Pipeline filters"
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
    self.apply_pipeline_preset_button.setText(
        self._tr(
            "apply_pipeline_preset_button",
            "Применить пресет фильтров" if self._ui_language == "ru" else "Apply filter preset",
        )
    )
    self.auto_tune_button.setText(
        self._tr(
            "auto_tune_button",
            "Автоподбор по рисунку" if self._ui_language == "ru" else "Auto-fit from drawing",
        )
    )
    self.auto_tune_button.setToolTip(
        self._tr(
            "auto_tune_button_tooltip",
            "Использовать текущие нарисованные полигоны как эталон"
            if self._ui_language == "ru"
            else "Use the currently drawn polygons as the fitting target",
        )
    )
    for widget, tooltip_key in (
        (self.save_pipeline_button, "save_json_button"),
        (self.load_pipeline_button, "load_json_button"),
        (self.auto_tune_button, "auto_tune_button"),
    ):
        tooltip = _localized_text(PIPELINE_CONTROL_TOOLTIPS, tooltip_key, self._ui_language)
        widget.setToolTip(tooltip)
        widget.setStatusTip(tooltip)
    self.pipeline_preset_combo.setToolTip(
        self._tr(
            "pipeline_preset_selector_tooltip",
            "Выберите готовый набор фильтров для типового сценария."
            if self._ui_language == "ru"
            else "Choose a ready filter chain for a typical scenario.",
        )
    )
    self.pipeline_preset_combo.setStatusTip(self.pipeline_preset_combo.toolTip())
    self.apply_pipeline_preset_button.setToolTip(self.pipeline_preset_combo.toolTip())
    self.apply_pipeline_preset_button.setStatusTip(self.pipeline_preset_combo.toolTip())
    self.parameters_group.setTitle(self._tr("step_parameters_group"))

    self.contour_group.setTitle(self._tr("contour_extraction_group"))
    self.basic_filters_group.setTitle(
        self._tr("basic_filters_group_title", "Базовые фильтры" if self._ui_language == "ru" else "Basic filters")
    )
    self.geometry_filters_group.setTitle(
        self._tr("geometry_filters_group_title", "Геометрия" if self._ui_language == "ru" else "Geometry")
    )
    self.via_group.setTitle(
        self._tr("via_constraints_group_title", "Ограничения via" if self._ui_language == "ru" else "Via constraints")
    )
    self.topology_group.setTitle(
        self._tr("topology_group_title", "Иерархия и отверстия" if self._ui_language == "ru" else "Hierarchy and holes")
    )
    if hasattr(self, "recognition_mode_combo"):
        self.recognition_mode_combo.setItemText(0, self._tr("recognition_mode_disabled"))
        self.recognition_mode_combo.setItemText(1, self._tr("recognition_mode_conductors"))
        self.recognition_mode_combo.setItemText(2, self._tr("extraction_profile_vias"))
    if self.retrieval_mode_label_widget is not None:
        self.retrieval_mode_label_widget.setText(self._tr("retrieval_mode_label"))
    if self.approximation_mode_label_widget is not None:
        self.approximation_mode_label_widget.setText(self._tr("approximation_mode_label"))
    if self.epsilon_label_widget is not None:
        self.epsilon_label_widget.setText(self._tr("epsilon_label"))
    if hasattr(self, "epsilon_left_label"):
        self.epsilon_left_label.setText(self._tr("epsilon_left_label"))
    if hasattr(self, "epsilon_right_label"):
        self.epsilon_right_label.setText(self._tr("epsilon_right_label"))
    if self.epsilon_mode_label_widget is not None:
        self.epsilon_mode_label_widget.setText(self._tr("epsilon_mode_label"))
    self.epsilon_relative_checkbox.setText(self._tr("epsilon_relative_checkbox"))
    if self.min_area_label_widget is not None:
        self.min_area_label_widget.setText(self._tr("area_range_label"))
    if self.min_perimeter_label_widget is not None:
        self.min_perimeter_label_widget.setText(
            self._tr("perimeter_range_label", "Диапазон периметра" if self._ui_language == "ru" else "Perimeter range")
        )
    if self.min_point_count_label_widget is not None:
        self.min_point_count_label_widget.setText(self._tr("min_point_count_label"))
    if getattr(self, "min_polygon_width_label_widget", None) is not None:
        self.min_polygon_width_label_widget.setText(self._tr("min_polygon_width_label"))
    if self.min_bbox_width_label_widget is not None:
        self.min_bbox_width_label_widget.setText(
            self._tr(
                "bbox_width_range_label",
                "Диапазон ширины bbox" if self._ui_language == "ru" else "BBox width range",
            )
        )
    if self.min_bbox_height_label_widget is not None:
        self.min_bbox_height_label_widget.setText(
            self._tr(
                "bbox_height_range_label",
                "Диапазон высоты bbox" if self._ui_language == "ru" else "BBox height range",
            )
        )
    if self.min_aspect_ratio_label_widget is not None:
        self.min_aspect_ratio_label_widget.setText(
            self._tr(
                "aspect_ratio_range_label",
                "Диапазон aspect ratio" if self._ui_language == "ru" else "Aspect ratio range",
            )
        )
    if self.border_handling_label_widget is not None:
        self.border_handling_label_widget.setText(self._tr("border_handling_label"))
    self.exclude_border_touching_checkbox.setText(
        self._tr("exclude_border_touching_checkbox_short", "Исключать" if self._ui_language == "ru" else "Exclude")
    )
    if self.min_solidity_label_widget is not None:
        self.min_solidity_label_widget.setText(self._tr("min_solidity_label"))
    if self.min_extent_label_widget is not None:
        self.min_extent_label_widget.setText(self._tr("min_extent_label"))
    if self.via_size_mode_label_widget is not None:
        self.via_size_mode_label_widget.setText(
            self._tr("via_size_mode_label", "Режим размеров via" if self._ui_language == "ru" else "Via size mode")
        )
    self.via_size_mode_combo.setItemText(
        0,
        self._tr("via_size_mode_range", "Диапазон" if self._ui_language == "ru" else "Range"),
    )
    self.via_size_mode_combo.setItemText(
        1,
        self._tr("via_size_mode_fixed", "Фиксированные значения" if self._ui_language == "ru" else "Fixed values"),
    )
    if getattr(self, "via_search_mode_label_widget", None) is not None:
        self.via_search_mode_label_widget.setText(
            self._tr("via_search_mode_label", "Режим поиска via" if self._ui_language == "ru" else "Via search mode")
        )
    if self.via_search_mode_combo.count() >= 2:
        self.via_search_mode_combo.setItemText(
            0,
            self._tr("via_search_mode_template", "По шаблону" if self._ui_language == "ru" else "Template"),
        )
        self.via_search_mode_combo.setItemText(
            1,
            self._tr("via_search_mode_heuristic", "Эвристический" if self._ui_language == "ru" else "Heuristic"),
        )
    if self.via_white_range_label_widget is not None:
        self.via_white_range_label_widget.setText(
            self._tr("via_white_range_label", "Диапазон белых" if self._ui_language == "ru" else "White range")
        )
    self.via_white_range_checkbox.setText("Вкл." if self._ui_language == "ru" else "Enabled")
    if self.via_black_range_label_widget is not None:
        self.via_black_range_label_widget.setText(
            self._tr("via_black_range_label", "Диапазон чёрных" if self._ui_language == "ru" else "Black range")
        )
    self.via_black_range_checkbox.setText("Вкл." if self._ui_language == "ru" else "Enabled")
    if getattr(self, "via_range_checkboxes_label_widget", None) is not None:
        self.via_range_checkboxes_label_widget.setText(
            self._tr("via_polarity_label", "Полярность" if self._ui_language == "ru" else "Polarity")
        )
    self.via_white_range_checkbox.setText(
        self._tr("via_white_range_method", "Диапазон белых" if self._ui_language == "ru" else "White range")
    )
    self.via_black_range_checkbox.setText(
        self._tr("via_black_range_method", "Диапазон черных" if self._ui_language == "ru" else "Black range")
    )
    if getattr(self, "via_min_score_label_widget", None) is not None:
        self.via_min_score_label_widget.setText(
            self._tr("via_min_score_label", "Мин. score" if self._ui_language == "ru" else "Min score")
        )
    if getattr(self, "via_min_contrast_label_widget", None) is not None:
        self.via_min_contrast_label_widget.setText(
            self._tr("via_min_contrast_label", "Мин. контраст" if self._ui_language == "ru" else "Min contrast")
        )
    if getattr(self, "via_min_edge_coverage_label_widget", None) is not None:
        self.via_min_edge_coverage_label_widget.setText(
            self._tr(
                "via_min_edge_coverage_label",
                "Мин. покрытие кромки" if self._ui_language == "ru" else "Min edge coverage",
            )
        )
    if self.via_spot_line_suppression_label_widget is not None:
        self.via_spot_line_suppression_label_widget.setText(
            self._tr(
                "via_spot_line_suppression_label",
                "\u0422\u043e\u0447\u043a\u0438: \u0434\u043e\u0440\u043e\u0436\u043a\u0438"
                if self._ui_language == "ru"
                else "Spots traces",
            )
        )
    if self.via_template_min_score_label_widget is not None:
        self.via_template_min_score_label_widget.setText(
            self._tr(
                "via_template_min_score_label",
                "Шаблон: совпадение" if self._ui_language == "ru" else "Template score",
            )
        )
    if self.via_templates_label_widget is not None:
        self.via_templates_label_widget.setText(
            self._tr("via_templates_label", "Шаблоны" if self._ui_language == "ru" else "Templates")
        )
    self.add_via_template_button.setText(
        self._tr("add_via_template_button", "Выделить шаблон" if self._ui_language == "ru" else "Pick template")
    )
    self.remove_via_template_button.setText(
        self._tr("remove_via_template_button", "Удалить выбранный" if self._ui_language == "ru" else "Remove selected")
    )
    self.clear_via_templates_button.setText(
        self._tr("clear_via_templates_button", "Удалить все" if self._ui_language == "ru" else "Clear all")
    )
    if getattr(self, "noisy_traces_via_preset_label_widget", None) is not None:
        self.noisy_traces_via_preset_label_widget.setText("")
    if self.via_preset_label_widget is not None:
        self.via_preset_label_widget.setText(
            self._tr("via_preset_label", "Пресеты поиска via" if self._ui_language == "ru" else "Via search presets")
        )
    self.apply_via_preset_button.setText(
        self._tr("apply_via_preset_button", "Применить" if self._ui_language == "ru" else "Apply")
    )
    self.save_via_preset_button.setText(
        self._tr("save_via_preset_button", "Сохранить" if self._ui_language == "ru" else "Save")
    )
    self.delete_via_preset_button.setText(
        self._tr("delete_via_preset_button", "Удалить" if self._ui_language == "ru" else "Delete")
    )
    self.noisy_traces_via_preset_button.setText(
        self._tr(
            "noisy_traces_via_preset_button",
            "Пресет: яркие via на дорожках" if self._ui_language == "ru" else "Preset: bright vias on traces",
        )
    )
    if getattr(self, "blurred_via_preset_label_widget", None) is not None:
        self.blurred_via_preset_label_widget.setText("")
    self.blurred_via_preset_button.setText(
        self._tr(
            "blurred_via_preset_button",
            "Пресет: слабые/размытые via" if self._ui_language == "ru" else "Preset: weak/blurred vias",
        )
    )
    self._refresh_via_preset_combo()
    if self.reset_via_search_label_widget is not None:
        self.reset_via_search_label_widget.setText("")
    self.reset_via_search_button.setText(
        self._tr(
            "reset_via_search_button",
            "Сбросить параметры поиска via" if self._ui_language == "ru" else "Reset via search parameters",
        )
    )
    if self.debug_candidates_label_widget is not None:
        self.debug_candidates_label_widget.setText(
            self._tr("debug_candidates_label", "Отладка via" if self._ui_language == "ru" else "Via debug")
        )
    self.debug_candidates_checkbox.setText(
        self._tr("debug_candidates_checkbox", "Проверять по клику" if self._ui_language == "ru" else "Inspect by click")
    )
    if getattr(self, "show_gradient_debug_label_widget", None) is not None:
        self.show_gradient_debug_label_widget.setText(
            self._tr("gradient_debug_label", "Карта градиента" if self._ui_language == "ru" else "Gradient map")
        )
    self.show_gradient_debug_button.setText(
        self._tr("gradient_debug_button", "Открыть карту" if self._ui_language == "ru" else "Show gradient map")
    )
    if getattr(self, "gradient_overlay_label_widget", None) is not None:
        self.gradient_overlay_label_widget.setText(
            self._tr("gradient_overlay_label", "Слой градиента" if self._ui_language == "ru" else "Gradient overlay")
        )
    self.gradient_overlay_checkbox.setText(
        self._tr(
            "gradient_overlay_checkbox",
            "Показывать на изображении" if self._ui_language == "ru" else "Overlay on image",
        )
    )
    self.gradient_overlay_mode_combo.setItemText(
        0, self._tr("gradient_overlay_mode_heatmap", "Тепловая карта" if self._ui_language == "ru" else "Heatmap")
    )
    self.gradient_overlay_mode_combo.setItemText(
        1,
        self._tr(
            "gradient_overlay_mode_threshold", "Маска по порогу" if self._ui_language == "ru" else "Threshold mask"
        ),
    )
    self.gradient_overlay_mode_combo.setItemText(
        2,
        self._tr("gradient_overlay_mode_elevation", "Серый градиент" if self._ui_language == "ru" else "Raw elevation"),
    )
    if self.via_roundness_label_widget is not None:
        self.via_roundness_label_widget.setText(
            self._tr("via_roundness_label", "Округлость" if self._ui_language == "ru" else "Roundness")
        )
    if self.min_via_width_label_widget is not None:
        self.min_via_width_label_widget.setText(
            self._tr("via_width_range_label", "Диапазон ширины via" if self._ui_language == "ru" else "Via width range")
        )
    if self.min_via_height_label_widget is not None:
        self.min_via_height_label_widget.setText(
            self._tr(
                "via_height_range_label", "Диапазон высоты via" if self._ui_language == "ru" else "Via height range"
            )
        )
    if self.fixed_vias_label_widget is not None:
        self.fixed_vias_label_widget.setText(
            self._tr("fixed_vias_label", "Фиксированные via" if self._ui_language == "ru" else "Fixed vias")
        )
    if self.min_hierarchy_depth_label_widget is not None:
        self.min_hierarchy_depth_label_widget.setText(self._tr("min_hierarchy_depth_label"))
    if self.min_inner_hole_area_label_widget is not None:
        self.min_inner_hole_area_label_widget.setText(self._tr("min_inner_hole_area_label"))
    if self.max_hierarchy_depth_label_widget is not None:
        self.max_hierarchy_depth_label_widget.setText(self._tr("max_hierarchy_depth_label"))
    if self.max_hole_area_ratio_label_widget is not None:
        self.max_hole_area_ratio_label_widget.setText(self._tr("max_hole_area_ratio_label"))
    self.save_group.setTitle(self._tr("save_options_group"))
    self.save_cif_checkbox.setText(self._tr("save_cif_checkbox"))
    self.save_cv_checkbox.setText(self._tr("save_cv_checkbox"))
    self.save_preview_checkbox.setText(self._tr("save_preview_checkbox"))
    self._set_common_tooltip(self.save_cif_checkbox, "save_cif")
    self._set_common_tooltip(self.save_cv_checkbox, "save_cv")
    self._set_common_tooltip(self.save_preview_checkbox, "save_preview")
    self._apply_extraction_tooltips()
    self._renumber_fixed_via_rows()
    self._update_extraction_profile_controls_state()

    if self.external_color_label_widget is not None:
        self.external_color_label_widget.setText(self._tr("external_contour_label"))
    if self.hole_color_label_widget is not None:
        self.hole_color_label_widget.setText(self._tr("hole_contour_label"))
    if self.selected_color_label_widget is not None:
        self.selected_color_label_widget.setText(self._tr("selected_contour_label"))
    if self.conductor_hover_highlight_label_widget is not None:
        self.conductor_hover_highlight_label_widget.setText(self._tr("conductor_hover_highlight_label"))
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
    self.random_object_colors_checkbox.setText(
        self._tr(
            "random_object_colors_checkbox",
            "Случайные цвета объектов" if self._ui_language == "ru" else "Random object colors",
        )
    )
    if hasattr(self, "show_frame_matrix_checkbox"):
        self.show_frame_matrix_checkbox.setText(
            self._tr(
                "show_frame_matrix_checkbox",
                "Show frame matrix",
            )
        )
    if hasattr(self, "show_frame_matrix_thumbnails_checkbox"):
        self.show_frame_matrix_thumbnails_checkbox.setText(
            self._tr(
                "show_frame_matrix_thumbnails_checkbox",
                "Load frame matrix thumbnails",
            )
        )
    self.show_neighbor_frames_checkbox.setText(
        self._tr(
            "show_neighbor_frames_checkbox",
            "Показывать соседние кадры" if self._ui_language == "ru" else "Show neighboring frames",
        )
    )
    if hasattr(self, "show_neighbor_vectors_checkbox"):
        self.show_neighbor_vectors_checkbox.setText(
            self._tr(
                "show_neighbor_vectors_checkbox",
                "Показывать векторы на соседних кадрах"
                if self._ui_language == "ru"
                else "Show vectors on neighboring frames",
            )
        )
    if self.neighbor_columns_label_widget is not None:
        self.neighbor_columns_label_widget.setText(
            self._tr("neighbor_columns_label", "Кадров в строке" if self._ui_language == "ru" else "Frames per row")
        )
    if self.neighbor_max_grid_label_widget is not None:
        self.neighbor_max_grid_label_widget.setText(
            self._tr("neighbor_max_grid_label", "Макс. сетка" if self._ui_language == "ru" else "Grid size")
        )
    if self.neighbor_opacity_label_widget is not None:
        self.neighbor_opacity_label_widget.setText(
            self._tr(
                "neighbor_opacity_label",
                "Прозрачность соседей" if self._ui_language == "ru" else "Neighbor opacity",
            )
        )
    if self.neighbor_overlap_label_widget is not None:
        self.neighbor_overlap_label_widget.setText(
            self._tr("neighbor_overlap_label", "Пересечение кадров" if self._ui_language == "ru" else "Frame overlap")
        )
    if self.extra_layers_label_widget is not None:
        self.extra_layers_label_widget.setText(
            self._tr("extra_layers_label", "Дополнительные слои" if self._ui_language == "ru" else "Additional layers")
        )
    if hasattr(self, "add_extra_layers_button"):
        self.add_extra_layers_button.setText("+")
        self.add_extra_layers_button.setToolTip("Добавить дополнительный слой из папки")
        self.add_extra_layers_button.setStatusTip(self.add_extra_layers_button.toolTip())
    for widget, tooltip_key in (
        (self.external_color_label_widget, "external_color"),
        (self.external_color_button, "external_color"),
        (self.hole_color_label_widget, "hole_color"),
        (self.hole_color_button, "hole_color"),
        (self.selected_color_label_widget, "selected_color"),
        (self.selected_color_button, "selected_color"),
        (self.conductor_hover_highlight_label_widget, "conductor_hover_highlight"),
        (self.conductor_hover_highlight_color_button, "conductor_hover_highlight"),
        (self.vertex_color_label_widget, "vertex_color"),
        (self.vertex_color_button, "vertex_color"),
        (self.line_width_label_widget, "line_width"),
        (self.line_width_spin, "line_width"),
        (self.vertex_size_label_widget, "vertex_size"),
        (self.vertex_size_spin, "vertex_size"),
        (self.fill_opacity_label_widget, "fill_opacity"),
        (self.fill_opacity_spin, "fill_opacity"),
        (self.show_vertices_checkbox, "show_vertices"),
        (self.show_labels_checkbox, "show_labels"),
    ):
        self._set_common_tooltip(widget, tooltip_key)
    for widget, tooltip in (
        (
            self.random_object_colors_checkbox,
            "Раскрашивает каждый объект отдельным цветом. Это удобно, когда нужно видеть, какие контуры остались отдельными после правки."
            if self._ui_language == "ru"
            else "Colors each object separately. Useful for seeing which contours remain separate after edits.",
        ),
        (
            self.show_neighbor_frames_checkbox,
            "Показывает соседние изображения вокруг текущего кадра на фоне. Текущий кадр остается в центре и отмечается желтой рамкой."
            if self._ui_language == "ru"
            else "Shows neighboring images around the current frame in the background. The current frame stays centered and has a yellow border.",
        ),
        (
            self.show_neighbor_vectors_checkbox,
            "Показывает CIF-векторы поверх соседних кадров."
            if self._ui_language == "ru"
            else "Shows matching CIF vectors over neighboring frames.",
        ),
        (
            self.neighbor_columns_spin,
            "Сколько кадров в одной строке исходной последовательности. Это нужно, чтобы правильно найти соседей сверху, снизу и по диагонали."
            if self._ui_language == "ru"
            else "How many frames are in one row of the source sequence. Used to locate top, bottom, and diagonal neighbors.",
        ),
        (
            self.neighbor_max_grid_spin,
            "Максимальный размер фоновой матрицы: 3, 5 или 7 кадров по стороне. При уменьшении масштаба сетка раскрывается до этого значения."
            if self._ui_language == "ru"
            else "Centered neighbor grid size: 3 shows one ring around the current frame, 5 shows two rings, 7 shows three rings.",
        ),
        (
            self.neighbor_opacity_spin,
            "Прозрачность соседних кадров на фоне. Меньше значение делает их менее заметными относительно основного кадра."
            if self._ui_language == "ru"
            else "Opacity of neighboring background frames. Lower values make them less prominent than the main frame.",
        ),
        (
            self.neighbor_overlap_spin,
            "Сколько пикселей соседние кадры заходят друг на друга. Ноль размещает кадры вплотную без пересечения."
            if self._ui_language == "ru"
            else "How many pixels neighboring frames overlap. Zero places frames edge to edge without overlap.",
        ),
        (
            self.extra_layers_widget,
            "Дополнительные слои загружаются только из папок и привязываются к базовым кадрам по номеру."
            if self._ui_language == "ru"
            else "Additional layers are loaded from folders and mapped to base frames by frame number.",
        ),
        (
            self.add_extra_layers_button,
            "Добавить дополнительный слой из папки"
            if self._ui_language == "ru"
            else "Add additional layer from folder",
        ),
    ):
        widget.setToolTip(tooltip)
        widget.setStatusTip(tooltip)

    if hasattr(self, "autosave_on_frame_transition_checkbox"):
        self.autosave_on_frame_transition_checkbox.setText(
            self._tr(
                "autosave_on_frame_transition_label",
                "Автосохранение при переходе к следующему кадру"
                if self._ui_language == "ru"
                else "Autosave on next frame",
            )
        )

    self.editor_group.setTitle(self._tr("editor_group_title"))
    self._update_tool_button_texts()
    self._update_action_button_texts()
    self.polygon_mode_label.setText("Полигон" if self._ui_language == "ru" else "Polygon")
    self.brush_mode_label.setText("Кисть" if self._ui_language == "ru" else "Brush")
    self.brush_size_label.setText("Толщина" if self._ui_language == "ru" else "Width")
    if hasattr(self, "trace_width_label"):
        self.trace_width_label.setText(self._tr("trace_width_label"))
    self.delete_vertex_mode_label.setText("Удаление" if self._ui_language == "ru" else "Delete")
    self.via_width_label.setText("Via W")
    self.via_height_label.setText("Via H")
    for widget, tooltip_key in (
        (self.polygon_mode_label, "polygon_mode"),
        (self.polygon_mode_combo, "polygon_mode"),
        (self.brush_mode_label, "brush_mode"),
        (self.brush_mode_combo, "brush_mode"),
        (self.brush_size_label, "brush_size"),
        (self.brush_size_spin, "brush_size"),
        (self.delete_vertex_mode_label, "delete_vertex_mode"),
        (self.delete_vertex_mode_combo, "delete_vertex_mode"),
        (self.via_width_label, "editor_via_width"),
        (self.via_width_spin, "editor_via_width"),
        (self.via_height_label, "editor_via_height"),
        (self.via_height_spin, "editor_via_height"),
    ):
        self._set_common_tooltip(widget, tooltip_key)
    self._on_editor_tool_changed(self.polygon_editor.current_tool)
    self._retranslate_editor_mode_combos()
    self.preview_busy_label.setText(self._busy_indicator_text())
    self._set_progress_status(self._progress_status_key, **self._progress_status_kwargs)

    self._populate_pipeline_operations()
    self._populate_pipeline_list()
    if selected_pipeline_row >= 0 and selected_pipeline_row < self.pipeline_list.count():
        self.pipeline_list.setCurrentRow(selected_pipeline_row)
    self._retranslate_contour_mode_combos()
    if selected_operation:
        target_item = self._find_operation_tree_item(selected_operation)
        if target_item is not None:
            self.operation_tree.setCurrentItem(target_item)
    self._update_pipeline_help_preview(self._selected_available_operation_name())
    self._refresh_help_menu()

