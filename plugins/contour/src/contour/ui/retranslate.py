"""Retranslate implementation for :class:`PolygonExtractionWidget`.

Split out of ``contour.widget`` during the production-ready refactor.
The function is bound as a method of the widget via attribute assignment, so
``self`` refers to the original widget and all attribute access is unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import QSignalBlocker
from PyQt6.QtWidgets import QComboBox, QFormLayout, QGroupBox

from ..vision.metal_recovery.strategy_registry import strategy_spec
from .bright_via_i18n import retranslate_bright_via_panel
from .i18n_content import PIPELINE_CONTROL_TOOLTIPS, _localized_text
from .metal_debug_i18n import retranslate_metal_debug_visual_combo
from .metal_strategy_i18n import choice_label, parameter_label, parameter_tooltip, strategy_name

if TYPE_CHECKING:
    from contour.widget import PolygonExtractionWidget


def _set_form_label(group: QGroupBox, field: object, text: str) -> None:
    form = group.layout()
    if not isinstance(form, QFormLayout):
        return
    label = form.labelForField(field)
    if label is not None:
        label.setText(text)


def _retranslate_strategy_parameter_pages(self: PolygonExtractionWidget) -> None:
    language = str(self._ui_language)
    for strategy_id, controls in getattr(self, "metal_strategy_parameter_widgets", {}).items():
        spec = strategy_spec(strategy_id)
        parameters = {parameter.key: parameter for parameter in spec.parameters}
        for key, control in controls.items():
            parameter = parameters[key]
            tooltip = parameter_tooltip(parameter, language)
            control.setToolTip(tooltip)
            parent = control.parentWidget()
            form = parent.layout() if parent is not None else None
            if isinstance(form, QFormLayout):
                label = form.labelForField(control)
                if label is not None:
                    label.setText(parameter_label(parameter, language))
                    label.setToolTip(tooltip)
            if isinstance(control, QComboBox):
                labels_by_value = dict(parameter.choices)
                for index in range(control.count()):
                    value = str(control.itemData(index))
                    source_label = labels_by_value.get(value)
                    if source_label is not None:
                        control.setItemText(index, choice_label(source_label, language))

        page_index = self.metal_strategy_parameter_pages.get(strategy_id)
        if page_index is None:
            continue
        page = self.metal_strategy_parameter_stack.widget(page_index)
        for group in page.findChildren(QGroupBox):
            advanced = group.objectName().endswith("_advanced_group")
            group.setTitle(
                ("Дополнительные" if advanced else "Основные")
                if language == "ru"
                else ("Advanced" if advanced else "Basic")
            )


def retranslate_ui(self: PolygonExtractionWidget) -> None:
    if not hasattr(self, "control_tabs"):
        return
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
    if getattr(self, "metal_adaptive_group", None) is not None:
        self.metal_adaptive_group.setTitle("Адаптивный порог" if self._ui_language == "ru" else "Adaptive threshold")
    if getattr(self, "metal_adaptive_block_label_widget", None) is not None:
        self.metal_adaptive_block_label_widget.setText(
            "Размер окна, px" if self._ui_language == "ru" else "Window size, px"
        )
    if getattr(self, "metal_adaptive_block_spin", None) is not None:
        self.metal_adaptive_block_spin.setSpecialValueText("Авто" if self._ui_language == "ru" else "Auto")
    if getattr(self, "metal_adaptive_c_label_widget", None) is not None:
        self.metal_adaptive_c_label_widget.setText(
            "Смещение порога C" if self._ui_language == "ru" else "Threshold offset C"
        )
    if getattr(self, "metal_adaptive_method_label_widget", None) is not None:
        self.metal_adaptive_method_label_widget.setText("Метод" if self._ui_language == "ru" else "Method")
    if getattr(self, "metal_adaptive_method_combo", None) is not None:
        self.metal_adaptive_method_combo.setItemText(0, "Гаусс" if self._ui_language == "ru" else "Gaussian")
        self.metal_adaptive_method_combo.setItemText(1, "Среднее" if self._ui_language == "ru" else "Mean")

    if hasattr(self, "metal_basic_group"):
        is_ru = self._ui_language == "ru"
        if hasattr(self, "metal_preset_group"):
            self.metal_preset_group.setTitle(
                "Пресеты распознавания" if is_ru else "Recognition presets"
            )
        self.metal_basic_group.setTitle("Основные параметры" if is_ru else "Basic parameters")
        self.metal_display_group.setTitle("Отображение" if is_ru else "Display")
        _set_form_label(
            self.metal_basic_group,
            self.metal_min_contrast_widget,
            "Мин. контраст проводника" if is_ru else "Minimum conductor contrast",
        )
        _set_form_label(
            self.metal_basic_group,
            self.metal_gap_bridge_spin,
            "Сшивка разрывов, пикс." if is_ru else "Gap bridging, px",
        )
        _set_form_label(
            self.metal_basic_group,
            self.metal_speckle_removal_spin,
            "Удаление шума, пикс." if is_ru else "Speckle removal, px",
        )
        _set_form_label(
            self.metal_basic_group,
            self.metal_width_row,
            "Ширина проводника, пикс." if is_ru else "Conductor width, px",
        )
        if hasattr(self, "metal_conductor_size_offset_widget"):
            _set_form_label(
                self.metal_basic_group,
                self.metal_conductor_size_offset_widget,
                "Размер проводника, пикс." if is_ru else "Conductor size, px",
            )
        _set_form_label(
            self.metal_basic_group,
            self.metal_epsilon_spin,
            "Точность аппроксимации" if is_ru else "Approximation epsilon",
        )
        self.metal_show_conductors_checkbox.setText("Показывать проводники" if is_ru else "Show conductors")
        self.metal_show_rejected_checkbox.setText("Показывать отклонённые" if is_ru else "Show rejected")
        self.metal_show_suspicious_checkbox.setText("Показывать подозрительные" if is_ru else "Show suspicious")
        self.metal_show_mask_checkbox.setText("Показывать маску" if is_ru else "Show mask")
        if getattr(self, "metal_show_border_checkbox", None) is not None:
            self.metal_show_border_checkbox.setText(
                "Подсветка у границы кадра" if is_ru else "Highlight objects at frame border"
            )
        if hasattr(self, "metal_debug_visual_combo"):
            retranslate_metal_debug_visual_combo(self.metal_debug_visual_combo, self._ui_language)
        _set_form_label(
            self.metal_display_group,
            self.metal_overlay_opacity_spin,
            "Прозрачность наложения" if is_ru else "Overlay opacity",
        )

        _set_form_label(
            self.metal_filter_group,
            self.metal_min_object_source_contrast_spin,
            "Мин. контраст объекта к фону" if is_ru else "Minimum object/background contrast",
        )
        _set_form_label(
            self.metal_filter_group,
            self.metal_min_area_spin,
            "Мин. площадь" if is_ru else "Minimum area",
        )
        _set_form_label(
            self.metal_filter_group,
            self.metal_max_area_spin,
            "Макс. площадь (0 — без ограничения)" if is_ru else "Maximum area (0 = unlimited)",
        )
        _set_form_label(
            self.metal_filter_group,
            self.metal_min_perimeter_spin,
            "Мин. периметр" if is_ru else "Minimum perimeter",
        )
        _set_form_label(
            self.metal_filter_group,
            self.metal_max_perimeter_spin,
            "Макс. периметр (0 — без ограничения)" if is_ru else "Maximum perimeter (0 = unlimited)",
        )
        _set_form_label(
            self.metal_filter_group,
            self.metal_min_length_spin,
            "Мин. длина проводника, пикс." if is_ru else "Minimum conductor length, px",
        )
        _set_form_label(
            self.metal_filter_group,
            self.metal_min_points_spin,
            "Мин. число точек" if is_ru else "Minimum point count",
        )
        _set_form_label(
            self.metal_filter_group,
            self.metal_min_angle_spin,
            "Мин. угол полигона, °" if is_ru else "Minimum polygon angle, °",
        )
        _set_form_label(
            self.metal_filter_group,
            self.metal_border_handling_combo,
            "Объекты у границы кадра" if is_ru else "Objects at image border",
        )
        border_labels = (
            ("Игнорировать", "Принимать", "Помечать отдельно") if is_ru else ("Ignore", "Accept", "Mark separately")
        )
        for index, text in enumerate(border_labels):
            self.metal_border_handling_combo.setItemText(index, text)

        self.metal_approximation_checkbox.setText("Аппроксимировать контуры" if is_ru else "Approximate contours")
        _set_form_label(
            self.metal_recognition_params_group,
            self.metal_hierarchy_combo,
            "Режим иерархии" if is_ru else "Hierarchy mode",
        )
        hierarchy_labels = (
            ("Полная иерархия", "Только внешние контуры") if is_ru else ("Full hierarchy", "External contours only")
        )
        for index, text in enumerate(hierarchy_labels):
            self.metal_hierarchy_combo.setItemText(index, text)
        _set_form_label(
            self.metal_recognition_params_group,
            self.min_inner_hole_area_spin,
            "Мин. площадь отверстия для заливки, пикс.²" if is_ru else "Minimum hole area to preserve, px²",
        )
        _set_form_label(
            self.metal_recognition_params_group,
            self.metal_min_hole_source_contrast_spin,
            "Мин. контраст отверстия" if is_ru else "Minimum hole contrast",
        )
        _set_form_label(
            self.metal_recognition_params_group,
            self.metal_min_hole_source_contrast_fraction_spin,
            "Доля контраста классов для отверстия" if is_ru else "Class-contrast fraction for hole",
        )

        self.metal_watershed_group.setTitle("Параметры водораздела" if is_ru else "Watershed parameters")
        self.metal_random_walker_group.setTitle("Случайное блуждание" if is_ru else "Random Walker")
        self.metal_graph_cut_group.setTitle("Графовый разрез" if is_ru else "Graph Cut")
        self.metal_reconstruction_group.setTitle("Морфологическая реконструкция" if is_ru else "Reconstruction")
        self.metal_closed_boundary_group.setTitle("Замкнутые границы" if is_ru else "Closed boundary")
        self.metal_adaptive_group.setTitle("Адаптивный порог" if is_ru else "Adaptive threshold")
        self.apply_metal_preset_button.setText("Применить" if is_ru else "Apply")
        self.save_metal_preset_button.setText("Сохранить" if is_ru else "Save")
        self.delete_metal_preset_button.setText("Удалить" if is_ru else "Delete")
        if hasattr(self, "export_metal_preset_button"):
            self.export_metal_preset_button.setText("Выгрузить" if is_ru else "Export")
        if hasattr(self, "import_metal_preset_button"):
            self.import_metal_preset_button.setText("Загрузить" if is_ru else "Import")
        self.metal_reset_params_button.setText("Сбросить параметры" if is_ru else "Reset parameters")
        if hasattr(self, "_refresh_metal_preset_combo"):
            self._refresh_metal_preset_combo()

        _retranslate_strategy_parameter_pages(self)
        from .builders import _sync_metal_strategy_panel

        _sync_metal_strategy_panel(self)
    if getattr(self, "metal_gradient_3d_button", None) is not None:
        self.metal_gradient_3d_button.setText("3D поле" if self._ui_language == "ru" else "3D field")
    if getattr(self, "metal_debug_visual_label_widget", None) is not None:
        self.metal_debug_visual_label_widget.setText(
            "Режим отладочного изображения" if self._ui_language == "ru" else "Debug image mode"
        )
    if getattr(self, "_gradient_field_3d_window", None) is not None:
        try:
            self._gradient_field_3d_window.set_ui_language(self._ui_language)
        except RuntimeError:
            self._gradient_field_3d_window = None
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
    if getattr(self, "via_search_mode_label_widget", None) is not None:
        self.via_search_mode_label_widget.setText(
            self._tr(
                "via_search_mode_label",
                "Режим поиска контактов" if self._ui_language == "ru" else "Contact search mode",
            )
        )
    if self.via_search_mode_combo.count() >= 3:
        if getattr(self, "bright_via_diameter_range_label_widget", None) is not None:
            self.bright_via_diameter_range_label_widget.setText(
                "Диаметр поиска" if self._ui_language == "ru" else "Candidate diameter range, px"
            )
        if getattr(self, "via_output_diameter_label_widget", None) is not None:
            self.via_output_diameter_label_widget.setText(
                "Диаметр сохранения" if self._ui_language == "ru" else "Saved contact diameter, px"
            )
        self.via_search_mode_combo.setItemText(
            0,
            self._tr("via_search_mode_heuristic", "Эвристический" if self._ui_language == "ru" else "Heuristic"),
        )
    if getattr(self, "bright_via_viamode_label_widget", None) is not None:
        self.bright_via_viamode_label_widget.setText(
            "Метод поиска контактов" if self._ui_language == "ru" else "Contact search method"
        )
        if getattr(self, "bright_via_polarity_label_widget", None) is not None:
            self.bright_via_polarity_label_widget.setText(
                "Полярность контакта" if self._ui_language == "ru" else "Contact polarity"
            )
        if getattr(self, "bright_via_diameter_mode_label_widget", None) is not None:
            self.bright_via_diameter_mode_label_widget.setText(
                "Размер контакта" if self._ui_language == "ru" else "Contact size"
            )
        if getattr(self, "bright_via_diameter_fixed_label_widget", None) is not None:
            self.bright_via_diameter_fixed_label_widget.setText(
                "Диаметр контакта, px" if self._ui_language == "ru" else "Contact diameter, px"
            )
        self.via_search_mode_combo.setItemText(
            1,
            self._tr("via_search_mode_template", "По шаблону" if self._ui_language == "ru" else "Template"),
        )
        self.via_search_mode_combo.setItemText(
            2,
            self._tr("via_search_mode_hybrid", "Смешанный" if self._ui_language == "ru" else "Mixed"),
        )
    elif self.via_search_mode_combo.count() >= 2:
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
        self._tr("via_white_range_method", "Распознавать светлые" if self._ui_language == "ru" else "Recognize bright")
    )
    self.via_black_range_checkbox.setText(
        self._tr("via_black_range_method", "Распознавать тёмные" if self._ui_language == "ru" else "Recognize dark")
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
    if hasattr(self, "via_template_table"):
        self.via_template_table.setHorizontalHeaderLabels(
            ["№", "Вид", "Похожесть", "Размер", "Удалить"]
            if self._ui_language == "ru"
            else ["No.", "Preview", "Similarity", "Size", "Delete"]
        )
        self._refresh_via_template_list()
    if getattr(self, "noisy_traces_via_preset_label_widget", None) is not None:
        self.noisy_traces_via_preset_label_widget.setText("")
    if self.via_preset_label_widget is not None:
        self.via_preset_label_widget.setText(
            self._tr(
                "via_preset_label",
                "Пресеты поиска контактов" if self._ui_language == "ru" else "Contact search presets",
            )
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
    self._refresh_via_preset_combo()
    if self.reset_via_search_label_widget is not None:
        self.reset_via_search_label_widget.setText("")
    self.reset_via_search_button.setText(
        self._tr(
            "reset_via_search_button",
            "Сбросить параметры поиска контактов" if self._ui_language == "ru" else "Reset contact search parameters",
        )
    )
    self.debug_candidates_checkbox.setText(
        self._tr("debug_candidates_checkbox", "Проверять по клику" if self._ui_language == "ru" else "Inspect by click")
    )
    self.via_show_detected_checkbox.setText(
        "Показывать найденные контакты" if self._ui_language == "ru" else "Show detected contacts"
    )
    retranslate_bright_via_panel(self)
    if getattr(self, "gradient_overlay_label_widget", None) is not None:
        self.gradient_overlay_label_widget.setText(
            self._tr(
                "gradient_overlay_label",
                "Вид" if self._ui_language == "ru" else "View",
            )
        )
    _overlay_names = {
        "source": ("Исходное изображение", "Source image"),
        "heatmap": ("Тепловая карта", "Heatmap"),
        "threshold": ("Маска по порогу", "Threshold mask"),
        "elevation": ("Серый градиент", "Raw elevation"),
        "mask": ("Итоговая маска", "Final mask"),
        "candidate_mask": ("Маска кандидатов", "Candidate mask"),
        "via_mask": ("Маска контактов", "Contact mask"),
        "metal_mask": ("Маска металла", "Metal mask"),
        "metal_binary_mask": ("Бинарная маска металла", "Metal binary mask"),
        "metal_filtered_mask": ("Отфильтрованная маска металла", "Filtered metal mask"),
        "metal_gradient_x": ("Градиент по X", "Gradient X"),
        "metal_gradient_y": ("Градиент по Y", "Gradient Y"),
        "metal_gradient_field": ("Градиентное поле", "Gradient field"),
        "tophat_mask": ("Маска Top-hat", "Top-hat mask"),
        "dog_mask": ("Маска DoG", "DoG mask"),
        "spot_response": ("Отклик светлых пятен", "Bright-spot response"),
        "spot_response_dark": ("Отклик тёмных пятен", "Dark-spot response"),
        "ring_response": ("Отклик колец", "Ring response"),
        "background_corrected": ("Скорректированный фон", "Background corrected"),
        "local_max_bright": ("Локальные светлые максимумы", "Local bright maxima"),
        "local_max_dark": ("Локальные тёмные максимумы", "Local dark maxima"),
        "binary_mask": ("Бинарная маска", "Binary mask"),
        "tophat": ("Top-hat", "Top-hat"),
        "dog": ("DoG", "DoG"),
        "radial_symmetry": ("Радиальная симметрия", "Radial symmetry"),
        "edge_likeness": ("Похожесть на край", "Edge likeness"),
        "line_likeness": ("Похожесть на линию", "Line likeness"),
        "distance_to_edge": ("Расстояние до края", "Distance to edge"),
        "scharr": ("Scharr", "Scharr"),
        "phase_congruency": ("Фазовая согласованность", "Phase congruency"),
        "structured": ("Структурные границы", "Structured edges"),
        "ridge": ("Гребни", "Ridges"),
        "processed": ("Обработанное изображение", "Processed image"),
        "raw_gray": ("Исходное изображение детектора", "Detector source image"),
        "source_gray": ("Исходное серое изображение", "Source grayscale"),
        "conductor_gradient_elevation": ("Градиент границ проводников", "Conductor edge gradient"),
        "template_count": ("Количество совпадений шаблонов", "Template match count"),
        "overlay": ("Наложение детектора", "Detector overlay"),
        "final_overlay": ("Итоговое наложение", "Final overlay"),
    }
    for _index in range(self.gradient_overlay_mode_combo.count()):
        _mode = str(self.gradient_overlay_mode_combo.itemData(_index) or "")
        _names = _overlay_names.get(_mode)
        if _names is not None:
            self.gradient_overlay_mode_combo.setItemText(
                _index,
                _names[0] if self._ui_language == "ru" else _names[1],
            )
    if self.via_roundness_label_widget is not None:
        self.via_roundness_label_widget.setText(
            self._tr("via_roundness_label", "Округлость" if self._ui_language == "ru" else "Roundness")
        )
    if self.min_via_width_label_widget is not None:
        self.min_via_width_label_widget.setText(
            self._tr(
                "via_width_range_label",
                "Диапазон ширины контакта" if self._ui_language == "ru" else "Contact width range",
            )
        )
    if self.min_via_height_label_widget is not None:
        self.min_via_height_label_widget.setText(
            self._tr(
                "via_height_range_label",
                "Диапазон высоты контакта" if self._ui_language == "ru" else "Contact height range",
            )
        )
    if self.fixed_vias_label_widget is not None:
        self.fixed_vias_label_widget.setText(
            self._tr(
                "fixed_vias_label",
                "Фиксированные контакты" if self._ui_language == "ru" else "Fixed contacts",
            )
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
    if self.via_selection_color_label_widget is not None:
        self.via_selection_color_label_widget.setText(self._tr("via_selection_color_label"))
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
        (self.via_selection_color_label_widget, "via_selection_color"),
        (self.via_selection_color_button, "via_selection_color"),
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

    if hasattr(self.editor_group, "setTitle"):
        self.editor_group.setTitle(self._tr("editor_group_title"))
    self._update_tool_button_texts()
    self._update_action_button_texts()
    self.polygon_mode_label.setText("Полигон" if self._ui_language == "ru" else "Polygon")
    self.brush_mode_label.setText("Кисть" if self._ui_language == "ru" else "Brush")
    self.brush_size_label.setText("Толщина" if self._ui_language == "ru" else "Width")
    if hasattr(self, "trace_width_label"):
        self.trace_width_label.setText(self._tr("trace_width_label"))
    self.delete_vertex_mode_label.setText("Удаление" if self._ui_language == "ru" else "Delete")
    self.via_width_label.setText("Ширина" if self._ui_language == "ru" else "Width")
    self.via_height_label.setText("Высота" if self._ui_language == "ru" else "Height")
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
