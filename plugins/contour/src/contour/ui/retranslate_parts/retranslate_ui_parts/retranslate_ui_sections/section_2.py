"""retranslate_ui section 2."""
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

def _retranslate_ui_section_2(self, index):
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
        from contour.ui.builders import _sync_metal_strategy_panel

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
