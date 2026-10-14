"""retranslate_ui section 3."""
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

def _retranslate_ui_section_3(self, tooltip_key, widget):
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
            "Контрастные цвета соседних объектов"
            if self._ui_language == "ru"
            else "Contrasting colors for nearby objects",
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
    return tooltip_key, widget
