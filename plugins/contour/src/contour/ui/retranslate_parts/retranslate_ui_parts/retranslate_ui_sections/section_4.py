"""retranslate_ui section 4."""
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

def _retranslate_ui_section_4(self, selected_operation, selected_pipeline_row, tooltip, tooltip_key, widget):
    for widget, tooltip in (
        (
            self.random_object_colors_checkbox,
            "Близкие отдельные проводники получают максимально различимые цвета. "
            "Это помогает замечать разрывы и лишние соединения."
            if self._ui_language == "ru"
            else "Nearby separate conductors get maximally distinct colors. "
            "This helps reveal breaks and unintended connections.",
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
    if hasattr(self, "trace_mode_label"):
        self.trace_mode_label.setText("Трасса" if self._ui_language == "ru" else "Trace")
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
        (self.trace_mode_label, "trace_mode"),
        (self.trace_mode_combo, "trace_mode"),
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
