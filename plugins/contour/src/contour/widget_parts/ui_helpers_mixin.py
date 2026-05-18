from __future__ import annotations

from ._imports import *  # noqa: F403


class WidgetUiHelpersMixin:
    def _build_visual_panel(self) -> QWidget:
        return build_visual_panel(self)

    def _build_editor_toolbar(self) -> QWidget:
        return build_editor_toolbar(self)

    def _sync_editor_via_size(self) -> None:
        self.polygon_editor.set_via_size(float(self.via_width_spin.value()), float(self.via_height_spin.value()))

    def _configure_toolbar_button(
        self,
        button: QToolButton,
        icon: QIcon,
        text: str,
        *,
        checkable: bool = False,
    ) -> None:
        button.setIcon(icon)
        button.setIconSize(QSize(self._toolbar_icon_size_px(), self._toolbar_icon_size_px()))
        button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        button.setToolTip(text)
        button.setStatusTip(text)
        button.setAccessibleName(text)
        button.setAutoRaise(False)
        button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        button.setFixedSize(self._toolbar_button_size_px(), self._toolbar_button_size_px())
        button.setCheckable(checkable)

    def _on_editor_tool_button_clicked(self, tool: EditorTool) -> None:
        shift_clicked = bool(QApplication.keyboardModifiers() & Qt.KeyboardModifier.ShiftModifier)
        self.polygon_editor.set_tool(tool)
        self.polygon_editor.setFocus(Qt.FocusReason.ShortcutFocusReason)
        if shift_clicked:
            self._cycle_editor_tool_mode(tool)

    def _cycle_editor_tool_mode(self, tool: EditorTool) -> None:
        combo = None
        if tool == EditorTool.ADD_POLYGON and hasattr(self, "polygon_mode_combo"):
            combo = self.polygon_mode_combo
        elif tool == EditorTool.BRUSH and hasattr(self, "brush_mode_combo"):
            combo = self.brush_mode_combo
        elif tool == EditorTool.DELETE_VERTEX and hasattr(self, "delete_vertex_mode_combo"):
            combo = self.delete_vertex_mode_combo
        if combo is None or combo.count() < 2:
            return
        combo.setCurrentIndex((combo.currentIndex() + 1) % combo.count())

    def _sync_polygon_mode_combo(self, mode: PolygonCreateMode) -> None:
        self._sync_mode_combo(self.polygon_mode_combo, mode)

    def _sync_brush_mode_combo(self, mode: BrushMode) -> None:
        self._sync_mode_combo(self.brush_mode_combo, mode)

    def _sync_delete_vertex_mode_combo(self, mode: DeleteVertexMode) -> None:
        self._sync_mode_combo(self.delete_vertex_mode_combo, mode)

    def _sync_mode_combo(self, combo: QComboBox, mode: object) -> None:
        index = combo.findData(mode)
        if index < 0 or index == combo.currentIndex():
            return
        blocker = QSignalBlocker(combo)
        try:
            combo.setCurrentIndex(index)
        finally:
            del blocker

    def _create_editor_tool_icon(self, tool: EditorTool) -> QIcon:
        return create_editor_tool_icon(tool)

    def _create_editor_action_icon(self, action: str) -> QIcon:
        return create_editor_action_icon(action)

    @staticmethod
    def _toolbar_icon_size_px() -> int:
        return TOOLBAR_ICON_SIZE_PX

    @staticmethod
    def _toolbar_button_size_px() -> int:
        return TOOLBAR_BUTTON_SIZE_PX

    @staticmethod
    def _toolbar_icon_canvas_size_px() -> int:
        return TOOLBAR_ICON_CANVAS_SIZE_PX

    def _tr(self, key: str, default: str = "", **kwargs) -> str:
        return tr(key, default=default, language=self._ui_language, **kwargs)

    def _set_common_tooltip(self, widget: QWidget | None, key: str) -> None:
        if widget is None:
            return
        tooltip = _localized_text(GENERAL_CONTROL_TOOLTIPS, key, self._ui_language)
        widget.setToolTip(tooltip)
        widget.setStatusTip(tooltip)

    def _mode_text(self, key: str) -> str:
        if self._ui_language == "ru":
            mapping = {
                "polygon_points": "По точкам",
                "polygon_rectangle": "Прямоугольник",
                "brush_freeform": "Произвольная",
                "brush_45deg": "45° шаг",
                "delete_single": "Вершина",
                "delete_area": "Область",
            }
        else:
            mapping = {
                "polygon_points": "By points",
                "polygon_rectangle": "Rectangle",
                "brush_freeform": "Freeform",
                "brush_45deg": "45° constrained",
                "delete_single": "Single vertex",
                "delete_area": "Area",
            }
        return mapping[key]

    def _busy_indicator_text(self) -> str:
        if self._busy_progress_stage:
            return self._busy_progress_stage
        return "Обработка..." if self._ui_language == "ru" else "Processing..."

    def _preview_progress_stages(self) -> list[tuple[int, str]]:
        ru = self._ui_language == "ru"
        if getattr(self, "_preview_running_signature", None) is not None:
            request = getattr(self, "_preview_running_request_for_progress", None)
            settings = getattr(request, "contour_settings", None)
            if getattr(settings, "object_type", "") == "via" or getattr(settings, "output_mode", "") == "box":
                return [
                    (12, "Подготовка изображения" if ru else "Preparing image"),
                    (32, "Поиск ярких/тёмных пиков via" if ru else "Finding via peaks"),
                    (58, "Проверка размера, формы и контраста" if ru else "Checking size, shape and contrast"),
                    (78, "Подавление дублей" if ru else "Merging duplicate candidates"),
                    (92, "Построение контуров via" if ru else "Building via contours"),
                ]
        return [
            (18, "Подготовка изображения" if ru else "Preparing image"),
            (48, "Построение маски" if ru else "Building mask"),
            (76, "Извлечение контуров" if ru else "Extracting contours"),
            (92, "Обновление редактора" if ru else "Updating editor"),
        ]

    def _reset_busy_progress(self, request: PreviewProcessingRequest | None = None) -> None:
        self._preview_running_request_for_progress = request
        self._busy_progress_value = 0
        self._busy_progress_stage = self._preview_progress_stages()[0][1]
        if hasattr(self, "preview_busy_progress"):
            self.preview_busy_progress.setRange(0, 100)
            self.preview_busy_progress.setValue(0)
            self.preview_busy_progress.setFormat("%p%")

    def _advance_busy_progress(self) -> None:
        if self._preview_running_request_id is None:
            return
        stages = self._preview_progress_stages()
        cap = stages[-1][0]
        if self._busy_progress_value < cap:
            step = 3 if self._busy_progress_value < 60 else 1
            self._busy_progress_value = min(cap, self._busy_progress_value + step)
        for threshold, label in stages:
            if self._busy_progress_value <= threshold:
                self._busy_progress_stage = label
                break
        if hasattr(self, "preview_busy_progress"):
            self.preview_busy_progress.setValue(self._busy_progress_value)
        if hasattr(self, "preview_busy_label"):
            self.preview_busy_label.setText(f"{self._busy_indicator_text()} — {self._busy_progress_value}%")

    def _set_progress_status(self, key: str, **kwargs) -> None:
        self._progress_status_key = key
        self._progress_status_kwargs = dict(kwargs)

    def set_ui_language(self, language: str | None) -> None:
        self._ui_language = active_language(language)
        self._batch_processor.set_ui_language(self._ui_language)
        if hasattr(self, "polygon_editor"):
            self.polygon_editor.set_ui_language(self._ui_language)
        self._retranslate_ui()
        if hasattr(self, "recognition_mode_combo"):
            self._update_extraction_profile_controls_state()

    def _retranslate_ui(self) -> None:
        retranslate_ui(self)

    def _update_tool_button_texts(self) -> None:
        texts = {
            EditorTool.RULER: self._tr("tool_ruler", "Ruler"),
            EditorTool.ADD_VIA: self._tr("tool_add_via", "Via"),
            EditorTool.SELECT: self._tr("tool_select", "Выбор" if self._ui_language == "ru" else "Select"),
            EditorTool.PAN: self._tr("tool_pan", "Панорамирование" if self._ui_language == "ru" else "Pan"),
            EditorTool.ADD_POLYGON: self._tr(
                "tool_add_polygon", "Полигон" if self._ui_language == "ru" else "Add polygon"
            ),
            EditorTool.BRUSH: self._tr("tool_brush", "Кисть" if self._ui_language == "ru" else "Brush"),
            EditorTool.ADD_VERTEX: self._tr(
                "tool_add_vertex", "Добавить вершину" if self._ui_language == "ru" else "Add vertex"
            ),
            EditorTool.DELETE_VERTEX: self._tr(
                "tool_delete_vertex", "Удалить вершину" if self._ui_language == "ru" else "Delete vertex"
            ),
            EditorTool.MOVE_VERTEX: self._tr(
                "tool_move_vertex", "Переместить вершину" if self._ui_language == "ru" else "Move vertex"
            ),
            EditorTool.DELETE_POLYGON: self._tr(
                "tool_delete_polygon", "Удалить полигон" if self._ui_language == "ru" else "Delete polygon"
            ),
        }
        for tool, button in self._tool_buttons.items():
            label = texts.get(tool, tool.value)
            if tool == EditorTool.RULER:
                label = self._tr("tool_ruler", "Линейка" if self._ui_language == "ru" else "Ruler")
            tooltip_pair = EDITOR_TOOL_TOOLTIPS.get(tool)
            base_tip = (tooltip_pair[0] if self._ui_language == "ru" else tooltip_pair[1]) if tooltip_pair else label
            shortcut_tip = tool_shortcut_native_text(tool)
            tooltip = append_shortcut_to_tooltip(base_tip, shortcut_tip)
            button.setToolTip(tooltip)
            button.setStatusTip(tooltip)
            button.setAccessibleName(label)

    def _update_action_button_texts(self) -> None:
        undo_key = QKeySequence(QKeySequence.StandardKey.Undo).toString(QKeySequence.SequenceFormat.NativeText)
        redo_key = QKeySequence(QKeySequence.StandardKey.Redo).toString(QKeySequence.SequenceFormat.NativeText)
        for button, label in [
            (self.undo_button, self._tr("undo_button", "Отменить" if self._ui_language == "ru" else "Undo")),
            (self.redo_button, self._tr("redo_button", "Повторить" if self._ui_language == "ru" else "Redo")),
            (self.zoom_in_button, self._tr("zoom_in_button", "Увеличить" if self._ui_language == "ru" else "Zoom in")),
            (
                self.zoom_out_button,
                self._tr("zoom_out_button", "Уменьшить" if self._ui_language == "ru" else "Zoom out"),
            ),
            (self.fit_button, self._tr("fit_button", "Подогнать" if self._ui_language == "ru" else "Fit")),
        ]:
            button.setAccessibleName(label)
        shortcuts_map = {
            self.undo_button: undo_key,
            self.redo_button: redo_key,
            self.zoom_in_button: "",
            self.zoom_out_button: "",
            self.fit_button: "",
        }
        for button, tooltip_key in (
            (self.undo_button, "undo_button"),
            (self.redo_button, "redo_button"),
            (self.zoom_in_button, "zoom_in_button"),
            (self.zoom_out_button, "zoom_out_button"),
            (self.fit_button, "fit_button"),
        ):
            tooltip = _localized_text(EDITOR_ACTION_TOOLTIPS, tooltip_key, self._ui_language)
            shortcut = shortcuts_map.get(button, "")
            full_tip = append_shortcut_to_tooltip(tooltip, shortcut) if shortcut else tooltip
            button.setToolTip(full_tip)
            button.setStatusTip(full_tip)

    def _on_editor_tool_changed(self, tool) -> None:
        is_ruler = tool == EditorTool.RULER
        self.ruler_status_label.setVisible(is_ruler)
        if is_ruler and not self.ruler_status_label.text():
            self.ruler_status_label.setText(
                self._tr(
                    "ruler_idle_label",
                    "Потяните на изображении для измерения"
                    if self._ui_language == "ru"
                    else "Drag on the image to measure",
                )
            )
        elif not is_ruler:
            self.ruler_status_label.clear()
        if hasattr(self, "_polygon_toolbar_block"):
            self._polygon_toolbar_block.setVisible(tool == EditorTool.ADD_POLYGON)
        if hasattr(self, "_brush_toolbar_block"):
            self._brush_toolbar_block.setVisible(tool == EditorTool.BRUSH)
        if hasattr(self, "_via_toolbar_block"):
            self._via_toolbar_block.setVisible(tool == EditorTool.ADD_VIA)
        if hasattr(self, "_delete_vertex_toolbar_block"):
            self._delete_vertex_toolbar_block.setVisible(tool == EditorTool.DELETE_VERTEX)
        self._place_active_tool_parameters_near_tool_button(tool)
        self._on_effective_polygon_create_mode_changed(self.polygon_editor.effective_polygon_create_mode())

    def _place_active_tool_parameters_near_tool_button(self, tool: EditorTool) -> None:
        if not hasattr(self, "_editor_toolbar_layout") or not hasattr(self, "_tool_parameter_blocks"):
            return
        layout = self._editor_toolbar_layout
        blocks = self._tool_parameter_blocks
        for block in blocks.values():
            if block is not None:
                layout.removeWidget(block)
        active_block = blocks.get(tool)
        active_button = getattr(self, "_tool_buttons", {}).get(tool)
        if active_block is None or active_button is None:
            return
        button_index = layout.indexOf(active_button)
        if button_index < 0:
            return
        layout.insertWidget(button_index + 1, active_block)
        active_block.setVisible(True)
        active_block.adjustSize()
        if hasattr(self, "editor_toolbar"):
            self.editor_toolbar.adjustSize()
            self.editor_toolbar.setMinimumWidth(self.editor_toolbar.sizeHint().width())
        if hasattr(self, "editor_toolbar_scroll"):
            top_left = active_block.mapTo(self.editor_toolbar, active_block.rect().topLeft())
            self.editor_toolbar_scroll.ensureVisible(int(top_left.x()), int(top_left.y()), 24, 0)

    def _on_effective_polygon_create_mode_changed(self, mode: PolygonCreateMode) -> None:
        if not hasattr(self, "polygon_draw_mode_indicator"):
            return
        # Keep mode switching/hotkeys functional, but do not render mode text in work area/toolbar.
        _ = mode
        self.polygon_draw_mode_indicator.clear()

    def _update_ruler_status(self, text: str) -> None:
        if not text:
            if self.polygon_editor.current_tool == EditorTool.RULER:
                self.ruler_status_label.setText(
                    self._tr(
                        "ruler_idle_label",
                        "Потяните на изображении для измерения"
                        if self._ui_language == "ru"
                        else "Drag on the image to measure",
                    )
                )
            else:
                self.ruler_status_label.clear()
            return
        self.ruler_status_label.setText(text)

    def _retranslate_editor_mode_combos(self) -> None:
        polygon_mode = self.polygon_mode_combo.currentData()
        brush_mode = self.brush_mode_combo.currentData()
        delete_mode = self.delete_vertex_mode_combo.currentData()

        self.polygon_mode_combo.setItemText(0, self._mode_text("polygon_points"))
        self.polygon_mode_combo.setItemText(1, self._mode_text("polygon_rectangle"))
        if self.brush_mode_combo.count() > 0:
            self.brush_mode_combo.setItemText(0, self._mode_text("brush_freeform"))
        if self.brush_mode_combo.count() > 1:
            self.brush_mode_combo.setItemText(1, self._mode_text("brush_45deg"))
        self.delete_vertex_mode_combo.setItemText(0, self._mode_text("delete_single"))
        self.delete_vertex_mode_combo.setItemText(1, self._mode_text("delete_area"))

        polygon_index = self.polygon_mode_combo.findData(polygon_mode)
        brush_index = self.brush_mode_combo.findData(brush_mode)
        delete_index = self.delete_vertex_mode_combo.findData(delete_mode)
        if polygon_index >= 0:
            self.polygon_mode_combo.setCurrentIndex(polygon_index)
        if brush_index >= 0:
            self.brush_mode_combo.setCurrentIndex(brush_index)
        elif self.brush_mode_combo.count() > 0:
            self.brush_mode_combo.setCurrentIndex(0)
        if delete_index >= 0:
            self.delete_vertex_mode_combo.setCurrentIndex(delete_index)

        self._on_effective_polygon_create_mode_changed(self.polygon_editor.effective_polygon_create_mode())

    def _retranslate_contour_mode_combos(self) -> None:
        current_retrieval = self.retrieval_mode_combo.currentData()
        for index in range(self.retrieval_mode_combo.count()):
            mode_name = str(self.retrieval_mode_combo.itemData(index))
            self.retrieval_mode_combo.setItemText(index, self._tr(f"retrieval_mode.{mode_name}", default=mode_name))
        if current_retrieval is not None:
            self.retrieval_mode_combo.setCurrentIndex(self.retrieval_mode_combo.findData(current_retrieval))

        current_approximation = self.approximation_mode_combo.currentData()
        for index in range(self.approximation_mode_combo.count()):
            mode_name = str(self.approximation_mode_combo.itemData(index))
            self.approximation_mode_combo.setItemText(
                index,
                self._tr(f"approximation_mode.{mode_name}", default=mode_name),
            )
        if current_approximation is not None:
            self.approximation_mode_combo.setCurrentIndex(self.approximation_mode_combo.findData(current_approximation))

    def _wrap_group(self, title: str, widget: QWidget) -> QWidget:
        group = QGroupBox(title)
        layout = QVBoxLayout(group)
        layout.addWidget(widget)
        return group

    def _build_checkbox_spin_row(self, checkbox: QCheckBox, spinbox: QAbstractSpinBox) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(checkbox, 1)
        layout.addWidget(spinbox)
        return widget

    def _build_checkbox_range_row(
        self,
        checkbox: QCheckBox,
        min_spinbox: QAbstractSpinBox,
        max_spinbox: QAbstractSpinBox,
    ) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(checkbox, 1)
        layout.addWidget(min_spinbox)
        layout.addWidget(max_spinbox)
        return widget

    def _build_range_row(self, min_spinbox: QAbstractSpinBox, max_spinbox: QAbstractSpinBox) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(min_spinbox)
        layout.addWidget(max_spinbox)
        return widget

    def _configure_icon_only_button(self, button: QPushButton, icon: QIcon) -> None:
        button.setText("")
        button.setIcon(icon)
        button.setIconSize(QSize(20, 20))
        button.setFixedWidth(36)
        button.setMinimumHeight(30)

    def _refresh_files_icon(self) -> QIcon:
        pixmap = QPixmap(24, 24)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        pen = QPen(QColor("#22C55E"), 2.4)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawArc(5, 5, 14, 14, 35 * 16, 285 * 16)
        arrow = QPolygonF([QPointF(18.0, 5.0), QPointF(18.2, 11.0), QPointF(13.2, 8.0)])
        painter.setBrush(QColor("#22C55E"))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPolygon(arrow)
        painter.end()
        return QIcon(pixmap)

    def _configure_compact_form(self, form: QFormLayout) -> None:
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        form.setVerticalSpacing(2)
        form.setHorizontalSpacing(6)

    def _disable_spinbox_wheel_changes(self) -> None:
        for spinbox in self.findChildren(QAbstractSpinBox):
            spinbox.installEventFilter(self)

    def _register_spinbox(self, spinbox: QAbstractSpinBox) -> None:
        spinbox.installEventFilter(self)

    def eventFilter(self, watched, event) -> bool:
        if isinstance(watched, QAbstractSpinBox) and event.type() == QEvent.Type.Wheel:
            event.ignore()
            return True
        return super().eventFilter(watched, event)

    def _build_color_button(self, color: str, handler) -> QPushButton:
        button = QPushButton(color)
        button.clicked.connect(handler)
        self._update_color_button(button, color)
        return button

    def _update_color_button(self, button: QPushButton, color_value: str) -> None:
        button.setText(color_value)
        button.setStyleSheet(f"background-color: {color_value}; color: #111111;")


