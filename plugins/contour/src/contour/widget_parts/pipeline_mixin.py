from __future__ import annotations

from ._imports import *  # noqa: F403


class WidgetPipelineMixin:
    def _populate_pipeline_operations(self) -> None:
        selected_operation = self._selected_available_operation_name()
        self.operation_tree.clear()
        for _group_key, labels, operations in PIPELINE_OPERATION_GROUPS:
            group_item = QTreeWidgetItem([labels[0] if self._ui_language == "ru" else labels[1]])
            group_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            for operation_name in operations:
                child_item = QTreeWidgetItem([get_operation_display_name(operation_name, self._ui_language)])
                child_item.setData(0, Qt.ItemDataRole.UserRole, operation_name)
                summary, use_case = self._operation_help_entry(operation_name)
                child_item.setToolTip(
                    0,
                    f"{summary}\n\n"
                    + (("Когда использовать: " if self._ui_language == "ru" else "When to use: ") + use_case),
                )
                group_item.addChild(child_item)
            group_item.setExpanded(True)
            self.operation_tree.addTopLevelItem(group_item)
        target_operation = selected_operation or self._all_operation_names()[0]
        target_item = self._find_operation_tree_item(target_operation)
        if target_item is not None:
            self.operation_tree.setCurrentItem(target_item)
            self._update_pipeline_help_preview(target_operation)

    def _populate_pipeline_list(self) -> None:
        self._ignore_pipeline_item_change = True
        self.pipeline_list.clear()
        for step in self._pipeline.steps:
            label = get_operation_display_name(step.operation, self._ui_language)
            item = QListWidgetItem(label)
            item.setFlags(
                item.flags()
                | Qt.ItemFlag.ItemIsUserCheckable
                | Qt.ItemFlag.ItemIsSelectable
                | Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsDragEnabled
            )
            item.setData(Qt.ItemDataRole.UserRole, self.pipeline_list.count())
            item.setData(Qt.ItemDataRole.UserRole + 1, step.operation)
            item.setCheckState(Qt.CheckState.Checked if step.enabled else Qt.CheckState.Unchecked)
            self.pipeline_list.addItem(item)
        self._ignore_pipeline_item_change = False
        if self.pipeline_list.count():
            self.pipeline_list.setCurrentRow(0)
            self._render_pipeline_parameters(0)
        else:
            self._clear_parameters_form()

    def _clear_parameters_form(self) -> None:
        while self.parameters_form.count():
            item = self.parameters_form.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._parameter_widgets.clear()

    def _on_pipeline_step_selected(self, row: int) -> None:
        self._render_pipeline_parameters(row)

    def _render_pipeline_parameters(self, row: int) -> None:
        self._clear_parameters_form()
        if row < 0 or row >= len(self._pipeline.steps):
            self._set_color_pick_active(None)
            return
        step = self._pipeline.steps[row]
        descriptor = get_operation_descriptor(step.operation)
        for spec in descriptor.parameters:
            value = step.parameters.get(spec.name, spec.default)
            if spec.kind == "bool":
                widget = QCheckBox()
                widget.setChecked(bool(value))
                widget.stateChanged.connect(
                    lambda _state, name=spec.name, row_index=row, w=widget: self._update_step_parameter(
                        row_index, name, w.isChecked()
                    )
                )
            elif spec.kind == "choice":
                widget = QComboBox()
                self._register_no_wheel_value_widget(widget)
                for option in spec.options:
                    widget.addItem(get_choice_display_label(spec.name, str(option), self._ui_language), option)
                selected_index = widget.findData(value)
                if selected_index >= 0:
                    widget.setCurrentIndex(selected_index)
                widget.currentIndexChanged.connect(
                    lambda _index, name=spec.name, row_index=row, w=widget: self._update_step_parameter(
                        row_index,
                        name,
                        w.currentData(),
                    )
                )
            elif spec.kind == "int":
                widget = QSpinBox()
                self._register_spinbox(widget)
                widget.setRange(int(spec.minimum or -1_000_000), int(spec.maximum or 1_000_000))
                widget.setSingleStep(int(spec.step or 1))
                widget.setValue(int(value))
                widget.valueChanged.connect(
                    lambda new_value, name=spec.name, row_index=row: self._update_step_parameter(
                        row_index, name, int(new_value)
                    )
                )
            else:
                widget = QDoubleSpinBox()
                self._register_spinbox(widget)
                widget.setDecimals(spec.decimals)
                widget.setRange(float(spec.minimum or -1_000_000), float(spec.maximum or 1_000_000))
                widget.setSingleStep(float(spec.step or 0.1))
                widget.setValue(float(value))
                widget.valueChanged.connect(
                    lambda new_value, name=spec.name, row_index=row: self._update_step_parameter(
                        row_index, name, float(new_value)
                    )
                )
            tooltip = spec.tooltip or self._pipeline_parameter_tooltip(step.operation, spec.name)
            widget.setToolTip(tooltip)
            self._parameter_widgets[spec.name] = widget
            label_widget = QLabel(get_parameter_display_label(spec, self._ui_language))
            label_widget.setToolTip(tooltip)
            self.parameters_form.addRow(label_widget, widget)
        if step.operation == "color_binarize":
            self._render_color_binarize_parameters(row)
        else:
            self._set_color_pick_active(None)

    def _update_step_parameter(self, row: int, parameter_name: str, value) -> None:
        if row < 0 or row >= len(self._pipeline.steps):
            return
        self._pipeline.steps[row].parameters[parameter_name] = value
        self._auto_apply_pipeline()

    def _color_selection_entries(self, row: int) -> list[dict[str, object]]:
        if row < 0 or row >= len(self._pipeline.steps):
            return []
        entries = self._pipeline.steps[row].parameters.get("selected_colors", [])
        if not isinstance(entries, list):
            entries = []
        normalized: list[dict[str, object]] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            rgb = entry.get("rgb")
            if not isinstance(rgb, (list, tuple)) or len(rgb) != 3:
                continue
            try:
                parsed_rgb = [max(0, min(255, int(channel))) for channel in rgb]
            except (TypeError, ValueError):
                continue
            normalized.append({"rgb": parsed_rgb, "enabled": bool(entry.get("enabled", True))})
        self._pipeline.steps[row].parameters["selected_colors"] = normalized
        return normalized

    def _render_color_binarize_parameters(self, row: int) -> None:
        entries = self._color_selection_entries(row)
        group = QGroupBox(
            self._tr(
                "color_binarize_group_title",
                "Цвета для бинаризации" if self._ui_language == "ru" else "Colors for binarization",
            )
        )
        layout = QVBoxLayout(group)
        hint = QLabel(
            self._tr(
                "color_binarize_hint",
                "Включите выбор и кликните по изображению, чтобы добавить цвет. Галочкой можно временно отключить цвет."
                if self._ui_language == "ru"
                else "Enable picking and click the image to add a color. Uncheck an item to disable it temporarily.",
            )
        )
        hint.setWordWrap(True)
        hint.setToolTip(
            "Цвета из списка используются для построения бинарной маски; допуск задается параметром delta."
            if self._ui_language == "ru"
            else "Colors in the list are used to build the binary mask; tolerance is controlled by delta."
        )
        layout.addWidget(hint)
        color_list = QListWidget()
        color_list.setToolTip(
            "Отмеченные цвета участвуют в бинаризации. Снимите галочку, чтобы временно исключить цвет из маски."
            if self._ui_language == "ru"
            else "Checked colors participate in binarization. Uncheck a color to temporarily exclude it from the mask."
        )
        color_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        for entry in entries:
            rgb = entry["rgb"]
            item = QListWidgetItem(f"#{int(rgb[0]):02X}{int(rgb[1]):02X}{int(rgb[2]):02X}")
            item.setToolTip(
                "Этот цвет добавляет похожие пиксели в маску; галочка включает или выключает его."
                if self._ui_language == "ru"
                else "This color adds similar pixels to the mask; the checkbox enables or disables it."
            )
            item.setData(Qt.ItemDataRole.UserRole, list(rgb))
            item.setFlags(
                item.flags()
                | Qt.ItemFlag.ItemIsUserCheckable
                | Qt.ItemFlag.ItemIsSelectable
                | Qt.ItemFlag.ItemIsEnabled
            )
            item.setCheckState(Qt.CheckState.Checked if entry.get("enabled", True) else Qt.CheckState.Unchecked)
            item.setBackground(QColor(int(rgb[0]), int(rgb[1]), int(rgb[2])))
            brightness = int(rgb[0]) * 0.299 + int(rgb[1]) * 0.587 + int(rgb[2]) * 0.114
            item.setForeground(QColor("#111111" if brightness > 150 else "#F8FAFC"))
            color_list.addItem(item)
        color_list.itemChanged.connect(
            lambda item, row_index=row, widget=color_list: self._on_color_entry_changed(row_index, widget, item)
        )
        layout.addWidget(color_list)

        buttons_row = QWidget()
        buttons_layout = QHBoxLayout(buttons_row)
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        pick_button = QPushButton(
            self._tr("pick_colors_button", "Выбор с изображения" if self._ui_language == "ru" else "Pick from image")
        )
        pick_button.setCheckable(True)
        pick_button.setToolTip(
            "Включает выбор цвета с изображения: кликните по нужному пикселю, чтобы добавить его в список."
            if self._ui_language == "ru"
            else "Enables picking from the image: click a pixel to add its color to the list."
        )
        pick_button.setChecked(self._color_pick_pipeline_row == row)
        pick_button.toggled.connect(
            lambda checked, row_index=row: self._set_color_pick_active(row_index if checked else None)
        )
        remove_button = QPushButton(
            self._tr(
                "remove_selected_color_button", "Удалить выбранный" if self._ui_language == "ru" else "Remove selected"
            )
        )
        remove_button.setToolTip(
            "Удаляет выбранный цвет из списка бинаризации."
            if self._ui_language == "ru"
            else "Removes the selected color from the binarization list."
        )
        remove_button.clicked.connect(
            lambda _checked=False, row_index=row, widget=color_list: self._remove_selected_color_entry(
                row_index, widget
            )
        )
        clear_button = QPushButton(
            self._tr("clear_colors_button", "Очистить список" if self._ui_language == "ru" else "Clear list")
        )
        clear_button.setToolTip(
            "Очищает весь список цветов для этого шага бинаризации."
            if self._ui_language == "ru"
            else "Clears the whole color list for this binarization step."
        )
        clear_button.clicked.connect(lambda _checked=False, row_index=row: self._clear_color_entries(row_index))
        buttons_layout.addWidget(pick_button)
        buttons_layout.addWidget(remove_button)
        buttons_layout.addWidget(clear_button)
        layout.addWidget(buttons_row)
        self.parameters_form.addRow(group)

    def _on_color_entry_changed(self, row: int, color_list: QListWidget, item: QListWidgetItem) -> None:
        entries = self._color_selection_entries(row)
        index = color_list.row(item)
        if index < 0 or index >= len(entries):
            return
        entries[index]["enabled"] = item.checkState() == Qt.CheckState.Checked
        self._pipeline.steps[row].parameters["selected_colors"] = entries
        self._auto_apply_pipeline()

    def _remove_selected_color_entry(self, row: int, color_list: QListWidget) -> None:
        index = color_list.currentRow()
        if index < 0:
            return
        entries = self._color_selection_entries(row)
        if index >= len(entries):
            return
        entries.pop(index)
        self._pipeline.steps[row].parameters["selected_colors"] = entries
        self._render_pipeline_parameters(row)
        self._auto_apply_pipeline()

    def _clear_color_entries(self, row: int) -> None:
        if row < 0 or row >= len(self._pipeline.steps):
            return
        self._pipeline.steps[row].parameters["selected_colors"] = []
        self._render_pipeline_parameters(row)
        self._auto_apply_pipeline()

    def _set_color_pick_active(self, row: int | None) -> None:
        self._color_pick_pipeline_row = row
        if hasattr(self, "polygon_editor"):
            self.polygon_editor.set_image_click_mode(row is not None)

    def _set_via_template_pick_active(self, enabled: bool) -> None:
        if enabled:
            self._set_color_pick_active(None)
        if hasattr(self, "polygon_editor"):
            self.polygon_editor.set_image_region_selection_mode(enabled)

    def _refresh_via_template_list(self) -> None:
        if not hasattr(self, "via_template_table"):
            return
        self._ensure_via_template_metadata()
        table = self.via_template_table
        table.clearSpans()
        table.setRowCount(0)
        if not self._via_template_images:
            table.insertRow(0)
            table.setRowHeight(0, 52)
            table.setSpan(0, 0, 1, 5)
            empty_label = QLabel(
                "Чтобы добавить шаблон удерживайте Ctrl и выделите область"
                if self._ui_language == "ru"
                else "Hold Ctrl and select an area to add a template"
            )
            empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            empty_label.setWordWrap(True)
            table.setCellWidget(0, 0, empty_label)
        for index, template in enumerate(self._via_template_images, start=1):
            row = index - 1
            table.insertRow(row)
            table.setRowHeight(row, 64)

            order_spin = QSpinBox()
            order_spin.setRange(1, len(self._via_template_images))
            order_spin.setValue(index)
            order_spin.setMinimumWidth(0)
            order_spin.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            order_spin.valueChanged.connect(
                lambda value, source_row=row: self._move_via_template(source_row, int(value) - 1)
            )
            table.setCellWidget(row, 0, order_spin)

            preview_pixmap = QPixmap.fromImage(cv_to_qimage(template)).scaled(
                56,
                56,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            preview_label = QLabel()
            preview_label.setPixmap(preview_pixmap)
            preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            preview_label.setFixedSize(60, 60)
            table.setCellWidget(row, 1, preview_label)

            similarity_spin = QDoubleSpinBox()
            similarity_spin.setRange(0.0, 1.0)
            similarity_spin.setDecimals(3)
            similarity_spin.setSingleStep(0.01)
            similarity_spin.setValue(self._via_template_min_scores[row])
            similarity_spin.setMinimumWidth(0)
            similarity_spin.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            similarity_spin.valueChanged.connect(
                lambda value, template_row=row: self._set_via_template_similarity(template_row, float(value))
            )
            table.setCellWidget(row, 2, similarity_spin)

            diameter_spin = QSpinBox()
            diameter_spin.setRange(1, 10_000)
            diameter_spin.setSuffix(" px")
            diameter_spin.setValue(self._via_template_diameters[row])
            diameter_spin.setMinimumWidth(0)
            diameter_spin.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            diameter_spin.valueChanged.connect(
                lambda value, template_row=row: self._set_via_template_diameter(template_row, int(value))
            )
            table.setCellWidget(row, 3, diameter_spin)

            remove_button = QPushButton("-")
            remove_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            remove_button.setMinimumSize(0, 0)
            remove_button.setStyleSheet(
                "QPushButton { background-color: #DC2626; color: white; font-weight: 700; "
                "border-radius: 0; padding: 0; }"
            )
            remove_button.setToolTip("Удалить шаблон" if self._ui_language == "ru" else "Remove template")
            remove_button.clicked.connect(
                lambda _checked=False, template_row=row: self._remove_via_template(template_row)
            )
            table.setCellWidget(row, 4, remove_button)
        table.setColumnWidth(1, 64)
        visible_rows = max(1, min(5, len(self._via_template_images)))
        row_height = 64 if self._via_template_images else 52
        header_height = max(24, table.horizontalHeader().sizeHint().height())
        table_height = header_height + visible_rows * row_height + table.frameWidth() * 2 + 2
        table.setMinimumHeight(table_height)
        table.setMaximumHeight(table_height)

    def _ensure_via_template_metadata(self) -> None:
        count = len(self._via_template_images)
        while len(self._via_template_min_scores) < count:
            fallback_score = self._via_template_min_scores[-1] if self._via_template_min_scores else 0.6
            self._via_template_min_scores.append(float(fallback_score))
        while len(self._via_template_diameters) < count:
            fallback_diameter = self._via_template_diameters[-1] if self._via_template_diameters else 8
            self._via_template_diameters.append(max(1, int(fallback_diameter)))
        del self._via_template_min_scores[count:]
        del self._via_template_diameters[count:]

    def _move_via_template(self, source_row: int, target_row: int) -> None:
        count = len(self._via_template_images)
        if source_row < 0 or source_row >= count:
            return
        target_row = max(0, min(count - 1, int(target_row)))
        if source_row == target_row:
            return
        self._ensure_via_template_metadata()
        for values in (self._via_template_images, self._via_template_min_scores, self._via_template_diameters):
            values.insert(target_row, values.pop(source_row))
        self._refresh_via_template_list()
        self._on_extraction_settings_changed()

    def _set_via_template_similarity(self, row: int, value: float) -> None:
        if 0 <= row < len(self._via_template_min_scores):
            self._via_template_min_scores[row] = max(0.0, min(1.0, float(value)))
            self._on_extraction_settings_changed()

    def _set_via_template_diameter(self, row: int, value: int) -> None:
        if 0 <= row < len(self._via_template_diameters):
            self._via_template_diameters[row] = max(1, int(value))
            self._on_extraction_settings_changed()

    def _normalize_via_template_images(self, payload: list[object]) -> list[np.ndarray]:
        templates: list[np.ndarray] = []
        for item in payload:
            try:
                image = np.asarray(item, dtype=np.uint8)
            except (TypeError, ValueError):
                continue
            if image.ndim == 3:
                if image.shape[2] >= 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    image = image[:, :, 0]
            if image.ndim != 2 or image.shape[0] < 2 or image.shape[1] < 2:
                continue
            templates.append(image.copy())
        return templates

    def _on_editor_image_region_selected(self, x_coord: float, y_coord: float, width: float, height: float) -> None:
        if hasattr(self, "polygon_editor"):
            self.polygon_editor.set_image_region_selection_mode(False)
        image = self._workspace.current_display_image()
        if image is None:
            return
        data = np.asarray(image)
        if data.size == 0:
            return
        left = max(0, int(np.floor(x_coord)))
        top = max(0, int(np.floor(y_coord)))
        right = min(data.shape[1], int(np.ceil(x_coord + width)))
        bottom = min(data.shape[0], int(np.ceil(y_coord + height)))
        if right - left < 2 or bottom - top < 2:
            return
        template = data[top:bottom, left:right].copy()
        if template.ndim == 3:
            if template.shape[2] >= 3:
                template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            else:
                template = template[:, :, 0]
        self._ensure_via_template_metadata()
        similarity = self._via_template_min_scores[-1] if self._via_template_min_scores else 0.6
        diameter = self._via_template_diameters[-1] if self._via_template_diameters else 8
        self._via_template_images.append(template.astype(np.uint8, copy=False))
        self._via_template_min_scores.append(float(similarity))
        self._via_template_diameters.append(max(1, int(diameter)))
        self._refresh_via_template_list()
        self._on_extraction_settings_changed()
        self._append_log(
            self._tr(
                "via_template_added_log",
                "Добавлен шаблон контакта {width}x{height}. Всего шаблонов: {count}."
                if self._ui_language == "ru"
                else "Added contact template {width}x{height}. Total templates: {count}.",
                width=right - left,
                height=bottom - top,
                count=len(self._via_template_images),
            )
        )

    def _clear_via_templates(self, *_args) -> None:
        self._via_template_images.clear()
        self._via_template_min_scores.clear()
        self._via_template_diameters.clear()
        self._refresh_via_template_list()
        self._on_extraction_settings_changed()

    def _remove_via_template(self, row: int) -> None:
        if row < 0 or row >= len(self._via_template_images):
            return
        self._ensure_via_template_metadata()
        self._via_template_images.pop(row)
        self._via_template_min_scores.pop(row)
        self._via_template_diameters.pop(row)
        self._refresh_via_template_list()
        self._on_extraction_settings_changed()

    def _remove_selected_via_template(self, *_args) -> None:
        row = self.via_template_table.currentRow() if hasattr(self, "via_template_table") else -1
        self._remove_via_template(row)

    def _built_in_via_presets(self) -> dict[str, dict[str, object]]:
        return built_in_via_presets(self._ui_language)

    def _noisy_traces_via_preset_payload(self) -> dict[str, object]:
        return noisy_traces_via_preset_payload()

    def _blurred_via_preset_payload(self) -> dict[str, object]:
        return blurred_via_preset_payload()

    def _load_user_via_presets(self) -> dict[str, dict[str, object]]:
        return {
            name: {
                str(key): value
                for key, value in payload.items()
                if str(key).startswith("heuristic_")
            }
            for name, payload in self._via_preset_settings_store.load().items()
        }

    def _save_user_via_presets(self) -> None:
        self._via_preset_settings_store.save(self._user_via_presets)

    def _refresh_via_preset_combo(self) -> None:
        if not hasattr(self, "via_preset_combo"):
            return
        current_name = self.via_preset_combo.currentText()
        self.via_preset_combo.clear()
        for name in self._built_in_via_presets():
            self.via_preset_combo.addItem(name, ("builtin", name))
        for name in sorted(self._user_via_presets):
            self.via_preset_combo.addItem(name, ("user", name))
        index = self.via_preset_combo.findText(current_name)
        if index >= 0:
            self.via_preset_combo.setCurrentIndex(index)

    def _current_via_preset_payload(self) -> dict[str, object]:
        payload = self._current_contour_settings().to_dict()
        return {
            key: value
            for key, value in payload.items()
            if key.startswith("heuristic_")
        }

    def _apply_via_preset_payload(self, payload: dict[str, object]) -> None:
        # A via preset is deliberately limited to the expert detector
        # parameters. Method, polarity, sizes, sensitivity, templates and
        # display controls belong to the recognition configuration itself.
        payload = {
            key: value
            for key, value in payload.items()
            if str(key).startswith("heuristic_")
        }
        if not payload:
            return
        widget_names = {
            "heuristic_line_penalty_scale": "heuristic_line_penalty_spin",
            "heuristic_border_penalty_scale": "heuristic_border_penalty_spin",
            "heuristic_use_bilateral": "heuristic_use_bilateral_checkbox",
        }
        preset_widgets = {
            key: getattr(
                self,
                widget_names.get(key, f"{key}_spin"),
                None,
            )
            for key in payload
        }
        preset_widgets = {
            key: widget
            for key, widget in preset_widgets.items()
            if widget is not None
        }
        preset_blockers = [QSignalBlocker(widget) for widget in preset_widgets.values()]
        try:
            for key, widget in preset_widgets.items():
                value = payload[key]
                if isinstance(widget, QCheckBox):
                    widget.setChecked(bool(value))
                else:
                    widget.setValue(float(value))
        finally:
            del preset_blockers
        if hasattr(self, "bright_via_advanced_outer"):
            self.bright_via_advanced_outer.setChecked(True)
        self._on_extraction_settings_changed()
        return

        blockers = [
            QSignalBlocker(self.via_search_mode_combo),
            QSignalBlocker(self.via_white_range_checkbox),
            QSignalBlocker(self.via_white_range_min_spin),
            QSignalBlocker(self.via_white_range_max_spin),
            QSignalBlocker(self.via_black_range_checkbox),
            QSignalBlocker(self.via_black_range_min_spin),
            QSignalBlocker(self.via_black_range_max_spin),
            QSignalBlocker(self.via_min_score_spin),
            QSignalBlocker(self.via_min_contrast_spin),
            QSignalBlocker(self.via_min_edge_coverage_spin),
            QSignalBlocker(self.via_spot_line_suppression_spin),
            QSignalBlocker(self.via_template_min_score_spin),
            QSignalBlocker(self.bright_via_diameter_min_spin),
            QSignalBlocker(self.bright_via_diameter_max_spin),
            QSignalBlocker(self.bright_via_clahe_clip_spin),
            QSignalBlocker(self.bright_via_clahe_tile_spin),
            QSignalBlocker(self.bright_via_median_kernel_spin),
            QSignalBlocker(self.bright_via_tophat_kernel_spin),
            QSignalBlocker(self.bright_via_dog_small_spin),
            QSignalBlocker(self.bright_via_dog_large_spin),
            QSignalBlocker(self.bright_via_threshold_percentile_spin),
            QSignalBlocker(self.bright_via_mask_combine_combo),
            QSignalBlocker(self.bright_via_min_area_factor_spin),
            QSignalBlocker(self.bright_via_max_area_factor_spin),
            QSignalBlocker(self.bright_via_min_circularity_spin),
            QSignalBlocker(self.bright_via_min_aspect_spin),
            QSignalBlocker(self.bright_via_max_aspect_spin),
            QSignalBlocker(self.bright_via_bright_center_score_spin),
            QSignalBlocker(self.bright_via_metal_constraint_combo),
            QSignalBlocker(self.bright_via_metal_fraction_spin),
            QSignalBlocker(self.bright_via_max_radial_asymmetry_spin),
            QSignalBlocker(self.bright_via_max_edge_likeness_spin),
            QSignalBlocker(self.bright_via_max_line_likeness_spin),
            QSignalBlocker(self.bright_via_nms_distance_spin),
            QSignalBlocker(self.bright_via_min_final_score_spin),
            QSignalBlocker(self.bright_via_show_rejected_checkbox),
            QSignalBlocker(self.bright_via_hard_asym_checkbox),
            QSignalBlocker(self.bright_via_hard_edge_checkbox),
            QSignalBlocker(self.bright_via_hard_line_checkbox),
            QSignalBlocker(self.debug_candidates_checkbox),
            QSignalBlocker(self.via_roundness_spin),
        ]
        heuristic_widgets = {
            "heuristic_background_sigma": "heuristic_background_sigma_spin",
            "heuristic_analysis_window_scale": "heuristic_analysis_window_scale_spin",
            "heuristic_min_center_brightness": "heuristic_min_center_brightness_spin",
            "heuristic_min_center_contrast": "heuristic_min_center_contrast_spin",
            "heuristic_min_peak_prominence": "heuristic_min_peak_prominence_spin",
            "heuristic_min_compactness": "heuristic_min_compactness_spin",
            "heuristic_min_circularity": "heuristic_min_circularity_spin",
            "heuristic_max_elongation": "heuristic_max_elongation_spin",
            "heuristic_line_penalty_scale": "heuristic_line_penalty_spin",
            "heuristic_border_penalty_scale": "heuristic_border_penalty_spin",
            "heuristic_local_binarize_percentile": "heuristic_local_binarize_percentile_spin",
            "heuristic_min_abs_peak": "heuristic_min_abs_peak_spin",
            **{
                name: f"{name}_spin"
                for name in payload
                if name.startswith("heuristic_") and hasattr(self, f"{name}_spin")
            },
        }
        blockers.extend(
            QSignalBlocker(widget)
            for widget_name in heuristic_widgets.values()
            if (widget := getattr(self, widget_name, None)) is not None
        )
        try:
            mode_index = self.via_search_mode_combo.findData(
                normalize_via_search_mode(payload.get("via_search_mode", self.via_search_mode_combo.currentData()))
            )
            if mode_index >= 0:
                self.via_search_mode_combo.setCurrentIndex(mode_index)
            self.via_white_range_checkbox.setChecked(
                bool(payload.get("via_white_range_enabled", self.via_white_range_checkbox.isChecked()))
            )
            self.via_white_range_min_spin.setValue(
                int(payload.get("via_white_range_min", self.via_white_range_min_spin.value()))
            )
            self.via_white_range_max_spin.setValue(
                int(payload.get("via_white_range_max", self.via_white_range_max_spin.value()))
            )
            self.via_black_range_checkbox.setChecked(
                bool(payload.get("via_black_range_enabled", self.via_black_range_checkbox.isChecked()))
            )
            self.via_black_range_min_spin.setValue(
                int(payload.get("via_black_range_min", self.via_black_range_min_spin.value()))
            )
            self.via_black_range_max_spin.setValue(
                int(payload.get("via_black_range_max", self.via_black_range_max_spin.value()))
            )
            self.via_min_score_spin.setValue(float(payload.get("via_min_score", self.via_min_score_spin.value())))
            self.via_min_contrast_spin.setValue(
                float(payload.get("via_min_contrast", self.via_min_contrast_spin.value()))
            )
            self.via_min_edge_coverage_spin.setValue(
                float(payload.get("via_min_edge_coverage", self.via_min_edge_coverage_spin.value()))
            )
            self.via_spot_line_suppression_spin.setValue(
                float(payload.get("via_spot_line_suppression", self.via_spot_line_suppression_spin.value()))
            )
            self.via_template_min_score_spin.setValue(
                float(payload.get("via_template_min_score", self.via_template_min_score_spin.value()))
            )
            self.via_roundness_spin.setValue(float(payload.get("via_min_roundness", self.via_roundness_spin.value())))
            self.bright_via_diameter_min_spin.setValue(
                int(payload.get("bright_via_diameter_min", self.bright_via_diameter_min_spin.value()))
            )
            self.bright_via_diameter_max_spin.setValue(
                int(payload.get("bright_via_diameter_max", self.bright_via_diameter_max_spin.value()))
            )
            self.bright_via_clahe_clip_spin.setValue(
                float(payload.get("bright_via_clahe_clip_limit", self.bright_via_clahe_clip_spin.value()))
            )
            self.bright_via_clahe_tile_spin.setValue(
                int(payload.get("bright_via_clahe_tile_grid_size", self.bright_via_clahe_tile_spin.value()))
            )
            self.bright_via_median_kernel_spin.setValue(
                int(payload.get("bright_via_median_blur_kernel", self.bright_via_median_kernel_spin.value()))
            )
            self.bright_via_tophat_kernel_spin.setValue(
                int(payload.get("bright_via_tophat_kernel_size", self.bright_via_tophat_kernel_spin.value()))
            )
            self.bright_via_dog_small_spin.setValue(
                float(payload.get("bright_via_dog_sigma_small", self.bright_via_dog_small_spin.value()))
            )
            self.bright_via_dog_large_spin.setValue(
                float(payload.get("bright_via_dog_sigma_large", self.bright_via_dog_large_spin.value()))
            )
            self.bright_via_threshold_percentile_spin.setValue(
                float(payload.get("bright_via_threshold_percentile", self.bright_via_threshold_percentile_spin.value()))
            )
            combine_index = self.bright_via_mask_combine_combo.findData(
                str(payload.get("bright_via_mask_combine_mode", self.bright_via_mask_combine_combo.currentData()))
            )
            if combine_index >= 0:
                self.bright_via_mask_combine_combo.setCurrentIndex(combine_index)
            self.bright_via_min_area_factor_spin.setValue(
                float(payload.get("bright_via_min_area_factor", self.bright_via_min_area_factor_spin.value()))
            )
            self.bright_via_max_area_factor_spin.setValue(
                float(payload.get("bright_via_max_area_factor", self.bright_via_max_area_factor_spin.value()))
            )
            self.bright_via_min_circularity_spin.setValue(
                float(payload.get("bright_via_min_circularity", self.bright_via_min_circularity_spin.value()))
            )
            self.bright_via_min_aspect_spin.setValue(
                float(payload.get("bright_via_min_aspect", self.bright_via_min_aspect_spin.value()))
            )
            self.bright_via_max_aspect_spin.setValue(
                float(payload.get("bright_via_max_aspect", self.bright_via_max_aspect_spin.value()))
            )
            self.bright_via_bright_center_score_spin.setValue(
                float(
                    payload.get(
                        "bright_via_bright_center_min_score",
                        self.bright_via_bright_center_score_spin.value(),
                    )
                )
            )
            metal_mode = _normalize_bright_via_metal_constraint_mode(
                payload.get("bright_via_metal_constraint_mode", self.bright_via_metal_constraint_combo.currentData())
            )
            metal_index = self.bright_via_metal_constraint_combo.findData(metal_mode)
            if metal_index >= 0:
                self.bright_via_metal_constraint_combo.setCurrentIndex(metal_index)
            self.bright_via_metal_fraction_spin.setValue(
                float(payload.get("bright_via_metal_fraction_min", self.bright_via_metal_fraction_spin.value()))
            )
            self.bright_via_max_radial_asymmetry_spin.setValue(
                float(
                    payload.get(
                        "bright_via_max_radial_asymmetry",
                        self.bright_via_max_radial_asymmetry_spin.value(),
                    )
                )
            )
            self.bright_via_max_edge_likeness_spin.setValue(
                float(payload.get("bright_via_max_edge_likeness", self.bright_via_max_edge_likeness_spin.value()))
            )
            self.bright_via_max_line_likeness_spin.setValue(
                float(payload.get("bright_via_max_line_likeness", self.bright_via_max_line_likeness_spin.value()))
            )
            self.bright_via_nms_distance_spin.setValue(
                int(payload.get("bright_via_nms_distance", self.bright_via_nms_distance_spin.value()))
            )
            self.bright_via_min_final_score_spin.setValue(
                float(payload.get("bright_via_min_final_score", self.bright_via_min_final_score_spin.value()))
            )
            self.bright_via_show_rejected_checkbox.setChecked(
                bool(payload.get("bright_via_show_rejected", self.bright_via_show_rejected_checkbox.isChecked()))
            )
            self.bright_via_hard_asym_checkbox.setChecked(
                bool(payload.get("bright_via_hard_reject_on_asymmetry", self.bright_via_hard_asym_checkbox.isChecked()))
            )
            self.bright_via_hard_edge_checkbox.setChecked(
                bool(payload.get("bright_via_hard_reject_on_edge", self.bright_via_hard_edge_checkbox.isChecked()))
            )
            self.bright_via_hard_line_checkbox.setChecked(
                bool(payload.get("bright_via_hard_reject_on_line", self.bright_via_hard_line_checkbox.isChecked()))
            )
            self.debug_candidates_checkbox.setChecked(
                bool(payload.get("debug_enabled", self.debug_candidates_checkbox.isChecked()))
            )
            if hasattr(self, "via_heuristic_polarity_combo") and "via_heuristic_polarity" in payload:
                polarity_index = self.via_heuristic_polarity_combo.findData(str(payload["via_heuristic_polarity"]))
                if polarity_index >= 0:
                    self.via_heuristic_polarity_combo.setCurrentIndex(polarity_index)
            for setting_name, widget_name in heuristic_widgets.items():
                widget = getattr(self, widget_name, None)
                if widget is not None and setting_name in payload:
                    widget.setValue(float(payload[setting_name]))
            if hasattr(self, "heuristic_use_bilateral_checkbox") and "heuristic_use_bilateral" in payload:
                self.heuristic_use_bilateral_checkbox.setChecked(bool(payload["heuristic_use_bilateral"]))
        finally:
            del blockers
        if hasattr(self, "bright_via_advanced_outer"):
            self.bright_via_advanced_outer.setChecked(True)
        self._update_bright_via_diameter_controls_state()
        self._update_via_threshold_controls_state()
        self._on_extraction_settings_changed()

    def _apply_selected_via_preset(self) -> None:
        data = self.via_preset_combo.currentData()
        if not isinstance(data, tuple) or len(data) != 2:
            return
        preset_type, preset_name = data
        payload = (
            self._built_in_via_presets().get(str(preset_name))
            if preset_type == "builtin"
            else self._user_via_presets.get(str(preset_name))
        )
        if payload:
            self._apply_via_preset_payload(payload)

    def _save_current_via_preset(self) -> None:
        name, ok = QInputDialog.getText(
            self,
            "Сохранить пресет" if self._ui_language == "ru" else "Save preset",
            "Имя пресета:" if self._ui_language == "ru" else "Preset name:",
        )
        name = str(name).strip()
        if not ok or not name:
            return
        expert_payload = self._current_via_preset_payload()
        safe_name = "".join(
            character if character not in '<>:"/\\|?*' else "_"
            for character in name
        ).strip(" .") or "via_expert_preset"
        start_directory = self._dialog_start_directory_from_line_edit(self.output_dir_edit)
        suggested_path = str(Path(start_directory) / f"{safe_name}.json")
        path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Сохранить пресет контактов" if self._ui_language == "ru" else "Save contact preset",
            suggested_path,
            self._tr("json_file_filter"),
        )
        if not path:
            return
        save_pipeline_config_to_path(
            path,
            {
                "format": "contour-via-expert-preset",
                "version": 1,
                "name": name,
                "expert_settings": expert_payload,
                # Kept as a complete snapshot for audit/reproducibility. Loading
                # this preset applies expert_settings only.
                "recognition_settings": self._current_contour_settings().to_dict(),
            },
        )
        self._user_via_presets[name] = expert_payload
        self._save_user_via_presets()
        self._refresh_via_preset_combo()
        index = self.via_preset_combo.findText(name)
        if index >= 0:
            self.via_preset_combo.setCurrentIndex(index)

    def _delete_selected_via_preset(self) -> None:
        data = self.via_preset_combo.currentData()
        if not isinstance(data, tuple) or len(data) != 2 or data[0] != "user":
            return
        self._user_via_presets.pop(str(data[1]), None)
        self._save_user_via_presets()
        self._refresh_via_preset_combo()

    def _built_in_metal_presets(self) -> dict[str, dict[str, object]]:
        return built_in_metal_presets(self._ui_language)

    def _load_user_metal_presets(self) -> dict[str, dict[str, object]]:
        return self._metal_preset_settings_store.load()

    def _save_user_metal_presets(self) -> None:
        self._metal_preset_settings_store.save(self._user_metal_presets)

    def _refresh_metal_preset_combo(self) -> None:
        if not hasattr(self, "metal_preset_combo"):
            return
        current_name = self.metal_preset_combo.currentText()
        self.metal_preset_combo.clear()
        for name in self._built_in_metal_presets():
            self.metal_preset_combo.addItem(name, ("builtin", name))
        for name in sorted(self._user_metal_presets):
            self.metal_preset_combo.addItem(name, ("user", name))
        index = self.metal_preset_combo.findText(current_name)
        if index >= 0:
            self.metal_preset_combo.setCurrentIndex(index)

    def _current_metal_preset_payload(self) -> dict[str, object]:
        payload = self._current_contour_settings().to_dict()
        keys = (
            "metal_preset",
            "metal_noise_suppression",
            "metal_min_contrast",
            "metal_min_object_source_contrast",
            "metal_min_object_rim_contrast",
            "metal_min_object_rim_area_fraction",
            "metal_min_hole_source_contrast",
            "metal_min_hole_source_contrast_fraction",
            "metal_segmentation_strategy",
            "metal_strategy_parameters",
            "metal_auto_contrast_step",
            "metal_auto_source_contrast_step",
            "metal_auto_directional_gap_bridge_px",
            "metal_auto_directional_gap_min_source_intensity",
            "metal_gap_bridge_px",
            "metal_speckle_removal_px",
            "metal_contour_smooth_px",
            "metal_min_trace_width_px",
            "metal_max_trace_width_px",
            "metal_min_trace_length_px",
            "metal_min_straightness",
            "metal_min_area",
            "metal_min_perimeter",
            "metal_use_wide_conductor_gradient",
            "metal_watershed_smoothing_sigma",
            "metal_watershed_core_margin",
            "metal_watershed_groove_margin",
            "metal_watershed_rim_probe_px",
            "metal_watershed_seed_speckle_px",
            "metal_watershed_valley_span_px",
            "metal_watershed_valley_depth",
            "metal_adaptive_block_size",
            "metal_adaptive_c",
            "metal_adaptive_method",
            "metal_random_walker_beta",
            "metal_random_walker_iterations",
            "metal_graph_cut_iterations",
            "metal_reconstruction_erode_px",
            "metal_boundary_relief",
            "metal_boundary_background_sigma",
            "metal_structural_variant",
            "metal_allowed_angles",
            "metal_angle_tolerance_deg",
            "epsilon",
        )
        return {key: payload[key] for key in keys if key in payload}

    def _apply_metal_preset_payload(self, payload: dict[str, object]) -> None:
        normalized_payload = dict(payload)
        if "metal_min_contrast" not in normalized_payload and "metal_contrast_bias" in normalized_payload:
            normalized_payload["metal_min_contrast"] = max(
                1.0,
                float(normalized_payload.get("metal_contrast_bias", 0.0)),
            )
        merged = ContourExtractionSettings.from_dict(
            self._current_contour_settings().to_dict() | normalized_payload
        )
        self._set_extraction_settings(merged)
        self._on_extraction_settings_changed()

    def _apply_selected_metal_preset(self) -> None:
        data = self.metal_preset_combo.currentData()
        if not isinstance(data, tuple) or len(data) != 2:
            return
        preset_type, preset_name = data
        payload = (
            self._built_in_metal_presets().get(str(preset_name))
            if preset_type == "builtin"
            else self._user_metal_presets.get(str(preset_name))
        )
        if payload:
            self._apply_metal_preset_payload(payload)

    def _save_current_metal_preset(self) -> None:
        name, ok = QInputDialog.getText(
            self,
            "Сохранить пресет" if self._ui_language == "ru" else "Save preset",
            "Имя пресета:" if self._ui_language == "ru" else "Preset name:",
        )
        name = str(name).strip()
        if not ok or not name:
            return
        self._user_metal_presets[name] = self._current_metal_preset_payload()
        self._save_user_metal_presets()
        self._refresh_metal_preset_combo()
        index = self.metal_preset_combo.findText(name)
        if index >= 0:
            self.metal_preset_combo.setCurrentIndex(index)

    def _delete_selected_metal_preset(self) -> None:
        data = self.metal_preset_combo.currentData()
        if not isinstance(data, tuple) or len(data) != 2 or data[0] != "user":
            return
        self._user_metal_presets.pop(str(data[1]), None)
        self._save_user_metal_presets()
        self._refresh_metal_preset_combo()

    def _apply_noisy_traces_via_preset(self, *_args) -> None:
        self._apply_via_preset_payload(self._noisy_traces_via_preset_payload())

    def _apply_blurred_via_preset(self, *_args) -> None:
        self._apply_via_preset_payload(self._blurred_via_preset_payload())

    def _reset_via_search_parameters(self, *_args) -> None:
        blockers = [
            QSignalBlocker(self.via_search_mode_combo),
            QSignalBlocker(self.via_min_score_spin),
            QSignalBlocker(self.via_min_contrast_spin),
            QSignalBlocker(self.via_min_edge_coverage_spin),
            QSignalBlocker(self.via_spot_line_suppression_spin),
            QSignalBlocker(self.via_template_min_score_spin),
            QSignalBlocker(self.via_roundness_spin),
        ]
        try:
            mode_index = self.via_search_mode_combo.findData("template")
            if mode_index >= 0:
                self.via_search_mode_combo.setCurrentIndex(mode_index)
            self.via_min_score_spin.setValue(0.35)
            self.via_min_contrast_spin.setValue(14.0)
            self.via_min_edge_coverage_spin.setValue(0.45)
            self.via_spot_line_suppression_spin.setValue(0.65)
            self.via_template_min_score_spin.setValue(0.35)
            self.via_roundness_spin.setValue(40.0)
        finally:
            del blockers
        self._update_via_threshold_controls_state()
        self._on_extraction_settings_changed()

    @staticmethod
    def _heuristic_default_values() -> dict[str, float]:
        return {
            "heuristic_background_sigma_spin": 25.0,
            "heuristic_analysis_window_scale_spin": 3.0,
            "heuristic_min_center_brightness_spin": 0.0,
            "heuristic_min_center_contrast_spin": 50.0,
            "heuristic_min_peak_prominence_spin": 50.0,
            "heuristic_min_compactness_spin": 0.9,
            "heuristic_min_circularity_spin": 0.40,
            "heuristic_max_elongation_spin": 2.5,
            "heuristic_line_penalty_spin": 3.0,
            "heuristic_border_penalty_spin": 1.0,
            "heuristic_local_binarize_percentile_spin": 88.0,
            "heuristic_min_abs_peak_spin": 0.0,
            "heuristic_size_tolerance_range_spin": 0.36,
            "heuristic_size_tolerance_fixed_spin": 0.26,
            "heuristic_max_center_drift_ratio_spin": 0.72,
            "heuristic_max_line_coherence_spin": 0.82,
            "heuristic_min_edge_sharpness_spin": 0.20,
            "heuristic_contrast_score_min_spin": 3.0,
            "heuristic_contrast_score_max_spin": 20.0,
            "heuristic_prominence_score_min_spin": 2.0,
            "heuristic_prominence_score_max_spin": 25.0,
            "heuristic_edge_snr_score_min_spin": 0.70,
            "heuristic_edge_snr_score_max_spin": 2.80,
            "heuristic_edge_quality_floor_spin": 0.55,
            "heuristic_border_balance_scale_spin": 2.0,
            "heuristic_seed_percentile_spin": 90.0,
            "heuristic_w_contrast_spin": 25.0,
            "heuristic_w_prominence_spin": 20.0,
            "heuristic_w_size_spin": 20.0,
            "heuristic_w_compact_spin": 15.0,
            "heuristic_w_round_spin": 10.0,
            "heuristic_w_balance_spin": 10.0,
            "heuristic_w_line_spin": 20.0,
            "heuristic_w_border_spin": 20.0,
        }

    def _reset_heuristic_parameters(self, *_args) -> None:
        defaults = self._heuristic_default_values()
        blockers = [
            QSignalBlocker(widget)
            for name in defaults
            if (widget := getattr(self, name, None)) is not None
        ]
        if hasattr(self, "heuristic_use_bilateral_checkbox"):
            blockers.append(QSignalBlocker(self.heuristic_use_bilateral_checkbox))
        try:
            for name, value in defaults.items():
                widget = getattr(self, name, None)
                if widget is not None:
                    widget.setValue(value)
            if hasattr(self, "heuristic_use_bilateral_checkbox"):
                self.heuristic_use_bilateral_checkbox.setChecked(False)
        finally:
            del blockers
        self._on_extraction_settings_changed()

    def _reset_bright_via_parameters(self, *_args) -> None:
        heuristic_defaults = self._heuristic_default_values()
        blockers = [
            QSignalBlocker(self.bright_via_diameter_min_spin),
            QSignalBlocker(self.bright_via_diameter_max_spin),
            QSignalBlocker(self.via_output_diameter_spin),
            QSignalBlocker(self.bright_via_clahe_clip_spin),
            QSignalBlocker(self.bright_via_clahe_tile_spin),
            QSignalBlocker(self.bright_via_median_kernel_spin),
            QSignalBlocker(self.bright_via_tophat_kernel_spin),
            QSignalBlocker(self.bright_via_dog_small_spin),
            QSignalBlocker(self.bright_via_dog_large_spin),
            QSignalBlocker(self.bright_via_threshold_percentile_spin),
            QSignalBlocker(self.bright_via_mask_combine_combo),
            QSignalBlocker(self.bright_via_min_area_factor_spin),
            QSignalBlocker(self.bright_via_max_area_factor_spin),
            QSignalBlocker(self.bright_via_min_circularity_spin),
            QSignalBlocker(self.bright_via_min_aspect_spin),
            QSignalBlocker(self.bright_via_max_aspect_spin),
            QSignalBlocker(self.bright_via_bright_center_score_spin),
            QSignalBlocker(self.bright_via_metal_constraint_combo),
            QSignalBlocker(self.bright_via_metal_fraction_spin),
            QSignalBlocker(self.bright_via_max_radial_asymmetry_spin),
            QSignalBlocker(self.bright_via_max_edge_likeness_spin),
            QSignalBlocker(self.bright_via_max_line_likeness_spin),
            QSignalBlocker(self.bright_via_nms_distance_spin),
            QSignalBlocker(self.bright_via_min_final_score_spin),
            QSignalBlocker(self.bright_via_show_rejected_checkbox),
            QSignalBlocker(self.bright_via_hard_asym_checkbox),
            QSignalBlocker(self.bright_via_hard_edge_checkbox),
            QSignalBlocker(self.bright_via_hard_line_checkbox),
        ]
        blockers.extend(
            QSignalBlocker(widget)
            for name in heuristic_defaults
            if (widget := getattr(self, name, None)) is not None
        )
        try:
            self.bright_via_diameter_min_spin.setValue(8)
            self.bright_via_diameter_max_spin.setValue(8)
            self.via_output_diameter_spin.setValue(8)
            self.bright_via_clahe_clip_spin.setValue(2.0)
            self.bright_via_clahe_tile_spin.setValue(8)
            self.bright_via_median_kernel_spin.setValue(3)
            self.bright_via_tophat_kernel_spin.setValue(11)
            self.bright_via_dog_small_spin.setValue(0.8)
            self.bright_via_dog_large_spin.setValue(2.0)
            self.bright_via_threshold_percentile_spin.setValue(99.0)
            combine_index = self.bright_via_mask_combine_combo.findData("OR")
            if combine_index >= 0:
                self.bright_via_mask_combine_combo.setCurrentIndex(combine_index)
            self.bright_via_min_area_factor_spin.setValue(0.45)
            self.bright_via_max_area_factor_spin.setValue(1.8)
            self.bright_via_min_circularity_spin.setValue(0.30)
            self.bright_via_min_aspect_spin.setValue(0.45)
            self.bright_via_max_aspect_spin.setValue(2.2)
            self.bright_via_bright_center_score_spin.setValue(140.0)
            self.via_white_range_checkbox.setChecked(True)
            self.via_white_range_min_spin.setValue(140)
            self.via_white_range_max_spin.setValue(255)
            self.via_black_range_checkbox.setChecked(False)
            metal_index = self.bright_via_metal_constraint_combo.findData("soft")
            if metal_index >= 0:
                self.bright_via_metal_constraint_combo.setCurrentIndex(metal_index)
            self.bright_via_metal_fraction_spin.setValue(0.3)
            self.bright_via_max_radial_asymmetry_spin.setValue(32.0)
            self.bright_via_max_edge_likeness_spin.setValue(35.0)
            self.bright_via_max_line_likeness_spin.setValue(65.0)
            self.bright_via_nms_distance_spin.setValue(5)
            self.bright_via_min_final_score_spin.setValue(38.0)
            self.bright_via_show_rejected_checkbox.setChecked(True)
            self.bright_via_hard_asym_checkbox.setChecked(True)
            self.bright_via_hard_edge_checkbox.setChecked(False)
            self.bright_via_hard_line_checkbox.setChecked(False)
            for name, value in heuristic_defaults.items():
                widget = getattr(self, name, None)
                if widget is not None:
                    widget.setValue(value)
            if hasattr(self, "heuristic_use_bilateral_checkbox"):
                self.heuristic_use_bilateral_checkbox.setChecked(False)
        finally:
            del blockers
        self._on_extraction_settings_changed()

    def _add_color_selection(self, row: int, rgb: tuple[int, int, int]) -> None:
        entries = self._color_selection_entries(row)
        for entry in entries:
            if tuple(entry["rgb"]) == tuple(rgb):
                entry["enabled"] = True
                self._pipeline.steps[row].parameters["selected_colors"] = entries
                self._render_pipeline_parameters(row)
                self._auto_apply_pipeline()
                return
        entries.append({"rgb": [int(rgb[0]), int(rgb[1]), int(rgb[2])], "enabled": True})
        self._pipeline.steps[row].parameters["selected_colors"] = entries
        self._render_pipeline_parameters(row)
        self._auto_apply_pipeline()

    def _on_editor_image_clicked(self, x_coord: float, y_coord: float) -> None:
        row = self._color_pick_pipeline_row
        if row is None or row < 0 or row >= len(self._pipeline.steps):
            return
        current_state = self._workspace.current_state
        if current_state is None or current_state.source_image is None:
            return
        image = np.asarray(current_state.source_image)
        x_index = round(x_coord)
        y_index = round(y_coord)
        if y_index < 0 or x_index < 0 or y_index >= image.shape[0] or x_index >= image.shape[1]:
            return
        if image.ndim == 2:
            value = int(image[y_index, x_index])
            rgb = (value, value, value)
        else:
            pixel = image[y_index, x_index]
            if image.shape[2] >= 3:
                rgb = (int(pixel[2]), int(pixel[1]), int(pixel[0]))
            else:
                value = int(pixel[0])
                rgb = (value, value, value)
        self._add_color_selection(row, rgb)
        self._append_log(
            self._tr(
                "color_picked_log",
                "Добавлен цвет {color}" if self._ui_language == "ru" else "Added color {color}",
                color=f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}",
            )
        )
