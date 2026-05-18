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
        self._refresh_pipeline_preset_combo()

    def _built_in_pipeline_presets(self) -> dict[str, dict[str, object]]:
        return built_in_pipeline_presets(self._ui_language)

    def _refresh_pipeline_preset_combo(self) -> None:
        if not hasattr(self, "pipeline_preset_combo"):
            return
        current_name = self.pipeline_preset_combo.currentText()
        self.pipeline_preset_combo.clear()
        for name in self._built_in_pipeline_presets():
            self.pipeline_preset_combo.addItem(name, name)
        index = self.pipeline_preset_combo.findText(current_name)
        if index >= 0:
            self.pipeline_preset_combo.setCurrentIndex(index)

    def _apply_selected_pipeline_preset(self) -> None:
        if not hasattr(self, "pipeline_preset_combo"):
            return
        preset_name = str(self.pipeline_preset_combo.currentData() or self.pipeline_preset_combo.currentText() or "")
        payload = self._built_in_pipeline_presets().get(preset_name)
        if not isinstance(payload, dict):
            return
        self._pipeline = PreprocessingPipeline.from_dict(payload)
        self._populate_pipeline_list()
        self.process_current_image(debounced=True)

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
        if not hasattr(self, "via_template_list"):
            return
        self.via_template_list.clear()
        for index, template in enumerate(self._via_template_images, start=1):
            height, width = template.shape[:2]
            item = QListWidgetItem(f"{index}: {width}x{height}")
            preview_pixmap = QPixmap.fromImage(cv_to_qimage(template)).scaled(
                56,
                56,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            item.setIcon(QIcon(preview_pixmap))
            item.setToolTip(
                f"Шаблон via #{index}: {width}x{height} пикс."
                if self._ui_language == "ru"
                else f"Via template #{index}: {width}x{height} px"
            )
            self.via_template_list.addItem(item)

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
        if hasattr(self, "add_via_template_button"):
            self.add_via_template_button.setChecked(False)
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
        self._via_template_images.append(template.astype(np.uint8, copy=False))
        self._refresh_via_template_list()
        self._on_extraction_settings_changed()
        self._append_log(
            self._tr(
                "via_template_added_log",
                "Добавлен шаблон via {width}x{height}. Всего шаблонов: {count}."
                if self._ui_language == "ru"
                else "Added via template {width}x{height}. Total templates: {count}.",
                width=right - left,
                height=bottom - top,
                count=len(self._via_template_images),
            )
        )

    def _clear_via_templates(self, *_args) -> None:
        self._via_template_images.clear()
        self._refresh_via_template_list()
        self._on_extraction_settings_changed()

    def _remove_selected_via_template(self, *_args) -> None:
        row = self.via_template_list.currentRow() if hasattr(self, "via_template_list") else -1
        if row < 0 or row >= len(self._via_template_images):
            return
        self._via_template_images.pop(row)
        self._refresh_via_template_list()
        if self._via_template_images:
            self.via_template_list.setCurrentRow(min(row, len(self._via_template_images) - 1))
        self._on_extraction_settings_changed()

    def _built_in_via_presets(self) -> dict[str, dict[str, object]]:
        return built_in_via_presets(self._ui_language)

    def _noisy_traces_via_preset_payload(self) -> dict[str, object]:
        return noisy_traces_via_preset_payload()

    def _blurred_via_preset_payload(self) -> dict[str, object]:
        return blurred_via_preset_payload()

    def _load_user_via_presets(self) -> dict[str, dict[str, object]]:
        return self._via_preset_settings_store.load()

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
        excluded_keys = {
            "via_template_images",
            "fixed_via_widths",
            "fixed_via_heights",
            "min_via_width",
            "max_via_width",
            "min_via_height",
            "max_via_height",
            "via_size_mode",
        }
        return {
            key: value
            for key, value in payload.items()
            if (key.startswith("via_") or key.startswith("bright_via_")) and key not in excluded_keys
        } | {
            "debug_enabled": self.debug_candidates_checkbox.isChecked()
        }

    def _apply_via_preset_payload(self, payload: dict[str, object]) -> None:
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
                float(
                    payload.get(
                        "bright_via_threshold_percentile", self.bright_via_threshold_percentile_spin.value()
                    )
                )
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
                bool(
                    payload.get(
                        "bright_via_hard_reject_on_asymmetry", self.bright_via_hard_asym_checkbox.isChecked()
                    )
                )
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
        finally:
            del blockers
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
        self._user_via_presets[name] = self._current_via_preset_payload()
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

    def _select_bright_via_mode(self) -> None:
        ridx = self.recognition_mode_combo.findData("via")
        if ridx >= 0 and self.recognition_mode_combo.currentIndex() != ridx:
            self.recognition_mode_combo.setCurrentIndex(ridx)
        mode_index = self.via_search_mode_combo.findData(VIA_SEARCH_MODE_BRIGHT_TOPHAT_DOG)
        if mode_index >= 0 and self.via_search_mode_combo.currentIndex() != mode_index:
            self.via_search_mode_combo.setCurrentIndex(mode_index)

    def _preview_bright_via_mask(self, *_args) -> None:
        self._select_bright_via_mode()
        self.debug_candidates_checkbox.setChecked(True)
        self._show_gradient_debug_window()

    def _reset_bright_via_parameters(self, *_args) -> None:
        blockers = [
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
        ]
        try:
            self.bright_via_diameter_min_spin.setValue(6)
            self.bright_via_diameter_max_spin.setValue(8)
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
            self.bright_via_bright_center_score_spin.setValue(6.0)
            metal_index = self.bright_via_metal_constraint_combo.findData("soft")
            if metal_index >= 0:
                self.bright_via_metal_constraint_combo.setCurrentIndex(metal_index)
            self.bright_via_metal_fraction_spin.setValue(0.3)
            self.bright_via_max_radial_asymmetry_spin.setValue(18.0)
            self.bright_via_max_edge_likeness_spin.setValue(35.0)
            self.bright_via_max_line_likeness_spin.setValue(65.0)
            self.bright_via_nms_distance_spin.setValue(5)
            self.bright_via_min_final_score_spin.setValue(38.0)
            self.bright_via_show_rejected_checkbox.setChecked(True)
            self.bright_via_hard_asym_checkbox.setChecked(False)
            self.bright_via_hard_edge_checkbox.setChecked(False)
            self.bright_via_hard_line_checkbox.setChecked(False)
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


