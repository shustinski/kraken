from __future__ import annotations

from ._imports import *  # noqa: F403


class WidgetHelpMixin:
    def _build_files_tab(self) -> QWidget:
        return build_files_tab(self)

    def _build_pipeline_tab(self) -> QWidget:
        return build_pipeline_tab(self)

    def _build_extraction_tab(self) -> QWidget:
        return build_extraction_tab(self)

    def _build_display_tab(self) -> QWidget:
        return build_display_tab(self)

    def _build_help_tab(self) -> QWidget:
        return build_help_tab(self)

    def _clear_layout_widgets(self, layout: QVBoxLayout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            child_layout = item.layout()
            if widget is not None:
                widget.deleteLater()
            elif child_layout is not None:
                self._clear_layout_widgets(child_layout)

    @staticmethod
    def _build_help_sample_image() -> np.ndarray:
        image = np.full((180, 260), 38, dtype=np.uint8)
        cv2.rectangle(image, (18, 18), (110, 90), 190, thickness=-1)
        cv2.circle(image, (176, 60), 26, 230, thickness=-1)
        cv2.circle(image, (176, 60), 10, 70, thickness=-1)
        cv2.line(image, (20, 136), (236, 120), 160, thickness=6)
        cv2.line(image, (22, 154), (236, 154), 210, thickness=4)
        cv2.putText(image, "A1", (126, 138), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 240, 2, cv2.LINE_AA)
        noise = np.random.default_rng(42).normal(0, 12, image.shape).astype(np.int16)
        return np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    def _operation_help_entry(self, operation_name: str) -> tuple[str, str]:
        entry = PIPELINE_OPERATION_HELP_TEXTS.get(operation_name, {})
        summary_pair = entry.get("summary", ("", ""))
        use_pair = entry.get("use", ("", ""))
        summary = summary_pair[0] if self._ui_language == "ru" else summary_pair[1]
        use_case = use_pair[0] if self._ui_language == "ru" else use_pair[1]
        if not summary:
            summary = (
                "Преобразование обрабатывает изображение перед извлечением контуров."
                if self._ui_language == "ru"
                else "This transformation preprocesses the image before contour extraction."
            )
        if not use_case:
            use_case = (
                "Используйте, когда этот эффект приближает изображение к удобной бинарной маске."
                if self._ui_language == "ru"
                else "Use it when the effect moves the image toward a cleaner binary mask."
            )
        return summary, use_case

    def _pipeline_parameter_tooltip(self, operation_name: str, parameter_name: str) -> str:
        del operation_name
        return _localized_text(PIPELINE_PARAMETER_HELP_TEXTS, parameter_name, self._ui_language)

    def _pixmap_for_help_image(self, image: np.ndarray) -> QPixmap:
        return QPixmap.fromImage(cv_to_qimage(image)).scaled(
            190,
            132,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

    def _rebuild_help_cards(self) -> None:
        if not hasattr(self, "help_layout"):
            return
        self._clear_layout_widgets(self.help_layout)
        intro = QLabel(
            "Ниже показано, как каждое преобразование меняет один и тот же тестовый кадр. Это помогает понять, когда шаг уместен в pipeline."
            if self._ui_language == "ru"
            else "Below, each transformation is applied to the same synthetic sample image so you can see what it changes and when to use it."
        )
        intro.setWordWrap(True)
        self.help_layout.addWidget(intro)
        sample_image = self._build_help_sample_image()
        before_pixmap = self._pixmap_for_help_image(sample_image)
        for descriptor in available_operations():
            card = QGroupBox(get_operation_display_name(descriptor.type_name, self._ui_language))
            card_layout = QVBoxLayout(card)
            summary, use_case = self._operation_help_entry(descriptor.type_name)
            summary_label = QLabel(summary)
            summary_label.setWordWrap(True)
            use_label = QLabel(("Когда использовать: " if self._ui_language == "ru" else "When to use: ") + use_case)
            use_label.setWordWrap(True)
            images_row = QWidget()
            images_layout = QHBoxLayout(images_row)
            images_layout.setContentsMargins(0, 0, 0, 0)
            before_box = QVBoxLayout()
            before_title = QLabel("До" if self._ui_language == "ru" else "Before")
            before_image = QLabel()
            before_image.setPixmap(before_pixmap)
            before_box.addWidget(before_title)
            before_box.addWidget(before_image)
            after_box = QVBoxLayout()
            after_title = QLabel("После" if self._ui_language == "ru" else "After")
            after_image = QLabel()
            try:
                processed = descriptor.handler(sample_image.copy(), descriptor.default_parameters())
            except Exception:
                processed = sample_image
            after_image.setPixmap(self._pixmap_for_help_image(processed))
            after_box.addWidget(after_title)
            after_box.addWidget(after_image)
            images_layout.addLayout(before_box)
            images_layout.addLayout(after_box)
            card_layout.addWidget(summary_label)
            card_layout.addWidget(use_label)
            card_layout.addWidget(images_row)
            self.help_layout.addWidget(card)
        self.help_layout.addStretch(1)

    def help_menu_title(self) -> str:
        return self._tr("tab_help")

    def attach_help_menu(self, menu: QMenu) -> None:
        self._help_menu = menu
        self._refresh_help_menu()

    def _refresh_help_menu(self) -> None:
        if self._help_menu is None:
            return
        self._help_menu.clear()
        postprocess_action = self._help_menu.addAction(
            "Постобработка ручных инструментов"
            if self._ui_language == "ru"
            else "Manual tool post-processing"
        )
        postprocess_action.setObjectName("manualToolPostprocessAction")
        postprocess_action.triggered.connect(lambda _checked=False: self._show_manual_tool_postprocess_dialog())
        self._help_menu.addSeparator()
        overview_action = self._help_menu.addAction(
            self._tr(
                "help_all_filters_action", "Все преобразования" if self._ui_language == "ru" else "All transformations"
            )
        )
        overview_action.triggered.connect(lambda _checked=False: self._show_help_dialog())
        hotkeys_action = self._help_menu.addAction(
            self._tr(
                "help_editor_hotkeys_action",
                "Горячие клавиши редактора" if self._ui_language == "ru" else "Editor hotkeys",
            )
        )
        hotkeys_action.triggered.connect(lambda _checked=False: self._show_editor_hotkeys_dialog())
        self._help_menu.addSeparator()
        for group_key, labels, operations in PIPELINE_OPERATION_GROUPS:
            submenu = self._help_menu.addMenu(labels[0] if self._ui_language == "ru" else labels[1])
            submenu.setObjectName(f"helpMenu_{group_key}")
            for operation_name in operations:
                action = submenu.addAction(get_operation_display_name(operation_name, self._ui_language))
                action.triggered.connect(lambda _checked=False, op=operation_name: self._show_help_dialog(op))

    def _show_help_dialog(self, operation_name: str | None = None) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle(
            self._tr("tab_help")
            if operation_name is None
            else get_operation_display_name(operation_name, self._ui_language)
        )
        dialog.resize(960, 720)
        layout = QVBoxLayout(dialog)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        help_layout = QVBoxLayout(container)
        help_layout.setContentsMargins(0, 0, 0, 0)
        scroll.setWidget(container)
        layout.addWidget(scroll, 1)
        self._populate_help_cards(
            help_layout,
            [operation_name] if operation_name is not None else self._all_operation_names(),
        )
        dialog.exec()

    def _show_editor_hotkeys_dialog(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle(
            self._tr(
                "help_editor_hotkeys_action",
                "Горячие клавиши редактора" if self._ui_language == "ru" else "Editor hotkeys",
            )
        )
        dialog.resize(520, 560)
        layout = QVBoxLayout(dialog)
        text = QTextEdit()
        text.setReadOnly(True)
        text.setPlainText(build_editor_hotkeys_plain_text(ru=self._ui_language == "ru"))
        layout.addWidget(text, 1)
        close_row = QHBoxLayout()
        close_row.addStretch(1)
        close_button = QPushButton("Закрыть" if self._ui_language == "ru" else "Close")
        close_button.clicked.connect(dialog.accept)
        close_row.addWidget(close_button)
        layout.addLayout(close_row)
        dialog.exec()

    def _populate_help_cards(self, layout: QVBoxLayout, operation_names: list[str]) -> None:
        self._clear_layout_widgets(layout)
        intro = QLabel(
            self._tr(
                "help_intro_text",
                "Ниже показано, как каждое преобразование меняет один и тот же тестовый кадр. Это помогает понять, когда шаг уместен в pipeline."
                if self._ui_language == "ru"
                else "Each transformation below is applied to the same sample image so you can see its effect and when to use it.",
            )
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        sample_image = self._build_help_sample_image()
        before_pixmap = self._pixmap_for_help_image(sample_image)
        for operation_name in operation_names:
            descriptor = get_operation_descriptor(operation_name)
            card = QGroupBox(get_operation_display_name(descriptor.type_name, self._ui_language))
            card_layout = QVBoxLayout(card)
            summary, use_case = self._operation_help_entry(descriptor.type_name)
            summary_label = QLabel(summary)
            summary_label.setWordWrap(True)
            use_label = QLabel(("Когда использовать: " if self._ui_language == "ru" else "When to use: ") + use_case)
            use_label.setWordWrap(True)
            images_row = QWidget()
            images_layout = QHBoxLayout(images_row)
            images_layout.setContentsMargins(0, 0, 0, 0)
            before_box = QVBoxLayout()
            before_title = QLabel("До" if self._ui_language == "ru" else "Before")
            before_image = QLabel()
            before_image.setPixmap(before_pixmap)
            before_box.addWidget(before_title)
            before_box.addWidget(before_image)
            after_box = QVBoxLayout()
            after_title = QLabel("После" if self._ui_language == "ru" else "After")
            after_image = QLabel()
            try:
                processed = descriptor.handler(sample_image.copy(), descriptor.default_parameters())
            except Exception:
                processed = sample_image
            after_image.setPixmap(self._pixmap_for_help_image(processed))
            after_box.addWidget(after_title)
            after_box.addWidget(after_image)
            images_layout.addLayout(before_box)
            images_layout.addLayout(after_box)
            card_layout.addWidget(summary_label)
            card_layout.addWidget(use_label)
            card_layout.addWidget(images_row)
            layout.addWidget(card)
        layout.addStretch(1)

    def _all_operation_names(self) -> list[str]:
        return [descriptor.type_name for descriptor in available_operations()]

    def _selected_available_operation_name(self) -> str | None:
        if not hasattr(self, "operation_tree"):
            return None
        item = self.operation_tree.currentItem()
        if item is None:
            return None
        operation_name = item.data(0, Qt.ItemDataRole.UserRole)
        return str(operation_name) if operation_name else None

    def _find_operation_tree_item(self, operation_name: str) -> QTreeWidgetItem | None:
        if not hasattr(self, "operation_tree"):
            return None
        for index in range(self.operation_tree.topLevelItemCount()):
            group_item = self.operation_tree.topLevelItem(index)
            for child_index in range(group_item.childCount()):
                child_item = group_item.child(child_index)
                if child_item.data(0, Qt.ItemDataRole.UserRole) == operation_name:
                    return child_item
        return None

    def _update_pipeline_help_preview(self, operation_name: str | None) -> None:
        if not hasattr(self, "pipeline_help_title"):
            return
        if not operation_name:
            self.pipeline_help_title.clear()
            self.pipeline_help_summary.clear()
            self.pipeline_help_use.clear()
            self.pipeline_help_before_image.clear()
            self.pipeline_help_after_image.clear()
            return
        descriptor = get_operation_descriptor(operation_name)
        summary, use_case = self._operation_help_entry(operation_name)
        sample_image = self._build_help_sample_image()
        self.pipeline_help_title.setText(get_operation_display_name(operation_name, self._ui_language))
        self.pipeline_help_summary.setText(summary)
        self.pipeline_help_use.setText(
            ("Когда использовать: " if self._ui_language == "ru" else "When to use: ") + use_case
        )
        self.pipeline_help_before_image.setPixmap(self._pixmap_for_help_image(sample_image))
        try:
            processed = descriptor.handler(sample_image.copy(), descriptor.default_parameters())
        except Exception:
            processed = sample_image
        self.pipeline_help_after_image.setPixmap(self._pixmap_for_help_image(processed))

    def _on_available_operation_selected(
        self, current: QTreeWidgetItem | None, _previous: QTreeWidgetItem | None
    ) -> None:
        operation_name = current.data(0, Qt.ItemDataRole.UserRole) if current is not None else None
        self._update_pipeline_help_preview(str(operation_name) if operation_name else None)

    def _on_available_operation_activated(self, item: QTreeWidgetItem, _column: int) -> None:
        if item.data(0, Qt.ItemDataRole.UserRole):
            self._add_pipeline_step()

    def _set_field_tooltip(self, label_widget: QLabel | None, field_widget: QWidget, help_key: str) -> None:
        tooltip = _localized_text(EXTRACTION_HELP_TEXTS, help_key, self._ui_language)
        if label_widget is not None:
            label_widget.setToolTip(tooltip)
        field_widget.setToolTip(tooltip)


