from __future__ import annotations

from ._imports import *  # noqa: F403


class WidgetExtractionControlsMixin:
    def _renumber_fixed_via_rows(self) -> None:
        for index, row in enumerate(self._fixed_via_rows, start=1):
            label = row["label"]
            if isinstance(label, QLabel):
                label.setText(f"via{index}")

    def _clear_fixed_via_rows(self) -> None:
        while self._fixed_via_rows:
            row = self._fixed_via_rows.pop()
            widget = row["widget"]
            if isinstance(widget, QWidget):
                self.fixed_via_rows_layout.removeWidget(widget)
                widget.deleteLater()

    def _fixed_via_pairs(self) -> list[tuple[int, int]]:
        pairs: list[tuple[int, int]] = []
        for row in self._fixed_via_rows:
            width_spin = row["width_spin"]
            height_spin = row["height_spin"]
            if isinstance(width_spin, QSpinBox) and isinstance(height_spin, QSpinBox):
                pairs.append((int(width_spin.value()), int(height_spin.value())))
        return pairs

    def _delete_fixed_via_row(self, row_widget: QWidget) -> None:
        for index, row in enumerate(self._fixed_via_rows):
            if row["widget"] is row_widget:
                self._fixed_via_rows.pop(index)
                self.fixed_via_rows_layout.removeWidget(row_widget)
                row_widget.deleteLater()
                self._renumber_fixed_via_rows()
                if not self._suspend_fixed_via_updates:
                    self._on_extraction_settings_changed()
                return

    def _add_fixed_via_row(self, *_args, width: int = 1, height: int = 1) -> None:
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(8)

        via_label = QLabel("")
        via_label.setMinimumWidth(44)
        width_spin = QSpinBox()
        width_spin.setRange(1, 100_000)
        width_spin.setValue(max(1, int(width)))
        width_spin.setPrefix("X ")
        height_spin = QSpinBox()
        height_spin.setRange(1, 100_000)
        height_spin.setValue(max(1, int(height)))
        height_spin.setPrefix("Y ")
        remove_button = QPushButton("-")
        remove_button.setFixedWidth(36)
        remove_button.setMinimumHeight(30)
        remove_button.setStyleSheet(
            "QPushButton { background-color: #d64545; color: white; font-size: 18px; font-weight: 700; border-radius: 6px; }"
            "QPushButton:hover { background-color: #bf3838; }"
            "QPushButton:pressed { background-color: #a93030; }"
        )

        width_spin.valueChanged.connect(self._on_extraction_settings_changed)
        height_spin.valueChanged.connect(self._on_extraction_settings_changed)
        remove_button.clicked.connect(lambda _checked=False, widget=row_widget: self._delete_fixed_via_row(widget))

        self._fixed_via_rows.append(
            {
                "widget": row_widget,
                "label": via_label,
                "width_spin": width_spin,
                "height_spin": height_spin,
                "remove_button": remove_button,
            }
        )

        row_layout.addWidget(via_label)
        row_layout.addWidget(width_spin, 1)
        row_layout.addWidget(height_spin, 1)
        row_layout.addWidget(remove_button)
        self.fixed_via_rows_layout.addWidget(row_widget)
        self._renumber_fixed_via_rows()

        width_spin.setToolTip(_localized_text(EXTRACTION_HELP_TEXTS, "fixed_via_widths", self._ui_language))
        height_spin.setToolTip(_localized_text(EXTRACTION_HELP_TEXTS, "fixed_via_heights", self._ui_language))
        remove_button.setToolTip(
            "Удаляет эту строку с допустимым размером via из списка."
            if self._ui_language == "ru"
            else "Removes this allowed via-size row from the list."
        )

        if not self._suspend_fixed_via_updates:
            self._on_extraction_settings_changed()

    def _apply_extraction_tooltips(self) -> None:
        self._set_field_tooltip(self.retrieval_mode_label_widget, self.retrieval_mode_combo, "retrieval_mode")
        self._set_field_tooltip(
            self.approximation_mode_label_widget, self.approximation_mode_combo, "approximation_mode"
        )
        self._set_field_tooltip(
            self.epsilon_label_widget,
            self.epsilon_row_widget if hasattr(self, "epsilon_row_widget") else self.epsilon_spin,
            "epsilon",
        )
        self._set_field_tooltip(self.epsilon_mode_label_widget, self.epsilon_relative_checkbox, "epsilon_mode")
        self._set_field_tooltip(self.min_area_label_widget, self.min_area_spin, "min_area")
        self._set_field_tooltip(self.max_area_label_widget, self.max_area_spin, "max_area")
        self._set_field_tooltip(self.min_perimeter_label_widget, self.min_perimeter_spin, "min_perimeter")
        self._set_field_tooltip(self.max_perimeter_label_widget, self.max_perimeter_spin, "max_perimeter")
        self._set_field_tooltip(self.min_point_count_label_widget, self.min_points_spin, "min_points")
        self._set_field_tooltip(
            self.min_polygon_width_label_widget, self.min_polygon_width_spin, "min_polygon_width"
        )
        self._set_field_tooltip(self.min_bbox_width_label_widget, self.min_bbox_width_spin, "min_bbox_width")
        self._set_field_tooltip(self.max_bbox_width_label_widget, self.max_bbox_width_spin, "max_bbox_width")
        self._set_field_tooltip(self.min_bbox_height_label_widget, self.min_bbox_height_spin, "min_bbox_height")
        self._set_field_tooltip(self.max_bbox_height_label_widget, self.max_bbox_height_spin, "max_bbox_height")
        self._set_field_tooltip(self.min_aspect_ratio_label_widget, self.min_aspect_ratio_spin, "min_aspect_ratio")
        self._set_field_tooltip(self.max_aspect_ratio_label_widget, self.max_aspect_ratio_spin, "max_aspect_ratio")
        self._set_field_tooltip(
            self.border_handling_label_widget, self.exclude_border_touching_checkbox, "exclude_border_touching"
        )
        self._set_field_tooltip(self.min_solidity_label_widget, self.min_solidity_spin, "min_solidity")
        self._set_field_tooltip(self.min_extent_label_widget, self.min_extent_spin, "min_extent")
        self._set_field_tooltip(self.via_size_mode_label_widget, self.via_size_mode_combo, "via_size_mode")
        if getattr(self, "via_search_mode_label_widget", None) is not None:
            self._set_field_tooltip(self.via_search_mode_label_widget, self.via_search_mode_combo, "via_search_mode")
        if hasattr(self, "bright_via_viamode_label_widget"):
            self._set_field_tooltip(self.bright_via_viamode_label_widget, self.via_search_mode_combo, "via_search_mode")
        self._set_field_tooltip(self.via_white_range_label_widget, self.via_white_range_widget, "via_white_range")
        self._set_field_tooltip(self.via_black_range_label_widget, self.via_black_range_widget, "via_black_range")
        self._set_field_tooltip(self.via_min_score_label_widget, self.via_min_score_spin, "via_min_score")
        self._set_field_tooltip(self.via_min_contrast_label_widget, self.via_min_contrast_spin, "via_min_contrast")
        self._set_field_tooltip(
            self.via_min_edge_coverage_label_widget,
            self.via_min_edge_coverage_spin,
            "via_min_edge_coverage",
        )
        self._set_field_tooltip(
            self.via_spot_line_suppression_label_widget,
            self.via_spot_line_suppression_spin,
            "via_spot_line_suppression",
        )
        self._set_field_tooltip(
            self.via_template_min_score_label_widget, self.via_template_min_score_spin, "via_template_min_score"
        )
        self._set_field_tooltip(self.via_templates_label_widget, self.via_templates_widget, "via_templates")
        self._set_field_tooltip(self.via_preset_label_widget, self.via_preset_widget, "via_preset_selector")
        if getattr(self, "noisy_traces_via_preset_label_widget", None) is not None:
            self._set_field_tooltip(
                self.noisy_traces_via_preset_label_widget,
                self.noisy_traces_via_preset_button,
                "via_noisy_traces_preset",
            )
        else:
            self.noisy_traces_via_preset_button.setToolTip(
                _localized_text(EXTRACTION_HELP_TEXTS, "via_noisy_traces_preset", self._ui_language)
            )
        if getattr(self, "blurred_via_preset_label_widget", None) is not None:
            self._set_field_tooltip(
                self.blurred_via_preset_label_widget,
                self.blurred_via_preset_button,
                "via_blurred_preset",
            )
        else:
            self.blurred_via_preset_button.setToolTip(
                _localized_text(EXTRACTION_HELP_TEXTS, "via_blurred_preset", self._ui_language)
            )
        self._set_field_tooltip(self.reset_via_search_label_widget, self.reset_via_search_button, "reset_via_search")
        self.add_via_template_button.setToolTip(
            _localized_text(EXTRACTION_HELP_TEXTS, "via_templates", self._ui_language)
        )
        self.remove_via_template_button.setToolTip(
            "Удаляет выбранный шаблон via из списка."
            if self._ui_language == "ru"
            else "Removes the selected via template from the list."
        )
        self.clear_via_templates_button.setToolTip(
            "Удаляет все сохраненные шаблоны via из списка."
            if self._ui_language == "ru"
            else "Removes all saved via templates from the list."
        )
        for checkbox, tooltip_key in (
            (self.via_white_range_checkbox, "via_white_range"),
            (self.via_black_range_checkbox, "via_black_range"),
        ):
            detector_tooltip = _localized_text(EXTRACTION_HELP_TEXTS, tooltip_key, self._ui_language)
            checkbox.setToolTip(detector_tooltip)
            checkbox.setStatusTip(detector_tooltip)
        self._set_field_tooltip(self.debug_candidates_label_widget, self.debug_candidates_checkbox, "debug_candidates")
        self._set_field_tooltip(self.via_roundness_label_widget, self.via_roundness_spin, "via_min_roundness")
        self._set_field_tooltip(self.min_via_width_label_widget, self.min_via_width_spin, "min_via_width")
        self._set_field_tooltip(self.max_via_width_label_widget, self.max_via_width_spin, "max_via_width")
        self._set_field_tooltip(self.min_via_height_label_widget, self.min_via_height_spin, "min_via_height")
        self._set_field_tooltip(self.max_via_height_label_widget, self.max_via_height_spin, "max_via_height")
        self._set_field_tooltip(self.fixed_vias_label_widget, self.fixed_vias_widget, "fixed_via_widths")
        self.fixed_via_add_button.setToolTip(
            "Добавляет еще одну допустимую пару ширины и высоты via."
            if self._ui_language == "ru"
            else "Adds another allowed via width and height pair."
        )
        for row in self._fixed_via_rows:
            width_spin = row["width_spin"]
            height_spin = row["height_spin"]
            remove_button = row["remove_button"]
            if isinstance(width_spin, QSpinBox):
                width_spin.setToolTip(_localized_text(EXTRACTION_HELP_TEXTS, "fixed_via_widths", self._ui_language))
            if isinstance(height_spin, QSpinBox):
                height_spin.setToolTip(_localized_text(EXTRACTION_HELP_TEXTS, "fixed_via_heights", self._ui_language))
            if isinstance(remove_button, QPushButton):
                remove_button.setToolTip(
                    "Удаляет эту строку с допустимым размером via из списка."
                    if self._ui_language == "ru"
                    else "Removes this allowed via-size row from the list."
                )
        self._set_field_tooltip(
            self.min_hierarchy_depth_label_widget, self.min_hierarchy_depth_spin, "min_hierarchy_depth"
        )
        self._set_field_tooltip(
            self.min_inner_hole_area_label_widget, self.min_inner_hole_area_spin, "min_inner_hole_area"
        )
        self._set_field_tooltip(
            self.max_hierarchy_depth_label_widget, self.max_hierarchy_depth_spin, "max_hierarchy_depth"
        )
        self._set_field_tooltip(
            self.max_hole_area_ratio_label_widget, self.max_hole_area_ratio_spin, "max_hole_area_ratio"
        )
        self._apply_bright_via_tooltips()

    def _apply_bright_via_tooltips(self) -> None:
        if not hasattr(self, "bright_via_diameter_min_spin"):
            return
        ru = self._ui_language == "ru"

        def tt(ru_text: str, en_text: str) -> str:
            return ru_text if ru else en_text

        self.bright_via_diameter_min_spin.setToolTip(
            tt(
                "Минимальный допустимый размер переходного отверстия в пикселях.\n"
                "Если значение слишком большое — маленькие via будут пропущены.\n"
                "Если слишком маленькое — появится больше ложных срабатываний на шуме.\n"
                "Обычно: 5–8 px.",
                "Minimum via diameter in pixels (typ. 5–8).",
            )
        )
        self.bright_via_diameter_max_spin.setToolTip(
            tt(
                "Максимальный допустимый размер via.\n"
                "Если слишком маленькое — крупные via будут пропущены.\n"
                "Если слишком большое — алгоритм начнёт принимать яркие фрагменты дорожек.\n"
                "Обычно: 8–14 px.",
                "Maximum via diameter in pixels (typ. 8–14).",
            )
        )
        self.bright_via_clahe_clip_spin.setToolTip(
            tt(
                "Предел усиления локального контраста (CLAHE).\n"
                "Больше значение — сильнее вытягиваются слабые детали, но растёт шум.\n"
                "Меньше — картинка ровнее, но слабые via могут стать незаметнее.\n"
                "Типично 1.5–3.5.",
                "CLAHE clip limit; higher emphasizes weak details and noise.",
            )
        )
        self.bright_via_clahe_tile_spin.setToolTip(
            tt(
                "Размер ячейки сетки CLAHE в пикселях.\n"
                "Меньше — контраст подстраивается локальнее (мелкие объекты), больше шума на мелкой текстуре.\n"
                "Больше — более глобально, меньше артефактов на зерне, но слабее локальный контраст.\n"
                "Часто 6–12.",
                "CLAHE tile size; smaller = more local adaptation.",
            )
        )
        self.bright_via_median_kernel_spin.setToolTip(
            tt(
                "Размер медианного фильтра (нечётное число; 1 = отключено по смыслу).\n"
                "Больше — сильнее подавление шума SEM, но мягче края via.\n"
                "Меньше — лучше сохраняются острые via, выше риск ложных точек.\n"
                "Типично 3.",
                "Median blur kernel (odd); larger removes more noise and softens edges.",
            )
        )
        self.bright_via_tophat_kernel_spin.setToolTip(
            tt(
                "Размер структурного элемента для белого top-hat (нечётное).\n"
                "Больше — подчёркиваются более крупные яркие вкрапления, фон на большей шкале.\n"
                "Меньше — чувствительнее к мелким пятнам и зерну.\n"
                "Сопоставляйте с ожидаемым диаметром via.",
                "White top-hat structuring size; match expected via scale.",
            )
        )
        self.bright_via_dog_small_spin.setToolTip(
            tt(
                "Меньшая сигма Гаусса в разности гауссов (DoG).\n"
                "Вместе с большой сигмой задаёт масштаб выделяемых ярких деталей.\n"
                "Слишком большая малая сигма — больше отклика на мелкий шум.\n"
                "Должна быть строго меньше «большой сигмы».",
                "DoG small sigma; must be < large sigma.",
            )
        )
        self.bright_via_dog_large_spin.setToolTip(
            tt(
                "Большая сигма Гаусса в DoG.\n"
                "Больше значение — сильнее сглаживание «крупного» масштаба, иначе выделяется фон.\n"
                "Меньше — остаётся больше мелких деталей в отклике.\n"
                "Подбирайте пару с малой сигмой под размер via.",
                "DoG large sigma; tune with small sigma for via size.",
            )
        )
        self.bright_via_threshold_percentile_spin.setToolTip(
            tt(
                "Определяет, насколько ярким должен быть пиксель, чтобы попасть в маску отклика.\n"
                "Большее значение → меньше ложных срабатываний, но больше пропусков.\n"
                "Меньшее значение → выше полнота поиска, но больше шума.\n"
                "Обычно: 97.5–99.2.",
                "Response percentile threshold (typ. 97.5–99.2).",
            )
        )
        self.bright_via_mask_combine_combo.setToolTip(
            tt(
                "ИЛИ — высокая полнота поиска, больше кандидатов.\n"
                "И — строгий режим, меньше ложных срабатываний, но больше пропусков.\n"
                "Обычно рекомендуется начинать с режима ИЛИ.",
                "OR = high recall; AND = stricter overlap of top-hat and DoG masks.",
            )
        )
        self.bright_via_min_area_factor_spin.setToolTip(
            tt(
                "Нижняя граница площади кандидата относительно площади идеального круга минимального диаметра.\n"
                "Больше — отсекаются слишком маленькие пятна (часто шум).\n"
                "Меньше — допускаются более мелкие объекты.\n"
                "Меняйте, если стабильно теряются мелкие via или наоборот много «крошек».",
                "Min area as a factor of π·(d_min/2)².",
            )
        )
        self.bright_via_max_area_factor_spin.setToolTip(
            tt(
                "Верхняя граница площади кандидата относительно площади круга максимального диаметра.\n"
                "Меньше — жёстче отсекаются крупные пятна (часто куски дорожек).\n"
                "Больше — допускаются более крупные отклики.\n"
                "Согласуйте с реальным размером via на SEM.",
                "Max area factor relative to max diameter.",
            )
        )
        self.bright_via_min_circularity_spin.setToolTip(
            tt(
                "Ожидаемая «круглость» контура (4π·area/perimeter²).\n"
                "Низкие значения допускают вытянутые пятна (часто артефакты дорожек).\n"
                "Высокие — ближе к диску, но реальные размытые via могут получать меньший балл.\n"
                "Обычно 0.15–0.45 в зависимости от качества изображения.",
                "Circularity expectation for blob shape (0–1).",
            )
        )
        self.bright_via_min_aspect_spin.setToolTip(
            tt(
                "Минимальное отношение ширины bounding box к высоте.\n"
                "Слишком большое — отсекаются слегка вытянутые via.\n"
                "Слишком маленькое — пропускаются сильно вытянутые ложные объекты реже.\n"
                "Для via обычно около 0.4–0.6.",
                "Min aspect ratio w/h of bbox.",
            )
        )
        self.bright_via_max_aspect_spin.setToolTip(
            tt(
                "Максимальное отношение сторон bbox.\n"
                "Меньше — строже к вытянутым контурам (меньше дорожных «колбас»).\n"
                "Больше — допускаются более вытянутые кандидаты.\n"
                "Слишком большое — растут ложные на границах дорожек.",
                "Max aspect ratio w/h of bbox.",
            )
        )
        self.bright_via_bright_center_score_spin.setToolTip(
            tt(
                "Центр via должен быть ярче окружающей области (разница средних по диску и кольцу).\n"
                "Увеличение значения уменьшает ложные срабатывания на слабом шуме,\n"
                "но может пропускать слабые или размытые via.\n"
                "Это жёсткий порог: ниже — кандидат отбрасывается сразу.",
                "Hard minimum center-vs-ring brightness delta.",
            )
        )
        self.bright_via_max_radial_asymmetry_spin.setToolTip(
            tt(
                "Проверяет симметричность яркости вокруг via (СКО по 8 направлениям).\n"
                "Настоящее via обычно симметрично, край дорожки — нет.\n"
                "Порог задаёт, насколько большой разброс ещё считается «похожим на via» в мягком режиме.\n"
                "Меньше значение в мягком режиме сильнее снижает итоговую оценку при асимметрии.\n"
                "Слишком жёсткий ручной отбор (если включить жёсткий режим) ведёт к пропускам на шуме.",
                "Reference level for radial brightness asymmetry (std).",
            )
        )
        self.bright_via_max_edge_likeness_spin.setToolTip(
            tt(
                "Ограничивает срабатывания на краях металлизации.\n"
                "Меньше значение — сильнее штраф в мягком режиме за «краевой» профиль.\n"
                "Больше — терпимее к via у границы дорожки.\n"
                "С жёстким режимом (если включён) пары с метрикой выше порога отбрасываются сразу.",
                "Edge-likeness cap / soft scale.",
            )
        )
        self.bright_via_max_line_likeness_spin.setToolTip(
            tt(
                "Отсекает объекты, похожие на куски дорожек (анизотропия градиентов в окне).\n"
                "Большее значение — мягче к вытянутым откликам, выше риск ложных срабатываний на трассы.\n"
                "Меньшее — жёстче к линиям, но больше риск пропуска via, слитых с трассой.\n"
                "В мягком режиме влияет на итоговый балл; в жёстком — и на немедленный отказ.",
                "Line-likeness (structure tensor) cap / scale.",
            )
        )
        self.bright_via_metal_constraint_combo.setToolTip(
            tt(
                "Определяет, использовать ли информацию о металлизации (Otsu+морфология).\n"
                "Отключено — не учитывать металл.\n"
                "Мягкая оценка — металл влияет только на итоговую оценку (бонус к баллу).\n"
                "Жёсткий фильтр — кандидаты вне металла с низкой долей покрытия отбрасываются.\n"
                "Если металл плохо виден, используйте «Отключено» или «Мягкая оценка».",
                "Metal mask: disabled / soft score / strict reject.",
            )
        )
        self.bright_via_metal_fraction_spin.setToolTip(
            tt(
                "Минимальная доля пикселей металла в окне вокруг кандидата для режима «Жёсткий фильтр».\n"
                "Выше — принимаются только via, лежащие на металлизации по маске.\n"
                "Ниже — больше кандидатов проходят, но растут ложные вне металла.\n"
                "В мягком режиме на порог ориентироваться не обязательно: используется непрерывный бонус.",
                "Min metal fraction for strict mode (0–1).",
            )
        )
        self.bright_via_min_final_score_spin.setToolTip(
            tt(
                "Главный параметр отбора итоговых via по суммарной оценке 0…100 (форма + локальные метрики).\n"
                "Увеличение → меньше ложных срабатываний, но больше пропусков.\n"
                "Уменьшение → больше найденных via, но больше кандидатов ниже порога (жёлтые на отладке).\n"
                "Обычно это один из самых важных параметров настройки.",
                "Minimum composite score (0–100) to accept a via.",
            )
        )
        self.bright_via_nms_distance_spin.setToolTip(
            tt(
                "Минимальное расстояние между двумя кандидатами после этапа слияния и подавления дублей.\n"
                "Если слишком маленькое — одно via может быть найдено несколько раз с разных откликов.\n"
                "Если слишком большое — соседние реальные via могут сливаться.\n"
                "Связывайте с ожидаемым шагом растра via.",
                "Non-maximum suppression distance in pixels.",
            )
        )
        self.bright_via_show_rejected_checkbox.setToolTip(
            tt(
                "Если включено, на итоговом наложении в отладке рисуются и отклонённые кандидаты: "
                "жёлтые — ниже порога итоговой оценки, красные — жёстко отброшенные по геометрии/контрасту/металлу.\n"
                "Если выключено — видны только принятые (зелёные).",
                "Show soft/hard rejected candidates on the debug overlay.",
            )
        )
        self.bright_via_hard_asym_checkbox.setToolTip(
            tt(
                "Если включено: при превышении «максимальной радиальной асимметрии» кандидат сразу отбрасывается.\n"
                "По умолчанию (выкл.) асимметрия влияет на балл, а не на мгновенный отказ.\n"
                "Включайте только если уверенно настроили порог по этой метрике.",
                "Hard-reject on radial asymmetry vs threshold.",
            )
        )
        self.bright_via_hard_edge_checkbox.setToolTip(
            tt(
                "Если включено: при слишком высокой «похожести на край» кандидат сразу отбрасывается.\n"
                "По умолчанию метрика только снижает итоговый балл.\n"
                "Полезно, если остаются устойчивые ложные на кромках металла после настройки мягкого скоринга.",
                "Hard-reject when edge-likeness exceeds cap.",
            )
        )
        self.bright_via_hard_line_checkbox.setToolTip(
            tt(
                "Если включено: при слишком высокой линейности (анизотропии градиентов) — мгновенный отказ.\n"
                "По умолчанию влияет на балл, чтобы не терять слабые круги на фоне трасс.\n"
                "Включайте при массовых ложных вдоль дорожек.",
                "Hard-reject when line-likeness exceeds cap.",
            )
        )
        self.preview_bright_via_mask_button.setToolTip(
            tt(
                "Переключает профиль на поиск via, режим «яркий top-hat/DoG», "
                "включает отладочные слои и открывает окно с картами (исходник, top-hat, DoG, маски, итог).",
                "Switch to bright via mode and open debug map window.",
            )
        )
        self.reset_bright_via_button.setToolTip(
            tt(
                "Сбрасывает параметры детектора к заводским значениям и запускает пересчёт (как при изменении настроек).",
                "Reset bright via parameters to defaults and re-run.",
            )
        )
        for w in (self.bright_via_diameter_range_widget,):
            w.setToolTip(
                tt(
                    "Пара min/max: см. подсказки у полей минимума и максимума диаметра.",
                    "Diameter range: see min and max tooltips.",
                )
            )
        if hasattr(self, "recognition_mode_combo"):
            self.recognition_mode_combo.setToolTip(
                tt(
                    "Выбор режима извлечения. По умолчанию включено «Без извлечения»; обработка запускается только после явного выбора режима.\n"
                    "Параметры на панели меняются в зависимости от режима.",
                    "Extraction mode. Defaults to No extraction; processing runs only after an explicit mode choice.",
                )
            )
        if hasattr(self, "via_search_sensitivity_combo"):
            self.via_search_sensitivity_combo.setToolTip(
                tt(
                    "Общий уровень агрессии поиска: «Низкая» — меньше ложных, больше пропусков; "
                    "«Средняя» — баланс; «Высокая» — больше срабатываний и кандидатов.\n"
                    "Меняет пороги и фильтры; в «Дополнительно» значения можно подправить вручную.",
                    "Coarse sensitivity for via search; adjust advanced fields manually if needed.",
                )
            )
        if hasattr(self, "via_show_detected_checkbox"):
            self.via_show_detected_checkbox.setToolTip(
                tt(
                    "Показывать на изображении полигоны via, найденные автоматически.",
                    "Show auto-detected via polygons on the image.",
                )
            )
        if hasattr(self, "via_debug_gradient_map_checkbox"):
            self.via_debug_gradient_map_checkbox.setToolTip(
                tt(
                    "Сохранять и показывать отладочные карты (градиент, маски) в окне «карта градиента» и при клике по отладке.",
                    "Enable extra debug image maps in the gradient / inspect views.",
                )
            )
        via_help: list[tuple[str, str, str]] = [
            (
                "heuristic_background_sigma_spin",
                "Размер размытия для оценки фона перед поиском локальных пиков.\n"
                "Увеличение убирает крупный фон и помогает на плавной засветке, но может ослабить близкие via.\n"
                "Уменьшение делает поиск локальнее и быстрее реагирует на мелкие перепады, но чаще принимает шум.",
                "Background blur sigma. Higher removes broad illumination; lower is more local and noisier.",
            ),
            (
                "heuristic_analysis_window_scale_spin",
                "Размер окна анализа вокруг найденного пика в долях диаметра via.\n"
                "Увеличение даёт больше контекста для формы и кольца, но медленнее и может захватить соседние дорожки.\n"
                "Уменьшение ускоряет проверку и лучше для плотных via, но хуже оценивает окружение.",
                "Analysis window in via diameters. Higher = more context/slower; lower = faster/tighter.",
            ),
            (
                "heuristic_min_center_contrast_spin",
                "Минимальная разница яркости центра и окружения.\n"
                "Увеличение уменьшает ложные срабатывания на слабой текстуре, но пропускает тусклые via.\n"
                "Уменьшение повышает полноту, но добавляет шумовые кандидаты.",
                "Minimum center-vs-surround contrast. Higher = cleaner; lower = more recall.",
            ),
            (
                "heuristic_min_peak_prominence_spin",
                "Насколько пик должен выделяться внутри локального окна.\n"
                "Увеличение отсекает плоские пятна и шум, но может потерять размытые via.\n"
                "Уменьшение принимает слабые пики и увеличивает число проверяемых кандидатов.",
                "Minimum local peak prominence. Higher = fewer candidates; lower = more recall/slower.",
            ),
            (
                "heuristic_min_compactness_spin",
                "Минимальная компактность локального компонента.\n"
                "Увеличение строже требует круглую/плотную форму via и режет вытянутые артефакты.\n"
                "Уменьшение допускает деформированные via, но чаще пропускает куски дорожек.",
                "Minimum component compactness. Higher = rounder; lower = more tolerant.",
            ),
            (
                "heuristic_max_elongation_spin",
                "Максимальная вытянутость компонента.\n"
                "Уменьшение сильнее отбрасывает линии и края дорожек.\n"
                "Увеличение допускает вытянутые/размытые via, но растит ложные на трассах.",
                "Maximum elongation. Lower rejects lines; higher tolerates stretched candidates.",
            ),
            (
                "heuristic_line_penalty_spin",
                "Штраф за похожесть на линию.\n"
                "Увеличение сильнее снижает балл кандидатов на дорожках.\n"
                "Уменьшение помогает via, слитым с проводником, но добавляет ложные вдоль линий.",
                "Line penalty. Higher suppresses trace-like detections; lower is more permissive.",
            ),
            (
                "heuristic_border_penalty_spin",
                "Штраф для кандидатов у края окна/кадра.\n"
                "Увеличение убирает неполные объекты у границ.\n"
                "Уменьшение сохраняет via около края, но может принять обрезанные артефакты.",
                "Border penalty. Higher rejects edge candidates; lower keeps edge vias.",
            ),
            (
                "heuristic_local_binarize_percentile_spin",
                "Процентиль локальной бинаризации компонента.\n"
                "Увеличение делает компонент меньше и строже по яркости.\n"
                "Уменьшение расширяет компонент и помогает слабым via, но может слить его с дорожкой.",
                "Local binarization percentile. Higher = stricter/smaller; lower = larger/more tolerant.",
            ),
            (
                "heuristic_min_abs_peak_spin",
                "Абсолютный минимум отклика пика до детальной проверки.\n"
                "Увеличение ускоряет поиск на шумных кадрах, потому что проверяется меньше seed-пиков.\n"
                "Уменьшение ищет слабые via, но может сильно замедлить обработку.",
                "Absolute seed floor. Higher is faster/stricter; lower finds weak vias but can be slower.",
            ),
            (
                "heuristic_use_bilateral_checkbox",
                "Билатеральное шумоподавление вместо медианного.\n"
                "Включение лучше сохраняет края, но обычно медленнее.\n"
                "Выключение быстрее и достаточно для большинства SEM-кадров.",
                "Use bilateral denoise. Preserves edges but is slower than median filtering.",
            ),
            (
                "via_template_min_score_spin",
                "Минимальная корреляция с шаблоном.\n"
                "Увеличение уменьшает ложные совпадения, но пропускает отличающиеся via.\n"
                "Уменьшение повышает полноту и число кандидатов.",
                "Template correlation threshold. Higher = cleaner; lower = more matches.",
            ),
            (
                "via_template_nms_distance_spin",
                "Расстояние подавления дублей для шаблонного поиска.\n"
                "Увеличение сильнее сливает близкие совпадения.\n"
                "Уменьшение сохраняет соседние via, но может давать дубли одного отверстия.",
                "Template NMS distance. Higher merges duplicates; lower keeps close matches.",
            ),
            (
                "via_template_scale_min_spin",
                "Минимальный масштаб шаблона.\n"
                "Уменьшение позволяет находить via меньше сохранённого шаблона, но добавляет лишние масштабы и замедляет поиск.\n"
                "Увеличение сужает поиск и ускоряет, но может пропустить маленькие via.",
                "Minimum template scale. Lower finds smaller vias but is slower.",
            ),
            (
                "via_template_scale_max_spin",
                "Максимальный масштаб шаблона.\n"
                "Увеличение позволяет находить via крупнее шаблона, но добавляет вычисления и ложные совпадения.\n"
                "Уменьшение ускоряет и делает поиск строже, но может пропустить крупные via.",
                "Maximum template scale. Higher finds larger vias but is slower/noisier.",
            ),
            (
                "via_template_scale_step_spin",
                "Шаг перебора масштаба шаблона.\n"
                "Увеличение ускоряет поиск, но может промахнуться по размеру.\n"
                "Уменьшение точнее, но заметно медленнее.",
                "Template scale step. Higher is faster; lower is more accurate but slower.",
            ),
        ]
        for attr, ru_text, en_text in via_help:
            widget = getattr(self, attr, None)
            if widget is not None:
                widget.setToolTip(tt(ru_text, en_text))
        if getattr(self, "metal_preset_combo", None) is not None:
            self.metal_preset_combo.setToolTip(
                tt(
                    "Готовый набор порогов и морфологии под тип слоя.\n"
                    "«Стандартный» — универсальный баланс; «Плотная металлизация» — чуть агрессивнее к шуму; "
                    "«Тонкие дорожки» — ниже минимальная ширина; «Шумное SEM» — жёстче отсев; "
                    "«Консервативный» — меньше ложных, выше пороги длины/прямолинейности.",
                    "Preset bundle for metal recovery.",
                )
            )
        if getattr(self, "metal_sensitivity_slider", None) is not None:
            self.metal_sensitivity_slider.setToolTip(
                tt(
                    "Единый регулятор чувствительности 0–100: увеличение добавляет пиксели в маску и чаще оставляет слабые дорожки, "
                    "но усиливает ложные срабатывания на зерне и артефактах; уменьшение убирает шум, но может проглотить тусклые реальные проводники.\n"
                    "Типичный диапазон 35–65; при «Шумном SEM» чаще 30–45, при контрастных кадрах 55–70.",
                    "Unified sensitivity 0–100 for internal thresholds.",
                )
            )
        if getattr(self, "metal_min_width_spin", None) is not None:
            self.metal_min_width_spin.setToolTip(
                tt(
                    "Оценка эффективной ширины по маске (медиальное ядро): объекты уже порога отбрасываются как шумовые царапины.\n"
                    "Увеличение убирает тонкие ложные сегменты, но может отрезать реальные узкие дорожки; уменьшение спасает тонкие линии, но пропускает больше мусора.\n"
                    "Стартуйте с 6–10 px для тонких технологий и 10–14 px для грубого SEM.",
                    "Minimum conductor width in pixels.",
                )
            )
        if getattr(self, "metal_max_width_spin", None) is not None:
            self.metal_max_width_spin.setToolTip(
                tt(
                    "Верхняя граница ширины: отсекает широкие заливки, контактные площадки и яркие «пятна», не являющиеся трассами.\n"
                    "0 или пусто — без ограничения. Уменьшайте максимум, если в результат попадают крупные артефакты; увеличивайте, если режет широкие шины.\n"
                    "Часто 40–120 px в зависимости от масштаба кадра.",
                    "Maximum trace width; 0 = unlimited.",
                )
            )
        if getattr(self, "metal_min_length_spin", None) is not None:
            self.metal_min_length_spin.setToolTip(
                tt(
                    "Минимальная длина по ограничивающему прямоугольнику: короткие фрагменты травления и одиночные засветы отсекаются.\n"
                    "Увеличение сильнее чистит шум; уменьшение сохраняет короткие, но реальные сегменты (перемычки, стабы).\n"
                    "Рабочий диапазон обычно 18–40 px.",
                    "Minimum trace length.",
                )
            )
        if getattr(self, "metal_use_wide_gradient_checkbox", None) is not None:
            self.metal_use_wide_gradient_checkbox.setToolTip(
                tt(
                    "Включает дополнительное восстановление широких проводников по ярким краям. Полезно для SEM, где ярко видны только границы проводника, "
                    "а центр похож на фон. Может находить широкие дорожки, которые пропускает обычная бинаризация, но при слишком шумном изображении "
                    "может добавить ложные срабатывания.",
                    "Wide conductor recovery from bright edges (SEM).",
                )
            )
        if getattr(self, "metal_wide_grad_radius_spin", None) is not None:
            self.metal_wide_grad_radius_spin.setToolTip(
                tt(
                    "Сколько пикселей по обе стороны от яркого края используется для анализа профиля яркости. Увеличение помогает для широких и размытых "
                    "проводников, но может захватывать соседние объекты.",
                    "Gradient profile half-width in pixels.",
                )
            )
        if getattr(self, "metal_wide_grad_conf_spin", None) is not None:
            self.metal_wide_grad_conf_spin.setToolTip(
                tt(
                    "Насколько явно одна сторона края похожа на фон, а другая — на внутреннюю часть проводника. Увеличение делает режим строже и уменьшает "
                    "ложные пары краёв, но может пропустить слабые проводники.",
                    "Minimum direction confidence.",
                )
            )
        if getattr(self, "metal_wide_grad_pair_len_spin", None) is not None:
            self.metal_wide_grad_pair_len_spin.setToolTip(
                tt(
                    "Минимальная длина двух параллельных границ, чтобы они считались сторонами широкого проводника. Увеличение отсекает короткие шумовые линии, "
                    "уменьшение помогает находить короткие проводники.",
                    "Minimum parallel edge length for pairing.",
                )
            )
        if getattr(self, "metal_wide_grad_parallel_spin", None) is not None:
            self.metal_wide_grad_parallel_spin.setToolTip(
                tt(
                    "Максимальное отличие углов двух границ. Меньшее значение требует почти параллельных краёв, большее допускает искажённые SEM-границы.",
                    "Parallelism tolerance in degrees.",
                )
            )
        if getattr(self, "metal_wide_grad_gap_spin", None) is not None:
            self.metal_wide_grad_gap_spin.setToolTip(
                tt(
                    "Позволяет соединять прерывистые яркие края. Увеличение помогает на шумных изображениях, но может ошибочно соединять разные объекты.",
                    "Max gap for Hough line linking.",
                )
            )
        if getattr(self, "metal_wide_grad_overlap_spin", None) is not None:
            self.metal_wide_grad_overlap_spin.setToolTip(
                tt(
                    "Минимальная доля перекрытия двух границ по длине. Увеличение делает поиск пар строже, уменьшение допускает частично видимые края.",
                    "Minimum overlap ratio of paired edges.",
                )
            )
        if getattr(self, "metal_segmentation_method_combo", None) is not None:
            self.metal_segmentation_method_combo.setToolTip(
                tt(
                    "По умолчанию — без глобальной пороговой сегментации: контуры строятся по границам на grayscale (Canny + локальная морфология), что лучше сохраняет топологию тонких проводников на SEM.\n"
                    "По умолчанию — Otsu: классическая пороговая сегментация яркости. Адаптивная — для неравномерного освещения, гибрид — объединяет границы и Otsu.",
                    "Default: Otsu thresholding. Adaptive handles uneven illumination. Hybrid combines edge mask with Otsu.",
                )
            )
        if getattr(self, "metal_sensitivity_combo", None) is not None:
            self.metal_sensitivity_combo.setToolTip(
                tt(
                    "Грубый уровень вместе со слайдером 0–100: «Низкая» — эрозия/порог жёстче, меньше ложных; «Высокая» — больше пикселей в маске.\n"
                    "Используйте как быстрый сдвиг до тонкой подстройки слайдером.",
                    "Coarse low/medium/high bias paired with slider.",
                )
            )
        if getattr(self, "metal_show_conductors_checkbox", None) is not None:
            self.metal_show_conductors_checkbox.setToolTip(
                tt("Показывать принятые полигоны проводников на сцене редактора.", "Show accepted conductor polygons.")
            )
        if getattr(self, "metal_show_rejected_checkbox", None) is not None:
            self.metal_show_rejected_checkbox.setToolTip(
                tt(
                    "Красным контуром показать отклонённые компоненты (после фильтров). Полезно понять, что алгоритм отбрасывает.",
                    "Draw rejected candidates in red.",
                )
            )
        if getattr(self, "metal_show_suspicious_checkbox", None) is not None:
            self.metal_show_suspicious_checkbox.setToolTip(
                tt(
                    "Жёлтым — объекты, прошедшие фильтр, но с пограничными углами или прямолинейностью; проверьте вручную.",
                    "Highlight borderline accepted traces in yellow.",
                )
            )
        if getattr(self, "metal_show_border_checkbox", None) is not None:
            self.metal_show_border_checkbox.setToolTip(
                tt(
                    "Синим — проводники, касающиеся края кадра (часто обрезаны SEM). Не ошибка, но требует осторожности при метриках.",
                    "Highlight border-touching traces in blue.",
                )
            )
        if getattr(self, "metal_show_mask_checkbox", None) is not None:
            self.metal_show_mask_checkbox.setToolTip(
                tt(
                    "Включить цветное наложение поверх изображения по выбранному режиму отладки (маска, контуры, фильтр и т.д.).",
                    "Enable debug / mask overlay on the image.",
                )
            )
        if getattr(self, "metal_debug_visual_combo", None) is not None:
            self.metal_debug_visual_combo.setToolTip(
                tt(
                    "Что именно рисуется в оверлее: итоговая смесь, сырая маска, контуры или этапы фильтрации.",
                    "Which debug channel is shown in the overlay.",
                )
            )
        if getattr(self, "metal_overlay_opacity_spin", None) is not None:
            self.metal_overlay_opacity_spin.setToolTip(
                tt("Прозрачность оверлея отладки/маски (0.05–1.0).", "Overlay opacity.")
            )
        if getattr(self, "metal_min_area_spin", None) is not None:
            self.metal_min_area_spin.setToolTip(
                tt(
                    "Минимальная площадь компонента в px² после бинаризации; отсекает мелкие засветы.\n"
                    "Увеличение — меньше шумовых островков; уменьшение — спасает тонкие, но короткие фрагменты.\n"
                    "Часто 40–120.",
                    "Minimum area filter.",
                )
            )
        if getattr(self, "metal_max_area_spin", None) is not None:
            self.metal_max_area_spin.setToolTip(
                tt("Максимальная площадь (0 = нет лимита); режет крупные заливки.", "Maximum area, 0 = off.")
            )
        if getattr(self, "metal_min_perimeter_spin", None) is not None:
            self.metal_min_perimeter_spin.setToolTip(
                tt("Минимальный периметр контура; дополнительный отсев «крошки» вокруг реальных трасс.", "Minimum perimeter.")
            )
        if getattr(self, "metal_max_perimeter_spin", None) is not None:
            self.metal_max_perimeter_spin.setToolTip(
                tt("Максимальный периметр (0 = нет); для отсечения огромных некорректных компонентов.", "Maximum perimeter.")
            )
        if getattr(self, "metal_epsilon_spin", None) is not None:
            self.metal_epsilon_spin.setToolTip(
                tt(
                    "Epsilon для Douglas–Peucker при упрощении цепочки контура перед проверками углов и топологии.\n"
                    "Больше — меньше вершин, устойчивее к зубцам; меньше — точнее геометрия, но шумнее углы.",
                    "Contour simplify epsilon.",
                )
            )
        if getattr(self, "metal_min_points_spin", None) is not None:
            self.metal_min_points_spin.setToolTip(
                tt("Минимальное число вершин упрощённого полигона для принятия.", "Minimum vertex count.")
            )
        if getattr(self, "metal_min_angle_spin", None) is not None:
            self.metal_min_angle_spin.setToolTip(
                tt(
                    "Подавляет острые «шипы» на контуре: вершины с меньшим внутренним углом выкидываются при упрощении.",
                    "Minimum interior angle at simplified vertices.",
                )
            )
        if getattr(self, "metal_approximation_checkbox", None) is not None:
            self.metal_approximation_checkbox.setToolTip(
                tt("Включить упрощение контура (approxPolyDP); выключите только для отладки сырой цепочки.", "Enable DP simplify.")
            )
        if getattr(self, "metal_hierarchy_combo", None) is not None:
            self.metal_hierarchy_combo.setToolTip(
                tt(
                    "Полная иерархия (RETR_TREE) учитывает вложенность контуров; только внешние — быстрее и проще, если дырки не нужны.",
                    "Contour hierarchy retrieval mode.",
                )
            )
        if getattr(self, "metal_allowed_angles_combo", None) is not None:
            self.metal_allowed_angles_combo.setToolTip(
                tt(
                    "Ограничение на углы трассировки после упрощения: ортогональ, 45°/90° или без ограничений.\n"
                    "Жёстче режим — меньше ложных изломанных контуров, но риск отсечь слегка «кривую» реальную дорожку.",
                    "Allowed routing angles.",
                )
            )
        if getattr(self, "metal_angle_tolerance_spin", None) is not None:
            self.metal_angle_tolerance_spin.setToolTip(
                tt(
                    "На сколько градусов можно отклониться от идеальных 0/45/90°, чтобы угол всё ещё считался допустимым.\n"
                    "Увеличьте при шумном крае; уменьшите, если просачиваются диагональные артефакты. Типично 5–10°.",
                    "Angular tolerance in degrees.",
                )
            )
        if getattr(self, "metal_straightness_spin", None) is not None:
            self.metal_straightness_spin.setToolTip(
                tt(
                    "Отношение «длина по minAreaRect» к периметру: низкие значения характерны для рыхлых, извилистых шумовых масок.\n"
                    "Повышение отсекает пятна и ветвистый мусор; понижение спасает сложные, но реальные формы. Старт 0.55–0.7.",
                    "Minimum straightness metric.",
                )
            )
        if getattr(self, "metal_t_junction_checkbox", None) is not None:
            self.metal_t_junction_checkbox.setToolTip(
                tt(
                    "Разрешать T-образные соединения в растровой маске (один связный компонент с разветвлением).\n"
                    "Выключение слегка ужесточает отбор по выпуклым дефектам — полезно, если шум даёт ложные «тройники» внутри одного контура.",
                    "Allow T-junction topology in mask components.",
                )
            )
        if getattr(self, "metal_border_handling_combo", None) is not None:
            self.metal_border_handling_combo.setToolTip(
                tt(
                    "«Игнорировать» — отбрасывать всё, что касается края кадра; «Принимать» — не отличать; "
                    "«Помечать» — принять, но выделить отдельно (часто обрезанные проводники).",
                    "How to treat image-border-touching components.",
                )
            )
        if getattr(self, "metal_validity_checkbox", None) is not None:
            self.metal_validity_checkbox.setToolTip(
                tt(
                    "Проверка простого замкнутого контура без самопересечений и лишних самокасаний на упрощённой цепочке.\n"
                    "Отключайте только временно для отладки сырой векторизации — иначе в выдачу могут попасть некорректные полигоны.",
                    "Validate simplified ring geometry.",
                )
            )
        if getattr(self, "metal_morph_close_spin", None) is not None:
            self.metal_morph_close_spin.setToolTip(
                tt(
                    "Радиус морфологического closing после порога: склеивает мелкие разрывы маски.\n"
                    "Держите низким (2–4), иначе сливаются близкие несвязанные объекты.",
                    "Closing radius; keep small.",
                )
            )
        if getattr(self, "metal_morph_open_spin", None) is not None:
            self.metal_morph_open_spin.setToolTip(
                tt("Opening для удаления тонкого соли-and-pepper шума; 0 — отключено.", "Opening radius, 0 = off.")
            )
        if getattr(self, "metal_preview_mask_button", None) is not None:
            self.metal_preview_mask_button.setToolTip(
                tt("Переключить оверлей на бинарную маску и включить показ.", "Jump to binary mask overlay.")
            )
        if getattr(self, "metal_reset_params_button", None) is not None:
            self.metal_reset_params_button.setToolTip(
                tt("Сбросить параметры восстановления к значениям по умолчанию.", "Reset metal parameters to defaults.")
            )

    def _update_via_size_controls_state(self) -> None:
        fixed_mode = normalize_via_size_mode(self.via_size_mode_combo.currentData()) == VIA_SIZE_MODE_FIXED
        range_widgets = [
            (self.min_via_width_label_widget, self.via_width_range_widget),
            (self.min_via_height_label_widget, self.via_height_range_widget),
        ]
        fixed_widgets = [
            (self.fixed_vias_label_widget, self.fixed_vias_widget),
        ]
        for label_widget, field_widget in range_widgets:
            if label_widget is not None:
                label_widget.setVisible(not fixed_mode)
            field_widget.setVisible(not fixed_mode)
        for label_widget, field_widget in fixed_widgets:
            if label_widget is not None:
                label_widget.setVisible(fixed_mode)
            field_widget.setVisible(fixed_mode)
        self._update_via_threshold_controls_state()

    def _update_via_threshold_controls_state(self) -> None:
        mode = normalize_via_search_mode(self.via_search_mode_combo.currentData())
        advanced = self._advanced_extraction_enabled()
        bright_enabled = mode in {VIA_SEARCH_MODE_HEURISTIC, "bright_tophat_dog"}
        blob_enabled = False
        template_enabled = mode == VIA_SEARCH_MODE_TEMPLATE
        for label_widget, field_widget in (
            (self.via_min_score_label_widget, self.via_min_score_spin),
            (self.via_min_contrast_label_widget, self.via_min_contrast_spin),
            (self.via_min_edge_coverage_label_widget, self.via_min_edge_coverage_spin),
            (self.via_spot_line_suppression_label_widget, self.via_spot_line_suppression_spin),
        ):
            if label_widget is not None:
                label_widget.setVisible(advanced and blob_enabled)
            field_widget.setVisible(advanced and blob_enabled)
        if self.via_template_min_score_label_widget is not None:
            self.via_template_min_score_label_widget.setVisible(advanced and template_enabled)
        self.via_template_min_score_spin.setVisible(advanced and template_enabled)
        if self.via_templates_label_widget is not None:
            self.via_templates_label_widget.setVisible(template_enabled)
        self.via_templates_widget.setVisible(template_enabled)
        if hasattr(self, "via_range_checkboxes_label_widget") and self.via_range_checkboxes_label_widget is not None:
            self.via_range_checkboxes_label_widget.setVisible(advanced and not bright_enabled)
        if hasattr(self, "via_range_checkboxes_widget"):
            self.via_range_checkboxes_widget.setVisible(advanced and not bright_enabled)

        white_enabled = self.via_white_range_checkbox.isChecked()
        self.via_white_range_min_spin.setEnabled(white_enabled)
        self.via_white_range_max_spin.setEnabled(white_enabled)
        if self.via_white_range_label_widget is not None:
            self.via_white_range_label_widget.setVisible(advanced and white_enabled and not bright_enabled)
        self.via_white_range_widget.setVisible(advanced and white_enabled and not bright_enabled)
        black_enabled = self.via_black_range_checkbox.isChecked()
        self.via_black_range_min_spin.setEnabled(black_enabled)
        self.via_black_range_max_spin.setEnabled(black_enabled)
        if self.via_black_range_label_widget is not None:
            self.via_black_range_label_widget.setVisible(advanced and black_enabled and not bright_enabled)
        self.via_black_range_widget.setVisible(advanced and black_enabled and not bright_enabled)
        if hasattr(self, "bright_via_group") and hasattr(self, "recognition_mode_combo"):
            self.bright_via_group.setVisible(
                self._active_extraction_profile == "vias"
                and str(self.recognition_mode_combo.currentData() or "") == "via"
            )

    def _update_extraction_profile_controls_state(self) -> None:
        rec = str(self.recognition_mode_combo.currentData() or "conductors") if hasattr(self, "recognition_mode_combo") else "conductors"
        is_via_profile = self._active_extraction_profile == "vias"
        advanced = self._advanced_extraction_enabled()
        show_legacy_via = is_via_profile and rec == "disabled"
        conductors_recognition = rec == "conductors"
        if hasattr(self, "advanced_extraction_checkbox"):
            self.advanced_extraction_checkbox.setVisible(not conductors_recognition)
        if conductors_recognition:
            self.basic_filters_group.setVisible(False)
            self.geometry_filters_group.setVisible(False)
            self.topology_group.setVisible(False)
        else:
            self.basic_filters_group.setVisible(advanced)
            self.geometry_filters_group.setVisible(advanced)
            self.topology_group.setVisible(advanced and (not is_via_profile or rec == "conductors"))
        self.conductor_group.setEnabled(False)
        self.conductor_group.setVisible(False)
        self.via_group.setEnabled(show_legacy_via)
        self.via_group.setVisible(show_legacy_via)
        advanced_via_widgets = [
            (self.via_range_checkboxes_label_widget, self.via_range_checkboxes_widget),
            (self.via_white_range_label_widget, self.via_white_range_widget),
            (self.via_black_range_label_widget, self.via_black_range_widget),
            (self.via_min_score_label_widget, self.via_min_score_spin),
            (self.via_min_contrast_label_widget, self.via_min_contrast_spin),
            (self.via_min_edge_coverage_label_widget, self.via_min_edge_coverage_spin),
            (self.via_spot_line_suppression_label_widget, self.via_spot_line_suppression_spin),
            (self.via_roundness_label_widget, self.via_roundness_spin),
        ]
        in_via_extraction = rec in ("via", "disabled")
        if hasattr(self, "contour_group"):
            if rec == "via":
                self.contour_group.setTitle("")
                self.contour_group.setFlat(True)
                self.contour_group.setStyleSheet(
                    "QGroupBox#contourExtractionGroup { border: 0; margin-top: 0; padding-top: 0; }"
                )
            else:
                self.contour_group.setTitle(self._tr("contour_extraction_group"))
                self.contour_group.setFlat(False)
                self.contour_group.setStyleSheet("")
        for label_widget, field_widget in advanced_via_widgets:
            if label_widget is not None:
                label_widget.setVisible(advanced and is_via_profile and in_via_extraction)
            field_widget.setVisible(advanced and is_via_profile and in_via_extraction)
        if hasattr(self, "bright_via_group"):
            self.bright_via_group.setVisible(is_via_profile and rec == "via")
        self._sync_recognition_stack_visibility()
        self._update_via_threshold_controls_state()

    def _sync_recognition_stack_visibility(self) -> None:
        if not hasattr(self, "recognition_mode_combo") or not hasattr(self, "recognition_stack"):
            return
        data = str(self.recognition_mode_combo.currentData() or "conductors")
        if data == "via":
            self.recognition_stack.setVisible(False)
        else:
            self.recognition_stack.setVisible(True)
            self.recognition_stack.setCurrentIndex(0 if data == "disabled" else 1)

    def _advanced_extraction_enabled(self) -> bool:
        return bool(hasattr(self, "advanced_extraction_checkbox") and self.advanced_extraction_checkbox.isChecked())

    def _on_advanced_extraction_toggled(self, *_args) -> None:
        self._update_extraction_profile_controls_state()


