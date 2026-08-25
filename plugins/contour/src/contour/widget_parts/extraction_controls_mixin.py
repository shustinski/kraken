from __future__ import annotations

from ._imports import *  # noqa: F403


class WidgetExtractionControlsMixin:
    def _renumber_fixed_via_rows(self) -> None:
        for index, row in enumerate(self._fixed_via_rows, start=1):
            label = row["label"]
            if isinstance(label, QLabel):
                label.setText(f"Контакт {index}" if self._ui_language == "ru" else f"Contact {index}")

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
            "Удаляет эту строку с допустимым размером контакта из списка."
            if self._ui_language == "ru"
            else "Removes this allowed contact-size row from the list."
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
        self._set_field_tooltip(self.min_polygon_width_label_widget, self.min_polygon_width_spin, "min_polygon_width")
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
        if getattr(self, "via_search_mode_label_widget", None) is not None:
            self._set_field_tooltip(self.via_search_mode_label_widget, self.via_search_mode_combo, "via_search_mode")
        if hasattr(self, "bright_via_viamode_label_widget"):
            self._set_field_tooltip(self.bright_via_viamode_label_widget, self.via_search_mode_combo, "via_search_mode")
        if hasattr(self, "bright_via_polarity_label_widget"):
            self._set_field_tooltip(
                self.bright_via_polarity_label_widget,
                self.via_heuristic_polarity_combo,
                "via_polarity",
            )
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
        self._set_field_tooltip(self.reset_via_search_label_widget, self.reset_via_search_button, "reset_via_search")
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
            "Добавляет ещё одну допустимую пару ширины и высоты контакта."
            if self._ui_language == "ru"
            else "Adds another allowed contact width and height pair."
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
                    "Удаляет эту строку с допустимым размером контакта из списка."
                    if self._ui_language == "ru"
                    else "Removes this allowed contact-size row from the list."
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
                "Если значение слишком большое — маленькие контакты будут пропущены.\n"
                "Если слишком маленькое — появится больше ложных срабатываний на шуме.\n"
                "Обычно: 5–8 px.",
                "Minimum contact diameter in pixels (typ. 5–8).",
            )
        )
        self.bright_via_diameter_max_spin.setToolTip(
            tt(
                "Максимальный допустимый размер контакта.\n"
                "Если слишком маленькое — крупные контакты будут пропущены.\n"
                "Если слишком большое — алгоритм начнёт принимать яркие фрагменты дорожек.\n"
                "Обычно: 8–14 px.",
                "Maximum contact diameter in pixels (typ. 8–14).",
            )
        )
        self.via_output_diameter_spin.setToolTip(
            tt(
                "Единый диаметр всех сохраняемых контактов. На поиск кандидатов не влияет.",
                "Uniform diameter of every saved contact. It does not affect candidate search.",
            )
        )
        self.bright_via_clahe_clip_spin.setToolTip(
            tt(
                "Предел усиления локального контраста (CLAHE).\n"
                "Больше значение — сильнее вытягиваются слабые детали, но растёт шум.\n"
                "Меньше — картинка ровнее, но слабые контакты могут стать незаметнее.\n"
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
                "Больше — сильнее подавление шума SEM, но мягче края контактов.\n"
                "Меньше — лучше сохраняются острые контакты, выше риск ложных точек.\n"
                "Типично 3.",
                "Median blur kernel (odd); larger removes more noise and softens edges.",
            )
        )
        self.bright_via_tophat_kernel_spin.setToolTip(
            tt(
                "Размер структурного элемента для белого top-hat (нечётное).\n"
                "Больше — подчёркиваются более крупные яркие вкрапления, фон на большей шкале.\n"
                "Меньше — чувствительнее к мелким пятнам и зерну.\n"
                "Сопоставляйте с ожидаемым диаметром контакта.",
                "White top-hat structuring size; match expected contact scale.",
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
                "Подбирайте пару с малой сигмой под размер контакта.",
                "DoG large sigma; tune with small sigma for contact size.",
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
                "Меняйте, если стабильно теряются мелкие контакты или, наоборот, много «крошек».",
                "Min area as a factor of π·(d_min/2)².",
            )
        )
        self.bright_via_max_area_factor_spin.setToolTip(
            tt(
                "Верхняя граница площади кандидата относительно площади круга максимального диаметра.\n"
                "Меньше — жёстче отсекаются крупные пятна (часто куски дорожек).\n"
                "Больше — допускаются более крупные отклики.\n"
                "Согласуйте с реальным размером контакта на SEM.",
                "Max area factor relative to max diameter.",
            )
        )
        self.bright_via_min_circularity_spin.setToolTip(
            tt(
                "Ожидаемая «круглость» контура (4π·area/perimeter²).\n"
                "Низкие значения допускают вытянутые пятна (часто артефакты дорожек).\n"
                "Высокие — ближе к диску, но реальные размытые контакты могут получать меньший балл.\n"
                "Обычно 0.15–0.45 в зависимости от качества изображения.",
                "Circularity expectation for blob shape (0–1).",
            )
        )
        self.bright_via_min_aspect_spin.setToolTip(
            tt(
                "Минимальное отношение ширины bounding box к высоте.\n"
                "Слишком большое — отсекаются слегка вытянутые контакты.\n"
                "Слишком маленькое — пропускаются сильно вытянутые ложные объекты реже.\n"
                "Для контактов обычно около 0.4–0.6.",
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
                "Минимальная абсолютная яркость ядра контакта (среднее по диску, шкала 0-255).\n"
                "Отсекает тусклые ложные срабатывания на текстуре металла и шуме,\n"
                "где настоящего контакта нет. Настоящие контакты обычно яркие (≈180-250).\n"
                "Увеличение убирает тусклые ложные пятна, но может пропустить слабые контакты.\n"
                "Это жёсткий порог: ниже — кандидат отбрасывается сразу.",
                "Hard minimum absolute contact-core brightness (0-255).",
            )
        )
        self.bright_via_max_radial_asymmetry_spin.setToolTip(
            tt(
                "Проверяет симметричность яркости вокруг контакта (СКО по 8 направлениям).\n"
                "Настоящий контакт обычно симметричен, край дорожки — нет.\n"
                "Порог задаёт, насколько большой разброс ещё считается «похожим на контакт» в мягком режиме.\n"
                "Меньше значение в мягком режиме сильнее снижает итоговую оценку при асимметрии.\n"
                "Слишком жёсткий ручной отбор (если включить жёсткий режим) ведёт к пропускам на шуме.",
                "Reference level for radial brightness asymmetry (std).",
            )
        )
        self.bright_via_max_edge_likeness_spin.setToolTip(
            tt(
                "Ограничивает срабатывания на краях металлизации.\n"
                "Меньше значение — сильнее штраф в мягком режиме за «краевой» профиль.\n"
                "Больше — терпимее к контактам у границы дорожки.\n"
                "С жёстким режимом (если включён) пары с метрикой выше порога отбрасываются сразу.",
                "Edge-likeness cap / soft scale.",
            )
        )
        self.bright_via_max_line_likeness_spin.setToolTip(
            tt(
                "Отсекает объекты, похожие на куски дорожек (анизотропия градиентов в окне).\n"
                "Большее значение — мягче к вытянутым откликам, выше риск ложных срабатываний на трассы.\n"
                "Меньшее — жёстче к линиям, но больше риск пропуска контактов, слитых с трассой.\n"
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
                "Выше — принимаются только контакты, лежащие на металлизации по маске.\n"
                "Ниже — больше кандидатов проходят, но растут ложные вне металла.\n"
                "В мягком режиме на порог ориентироваться не обязательно: используется непрерывный бонус.",
                "Min metal fraction for strict mode (0–1).",
            )
        )
        self.bright_via_min_final_score_spin.setToolTip(
            tt(
                "Главный параметр отбора итоговых контактов по суммарной оценке 0…100 (форма + локальные метрики).\n"
                "Увеличение → меньше ложных срабатываний, но больше пропусков.\n"
                "Уменьшение → больше найденных контактов, но больше кандидатов ниже порога (жёлтые на отладке).\n"
                "Обычно это один из самых важных параметров настройки.",
                "Minimum composite score (0–100) to accept a contact.",
            )
        )
        self.bright_via_nms_distance_spin.setToolTip(
            tt(
                "Минимальное расстояние между двумя кандидатами после этапа слияния и подавления дублей.\n"
                "Если слишком маленькое — один контакт может быть найден несколько раз с разных откликов.\n"
                "Если слишком большое — соседние реальные контакты могут сливаться.\n"
                "Связывайте с ожидаемым шагом растра контактов.",
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
        self.reset_bright_via_button.setToolTip(
            tt(
                "Сбрасывает параметры детектора к заводским значениям и запускает пересчёт (как при изменении настроек).",
                "Reset bright contact parameters to defaults and re-run.",
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
        if hasattr(self, "via_show_detected_checkbox"):
            self.via_show_detected_checkbox.setToolTip(
                tt(
                "Показывать на изображении автоматически найденные контакты.",
                "Show auto-detected contacts on the image.",
                )
            )
        via_help: list[tuple[str, str, str]] = [
            (
                "heuristic_background_sigma_spin",
                "Размер размытия для оценки фона перед поиском локальных пиков.\n"
                "Увеличение убирает крупный фон и помогает на плавной засветке, но может ослабить близкие контакты.\n"
                "Уменьшение делает поиск локальнее и быстрее реагирует на мелкие перепады, но чаще принимает шум.",
                "Background blur sigma. Higher removes broad illumination; lower is more local and noisier.",
            ),
            (
                "heuristic_analysis_window_scale_spin",
                "Размер окна анализа вокруг найденного пика в долях диаметра контакта.\n"
                "Увеличение даёт больше контекста для формы и кольца, но медленнее и может захватить соседние дорожки.\n"
                "Уменьшение ускоряет проверку и лучше для плотных контактов, но хуже оценивает окружение.",
                "Analysis window in contact diameters. Higher = more context/slower; lower = faster/tighter.",
            ),
            (
                "heuristic_min_center_brightness_spin",
                "Минимальная допустимая яркость пикселя в центре контакта по шкале от 0 до 255.\n"
                "Кандидат с яркостью центра ниже порога отклоняется. Увеличение оставляет только более светлые центры, "
                "уменьшение допускает более тёмные. Значение 0 отключает этот фильтр.",
                "Minimum allowed center-pixel brightness from 0 to 255. Candidates below it are rejected; zero disables the filter.",
            ),
            (
                "heuristic_min_center_contrast_spin",
                "Минимальная разница яркости центра и окружения.\n"
                "Увеличение уменьшает ложные срабатывания на слабой текстуре, но пропускает тусклые контакты.\n"
                "Уменьшение повышает полноту, но добавляет шумовые кандидаты.",
                "Minimum center-vs-surround contrast. Higher = cleaner; lower = more recall.",
            ),
            (
                "heuristic_min_peak_prominence_spin",
                "Насколько пик должен выделяться внутри локального окна.\n"
                "Увеличение отсекает плоские пятна и шум, но может потерять размытые контакты.\n"
                "Уменьшение принимает слабые пики и увеличивает число проверяемых кандидатов.",
                "Minimum local peak prominence. Higher = fewer candidates; lower = more recall/slower.",
            ),
            (
                "heuristic_min_compactness_spin",
                "Минимальная компактность локального компонента.\n"
                "Увеличение строже требует круглую/плотную форму контакта и режет вытянутые артефакты.\n"
                "Уменьшение допускает деформированные контакты, но чаще пропускает куски дорожек.",
                "Minimum component compactness. Higher = rounder; lower = more tolerant.",
            ),
            (
                "heuristic_min_circularity_spin",
                "Минимальная округлость контура от 0 до 1: 1 соответствует идеальной окружности, 0 — сильно неровной или вытянутой форме.\n"
                "Кандидат со значением ниже порога отклоняется. Увеличение оставляет только более круглые контакты, уменьшение допускает повреждённые и размытые края.\n"
                "Значение 0 отключает обязательную фильтрацию, но округлость продолжает участвовать в итоговой оценке через параметр «Вес округлости».",
                "Minimum contour circularity from 0 to 1. Values below it are rejected; 0 disables the hard filter.",
            ),
            (
                "heuristic_max_elongation_spin",
                "Максимальная вытянутость компонента.\n"
                "Уменьшение сильнее отбрасывает линии и края дорожек.\n"
                "Увеличение допускает вытянутые/размытые контакты, но растит ложные на трассах.",
                "Maximum elongation. Lower rejects lines; higher tolerates stretched candidates.",
            ),
            (
                "heuristic_line_penalty_spin",
                "Штраф за похожесть на линию.\n"
                "Увеличение сильнее снижает балл кандидатов на дорожках.\n"
                "Уменьшение помогает контактам, слитым с проводником, но добавляет ложные вдоль линий.",
                "Line penalty. Higher suppresses trace-like detections; lower is more permissive.",
            ),
            (
                "heuristic_border_penalty_spin",
                "Штраф для кандидатов у края окна/кадра.\n"
                "Увеличение убирает неполные объекты у границ.\n"
                "Уменьшение сохраняет контакты около края, но может принять обрезанные артефакты.",
                "Border penalty. Higher rejects edge candidates; lower keeps edge contacts.",
            ),
            (
                "heuristic_local_binarize_percentile_spin",
                "Процентиль локальной бинаризации компонента.\n"
                "Увеличение делает компонент меньше и строже по яркости.\n"
                "Уменьшение расширяет компонент и помогает слабым контактам, но может слить его с дорожкой.",
                "Local binarization percentile. Higher = stricter/smaller; lower = larger/more tolerant.",
            ),
            (
                "heuristic_min_abs_peak_spin",
                "Абсолютный минимум отклика пика до детальной проверки.\n"
                "Увеличение ускоряет поиск на шумных кадрах, потому что проверяется меньше seed-пиков.\n"
                "Уменьшение ищет слабые контакты, но может сильно замедлить обработку.",
                "Absolute seed floor. Higher is faster/stricter; lower finds weak contacts but can be slower.",
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
                "Увеличение уменьшает ложные совпадения, но пропускает отличающиеся контакты.\n"
                "Уменьшение повышает полноту и число кандидатов.",
                "Template correlation threshold. Higher = cleaner; lower = more matches.",
            ),
            (
                "via_template_nms_distance_spin",
                "Расстояние подавления дублей для шаблонного поиска.\n"
                "Увеличение сильнее сливает близкие совпадения.\n"
                "Уменьшение сохраняет соседние контакты, но может давать дубли одного контакта.",
                "Template NMS distance. Higher merges duplicates; lower keeps close matches.",
            ),
            (
                "via_template_scale_min_spin",
                "Минимальный масштаб шаблона.\n"
                "Уменьшение позволяет находить контакты меньше сохранённого шаблона, но добавляет лишние масштабы и замедляет поиск.\n"
                "Увеличение сужает поиск и ускоряет, но может пропустить маленькие контакты.",
                "Minimum template scale. Lower finds smaller contacts but is slower.",
            ),
            (
                "via_template_scale_max_spin",
                "Максимальный масштаб шаблона.\n"
                "Увеличение позволяет находить контакты крупнее шаблона, но добавляет вычисления и ложные совпадения.\n"
                "Уменьшение ускоряет и делает поиск строже, но может пропустить крупные контакты.",
                "Maximum template scale. Higher finds larger contacts but is slower/noisier.",
            ),
            (
                "via_template_scale_step_spin",
                "Шаг перебора масштаба шаблона.\n"
                "Увеличение ускоряет поиск, но может промахнуться по размеру.\n"
                "Уменьшение точнее, но заметно медленнее.",
                "Template scale step. Higher is faster; lower is more accurate but slower.",
            ),
        ]
        via_help.extend(
            [
                ("heuristic_size_tolerance_range_spin", "Допустимое относительное отличие найденного диаметра от проверяемого при поиске в диапазоне. 0,36 означает 36 %. Большее значение сохраняет сильнее искажённые контакты, но чаще принимает объекты неверного размера; меньшее — строже фильтрует размер.", "Allowed relative diameter error in range mode. Higher is more tolerant; lower is stricter."),
                ("heuristic_size_tolerance_fixed_spin", "Допустимое относительное отличие найденного диаметра от заданного фиксированного размера. 0,26 означает 26 %. Кандидат принимается, пока ошибка не превышает порог. Увеличение добавляет допуск, уменьшение требует точного размера.", "Allowed relative diameter error in fixed mode. Candidates above the threshold are rejected."),
                ("heuristic_max_center_drift_ratio_spin", "Максимальное смещение уточнённого центра от исходного пика, в долях диаметра. Смещение выше порога отклоняет кандидата. Большее значение терпимее к асимметрии, меньшее лучше отсекает соседние компоненты.", "Maximum center drift as a diameter fraction. Candidates above it are rejected."),
                ("heuristic_max_line_coherence_spin", "Максимальная направленность градиентов от 0 до 1: 0 — направления распределены равномерно, 1 — почти одна прямая линия. Значение выше порога отклоняется. Уменьшение сильнее подавляет дорожки, увеличение сохраняет контакты на линиях.", "Maximum gradient direction coherence, 0..1. Values above it are rejected."),
                ("heuristic_min_edge_sharpness_spin", "Минимальная резкость границы контакта относительно его контраста. Значение ниже порога отклоняется как размытое пятно. Увеличение требует резкой кромки, уменьшение помогает размытым контактам.", "Minimum edge sharpness. Values below it are rejected as diffuse spots."),
                ("heuristic_contrast_score_min_spin", "Контраст, соответствующий нулевому вкладу контраста в оценку, в уровнях яркости 0–255. Вместе с верхней границей задаёт шкалу. Увеличение делает слабые контакты менее значимыми.", "Contrast mapped to zero score contribution; must not exceed the upper bound."),
                ("heuristic_contrast_score_max_spin", "Контраст, после которого вклад контраста считается максимальным, в уровнях яркости 0–255. Должен быть не меньше нижней границы. Уменьшение быстрее насыщает оценку, увеличение лучше различает очень контрастные контакты.", "Contrast mapped to full score contribution; must be at least the lower bound."),
                ("heuristic_prominence_score_min_spin", "Минимальная выраженность локального пика для начала роста её вклада в оценку, в уровнях яркости. Это отличие пика от типичного значения в окне. Увеличение снижает балл слабых пиков.", "Local peak prominence mapped to zero contribution; must not exceed the upper bound."),
                ("heuristic_prominence_score_max_spin", "Выраженность локального пика, дающая максимальный вклад в оценку, в уровнях яркости. Должна быть не меньше нижней границы. Уменьшение быстрее насыщает вклад.", "Local peak prominence mapped to full contribution; must be at least the lower bound."),
                ("heuristic_edge_snr_score_min_spin", "Нижняя граница отношения силы края к фоновому шуму. На ней качество края даёт только минимальный вклад. Увеличение сильнее наказывает шумные и слабые границы.", "Lower edge-to-noise ratio used for score normalization."),
                ("heuristic_edge_snr_score_max_spin", "Верхняя граница отношения силы края к фоновому шуму, после которой качество края считается максимальным. Должна быть не меньше нижней границы.", "Upper edge-to-noise ratio used for score normalization."),
                ("heuristic_edge_quality_floor_spin", "Минимальная доля от 0 до 1, сохраняемая у вклада выраженности пика даже при шумном крае. Увеличение меньше учитывает качество края, уменьшение сильнее штрафует шум.", "Minimum retained peak contribution when edge quality is poor, 0..1."),
                ("heuristic_border_balance_scale_spin", "Чувствительность оценки к разнице яркости слева/справа и сверху/снизу. Увеличение сильнее снижает балл асимметричных кандидатов; уменьшение терпимее к неравномерному фону.", "Sensitivity to left/right and top/bottom intensity imbalance."),
                ("heuristic_seed_percentile_spin", "Порог отбора локальных пиков как процентиль карты отклика, 0–100 %. Принимаются пики не ниже выбранного процентиля. Увеличение оставляет меньше сильных кандидатов и ускоряет поиск, но повышает риск пропусков; уменьшение добавляет слабые кандидаты, ложные срабатывания и увеличивает время обработки.", "Response-map percentile for candidate peaks, 0..100%. Higher is stricter and faster; lower finds more weak candidates and costs more time."),
                ("heuristic_w_contrast_spin", "Вес контраста в итоговой оценке 0–100. Большее значение усиливает влияние отличия центра от окружения; меньшее делает контраст менее важным.", "Final-score contrast weight. Higher increases its influence."),
                ("heuristic_w_prominence_spin", "Вес выраженности локального пика в итоговой оценке 0–100. Большее значение усиливает влияние отчётливых пиков.", "Final-score peak prominence weight. Higher increases its influence."),
                ("heuristic_w_size_spin", "Вес соответствия найденного диаметра ожидаемому в итоговой оценке 0–100. Большее значение сильнее предпочитает заданный размер.", "Final-score expected-size weight. Higher increases its influence."),
                ("heuristic_w_compact_spin", "Вес плотности и заполненности формы в итоговой оценке 0–100. Большее значение усиливает влияние компактности.", "Final-score compactness weight. Higher increases its influence."),
                ("heuristic_w_round_spin", "Вес округлости контура в итоговой оценке 0–100. Большее значение сильнее предпочитает круглые контакты.", "Final-score roundness weight. Higher increases its influence."),
                ("heuristic_w_balance_spin", "Вес симметрии яркости вокруг центра в итоговой оценке 0–100. Большее значение усиливает влияние баланса.", "Final-score intensity-balance weight. Higher increases its influence."),
                ("heuristic_w_line_spin", "Вес вычитаемого штрафа за похожесть на прямую линию, 0–100. Большее значение сильнее подавляет кандидаты на дорожках; 0 отключает этот штраф.", "Line-penalty weight. Higher subtracts more; zero disables it."),
                ("heuristic_w_border_spin", "Вес вычитаемого штрафа за дисбаланс границы, 0–100. Большее значение сильнее снижает оценку неполных и асимметричных кандидатов; 0 отключает штраф.", "Border-imbalance penalty weight. Higher subtracts more; zero disables it."),
            ]
        )
        for attr, ru_text, en_text in via_help:
            widget = getattr(self, attr, None)
            if widget is not None:
                tooltip = tt(ru_text, en_text)
                widget.setToolTip(tooltip)
                for form_name in ("bright_via_form", "bright_via_basics_form", "bright_via_quality_form"):
                    form = getattr(self, form_name, None)
                    if form is None:
                        continue
                    label = form.labelForField(widget)
                    if label is not None:
                        label.setToolTip(tooltip)
        if getattr(self, "metal_preset_combo", None) is not None:
            self.metal_preset_combo.setToolTip(
                tt(
                    "Сценарный пресет: подготовка, сегментация, топология и геометрические фильтры.\n"
                    "«Стандартный» — чистые кадры; «Шумное SEM» — сильное подавление зерна; "
                    "«Тонкие дорожки» — узкие проводники; «Широкие заливки» — толстые полигоны.",
                    "Scenario preset for the metal recovery pipeline.",
                )
            )
        if getattr(self, "metal_noise_suppression_slider", None) is not None:
            self.metal_noise_suppression_slider.setToolTip(
                tt(
                    "Подавление зернистости SEM: выравнивание освещения и шумоподавление перед сегментацией.\n"
                    "Выше — меньше шума в маске, но возможна потеря тонких деталей.",
                    "SEM noise suppression before segmentation.",
                )
            )
        if getattr(self, "metal_min_contrast_slider", None) is not None:
            self.metal_min_contrast_slider.setToolTip(
                tt(
                    "Минимальная разница яркости проводника и фона (1–255). "
                    "Распознаются области с таким или большим контрастом. При уменьшении порога "
                    "соседние фрагменты могут объединиться, поэтому площадь маски растёт, а число контуров иногда уменьшается.",
                    "Minimum conductor-to-background brightness difference (1–255). "
                    "Areas at or above the threshold are recognized. Lowering the threshold can merge adjacent fragments, "
                    "so mask area grows even though the contour count can decrease.",
                )
            )
        if getattr(self, "metal_min_object_source_contrast_spin", None) is not None:
            self.metal_min_object_source_contrast_spin.setToolTip(
                tt(
                    "Минимальная разница медианной яркости объекта и локального фона. "
                    "Повышайте, если распознаются тёмные тени рядом с проводниками.",
                    "Minimum median brightness difference between an object and its local background. "
                    "Increase it when dark shadows beside conductors are detected.",
                )
            )
        if getattr(self, "metal_min_object_rim_contrast_spin", None) is not None:
            self.metal_min_object_rim_contrast_spin.setToolTip(
                tt(
                    "\u041c\u0438\u043d\u0438\u043c\u0430\u043b\u044c\u043d\u044b\u0439 \u043a\u043e\u043d\u0442\u0440\u0430\u0441\u0442 \u044f\u0440\u043a\u043e\u0439 \u043a\u0440\u043e\u043c\u043a\u0438, \u043f\u043e\u0434\u0442\u0432\u0435\u0440\u0436\u0434\u0430\u044e\u0449\u0435\u0439 \u0441\u043b\u0430\u0431\u043e\u043a\u043e\u043d\u0442\u0440\u0430\u0441\u0442\u043d\u044b\u0439 \u043f\u0440\u043e\u0432\u043e\u0434\u043d\u0438\u043a.",
                    "Minimum bright-rim contrast used to confirm a low-contrast conductor.",
                )
            )
        if getattr(self, "metal_min_object_rim_area_fraction_spin", None) is not None:
            self.metal_min_object_rim_area_fraction_spin.setToolTip(
                tt(
                    "\u041c\u0438\u043d\u0438\u043c\u0430\u043b\u044c\u043d\u0430\u044f \u0434\u043e\u043b\u044f \u043a\u0430\u0434\u0440\u0430, \u043f\u0440\u0438 \u043a\u043e\u0442\u043e\u0440\u043e\u0439 \u043e\u0431\u044a\u0435\u043a\u0442 \u0441\u0447\u0438\u0442\u0430\u0435\u0442\u0441\u044f \u0434\u043e\u0441\u0442\u0430\u0442\u043e\u0447\u043d\u043e \u043a\u0440\u0443\u043f\u043d\u044b\u043c \u0434\u043b\u044f \u043a\u0440\u043e\u043c\u043e\u0447\u043d\u043e\u0439 \u043f\u0440\u043e\u0432\u0435\u0440\u043a\u0438.",
                    "Minimum image-area fraction for rim-backed object filtering.",
                )
            )
        if getattr(self, "metal_min_hole_source_contrast_spin", None) is not None:
            self.metal_min_hole_source_contrast_spin.setToolTip(
                tt(
                    "Минимальная разница яркости между проводником и внутренним отверстием.",
                    "Minimum brightness difference between a conductor and an inner hole.",
                )
            )
        if getattr(self, "metal_min_hole_source_contrast_fraction_spin", None) is not None:
            self.metal_min_hole_source_contrast_fraction_spin.setToolTip(
                tt(
                    "Минимальный контраст отверстия как доля общего разделения светлого и тёмного классов.",
                    "Minimum hole contrast as a fraction of the source class separation.",
                )
            )
        if getattr(self, "metal_segmentation_strategy_combo", None) is not None:
            self.metal_segmentation_strategy_combo.setToolTip(
                tt(
                    "Алгоритм сегментации проводников.\n"
                    "«Авто» сравнивает Otsu и Watershed по количеству, плотности и связности объектов.\n"
                    "«Порог Otsu» — классическая бинаризация по яркости.\n"
                    "«Адаптивный порог» считает локальный порог в окрестности каждого пикселя "
                    "и лучше держит неравномерную засветку SEM.\n"
                    "Watershed, Random Walker, Graph Cut и Reconstruction растят ядра металла "
                    "до затравок в зазорах и лучше держат бледную заливку SEM с яркими кромками.\n"
                    "«Замкнутые границы» вообще не опирается на яркость заливки: он замыкает края "
                    "в стенки и заливает охваченные ими области. Это единственный вариант, когда "
                    "проводник по яркости неотличим от подложки и виден только его контур.",
                    "Conductor segmentation algorithm.\n"
                    "Auto compares Otsu and Watershed using object count, density, and topology.\n"
                    "Otsu threshold is classical intensity binarization.\n"
                    "Adaptive threshold uses a local neighbourhood of each pixel "
                    "and holds uneven SEM illumination better.\n"
                    "Watershed, Random Walker, Graph Cut and Reconstruction grow metal cores "
                    "until they meet gap seeds and recover pale SEM fills outlined by bright rims.\n"
                    "Closed boundary ignores the fill brightness altogether: it seals edges into "
                    "walls and fills whatever they enclose. It is the only option when a conductor "
                    "is indistinguishable from the substrate and only its outline is visible.",
                )
            )
            if getattr(self, "metal_segmentation_strategy_label_widget", None) is not None:
                self.metal_segmentation_strategy_label_widget.setToolTip(
                    self.metal_segmentation_strategy_combo.toolTip()
                )
        if getattr(self, "metal_auto_contrast_step_spin", None) is not None:
            self.metal_auto_contrast_step_spin.setToolTip(
                tt(
                    "Дополнительное увеличение минимального контраста, которое режим Auto проверяет при признаках слияния проводников. "
                    "Уточнённый результат принимается только при стабильном числе объектов и лучшем согласии с Watershed. "
                    "Значение 0 отключает эту проверку.",
                    "Additional minimum-contrast increase tested by Auto when conductor merges are suspected. "
                    "The refined result is accepted only when object count stays stable and agreement with Watershed improves. "
                    "Set to 0 to disable this check.",
                )
            )
            if getattr(self, "metal_auto_contrast_step_label_widget", None) is not None:
                self.metal_auto_contrast_step_label_widget.setToolTip(
                    self.metal_auto_contrast_step_spin.toolTip()
                )
        if getattr(self, "metal_auto_source_contrast_step_spin", None) is not None:
            self.metal_auto_source_contrast_step_spin.setToolTip(
                tt(
                    "Шаг повышения порога контраста объектов для дополнительной проверки локальной сегментации в режиме Auto. "
                    "Результат принимается только после стабилизации числа объектов и при близкой маске. "
                    "Значение 0 отключает проверку.",
                    "Source-contrast increment used to validate a local-segmentation candidate in Auto mode. "
                    "The result is accepted only after object count stabilizes and the mask remains close. "
                    "Set to 0 to disable this check.",
                )
            )
            if getattr(self, "metal_auto_source_contrast_step_label_widget", None) is not None:
                self.metal_auto_source_contrast_step_label_widget.setToolTip(
                    self.metal_auto_source_contrast_step_spin.toolTip()
                )
        if getattr(self, "metal_auto_directional_gap_bridge_spin", None) is not None:
            self.metal_auto_directional_gap_bridge_spin.setToolTip(
                tt(
                    "Радиус направленной сшивки горизонтальных разрывов в режиме Auto. "
                    "Сшивка применяется только при горизонтальной структуре кадра и подтверждении исходной яркостью. "
                    "Значение 0 отключает сшивку.",
                    "Maximum directional bridge radius for horizontal gaps in Auto mode. "
                    "It is applied only to horizontally structured frames and requires source-intensity evidence. "
                    "Set to 0 to disable bridging.",
                )
            )
            if getattr(self, "metal_auto_directional_gap_bridge_label_widget", None) is not None:
                self.metal_auto_directional_gap_bridge_label_widget.setToolTip(
                    self.metal_auto_directional_gap_bridge_spin.toolTip()
                )
        if getattr(self, "metal_auto_directional_gap_min_source_spin", None) is not None:
            self.metal_auto_directional_gap_min_source_spin.setToolTip(
                tt(
                    "Минимальная медианная яркость исходных пикселей внутри добавляемого мостика. "
                    "Повышение значения уменьшает риск слияния соседних проводников.",
                    "Minimum median source intensity inside an added bridge. "
                    "Increase it to reduce the risk of merging neighboring conductors.",
                )
            )
            if getattr(self, "metal_auto_directional_gap_min_source_label_widget", None) is not None:
                self.metal_auto_directional_gap_min_source_label_widget.setToolTip(
                    self.metal_auto_directional_gap_min_source_spin.toolTip()
                )
        if getattr(self, "metal_gap_bridge_spin", None) is not None:
            self.metal_gap_bridge_spin.setToolTip(
                tt(
                    "Сшивка разрывов в маске (morphological close): соединяет обрывы дорожек.\n"
                    "Слишком большое значение может слить соседние проводники.",
                    "Gap bridging radius in pixels.",
                )
            )
        if getattr(self, "metal_speckle_removal_spin", None) is not None:
            self.metal_speckle_removal_spin.setToolTip(
                tt(
                    "Удаление мелкого шума (morphological open): убирает одиночные пиксели и короткие отростки.",
                    "Speckle removal radius in pixels.",
                )
            )
        if getattr(self, "metal_contour_smooth_spin", None) is not None:
            self.metal_contour_smooth_spin.setToolTip(
                tt(
                    "Сглаживание контура перед векторизацией: убирает зубчатость от шума SEM.",
                    "Contour smoothing before polygon simplification.",
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
        if getattr(self, "metal_adaptive_block_spin", None) is not None:
            self.metal_adaptive_block_spin.setToolTip(
                tt(
                    "Размер окна локального порога. 0 — автоматически от размера кадра (как раньше: нечётное окно около 15–63 px).\n"
                    "Увеличение сглаживает неравномерную засветку, но может залить узкие зазоры.\n"
                    "Уменьшение лучше держит тонкие щели, но сильнее реагирует на зерно SEM.",
                    "Local-threshold window size. 0 = automatic from the frame size (odd 15–63 px).\n"
                    "Larger windows flatten uneven illumination but may fill narrow gaps.\n"
                    "Smaller windows keep thin seams but follow SEM grain more.",
                )
            )
        if getattr(self, "metal_adaptive_c_spin", None) is not None:
            self.metal_adaptive_c_spin.setToolTip(
                tt(
                    "Смещение C в формуле адаптивного порога. Положительное значение делает маску строже (меньше металла).\n"
                    "Отрицательное — мягче, больше заливки и риск слияний.",
                    "Offset C in the adaptive-threshold formula. Positive values make the mask stricter.\n"
                    "Negative values recover more fill and increase merge risk.",
                )
            )
        if getattr(self, "metal_adaptive_method_combo", None) is not None:
            self.metal_adaptive_method_combo.setToolTip(
                tt(
                    "Как считается локальный фон: взвешенное гауссово среднее или простое среднее по окну.",
                    "How the local background is estimated: Gaussian-weighted mean or plain window mean.",
                )
            )
        if getattr(self, "metal_ws_smoothing_spin", None) is not None:
            self.metal_ws_smoothing_spin.setToolTip(
                tt(
                    "Сглаживание кадра перед выбором затравок водораздела. Убирает зерно SEM, чтобы ядра и зазоры не прыгали по шуму.\n"
                    "Увеличение: границы спокойнее, меньше ложных затравок; узкие щели могут заплыть, соседние проводники — слиться.\n"
                    "Уменьшение: тонкие зазоры и мелкие детали сохраняются лучше, но шум чаще даёт рваные контуры и лишние полигоны.",
                    "Gaussian smoothing before watershed seeds.\n"
                    "Increase: calmer borders, fewer noisy seeds; narrow gaps may close and neighbouring traces may merge.\n"
                    "Decrease: keeps thin gaps and fine detail, but noise more often splits polygons.",
                )
            )
        if getattr(self, "metal_ws_core_margin_spin", None) is not None:
            self.metal_ws_core_margin_spin.setToolTip(
                tt(
                    "На сколько единиц яркости ниже порога металла начинать ядро. Ядро — внутренняя затравка, от которой растёт полигон.\n"
                    "Увеличение: ядра крупнее, бледная середина широких заливок заполняется надёжнее; слишком большое значение может перепрыгнуть узкий зазор и слить трассы.\n"
                    "Уменьшение: ядра остаются только на самых ярких кромках, полигоны реже сливаются; бледная заливка может не заполниться или развалиться на фрагменты.",
                    "How far below the metal Otsu limit a pixel still counts as a metal core seed.\n"
                    "Increase: cores grow, pale fills recover better; too high and cores jump the gap and merge traces.\n"
                    "Decrease: cores stay on the brightest rims, fewer merges; pale interiors may stay empty or fragment.",
                )
            )
        if getattr(self, "metal_ws_groove_margin_spin", None) is not None:
            self.metal_ws_groove_margin_spin.setToolTip(
                tt(
                    "На сколько единиц яркости выше порога подложки считать тёмную точку затравкой зазора. Затравка зазора останавливает рост металла.\n"
                    "Увеличение: серые канавки между трассами находятся увереннее; тёмная текстура внутри заливки чаще принимается за зазор и крошит широкий полигон.\n"
                    "Уменьшение: затравками остаются только самые тёмные канавки, широкие заливки целее; узкие или светлые зазоры могут не разделить соседние проводники.",
                    "How far above the substrate Otsu limit a pixel still counts as a gap seed that stops metal growth.\n"
                    "Increase: grey channels between traces are found more reliably; texture inside a pour is more likely to split it.\n"
                    "Decrease: only the darkest grooves remain seeds, pours stay whole; faint gaps may fail to separate neighbours.",
                )
            )
        if getattr(self, "metal_ws_rim_probe_spin", None) is not None:
            self.metal_ws_rim_probe_spin.setToolTip(
                tt(
                    "Радиус кольца вокруг тёмной затравки. Если кольцо яркое (кромки соседей), это настоящий зазор; если вокруг заливка — пятно текстуры внутри металла, затравку отбрасываем.\n"
                    "Увеличение: лучше отсекает тёмные пятна внутри широких заливок; слишком большое кольцо может «увидеть» далёкий металл и оставить ложную затравку.\n"
                    "Уменьшение: требует, чтобы яркая кромка была вплотную к канавке — узкие зазоры сохраняются, но пятна внутри заливки чаще проходят как зазоры.",
                    "How far around a dark seed we look for a bright rim. A real gap is lined with metal edges; a speck inside a pour is not.\n"
                    "Increase: drops more texture specks inside wide fills; too large and a distant rim may keep a false gap seed.\n"
                    "Decrease: only tightly rim-lined grooves survive, so narrow gaps stay; interior specks more often split a pour.",
                )
            )
        if getattr(self, "metal_ws_seed_speckle_spin", None) is not None:
            self.metal_ws_seed_speckle_spin.setToolTip(
                tt(
                    "Морфологическое opening затравок ядер и зазоров: выкидывает одиночные пиксели до водораздела.\n"
                    "Увеличение: меньше шумных затравок, контуры спокойнее; тонкие проводники и узкие канавки могут потерять затравку и пропасть либо слиться.\n"
                    "Уменьшение (0 — выкл.): сохраняет мелкие ядра и щели, но шум даёт лишние полигоны и рваные края.",
                    "Morphological opening of core and gap seeds before the watershed.\n"
                    "Increase: fewer noisy seeds and calmer outlines; thin traces or narrow grooves may lose their seed and vanish or merge.\n"
                    "Decrease (0 = off): keeps tiny cores and slits, but noise more often creates extra polygons.",
                )
            )
        if getattr(self, "metal_ws_valley_span_spin", None) is not None:
            self.metal_ws_valley_span_spin.setToolTip(
                tt(
                    "Полуширина узкого шва между близкими проводниками. Такой шов не доходит до яркости подложки, поэтому обычная затравка зазора его пропускает и трассы слипаются.\n"
                    "Увеличение: разделяются и более широкие промежутки; слишком большое значение начинает резать сами проводники по внутренней текстуре.\n"
                    "Уменьшение (0 — выкл.): ищутся только самые тонкие швы; близко идущие трассы чаще остаются одним полигоном.",
                    "Half-width of the thin seam between closely spaced conductors. Such a seam never reaches substrate brightness, so the ordinary gap seed misses it and traces fuse.\n"
                    "Increase: wider separations are split too; too large and the detector starts cutting conductors along their own texture.\n"
                    "Decrease (0 = off): only the thinnest seams are found; adjacent traces more often stay one polygon.",
                )
            )
        if getattr(self, "metal_ws_valley_depth_spin", None) is not None:
            self.metal_ws_valley_depth_spin.setToolTip(
                tt(
                    "Насколько шов должен быть темнее окружающего металла, чтобы считаться разделителем. Это и отличает настоящий зазор от тёмной текстуры внутри заливки.\n"
                    "Увеличение: разделителями считаются только явно тёмные швы, широкие заливки целее; слабые светлые швы не разделят соседние трассы.\n"
                    "Уменьшение: находятся и бледные швы; тёмная текстура внутри проводника может ошибочно разрезать полигон.",
                    "How much darker than the surrounding metal a seam must be to count as a separator. This is what distinguishes a real gap from dark texture inside a fill.\n"
                    "Increase: only clearly dark seams separate, pours stay whole; faint seams no longer split neighbours.\n"
                    "Decrease: pale seams are found too, but texture inside a conductor may wrongly cut the polygon.",
                )
            )
        if getattr(self, "metal_rw_beta_spin", None) is not None:
            self.metal_rw_beta_spin.setToolTip(
                tt(
                    "Насколько Random Walker держится перепада яркости (шкала ~32 серых уровня). Больше β — граница садится на кромку и не перетекает через слабый зазор; слишком большое значение оставляет дыры в бледной заливке.\n"
                    "Уменьшение: заливка восстанавливается смелее, но близкие проводники чаще сливаются.",
                    "Random Walker boundary stiffness β, scaled so a ~32 grey-level edge is a barrier at the default value.\n"
                    "Increase: the cut sits on the rim and leaks less through a faint gap; too high and pale fills stay hollow.\n"
                    "Decrease: fills recover more readily, but neighbouring traces merge more often.",
                )
            )
            if getattr(self, "metal_rw_beta_label_widget", None) is not None:
                self.metal_rw_beta_label_widget.setToolTip(self.metal_rw_beta_spin.toolTip())
        if getattr(self, "metal_rw_iterations_spin", None) is not None:
            self.metal_rw_iterations_spin.setToolTip(
                tt(
                    "Число итераций Якоби. Больше — границы стабильнее на широких кадрах; меньше — быстрее, но маска может не дойти до равновесия.",
                    "Jacobi iterations for Random Walker. More iterations stabilize wide frames; fewer run faster but may stop short of equilibrium.",
                )
            )
            if getattr(self, "metal_rw_iterations_label_widget", None) is not None:
                self.metal_rw_iterations_label_widget.setToolTip(self.metal_rw_iterations_spin.toolTip())
        if getattr(self, "metal_gc_iterations_spin", None) is not None:
            self.metal_gc_iterations_spin.setToolTip(
                tt(
                    "Итерации OpenCV GrabCut. Обычно хватает 3–5; больше почти не меняет контур, но замедляет предпросмотр.",
                    "OpenCV GrabCut iterations. 3–5 is typical; more rarely changes the contour and only slows the preview.",
                )
            )
            if getattr(self, "metal_gc_iterations_label_widget", None) is not None:
                self.metal_gc_iterations_label_widget.setToolTip(self.metal_gc_iterations_spin.toolTip())
        if getattr(self, "metal_recon_erode_spin", None) is not None:
            self.metal_recon_erode_spin.setToolTip(
                tt(
                    "Сжать ядра металла перед геодезической реконструкцией. 0 оставляет ядра как есть; увеличение уводит затравку от кромки и реже перепрыгивает узкий зазор.",
                    "Erode metal cores before geodesic reconstruction. 0 keeps cores as-is; increasing pulls the seed off the rim and is less likely to jump a narrow gap.",
                )
            )
            if getattr(self, "metal_recon_erode_label_widget", None) is not None:
                self.metal_recon_erode_label_widget.setToolTip(self.metal_recon_erode_spin.toolTip())
        if getattr(self, "metal_boundary_relief_spin", None) is not None:
            self.metal_boundary_relief_spin.setToolTip(
                tt(
                    "На сколько серых уровней край должен выступать над локальным фоном, чтобы считаться границей проводника. Алгоритм «Замкнутые границы» ищет пару «светлый гребень со стороны металла — тёмная канавка со стороны подложки» и заливает то, что этими стенками охвачено, поэтому работает там, где заливка не отличается по яркости от подложки.\n"
                    "Увеличение: остаются только явные края, посторонние стенки исчезают; слабые границы рвутся, и области перетекают друг в друга.\n"
                    "Уменьшение: ловятся бледные края, но шум даёт лишние стенки и дробит крупные заливки.",
                    "How many grey levels an edge must rise above the local background to count as a conductor boundary. The closed-boundary algorithm looks for the pair \"bright ridge on the metal side, dark trench on the substrate side\" and fills whatever those walls enclose, so it works where the fill has the same brightness as the substrate.\n"
                    "Increase: only distinct edges remain and spurious walls disappear, but weak boundaries break and regions leak into each other.\n"
                    "Decrease: faint edges are caught, but noise adds walls and shatters large pours.",
                )
            )
            if getattr(self, "metal_boundary_relief_label_widget", None) is not None:
                self.metal_boundary_relief_label_widget.setToolTip(
                    self.metal_boundary_relief_spin.toolTip()
                )
        if getattr(self, "metal_boundary_background_spin", None) is not None:
            self.metal_boundary_background_spin.setToolTip(
                tt(
                    "Размытие, задающее локальный фон, относительно которого меряется рельеф края. Оно определяет самую широкую структуру, которую алгоритм ещё считает границей, а не заливкой.\n"
                    "Увеличение: границы широких проводников распознаются увереннее, но у соседних тонких трасс края начинают сливаться.\n"
                    "Уменьшение: лучше разделяются плотные тонкие трассы; у широких заливок край может распасться на два отдельных вала.",
                    "The blur that defines the local background against which the edge relief is measured. It sets the widest structure the algorithm still treats as a boundary rather than as fill.\n"
                    "Increase: boundaries of wide conductors are found more reliably, but edges of neighbouring thin traces start to merge.\n"
                    "Decrease: dense thin traces separate better, while the edge of a wide pour may split into two separate walls.",
                )
            )
            if getattr(self, "metal_boundary_background_label_widget", None) is not None:
                self.metal_boundary_background_label_widget.setToolTip(
                    self.metal_boundary_background_spin.toolTip()
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
                    "Что именно рисуется в оверлее: итоговая смесь, градиенты по X/Y, градиентное поле, сырая маска, контуры или этапы фильтрации.",
                    "Which debug channel is shown in the overlay, including X/Y gradients and the gradient field.",
                )
            )
        if getattr(self, "metal_gradient_3d_button", None) is not None:
            self.metal_gradient_3d_button.setToolTip(
                tt(
                    "Открыть отдельное окно с 3D-поверхностью модуля градиента и линиями тока по направлению Sobel.",
                    "Open a separate window with a 3D magnitude surface and Sobel streamlines.",
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
                tt(
                    "Минимальный периметр контура; дополнительный отсев «крошки» вокруг реальных трасс.",
                    "Minimum perimeter.",
                )
            )
        if getattr(self, "metal_max_perimeter_spin", None) is not None:
            self.metal_max_perimeter_spin.setToolTip(
                tt(
                    "Максимальный периметр (0 = нет); для отсечения огромных некорректных компонентов.",
                    "Maximum perimeter.",
                )
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
                tt(
                    "Включить упрощение контура (approxPolyDP); выключите только для отладки сырой цепочки.",
                    "Enable DP simplify.",
                )
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
        range_widgets = [
            (self.min_via_width_label_widget, self.via_width_range_widget),
            (self.min_via_height_label_widget, self.via_height_range_widget),
        ]
        fixed_widgets = [
            (self.fixed_vias_label_widget, self.fixed_vias_widget),
        ]
        for label_widget, field_widget in range_widgets:
            if label_widget is not None:
                label_widget.setVisible(True)
            field_widget.setVisible(True)
        for label_widget, field_widget in fixed_widgets:
            if label_widget is not None:
                label_widget.setVisible(False)
            field_widget.setVisible(False)
        self._update_bright_via_diameter_controls_state()
        self._update_via_threshold_controls_state()

    def _update_bright_via_diameter_controls_state(self) -> None:
        if getattr(self, "bright_via_diameter_range_label_widget", None) is not None:
            self.bright_via_diameter_range_label_widget.setVisible(True)
        if hasattr(self, "bright_via_diameter_range_widget"):
            self.bright_via_diameter_range_widget.setVisible(True)
        if getattr(self, "via_output_diameter_label_widget", None) is not None:
            self.via_output_diameter_label_widget.setVisible(True)
        self.via_output_diameter_spin.setVisible(True)

    def _update_via_threshold_controls_state(self) -> None:
        mode = normalize_via_search_mode(self.via_search_mode_combo.currentData())
        advanced = self._advanced_extraction_enabled()
        bright_enabled = mode == VIA_SEARCH_MODE_BRIGHT_TOPHAT_DOG
        heuristic_mode = mode in (VIA_SEARCH_MODE_HEURISTIC, VIA_SEARCH_MODE_HYBRID)
        blob_enabled = False
        template_enabled = mode in (VIA_SEARCH_MODE_TEMPLATE, VIA_SEARCH_MODE_HYBRID)
        in_via_recognition = (
            hasattr(self, "recognition_mode_combo")
            and str(self.recognition_mode_combo.currentData() or "") == "via"
        )
        if hasattr(self, "polygon_editor"):
            self.polygon_editor.set_ctrl_image_region_selection_enabled(
                in_via_recognition and template_enabled
            )
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
            self.via_range_checkboxes_label_widget.setVisible(False)
        if hasattr(self, "via_range_checkboxes_widget"):
            self.via_range_checkboxes_widget.setVisible(False)

        self._update_via_brightness_range_controls_state()
        if hasattr(self, "bright_via_group") and hasattr(self, "recognition_mode_combo"):
            self.bright_via_group.setVisible(
                self._active_extraction_profile == "vias"
                and str(self.recognition_mode_combo.currentData() or "") == "via"
            )
        template_only = mode == VIA_SEARCH_MODE_TEMPLATE
        if hasattr(self, "bright_via_basics_form"):
            for row in range(self.bright_via_basics_form.rowCount()):
                self.bright_via_basics_form.setRowVisible(row, row < 2 or not template_only)
            if hasattr(self, "bright_via_mode_stack"):
                self.bright_via_basics_form.setRowVisible(self.bright_via_mode_stack, template_enabled)
        # The generic row visibility above must not re-open the diameter range
        # that is irrelevant in fixed-size mode.
        if not template_only:
            self._update_bright_via_diameter_controls_state()
        if hasattr(self, "via_heuristic_polarity_combo"):
            self.via_heuristic_polarity_combo.setVisible(heuristic_mode)
        if getattr(self, "bright_via_polarity_label_widget", None) is not None:
            self.bright_via_polarity_label_widget.setVisible(heuristic_mode)
        if getattr(self, "bright_via_mode_stack_label_widget", None) is not None:
            self.bright_via_mode_stack_label_widget.setVisible(not template_only)
        sem_mode = mode == VIA_SEARCH_MODE_BRIGHT_TOPHAT_DOG
        if hasattr(self, "bright_via_quality_group"):
            self.bright_via_quality_group.setVisible(sem_mode and not template_only)
        if hasattr(self, "bright_via_display_group"):
            self.bright_via_display_group.setVisible(not template_only)
        if hasattr(self, "bright_via_advanced_outer"):
            self.bright_via_advanced_outer.setVisible(heuristic_mode and not template_only)
        if sem_mode:
            self._update_bright_via_diameter_controls_state()

    def _update_via_brightness_range_controls_state(self) -> None:
        if not hasattr(self, "via_white_range_checkbox"):
            return

        white_checked = self.via_white_range_checkbox.isChecked()
        self.via_white_range_min_spin.setEnabled(white_checked)
        self.via_white_range_max_spin.setEnabled(white_checked)
        black_checked = self.via_black_range_checkbox.isChecked()
        self.via_black_range_min_spin.setEnabled(black_checked)
        self.via_black_range_max_spin.setEnabled(black_checked)

        in_via_recognition = (
            hasattr(self, "recognition_mode_combo") and str(self.recognition_mode_combo.currentData() or "") == "via"
        )
        show_in_basics = in_via_recognition
        self.via_white_range_checkbox.setVisible(show_in_basics)
        self.via_white_range_widget.setVisible(show_in_basics)
        self.via_black_range_checkbox.setVisible(show_in_basics)
        self.via_black_range_widget.setVisible(show_in_basics)
        if getattr(self, "bright_via_white_range_label_widget", None) is not None:
            self.bright_via_white_range_label_widget.setVisible(show_in_basics)
        if getattr(self, "bright_via_black_range_label_widget", None) is not None:
            self.bright_via_black_range_label_widget.setVisible(show_in_basics)

        # Legacy via panel duplicates (recognition disabled).
        legacy_visible = (
            not in_via_recognition and self._advanced_extraction_enabled() and self._active_extraction_profile == "vias"
        )
        if self.via_white_range_label_widget is not None:
            self.via_white_range_label_widget.setVisible(legacy_visible and white_checked)
        if self.via_black_range_label_widget is not None:
            self.via_black_range_label_widget.setVisible(legacy_visible and black_checked)

    def _update_extraction_profile_controls_state(self) -> None:
        rec = (
            str(self.recognition_mode_combo.currentData() or "conductors")
            if hasattr(self, "recognition_mode_combo")
            else "conductors"
        )
        if hasattr(self, "polygon_editor"):
            self.polygon_editor.set_contact_recognition_mode(rec == "via")
            self.polygon_editor.set_conductor_recognition_mode(rec == "conductors")
        is_via_profile = self._active_extraction_profile == "vias"
        advanced = self._advanced_extraction_enabled()
        if hasattr(self, "advanced_extraction_checkbox"):
            self.advanced_extraction_checkbox.hide()
        self.basic_filters_group.hide()
        self.geometry_filters_group.hide()
        self.topology_group.hide()
        self.conductor_group.setEnabled(False)
        self.conductor_group.hide()
        self.via_group.setEnabled(False)
        self.via_group.hide()
        advanced_via_widgets = [
            (self.via_range_checkboxes_label_widget, self.via_range_checkboxes_widget),
            (self.via_min_score_label_widget, self.via_min_score_spin),
            (self.via_min_contrast_label_widget, self.via_min_contrast_spin),
            (self.via_min_edge_coverage_label_widget, self.via_min_edge_coverage_spin),
            (self.via_spot_line_suppression_label_widget, self.via_spot_line_suppression_spin),
            (self.via_roundness_label_widget, self.via_roundness_spin),
        ]
        in_via_extraction = rec in ("via", "disabled")
        if hasattr(self, "contour_group"):
            self.contour_group.hide()
        for label_widget, field_widget in advanced_via_widgets:
            if label_widget is not None:
                label_widget.setVisible(advanced and is_via_profile and in_via_extraction and rec == "disabled")
            field_widget.setVisible(advanced and is_via_profile and in_via_extraction and rec == "disabled")
        if hasattr(self, "bright_via_group"):
            self.bright_via_group.setVisible(is_via_profile and rec == "via")
        if hasattr(self, "metal_display_group"):
            self.metal_display_group.setEnabled(rec == "conductors")
        self._sync_recognition_stack_visibility()
        self._update_via_threshold_controls_state()
        self._update_via_brightness_range_controls_state()

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
