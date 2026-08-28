"""Retranslate the bright-contact search panel that builders create in Russian."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from PyQt6.QtWidgets import QComboBox, QFormLayout, QWidget

if TYPE_CHECKING:
    from contour.widget import PolygonExtractionWidget

_POLARITY_LABELS = {
    "bright": ("Светлые", "Bright"),
    "dark": ("Тёмные", "Dark"),
    "ring_light_ring": ("Светлое кольцо / тёмный центр", "Bright ring / dark center"),
    "ring_dark_ring": ("Тёмное кольцо / светлый центр", "Dark ring / bright center"),
    "auto": ("Авто", "Auto"),
}
_MASK_COMBINE_LABELS = {
    "OR": ("ИЛИ", "OR"),
    "AND": ("И", "AND"),
}
_METAL_CONSTRAINT_LABELS = {
    "disabled": ("Отключено", "Disabled"),
    "soft": ("Мягкая оценка", "Soft score"),
    "strict": ("Жёсткий фильтр", "Hard filter"),
}


def _text(language: str, russian: str, english: str) -> str:
    return russian if language == "ru" else english


def _set_combo_item_texts(combo: QComboBox, labels: dict[str, tuple[str, str]], language: str) -> None:
    for index in range(combo.count()):
        pair = labels.get(str(combo.itemData(index) or ""))
        if pair is not None:
            combo.setItemText(index, _text(language, *pair))


def _set_form_field_label(container: QWidget | None, field: object, text: str) -> None:
    if container is None:
        return
    form = container.layout()
    if not isinstance(form, QFormLayout):
        return
    label = form.labelForField(field)
    if label is not None:
        label.setText(text)


def retranslate_bright_via_panel(self: PolygonExtractionWidget) -> None:
    if not hasattr(self, "bright_via_basics_group"):
        return
    language = str(self._ui_language)
    self.bright_via_basics_group.setTitle(_text(language, "Основные параметры", "Basic parameters"))
    self.bright_via_quality_group.setTitle(_text(language, "Фильтры качества", "Quality filters"))
    self.bright_via_display_group.setTitle(_text(language, "Отображение", "Display"))
    self.bright_via_advanced_outer.setTitle(_text(language, "Экспертные параметры", "Expert parameters"))
    if getattr(self, "bright_via_mode_hint", None) is not None:
        self.bright_via_mode_hint.setText(
            _text(
                language,
                "Оптимизирован для светлых круглых контактов на SEM-кадрах. Полярность не требуется.",
                "Optimized for bright round contacts on SEM frames. Polarity is not required.",
            )
        )
    _set_combo_item_texts(self.via_heuristic_polarity_combo, _POLARITY_LABELS, language)
    _set_combo_item_texts(self.bright_via_mask_combine_combo, _MASK_COMBINE_LABELS, language)
    _set_combo_item_texts(self.bright_via_metal_constraint_combo, _METAL_CONSTRAINT_LABELS, language)
    self.bright_via_show_rejected_checkbox.setText(
        _text(language, "Показывать отклонённые кандидаты", "Show rejected candidates")
    )
    self.bright_via_hard_asym_checkbox.setText(
        _text(language, "Жёстко фильтровать по асимметрии", "Hard-filter by asymmetry")
    )
    self.bright_via_hard_edge_checkbox.setText(
        _text(language, "Подавление границ падов", "Suppress pad edges")
    )
    self.bright_via_hard_line_checkbox.setText(_text(language, "Подавление дорожек", "Suppress traces"))
    self.heuristic_use_bilateral_checkbox.setText(
        _text(language, "Bilateral вместо медианы", "Bilateral instead of median")
    )
    self.reset_bright_via_button.setText(_text(language, "Сбросить параметры", "Reset parameters"))
    if hasattr(self, "heuristic_defaults_button"):
        self.heuristic_defaults_button.setText(_text(language, "По умолчанию", "Defaults"))
        self.heuristic_defaults_button.setToolTip(
            _text(
                language,
                "Восстанавливает значения по умолчанию только для экспертных параметров "
                "эвристического распознавания. Основные настройки, полярность, размеры "
                "контактов и шаблоны не изменяются.",
                "Restores defaults only for heuristic expert parameters. Basic settings, "
                "polarity, contact sizes and templates stay unchanged.",
            )
        )
    _set_form_field_label(
        self.bright_via_basics_group,
        self.bright_via_min_final_score_spin,
        _text(language, "Минимальная итоговая оценка", "Minimum final score"),
    )
    _set_form_field_label(
        self.bright_via_basics_group,
        self.bright_via_nms_distance_spin,
        _text(language, "Расстояние подавления дублей", "Duplicate suppression distance"),
    )
    _set_form_field_label(
        self.bright_via_basics_group,
        self.via_preset_widget,
        _text(language, "Пресет поиска", "Search preset"),
    )
    _set_form_field_label(
        self.bright_via_basics_group,
        self.reset_bright_via_button,
        _text(language, "Действия", "Actions"),
    )
    preset_actions = getattr(self, "bright_via_preset_actions_widget", None)
    if preset_actions is not None:
        _set_form_field_label(
            self.bright_via_basics_group,
            preset_actions,
            _text(language, "Управление пресетом", "Preset controls"),
        )
    _set_form_field_label(
        self.bright_via_quality_group,
        self.bright_via_threshold_percentile_spin,
        _text(language, "Порог яркости пятна", "Spot brightness percentile"),
    )
    _set_form_field_label(
        self.bright_via_quality_group,
        self.bright_via_min_circularity_spin,
        _text(language, "Мин. круглость", "Minimum circularity"),
    )
    _set_form_field_label(
        self.bright_via_quality_group,
        self.bright_via_min_isolation_spin,
        _text(language, "Мин. изолированность пятна", "Minimum spot isolation"),
    )
    if getattr(self, "bright_via_expert_candidates_header", None) is not None:
        self.bright_via_expert_candidates_header.setText(
            _text(language, "<b>Генерация кандидатов</b>", "<b>Candidate generation</b>")
        )
    if getattr(self, "bright_via_expert_geometry_header", None) is not None:
        self.bright_via_expert_geometry_header.setText(
            _text(language, "<b>Геометрия и обязательные фильтры</b>", "<b>Geometry and required filters</b>")
        )
    if getattr(self, "bright_via_expert_score_header", None) is not None:
        self.bright_via_expert_score_header.setText(
            _text(language, "<b>Итоговая оценка</b>", "<b>Final score</b>")
        )
    expert_labels: tuple[tuple[Any, str, str], ...] = (
        (self.heuristic_background_sigma_spin, "Сигма коррекции фона", "Background correction sigma"),
        (
            self.heuristic_analysis_window_scale_spin,
            "Размер окна анализа (множитель диаметра)",
            "Analysis window size (diameter multiplier)",
        ),
        (self.heuristic_min_center_brightness_spin, "Минимальная яркость центра", "Minimum center brightness"),
        (self.heuristic_min_center_contrast_spin, "Минимальный контраст центра", "Minimum center contrast"),
        (self.heuristic_min_peak_prominence_spin, "Минимальная выраженность пика", "Minimum peak prominence"),
        (
            self.heuristic_local_binarize_percentile_spin,
            "Локальный процентиль бинаризации",
            "Local binarization percentile",
        ),
        (
            self.heuristic_min_abs_peak_spin,
            "Минимальная яркость пика (карта отклика)",
            "Minimum peak brightness (response map)",
        ),
        (self.heuristic_seed_percentile_spin, "Порог пиков", "Peak threshold"),
        (self.heuristic_min_compactness_spin, "Минимальная компактность", "Minimum compactness"),
        (self.heuristic_min_circularity_spin, "Минимальная округлость формы", "Minimum shape circularity"),
        (self.heuristic_max_elongation_spin, "Максимальная вытянутость", "Maximum elongation"),
        (self.heuristic_size_tolerance_range_spin, "Допуск размера, диапазон", "Size tolerance, range"),
        (self.heuristic_size_tolerance_fixed_spin, "Допуск размера, фиксированный", "Size tolerance, fixed"),
        (self.heuristic_max_center_drift_ratio_spin, "Допустимое смещение центра", "Allowed center drift"),
        (self.heuristic_max_line_coherence_spin, "Макс. направленность границ", "Maximum edge directionality"),
        (self.heuristic_min_edge_sharpness_spin, "Мин. резкость края", "Minimum edge sharpness"),
        (self.heuristic_line_penalty_spin, "Множитель штрафа за линию", "Line penalty multiplier"),
        (self.heuristic_border_penalty_spin, "Множитель штрафа за границу", "Border penalty multiplier"),
        (self.heuristic_contrast_score_min_spin, "Контраст: нижняя граница оценки", "Contrast: score floor"),
        (self.heuristic_contrast_score_max_spin, "Контраст: верхняя граница оценки", "Contrast: score ceiling"),
        (self.heuristic_prominence_score_min_spin, "Выраженность: нижняя граница", "Prominence: score floor"),
        (self.heuristic_prominence_score_max_spin, "Выраженность: верхняя граница", "Prominence: score ceiling"),
        (self.heuristic_edge_snr_score_min_spin, "Край/шум: нижняя граница", "Edge/noise: score floor"),
        (self.heuristic_edge_snr_score_max_spin, "Край/шум: верхняя граница", "Edge/noise: score ceiling"),
        (self.heuristic_edge_quality_floor_spin, "Минимальный вклад качества края", "Minimum edge-quality contribution"),
        (self.heuristic_border_balance_scale_spin, "Чувствительность к дисбалансу", "Imbalance sensitivity"),
        (self.heuristic_w_contrast_spin, "Вес контраста", "Contrast weight"),
        (self.heuristic_w_prominence_spin, "Вес выраженности пика", "Peak prominence weight"),
        (self.heuristic_w_size_spin, "Вес соответствия размеру", "Size-match weight"),
        (self.heuristic_w_compact_spin, "Вес компактности", "Compactness weight"),
        (self.heuristic_w_round_spin, "Вес округлости", "Roundness weight"),
        (self.heuristic_w_balance_spin, "Вес баланса", "Balance weight"),
        (self.heuristic_w_line_spin, "Вес штрафа за линию", "Line-penalty weight"),
        (self.heuristic_w_border_spin, "Вес штрафа за границу", "Border-penalty weight"),
        (self.bright_via_clahe_clip_spin, "CLAHE clip", "CLAHE clip"),
        (self.bright_via_clahe_tile_spin, "CLAHE плитка", "CLAHE tile"),
        (self.bright_via_median_kernel_spin, "Ядро медианы", "Median kernel"),
        (self.bright_via_tophat_kernel_spin, "Ядро Top-hat", "Top-hat kernel"),
        (self.bright_via_dog_small_spin, "DoG σ малая", "DoG σ small"),
        (self.bright_via_dog_large_spin, "DoG σ большая", "DoG σ large"),
        (self.bright_via_min_area_factor_spin, "Мин. площадь (×)", "Min area (×)"),
        (self.bright_via_max_area_factor_spin, "Макс. площадь (×)", "Max area (×)"),
        (self.bright_via_min_aspect_spin, "Мин. соотношение сторон", "Min aspect ratio"),
        (self.bright_via_max_aspect_spin, "Макс. соотношение сторон", "Max aspect ratio"),
        (self.bright_via_bright_center_score_spin, "Яркость центра", "Center brightness"),
        (self.bright_via_metal_constraint_combo, "Ограничение по металлу", "Metal constraint"),
        (self.bright_via_metal_fraction_spin, "Доля металла", "Metal fraction"),
        (self.bright_via_max_radial_asymmetry_spin, "Макс. радиальная асимметрия", "Max radial asymmetry"),
        (self.bright_via_max_edge_likeness_spin, "Макс. похожесть на край", "Max edge likeness"),
        (self.bright_via_max_line_likeness_spin, "Макс. похожесть на линию", "Max line likeness"),
        (self.bright_via_mask_combine_combo, "Объединение масок", "Mask combine"),
    )
    expert_form_host = getattr(self, "bright_via_advanced_inner", None)
    for field, russian, english in expert_labels:
        _set_form_field_label(expert_form_host, field, _text(language, russian, english))
        _set_form_field_label(self.bright_via_quality_group, field, _text(language, russian, english))
        _set_form_field_label(self.bright_via_basics_group, field, _text(language, russian, english))
    if getattr(self, "metal_min_width_caption", None) is not None:
        self.metal_min_width_caption.setText(_text(language, "Мин.", "Min"))
    if getattr(self, "metal_max_width_caption", None) is not None:
        self.metal_max_width_caption.setText(_text(language, "Макс.", "Max"))
