"""build_extraction_tab section 3."""
from __future__ import annotations
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QStackedWidget,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)
from .....application.processing import (
    VIA_SEARCH_MODE_HEURISTIC,
    VIA_SEARCH_MODE_HYBRID,
    VIA_SEARCH_MODE_TEMPLATE,
)
from contour.contour_extractor import APPROXIMATION_MODE_MAP, RETRIEVAL_MODE_MAP
from .....vision.metal_recovery.strategy_registry import (
    IMPLEMENTED_NEW_STRATEGIES,
    strategy_spec,
    visible_strategy_specs,
)
from contour.ui.no_wheel_controls import NoWheelComboBox as QComboBox
from contour.ui.no_wheel_controls import NoWheelDoubleSpinBox as QDoubleSpinBox
from contour.ui.no_wheel_controls import NoWheelSpinBox as QSpinBox
from contour.ui.builders_parts.build_strategy_parameter_page import _build_strategy_parameter_page
from contour.ui.builders_parts.dock_width_combo_box import _DockWidthComboBox
from contour.ui.builders_parts.sync_metal_strategy_panel import _sync_metal_strategy_panel

def _build_extraction_tab_section_3(self, _expert_widget, _label):
    self.epsilon_relative_checkbox.stateChanged.connect(self._on_extraction_settings_changed)
    self.min_area_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.max_area_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.min_perimeter_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.min_points_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.min_polygon_width_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.max_perimeter_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.min_bbox_width_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.max_bbox_width_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.min_bbox_height_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.max_bbox_height_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.min_aspect_ratio_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.max_aspect_ratio_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.exclude_border_touching_checkbox.stateChanged.connect(self._on_extraction_settings_changed)
    self.min_solidity_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.min_extent_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.min_polygon_angle_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.conductor_gradient_checkbox.stateChanged.connect(self._on_extraction_settings_changed)
    self.conductor_gradient_min_strength_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.conductor_gradient_band_radius_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.via_search_mode_combo.currentIndexChanged.connect(self._on_via_search_method_changed)
    self.via_search_mode_combo.currentIndexChanged.connect(self._on_extraction_settings_changed)
    self.via_heuristic_polarity_combo.currentIndexChanged.connect(self._on_extraction_settings_changed)
    self.via_template_nms_distance_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.via_template_scale_min_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.via_template_scale_max_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.via_template_scale_step_spin.valueChanged.connect(self._on_extraction_settings_changed)
    for _expert_widget in self._heuristic_expert_parameter_widgets:
        if isinstance(_expert_widget, QCheckBox):
            _expert_widget.stateChanged.connect(self._on_extraction_settings_changed)
        else:
            _expert_widget.valueChanged.connect(self._on_extraction_settings_changed)
    self.via_white_range_checkbox.stateChanged.connect(self._on_via_brightness_range_changed)
    self.via_white_range_min_spin.valueChanged.connect(self._on_via_brightness_range_changed)
    self.via_white_range_max_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.via_black_range_checkbox.stateChanged.connect(self._on_via_brightness_range_changed)
    self.via_black_range_min_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.via_black_range_max_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.via_min_score_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.via_min_contrast_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.via_min_edge_coverage_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.via_spot_line_suppression_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.via_template_min_score_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.apply_via_preset_button.clicked.connect(self._apply_selected_via_preset)
    self.save_via_preset_button.clicked.connect(self._save_current_via_preset)
    self.delete_via_preset_button.clicked.connect(self._delete_selected_via_preset)
    self.reset_via_search_button.clicked.connect(self._reset_via_search_parameters)
    self.debug_candidates_checkbox.stateChanged.connect(self._on_extraction_settings_changed)
    self.bright_via_diameter_min_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.bright_via_diameter_max_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.via_output_diameter_spin.valueChanged.connect(self._on_via_output_diameter_changed)
    self.bright_via_clahe_clip_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.bright_via_clahe_tile_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.bright_via_median_kernel_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.bright_via_tophat_kernel_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.bright_via_dog_small_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.bright_via_dog_large_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.bright_via_threshold_percentile_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.bright_via_mask_combine_combo.currentIndexChanged.connect(self._on_extraction_settings_changed)
    self.bright_via_min_area_factor_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.bright_via_max_area_factor_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.bright_via_min_circularity_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.bright_via_min_aspect_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.bright_via_max_aspect_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.bright_via_bright_center_score_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.bright_via_metal_constraint_combo.currentIndexChanged.connect(self._on_extraction_settings_changed)
    self.bright_via_metal_fraction_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.bright_via_max_radial_asymmetry_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.bright_via_max_edge_likeness_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.bright_via_max_line_likeness_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.bright_via_nms_distance_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.bright_via_min_final_score_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.bright_via_show_rejected_checkbox.stateChanged.connect(self._on_extraction_settings_changed)
    self.bright_via_min_isolation_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.bright_via_hard_asym_checkbox.stateChanged.connect(self._on_extraction_settings_changed)
    self.bright_via_hard_edge_checkbox.stateChanged.connect(self._on_extraction_settings_changed)
    self.bright_via_hard_line_checkbox.stateChanged.connect(self._on_extraction_settings_changed)
    self.via_show_detected_checkbox.stateChanged.connect(self._on_via_display_settings_changed)
    self.reset_bright_via_button.clicked.connect(self._reset_bright_via_parameters)
    self.via_roundness_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.min_via_width_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.max_via_width_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.min_via_height_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.max_via_height_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.min_hierarchy_depth_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.min_inner_hole_area_spin.valueChanged.connect(self._on_min_inner_hole_area_changed)
    self.max_hierarchy_depth_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.max_hole_area_ratio_spin.valueChanged.connect(self._on_extraction_settings_changed)

    self.basic_filters_form.addRow("Retrieval mode", self.retrieval_mode_combo)
    self.retrieval_mode_label_widget = self.basic_filters_form.labelForField(self.retrieval_mode_combo)
    self.basic_filters_form.addRow("Approximation mode", self.approximation_mode_combo)
    self.approximation_mode_label_widget = self.basic_filters_form.labelForField(self.approximation_mode_combo)
    self.basic_filters_form.addRow("Epsilon", self.epsilon_row_widget)
    self.epsilon_label_widget = self.basic_filters_form.labelForField(self.epsilon_row_widget)
    self.basic_filters_form.addRow("Epsilon mode", self.epsilon_relative_checkbox)
    self.epsilon_mode_label_widget = self.basic_filters_form.labelForField(self.epsilon_relative_checkbox)
    self.basic_filters_form.addRow("Area range", self.area_range_widget)
    self.min_area_label_widget = self.basic_filters_form.labelForField(self.area_range_widget)
    self.max_area_label_widget = None
    self.basic_filters_form.addRow("Perimeter range", self.perimeter_range_widget)
    self.min_perimeter_label_widget = self.basic_filters_form.labelForField(self.perimeter_range_widget)
    self.max_perimeter_label_widget = None
    self.basic_filters_form.addRow("Min point count", self.min_points_spin)
    self.min_point_count_label_widget = self.basic_filters_form.labelForField(self.min_points_spin)
    self.basic_filters_form.addRow("Min polygon width", self.min_polygon_width_spin)
    self.min_polygon_width_label_widget = self.basic_filters_form.labelForField(self.min_polygon_width_spin)

    self.geometry_filters_form.addRow("BBox width range", self.bbox_width_range_widget)
    self.min_bbox_width_label_widget = self.geometry_filters_form.labelForField(self.bbox_width_range_widget)
    self.max_bbox_width_label_widget = None
    self.geometry_filters_form.addRow("BBox height range", self.bbox_height_range_widget)
    self.min_bbox_height_label_widget = self.geometry_filters_form.labelForField(self.bbox_height_range_widget)
    self.max_bbox_height_label_widget = None
    self.geometry_filters_form.addRow("Aspect ratio range", self.aspect_ratio_range_widget)
    self.min_aspect_ratio_label_widget = self.geometry_filters_form.labelForField(self.aspect_ratio_range_widget)
    self.max_aspect_ratio_label_widget = None
    self.geometry_filters_form.addRow("Border handling", self.exclude_border_touching_checkbox)
    self.border_handling_label_widget = self.geometry_filters_form.labelForField(self.exclude_border_touching_checkbox)
    self.geometry_filters_form.addRow("Min solidity", self.min_solidity_spin)
    self.min_solidity_label_widget = self.geometry_filters_form.labelForField(self.min_solidity_spin)
    self.geometry_filters_form.addRow("Min extent", self.min_extent_spin)
    self.min_extent_label_widget = self.geometry_filters_form.labelForField(self.min_extent_spin)
    self.geometry_filters_form.addRow("Min polygon angle", self.min_polygon_angle_spin)
    self.min_polygon_angle_label_widget = self.geometry_filters_form.labelForField(self.min_polygon_angle_spin)

    self.conductor_form.addRow("Gradient boundaries", self.conductor_gradient_checkbox)
    self.conductor_gradient_enabled_label_widget = self.conductor_form.labelForField(self.conductor_gradient_checkbox)
    self.conductor_form.addRow("Min edge", self.conductor_gradient_min_strength_spin)
    self.conductor_gradient_min_strength_label_widget = self.conductor_form.labelForField(
        self.conductor_gradient_min_strength_spin
    )
    self.conductor_form.addRow("Boundary band", self.conductor_gradient_band_radius_spin)
    self.conductor_gradient_band_radius_label_widget = self.conductor_form.labelForField(
        self.conductor_gradient_band_radius_spin
    )

    self.via_size_mode_label_widget = None
    self.via_search_mode_label_widget = None
    self.via_form.addRow("Polarity", self.via_range_checkboxes_widget)
    self.via_range_checkboxes_label_widget = self.via_form.labelForField(self.via_range_checkboxes_widget)
    self.via_form.addRow("White range", self.via_white_range_widget)
    self.via_white_range_label_widget = self.via_form.labelForField(self.via_white_range_widget)
    self.via_form.addRow("Black range", self.via_black_range_widget)
    self.via_black_range_label_widget = self.via_form.labelForField(self.via_black_range_widget)
    self.via_form.addRow("Min score", self.via_min_score_spin)
    self.via_min_score_label_widget = self.via_form.labelForField(self.via_min_score_spin)
    self.via_form.addRow("Min contrast", self.via_min_contrast_spin)
    self.via_min_contrast_label_widget = self.via_form.labelForField(self.via_min_contrast_spin)
    self.via_form.addRow("Min edge coverage", self.via_min_edge_coverage_spin)
    self.via_min_edge_coverage_label_widget = self.via_form.labelForField(self.via_min_edge_coverage_spin)
    self.via_form.addRow("Spot trace suppression", self.via_spot_line_suppression_spin)
    self.via_spot_line_suppression_label_widget = self.via_form.labelForField(self.via_spot_line_suppression_spin)
    self.via_template_min_score_label_widget = None
    self.via_templates_label_widget = None
    self.via_form.addRow("Reset", self.reset_via_search_button)
    self.reset_via_search_label_widget = self.via_form.labelForField(self.reset_via_search_button)
    self.bright_via_basics_form.addRow("Метод поиска контактов", self.via_search_mode_combo)
    self.bright_via_viamode_label_widget = self.bright_via_basics_form.labelForField(self.via_search_mode_combo)
    self.bright_via_basics_form.addRow(self.bright_via_mode_stack)
    self.bright_via_mode_stack_label_widget = None
    self.bright_via_basics_form.addRow("Полярность контакта", self.via_heuristic_polarity_combo)
    self.bright_via_polarity_label_widget = self.bright_via_basics_form.labelForField(self.via_heuristic_polarity_combo)
    self.bright_via_diameter_mode_label_widget = None
    self.bright_via_diameter_fixed_label_widget = None
    self.bright_via_basics_form.addRow("Диаметр поиска", self.bright_via_diameter_range_widget)
    self.bright_via_diameter_range_label_widget = self.bright_via_basics_form.labelForField(
        self.bright_via_diameter_range_widget
    )
    self.bright_via_basics_form.addRow("Диаметр сохранения", self.via_output_diameter_spin)
    self.via_output_diameter_label_widget = self.bright_via_basics_form.labelForField(self.via_output_diameter_spin)
    self.bright_via_basics_form.addRow(self.via_white_range_checkbox, self.via_white_range_widget)
    self.bright_via_white_range_label_widget = self.bright_via_basics_form.labelForField(self.via_white_range_widget)
    self.bright_via_basics_form.addRow(self.via_black_range_checkbox, self.via_black_range_widget)
    self.bright_via_black_range_label_widget = self.bright_via_basics_form.labelForField(self.via_black_range_widget)
    self.bright_via_basics_form.addRow("Минимальная итоговая оценка", self.bright_via_min_final_score_spin)
    self.bright_via_basics_form.addRow("Расстояние подавления дублей", self.bright_via_nms_distance_spin)
    self.bright_via_basics_form.addRow("Пресет поиска", self.via_preset_widget)
    self.via_preset_label_widget = self.bright_via_basics_form.labelForField(self.via_preset_widget)
    self.bright_via_preset_actions_widget = QWidget()
    _via_preset_actions_layout = QHBoxLayout(self.bright_via_preset_actions_widget)
    _via_preset_actions_layout.setContentsMargins(0, 0, 0, 0)
    _via_preset_actions_layout.addWidget(self.apply_via_preset_button)
    _via_preset_actions_layout.addWidget(self.save_via_preset_button)
    _via_preset_actions_layout.addWidget(self.delete_via_preset_button)
    self.bright_via_basics_form.addRow("Управление пресетом", self.bright_via_preset_actions_widget)
    self.bright_via_basics_form.addRow("Действия", self.reset_bright_via_button)
    self.bright_via_basics_label_width = 190
    self.bright_via_basics_editor_width = 136
    for _basic_editor in (
        self.via_search_mode_combo,
        self.via_heuristic_polarity_combo,
        self.via_output_diameter_spin,
        self.bright_via_min_final_score_spin,
        self.bright_via_nms_distance_spin,
    ):
        _basic_editor.setFixedWidth(self.bright_via_basics_editor_width)
    for _row in range(self.bright_via_basics_form.rowCount()):
        _label_item = self.bright_via_basics_form.itemAt(_row, QFormLayout.ItemRole.LabelRole)
        _label_widget = _label_item.widget() if _label_item is not None else None
        if _label_widget is None:
            continue
        _label_widget.setFixedWidth(self.bright_via_basics_label_width)
        if isinstance(_label_widget, QLabel):
            _label_widget.setWordWrap(True)
    self.bright_via_quality_form.addRow("Порог яркости пятна", self.bright_via_threshold_percentile_spin)
    self.bright_via_quality_form.addRow("Мин. круглость", self.bright_via_min_circularity_spin)
    self.bright_via_quality_form.addRow("Мин. изолированность пятна", self.bright_via_min_isolation_spin)
    self.bright_via_quality_form.addRow("", self.bright_via_hard_line_checkbox)
    self.bright_via_quality_form.addRow("", self.bright_via_hard_edge_checkbox)
    self.bright_via_display_form.addRow(self.via_show_detected_checkbox)
    self.bright_via_display_form.addRow(self.debug_candidates_checkbox)
    self.debug_candidates_label_widget = None
    self.bright_via_display_form.addRow(self.bright_via_show_rejected_checkbox)
    self.bright_via_display_form.addRow("Вид", self.gradient_overlay_mode_combo)
    self.gradient_overlay_label_widget = self.bright_via_display_form.labelForField(self.gradient_overlay_mode_combo)
    self.bright_via_expert_candidates_header = QLabel("<b>Генерация кандидатов</b>")
    self.bright_via_form.addRow(self.bright_via_expert_candidates_header)
    self.bright_via_form.addRow("Сигма коррекции фона", self.heuristic_background_sigma_spin)
    self.bright_via_form.addRow("Размер окна анализа (множитель диаметра)", self.heuristic_analysis_window_scale_spin)
    self.bright_via_form.addRow("Минимальная яркость центра", self.heuristic_min_center_brightness_spin)
    self.bright_via_form.addRow("Минимальный контраст центра", self.heuristic_min_center_contrast_spin)
    self.bright_via_form.addRow("Минимальная выраженность пика", self.heuristic_min_peak_prominence_spin)
    self.bright_via_form.addRow("Локальный процентиль бинаризации", self.heuristic_local_binarize_percentile_spin)
    self.bright_via_form.addRow("Минимальная яркость пика (карта отклика)", self.heuristic_min_abs_peak_spin)
    self.bright_via_form.addRow(self.heuristic_use_bilateral_checkbox)
    self.bright_via_form.addRow("Порог пиков", self.heuristic_seed_percentile_spin)
    self.bright_via_expert_geometry_header = QLabel("<b>Геометрия и обязательные фильтры</b>")
    self.bright_via_form.addRow(self.bright_via_expert_geometry_header)
    self.bright_via_form.addRow("Минимальная компактность", self.heuristic_min_compactness_spin)
    self.bright_via_form.addRow("Минимальная округлость формы", self.heuristic_min_circularity_spin)
    self.bright_via_form.addRow("Максимальная вытянутость", self.heuristic_max_elongation_spin)
    self.bright_via_form.addRow("Допуск размера, диапазон", self.heuristic_size_tolerance_range_spin)
    self.bright_via_form.addRow("Допуск размера, фиксированный", self.heuristic_size_tolerance_fixed_spin)
    self.bright_via_form.addRow("Допустимое смещение центра", self.heuristic_max_center_drift_ratio_spin)
    self.bright_via_form.addRow("Макс. направленность границ", self.heuristic_max_line_coherence_spin)
    self.bright_via_form.addRow("Мин. резкость края", self.heuristic_min_edge_sharpness_spin)
    self.bright_via_expert_score_header = QLabel("<b>Итоговая оценка</b>")
    self.bright_via_form.addRow(self.bright_via_expert_score_header)
    self.bright_via_form.addRow("Множитель штрафа за линию", self.heuristic_line_penalty_spin)
    self.bright_via_form.addRow("Множитель штрафа за границу", self.heuristic_border_penalty_spin)
    self.bright_via_form.addRow("Контраст: нижняя граница оценки", self.heuristic_contrast_score_min_spin)
    self.bright_via_form.addRow("Контраст: верхняя граница оценки", self.heuristic_contrast_score_max_spin)
    self.bright_via_form.addRow("Выраженность: нижняя граница", self.heuristic_prominence_score_min_spin)
    self.bright_via_form.addRow("Выраженность: верхняя граница", self.heuristic_prominence_score_max_spin)
    self.bright_via_form.addRow("Край/шум: нижняя граница", self.heuristic_edge_snr_score_min_spin)
    self.bright_via_form.addRow("Край/шум: верхняя граница", self.heuristic_edge_snr_score_max_spin)
    self.bright_via_form.addRow("Минимальный вклад качества края", self.heuristic_edge_quality_floor_spin)
    self.bright_via_form.addRow("Чувствительность к дисбалансу", self.heuristic_border_balance_scale_spin)
    for _label, _attr in (
        ("Вес контраста", "heuristic_w_contrast_spin"),
        ("Вес выраженности пика", "heuristic_w_prominence_spin"),
        ("Вес соответствия размеру", "heuristic_w_size_spin"),
        ("Вес компактности", "heuristic_w_compact_spin"),
        ("Вес округлости", "heuristic_w_round_spin"),
        ("Вес баланса", "heuristic_w_balance_spin"),
        ("Вес штрафа за линию", "heuristic_w_line_spin"),
        ("Вес штрафа за границу", "heuristic_w_border_spin"),
    ):
        self.bright_via_form.addRow(_label, getattr(self, _attr))
    self.heuristic_expert_label_width = 268
    for _expert_widget in self._heuristic_expert_parameter_widgets:
        if not isinstance(_expert_widget, (QSpinBox, QDoubleSpinBox)):
            continue
        _expert_label = self.bright_via_form.labelForField(_expert_widget)
        if isinstance(_expert_label, QLabel):
            _expert_label.setFixedWidth(self.heuristic_expert_label_width)
            _expert_label.setWordWrap(True)
    self.heuristic_defaults_button = QPushButton("По умолчанию")
    self.heuristic_defaults_button.setToolTip(
        "Восстанавливает значения по умолчанию только для экспертных параметров "
        "эвристического распознавания. Основные настройки, полярность, размеры "
        "контактов и шаблоны не изменяются."
    )
    self.heuristic_defaults_button.clicked.connect(self._reset_heuristic_parameters)
    self.bright_via_form.addRow(self.heuristic_defaults_button)
    self.recognition_mode_combo = _DockWidthComboBox()
    self.recognition_mode_combo.addItem("Отключено", "disabled")
    self.recognition_mode_combo.addItem("Проводники", "conductors")
    self.recognition_mode_combo.addItem("Контакты", "via")
    self.recognition_mode_combo.setCurrentIndex(1)
    _rm_row = QHBoxLayout()
    self.recognition_mode_label = QLabel("Распознавание")
    _rm_row.addWidget(self.recognition_mode_label)
    _rm_row.addWidget(self.recognition_mode_combo, 1)
    self.recognition_stack = QStackedWidget()
    self.recognition_page_off = QWidget()
    _off_l = QVBoxLayout(self.recognition_page_off)
    self.recognition_page_conductors = QWidget()
    _cond_l = QVBoxLayout(self.recognition_page_conductors)
    _cond_l.setContentsMargins(0, 0, 0, 0)
    _cond_l.setSpacing(8)
    self.metal_preset_group = QGroupBox("Пресеты распознавания")
    _metal_preset_outer = QVBoxLayout(self.metal_preset_group)
    _metal_preset_outer.setContentsMargins(8, 8, 8, 8)
    _metal_preset_outer.setSpacing(6)
    self.metal_preset_combo = QComboBox()
    self.apply_metal_preset_button = QPushButton("Применить")
    self.save_metal_preset_button = QPushButton("Сохранить")
    self.delete_metal_preset_button = QPushButton("Удалить")
    self.export_metal_preset_button = QPushButton("Выгрузить")
    self.import_metal_preset_button = QPushButton("Загрузить")
    self.metal_preset_widget = QWidget()
    _metal_preset_layout = QGridLayout(self.metal_preset_widget)
    _metal_preset_layout.setContentsMargins(0, 0, 0, 0)
    _metal_preset_layout.setHorizontalSpacing(6)
    _metal_preset_layout.setVerticalSpacing(6)
    _metal_preset_layout.addWidget(self.metal_preset_combo, 0, 0, 1, 3)
    _metal_preset_layout.addWidget(self.apply_metal_preset_button, 1, 0)
    _metal_preset_layout.addWidget(self.save_metal_preset_button, 1, 1)
    _metal_preset_layout.addWidget(self.delete_metal_preset_button, 1, 2)
    _metal_preset_layout.addWidget(self.export_metal_preset_button, 2, 0)
    _metal_preset_layout.addWidget(self.import_metal_preset_button, 2, 1)
    _metal_preset_outer.addWidget(self.metal_preset_widget)
    self._refresh_metal_preset_combo()

    self.metal_basic_group = QGroupBox("Основные параметры")
    _metal_basic_form = QFormLayout(self.metal_basic_group)

    self.metal_min_contrast_slider = QSlider(Qt.Orientation.Horizontal)
    self.metal_min_contrast_slider.setRange(1, 255)
    self.metal_min_contrast_slider.setPageStep(5)
    self.metal_min_contrast_slider.setTickInterval(25)
    self.metal_min_contrast_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
    self.metal_min_contrast_slider.setValue(50)
    self.metal_min_contrast_value_label = QLabel("50")
    return _cond_l, _metal_basic_form, _rm_row
