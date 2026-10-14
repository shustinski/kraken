"""build_extraction_tab section 2."""
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

def _build_extraction_tab_section_2(self):
    self.bright_via_diameter_range_widget = self._build_range_row(
        self.bright_via_diameter_min_spin, self.bright_via_diameter_max_spin
    )
    self.bright_via_clahe_clip_spin = QDoubleSpinBox()
    self.bright_via_clahe_clip_spin.setRange(0.01, 20.0)
    self.bright_via_clahe_clip_spin.setDecimals(2)
    self.bright_via_clahe_clip_spin.setSingleStep(0.1)
    self.bright_via_clahe_clip_spin.setValue(2.0)
    self.bright_via_clahe_tile_spin = QSpinBox()
    self.bright_via_clahe_tile_spin.setRange(1, 128)
    self.bright_via_clahe_tile_spin.setValue(8)
    self.bright_via_median_kernel_spin = QSpinBox()
    self.bright_via_median_kernel_spin.setRange(1, 51)
    self.bright_via_median_kernel_spin.setSingleStep(2)
    self.bright_via_median_kernel_spin.setValue(3)
    self.bright_via_tophat_kernel_spin = QSpinBox()
    self.bright_via_tophat_kernel_spin.setRange(3, 101)
    self.bright_via_tophat_kernel_spin.setSingleStep(2)
    self.bright_via_tophat_kernel_spin.setValue(11)
    self.bright_via_dog_small_spin = QDoubleSpinBox()
    self.bright_via_dog_small_spin.setRange(0.01, 50.0)
    self.bright_via_dog_small_spin.setDecimals(2)
    self.bright_via_dog_small_spin.setSingleStep(0.1)
    self.bright_via_dog_small_spin.setValue(0.8)
    self.bright_via_dog_large_spin = QDoubleSpinBox()
    self.bright_via_dog_large_spin.setRange(0.02, 80.0)
    self.bright_via_dog_large_spin.setDecimals(2)
    self.bright_via_dog_large_spin.setSingleStep(0.1)
    self.bright_via_dog_large_spin.setValue(2.0)
    self.bright_via_threshold_percentile_spin = QDoubleSpinBox()
    self.bright_via_threshold_percentile_spin.setRange(90.0, 99.9)
    self.bright_via_threshold_percentile_spin.setDecimals(1)
    self.bright_via_threshold_percentile_spin.setSingleStep(0.1)
    self.bright_via_threshold_percentile_spin.setValue(99.0)
    self.bright_via_mask_combine_combo = QComboBox()
    self.bright_via_mask_combine_combo.addItem("ИЛИ", "OR")
    self.bright_via_mask_combine_combo.addItem("И", "AND")
    self.bright_via_min_area_factor_spin = QDoubleSpinBox()
    self.bright_via_min_area_factor_spin.setRange(0.01, 10.0)
    self.bright_via_min_area_factor_spin.setDecimals(2)
    self.bright_via_min_area_factor_spin.setSingleStep(0.05)
    self.bright_via_min_area_factor_spin.setValue(0.45)
    self.bright_via_max_area_factor_spin = QDoubleSpinBox()
    self.bright_via_max_area_factor_spin.setRange(0.01, 20.0)
    self.bright_via_max_area_factor_spin.setDecimals(2)
    self.bright_via_max_area_factor_spin.setSingleStep(0.05)
    self.bright_via_max_area_factor_spin.setValue(1.8)
    self.bright_via_min_circularity_spin = QDoubleSpinBox()
    self.bright_via_min_circularity_spin.setRange(0.0, 1.5)
    self.bright_via_min_circularity_spin.setDecimals(2)
    self.bright_via_min_circularity_spin.setSingleStep(0.05)
    self.bright_via_min_circularity_spin.setValue(0.30)
    self.bright_via_min_aspect_spin = QDoubleSpinBox()
    self.bright_via_min_aspect_spin.setRange(0.01, 20.0)
    self.bright_via_min_aspect_spin.setDecimals(2)
    self.bright_via_min_aspect_spin.setSingleStep(0.05)
    self.bright_via_min_aspect_spin.setValue(0.45)
    self.bright_via_max_aspect_spin = QDoubleSpinBox()
    self.bright_via_max_aspect_spin.setRange(0.01, 20.0)
    self.bright_via_max_aspect_spin.setDecimals(2)
    self.bright_via_max_aspect_spin.setSingleStep(0.05)
    self.bright_via_max_aspect_spin.setValue(2.2)
    self.bright_via_bright_center_score_spin = QDoubleSpinBox()
    self.bright_via_bright_center_score_spin.setRange(0.0, 255.0)
    self.bright_via_bright_center_score_spin.setDecimals(1)
    self.bright_via_bright_center_score_spin.setSingleStep(1.0)
    self.bright_via_bright_center_score_spin.setValue(140.0)
    self.bright_via_metal_constraint_combo = QComboBox()
    self.bright_via_metal_constraint_combo.addItem("Отключено", "disabled")
    self.bright_via_metal_constraint_combo.addItem("Мягкая оценка", "soft")
    self.bright_via_metal_constraint_combo.addItem("Жёсткий фильтр", "strict")
    self.bright_via_metal_constraint_combo.setCurrentIndex(1)
    self.bright_via_metal_fraction_spin = QDoubleSpinBox()
    self.bright_via_metal_fraction_spin.setRange(0.0, 1.0)
    self.bright_via_metal_fraction_spin.setDecimals(2)
    self.bright_via_metal_fraction_spin.setSingleStep(0.05)
    self.bright_via_metal_fraction_spin.setValue(0.3)
    self.bright_via_max_radial_asymmetry_spin = QDoubleSpinBox()
    self.bright_via_max_radial_asymmetry_spin.setRange(0.0, 255.0)
    self.bright_via_max_radial_asymmetry_spin.setDecimals(1)
    self.bright_via_max_radial_asymmetry_spin.setSingleStep(1.0)
    self.bright_via_max_radial_asymmetry_spin.setValue(32.0)
    self.bright_via_max_edge_likeness_spin = QDoubleSpinBox()
    self.bright_via_max_edge_likeness_spin.setRange(0.0, 255.0)
    self.bright_via_max_edge_likeness_spin.setDecimals(1)
    self.bright_via_max_edge_likeness_spin.setSingleStep(1.0)
    self.bright_via_max_edge_likeness_spin.setValue(35.0)
    self.bright_via_max_line_likeness_spin = QDoubleSpinBox()
    self.bright_via_max_line_likeness_spin.setRange(0.0, 100.0)
    self.bright_via_max_line_likeness_spin.setDecimals(1)
    self.bright_via_max_line_likeness_spin.setSingleStep(1.0)
    self.bright_via_max_line_likeness_spin.setValue(65.0)
    self.bright_via_nms_distance_spin = QSpinBox()
    self.bright_via_nms_distance_spin.setRange(0, 100_000)
    self.bright_via_nms_distance_spin.setValue(5)
    self.bright_via_min_final_score_spin = QDoubleSpinBox()
    self.bright_via_min_final_score_spin.setRange(0.0, 100.0)
    self.bright_via_min_final_score_spin.setDecimals(1)
    self.bright_via_min_final_score_spin.setSingleStep(1.0)
    self.bright_via_min_final_score_spin.setValue(38.0)
    self.bright_via_show_rejected_checkbox = QCheckBox("Показывать отклонённые кандидаты")
    self.bright_via_show_rejected_checkbox.setChecked(True)
    self.bright_via_hard_asym_checkbox = QCheckBox("Жёстко фильтровать по асимметрии")
    self.bright_via_hard_asym_checkbox.setChecked(True)
    self.bright_via_hard_edge_checkbox = QCheckBox("Подавление границ падов")
    self.bright_via_hard_line_checkbox = QCheckBox("Подавление дорожек")
    self.bright_via_min_isolation_spin = QDoubleSpinBox()
    self.bright_via_min_isolation_spin.setRange(0.0, 1.0)
    self.bright_via_min_isolation_spin.setDecimals(2)
    self.bright_via_min_isolation_spin.setSingleStep(0.01)
    self.bright_via_min_isolation_spin.setValue(0.38)
    self.heuristic_background_sigma_spin = QDoubleSpinBox()
    self.heuristic_background_sigma_spin.setRange(1.0, 200.0)
    self.heuristic_background_sigma_spin.setDecimals(1)
    self.heuristic_background_sigma_spin.setValue(25.0)
    self.heuristic_analysis_window_scale_spin = QDoubleSpinBox()
    self.heuristic_analysis_window_scale_spin.setRange(1.0, 12.0)
    self.heuristic_analysis_window_scale_spin.setDecimals(2)
    self.heuristic_analysis_window_scale_spin.setValue(3.0)
    self.heuristic_min_center_brightness_spin = QDoubleSpinBox()
    self.heuristic_min_center_brightness_spin.setRange(0.0, 255.0)
    self.heuristic_min_center_brightness_spin.setDecimals(1)
    self.heuristic_min_center_brightness_spin.setValue(0.0)
    self.heuristic_min_center_contrast_spin = QDoubleSpinBox()
    self.heuristic_min_center_contrast_spin.setRange(0.0, 255.0)
    self.heuristic_min_center_contrast_spin.setDecimals(1)
    self.heuristic_min_center_contrast_spin.setValue(50.0)
    self.heuristic_min_peak_prominence_spin = QDoubleSpinBox()
    self.heuristic_min_peak_prominence_spin.setRange(0.0, 255.0)
    self.heuristic_min_peak_prominence_spin.setDecimals(1)
    self.heuristic_min_peak_prominence_spin.setValue(50.0)
    self.heuristic_min_compactness_spin = QDoubleSpinBox()
    self.heuristic_min_compactness_spin.setRange(0.0, 1.0)
    self.heuristic_min_compactness_spin.setDecimals(3)
    self.heuristic_min_compactness_spin.setSingleStep(0.1)
    self.heuristic_min_compactness_spin.setValue(0.9)
    self.heuristic_min_circularity_spin = QDoubleSpinBox()
    self.heuristic_min_circularity_spin.setRange(0.0, 1.0)
    self.heuristic_min_circularity_spin.setDecimals(3)
    self.heuristic_min_circularity_spin.setSingleStep(0.01)
    self.heuristic_min_circularity_spin.setValue(0.40)
    self.heuristic_max_elongation_spin = QDoubleSpinBox()
    self.heuristic_max_elongation_spin.setRange(1.0, 20.0)
    self.heuristic_max_elongation_spin.setDecimals(2)
    self.heuristic_max_elongation_spin.setValue(2.5)
    self.heuristic_line_penalty_spin = QDoubleSpinBox()
    self.heuristic_line_penalty_spin.setRange(0.0, 5.0)
    self.heuristic_line_penalty_spin.setDecimals(2)
    self.heuristic_line_penalty_spin.setValue(3.0)
    self.heuristic_border_penalty_spin = QDoubleSpinBox()
    self.heuristic_border_penalty_spin.setRange(0.0, 5.0)
    self.heuristic_border_penalty_spin.setDecimals(2)
    self.heuristic_border_penalty_spin.setValue(1.0)
    self.heuristic_local_binarize_percentile_spin = QDoubleSpinBox()
    self.heuristic_local_binarize_percentile_spin.setRange(50.0, 99.5)
    self.heuristic_local_binarize_percentile_spin.setDecimals(1)
    self.heuristic_local_binarize_percentile_spin.setValue(88.0)
    self.heuristic_min_abs_peak_spin = QDoubleSpinBox()
    self.heuristic_min_abs_peak_spin.setRange(0.0, 255.0)
    self.heuristic_min_abs_peak_spin.setDecimals(1)
    self.heuristic_min_abs_peak_spin.setValue(0.0)
    self.heuristic_use_bilateral_checkbox = QCheckBox("Bilateral вместо медианы")

    def _expert_spin(name: str, minimum: float, maximum: float, value: float, decimals: int = 2):
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        spin.setSingleStep(0.01 if decimals >= 2 else 0.1)
        spin.setValue(value)
        setattr(self, name, spin)
        return spin

    _expert_spin("heuristic_size_tolerance_range_spin", 0.05, 0.95, 0.36)
    _expert_spin("heuristic_size_tolerance_fixed_spin", 0.05, 0.95, 0.26)
    _expert_spin("heuristic_max_center_drift_ratio_spin", 0.10, 1.50, 0.72)
    _expert_spin("heuristic_max_line_coherence_spin", 0.0, 1.0, 0.82)
    _expert_spin("heuristic_min_edge_sharpness_spin", 0.0, 10.0, 0.20)
    _expert_spin("heuristic_contrast_score_min_spin", 0.0, 255.0, 3.0, 1)
    _expert_spin("heuristic_contrast_score_max_spin", 0.0, 255.0, 20.0, 1)
    _expert_spin("heuristic_prominence_score_min_spin", 0.0, 255.0, 2.0, 1)
    _expert_spin("heuristic_prominence_score_max_spin", 0.0, 255.0, 25.0, 1)
    _expert_spin("heuristic_edge_snr_score_min_spin", 0.0, 20.0, 0.70)
    _expert_spin("heuristic_edge_snr_score_max_spin", 0.0, 20.0, 2.80)
    _expert_spin("heuristic_edge_quality_floor_spin", 0.0, 1.0, 0.55)
    _expert_spin("heuristic_border_balance_scale_spin", 0.0, 10.0, 2.0)
    _expert_spin("heuristic_seed_percentile_spin", 0.0, 100.0, 90.0, 1)
    for _weight_name, _weight_default in (
        ("contrast", 25.0),
        ("prominence", 20.0),
        ("size", 20.0),
        ("compact", 15.0),
        ("round", 10.0),
        ("balance", 10.0),
        ("line", 20.0),
        ("border", 20.0),
    ):
        _expert_spin(f"heuristic_w_{_weight_name}_spin", 0.0, 100.0, _weight_default, 1)
    self._heuristic_expert_parameter_widgets = tuple(
        widget
        for name, widget in vars(self).items()
        if name.startswith("heuristic_") and (name.endswith("_spin") or name.endswith("_checkbox"))
    )
    self.heuristic_expert_spinbox_width = 136
    for _expert_widget in self._heuristic_expert_parameter_widgets:
        if isinstance(_expert_widget, (QSpinBox, QDoubleSpinBox)):
            _expert_widget.setFixedWidth(self.heuristic_expert_spinbox_width)
    for _lower, _upper in (
        (self.heuristic_contrast_score_min_spin, self.heuristic_contrast_score_max_spin),
        (self.heuristic_prominence_score_min_spin, self.heuristic_prominence_score_max_spin),
        (self.heuristic_edge_snr_score_min_spin, self.heuristic_edge_snr_score_max_spin),
    ):
        _lower.setMaximum(_upper.value())
        _upper.setMinimum(_lower.value())
        _lower.valueChanged.connect(_upper.setMinimum)
        _upper.valueChanged.connect(_lower.setMaximum)
    self.reset_bright_via_button = QPushButton("Сбросить параметры")
    self.gradient_overlay_mode_combo = QComboBox()
    self.gradient_overlay_mode_combo.addItem("Source image", "source")
    self.gradient_overlay_mode_combo.addItem("Heatmap", "heatmap")
    self.gradient_overlay_mode_combo.addItem("Threshold mask", "threshold")
    self.gradient_overlay_mode_combo.addItem("Raw elevation", "elevation")
    for _label, _map_key in (
        ("Итоговая маска", "mask"),
        ("Маска кандидатов", "candidate_mask"),
        ("Маска контактов", "via_mask"),
        ("Маска металла", "metal_mask"),
        ("Бинарная маска металла", "metal_binary_mask"),
        ("Отфильтрованная маска металла", "metal_filtered_mask"),
        ("Маска Top-hat", "tophat_mask"),
        ("Маска DoG", "dog_mask"),
        ("Отклик светлых пятен", "spot_response"),
        ("Отклик тёмных пятен", "spot_response_dark"),
        ("Отклик колец", "ring_response"),
        ("Скорректированный фон", "background_corrected"),
        ("Локальные светлые максимумы", "local_max_bright"),
        ("Локальные тёмные максимумы", "local_max_dark"),
        ("Бинарная маска", "binary_mask"),
        ("Top-hat", "tophat"),
        ("DoG", "dog"),
        ("Радиальная симметрия", "radial_symmetry"),
        ("Похожесть на край", "edge_likeness"),
        ("Похожесть на линию", "line_likeness"),
        ("Расстояние до края", "distance_to_edge"),
        ("Scharr", "scharr"),
        ("Фазовая согласованность", "phase_congruency"),
        ("Структурные границы", "structured"),
        ("Гребни", "ridge"),
        ("Обработанное изображение", "processed"),
        ("Исходное изображение детектора", "raw_gray"),
        ("Исходное серое изображение", "source_gray"),
        ("Градиент границ проводников", "conductor_gradient_elevation"),
        ("Количество совпадений шаблонов", "template_count"),
        ("Наложение детектора", "overlay"),
        ("Итоговое наложение", "final_overlay"),
    ):
        self.gradient_overlay_mode_combo.addItem(_label, _map_key)
    self.via_roundness_spin = QDoubleSpinBox()
    self.via_roundness_spin.setRange(0.0, 100.0)
    self.via_roundness_spin.setDecimals(1)
    self.via_roundness_spin.setSingleStep(1.0)
    self.via_roundness_spin.setValue(5.0)
    self.min_via_width_spin = QSpinBox()
    self.min_via_width_spin.setRange(0, 100_000)
    self.min_via_width_spin.setValue(0)
    self.max_via_width_spin = QSpinBox()
    self.max_via_width_spin.setRange(0, 100_000)
    self.max_via_width_spin.setValue(0)
    self.min_via_height_spin = QSpinBox()
    self.min_via_height_spin.setRange(0, 100_000)
    self.min_via_height_spin.setValue(0)
    self.max_via_height_spin = QSpinBox()
    self.max_via_height_spin.setRange(0, 100_000)
    self.max_via_height_spin.setValue(0)
    self.via_width_range_widget = self._build_range_row(self.min_via_width_spin, self.max_via_width_spin)
    self.via_height_range_widget = self._build_range_row(self.min_via_height_spin, self.max_via_height_spin)
    self.fixed_vias_widget = QWidget()
    self.fixed_vias_widget.setObjectName("fixedViaArea")
    self.fixed_vias_layout = QVBoxLayout(self.fixed_vias_widget)
    self.fixed_vias_layout.setContentsMargins(10, 10, 10, 10)
    self.fixed_vias_layout.setSpacing(8)
    self.fixed_vias_widget.setStyleSheet(
        "#fixedViaArea { background-color: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.10); border-radius: 8px; }"
        "#fixedViaArea QLabel, #fixedViaArea QSpinBox, #fixedViaArea QPushButton { border: none; background: transparent; }"
    )
    self.fixed_via_rows_widget = QWidget()
    self.fixed_via_rows_layout = QVBoxLayout(self.fixed_via_rows_widget)
    self.fixed_via_rows_layout.setContentsMargins(0, 0, 0, 0)
    self.fixed_via_rows_layout.setSpacing(6)
    self.fixed_vias_layout.addWidget(self.fixed_via_rows_widget)
    self.fixed_via_add_button = QPushButton("+")
    self.fixed_via_add_button.setMinimumHeight(38)
    self.fixed_via_add_button.setStyleSheet(
        "QPushButton { background-color: #2fbf71; color: white; font-size: 22px; font-weight: 700; border-radius: 8px; }"
        "QPushButton:hover { background-color: #28a764; }"
        "QPushButton:pressed { background-color: #229157; }"
    )
    self.fixed_via_add_button.clicked.connect(self._add_fixed_via_row)
    self.fixed_vias_layout.addWidget(self.fixed_via_add_button)

    self.topology_group = QGroupBox("Hierarchy and holes")
    self.topology_form = QFormLayout(self.topology_group)
    self._configure_compact_form(self.topology_form)
    self.min_hierarchy_depth_spin = QSpinBox()
    self.min_hierarchy_depth_spin.setRange(0, 100)
    self.min_hierarchy_depth_spin.setValue(0)
    self.min_inner_hole_area_spin = QDoubleSpinBox()
    self.min_inner_hole_area_spin.setRange(0.0, 1_000_000_000.0)
    self.min_inner_hole_area_spin.setDecimals(1)
    self.min_inner_hole_area_spin.setSingleStep(10.0)
    self.min_inner_hole_area_spin.setValue(100.0)
    self.max_hierarchy_depth_spin = QSpinBox()
    self.max_hierarchy_depth_spin.setRange(0, 100)
    self.max_hierarchy_depth_spin.setValue(0)
    self.max_hole_area_ratio_spin = QDoubleSpinBox()
    self.max_hole_area_ratio_spin.setRange(0.0, 10.0)
    self.max_hole_area_ratio_spin.setDecimals(3)
    self.max_hole_area_ratio_spin.setSingleStep(0.05)
    self.max_hole_area_ratio_spin.setValue(0.0)
    self.advanced_extraction_checkbox.stateChanged.connect(self._on_advanced_extraction_toggled)
    self.retrieval_mode_combo.currentIndexChanged.connect(self._on_extraction_settings_changed)
    self.approximation_mode_combo.currentIndexChanged.connect(self._on_extraction_settings_changed)
    self.epsilon_spin.valueChanged.connect(self._on_epsilon_spin_value_changed)
    self.epsilon_slider.valueChanged.connect(self._on_epsilon_slider_value_changed)
    return _expert_widget, _label
