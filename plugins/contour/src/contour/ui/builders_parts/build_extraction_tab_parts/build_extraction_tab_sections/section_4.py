"""build_extraction_tab section 4."""
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

def _build_extraction_tab_section_4(self, _metal_basic_form):
    self.metal_min_contrast_value_label.setMinimumWidth(28)
    self.metal_min_contrast_value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    self.metal_min_contrast_widget = QWidget()
    _metal_contrast_layout = QHBoxLayout(self.metal_min_contrast_widget)
    _metal_contrast_layout.setContentsMargins(0, 0, 0, 0)
    _metal_contrast_layout.addWidget(self.metal_min_contrast_slider, 1)
    _metal_contrast_layout.addWidget(self.metal_min_contrast_value_label)

    self.metal_gap_bridge_spin = QSpinBox()
    self.metal_gap_bridge_spin.setRange(0, 25)
    self.metal_gap_bridge_spin.setValue(1)
    self.metal_speckle_removal_spin = QSpinBox()
    self.metal_speckle_removal_spin.setRange(0, 25)
    self.metal_speckle_removal_spin.setValue(1)

    self.metal_min_width_spin = QDoubleSpinBox()
    self.metal_min_width_spin.setRange(0.5, 200.0)
    self.metal_min_width_spin.setDecimals(1)
    self.metal_min_width_spin.setValue(8.0)
    self.metal_max_width_spin = QDoubleSpinBox()
    self.metal_max_width_spin.setRange(0.0, 500.0)
    self.metal_max_width_spin.setDecimals(1)
    self.metal_max_width_spin.setSpecialValueText("—")
    self.metal_max_width_spin.setValue(0.0)
    self.metal_width_row = QWidget()
    _metal_width_layout = QHBoxLayout(self.metal_width_row)
    _metal_width_layout.setContentsMargins(0, 0, 0, 0)
    self.metal_min_width_caption = QLabel("Мин.")
    self.metal_max_width_caption = QLabel("Макс.")
    _metal_width_layout.addWidget(self.metal_min_width_caption)
    _metal_width_layout.addWidget(self.metal_min_width_spin, 1)
    _metal_width_layout.addWidget(self.metal_max_width_caption)
    _metal_width_layout.addWidget(self.metal_max_width_spin, 1)

    self.metal_epsilon_spin = QDoubleSpinBox()
    self.metal_epsilon_spin.setRange(0.0, 50.0)
    self.metal_epsilon_spin.setDecimals(2)
    self.metal_epsilon_spin.setValue(2.0)

    _metal_basic_form.addRow("Мин. контраст проводника", self.metal_min_contrast_widget)
    _metal_basic_form.addRow("Сшивка разрывов, px", self.metal_gap_bridge_spin)
    _metal_basic_form.addRow("Удаление шума (opening), px", self.metal_speckle_removal_spin)
    _metal_basic_form.addRow("Ширина проводника, px", self.metal_width_row)
    self.metal_conductor_size_offset_slider = QSlider(Qt.Orientation.Horizontal)
    self.metal_conductor_size_offset_slider.setRange(-20, 20)
    self.metal_conductor_size_offset_slider.setPageStep(1)
    self.metal_conductor_size_offset_slider.setTickInterval(5)
    self.metal_conductor_size_offset_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
    self.metal_conductor_size_offset_slider.setValue(0)
    self.metal_conductor_size_offset_value_label = QLabel("0")
    self.metal_conductor_size_offset_value_label.setMinimumWidth(28)
    self.metal_conductor_size_offset_value_label.setAlignment(
        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
    )
    self.metal_conductor_size_offset_widget = QWidget()
    _conductor_size_layout = QHBoxLayout(self.metal_conductor_size_offset_widget)
    _conductor_size_layout.setContentsMargins(0, 0, 0, 0)
    _conductor_size_layout.addWidget(self.metal_conductor_size_offset_slider, 1)
    _conductor_size_layout.addWidget(self.metal_conductor_size_offset_value_label)
    _metal_basic_form.addRow("Размер проводника, px", self.metal_conductor_size_offset_widget)
    _metal_basic_form.addRow("Epsilon (approxPolyDP)", self.metal_epsilon_spin)
    self.metal_segmentation_strategy_combo = QComboBox()
    self.metal_segmentation_strategy_combo.addItem("Авто (контроль связности)", "auto")
    self.metal_segmentation_strategy_combo.addItem("Порог Otsu", "legacy_otsu")
    self.metal_segmentation_strategy_combo.addItem("Адаптивный порог", "local_adaptive")
    self.metal_segmentation_strategy_combo.addItem("Watershed", "gradient_watershed")
    self.metal_segmentation_strategy_combo.addItem("Random Walker", "random_walker")
    self.metal_segmentation_strategy_combo.addItem("Graph Cut", "graph_cut")
    self.metal_segmentation_strategy_combo.addItem("Reconstruction", "reconstruction")
    self.metal_segmentation_strategy_combo.addItem("Замкнутые границы", "closed_boundary")
    self.metal_segmentation_strategy_combo.addItem("Структурный водораздел", "structural_watershed")
    for _strategy_spec in visible_strategy_specs():
        if self.metal_segmentation_strategy_combo.findData(_strategy_spec.strategy_id) < 0:
            self.metal_segmentation_strategy_combo.addItem(
                _strategy_spec.display_name,
                _strategy_spec.strategy_id,
            )
    self.metal_segmentation_strategy_combo.setCurrentIndex(0)
    _metal_basic_form.addRow("Алгоритм распознавания", self.metal_segmentation_strategy_combo)
    self.metal_segmentation_strategy_label_widget = _metal_basic_form.labelForField(
        self.metal_segmentation_strategy_combo
    )
    self.metal_strategy_description_label = QLabel()
    self.metal_strategy_description_label.setWordWrap(True)
    self.metal_strategy_description_label.setStyleSheet("color: #7f8c8d;")
    _metal_basic_form.addRow(self.metal_strategy_description_label)
    self.metal_auto_contrast_step_spin = QDoubleSpinBox()
    self.metal_auto_contrast_step_spin.setRange(0.0, 255.0)
    self.metal_auto_contrast_step_spin.setDecimals(1)
    self.metal_auto_contrast_step_spin.setSingleStep(1.0)
    self.metal_auto_contrast_step_spin.setSpecialValueText("Off")
    self.metal_auto_contrast_step_spin.setValue(10.0)
    _metal_basic_form.addRow("Auto contrast step", self.metal_auto_contrast_step_spin)
    self.metal_auto_contrast_step_label_widget = _metal_basic_form.labelForField(self.metal_auto_contrast_step_spin)
    self.metal_auto_source_contrast_step_spin = QDoubleSpinBox()
    self.metal_auto_source_contrast_step_spin.setRange(0.0, 255.0)
    self.metal_auto_source_contrast_step_spin.setDecimals(1)
    self.metal_auto_source_contrast_step_spin.setSingleStep(1.0)
    self.metal_auto_source_contrast_step_spin.setSpecialValueText("Off")
    self.metal_auto_source_contrast_step_spin.setValue(4.0)
    _metal_basic_form.addRow(
        "Auto object-filter step",
        self.metal_auto_source_contrast_step_spin,
    )
    self.metal_auto_source_contrast_step_label_widget = _metal_basic_form.labelForField(
        self.metal_auto_source_contrast_step_spin
    )
    self.metal_auto_directional_gap_bridge_spin = QSpinBox()
    self.metal_auto_directional_gap_bridge_spin.setRange(0, 64)
    self.metal_auto_directional_gap_bridge_spin.setSpecialValueText("Off")
    self.metal_auto_directional_gap_bridge_spin.setValue(3)
    _metal_basic_form.addRow(
        "Auto directional gap",
        self.metal_auto_directional_gap_bridge_spin,
    )
    self.metal_auto_directional_gap_bridge_label_widget = _metal_basic_form.labelForField(
        self.metal_auto_directional_gap_bridge_spin
    )
    self.metal_auto_directional_gap_min_source_spin = QDoubleSpinBox()
    self.metal_auto_directional_gap_min_source_spin.setRange(0.0, 255.0)
    self.metal_auto_directional_gap_min_source_spin.setDecimals(1)
    self.metal_auto_directional_gap_min_source_spin.setSingleStep(1.0)
    self.metal_auto_directional_gap_min_source_spin.setValue(45.0)
    _metal_basic_form.addRow(
        "Directional gap source minimum",
        self.metal_auto_directional_gap_min_source_spin,
    )
    self.metal_auto_directional_gap_min_source_label_widget = _metal_basic_form.labelForField(
        self.metal_auto_directional_gap_min_source_spin
    )

    self.metal_display_group = QGroupBox("Отображение")
    _metal_disp_form = QFormLayout(self.metal_display_group)
    self.metal_show_conductors_checkbox = QCheckBox("Показывать проводники")
    self.metal_show_conductors_checkbox.setChecked(True)
    self.metal_show_rejected_checkbox = QCheckBox("Показывать отклонённые")
    self.metal_show_rejected_checkbox.setChecked(True)
    self.metal_show_suspicious_checkbox = QCheckBox("Показывать подозрительные")
    self.metal_show_suspicious_checkbox.setChecked(True)
    self.metal_show_border_checkbox = QCheckBox("Подсветка у границы кадра")
    self.metal_show_border_checkbox.setChecked(False)
    self.metal_show_mask_checkbox = QCheckBox("Наложение маски (градиент)")
    self.metal_show_mask_checkbox.setChecked(False)
    self.metal_debug_visual_combo = QComboBox()
    for _l, _d in (
        ("Итоговое наложение", "overlay"),
        ("Исходное (серое)", "metal_source_gray"),
        ("Градиент по X", "metal_gradient_x"),
        ("Градиент по Y", "metal_gradient_y"),
        ("Градиентное поле", "metal_gradient_field"),
        ("После подготовки", "metal_preprocessed"),
        ("Сегментация (сырая)", "metal_raw_segmentation"),
        ("После топологии", "metal_after_topology"),
        ("Бинарная маска", "metal_binary_mask"),
        ("Порог Otsu", "metal_threshold_mask"),
        ("Контуры (сырые)", "metal_contours_raw"),
        ("После фильтрации", "metal_filtered_mask"),
        ("Проверка ширины", "metal_width_check"),
        ("Структура: денойз", "metal_structural_denoised"),
        ("Структура: Gx", "metal_structural_gx"),
        ("Структура: Gy", "metal_structural_gy"),
        ("Структура: |G|", "metal_structural_gradient_magnitude"),
        ("Структура: ориентация", "metal_structural_orientation"),
        ("Структура: когерентность", "metal_structural_coherence"),
        ("Структура: ridge response", "metal_structural_ridge_response"),
        ("Структура: ridge markers", "metal_structural_ridge_markers"),
        ("Структура: ridge fragments", "metal_structural_ridge_fragments"),
        ("Структура: ridge links", "metal_structural_ridge_links_accepted"),
        ("Структура: rejected links", "metal_structural_ridge_links_rejected"),
        ("Структура: boundary veto", "metal_structural_ridge_links_boundary_veto"),
        ("Структура: logical ridge", "metal_structural_logical_ridge"),
        ("Структура: wide interior", "metal_structural_wide_interior_markers"),
        ("Структура: wide fragments", "metal_structural_wide_fragments"),
        ("Структура: logical wide", "metal_structural_logical_wide"),
        ("Структура: logical markers", "metal_structural_logical_markers"),
        ("Структура: conductor bands", "metal_structural_conductor_bands"),
        ("Структура: transverse samples", "metal_structural_transverse_samples"),
        ("Структура: accepted band groups", "metal_structural_band_groups_accepted"),
        ("Структура: rejected band groups", "metal_structural_band_groups_rejected"),
        ("Структура: foreground markers", "metal_structural_foreground_markers"),
        ("Структура: background markers", "metal_structural_background_markers"),
        ("Структура: boundary cost", "metal_structural_boundary_cost"),
        ("Структура: watershed labels", "metal_structural_watershed_labels"),
        ("Структура: instance labels", "metal_structural_instance_labels"),
        ("Структура: label boundary", "metal_structural_label_boundary"),
        ("Структура: final mask", "metal_structural_final_mask"),
        ("OWT-UCM: oriented boundaries", "metal_owt_oriented_boundaries"),
        ("OWT-UCM: initial watershed", "metal_owt_initial_watershed"),
        ("OWT-UCM: UCM", "metal_owt_ucm"),
        ("OWT-UCM: selected hierarchy", "metal_owt_selected_hierarchy"),
        ("Multi-Separator: separator cost", "metal_msp_separator_cost"),
        ("Multi-Separator: selected separators", "metal_msp_selected_separators"),
        ("Multi-Separator: regions", "metal_msp_regions"),
        ("GASP: attractive affinity", "metal_gasp_attractive_affinity"),
        ("GASP: repulsive affinity", "metal_gasp_repulsive_affinity"),
        ("GASP: final labels", "metal_gasp_final_labels"),
        ("MWS: attractive affinity", "metal_mutex_watershed_attractive_affinity"),
        ("MWS: repulsive affinity", "metal_mutex_watershed_repulsive_affinity"),
        ("MWS: long-range mutex", "metal_mutex_watershed_long_range_mutex"),
        ("MWS: final labels", "metal_mutex_watershed_final_labels"),
        ("Multicut: attractive affinity", "metal_multicut_attractive_affinity"),
        ("Multicut: repulsive affinity", "metal_multicut_repulsive_affinity"),
        ("Multicut: final labels", "metal_multicut_final_labels"),
        ("Lifted Multicut: lifted relations", "metal_lifted_multicut_lifted_relations"),
        ("Lifted Multicut: final labels", "metal_lifted_multicut_final_labels"),
        ("Material: confidence", "metal_material_confidence"),
        ("Material: core evidence", "metal_material_core_evidence"),
        ("Material: substrate evidence", "metal_material_substrate_evidence"),
    ):
        self.metal_debug_visual_combo.addItem(_l, _d)
    self.metal_gradient_3d_button = QPushButton("3D поле")
    _metal_debug_row = QWidget()
    _metal_debug_layout = QHBoxLayout(_metal_debug_row)
    _metal_debug_layout.setContentsMargins(0, 0, 0, 0)
    _metal_debug_layout.setSpacing(8)
    _metal_debug_layout.addWidget(self.metal_debug_visual_combo, 1)
    _metal_debug_layout.addWidget(self.metal_gradient_3d_button, 0)
    self.metal_overlay_opacity_spin = QDoubleSpinBox()
    self.metal_overlay_opacity_spin.setRange(0.05, 1.0)
    self.metal_overlay_opacity_spin.setSingleStep(0.05)
    self.metal_overlay_opacity_spin.setValue(0.45)
    _metal_disp_form.addRow(self.metal_show_conductors_checkbox)
    _metal_disp_form.addRow(self.metal_show_rejected_checkbox)
    _metal_disp_form.addRow(self.metal_show_suspicious_checkbox)
    _metal_disp_form.addRow(self.metal_show_mask_checkbox)
    _metal_disp_form.addRow("Режим отладочного изображения", _metal_debug_row)
    self.metal_debug_visual_label_widget = _metal_disp_form.labelForField(_metal_debug_row)
    _metal_disp_form.addRow("Прозрачность наложения", self.metal_overlay_opacity_spin)

    self.metal_advanced_group = QGroupBox("Дополнительно")
    self.metal_advanced_group.setCheckable(True)
    self.metal_advanced_group.setChecked(False)
    self.metal_min_area_spin = QDoubleSpinBox()
    self.metal_min_area_spin.setRange(0.0, 1_000_000.0)
    self.metal_min_area_spin.setDecimals(1)
    self.metal_min_area_spin.setValue(60.0)
    self.metal_max_area_spin = QDoubleSpinBox()
    self.metal_max_area_spin.setRange(0.0, 1_000_000.0)
    self.metal_max_area_spin.setDecimals(1)
    self.metal_max_area_spin.setSpecialValueText("—")
    self.metal_max_area_spin.setValue(0.0)
    self.metal_min_perimeter_spin = QDoubleSpinBox()
    self.metal_min_perimeter_spin.setRange(0.0, 1_000_000.0)
    self.metal_min_perimeter_spin.setDecimals(1)
    self.metal_min_perimeter_spin.setValue(32.0)
    self.metal_max_perimeter_spin = QDoubleSpinBox()
    self.metal_max_perimeter_spin.setRange(0.0, 1_000_000.0)
    self.metal_max_perimeter_spin.setDecimals(1)
    self.metal_max_perimeter_spin.setSpecialValueText("—")
    self.metal_max_perimeter_spin.setValue(0.0)
    self.metal_min_length_spin = QDoubleSpinBox()
    self.metal_min_length_spin.setRange(1.0, 2000.0)
    self.metal_min_length_spin.setDecimals(1)
    self.metal_min_length_spin.setValue(8.0)
    self.metal_min_points_spin = QSpinBox()
    self.metal_min_points_spin.setRange(3, 256)
    self.metal_min_points_spin.setValue(4)
    self.metal_min_angle_spin = QDoubleSpinBox()
    self.metal_min_angle_spin.setRange(0.0, 180.0)
    self.metal_min_angle_spin.setValue(30.0)
    self.metal_min_object_source_contrast_spin = QDoubleSpinBox()
    self.metal_min_object_source_contrast_spin.setRange(0.0, 255.0)
    self.metal_min_object_source_contrast_spin.setDecimals(1)
    self.metal_min_object_source_contrast_spin.setValue(12.0)
    self.metal_min_object_rim_contrast_spin = QDoubleSpinBox()
    self.metal_min_object_rim_contrast_spin.setRange(0.0, 255.0)
    self.metal_min_object_rim_contrast_spin.setDecimals(1)
    self.metal_min_object_rim_contrast_spin.setValue(36.0)
    self.metal_min_object_rim_area_fraction_spin = QDoubleSpinBox()
    self.metal_min_object_rim_area_fraction_spin.setRange(0.000001, 1.0)
    self.metal_min_object_rim_area_fraction_spin.setDecimals(6)
    self.metal_min_object_rim_area_fraction_spin.setSingleStep(0.0005)
    self.metal_min_object_rim_area_fraction_spin.setValue(0.001)
    self.metal_min_hole_source_contrast_spin = QDoubleSpinBox()
    self.metal_min_hole_source_contrast_spin.setRange(0.0, 255.0)
    self.metal_min_hole_source_contrast_spin.setDecimals(1)
    self.metal_min_hole_source_contrast_spin.setValue(8.0)
    self.metal_min_hole_source_contrast_fraction_spin = QDoubleSpinBox()
    self.metal_min_hole_source_contrast_fraction_spin.setRange(0.0, 1.0)
    self.metal_min_hole_source_contrast_fraction_spin.setDecimals(2)
    self.metal_min_hole_source_contrast_fraction_spin.setSingleStep(0.05)
    self.metal_min_hole_source_contrast_fraction_spin.setValue(0.35)
    self.metal_approximation_checkbox = QCheckBox("Режим аппроксимации контуров")
    self.metal_approximation_checkbox.setChecked(True)
    self.metal_ws_smoothing_spin = QDoubleSpinBox()
    self.metal_ws_smoothing_spin.setRange(0.1, 8.0)
    self.metal_ws_smoothing_spin.setDecimals(1)
    self.metal_ws_smoothing_spin.setSingleStep(0.1)
    self.metal_ws_smoothing_spin.setValue(1.2)
    self.metal_ws_core_margin_spin = QDoubleSpinBox()
    self.metal_ws_core_margin_spin.setRange(0.0, 40.0)
    self.metal_ws_core_margin_spin.setDecimals(1)
    self.metal_ws_core_margin_spin.setSingleStep(1.0)
    self.metal_ws_core_margin_spin.setValue(23.0)
    self.metal_ws_groove_margin_spin = QDoubleSpinBox()
    self.metal_ws_groove_margin_spin.setRange(0.0, 40.0)
    self.metal_ws_groove_margin_spin.setDecimals(1)
    self.metal_ws_groove_margin_spin.setSingleStep(1.0)
    self.metal_ws_groove_margin_spin.setValue(19.0)
    self.metal_ws_rim_probe_spin = QSpinBox()
    self.metal_ws_rim_probe_spin.setRange(1, 32)
    self.metal_ws_rim_probe_spin.setValue(2)
    self.metal_ws_seed_speckle_spin = QSpinBox()
    self.metal_ws_seed_speckle_spin.setRange(0, 8)
    self.metal_ws_seed_speckle_spin.setValue(1)
    self.metal_ws_valley_span_spin = QSpinBox()
    self.metal_ws_valley_span_spin.setRange(0, 16)
    self.metal_ws_valley_span_spin.setValue(5)
    self.metal_ws_valley_depth_spin = QDoubleSpinBox()
    self.metal_ws_valley_depth_spin.setRange(0.0, 160.0)
    self.metal_ws_valley_depth_spin.setDecimals(1)
    self.metal_ws_valley_depth_spin.setSingleStep(5.0)
    self.metal_ws_valley_depth_spin.setValue(65.0)
    self.metal_rw_beta_spin = QDoubleSpinBox()
    self.metal_rw_beta_spin.setRange(1.0, 400.0)
    self.metal_rw_beta_spin.setDecimals(0)
    self.metal_rw_beta_spin.setSingleStep(10.0)
    self.metal_rw_beta_spin.setValue(90.0)
    self.metal_rw_iterations_spin = QSpinBox()
    self.metal_rw_iterations_spin.setRange(8, 400)
    self.metal_rw_iterations_spin.setValue(160)
    self.metal_gc_iterations_spin = QSpinBox()
    self.metal_gc_iterations_spin.setRange(1, 16)
    self.metal_gc_iterations_spin.setValue(5)
    self.metal_recon_erode_spin = QSpinBox()
    self.metal_recon_erode_spin.setRange(0, 16)
