"""build_extraction_tab section 5."""
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

def _build_extraction_tab_section_5(self, _cond_l, _rm_row, contour_layout, layout, tab):
    self.metal_recon_erode_spin.setValue(0)
    self.metal_boundary_relief_spin = QDoubleSpinBox()
    self.metal_boundary_relief_spin.setRange(1.0, 80.0)
    self.metal_boundary_relief_spin.setDecimals(1)
    self.metal_boundary_relief_spin.setSingleStep(1.0)
    self.metal_boundary_relief_spin.setValue(16.0)
    self.metal_boundary_background_spin = QDoubleSpinBox()
    self.metal_boundary_background_spin.setRange(2.0, 60.0)
    self.metal_boundary_background_spin.setDecimals(1)
    self.metal_boundary_background_spin.setSingleStep(1.0)
    self.metal_boundary_background_spin.setValue(12.0)
    self.metal_adaptive_block_spin = QSpinBox()
    self.metal_adaptive_block_spin.setRange(0, 255)
    self.metal_adaptive_block_spin.setSingleStep(2)
    self.metal_adaptive_block_spin.setSpecialValueText("Авто")
    self.metal_adaptive_block_spin.setValue(0)
    self.metal_adaptive_c_spin = QDoubleSpinBox()
    self.metal_adaptive_c_spin.setRange(-64.0, 64.0)
    self.metal_adaptive_c_spin.setDecimals(1)
    self.metal_adaptive_c_spin.setSingleStep(1.0)
    self.metal_adaptive_c_spin.setValue(0.0)
    self.metal_adaptive_method_combo = QComboBox()
    self.metal_adaptive_method_combo.addItem("Гаусс", "gaussian")
    self.metal_adaptive_method_combo.addItem("Среднее", "mean")
    self.metal_hierarchy_combo = QComboBox()
    self.metal_hierarchy_combo.addItem("Полная иерархия", "full")
    self.metal_hierarchy_combo.addItem("Только внешние контуры", "external")
    self.metal_hierarchy_combo.setCurrentIndex(0)
    self.metal_border_handling_combo = QComboBox()
    self.metal_border_handling_combo.addItem("Игнорировать объекты у края", "ignore")
    self.metal_border_handling_combo.addItem("Принимать как обычно", "accept")
    self.metal_border_handling_combo.addItem("Отдельно помечать", "mark")
    self.metal_border_handling_combo.setCurrentIndex(2)

    self.metal_filter_group = QGroupBox("Фильтрация распознанных")
    _metal_filter_form = QFormLayout(self.metal_filter_group)
    self._configure_compact_form(_metal_filter_form)
    _metal_filter_form.addRow(
        "Мин. контраст объекта к фону",
        self.metal_min_object_source_contrast_spin,
    )
    _metal_filter_form.addRow("Bright rim minimum contrast", self.metal_min_object_rim_contrast_spin)
    self.metal_min_object_rim_contrast_label_widget = _metal_filter_form.labelForField(
        self.metal_min_object_rim_contrast_spin
    )
    _metal_filter_form.addRow(
        "Rim-filter minimum area fraction",
        self.metal_min_object_rim_area_fraction_spin,
    )
    self.metal_min_object_rim_area_fraction_label_widget = _metal_filter_form.labelForField(
        self.metal_min_object_rim_area_fraction_spin
    )
    _metal_filter_form.addRow("Мин. площадь", self.metal_min_area_spin)
    _metal_filter_form.addRow("Макс. площадь (0 = нет)", self.metal_max_area_spin)
    _metal_filter_form.addRow("Мин. периметр", self.metal_min_perimeter_spin)
    _metal_filter_form.addRow("Макс. периметр (0 = нет)", self.metal_max_perimeter_spin)
    _metal_filter_form.addRow("Мин. длина проводника, px", self.metal_min_length_spin)
    _metal_filter_form.addRow("Мин. число точек", self.metal_min_points_spin)
    _metal_filter_form.addRow("Мин. угол полигона, °", self.metal_min_angle_spin)
    _metal_filter_form.addRow("Обработка объектов на границе", self.metal_border_handling_combo)

    self.metal_recognition_params_group = QGroupBox("Параметры распознавания")
    _metal_recog_form = QFormLayout(self.metal_recognition_params_group)
    self._configure_compact_form(_metal_recog_form)
    _metal_recog_form.addRow(self.metal_approximation_checkbox)
    _metal_recog_form.addRow("Режим иерархии", self.metal_hierarchy_combo)
    _metal_recog_form.addRow("Мин. площадь отверстия для заливки, px²", self.min_inner_hole_area_spin)
    self.min_inner_hole_area_label_widget = _metal_recog_form.labelForField(self.min_inner_hole_area_spin)
    _metal_recog_form.addRow("Мин. контраст отверстия", self.metal_min_hole_source_contrast_spin)
    _metal_recog_form.addRow(
        "Доля контраста классов для отверстия",
        self.metal_min_hole_source_contrast_fraction_spin,
    )

    self.metal_watershed_group = QGroupBox("Watershed")
    _metal_ws_form = QFormLayout(self.metal_watershed_group)
    self._configure_compact_form(_metal_ws_form)
    _metal_ws_form.addRow("Сглаживание, σ", self.metal_ws_smoothing_spin)
    self.metal_ws_smoothing_label_widget = _metal_ws_form.labelForField(self.metal_ws_smoothing_spin)
    _metal_ws_form.addRow("Отступ ядер металла", self.metal_ws_core_margin_spin)
    self.metal_ws_core_margin_label_widget = _metal_ws_form.labelForField(self.metal_ws_core_margin_spin)
    _metal_ws_form.addRow("Отступ затравок зазора", self.metal_ws_groove_margin_spin)
    self.metal_ws_groove_margin_label_widget = _metal_ws_form.labelForField(self.metal_ws_groove_margin_spin)
    _metal_ws_form.addRow("Радиус кольца кромки, px", self.metal_ws_rim_probe_spin)
    self.metal_ws_rim_probe_label_widget = _metal_ws_form.labelForField(self.metal_ws_rim_probe_spin)
    _metal_ws_form.addRow("Очистка затравок, px", self.metal_ws_seed_speckle_spin)
    self.metal_ws_seed_speckle_label_widget = _metal_ws_form.labelForField(self.metal_ws_seed_speckle_spin)
    _metal_ws_form.addRow("Ширина узкого зазора, px", self.metal_ws_valley_span_spin)
    self.metal_ws_valley_span_label_widget = _metal_ws_form.labelForField(self.metal_ws_valley_span_spin)
    _metal_ws_form.addRow("Глубина узкого зазора", self.metal_ws_valley_depth_spin)
    self.metal_ws_valley_depth_label_widget = _metal_ws_form.labelForField(self.metal_ws_valley_depth_spin)

    self.metal_random_walker_group = QGroupBox("Random Walker")
    _metal_rw_form = QFormLayout(self.metal_random_walker_group)
    self._configure_compact_form(_metal_rw_form)
    _metal_rw_form.addRow("Жёсткость границ, β", self.metal_rw_beta_spin)
    self.metal_rw_beta_label_widget = _metal_rw_form.labelForField(self.metal_rw_beta_spin)
    _metal_rw_form.addRow("Итерации", self.metal_rw_iterations_spin)
    self.metal_rw_iterations_label_widget = _metal_rw_form.labelForField(self.metal_rw_iterations_spin)

    self.metal_graph_cut_group = QGroupBox("Graph Cut")
    _metal_gc_form = QFormLayout(self.metal_graph_cut_group)
    self._configure_compact_form(_metal_gc_form)
    _metal_gc_form.addRow("Итерации GrabCut", self.metal_gc_iterations_spin)
    self.metal_gc_iterations_label_widget = _metal_gc_form.labelForField(self.metal_gc_iterations_spin)

    self.metal_reconstruction_group = QGroupBox("Reconstruction")
    _metal_recon_form = QFormLayout(self.metal_reconstruction_group)
    self._configure_compact_form(_metal_recon_form)
    _metal_recon_form.addRow("Эрозия ядер, px", self.metal_recon_erode_spin)
    self.metal_recon_erode_label_widget = _metal_recon_form.labelForField(self.metal_recon_erode_spin)

    self.metal_closed_boundary_group = QGroupBox("Замкнутые границы")
    _metal_cb_form = QFormLayout(self.metal_closed_boundary_group)
    self._configure_compact_form(_metal_cb_form)
    _metal_cb_form.addRow("Высота рельефа границы", self.metal_boundary_relief_spin)
    self.metal_boundary_relief_label_widget = _metal_cb_form.labelForField(self.metal_boundary_relief_spin)
    _metal_cb_form.addRow("Масштаб фона, σ", self.metal_boundary_background_spin)
    self.metal_boundary_background_label_widget = _metal_cb_form.labelForField(self.metal_boundary_background_spin)

    self.metal_adaptive_group = QGroupBox("Адаптивный порог")
    _metal_ad_form = QFormLayout(self.metal_adaptive_group)
    self._configure_compact_form(_metal_ad_form)
    _metal_ad_form.addRow("Размер окна, px", self.metal_adaptive_block_spin)
    self.metal_adaptive_block_label_widget = _metal_ad_form.labelForField(self.metal_adaptive_block_spin)
    _metal_ad_form.addRow("Смещение порога C", self.metal_adaptive_c_spin)
    self.metal_adaptive_c_label_widget = _metal_ad_form.labelForField(self.metal_adaptive_c_spin)
    _metal_ad_form.addRow("Метод", self.metal_adaptive_method_combo)
    self.metal_adaptive_method_label_widget = _metal_ad_form.labelForField(self.metal_adaptive_method_combo)

    self.metal_strategy_parameters_group = QGroupBox("Strategy parameters")
    _strategy_parameters_layout = QVBoxLayout(self.metal_strategy_parameters_group)
    _strategy_parameters_layout.setContentsMargins(4, 4, 4, 4)
    self.metal_strategy_parameter_stack = QStackedWidget()
    self.metal_strategy_parameter_widgets: dict[str, dict[str, QWidget]] = {}
    self.metal_strategy_parameter_pages: dict[str, int] = {}
    for _strategy_id in sorted(IMPLEMENTED_NEW_STRATEGIES):
        _page = _build_strategy_parameter_page(self, strategy_spec(_strategy_id))
        self.metal_strategy_parameter_pages[_strategy_id] = self.metal_strategy_parameter_stack.addWidget(_page)
    _strategy_parameters_layout.addWidget(self.metal_strategy_parameter_stack)

    _metal_adv_wrap = QWidget()
    _adv_box_l = QVBoxLayout(_metal_adv_wrap)
    _adv_box_l.setContentsMargins(8, 4, 8, 4)
    _adv_box_l.setSpacing(8)
    _adv_box_l.addWidget(self.metal_filter_group)
    _adv_box_l.addWidget(self.metal_recognition_params_group)
    _adv_box_l.addWidget(self.metal_strategy_parameters_group)
    _adv_box_l.addWidget(self.metal_adaptive_group)
    _adv_box_l.addWidget(self.metal_watershed_group)
    _adv_box_l.addWidget(self.metal_random_walker_group)
    _adv_box_l.addWidget(self.metal_graph_cut_group)
    _adv_box_l.addWidget(self.metal_reconstruction_group)
    _adv_box_l.addWidget(self.metal_closed_boundary_group)
    _adv_outer = QVBoxLayout(self.metal_advanced_group)
    _adv_outer.setContentsMargins(0, 0, 0, 0)
    _adv_outer.addWidget(_metal_adv_wrap)
    self.metal_advanced_group.toggled.connect(_metal_adv_wrap.setVisible)
    _metal_adv_wrap.setVisible(self.metal_advanced_group.isChecked())

    _metal_actions = QHBoxLayout()
    self.metal_preview_mask_button = QPushButton("Предпросмотр маски")
    self.metal_preview_mask_button.setVisible(False)
    self.metal_reset_params_button = QPushButton("Сбросить параметры")
    _metal_actions.addWidget(self.metal_reset_params_button)

    _cond_l.addWidget(self.metal_preset_group, 0)
    _cond_l.addWidget(self.metal_basic_group, 0)
    _cond_l.addWidget(self.metal_display_group, 0)
    _cond_l.addWidget(self.metal_advanced_group, 0)
    _cond_l.addLayout(_metal_actions)

    self.metal_preset_combo.currentIndexChanged.connect(self._on_metal_preset_changed)
    self.apply_metal_preset_button.clicked.connect(self._apply_selected_metal_preset)
    self.save_metal_preset_button.clicked.connect(self._save_current_metal_preset)
    self.delete_metal_preset_button.clicked.connect(self._delete_selected_metal_preset)
    self.export_metal_preset_button.clicked.connect(self._export_selected_metal_preset)
    self.import_metal_preset_button.clicked.connect(self._import_metal_preset)
    self.metal_min_contrast_slider.valueChanged.connect(
        lambda value: self.metal_min_contrast_value_label.setText(str(value))
    )
    self.metal_min_contrast_slider.valueChanged.connect(self._on_extraction_settings_changed)
    self.metal_auto_contrast_step_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.metal_auto_source_contrast_step_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.metal_auto_directional_gap_bridge_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.metal_auto_directional_gap_min_source_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.metal_gap_bridge_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.metal_speckle_removal_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.metal_epsilon_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.metal_min_width_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.metal_max_width_spin.valueChanged.connect(self._on_extraction_settings_changed)
    self.metal_conductor_size_offset_slider.valueChanged.connect(self._on_conductor_size_offset_changed)
    for _w in (
        self.metal_show_conductors_checkbox,
        self.metal_show_rejected_checkbox,
        self.metal_show_suspicious_checkbox,
        self.metal_show_border_checkbox,
        self.metal_show_mask_checkbox,
    ):
        _w.stateChanged.connect(self._on_metal_display_changed)
    self.metal_debug_visual_combo.currentIndexChanged.connect(self._on_metal_display_changed)
    self.metal_gradient_3d_button.clicked.connect(self._open_metal_gradient_field_3d)
    self.metal_overlay_opacity_spin.valueChanged.connect(self._on_metal_overlay_opacity_changed)
    for _w in (
        self.metal_min_area_spin,
        self.metal_max_area_spin,
        self.metal_min_perimeter_spin,
        self.metal_max_perimeter_spin,
        self.metal_min_length_spin,
        self.metal_min_points_spin,
        self.metal_min_angle_spin,
        self.metal_min_object_source_contrast_spin,
        self.metal_min_object_rim_contrast_spin,
        self.metal_min_object_rim_area_fraction_spin,
        self.metal_min_hole_source_contrast_spin,
        self.metal_min_hole_source_contrast_fraction_spin,
        self.metal_ws_smoothing_spin,
        self.metal_ws_core_margin_spin,
        self.metal_ws_groove_margin_spin,
        self.metal_ws_rim_probe_spin,
        self.metal_ws_seed_speckle_spin,
        self.metal_ws_valley_span_spin,
        self.metal_ws_valley_depth_spin,
        self.metal_rw_beta_spin,
        self.metal_rw_iterations_spin,
        self.metal_gc_iterations_spin,
        self.metal_recon_erode_spin,
        self.metal_boundary_relief_spin,
        self.metal_boundary_background_spin,
        self.metal_adaptive_block_spin,
        self.metal_adaptive_c_spin,
    ):
        _w.valueChanged.connect(self._on_extraction_settings_changed)
    self.metal_approximation_checkbox.stateChanged.connect(self._on_extraction_settings_changed)
    self.metal_adaptive_method_combo.currentIndexChanged.connect(self._on_extraction_settings_changed)
    self.metal_segmentation_strategy_combo.currentIndexChanged.connect(self._on_extraction_settings_changed)
    self.metal_segmentation_strategy_combo.currentIndexChanged.connect(lambda _index: _sync_metal_strategy_panel(self))
    self.metal_segmentation_strategy_combo.currentIndexChanged.connect(
        lambda: self.metal_auto_contrast_step_spin.setEnabled(
            self.metal_segmentation_strategy_combo.currentData() == "auto"
        )
    )
    self.metal_segmentation_strategy_combo.currentIndexChanged.connect(
        lambda: self.metal_auto_source_contrast_step_spin.setEnabled(
            self.metal_segmentation_strategy_combo.currentData() == "auto"
        )
    )
    self.metal_segmentation_strategy_combo.currentIndexChanged.connect(
        lambda: self.metal_auto_directional_gap_bridge_spin.setEnabled(
            self.metal_segmentation_strategy_combo.currentData() == "auto"
        )
    )
    self.metal_segmentation_strategy_combo.currentIndexChanged.connect(
        lambda: self.metal_auto_directional_gap_min_source_spin.setEnabled(
            self.metal_segmentation_strategy_combo.currentData() == "auto"
        )
    )
    self.metal_hierarchy_combo.currentIndexChanged.connect(self._on_extraction_settings_changed)
    self.metal_border_handling_combo.currentIndexChanged.connect(self._on_extraction_settings_changed)
    self.metal_preview_mask_button.clicked.connect(self._preview_metal_mask)
    self.metal_reset_params_button.clicked.connect(self._reset_metal_parameters)
    self.recognition_stack.addWidget(self.recognition_page_off)
    self.recognition_stack.addWidget(self.recognition_page_conductors)
    self.recognition_stack.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
    self.recognition_stack.setCurrentIndex(1)
    _sync_metal_strategy_panel(self)
    self.recognition_mode_combo.currentIndexChanged.connect(self._update_extraction_profile_controls_state)
    self.recognition_mode_combo.currentIndexChanged.connect(self._on_recognition_mode_changed)
    self.gradient_overlay_mode_combo.currentIndexChanged.connect(self._refresh_gradient_overlay)
    self.via_form.addRow("Roundness", self.via_roundness_spin)
    self.via_roundness_label_widget = self.via_form.labelForField(self.via_roundness_spin)
    self.via_form.addRow("Contact width range", self.via_width_range_widget)
    self.min_via_width_label_widget = self.via_form.labelForField(self.via_width_range_widget)
    self.max_via_width_label_widget = None
    self.via_form.addRow("Contact height range", self.via_height_range_widget)
    self.min_via_height_label_widget = self.via_form.labelForField(self.via_height_range_widget)
    self.max_via_height_label_widget = None
    self.via_form.addRow("Fixed contacts", self.fixed_vias_widget)
    self.fixed_vias_label_widget = self.via_form.labelForField(self.fixed_vias_widget)
    self._update_via_size_controls_state()

    self.topology_form.addRow("Min hierarchy depth", self.min_hierarchy_depth_spin)
    self.min_hierarchy_depth_label_widget = self.topology_form.labelForField(self.min_hierarchy_depth_spin)
    self.topology_form.addRow("Max hierarchy depth (0 = unlimited)", self.max_hierarchy_depth_spin)
    self.max_hierarchy_depth_label_widget = self.topology_form.labelForField(self.max_hierarchy_depth_spin)
    self.topology_form.addRow("Max hole area ratio (0 = unlimited)", self.max_hole_area_ratio_spin)
    self.max_hole_area_ratio_label_widget = self.topology_form.labelForField(self.max_hole_area_ratio_spin)

    for group in [
        self.basic_filters_group,
        self.geometry_filters_group,
        self.via_group,
        self.topology_group,
    ]:
        contour_layout.addWidget(group)
    contour_layout.addWidget(self.conductor_group)
    self.conductor_group.setVisible(False)
    layout.addLayout(_rm_row)
    layout.addWidget(self.recognition_stack, 0)
    layout.addWidget(self.bright_via_group, 0)
    self.contour_group.setParent(tab)
    self.contour_group.hide()
    layout.addStretch(1)

    self.save_group = QGroupBox("Save options")
    save_layout = QVBoxLayout(self.save_group)
    self.save_cif_checkbox = QCheckBox("CIF")
    self.save_cif_checkbox.setChecked(True)
    self.save_cv_checkbox = QCheckBox("CV")
    self.save_preview_checkbox = QCheckBox("Overlay preview image")
    self.save_preview_checkbox.setChecked(False)
    for checkbox in [
        self.save_cif_checkbox,
        self.save_cv_checkbox,
        self.save_preview_checkbox,
    ]:
        save_layout.addWidget(checkbox)
