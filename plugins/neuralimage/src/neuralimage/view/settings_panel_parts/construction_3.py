"""settings_panel: construction 3."""
from __future__ import annotations
from PyQt6.QtCore import QStringListModel, QTimer, Qt
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QWidget,
    QCheckBox,
    QGridLayout,
    QButtonGroup,
    QPushButton,
    QGroupBox,
    QTabWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLineEdit,
    QCompleter,
    QToolButton,
    QRadioButton,
)
from neuralimage.lib.data_interfaces import (
    RANDOM_ARTIFACT_TYPES,
)
from neuralimage.view.settings_panel_widgets import (
    SettingsCard,
    SlidingPanel,  # noqa: F401 - compatibility re-export from neuralimage.view
    )
from neuralimage.view.compact_settings_row import add_compact_row
from neuralimage.view.training_transform_editors import (
    AugmentationBlock,
    AugmentationRow,
    SemNormalizationEditor,
    TrainingAugmentationEditor,
)

from neuralimage.view.settings_panel_parts.module_state import (
    CONTENT_LAYOUT_MARGINS,
    CONTENT_LAYOUT_SPACING,
    IC_DEFECT_WEIGHT_FIELDS,
    PCB_DEFECT_WEIGHT_FIELDS,
)

from neuralimage.view.settings_panel_parts.settings_panel_qsettings import _settings_panel_qsettings


class Construction3Mixin:
    def _init_layout(self) -> None:
        # Backward-compatible alias (legacy code expects training patch widget).
        self.sample_size_widget = self.train_patch_size_widget

        self.train_patch_size_groupbox = QGroupBox('')
        train_patch_layout = QVBoxLayout(self.train_patch_size_groupbox)
        train_patch_layout.setContentsMargins(*CONTENT_LAYOUT_MARGINS)
        train_patch_layout.setSpacing(CONTENT_LAYOUT_SPACING)
        train_patch_layout.addWidget(self.train_patch_size_widget, alignment=Qt.AlignmentFlag.AlignCenter)

        self.random_patch_widgets_container = QWidget()
        random_patch_widgets_layout = QHBoxLayout(self.random_patch_widgets_container)
        random_patch_widgets_layout.setContentsMargins(0, 0, 0, 0)
        random_patch_widgets_layout.setSpacing(CONTENT_LAYOUT_SPACING)
        self.random_patch_min_size_groupbox = QGroupBox('')
        random_min_layout = QVBoxLayout(self.random_patch_min_size_groupbox)
        random_min_layout.setContentsMargins(*CONTENT_LAYOUT_MARGINS)
        random_min_layout.addWidget(self.random_patch_min_size_widget, alignment=Qt.AlignmentFlag.AlignCenter)
        self.random_patch_max_size_groupbox = QGroupBox('')
        random_max_layout = QVBoxLayout(self.random_patch_max_size_groupbox)
        random_max_layout.setContentsMargins(*CONTENT_LAYOUT_MARGINS)
        random_max_layout.addWidget(self.random_patch_max_size_widget, alignment=Qt.AlignmentFlag.AlignCenter)
        random_patch_widgets_layout.addWidget(self.random_patch_min_size_groupbox, 1)
        random_patch_widgets_layout.addWidget(self.random_patch_max_size_groupbox, 1)
        train_patch_layout.addWidget(self.random_patch_widgets_container)

        self.recognition_patch_size_groupbox = QGroupBox('')
        recognition_patch_layout = QVBoxLayout(self.recognition_patch_size_groupbox)
        recognition_patch_layout.setContentsMargins(*CONTENT_LAYOUT_MARGINS)
        recognition_patch_layout.setSpacing(CONTENT_LAYOUT_SPACING)
        recognition_patch_layout.addWidget(
            self.recognition_patch_size_widget,
            alignment=Qt.AlignmentFlag.AlignCenter,
        )

        self.recognition_tab_patch_size_groupbox = QGroupBox('')
        recognition_tab_patch_layout = QVBoxLayout(self.recognition_tab_patch_size_groupbox)
        recognition_tab_patch_layout.setContentsMargins(*CONTENT_LAYOUT_MARGINS)
        recognition_tab_patch_layout.setSpacing(CONTENT_LAYOUT_SPACING)
        recognition_tab_patch_layout.addWidget(
            self.recognition_tab_patch_size_widget,
            alignment=Qt.AlignmentFlag.AlignCenter,
        )

        # Size controls use titled groups instead of a description label on the left.
        # Keep the field-row mapping so existing visibility and enablement policy
        # applies to the complete group, including its title.
        self._field_rows[self.train_patch_size_widget] = self.train_patch_size_groupbox
        self._field_rows[self.recognition_patch_size_widget] = self.recognition_patch_size_groupbox
        self._field_rows[self.recognition_tab_patch_size_widget] = self.recognition_tab_patch_size_groupbox
        self._field_rows[self.random_patch_min_size_widget] = self.random_patch_min_size_groupbox
        self._field_rows[self.random_patch_max_size_widget] = self.random_patch_max_size_groupbox

        self.general_groupbox = QGroupBox('')
        self.general_form = self._create_form_layout()
        self.general_groupbox.setLayout(self.general_form)
        self._add_labeled_row(self.general_form, self.nn_model_type, 'model')
        self.training_feature_toggles_widget = QWidget()
        self.training_feature_toggles_layout = QHBoxLayout(self.training_feature_toggles_widget)
        self.training_feature_toggles_layout.setContentsMargins(0, 0, 0, 0)
        self.training_feature_toggles_layout.setSpacing(CONTENT_LAYOUT_SPACING)
        self.training_feature_toggles_layout.addWidget(self.early_stopping_check_box)
        self.training_feature_toggles_layout.addWidget(self.validation_check_box)
        self.training_feature_toggles_layout.addStretch(1)
        self.general_form.addRow(self.training_feature_toggles_widget)
        self._add_labeled_row(self.general_form, self.epochs_spinbox, 'epochs')
        self.general_form.addRow(self.early_stopping_control_warning)
        self._add_labeled_row(self.general_form, self.color_type, 'image_format')
        self.patch_size_section_widget = QWidget()
        patch_size_section_layout = QVBoxLayout(self.patch_size_section_widget)
        patch_size_section_layout.setContentsMargins(0, 0, 0, 0)
        patch_size_section_layout.setSpacing(CONTENT_LAYOUT_SPACING)
        patch_options_layout = QHBoxLayout()
        patch_options_layout.setContentsMargins(0, 0, 0, 0)
        patch_options_layout.setSpacing(CONTENT_LAYOUT_SPACING)
        patch_options_layout.addWidget(self.random_patch_size_check_box)
        patch_options_layout.addStretch(1)
        self.sync_patch_sizes_check_box = QCheckBox('')
        self.sync_patch_sizes_check_box.setChecked(True)
        patch_options_layout.addWidget(self.sync_patch_sizes_check_box)
        patch_size_section_layout.addLayout(patch_options_layout)
        self.patch_size_groups_layout = QGridLayout()
        self.patch_size_groups_layout.setContentsMargins(0, 0, 0, 0)
        self.patch_size_groups_layout.setHorizontalSpacing(CONTENT_LAYOUT_SPACING)
        self.patch_size_groups_layout.setVerticalSpacing(CONTENT_LAYOUT_SPACING)
        patch_size_section_layout.addLayout(self.patch_size_groups_layout)
        self.general_form.addRow(self.patch_size_section_widget)
        self.model_variants_groupbox = QGroupBox('')
        self.model_variants_form = self._create_form_layout()
        self.model_variants_groupbox.setLayout(self.model_variants_form)
        self._add_labeled_row(self.model_variants_form, self.deprecated_model_type, 'deprecated_models')
        self._add_labeled_row(self.model_variants_form, self.experimental_model_type, 'experimental_models')
        self.model_variants_form.addRow(self.sem_segmentation_section_groupboxes['context'])

        self.augmentation_groupbox = QGroupBox('')
        self.augmentation_form = self._create_form_layout()
        self.augmentation_groupbox.setLayout(self.augmentation_form)
        self.shuffle_groupbox = QGroupBox('')
        self.shuffle_form = self._create_form_layout()
        self.shuffle_groupbox.setLayout(self.shuffle_form)
        self.shuffle_form.addRow(self.shuffle_frames_check_box)
        self.shuffle_form.addRow(self.shuffle_patches_in_frame_check_box)
        self.spatial_groupbox = QGroupBox('')
        spatial_layout = QVBoxLayout(self.spatial_groupbox)
        spatial_layout.setContentsMargins(6, 6, 6, 6)
        spatial_layout.setSpacing(6)
        add_compact_row(
            spatial_layout,
            controls=(self.horizontal_rotation, self.vertical_rotation, self.flip_x, self.flip_y),
            stretch_label=False,
        )
        add_compact_row(
            spatial_layout,
            checkbox=self.random_crop_check_box,
            controls=(self._field_with_description(self.crops_per_image_spinbox, 'crops_per_image'),),
        )
        add_compact_row(
            spatial_layout,
            controls=(self._field_with_description(self.shift_spinbox, 'shift'),),
        )
        add_compact_row(
            spatial_layout,
            checkbox=self.scale_augmentation_check_box,
            controls=(self._field_with_description(self.scale_augmentation_strength_spinbox, 'scale_augmentation_strength'),),
        )
        add_compact_row(
            spatial_layout,
            controls=(self._field_with_description(self.augmentation_multiplier_spinbox, 'augmentation_multiplier'),),
        )
        # Legacy state holder; its controls are displayed by the shared editor.
        self.photometric_groupbox = QGroupBox('', self)
        self.photometric_groupbox.hide()
        self.photometric_groupbox.setCheckable(True)
        self.photometric_groupbox.setChecked(False)
        photometric_layout = QVBoxLayout(self.photometric_groupbox)
        photometric_layout.setContentsMargins(6, 6, 6, 6)
        photometric_layout.setSpacing(6)
        add_compact_row(
            photometric_layout,
            controls=(self._field_with_description(self.augmentation_brightness_spinbox, 'augmentation_brightness_strength'),),
        )
        add_compact_row(
            photometric_layout,
            controls=(self._field_with_description(self.augmentation_contrast_spinbox, 'augmentation_contrast_strength'),),
        )
        add_compact_row(
            photometric_layout,
            controls=(self._field_with_description(self.augmentation_gamma_spinbox, 'augmentation_gamma_strength'),),
        )
        self._photometric_noise_row = add_compact_row(
            photometric_layout,
            controls=(
                self._field_with_description(self.augmentation_noise_probability_spinbox, 'augmentation_noise_probability'),
                self._field_with_description(self.augmentation_noise_sigma_spinbox, 'augmentation_noise_sigma'),
            ),
            stretch_label=False,
        )
        self._photometric_blur_row = add_compact_row(
            photometric_layout,
            controls=(
                self._field_with_description(self.augmentation_blur_probability_spinbox, 'augmentation_blur_probability'),
                self._field_with_description(self.augmentation_blur_radius_spinbox, 'augmentation_blur_radius'),
            ),
            stretch_label=False,
        )
        self.tech_augmentation_groupbox = QGroupBox('')
        self.tech_augmentation_groupbox.setCheckable(True)
        self.tech_augmentation_groupbox.setChecked(False)
        tech_layout = QVBoxLayout(self.tech_augmentation_groupbox)
        tech_layout.setContentsMargins(6, 6, 6, 6)
        tech_layout.setSpacing(6)
        add_compact_row(tech_layout, checkbox=self.tech_augmentation_debug_pair_check_box)
        add_compact_row(
            tech_layout,
            controls=(
                self._field_with_description(self.tech_aug_min_operations_spinbox, 'tech_aug_min_operations'),
                self._field_with_description(self.tech_aug_max_operations_spinbox, 'tech_aug_max_operations'),
                self._field_with_description(self.tech_aug_max_changed_pixels_ratio_spinbox, 'tech_aug_max_changed_pixels_ratio'),
                self._field_with_description(self.tech_aug_max_foreground_ratio_delta_spinbox, 'tech_aug_max_foreground_ratio_delta'),
            ),
            stretch_label=False,
        )
        for control, key in (
            (self.tech_aug_global_width_probability_spinbox, 'tech_aug_global_width_probability'),
            (self.tech_aug_scale_rethreshold_probability_spinbox, 'tech_aug_scale_rethreshold_probability'),
            (self.tech_aug_blur_threshold_probability_spinbox, 'tech_aug_blur_threshold_probability'),
            (self.tech_aug_boundary_aware_probability_spinbox, 'tech_aug_boundary_aware_probability'),
            (self.tech_aug_local_morphology_probability_spinbox, 'tech_aug_local_morphology_probability'),
            (self.tech_aug_gap_variation_probability_spinbox, 'tech_aug_gap_variation_probability'),
        ):
            add_compact_row(
                tech_layout,
                controls=(self._field_with_description(control, key),),
            )
        self.synthetic_defect_generator_groupbox = QGroupBox('')
        self.synthetic_defect_generator_groupbox.setCheckable(True)
        self.synthetic_defect_generator_groupbox.setChecked(False)
        synthetic_layout = QVBoxLayout(self.synthetic_defect_generator_groupbox)
        synthetic_layout.setContentsMargins(6, 6, 6, 6)
        synthetic_layout.setSpacing(6)
        for field, key in (
            (self.synthetic_topology_domain_combo, 'synthetic_topology_domain'),
            (self.pcb_topology_family_combo, 'pcb_topology_family'),
            (self.ic_topology_family_combo, 'ic_topology_family'),
            (self.synthetic_dataset_factor_spinbox, 'synthetic_dataset_factor'),
            (self.synthetic_image_size_widget, 'synthetic_image_size'),
            (self.synthetic_trace_count_range_widget, 'synthetic_trace_count'),
            (self.synthetic_segment_count_range_widget, 'synthetic_segment_count'),
            (self.synthetic_trace_half_width_range_widget, 'synthetic_trace_half_width'),
            (self.synthetic_background_noise_sigma_range_widget, 'synthetic_background_noise_sigma'),
            (self.synthetic_trace_noise_sigma_range_widget, 'synthetic_trace_noise_sigma'),
        ):
            add_compact_row(
                synthetic_layout,
                controls=(self._field_with_description(field, key),),
            )
        add_compact_row(
            synthetic_layout,
            controls=(
                self._field_with_description(self.pcb_defects_probability_spinbox, 'pcb_defects_probability'),
                self._field_with_description(self.pcb_defects_min_count_spinbox, 'pcb_defects_min_count'),
                self._field_with_description(self.pcb_defects_max_count_spinbox, 'pcb_defects_max_count'),
            ),
            stretch_label=False,
        )
        for defect_name, label_key in PCB_DEFECT_WEIGHT_FIELDS:
            add_compact_row(
                synthetic_layout,
                checkbox=self.pcb_defect_type_checkboxes[defect_name],
                controls=(self._field_with_description(self.pcb_defect_type_spinboxes[defect_name], label_key),),
            )
        for defect_name, label_key in IC_DEFECT_WEIGHT_FIELDS:
            add_compact_row(
                synthetic_layout,
                checkbox=self.ic_defect_type_checkboxes[defect_name],
                controls=(self._field_with_description(self.ic_defect_type_spinboxes[defect_name], label_key),),
            )
        self.cutout_groupbox = QGroupBox('')
        self.cutout_groupbox.setCheckable(True)
        self.cutout_groupbox.setChecked(False)
        cutout_layout = QVBoxLayout(self.cutout_groupbox)
        cutout_layout.setContentsMargins(6, 6, 6, 6)
        cutout_layout.setSpacing(6)
        add_compact_row(
            cutout_layout,
            controls=(
                self._field_with_description(self.cutout_probability_spinbox, 'cutout_probability'),
                self._field_with_description(self.cutout_holes_spinbox, 'cutout_holes'),
                self._field_with_description(self.cutout_size_ratio_spinbox, 'cutout_size_ratio'),
            ),
            stretch_label=False,
        )
        self.random_artifacts_groupbox = QGroupBox('')
        self.random_artifacts_groupbox.setCheckable(True)
        self.random_artifacts_groupbox.setChecked(False)
        random_artifacts_layout = QVBoxLayout(self.random_artifacts_groupbox)
        random_artifacts_layout.setContentsMargins(6, 6, 6, 6)
        random_artifacts_layout.setSpacing(6)
        add_compact_row(
            random_artifacts_layout,
            controls=tuple(self.random_artifact_type_checkboxes[artifact_name] for artifact_name in RANDOM_ARTIFACT_TYPES),
            stretch_label=False,
        )
        add_compact_row(
            random_artifacts_layout,
            controls=(
                self._field_with_description(self.random_artifacts_probability_spinbox, 'random_artifacts_probability'),
                self._field_with_description(self.random_artifacts_count_spinbox, 'random_artifacts_count'),
                self._field_with_description(self.random_artifacts_size_ratio_spinbox, 'random_artifacts_size_ratio'),
            ),
            stretch_label=False,
        )
        self.mixup_groupbox = QGroupBox('')
        self.mixup_groupbox.setCheckable(True)
        self.mixup_groupbox.setChecked(False)
        mixup_layout = QVBoxLayout(self.mixup_groupbox)
        mixup_layout.setContentsMargins(6, 6, 6, 6)
        mixup_layout.setSpacing(6)
        add_compact_row(
            mixup_layout,
            controls=(
                self._field_with_description(self.mixup_probability_spinbox, 'mixup_probability'),
                self._field_with_description(self.mixup_alpha_spinbox, 'mixup_alpha'),
            ),
            stretch_label=False,
        )
        self.additional_augmentation_check_box = self.photometric_groupbox
        self.tech_augmentation_check_box = self.tech_augmentation_groupbox
        self.synthetic_defect_generator_check_box = self.synthetic_defect_generator_groupbox
        self.cutout_check_box = self.cutout_groupbox
        self.random_artifacts_check_box = self.random_artifacts_groupbox
        self.mixup_check_box = self.mixup_groupbox
        self.batch_augmentation_widget = QWidget()
        batch_augmentation_layout = QVBoxLayout(self.batch_augmentation_widget)
        batch_augmentation_layout.setContentsMargins(0, 0, 0, 0)
        batch_augmentation_layout.setSpacing(CONTENT_LAYOUT_SPACING)
        batch_augmentation_layout.addWidget(self.cutout_groupbox)
        batch_augmentation_layout.addWidget(self.random_artifacts_groupbox)
        batch_augmentation_layout.addWidget(self.mixup_groupbox)
        self.sem_normalization_editor = SemNormalizationEditor(
            self.preprocessing_groupbox, title='SEM normalization'
        )

        def _pair(*controls: QWidget) -> QWidget:
            widget = QWidget()
            widget.setMaximumWidth(180)
            row = QHBoxLayout(widget)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(4)
            for control in controls:
                row.addWidget(control)
            return widget

        def _stack(*controls: QWidget) -> QWidget:
            widget = QWidget()
            widget.setMaximumWidth(180)
            column = QVBoxLayout(widget)
            column.setContentsMargins(0, 0, 0, 0)
            column.setSpacing(2)
            for control in controls:
                column.addWidget(control)
            return widget

        sem_controls = self.sem_segmentation_controls
        self.training_augmentation_editor = TrainingAugmentationEditor(
            (
                AugmentationBlock('spatial', 'Пространственные', 'Spatial'),
                AugmentationBlock('photometric', 'Фотометрические', 'Photometric'),
                AugmentationBlock(
                    'sem',
                    'SEM-аугментации',
                    'SEM augmentations',
                    sem_controls['aug_enabled'],
                    sem_controls['aug_plan'],
                ),
            ),
            (
                AugmentationRow('rotate_90', 'spatial', 'Поворот на 90°', 'Rotate 90°', self.horizontal_rotation, self.horizontal_rotation_probability_spinbox),
                AugmentationRow('rotate_180', 'spatial', 'Поворот на 180°', 'Rotate 180°', self.vertical_rotation, self.vertical_rotation_probability_spinbox),
                AugmentationRow('flip_x', 'spatial', 'Отражение по горизонтали', 'Horizontal flip', self.flip_x, self.flip_x_probability_spinbox),
                AugmentationRow('flip_y', 'spatial', 'Отражение по вертикали', 'Vertical flip', self.flip_y, self.flip_y_probability_spinbox),
                AugmentationRow('scale', 'spatial', 'Изменение масштаба', 'Scale', self.scale_augmentation_check_box, self.scale_augmentation_probability_spinbox, self.scale_augmentation_strength_spinbox),
                AugmentationRow('brightness', 'photometric', 'Яркость', 'Brightness', self.brightness_augmentation_check_box, self.augmentation_brightness_probability_spinbox, self.augmentation_brightness_spinbox),
                AugmentationRow('contrast', 'photometric', 'Контраст', 'Contrast', self.contrast_augmentation_check_box, self.augmentation_contrast_probability_spinbox, self.augmentation_contrast_spinbox),
                AugmentationRow('gamma', 'photometric', 'Гамма', 'Gamma', self.gamma_augmentation_check_box, self.augmentation_gamma_probability_spinbox, self.augmentation_gamma_spinbox),
                AugmentationRow('noise', 'photometric', 'Гауссов шум', 'Gaussian noise', self.noise_augmentation_check_box, self.augmentation_noise_probability_spinbox, self.augmentation_noise_sigma_spinbox),
                AugmentationRow('blur', 'photometric', 'Размытие', 'Blur', self.blur_augmentation_check_box, self.augmentation_blur_probability_spinbox, self.augmentation_blur_radius_spinbox),
                AugmentationRow('cutout', 'photometric', 'Cutout', 'Cutout', self.cutout_groupbox, self.cutout_probability_spinbox, _pair(self.cutout_holes_spinbox, self.cutout_size_ratio_spinbox)),
                AugmentationRow('random_artifacts', 'photometric', 'Случайные артефакты', 'Random artifacts', self.random_artifacts_groupbox, self.random_artifacts_probability_spinbox, _stack(*self.random_artifact_type_checkboxes.values(), _pair(self.random_artifacts_count_spinbox, self.random_artifacts_size_ratio_spinbox))),
                AugmentationRow('mixup', 'photometric', 'MixUp', 'MixUp', self.mixup_groupbox, self.mixup_probability_spinbox, self.mixup_alpha_spinbox),
                AugmentationRow('topology_width', 'spatial', 'Изменение ширины проводников', 'Conductor width variation', self.tech_augmentation_operation_checkboxes['global_width'], self.tech_aug_global_width_probability_spinbox, _pair(self.tech_aug_min_operations_spinbox, self.tech_aug_max_operations_spinbox)),
                AugmentationRow('topology_scale', 'spatial', 'Масштабирование с бинаризацией', 'Scale and re-threshold', self.tech_augmentation_operation_checkboxes['scale_rethreshold'], self.tech_aug_scale_rethreshold_probability_spinbox, self.tech_aug_max_changed_pixels_ratio_spinbox),
                AugmentationRow('topology_blur', 'spatial', 'Размытие границы маски', 'Blur and threshold mask', self.tech_augmentation_operation_checkboxes['blur_threshold'], self.tech_aug_blur_threshold_probability_spinbox, self.tech_aug_max_foreground_ratio_delta_spinbox),
                AugmentationRow('topology_boundary', 'spatial', 'Вариация границ', 'Boundary-aware variation', self.tech_augmentation_operation_checkboxes['boundary_aware'], self.tech_aug_boundary_aware_probability_spinbox),
                AugmentationRow('topology_local', 'spatial', 'Локальная морфология', 'Local morphology', self.tech_augmentation_operation_checkboxes['local_morphology'], self.tech_aug_local_morphology_probability_spinbox),
                AugmentationRow('topology_gap', 'spatial', 'Вариация разрывов', 'Gap variation', self.tech_augmentation_operation_checkboxes['gap_variation'], self.tech_aug_gap_variation_probability_spinbox),
                AugmentationRow('charging', 'sem', 'Эффект заряда', 'Charging artifacts', sem_controls['aug_charging'], sem_controls['aug_charging_probability'], sem_controls['aug_charging_strength']),
                AugmentationRow('drift', 'sem', 'Дрейф сканирования', 'Scan drift', sem_controls['aug_drift'], sem_controls['aug_drift_probability'], sem_controls['aug_drift_pixels']),
                AugmentationRow('focus', 'sem', 'Локальная расфокусировка', 'Local focus variation', sem_controls['aug_focus'], sem_controls['aug_focus_probability'], sem_controls['aug_focus_sigma']),
                AugmentationRow('detector_noise', 'sem', 'Шум детектора', 'Detector noise', sem_controls['aug_noise'], sem_controls['aug_noise_probability'], _pair(sem_controls['aug_peak_electrons'], sem_controls['aug_read_noise'])),
                AugmentationRow('gain', 'sem', 'Градиент яркости', 'Brightness gradient', sem_controls['aug_gradient'], sem_controls['aug_gradient_probability'], sem_controls['aug_gain_strength']),
                AugmentationRow('scan_defects', 'sem', 'Дефекты сканирования', 'Realistic scan defects', sem_controls['aug_defects'], sem_controls['aug_defects_probability']),
            ),
            reset_defaults=self._reset_training_augmentation_defaults,
            title='Training augmentations',
        )

        self.patch_sampling_mode_groupbox = QGroupBox('Режим нарезки')
        sampling_mode_layout = QGridLayout(self.patch_sampling_mode_groupbox)
        self.grid_sampling_radio = QRadioButton('По сетке')
        self.random_sampling_radio = QRadioButton('Случайная')
        self.patch_sampling_mode_buttons = QButtonGroup(self.patch_sampling_mode_groupbox)
        self.patch_sampling_mode_buttons.addButton(self.grid_sampling_radio)
        self.patch_sampling_mode_buttons.addButton(self.random_sampling_radio)
        sampling_mode_layout.addWidget(self.grid_sampling_radio, 0, 0)
        sampling_mode_layout.addWidget(self.shift_spinbox, 0, 1)
        sampling_mode_layout.addWidget(self.random_sampling_radio, 1, 0)
        sampling_mode_layout.addWidget(self.crops_per_image_spinbox, 1, 1)
        self.grid_sampling_radio.setToolTip('Патчи располагаются по регулярной сетке; справа задаётся шаг в пикселях.')
        self.shift_spinbox.setToolTip('Шаг сетки между соседними патчами, пикс.')
        self.random_sampling_radio.setToolTip('Для каждого кадра выбираются случайные позиции патчей.')
        self.crops_per_image_spinbox.setToolTip('Количество случайных фрагментов на один кадр.')
        self.grid_sampling_radio.setChecked(not self.random_crop_check_box.isChecked())
        self.random_sampling_radio.setChecked(self.random_crop_check_box.isChecked())
        self.random_sampling_radio.toggled.connect(self.random_crop_check_box.setChecked)
        self.random_crop_check_box.toggled.connect(self._sync_patch_sampling_mode)
        self.random_sampling_radio.toggled.connect(self._sync_patch_sampling_mode)
        self.pcb_defects_groupbox = QGroupBox('')
        self.pcb_defects_form = self._create_form_layout()
        self.pcb_defects_groupbox.setLayout(self.pcb_defects_form)
        self.additional_augmentation_check_box.toggled.connect(self._sync_augmentation_controls)
        for checkbox in (
            self.brightness_augmentation_check_box,
            self.contrast_augmentation_check_box,
            self.gamma_augmentation_check_box,
            self.noise_augmentation_check_box,
            self.blur_augmentation_check_box,
        ):
            checkbox.toggled.connect(self._sync_basic_photometric_master)
        for checkbox in self.tech_augmentation_operation_checkboxes.values():
            checkbox.toggled.connect(self._sync_tech_operation_controls)
        self.cutout_check_box.toggled.connect(self._sync_training_augmentation_controls)
        self.random_artifacts_check_box.toggled.connect(self._sync_training_augmentation_controls)
        for checkbox in self.random_artifact_type_checkboxes.values():
            checkbox.toggled.connect(self._sync_training_augmentation_controls)
        self.mixup_check_box.toggled.connect(self._sync_training_augmentation_controls)
        self.synthetic_defect_generator_check_box.toggled.connect(self._sync_synthetic_defect_generator_controls)
        self.synthetic_topology_domain_combo.currentIndexChanged.connect(self._sync_synthetic_domain_controls)
        self.pcb_defects_min_count_spinbox.valueChanged.connect(self._sync_pcb_defect_count_bounds)
        self.pcb_defects_max_count_spinbox.valueChanged.connect(self._sync_pcb_defect_count_bounds)
        for checkbox in self.pcb_defect_type_checkboxes.values():
            checkbox.toggled.connect(self._sync_synthetic_defect_generator_controls)
        for checkbox in self.ic_defect_type_checkboxes.values():
            checkbox.toggled.connect(self._sync_synthetic_defect_generator_controls)
        self.augmentation_blur_probability_spinbox.valueChanged.connect(
            lambda *_args, **_kwargs: self._sync_augmentation_controls(self.additional_augmentation_check_box.isChecked())
        )
        self._sync_synthetic_domain_controls()
        self._sync_augmentation_controls(self.additional_augmentation_check_box.isChecked())
        self._sync_synthetic_defect_generator_controls(self.synthetic_defect_generator_check_box.isChecked())
        self._sync_training_augmentation_controls()

        self.validation_groupbox = QGroupBox('')
        self.validation_form = self._create_form_layout()
        self.validation_groupbox.setLayout(self.validation_form)
        self._add_labeled_row(self.validation_form, self.validation_mode_combo, 'validation_source')
        self._add_labeled_row(self.validation_form, self.validation_spinbox, 'validation_percent')
        self._add_labeled_row(self.validation_form, self.validation_image_path_label, 'validation_image_path')
        self._add_labeled_row(self.validation_form, self.validation_label_path_label, 'validation_label_path')
        self.validation_form.addRow(self.save_validation_binary_images_check_box)
        self.validation_form.addRow(self.sem_segmentation_section_groupboxes['validation'])
        self.validation_form.addRow(self.sem_segmentation_section_groupboxes['experiment'])

        self.preprocessing_groupbox = self.sem_segmentation_section_groupboxes['preprocessing']
        self.preprocessing_groupbox.toggled.connect(self._on_preprocessing_group_toggled)

        self.main_form = self.general_form
        self.reset_defaults_button = QPushButton('')

        self.settings_tabs = QTabWidget()
        self.settings_tabs.setDocumentMode(True)

        def _container(*widgets: QWidget) -> QWidget:
            container = QWidget()
            container_layout = QVBoxLayout(container)
            container_layout.setContentsMargins(*CONTENT_LAYOUT_MARGINS)
            container_layout.setSpacing(CONTENT_LAYOUT_SPACING)
            for widget in widgets:
                container_layout.addWidget(widget)
            return container

        def _page() -> tuple[QWidget, QVBoxLayout]:
            page = QWidget()
            page_layout = QVBoxLayout(page)
            page_layout.setContentsMargins(0, 0, 0, 0)
            page_layout.setSpacing(CONTENT_LAYOUT_SPACING)
            return page, page_layout

        def _card(
            key: str,
            title: str,
            *widgets: QWidget,
            expanded: bool = True,
            expert_only: bool = False,
        ) -> SettingsCard:
            card = SettingsCard(title, _container(*widgets), expanded=expanded)
            card.setProperty('settingsCardKey', key)
            card.setProperty('expertOnly', expert_only)
            self._settings_cards[key] = card
            if expert_only:
                self._expert_widgets.append(card)
            return card

        # Data owns image representation, patch generation, augmentation and sampling.
        self.data_page, self.data_page_layout = _page()
        self.data_page_layout.addWidget(
            _card(
                'data_source',
                'Dataset and patches',
                self.samples_number,
                self._field_rows[self.color_type],
                self.patch_size_section_widget,
                self.prepare_samples_groupbox,
                self.synthetic_defect_generator_groupbox,
            )
        )
        self.data_page_layout.addWidget(_card('preprocessing', 'SEM normalization', self.sem_normalization_editor))
        self.data_page_layout.addWidget(
            _card(
                'augmentation',
                'Training augmentation',
                self.training_augmentation_editor,
                self.augmentation_preview_button,
                expanded=False,
            )
        )
        self.data_page_layout.addWidget(
            _card(
                'sampling',
                'Patch sampling',
                self.patch_sampling_mode_groupbox,
                self.shuffle_groupbox,
                self.rare_patch_groupbox,
                self.hard_mining_groupbox,
                expanded=False,
            )
        )
        self.data_page_layout.addWidget(
            _card('validation_data', 'Validation data', self.validation_check_box, self.validation_groupbox)
        )
        self.data_page_layout.addStretch(1)

        # Model/training owns architecture, supervision, optimization and metrics.
        self.training_page, self.training_page_layout = _page()
        self.training_page_layout.addWidget(
            _card(
                'architecture',
                'Architecture',
                self._field_rows[self.nn_model_type],
                self.model_variants_groupbox,
            )
        )
        self.training_page_layout.addWidget(
            _card(
                'training',
                'Training process',
                self._field_rows[self.epochs_spinbox],
                self.early_stopping_check_box,
                self.early_stopping_control_warning,
                self.optimizer_groupbox,
            )
        )
        self.training_page_layout.addWidget(
            _card(
                'supervision',
                'Topology supervision',
                self.sem_segmentation_section_groupboxes['basic_targets'],
                self.sem_segmentation_section_groupboxes['geometry_targets'],
                expanded=False,
                expert_only=True,
            )
        )
        self.training_page_layout.addWidget(_card('losses', 'Loss functions', self.precision_loss_groupbox))
        self.training_page_layout.addWidget(
            _card(
                'schedule',
                'Learning-rate schedule',
                self.warmup_groupbox,
                self.scheduler_groupbox,
                expanded=False,
                expert_only=True,
            )
        )
        self.training_page_layout.addWidget(
            _card(
                'confidence_training',
                'Confidence-head training',
                self.sem_segmentation_section_contents['confidence_training'],
                expanded=False,
                expert_only=True,
            )
        )
        self.training_page_layout.addWidget(
            _card(
                'validation_metrics',
                'Validation metrics and experiment',
                self.sem_segmentation_section_groupboxes['validation'],
                self.sem_segmentation_section_groupboxes['experiment'],
                expanded=False,
                expert_only=True,
            )
        )
        self.training_page_layout.addWidget(
            _card('runtime', 'Runtime and GPU', self.runtime_groupbox, expanded=False, expert_only=True)
        )
        self.training_page_layout.addStretch(1)

        # Recognition owns prediction, uncertainty and annotation export only.
        self.recognition_page, self.recognition_page_layout = _page()
        self.recognition_page_layout.addWidget(
            _card(
                'recognition',
                'Mask prediction',
                self.recognition_tab_patch_size_groupbox,
                self.recognition_groupbox,
            )
        )
        self.recognition_page_layout.addWidget(
            _card(
                'inference_uncertainty',
                'Prediction uncertainty',
                self.sem_segmentation_section_contents['inference_uncertainty'],
                expanded=False,
            )
        )
        self.recognition_page_layout.addWidget(
            _card(
                'active_learning',
                'NeedsAnnotation export',
                self.sem_segmentation_section_contents['active_learning'],
                expanded=False,
                expert_only=True,
            )
        )
        self.recognition_page_layout.addStretch(1)
        for section_key in ('confidence_training', 'inference_uncertainty', 'active_learning'):
            self.sem_segmentation_section_contents[section_key].setVisible(True)

        self._expert_widgets.extend(
            (
                self.cutout_groupbox,
                self.random_artifacts_groupbox,
                self.synthetic_defect_generator_groupbox,
                self.mixup_groupbox,
                self.model_variants_groupbox,
                self.optimizer_advanced_groupbox,
                self.loss_advanced_groupbox,
                self.hard_mining_groupbox,
            )
        )
        for widget in self._expert_widgets:
            widget.setProperty('expertOnly', True)

        self._page_indexes = {
            'data': self.settings_tabs.addTab(self.data_page, ''),
            'training': self.settings_tabs.addTab(self.training_page, ''),
            'recognition': self.settings_tabs.addTab(self.recognition_page, ''),
        }

        self.settings_search = QLineEdit()
        self.settings_search.setClearButtonEnabled(True)
        self.settings_search_completer = QCompleter(self)
        self.settings_search_completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.settings_search_completer.setFilterMode(Qt.MatchFlag.MatchContains)
        self.settings_search_model = QStringListModel(self)
        self.settings_search_completer.setModel(self.settings_search_model)
        self.settings_search.setCompleter(self.settings_search_completer)
        self.expert_mode_check_box = QCheckBox('Expert mode')
        # Attribute compatibility only: the old container no longer exists in the UI.
        self.expert_groupbox = self.expert_mode_check_box
        settings = _settings_panel_qsettings()
        self.expert_mode_check_box.setChecked(settings.value('expert_mode', False, type=bool))
        self.configuration_status_button = QToolButton()
        self.configuration_status_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.configuration_status_button.setAutoRaise(True)

        header_layout = QGridLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(CONTENT_LAYOUT_SPACING)
        header_layout.addWidget(self.settings_search, 0, 0, 1, 2)
        header_layout.addWidget(self.expert_mode_check_box, 1, 0)
        header_layout.addWidget(
            self.configuration_status_button,
            1,
            1,
            alignment=Qt.AlignmentFlag.AlignRight,
        )

        layout = QVBoxLayout()
        layout.setContentsMargins(*CONTENT_LAYOUT_MARGINS)
        layout.setSpacing(CONTENT_LAYOUT_SPACING)
        layout.addLayout(header_layout)
        layout.addWidget(self.settings_tabs)

        reset_row = QHBoxLayout()
        reset_row.setContentsMargins(0, 0, 0, 0)
        reset_row.addStretch(1)
        reset_row.addWidget(self.reset_defaults_button, 0, Qt.AlignmentFlag.AlignRight)
        layout.addLayout(reset_row)
        layout.addStretch(1)

        self.general_form.setAlignment(self.nn_model_type, Qt.AlignmentFlag.AlignRight)
        self.general_form.setAlignment(self.epochs_spinbox, Qt.AlignmentFlag.AlignRight)
        self.general_form.setAlignment(self.color_type, Qt.AlignmentFlag.AlignRight)
        self.model_variants_form.setAlignment(self.deprecated_model_type, Qt.AlignmentFlag.AlignRight)
        self.model_variants_form.setAlignment(self.experimental_model_type, Qt.AlignmentFlag.AlignRight)
        self.validation_form.setAlignment(self.validation_mode_combo, Qt.AlignmentFlag.AlignRight)
        self.validation_form.setAlignment(self.validation_spinbox, Qt.AlignmentFlag.AlignRight)
        self.recognition_form.setAlignment(self.recognition_batch_spinbox, Qt.AlignmentFlag.AlignRight)
        self.recognition_form.setAlignment(self.overlap_spinbox, Qt.AlignmentFlag.AlignRight)
        self.recognition_form.setAlignment(self.recognition_jpeg_quality_spinbox, Qt.AlignmentFlag.AlignRight)
        self.expert_mode_check_box.toggled.connect(self._sync_expert_mode)
        self.settings_search.returnPressed.connect(self._activate_first_search_result)
        self.settings_search_completer.activated.connect(self._activate_search_result)
        self.configuration_status_button.clicked.connect(self._focus_first_configuration_error)
        self.epochs_spinbox.valueChanged.connect(self._update_card_summaries)
        self.nn_model_type.currentIndexChanged.connect(self._update_card_summaries)
        self.optimizer_type.currentIndexChanged.connect(self._update_card_summaries)
        self.recognition_batch_spinbox.valueChanged.connect(self._update_card_summaries)
        self._search_shortcut = QShortcut(QKeySequence.StandardKey.Find, self)
        self._search_shortcut.activated.connect(self.settings_search.setFocus)
        self.random_patch_size_check_box.toggled.connect(self._sync_random_patch_size_controls)
        self._sync_expert_mode(self.expert_mode_check_box.isChecked())
        self._sync_random_patch_size_controls()

        self._content_widget.setLayout(layout)
        self._rebuild_settings_registry()
        for card in self._settings_cards.values():
            card.capture_defaults()
        QTimer.singleShot(0, self._validate_configuration_live)
