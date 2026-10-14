"""settings_panel: construction 2."""
from __future__ import annotations
from PyQt6.QtWidgets import (
    QWidget,
    QCheckBox,
    QFormLayout,
    QLabel,
    QPushButton,
    QGroupBox,
    QVBoxLayout,
    QHBoxLayout,
)
from neuralimage.view.settings_panel_widgets import (
    NoWheelComboBox,
    SlidingPanel,  # noqa: F401 - compatibility re-export from neuralimage.view
    create_double_spinbox,
    create_spinbox,
)
from neuralimage.view.sem_compact_section_editor import CompactSemSectionEditor
from neuralimage.configuration import (
    SEM_UI_SECTIONS,
    sem_ui_section_help,
)

from neuralimage.view.settings_panel_parts.module_state import (
    CONFIDENCE_SAVE_MODES,
    CONTENT_LAYOUT_SPACING,
    FORM_DEFAULT_MARGINS,
    LOSS_FUNCTIONS,
    MAX_BATCH,
    MAX_DATALOADER_WORKERS,
    MAX_HARD_PIXEL_KEEP_RATIO,
    MAX_JPEG_QUALITY,
    MAX_LEARNING_RATE,
    MAX_LOG_UPDATE_FREQUENCY,
    MAX_OVERLAP,
    MAX_POSTPROCESS_KERNEL_SIZE,
    MAX_RARE_PATCH_OVERSAMPLING_FACTOR,
    MAX_RECOGNITION_THRESHOLD,
    MAX_SCHEDULER_COOLDOWN,
    MAX_SCHEDULER_DIV_FACTOR,
    MAX_SCHEDULER_FACTOR,
    MAX_SCHEDULER_FINAL_DIV_FACTOR,
    MAX_SCHEDULER_MIN_LR,
    MAX_SCHEDULER_PATIENCE,
    MAX_SCHEDULER_STEP_SIZE,
    MAX_SCHEDULER_THRESHOLD,
    MAX_SCHEDULER_T_MAX,
    MAX_WARMUP_EPOCHS,
    MAX_WARMUP_START_FACTOR,
    MAX_WEIGHT_DECAY,
    MIN_BATCH,
    MIN_DATALOADER_WORKERS,
    MIN_HARD_PIXEL_KEEP_RATIO,
    MIN_JPEG_QUALITY,
    MIN_LEARNING_RATE,
    MIN_LOG_UPDATE_FREQUENCY,
    MIN_OVERLAP,
    MIN_POSTPROCESS_KERNEL_SIZE,
    MIN_RARE_PATCH_OVERSAMPLING_FACTOR,
    MIN_RECOGNITION_THRESHOLD,
    MIN_SCHEDULER_COOLDOWN,
    MIN_SCHEDULER_DIV_FACTOR,
    MIN_SCHEDULER_FACTOR,
    MIN_SCHEDULER_FINAL_DIV_FACTOR,
    MIN_SCHEDULER_MIN_LR,
    MIN_SCHEDULER_PATIENCE,
    MIN_SCHEDULER_STEP_SIZE,
    MIN_SCHEDULER_THRESHOLD,
    MIN_SCHEDULER_T_MAX,
    MIN_WARMUP_EPOCHS,
    MIN_WARMUP_START_FACTOR,
    MIN_WEIGHT_DECAY,
    MIXED_PRECISION_MODES,
    MULTI_GPU_MODES,
    ONE_CYCLE_ANNEAL_STRATEGIES,
    OPTIMIZERS,
    OPTIMIZER_PRESETS,
    SCHEDULER_NAMES,
    VISIBLE_OPTIMIZER_PRESET_NAMES,
)


class Construction2Mixin:
    def _init_nn_auxilary_settings(self) -> None:
        self.nn_auxilary_settings_groupbox = QGroupBox('')
        nn_sections_layout = QVBoxLayout()
        nn_sections_layout.setContentsMargins(*FORM_DEFAULT_MARGINS)
        nn_sections_layout.setSpacing(CONTENT_LAYOUT_SPACING)
        self.nn_auxilary_settings_groupbox.setLayout(nn_sections_layout)

        def _build_subgroup() -> tuple[QGroupBox, QFormLayout]:
            group = QGroupBox('')
            form = self._create_form_layout()
            group.setLayout(form)
            return group, form

        self.optimizer_preset_buttons: list[QPushButton] = []
        self._visible_optimizer_presets: list[tuple[str, str, float, float]] = []
        self.optimizer_presets_widget = QWidget()
        preset_layout = QHBoxLayout(self.optimizer_presets_widget)
        preset_layout.setContentsMargins(0, 0, 0, 0)
        preset_layout.setSpacing(6)
        for title, optimizer_name, learning_rate, weight_decay in OPTIMIZER_PRESETS:
            if optimizer_name not in VISIBLE_OPTIMIZER_PRESET_NAMES:
                continue
            btn = QPushButton(title)
            btn.setCheckable(True)
            btn.setProperty("selectionRole", "mode")
            btn.setProperty("presetRole", "quickPreset")
            btn.clicked.connect(
                lambda _checked=False, n=optimizer_name, lr=learning_rate, wd=weight_decay: self._apply_optimizer_preset(
                    n, lr, wd
                )
            )
            self.optimizer_preset_buttons.append(btn)
            self._visible_optimizer_presets.append((title, optimizer_name, learning_rate, weight_decay))
            preset_layout.addWidget(btn)
        preset_layout.addStretch(1)

        self.optimizer_type = NoWheelComboBox()
        self.optimizer_type.addItems(list(OPTIMIZERS))
        self.mixed_precision_type = NoWheelComboBox()
        self.mixed_precision_type.addItems(list(MIXED_PRECISION_MODES))
        self.mixed_precision_type.setCurrentText('off')
        self.deep_supervision_check_box = QCheckBox('')
        self.deep_supervision_check_box.setChecked(False)
        self.loss_function_type = NoWheelComboBox()
        self.loss_function_type.addItems(list(LOSS_FUNCTIONS))
        self.dice_loss_weight_spinbox = create_double_spinbox((0.0, 1.0), step=0.05, default_value=0.5, decimals=2)
        self.iou_loss_weight_spinbox = create_double_spinbox((0.0, 1.0), step=0.05, default_value=0.5, decimals=2)
        self.loss_function_type.setVisible(False)
        self.dice_loss_weight_spinbox.setVisible(False)
        self.iou_loss_weight_spinbox.setVisible(False)
        self.loss_terms_widget = self._build_loss_terms_widget()
        self.loss_terms_groupbox = QGroupBox('')
        loss_terms_group_layout = QVBoxLayout(self.loss_terms_groupbox)
        loss_terms_group_layout.setContentsMargins(8, 8, 8, 8)
        loss_terms_group_layout.setSpacing(6)
        loss_terms_group_layout.addWidget(self.loss_terms_widget)

        self.learning_rate_spinbox = create_double_spinbox(
            (MIN_LEARNING_RATE, MAX_LEARNING_RATE), step=1e-4, default_value=1e-3, decimals=6
        )
        self.weight_decay_spinbox = create_double_spinbox(
            (MIN_WEIGHT_DECAY, MAX_WEIGHT_DECAY), step=1e-4, default_value=0.0, decimals=6
        )

        self.train_batch_spinbox = create_spinbox((MIN_BATCH, MAX_BATCH), step=1, default_value=16)
        self.dataloader_num_workers_spinbox = create_spinbox(
            (MIN_DATALOADER_WORKERS, MAX_DATALOADER_WORKERS),
            step=1,
            default_value=-1,
        )
        self.dataloader_num_workers_spinbox.setSpecialValueText('auto')
        self.recognition_batch_spinbox = create_spinbox((MIN_BATCH, MAX_BATCH), step=1, default_value=16)
        # Backward-compatible alias (legacy code expects training batch control).
        self.batch_spinbox = self.train_batch_spinbox
        self.overlap_spinbox = create_spinbox((MIN_OVERLAP, MAX_OVERLAP), step=4, default_value=8)
        self.recognition_jpeg_quality_spinbox = create_spinbox(
            (MIN_JPEG_QUALITY, MAX_JPEG_QUALITY),
            step=1,
            default_value=95,
        )
        self.recognition_multiprocessing_check_box = QCheckBox('')
        self.recognition_multiprocessing_check_box.setChecked(True)
        self.recognition_binarize_output_check_box = QCheckBox('')
        self.recognition_binarize_output_check_box.setChecked(True)
        self.recognition_use_auto_threshold_check_box = QCheckBox('')
        self.recognition_use_auto_threshold_check_box.setChecked(True)
        self.recognition_threshold_spinbox = create_double_spinbox(
            (MIN_RECOGNITION_THRESHOLD, MAX_RECOGNITION_THRESHOLD),
            step=0.05,
            default_value=0.5,
            decimals=2,
        )
        self.recognition_tta_check_box = QCheckBox('')
        self.recognition_tta_check_box.setChecked(False)
        self.confidence_save_mode_combo = NoWheelComboBox()
        for value in CONFIDENCE_SAVE_MODES:
            self.confidence_save_mode_combo.addItem(value, value)
        self.recognition_postprocess_check_box = QCheckBox('')
        self.recognition_postprocess_kernel_size_spinbox = create_spinbox(
            (MIN_POSTPROCESS_KERNEL_SIZE, MAX_POSTPROCESS_KERNEL_SIZE),
            step=2,
            default_value=3,
        )
        self.log_update_frequency_spinbox = create_spinbox(
            (MIN_LOG_UPDATE_FREQUENCY, MAX_LOG_UPDATE_FREQUENCY),
            step=1,
            default_value=0,
        )
        self.multi_gpu_mode_combo = NoWheelComboBox()
        self.multi_gpu_mode_combo.addItems(list(MULTI_GPU_MODES))
        # Backward-compatible alias used by tests/older code paths.
        self.multi_gpu_check_box = self.multi_gpu_mode_combo
        self.torch_compile_check_box = QCheckBox('')
        self.warmup_check_box = QCheckBox('')
        self.warmup_epochs_spinbox = create_spinbox((MIN_WARMUP_EPOCHS, MAX_WARMUP_EPOCHS), step=1, default_value=3)
        self.warmup_start_factor_spinbox = create_double_spinbox(
            (MIN_WARMUP_START_FACTOR, MAX_WARMUP_START_FACTOR),
            step=0.01,
            default_value=0.1,
            decimals=3,
        )
        self.scheduler_type_combo = NoWheelComboBox()
        for value in SCHEDULER_NAMES:
            self.scheduler_type_combo.addItem(value, value)
        self.scheduler_plateau_factor_spinbox = create_double_spinbox(
            (MIN_SCHEDULER_FACTOR, MAX_SCHEDULER_FACTOR),
            step=0.05,
            default_value=0.5,
            decimals=2,
        )
        self.scheduler_plateau_patience_spinbox = create_spinbox(
            (MIN_SCHEDULER_PATIENCE, MAX_SCHEDULER_PATIENCE),
            step=1,
            default_value=3,
        )
        self.scheduler_plateau_threshold_spinbox = create_double_spinbox(
            (MIN_SCHEDULER_THRESHOLD, MAX_SCHEDULER_THRESHOLD),
            step=1e-4,
            default_value=1e-4,
            decimals=6,
        )
        self.scheduler_plateau_min_lr_spinbox = create_double_spinbox(
            (MIN_SCHEDULER_MIN_LR, MAX_SCHEDULER_MIN_LR),
            step=1e-5,
            default_value=1e-6,
            decimals=6,
        )
        self.scheduler_plateau_cooldown_spinbox = create_spinbox(
            (MIN_SCHEDULER_COOLDOWN, MAX_SCHEDULER_COOLDOWN),
            step=1,
            default_value=0,
        )
        self.scheduler_cosine_t_max_spinbox = create_spinbox(
            (MIN_SCHEDULER_T_MAX, MAX_SCHEDULER_T_MAX),
            step=1,
            default_value=10,
        )
        self.scheduler_cosine_eta_min_spinbox = create_double_spinbox(
            (MIN_SCHEDULER_MIN_LR, MAX_SCHEDULER_MIN_LR),
            step=1e-5,
            default_value=1e-6,
            decimals=6,
        )
        self.scheduler_one_cycle_max_lr_spinbox = create_double_spinbox(
            (MIN_LEARNING_RATE, MAX_LEARNING_RATE),
            step=1e-4,
            default_value=1e-3,
            decimals=6,
        )
        self.scheduler_one_cycle_pct_start_spinbox = create_double_spinbox(
            (0.0, 1.0),
            step=0.05,
            default_value=0.3,
            decimals=2,
        )
        self.scheduler_one_cycle_anneal_strategy_combo = NoWheelComboBox()
        for value in ONE_CYCLE_ANNEAL_STRATEGIES:
            self.scheduler_one_cycle_anneal_strategy_combo.addItem(value, value)
        self.scheduler_one_cycle_div_factor_spinbox = create_double_spinbox(
            (MIN_SCHEDULER_DIV_FACTOR, MAX_SCHEDULER_DIV_FACTOR),
            step=1.0,
            default_value=25.0,
            decimals=2,
        )
        self.scheduler_one_cycle_final_div_factor_spinbox = create_double_spinbox(
            (MIN_SCHEDULER_FINAL_DIV_FACTOR, MAX_SCHEDULER_FINAL_DIV_FACTOR),
            step=100.0,
            default_value=10000.0,
            decimals=2,
        )
        self.scheduler_one_cycle_three_phase_check_box = QCheckBox('')
        self.scheduler_step_lr_step_size_spinbox = create_spinbox(
            (MIN_SCHEDULER_STEP_SIZE, MAX_SCHEDULER_STEP_SIZE),
            step=1,
            default_value=10,
        )
        self.scheduler_step_lr_gamma_spinbox = create_double_spinbox(
            (MIN_SCHEDULER_FACTOR, MAX_WARMUP_START_FACTOR),
            step=0.05,
            default_value=0.1,
            decimals=2,
        )
        self.hard_mining_check_box = QCheckBox('')
        self.hard_pixel_mining_check_box = QCheckBox('')
        self.hard_pixel_mining_ratio_spinbox = create_double_spinbox(
            (MIN_HARD_PIXEL_KEEP_RATIO, MAX_HARD_PIXEL_KEEP_RATIO),
            step=0.05,
            default_value=0.25,
            decimals=2,
        )
        self.skip_uniform_labels_check_box = QCheckBox('')
        self.rare_patch_oversampling_check_box = QCheckBox('')
        self.rare_patch_oversampling_factor_spinbox = create_spinbox(
            (MIN_RARE_PATCH_OVERSAMPLING_FACTOR, MAX_RARE_PATCH_OVERSAMPLING_FACTOR),
            step=1,
            default_value=2,
        )
        self.edit_rare_regions_button = QPushButton('')
        self.early_stopping_check_box = QCheckBox('')
        self.early_stopping_control_warning = QLabel('')
        self.early_stopping_control_warning.setWordWrap(True)

        self.optimizer_groupbox, self.optimizer_form = _build_subgroup()
        self.precision_loss_groupbox, self.precision_loss_form = _build_subgroup()
        self.rare_patch_groupbox, self.rare_patch_form = _build_subgroup()
        self.recognition_groupbox, self.recognition_form = _build_subgroup()
        self.runtime_groupbox, self.runtime_form = _build_subgroup()
        self.warmup_groupbox, self.warmup_form = _build_subgroup()
        self.scheduler_groupbox, self.scheduler_form = _build_subgroup()
        self.hard_mining_groupbox, self.hard_mining_form = _build_subgroup()
        self.sem_segmentation_groupbox, self.sem_segmentation_form = _build_subgroup()
        self.sem_segmentation_validation_label = QLabel('')
        self.sem_segmentation_validation_label.setWordWrap(True)
        self.sem_segmentation_controls: dict[str, QWidget] = {}
        self.sem_segmentation_field_labels: dict[str, QLabel] = {}
        self.sem_segmentation_error_labels: dict[str, QLabel] = {}
        self.sem_segmentation_section_groupboxes: dict[str, QGroupBox] = {}
        self.sem_segmentation_section_contents: dict[str, QWidget] = {}
        self.sem_segmentation_section_forms: dict[str, QFormLayout] = {}
        self.sem_segmentation_section_editors: dict[str, CompactSemSectionEditor] = {}
        self._sem_segmentation_update_guard = True
        for section_key, _section_title in SEM_UI_SECTIONS:
            section_editor = CompactSemSectionEditor(section_key)
            section_editor.changed.connect(self._on_sem_segmentation_control_changed)
            section_groupbox = section_editor
            content_widget = section_editor
            if section_key != 'augmentation':
                section_groupbox.setToolTip(sem_ui_section_help(section_key, 'en'))
            else:
                section_editor.setToolTip(sem_ui_section_help(section_key, 'en'))
            self.sem_segmentation_section_groupboxes[section_key] = section_groupbox
            self.sem_segmentation_section_contents[section_key] = content_widget
            self.sem_segmentation_section_editors[section_key] = section_editor
            self.sem_segmentation_controls.update(section_editor.controls)
            self.sem_segmentation_field_labels.update(section_editor.labels)
            self.sem_segmentation_error_labels.update(section_editor.error_labels)
        self.sem_segmentation_section_editors['validation'].set_field_visible(
            'validation_enabled',
            False,
        )
        self.sem_segmentation_form.addRow(self.sem_segmentation_section_groupboxes['basic_targets'])
        self.sem_segmentation_form.addRow(self.sem_segmentation_section_groupboxes['geometry_targets'])
        self.preprocessing_groupbox = self.sem_segmentation_section_groupboxes['preprocessing']
        self.set_sem_segmentation_config({})
        self.early_stopping_groupbox, self.early_stopping_form = _build_subgroup()

        # Keep backward compatibility for code that may inspect this attr.
        self.nn_aux_form = self.optimizer_form

        self.optimizer_form.addRow(self.optimizer_presets_widget)
        (
            self.optimizer_advanced_groupbox,
            self.optimizer_advanced_layout,
            self.optimizer_advanced_content_widget,
        ) = self._create_collapsible_groupbox()
        self.optimizer_advanced_form = self._create_form_layout(margins=(0, 0, 0, 0))
        self.optimizer_advanced_layout.addLayout(self.optimizer_advanced_form)
        self._add_labeled_row(self.optimizer_advanced_form, self.learning_rate_spinbox, 'learning_rate')
        self._add_labeled_row(self.optimizer_advanced_form, self.weight_decay_spinbox, 'weight_decay')
        self.optimizer_form.addRow(self.optimizer_advanced_groupbox)

        self.precision_loss_form.addRow(self.loss_presets_widget)
        (
            self.loss_advanced_groupbox,
            self.loss_advanced_layout,
            self.loss_advanced_content_widget,
        ) = self._create_collapsible_groupbox()
        self.loss_advanced_form = self._create_form_layout(margins=(0, 0, 0, 0))
        self.loss_advanced_layout.addLayout(self.loss_advanced_form)
        self.loss_advanced_form.addRow(self.deep_supervision_check_box)
        self.loss_advanced_form.addRow(self.loss_terms_groupbox)
        self.loss_advanced_form.addRow(self.sem_segmentation_section_groupboxes['losses'])
        self.precision_loss_form.addRow(self.loss_advanced_groupbox)

        self.rare_patch_form.addRow(self.skip_uniform_labels_check_box)
        self.rare_patch_form.addRow(self.rare_patch_oversampling_check_box)
        self._add_labeled_row(
            self.rare_patch_form,
            self.rare_patch_oversampling_factor_spinbox,
            'rare_patch_oversampling_factor',
        )
        self.rare_patch_form.addRow(self.edit_rare_regions_button)

        self._add_labeled_row(self.recognition_form, self.recognition_batch_spinbox, 'recognition_batch_size')
        self._add_labeled_row(self.recognition_form, self.overlap_spinbox, 'overlap')
        self._add_labeled_row(self.recognition_form, self.recognition_jpeg_quality_spinbox, 'recognition_jpeg_quality')
        self.recognition_form.addRow(self.recognition_multiprocessing_check_box)
        self.recognition_form.addRow(self.recognition_binarize_output_check_box)
        self.recognition_form.addRow(self.recognition_use_auto_threshold_check_box)
        self._add_labeled_row(self.recognition_form, self.recognition_threshold_spinbox, 'recognition_threshold')
        self.recognition_form.addRow(self.recognition_tta_check_box)
        self._add_labeled_row(self.recognition_form, self.confidence_save_mode_combo, 'confidence_save_mode')
        # The confidence output combo is retained only as an adapter for old
        # snapshots. Recognition TTA remains a separate mask-averaging option.
        self._set_field_visible(self.confidence_save_mode_combo, False)
        self.recognition_form.addRow(self.recognition_postprocess_check_box)
        self._add_labeled_row(
            self.recognition_form,
            self.recognition_postprocess_kernel_size_spinbox,
            'recognition_postprocess_kernel_size',
        )
        self.recognition_form.addRow(self.sem_segmentation_section_groupboxes['inference_uncertainty'])
        self.recognition_form.addRow(self.sem_segmentation_section_groupboxes['active_learning'])
        self._add_labeled_row(self.runtime_form, self.train_batch_spinbox, 'train_batch_size')
        self._add_labeled_row(self.runtime_form, self.dataloader_num_workers_spinbox, 'dataloader_num_workers')
        self._add_labeled_row(self.runtime_form, self.multi_gpu_mode_combo, 'multi_gpu')
        self._add_labeled_row(self.runtime_form, self.mixed_precision_type, 'mixed_precision')
        self._add_labeled_row(self.runtime_form, self.log_update_frequency_spinbox, 'log_update_frequency')
        self.runtime_form.addRow(self.torch_compile_check_box)

        self.warmup_form.addRow(self.warmup_check_box)
        self._add_labeled_row(self.warmup_form, self.warmup_epochs_spinbox, 'warmup_epochs')
        self._add_labeled_row(self.warmup_form, self.warmup_start_factor_spinbox, 'warmup_start_factor')

        self._add_labeled_row(self.scheduler_form, self.scheduler_type_combo, 'scheduler_name')
        self._add_labeled_row(self.scheduler_form, self.scheduler_plateau_factor_spinbox, 'scheduler_plateau_factor')
        self._add_labeled_row(self.scheduler_form, self.scheduler_plateau_patience_spinbox, 'scheduler_plateau_patience')
        self._add_labeled_row(
            self.scheduler_form,
            self.scheduler_plateau_threshold_spinbox,
            'scheduler_plateau_threshold',
        )
        self._add_labeled_row(self.scheduler_form, self.scheduler_plateau_min_lr_spinbox, 'scheduler_plateau_min_lr')
        self._add_labeled_row(
            self.scheduler_form,
            self.scheduler_plateau_cooldown_spinbox,
            'scheduler_plateau_cooldown',
        )
        self._add_labeled_row(self.scheduler_form, self.scheduler_cosine_t_max_spinbox, 'scheduler_cosine_t_max')
        self._add_labeled_row(self.scheduler_form, self.scheduler_cosine_eta_min_spinbox, 'scheduler_cosine_eta_min')
        self._add_labeled_row(self.scheduler_form, self.scheduler_one_cycle_max_lr_spinbox, 'scheduler_one_cycle_max_lr')
        self._add_labeled_row(
            self.scheduler_form,
            self.scheduler_one_cycle_pct_start_spinbox,
            'scheduler_one_cycle_pct_start',
        )
        self._add_labeled_row(
            self.scheduler_form,
            self.scheduler_one_cycle_anneal_strategy_combo,
            'scheduler_one_cycle_anneal_strategy',
        )
        self._add_labeled_row(
            self.scheduler_form,
            self.scheduler_one_cycle_div_factor_spinbox,
            'scheduler_one_cycle_div_factor',
        )
        self._add_labeled_row(
            self.scheduler_form,
            self.scheduler_one_cycle_final_div_factor_spinbox,
            'scheduler_one_cycle_final_div_factor',
        )
        self._add_labeled_row(
            self.scheduler_form,
            self.scheduler_one_cycle_three_phase_check_box,
            'scheduler_one_cycle_three_phase',
        )
        self._add_labeled_row(
            self.scheduler_form,
            self.scheduler_step_lr_step_size_spinbox,
            'scheduler_step_lr_step_size',
        )
        self._add_labeled_row(self.scheduler_form, self.scheduler_step_lr_gamma_spinbox, 'scheduler_step_lr_gamma')

        self.hard_mining_form.addRow(self.hard_mining_check_box)
        self.hard_mining_form.addRow(self.hard_pixel_mining_check_box)
        self._add_labeled_row(self.hard_mining_form, self.hard_pixel_mining_ratio_spinbox, 'hard_pixel_mining_ratio')
        self.hard_mining_form.addRow(self.sem_segmentation_section_groupboxes['hard_mining'])

        nn_sections_layout.addWidget(self.optimizer_groupbox)
        nn_sections_layout.addWidget(self.precision_loss_groupbox)
        nn_sections_layout.addWidget(self.rare_patch_groupbox)
        nn_sections_layout.addWidget(self.warmup_groupbox)
        nn_sections_layout.addWidget(self.scheduler_groupbox)
        nn_sections_layout.addWidget(self.hard_mining_groupbox)
        nn_sections_layout.addWidget(self.sem_segmentation_groupbox)
        nn_sections_layout.addWidget(self.runtime_groupbox)
        nn_sections_layout.addStretch(1)

        self._sync_active_optimizer_preset()
        self.warmup_check_box.toggled.connect(self._sync_warmup_controls)
        self.scheduler_type_combo.currentIndexChanged.connect(self._sync_scheduler_controls)
        self.hard_mining_check_box.toggled.connect(self._sync_hard_mining_controls)
        self.hard_pixel_mining_check_box.toggled.connect(self._sync_hard_mining_controls)
        self.validation_check_box.toggled.connect(lambda _checked: self._sync_sem_segmentation_controls())
        self.early_stopping_check_box.toggled.connect(self._sync_early_stopping_controls)
        self.rare_patch_oversampling_check_box.toggled.connect(self._sync_rare_patch_oversampling_controls)
        self.recognition_binarize_output_check_box.toggled.connect(self._sync_recognition_output_controls)
        self.recognition_use_auto_threshold_check_box.toggled.connect(self._sync_recognition_output_controls)
        self.recognition_postprocess_check_box.toggled.connect(self._sync_recognition_output_controls)
        self.confidence_save_mode_combo.currentIndexChanged.connect(
            self._sync_inference_uncertainty_from_legacy
        )
        self._sync_warmup_controls(self.warmup_check_box.isChecked())
        self._sync_scheduler_controls()
        self._sync_loss_controls()
        self._sync_hard_mining_controls(self.hard_mining_check_box.isChecked())
        self._sync_early_stopping_controls(self.early_stopping_check_box.isChecked())
        self._sync_rare_patch_oversampling_controls(self.rare_patch_oversampling_check_box.isChecked())
        self._sync_recognition_output_controls()
