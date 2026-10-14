"""apply_settings_panel_texts section 4."""
from __future__ import annotations
from typing import Any
from neuralimage.configuration import (
    SEM_UI_SECTIONS,
    sem_ui_section_help,
    sem_ui_section_label,
)
from neuralimage.lib.ui_texts import get_ui_language
from neuralimage.view.settings_panel_i18n_parts.module_state import (
    TEXT_FIELD_LOG_UPDATE_FREQUENCY,
    TEXT_FIELD_RECOGNITION_JPEG_QUALITY,
)
from neuralimage.view.settings_panel_i18n_parts.apply_description_tooltips import (
    _copy_text_dict,
    _read_text_from_mappings,
    apply_description_tooltips,
)

def _apply_settings_panel_texts_section_4(panel, t):
    if panel.pcb_topology_family_combo.count() >= 3:
        panel.pcb_topology_family_combo.setItemText(0, str(t.get('pcb_topology_family_parallel', 'Parallel traces')))
        panel.pcb_topology_family_combo.setItemText(1, str(t.get('pcb_topology_family_independent', 'Independent traces')))
        panel.pcb_topology_family_combo.setItemText(2, str(t.get('pcb_topology_family_fanout', 'Fanout tree')))
    if panel.ic_topology_family_combo.count() >= 3:
        panel.ic_topology_family_combo.setItemText(0, str(t.get('ic_topology_family_channels', 'Routing channels')))
        panel.ic_topology_family_combo.setItemText(1, str(t.get('ic_topology_family_cell_array', 'Cell array')))
        panel.ic_topology_family_combo.setItemText(2, str(t.get('ic_topology_family_tree', 'Tree routing')))
    panel.synthetic_trace_count_range_widget.setToolTip(str(t.get('synthetic_trace_count_tip', '')))
    panel.synthetic_segment_count_range_widget.setToolTip(str(t.get('synthetic_segment_count_tip', '')))
    panel.synthetic_trace_half_width_range_widget.setToolTip(str(t.get('synthetic_trace_half_width_tip', '')))
    panel.synthetic_background_noise_sigma_range_widget.setToolTip(
        str(t.get('synthetic_background_noise_sigma_tip', ''))
    )
    panel.synthetic_trace_noise_sigma_range_widget.setToolTip(str(t.get('synthetic_trace_noise_sigma_tip', '')))
    panel.cutout_probability_spinbox.setToolTip(str(t.get('cutout_probability_tip', '')))
    panel.cutout_holes_spinbox.setToolTip(str(t.get('cutout_holes_tip', '')))
    panel.cutout_size_ratio_spinbox.setToolTip(str(t.get('cutout_size_ratio_tip', '')))
    panel.random_artifacts_probability_spinbox.setToolTip(str(t.get('random_artifacts_probability_tip', '')))
    panel.random_artifacts_count_spinbox.setToolTip(str(t.get('random_artifacts_count_tip', '')))
    panel.random_artifacts_size_ratio_spinbox.setToolTip(str(t.get('random_artifacts_size_ratio_tip', '')))
    panel.mixup_probability_spinbox.setToolTip(str(t.get('mixup_probability_tip', '')))
    panel.mixup_alpha_spinbox.setToolTip(str(t.get('mixup_alpha_tip', '')))
    panel.pcb_defects_probability_spinbox.setToolTip(str(t.get('pcb_defects_probability_tip', '')))
    panel.pcb_defects_min_count_spinbox.setToolTip(str(t.get('pcb_defects_min_count_tip', '')))
    panel.pcb_defects_max_count_spinbox.setToolTip(str(t.get('pcb_defects_max_count_tip', '')))
    panel.synthetic_defect_generator_groupbox.setToolTip(
        str(
            t.get(
                'synthetic_defect_generator_group_tip',
                'Generate additional fully synthetic topology samples and image-only synthetic defects for training.',
            )
        )
    )
    for defect_name, text_key in (
        ('break', 'pcb_break_severity_tip'),
        ('short', 'pcb_short_severity_tip'),
        ('missing_copper', 'pcb_missing_copper_severity_tip'),
        ('excess_copper', 'pcb_excess_copper_severity_tip'),
        ('pinhole', 'pcb_pinhole_severity_tip'),
        ('spurious_copper', 'pcb_spurious_copper_severity_tip'),
        ('via', 'pcb_via_severity_tip'),
        ('misalignment', 'pcb_misalignment_severity_tip'),
    ):
        panel.pcb_defect_type_spinboxes[defect_name].setToolTip(str(t.get(text_key, '')))
    for defect_name, text_key in (
        ('line_break', 'ic_line_break_severity_tip'),
        ('bridge', 'ic_bridge_severity_tip'),
        ('necking', 'ic_necking_severity_tip'),
        ('missing_metal', 'ic_missing_metal_severity_tip'),
        ('spur', 'ic_spur_severity_tip'),
        ('pinhole', 'ic_pinhole_severity_tip'),
        ('via_open', 'ic_via_open_severity_tip'),
        ('line_shift', 'ic_line_shift_severity_tip'),
    ):
        panel.ic_defect_type_spinboxes[defect_name].setToolTip(str(t.get(text_key, '')))
    panel.shift_spinbox.setToolTip(str(t.get('shift_tip', '')))
    panel.validation_check_box.setText(str(t.get('validation', 'Use validation during training')))
    panel.validation_check_box.setToolTip(str(t.get('validation_tip', '')))
    panel.save_validation_binary_images_check_box.setText(
        str(t.get('save_validation_binary_images', 'Save binary validation predictions after each epoch'))
    )
    panel.save_validation_binary_images_check_box.setToolTip(
        str(
            t.get(
                'save_validation_binary_images_tip',
                'Save binary predictions for validation samples after each epoch into the run artifact folder.',
            )
        )
    )
    panel.validation_mode_combo.setToolTip(
        str(
            t.get(
                'validation_source_tip',
                'Choose between validation split from the training dataset or a separate validation dataset.',
            )
        )
    )
    split_text = str(t.get('validation_source_split', 'Split from training dataset'))
    external_text = str(t.get('validation_source_external', 'Use external image/label folders'))
    panel.validation_mode_combo.setItemText(0, split_text)
    panel.validation_mode_combo.setItemText(1, external_text)
    panel.validation_image_path_label.setToolTip(
        str(
            t.get(
                'validation_image_path_tip',
                'Folder with validation images used when external validation mode is selected.',
            )
        )
    )
    panel.validation_label_path_label.setToolTip(
        str(
            t.get(
                'validation_label_path_tip',
                'Folder with validation labels used when external validation mode is selected.',
            )
        )
    )
    panel.general_groupbox.setTitle(str(t.get('general_group', 'Data and model')))
    panel.augmentation_groupbox.setTitle(str(t.get('augmentation_group', 'Augmentation and shift')))
    panel.shuffle_groupbox.setTitle(str(t.get('shuffle_group', 'Shuffle order')))
    panel.spatial_groupbox.setTitle(str(t.get('spatial_group', 'Spatial augmentations')))
    panel.photometric_groupbox.setTitle(str(t.get('photometric_group', 'Photometric augmentations')))
    if hasattr(panel, 'tech_augmentation_groupbox'):
        panel.tech_augmentation_groupbox.setTitle(
            str(t.get('tech_augmentation', 'Topology variation augmentation'))
        )
    panel.cutout_groupbox.setTitle(str(t.get('cutout', 'Cutout')))
    panel.random_artifacts_groupbox.setTitle(str(t.get('random_artifacts', 'Random artifacts')))
    panel.mixup_groupbox.setTitle(str(t.get('mixup', 'Mixup')))
    panel.synthetic_defect_generator_groupbox.setTitle(
        str(t.get('synthetic_defect_generator_group', 'Synthetic topology'))
    )
    panel.augmentation_preview_button.setText(
        str(t.get('augmentation_preview_button', 'Open augmentation preview'))
    )
    panel.augmentation_preview_button.setToolTip(
        str(
            t.get(
                'augmentation_preview_button_tip',
                'Open a preview window that applies the selected training augmentations to dataset samples.',
            )
        )
    )
    panel.validation_groupbox.setTitle(str(t.get('validation_group', 'Validation')))
    panel.prepare_samples_groupbox.setTitle(str(t.get('preprocess_group', 'Sample preprocessing')))
    panel.enable_crop_processing.setText(str(t.get('preprocess_crop_enable', 'Enable edge crop')))
    panel.enable_resize_processing.setText(str(t.get('preprocess_resize_enable', 'Enable compression resize')))
    panel.recursive_file_search_check_box.setText(str(t.get('recursive_file_search', 'Search in subfolders')))
    panel.nn_auxilary_settings_groupbox.setTitle(str(t.get('aux_group', 'Additional settings')))
    panel.optimizer_groupbox.setTitle(str(t.get('optimizer_group', 'Optimizer and batch')))
    panel.precision_loss_groupbox.setTitle(str(t.get('loss_precision_group', 'Loss function')))
    panel.optimizer_advanced_groupbox.setTitle(str(t.get('advanced_group', 'Advanced settings')))
    panel.loss_advanced_groupbox.setTitle(str(t.get('advanced_group', 'Advanced settings')))
    panel.rare_patch_groupbox.setTitle(str(t.get('rare_patch_group', 'Rare patch oversampling')))
    panel.model_variants_groupbox.setTitle(str(t.get('model_variants_group', 'Additional model families')))
    panel.recognition_groupbox.setTitle(str(t.get('recognition_group', 'Recognition')))
    panel.runtime_groupbox.setTitle(str(t.get('runtime_group', 'Runtime and filtering')))
    panel.warmup_groupbox.setTitle(str(t.get('warmup_group', 'Warmup')))
    panel.scheduler_groupbox.setTitle(str(t.get('scheduler_group', 'Scheduler')))
    panel.hard_mining_groupbox.setTitle(str(t.get('hard_mining_group', 'Hard mining')))
    if hasattr(panel, 'sem_segmentation_groupbox'):
        language = get_ui_language()
        panel.sem_segmentation_groupbox.setTitle(
            str(t.get('sem_segmentation_group', 'SEM topology segmentation'))
        )
        if hasattr(panel, 'sem_segmentation_section_groupboxes'):
            for section_key, english_label in SEM_UI_SECTIONS:
                groupbox = panel.sem_segmentation_section_groupboxes.get(section_key)
                if groupbox is None:
                    continue
                editor = panel.sem_segmentation_section_editors.get(section_key)
                if editor is not None and hasattr(editor, 'set_language'):
                    editor.set_language(language)
                elif groupbox is not editor:
                    groupbox.setTitle(sem_ui_section_label(section_key, english_label, language))
                    groupbox.setToolTip(sem_ui_section_help(section_key, language))
    panel.early_stopping_groupbox.setTitle(str(t.get('early_stopping_group', 'Early stopping')))
    if hasattr(panel, 'settings_tabs'):
        language = get_ui_language()
        data_index = panel._page_indexes.get('data')
        training_index = panel._page_indexes.get('training')
        recognition_index = panel._page_indexes.get('recognition')
        if data_index is not None:
            panel.settings_tabs.setTabText(data_index, 'Данные' if language == 'ru' else 'Data')
        if training_index is not None:
            panel.settings_tabs.setTabText(
                training_index,
                'Модель и обучение' if language == 'ru' else 'Model and training',
            )
        if recognition_index is not None:
            panel.settings_tabs.setTabText(
                recognition_index,
                str(t.get('tab_recognition', 'Распознавание')),
            )
        panel.settings_search.setPlaceholderText(
            'Поиск настроек…' if language == 'ru' else 'Search settings…'
        )
        panel.settings_search.setToolTip(
            'Поиск по русским и английским названиям и описаниям. Ctrl+F.'
            if language == 'ru'
            else 'Search Russian and English labels and descriptions. Ctrl+F.'
        )
        panel.expert_mode_check_box.setText('Экспертный режим' if language == 'ru' else 'Expert mode')
        panel.expert_mode_check_box.setToolTip(
            'Показывает редкие и ресурсоёмкие параметры, не изменяя их значения.'
            if language == 'ru'
            else 'Shows uncommon and resource-intensive settings without changing their values.'
        )
        card_titles = {
            'data_source': ('Датасет и патчи', 'Dataset and patches'),
            'preprocessing': ('Нормализация SEM', 'SEM normalization'),
            'augmentation': ('Аугментация обучения', 'Training augmentation'),
            'sampling': ('Выбор патчей', 'Patch sampling'),
            'validation_data': ('Данные валидации', 'Validation data'),
            'architecture': ('Архитектура', 'Architecture'),
            'training': ('Процесс обучения', 'Training process'),
            'supervision': ('Топологический надзор', 'Topology supervision'),
            'losses': ('Функции потерь', 'Loss functions'),
            'schedule': ('Расписание learning rate', 'Learning-rate schedule'),
            'confidence_training': ('Обучение уверенности', 'Confidence-head training'),
            'validation_metrics': ('Метрики и эксперимент', 'Validation metrics and experiment'),
            'runtime': ('Ресурсы и GPU', 'Runtime and GPU'),
            'recognition': ('Предсказание маски', 'Mask prediction'),
            'inference_uncertainty': ('Неопределённость прогноза', 'Prediction uncertainty'),
            'active_learning': ('Экспорт NeedsAnnotation', 'NeedsAnnotation export'),
        }
        for key, (ru_title, en_title) in card_titles.items():
            card = panel._settings_cards.get(key)
            if card is not None:
                card.set_title(ru_title if language == 'ru' else en_title)
                card.reset_button.setText('Сбросить' if language == 'ru' else 'Reset')
                card.reset_button.setToolTip(
                    'Вернуть значения этой карточки по умолчанию.'
                    if language == 'ru'
                    else 'Restore this card to its default values.'
                )
    panel.optimizer_type.setToolTip(str(t.get('optimizer_tip', '')))
    panel.learning_rate_spinbox.setToolTip(str(t.get('lr_tip', '')))
    panel.weight_decay_spinbox.setToolTip(str(t.get('wd_tip', '')))
    panel.train_batch_spinbox.setToolTip(str(t.get('train_batch_tip', t.get('batch_tip', ''))))
    panel.dataloader_num_workers_spinbox.setToolTip(
        str(
            t.get(
                'dataloader_num_workers_tip',
                'Number of worker processes used by the training DataLoader. Use -1 for automatic selection.',
            )
        )
    )
    panel.dataloader_num_workers_spinbox.setSpecialValueText(str(t.get('auto_value', 'auto')))
    panel.scheduler_type_combo.setToolTip(
        str(
            t.get(
                'scheduler_tip',
                'Learning-rate schedule applied after optimizer updates.',
            )
        )
    )
    panel.scheduler_type_combo.setItemText(0, str(t.get('scheduler_off', 'Off')))
    panel.scheduler_type_combo.setItemText(1, str(t.get('scheduler_reduce_on_plateau', 'ReduceLROnPlateau')))
    panel.scheduler_type_combo.setItemText(2, str(t.get('scheduler_cosine_annealing', 'CosineAnnealingLR')))
    panel.scheduler_type_combo.setItemText(3, str(t.get('scheduler_one_cycle', 'OneCycleLR')))
    panel.scheduler_type_combo.setItemText(4, str(t.get('scheduler_step_lr', 'StepLR')))
    panel.scheduler_plateau_factor_spinbox.setToolTip(str(t.get('scheduler_plateau_factor_tip', '')))
    panel.scheduler_plateau_patience_spinbox.setToolTip(str(t.get('scheduler_plateau_patience_tip', '')))
    panel.scheduler_plateau_threshold_spinbox.setToolTip(str(t.get('scheduler_plateau_threshold_tip', '')))
    panel.scheduler_plateau_min_lr_spinbox.setToolTip(str(t.get('scheduler_plateau_min_lr_tip', '')))
    panel.scheduler_plateau_cooldown_spinbox.setToolTip(str(t.get('scheduler_plateau_cooldown_tip', '')))
    panel.scheduler_cosine_t_max_spinbox.setToolTip(str(t.get('scheduler_cosine_t_max_tip', '')))
    panel.scheduler_cosine_eta_min_spinbox.setToolTip(str(t.get('scheduler_cosine_eta_min_tip', '')))
    panel.scheduler_one_cycle_max_lr_spinbox.setToolTip(str(t.get('scheduler_one_cycle_max_lr_tip', '')))
    panel.scheduler_one_cycle_pct_start_spinbox.setToolTip(str(t.get('scheduler_one_cycle_pct_start_tip', '')))
    panel.scheduler_one_cycle_anneal_strategy_combo.setToolTip(
        str(t.get('scheduler_one_cycle_anneal_strategy_tip', ''))
    )
    panel.scheduler_one_cycle_anneal_strategy_combo.setItemText(
        0,
        str(t.get('scheduler_one_cycle_anneal_cos', 'Cosine')),
    )
    panel.scheduler_one_cycle_anneal_strategy_combo.setItemText(
        1,
        str(t.get('scheduler_one_cycle_anneal_linear', 'Linear')),
    )
    panel.scheduler_one_cycle_div_factor_spinbox.setToolTip(str(t.get('scheduler_one_cycle_div_factor_tip', '')))
    panel.scheduler_one_cycle_final_div_factor_spinbox.setToolTip(
        str(t.get('scheduler_one_cycle_final_div_factor_tip', ''))
    )
    panel.scheduler_one_cycle_three_phase_check_box.setToolTip(
        str(t.get('scheduler_one_cycle_three_phase_tip', ''))
    )
    panel.scheduler_step_lr_step_size_spinbox.setToolTip(str(t.get('scheduler_step_lr_step_size_tip', '')))
    panel.scheduler_step_lr_gamma_spinbox.setToolTip(str(t.get('scheduler_step_lr_gamma_tip', '')))
    panel.recognition_batch_spinbox.setToolTip(str(t.get('recognition_batch_tip', t.get('batch_tip', ''))))
    panel.overlap_spinbox.setToolTip(str(t.get('overlap_tip', '')))
    panel.recognition_jpeg_quality_spinbox.setToolTip(
        str(t.get('recognition_jpeg_quality_tip', 'JPEG quality for recognition output (1..100).'))
    )
    panel.recognition_multiprocessing_check_box.setText(
        str(t.get('recognition_multiprocessing_enable', 'Use multiprocessing recognition'))
    )
    panel.recognition_multiprocessing_check_box.setToolTip(
        str(
            t.get(
                'recognition_multiprocessing_tip',
                'Speeds up recognition via separate cut/predict/sew processes and automatically falls back to single-thread mode when unsuitable.',
            )
        )
    )
    panel.recognition_binarize_output_check_box.setText(
        str(t.get('recognition_binarize_output', 'Binarize recognition output'))
    )
    panel.recognition_binarize_output_check_box.setToolTip(
        str(t.get('recognition_binarize_output_tip', 'Convert probabilities into a binary mask before saving.'))
    )
    panel.recognition_use_auto_threshold_check_box.setText(
        str(t.get('recognition_use_auto_threshold', 'Use recommended threshold from model'))
    )
    panel.recognition_use_auto_threshold_check_box.setToolTip(
        str(
            t.get(
                'recognition_use_auto_threshold_tip',
                'Use the threshold selected on validation and saved inside the model artifact.',
            )
        )
    )
    panel.recognition_threshold_spinbox.setToolTip(
        str(
            t.get(
                'recognition_threshold_tip',
                'Probability threshold used to convert recognition output into a binary mask.',
            )
        )
    )
    panel.recognition_tta_check_box.setText(
        str(t.get('recognition_tta', 'Use TTA for recognition'))
    )
