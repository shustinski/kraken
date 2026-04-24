from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from application.dto import (
    MainWindowState,
    SettingsState,
    build_main_window_mode_state_entry,
    build_main_window_mode_state_entry_from_state,
    clone_main_window_state,
    normalize_main_window_mode_state,
    resolve_main_window_mode_state_entry,
)
from lib.data_interfaces import (
    SampleCutMode,
    WorkMode,
    normalize_multi_gpu_mode,
    normalize_validation_source,
    normalize_work_mode,
)
from lib.loss_config import dominant_loss_function, resolve_loss_term_weights


def load_initial_state(presenter) -> None:
    load_main_window_settings(presenter)
    load_settings_panel_settings(presenter)

    presenter.view.restore_from_dataclass(presenter.main_window_state)
    apply_settings_to_panel(presenter)
    presenter.view.set_batch_preview_enabled(presenter.settings_state.show_batch_preview)
    presenter.settings_panel.set_model(presenter.settings_state.model)
    restore_work_mode_ui(presenter)
    presenter.view.apply_ui_mode(getattr(presenter.main_window_state, 'ui_mode', 'simple'))
    presenter.view.set_simple_workflow_profile(None)

    presenter.settings_panel.connect_internal_signals()
    presenter.view.connect_internal_signals()

    presenter._refresh_queue_view()
    presenter._validate_start_button()


def load_main_window_settings(presenter) -> None:
    presenter.main_window_state = presenter._state_store.load_main_window_state()
    presenter.main_window_state.mode_state = normalize_main_window_mode_state(
        getattr(presenter.main_window_state, 'mode_state', {})
    )
    current_mode = normalize_work_mode(getattr(presenter.main_window_state, 'work_mode', ''))
    if current_mode:
        mode_state = dict(presenter.main_window_state.mode_state)
        if current_mode not in mode_state:
            mode_state[current_mode] = build_main_window_mode_state_entry_from_state(
                presenter.main_window_state
            )
            presenter.main_window_state.mode_state = mode_state
    sample_folder = str(getattr(presenter.main_window_state, 'sample_folder', '') or '').strip()
    presenter.sample_calculator.set_path(Path(sample_folder) if sample_folder else None)


def load_settings_panel_settings(presenter) -> None:
    presenter.settings_state = presenter._state_store.load_settings_state()


def set_initial_sample_count_state(presenter) -> None:
    sample_folder = str(getattr(presenter.main_window_state, 'sample_folder', '') or '').strip()
    if sample_folder and Path(sample_folder).is_dir():
        presenter.settings_panel.set_samples_count_loading()
        if hasattr(presenter.view, 'set_samples_count_loading'):
            presenter.view.set_samples_count_loading()
        return
    presenter.settings_panel.set_samples_count(0)
    if hasattr(presenter.view, 'set_samples_count'):
        presenter.view.set_samples_count(0)


def current_view_main_window_mode_entry(presenter) -> dict[str, object]:
    view = presenter.view
    epochs_widget = getattr(getattr(presenter, 'settings_panel', None), 'epochs_spinbox', None)
    epochs_value = epochs_widget.value() if epochs_widget is not None else view.le_epochs.value()
    return build_main_window_mode_state_entry(
        source_folder=view.lbl_source.text(),
        result_folder=view.lbl_result.text(),
        model_path=view.model_path.text(),
        label_folder=view.label_path.text(),
        sample_folder=view.sample_path.text(),
        epochs=epochs_value,
    )


def store_view_state_for_mode(presenter, mode: str) -> None:
    normalized_mode = normalize_work_mode(mode)
    if not normalized_mode:
        return
    mode_state = normalize_main_window_mode_state(getattr(presenter.main_window_state, 'mode_state', {}))
    mode_state[normalized_mode] = current_view_main_window_mode_entry(presenter)
    presenter.main_window_state.mode_state = mode_state


def restore_mode_state_to_view(presenter, mode: str) -> None:
    entry = resolve_main_window_mode_state_entry(presenter.main_window_state, mode)
    presenter.view.set_source_path(str(entry.get('source_folder', '')))
    presenter.view.set_result_path(str(entry.get('result_folder', '')))
    presenter.view.set_label_path(str(entry.get('label_folder', '')))
    presenter.view.set_jpg_path(str(entry.get('sample_folder', '')))
    presenter.view.model_path.setText(str(entry.get('model_path', '')))
    try:
        epochs = int(entry.get('epochs', MainWindowState().epochs))
    except (TypeError, ValueError):
        epochs = MainWindowState().epochs
    presenter.view.le_epochs.setValue(epochs)
    if hasattr(presenter, 'settings_panel') and hasattr(presenter.settings_panel, 'epochs_spinbox'):
        presenter.settings_panel.epochs_spinbox.setValue(epochs)
    presenter.main_window_state.source_folder = str(entry.get('source_folder', ''))
    presenter.main_window_state.result_folder = str(entry.get('result_folder', ''))
    presenter.main_window_state.label_folder = str(entry.get('label_folder', ''))
    presenter.main_window_state.sample_folder = str(entry.get('sample_folder', ''))
    presenter.main_window_state.model_path = str(entry.get('model_path', ''))
    presenter.main_window_state.epochs = epochs
    sample_folder = str(entry.get('sample_folder', '') or '').strip()
    presenter.sample_calculator.set_path(Path(sample_folder) if sample_folder else None)


def update_main_window_state(presenter) -> None:
    view = presenter.view
    work_mode = update_work_mode(presenter)
    mode_state = normalize_main_window_mode_state(getattr(presenter.main_window_state, 'mode_state', {}))
    if work_mode:
        mode_state[work_mode] = current_view_main_window_mode_entry(presenter)
    presenter.main_window_state.work_mode = work_mode
    presenter.main_window_state.source_folder = view.lbl_source.text()
    presenter.main_window_state.result_folder = view.lbl_result.text()
    presenter.main_window_state.label_folder = view.label_path.text()
    presenter.main_window_state.sample_folder = view.sample_path.text()
    presenter.main_window_state.model_path = view.model_path.text()
    epochs_widget = getattr(getattr(presenter, 'settings_panel', None), 'epochs_spinbox', None)
    presenter.main_window_state.epochs = epochs_widget.value() if epochs_widget is not None else view.le_epochs.value()
    presenter.main_window_state.ui_mode = presenter.view.current_ui_mode()
    presenter.main_window_state.mode_state = mode_state


def restore_work_mode_ui(presenter) -> None:
    mode = normalize_work_mode(getattr(presenter.main_window_state, 'work_mode', ''))
    view = presenter.view
    mode_to_button = {
        WorkMode.train_and_recognition.value: view.rb_train_and_recognition,
        WorkMode.train_only.value: view.rb_train_only,
        WorkMode.further_training.value: view.rb_further_train_model,
        WorkMode.recognition_only.value: view.rb_recognition,
    }
    if mode not in mode_to_button:
        mode = WorkMode.train_and_recognition.value
        presenter.main_window_state.work_mode = mode
    mode_to_button[mode].setChecked(True)
    presenter._on_sample_type_changed(mode)


def apply_settings_to_panel(presenter) -> None:
    panel = presenter.settings_panel
    state = presenter.settings_state

    panel.horizontal_rotation.setChecked(state.horizontal_rotation)
    panel.vertical_rotation.setChecked(state.vertical_rotation)
    panel.flip_x.setChecked(bool(getattr(state, 'flip_x', False)))
    panel.flip_y.setChecked(bool(getattr(state, 'flip_y', False)))
    panel.additional_augmentation_check_box.setChecked(state.additional_augmentation)
    panel.augmentation_brightness_spinbox.setValue(state.augmentation_brightness_strength)
    panel.augmentation_contrast_spinbox.setValue(state.augmentation_contrast_strength)
    panel.augmentation_gamma_spinbox.setValue(float(getattr(state, 'augmentation_gamma_strength', 0.15)))
    panel.augmentation_noise_probability_spinbox.setValue(state.augmentation_noise_probability)
    panel.augmentation_noise_sigma_spinbox.setValue(state.augmentation_noise_sigma)
    panel.augmentation_blur_probability_spinbox.setValue(
        float(getattr(state, 'augmentation_blur_probability', 0.25))
    )
    panel.augmentation_blur_radius_spinbox.setValue(float(getattr(state, 'augmentation_blur_radius', 1.0)))
    if hasattr(panel, '_sync_augmentation_controls'):
        panel._sync_augmentation_controls(state.additional_augmentation)

    train_patch_size = tuple(getattr(state, 'train_patch_size', None) or state.sample_size)
    recognition_patch_size = tuple(getattr(state, 'recognition_patch_size', None) or state.sample_size)
    panel.shift_spinbox.setValue(state.step)
    panel.train_patch_x_size.setValue(train_patch_size[0])
    panel.train_patch_y_size.setValue(train_patch_size[1])
    panel.recognition_patch_x_size.setValue(recognition_patch_size[0])
    panel.recognition_patch_y_size.setValue(recognition_patch_size[1])
    main_window_state = getattr(presenter, '__dict__', {}).get('main_window_state')
    panel.epochs_spinbox.setValue(int(getattr(main_window_state, 'epochs', MainWindowState().epochs)))

    panel.set_model(state.model)
    if hasattr(panel, 'set_color_mode_value'):
        panel.set_color_mode_value(state.color_mode)
    else:
        panel.color_type.setCurrentText(state.color_mode)
    panel.shuffle_frames_check_box.setChecked(bool(getattr(state, 'shuffle', True)))
    panel.shuffle_patches_in_frame_check_box.setChecked(
        bool(getattr(state, 'shuffle_patches_in_frame', getattr(state, 'shuffle', True)))
    )
    panel.random_crop_check_box.setChecked(bool(getattr(state, 'random_crop', False)))
    panel.crops_per_image_spinbox.setValue(int(getattr(state, 'crops_per_image', 64)))
    panel.scale_augmentation_check_box.setChecked(bool(getattr(state, 'scale_augmentation', False)))
    panel.scale_augmentation_strength_spinbox.setValue(
        float(getattr(state, 'scale_augmentation_strength', 0.2))
    )
    if hasattr(panel, 'set_synthetic_defect_generator_config'):
        panel.set_synthetic_defect_generator_config(getattr(state, 'synthetic_defect_generator', {}))
    panel.cutout_check_box.setChecked(bool(getattr(state, 'cutout_enabled', False)))
    panel.cutout_probability_spinbox.setValue(float(getattr(state, 'cutout_probability', 1.0)))
    panel.cutout_holes_spinbox.setValue(int(getattr(state, 'cutout_holes', 1)))
    panel.cutout_size_ratio_spinbox.setValue(float(getattr(state, 'cutout_size_ratio', 0.25)))
    panel.random_artifacts_check_box.setChecked(bool(getattr(state, 'random_artifacts_enabled', False)))
    panel.random_artifacts_probability_spinbox.setValue(
        float(getattr(state, 'random_artifacts_probability', 1.0))
    )
    panel.random_artifacts_count_spinbox.setValue(int(getattr(state, 'random_artifacts_count', 1)))
    panel.random_artifacts_size_ratio_spinbox.setValue(
        float(getattr(state, 'random_artifacts_size_ratio', 0.25))
    )
    for artifact_name, checkbox in getattr(panel, 'random_artifact_type_checkboxes', {}).items():
        checkbox.setChecked(bool(getattr(state, f'random_artifacts_{artifact_name}_enabled', True)))
    panel.mixup_check_box.setChecked(bool(getattr(state, 'mixup_enabled', False)))
    panel.mixup_probability_spinbox.setValue(float(getattr(state, 'mixup_probability', 1.0)))
    panel.mixup_alpha_spinbox.setValue(float(getattr(state, 'mixup_alpha', 0.2)))

    panel.validation_check_box.setChecked(state.use_validation)
    panel.validation_spinbox.setValue(state.validation_percent)
    panel.set_validation_source_value(
        normalize_validation_source(getattr(state, 'validation_source', 'split'))
    )
    panel.set_validation_image_path(str(getattr(state, 'validation_image_folder', '')))
    panel.set_validation_label_path(str(getattr(state, 'validation_label_folder', '')))
    panel.save_validation_binary_images_check_box.setChecked(
        bool(getattr(state, 'save_validation_binary_images', False))
    )

    panel.restore_cut_mode(state.sample_cut_mode)
    if hasattr(panel, '_sync_augmentation_controls'):
        panel._sync_augmentation_controls(state.additional_augmentation)
    if hasattr(panel, '_sync_training_augmentation_controls'):
        panel._sync_training_augmentation_controls()

    train_batch_size = int(getattr(state, 'train_batch_size', None) or state.batch_size)
    recognition_batch_size = int(getattr(state, 'recognition_batch_size', None) or state.batch_size)
    panel.train_batch_spinbox.setValue(train_batch_size)
    panel.dataloader_num_workers_spinbox.setValue(int(getattr(state, 'dataloader_num_workers', -1)))
    panel.recognition_batch_spinbox.setValue(recognition_batch_size)
    panel.overlap_spinbox.setValue(state.overlap)
    panel.recognition_jpeg_quality_spinbox.setValue(int(getattr(state, 'recognition_jpeg_quality', 95)))
    panel.recognition_multiprocessing_check_box.setChecked(
        bool(getattr(state, 'recognition_multiprocessing_enabled', True))
    )
    panel.recognition_binarize_output_check_box.setChecked(
        bool(getattr(state, 'recognition_binarize_output', True))
    )
    panel.recognition_use_auto_threshold_check_box.setChecked(
        bool(getattr(state, 'recognition_use_auto_threshold', True))
    )
    panel.recognition_threshold_spinbox.setValue(float(getattr(state, 'recognition_threshold', 0.5)))
    panel.recognition_tta_check_box.setChecked(bool(getattr(state, 'recognition_tta_enabled', False)))
    panel.recognition_postprocess_check_box.setChecked(bool(getattr(state, 'recognition_postprocess', False)))
    panel.recognition_postprocess_kernel_size_spinbox.setValue(
        int(getattr(state, 'recognition_postprocess_kernel_size', 3))
    )
    if hasattr(panel, 'set_confidence_output_mode'):
        panel.set_confidence_output_mode(
            str(getattr(state, 'confidence_save_mode', 'off')),
            bool(getattr(state, 'confidence_tta_enabled', False)),
        )
    elif hasattr(panel, 'set_confidence_save_mode_value'):
        panel.set_confidence_save_mode_value(str(getattr(state, 'confidence_save_mode', 'off')))
    panel.log_update_frequency_spinbox.setValue(state.log_update_frequency)
    panel.optimizer_type.setCurrentText(state.optimizer_name)
    panel.mixed_precision_type.setCurrentText(state.mixed_precision)
    panel.deep_supervision_check_box.setChecked(bool(getattr(state, 'deep_supervision', False)))
    if hasattr(panel, 'set_loss_term_weights'):
        panel.set_loss_term_weights(
            resolve_loss_term_weights(
                getattr(state, 'loss_term_weights', None),
                fallback_loss_function=state.loss_function,
            )
        )
    if hasattr(panel, '_sync_loss_controls'):
        panel._sync_loss_controls()
    panel.learning_rate_spinbox.setValue(state.learning_rate)
    panel.weight_decay_spinbox.setValue(state.weight_decay)
    panel.early_stopping_check_box.setChecked(state.early_stopping_enabled)
    panel.early_stopping_patience_spinbox.setValue(state.early_stopping_patience)
    panel.early_stopping_min_delta_spinbox.setValue(state.early_stopping_min_delta)
    panel.restore_best_weights_check_box.setChecked(state.early_stopping_restore_best_weights)
    panel.warmup_check_box.setChecked(state.warmup_enabled)
    panel.warmup_epochs_spinbox.setValue(state.warmup_epochs)
    panel.warmup_start_factor_spinbox.setValue(state.warmup_start_factor)
    panel.set_scheduler_value(str(getattr(state, 'scheduler_name', 'off')))
    panel.scheduler_plateau_factor_spinbox.setValue(float(getattr(state, 'scheduler_plateau_factor', 0.5)))
    panel.scheduler_plateau_patience_spinbox.setValue(int(getattr(state, 'scheduler_plateau_patience', 3)))
    panel.scheduler_plateau_threshold_spinbox.setValue(
        float(getattr(state, 'scheduler_plateau_threshold', 1e-4))
    )
    panel.scheduler_plateau_min_lr_spinbox.setValue(float(getattr(state, 'scheduler_plateau_min_lr', 1e-6)))
    panel.scheduler_plateau_cooldown_spinbox.setValue(int(getattr(state, 'scheduler_plateau_cooldown', 0)))
    panel.scheduler_cosine_t_max_spinbox.setValue(int(getattr(state, 'scheduler_cosine_t_max', 10)))
    panel.scheduler_cosine_eta_min_spinbox.setValue(float(getattr(state, 'scheduler_cosine_eta_min', 1e-6)))
    panel.scheduler_one_cycle_max_lr_spinbox.setValue(
        float(getattr(state, 'scheduler_one_cycle_max_lr', 1e-3))
    )
    panel.scheduler_one_cycle_pct_start_spinbox.setValue(
        float(getattr(state, 'scheduler_one_cycle_pct_start', 0.3))
    )
    panel.set_scheduler_one_cycle_anneal_strategy_value(
        str(getattr(state, 'scheduler_one_cycle_anneal_strategy', 'cos'))
    )
    panel.scheduler_one_cycle_div_factor_spinbox.setValue(
        float(getattr(state, 'scheduler_one_cycle_div_factor', 25.0))
    )
    panel.scheduler_one_cycle_final_div_factor_spinbox.setValue(
        float(getattr(state, 'scheduler_one_cycle_final_div_factor', 10000.0))
    )
    panel.scheduler_one_cycle_three_phase_check_box.setChecked(
        bool(getattr(state, 'scheduler_one_cycle_three_phase', False))
    )
    panel.scheduler_step_lr_step_size_spinbox.setValue(
        int(getattr(state, 'scheduler_step_lr_step_size', 10))
    )
    panel.scheduler_step_lr_gamma_spinbox.setValue(float(getattr(state, 'scheduler_step_lr_gamma', 0.1)))
    panel.hard_mining_check_box.setChecked(state.hard_mining_enabled)
    panel.hard_mining_strength_spinbox.setValue(state.hard_mining_strength)
    panel.hard_mining_ema_alpha_spinbox.setValue(state.hard_mining_ema_alpha)
    panel.hard_pixel_mining_check_box.setChecked(bool(getattr(state, 'hard_pixel_mining_enabled', False)))
    panel.hard_pixel_mining_ratio_spinbox.setValue(float(getattr(state, 'hard_pixel_mining_ratio', 0.25)))
    panel.skip_uniform_labels_check_box.setChecked(state.skip_uniform_labels)
    panel.rare_patch_oversampling_check_box.setChecked(
        bool(getattr(state, 'rare_patch_oversampling_enabled', False))
    )
    panel.rare_patch_oversampling_factor_spinbox.setValue(
        int(getattr(state, 'rare_patch_oversampling_factor', 2))
    )
    multi_gpu_mode = normalize_multi_gpu_mode(
        getattr(state, 'multi_gpu_mode', ''),
        use_multi_gpu_fallback=bool(getattr(state, 'use_multi_gpu', False)),
    )
    panel.sync_patch_sizes_check_box.setChecked(bool(getattr(state, 'sync_patch_sizes', True)))
    view = presenter.__dict__.get('view')
    if view is not None and hasattr(view, 'set_recursive_file_search'):
        view.set_recursive_file_search(bool(getattr(state, 'recursive_file_search', False)))
    elif hasattr(panel, 'recursive_file_search_check_box'):
        panel.recursive_file_search_check_box.setChecked(bool(getattr(state, 'recursive_file_search', False)))
    panel.multi_gpu_mode_combo.setCurrentText(multi_gpu_mode)
    panel.torch_compile_check_box.setChecked(state.torch_compile_enabled)
    if view is not None and hasattr(view, 'set_batch_preview_enabled'):
        view.set_batch_preview_enabled(state.show_batch_preview)

    panel.enable_crop_processing.setChecked(state.crop_enabled)
    panel.enable_resize_processing.setChecked(state.resize_enabled)
    if hasattr(panel, '_sync_preprocess_controls'):
        panel._sync_preprocess_controls()
    if hasattr(panel, '_sync_patch_size_controls'):
        panel._sync_patch_size_controls()
    if hasattr(panel, '_sync_rare_patch_oversampling_controls'):
        panel._sync_rare_patch_oversampling_controls()
    if hasattr(panel, '_sync_recognition_output_controls'):
        panel._sync_recognition_output_controls()
    if hasattr(panel, 'sync_business_logic_controls'):
        main_state = presenter.__dict__.get('main_window_state')
        work_mode = getattr(main_state, 'work_mode', '')
        panel.sync_business_logic_controls(work_mode)

    panel.cut_corner_spinbox.setValue(state.edge_cut_size)
    panel.compression_factor_spinbox.setValue(max(1, int(getattr(state, 'compression_factor', 1))))


def update_work_mode(presenter) -> str:
    view = presenter.view
    if view.rb_train_and_recognition.isChecked():
        return WorkMode.train_and_recognition.value
    if view.rb_train_only.isChecked():
        return WorkMode.train_only.value
    if view.rb_further_train_model.isChecked():
        return WorkMode.further_training.value
    if view.rb_recognition.isChecked():
        return WorkMode.recognition_only.value
    return ''


def update_cut_mode(presenter) -> str:
    panel = presenter.settings_panel
    if panel.cut_dataset_type.isChecked():
        return SampleCutMode.disk.value
    if panel.no_cut_dataset_type.isChecked():
        return SampleCutMode.online.value
    return SampleCutMode.online.value


def update_settings_window_state(presenter) -> None:
    panel = presenter.settings_panel

    horizontal_rotation = panel.horizontal_rotation.isChecked()
    vertical_rotation = panel.vertical_rotation.isChecked()
    flip_x = panel.flip_x.isChecked()
    flip_y = panel.flip_y.isChecked()
    additional_augmentation = panel.additional_augmentation_check_box.isChecked()
    augmentation_brightness_strength = panel.augmentation_brightness_spinbox.value()
    augmentation_contrast_strength = panel.augmentation_contrast_spinbox.value()
    augmentation_gamma_strength = panel.augmentation_gamma_spinbox.value()
    augmentation_noise_probability = panel.augmentation_noise_probability_spinbox.value()
    augmentation_noise_sigma = panel.augmentation_noise_sigma_spinbox.value()
    augmentation_blur_probability = panel.augmentation_blur_probability_spinbox.value()
    augmentation_blur_radius = panel.augmentation_blur_radius_spinbox.value()
    step = panel.shift_spinbox.value()
    train_patch_size = (panel.train_patch_x_size.value(), panel.train_patch_y_size.value())
    recognition_patch_size = (
        panel.recognition_patch_x_size.value(),
        panel.recognition_patch_y_size.value(),
    )
    model = panel.get_selected_model()
    color_mode = (
        panel.get_color_mode_value() if hasattr(panel, 'get_color_mode_value') else panel.color_type.currentText()
    )
    shuffle_frames = panel.shuffle_frames_check_box.isChecked()
    shuffle_patches_in_frame = panel.shuffle_patches_in_frame_check_box.isChecked()
    random_crop = panel.random_crop_check_box.isChecked()
    crops_per_image = panel.crops_per_image_spinbox.value()
    scale_augmentation = panel.scale_augmentation_check_box.isChecked()
    scale_augmentation_strength = panel.scale_augmentation_strength_spinbox.value()
    synthetic_defect_generator = (
        panel.get_synthetic_defect_generator_config()
        if hasattr(panel, 'get_synthetic_defect_generator_config')
        else {}
    )
    cutout_enabled = panel.cutout_check_box.isChecked()
    cutout_probability = panel.cutout_probability_spinbox.value()
    cutout_holes = panel.cutout_holes_spinbox.value()
    cutout_size_ratio = panel.cutout_size_ratio_spinbox.value()
    random_artifacts_enabled = panel.random_artifacts_check_box.isChecked()
    random_artifacts_probability = panel.random_artifacts_probability_spinbox.value()
    random_artifacts_count = panel.random_artifacts_count_spinbox.value()
    random_artifacts_size_ratio = panel.random_artifacts_size_ratio_spinbox.value()
    random_artifact_type_enabled = {
        artifact_name: checkbox.isChecked()
        for artifact_name, checkbox in getattr(panel, 'random_artifact_type_checkboxes', {}).items()
    }
    mixup_enabled = panel.mixup_check_box.isChecked()
    mixup_probability = panel.mixup_probability_spinbox.value()
    mixup_alpha = panel.mixup_alpha_spinbox.value()
    validation = panel.validation_check_box.isChecked()
    validation_percent = panel.validation_spinbox.value()
    validation_source = normalize_validation_source(panel.get_validation_source_value())
    validation_image_folder = panel.validation_image_path()
    validation_label_folder = panel.validation_label_path()
    save_validation_binary_images = panel.save_validation_binary_images_check_box.isChecked()
    cut_mode = update_cut_mode(presenter)
    train_batch_size = panel.train_batch_spinbox.value()
    dataloader_num_workers = panel.dataloader_num_workers_spinbox.value()
    recognition_batch_size = panel.recognition_batch_spinbox.value()
    sync_patch_sizes = panel.sync_patch_sizes_check_box.isChecked()
    if sync_patch_sizes:
        recognition_patch_size = train_patch_size
    overlap = panel.overlap_spinbox.value()
    recognition_jpeg_quality = panel.recognition_jpeg_quality_spinbox.value()
    recognition_multiprocessing_enabled = panel.recognition_multiprocessing_check_box.isChecked()
    recognition_binarize_output = panel.recognition_binarize_output_check_box.isChecked()
    recognition_use_auto_threshold = panel.recognition_use_auto_threshold_check_box.isChecked()
    recognition_threshold = panel.recognition_threshold_spinbox.value()
    recognition_tta_enabled = panel.recognition_tta_check_box.isChecked()
    confidence_tta_enabled = (
        panel.is_confidence_tta_enabled()
        if hasattr(panel, 'is_confidence_tta_enabled')
        else False
    )
    recognition_postprocess = panel.recognition_postprocess_check_box.isChecked()
    recognition_postprocess_kernel_size = panel.recognition_postprocess_kernel_size_spinbox.value()
    confidence_save_mode = (
        panel.get_confidence_save_mode_value()
        if hasattr(panel, 'get_confidence_save_mode_value')
        else 'off'
    )
    log_update_frequency = panel.log_update_frequency_spinbox.value()
    optimizer_name = panel.optimizer_type.currentText()
    mixed_precision = panel.mixed_precision_type.currentText()
    deep_supervision = panel.deep_supervision_check_box.isChecked()
    current_state = getattr(presenter, 'settings_state', SettingsState())
    loss_term_weights = (
        panel.get_loss_term_weights()
        if hasattr(panel, 'get_loss_term_weights')
        else resolve_loss_term_weights({}, fallback_loss_function='bce')
    )
    loss_function = dominant_loss_function(
        loss_term_weights,
        fallback=getattr(current_state, 'loss_function', 'bce'),
    )
    dice_loss_weight = float(getattr(current_state, 'dice_loss_weight', 0.5))
    iou_loss_weight = float(getattr(current_state, 'iou_loss_weight', 0.5))
    learning_rate = panel.learning_rate_spinbox.value()
    weight_decay = panel.weight_decay_spinbox.value()
    early_stopping_enabled = panel.early_stopping_check_box.isChecked()
    early_stopping_patience = panel.early_stopping_patience_spinbox.value()
    early_stopping_min_delta = panel.early_stopping_min_delta_spinbox.value()
    early_stopping_restore_best_weights = panel.restore_best_weights_check_box.isChecked()
    warmup_enabled = panel.warmup_check_box.isChecked()
    warmup_epochs = panel.warmup_epochs_spinbox.value()
    warmup_start_factor = panel.warmup_start_factor_spinbox.value()
    scheduler_name = panel.get_scheduler_value()
    scheduler_plateau_factor = panel.scheduler_plateau_factor_spinbox.value()
    scheduler_plateau_patience = panel.scheduler_plateau_patience_spinbox.value()
    scheduler_plateau_threshold = panel.scheduler_plateau_threshold_spinbox.value()
    scheduler_plateau_min_lr = panel.scheduler_plateau_min_lr_spinbox.value()
    scheduler_plateau_cooldown = panel.scheduler_plateau_cooldown_spinbox.value()
    scheduler_cosine_t_max = panel.scheduler_cosine_t_max_spinbox.value()
    scheduler_cosine_eta_min = panel.scheduler_cosine_eta_min_spinbox.value()
    scheduler_one_cycle_max_lr = panel.scheduler_one_cycle_max_lr_spinbox.value()
    scheduler_one_cycle_pct_start = panel.scheduler_one_cycle_pct_start_spinbox.value()
    scheduler_one_cycle_anneal_strategy = panel.get_scheduler_one_cycle_anneal_strategy_value()
    scheduler_one_cycle_div_factor = panel.scheduler_one_cycle_div_factor_spinbox.value()
    scheduler_one_cycle_final_div_factor = panel.scheduler_one_cycle_final_div_factor_spinbox.value()
    scheduler_one_cycle_three_phase = panel.scheduler_one_cycle_three_phase_check_box.isChecked()
    scheduler_step_lr_step_size = panel.scheduler_step_lr_step_size_spinbox.value()
    scheduler_step_lr_gamma = panel.scheduler_step_lr_gamma_spinbox.value()
    hard_mining_enabled = panel.hard_mining_check_box.isChecked()
    hard_mining_strength = panel.hard_mining_strength_spinbox.value()
    hard_mining_ema_alpha = panel.hard_mining_ema_alpha_spinbox.value()
    hard_pixel_mining_enabled = panel.hard_pixel_mining_check_box.isChecked()
    hard_pixel_mining_ratio = panel.hard_pixel_mining_ratio_spinbox.value()
    skip_uniform_labels = panel.skip_uniform_labels_check_box.isChecked()
    rare_patch_oversampling_enabled = panel.rare_patch_oversampling_check_box.isChecked()
    rare_patch_oversampling_factor = panel.rare_patch_oversampling_factor_spinbox.value()
    multi_gpu_mode = normalize_multi_gpu_mode(panel.multi_gpu_mode_combo.currentText())
    use_multi_gpu = multi_gpu_mode != 'off'
    torch_compile_enabled = panel.torch_compile_check_box.isChecked()
    show_batch_preview = presenter.view.is_batch_preview_enabled()
    crop_enabled = panel.enable_crop_processing.isChecked()
    resize_enabled = panel.enable_resize_processing.isChecked()
    edge_cut_size = panel.cut_corner_spinbox.value()
    compression_factor = panel.compression_factor_spinbox.value()
    view = presenter.__dict__.get('view')
    if view is not None and hasattr(view, 'is_recursive_file_search_enabled'):
        recursive_file_search = view.is_recursive_file_search_enabled()
    else:
        recursive_file_search = panel.recursive_file_search_check_box.isChecked()

    presenter.settings_state = SettingsState(
        step=step,
        vertical_rotation=vertical_rotation,
        horizontal_rotation=horizontal_rotation,
        flip_x=flip_x,
        flip_y=flip_y,
        additional_augmentation=additional_augmentation,
        augmentation_brightness_strength=augmentation_brightness_strength,
        augmentation_contrast_strength=augmentation_contrast_strength,
        augmentation_gamma_strength=augmentation_gamma_strength,
        augmentation_noise_probability=augmentation_noise_probability,
        augmentation_noise_sigma=augmentation_noise_sigma,
        augmentation_blur_probability=augmentation_blur_probability,
        augmentation_blur_radius=augmentation_blur_radius,
        sample_size=train_patch_size,
        train_patch_size=train_patch_size,
        recognition_patch_size=recognition_patch_size,
        model=model,
        color_mode=color_mode,
        shuffle=shuffle_frames,
        shuffle_patches_in_frame=shuffle_patches_in_frame,
        random_crop=random_crop,
        crops_per_image=crops_per_image,
        scale_augmentation=scale_augmentation,
        scale_augmentation_strength=scale_augmentation_strength,
        synthetic_defect_generator=synthetic_defect_generator,
        cutout_enabled=cutout_enabled,
        cutout_probability=cutout_probability,
        cutout_holes=cutout_holes,
        cutout_size_ratio=cutout_size_ratio,
        random_artifacts_enabled=random_artifacts_enabled,
        random_artifacts_probability=random_artifacts_probability,
        random_artifacts_count=random_artifacts_count,
        random_artifacts_size_ratio=random_artifacts_size_ratio,
        random_artifacts_dust_enabled=bool(random_artifact_type_enabled.get('dust', True)),
        random_artifacts_resist_residue_enabled=bool(
            random_artifact_type_enabled.get('resist_residue', True)
        ),
        random_artifacts_etch_residue_enabled=bool(
            random_artifact_type_enabled.get('etch_residue', True)
        ),
        random_artifacts_particle_cluster_enabled=bool(
            random_artifact_type_enabled.get('particle_cluster', True)
        ),
        random_artifacts_flake_enabled=bool(random_artifact_type_enabled.get('flake', True)),
        mixup_enabled=mixup_enabled,
        mixup_probability=mixup_probability,
        mixup_alpha=mixup_alpha,
        use_validation=validation,
        validation_percent=validation_percent,
        validation_source=validation_source,
        validation_image_folder=validation_image_folder,
        validation_label_folder=validation_label_folder,
        save_validation_binary_images=save_validation_binary_images,
        sample_cut_mode=cut_mode,
        batch_size=train_batch_size,
        dataloader_num_workers=dataloader_num_workers,
        train_batch_size=train_batch_size,
        recognition_batch_size=recognition_batch_size,
        sync_patch_sizes=sync_patch_sizes,
        patch_batch_sync_mode='patch' if sync_patch_sizes else 'off',
        overlap=overlap,
        recognition_jpeg_quality=recognition_jpeg_quality,
        recognition_multiprocessing_enabled=recognition_multiprocessing_enabled,
        recognition_binarize_output=recognition_binarize_output,
        recognition_use_auto_threshold=recognition_use_auto_threshold,
        recognition_threshold=recognition_threshold,
        recognition_tta_enabled=recognition_tta_enabled,
        confidence_tta_enabled=confidence_tta_enabled,
        recognition_postprocess=recognition_postprocess,
        recognition_postprocess_kernel_size=recognition_postprocess_kernel_size,
        confidence_save_mode=confidence_save_mode,
        crop_enabled=crop_enabled,
        resize_enabled=resize_enabled,
        edge_cut_size=edge_cut_size,
        compression_factor=compression_factor,
        recursive_file_search=recursive_file_search,
        optimizer_name=optimizer_name,
        mixed_precision=mixed_precision,
        deep_supervision=deep_supervision,
        loss_function=loss_function,
        loss_term_weights=loss_term_weights,
        dice_loss_weight=dice_loss_weight,
        iou_loss_weight=iou_loss_weight,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        early_stopping_enabled=early_stopping_enabled,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        early_stopping_restore_best_weights=early_stopping_restore_best_weights,
        warmup_enabled=warmup_enabled,
        warmup_epochs=warmup_epochs,
        warmup_start_factor=warmup_start_factor,
        scheduler_name=scheduler_name,
        scheduler_plateau_factor=scheduler_plateau_factor,
        scheduler_plateau_patience=scheduler_plateau_patience,
        scheduler_plateau_threshold=scheduler_plateau_threshold,
        scheduler_plateau_min_lr=scheduler_plateau_min_lr,
        scheduler_plateau_cooldown=scheduler_plateau_cooldown,
        scheduler_cosine_t_max=scheduler_cosine_t_max,
        scheduler_cosine_eta_min=scheduler_cosine_eta_min,
        scheduler_one_cycle_max_lr=scheduler_one_cycle_max_lr,
        scheduler_one_cycle_pct_start=scheduler_one_cycle_pct_start,
        scheduler_one_cycle_anneal_strategy=scheduler_one_cycle_anneal_strategy,
        scheduler_one_cycle_div_factor=scheduler_one_cycle_div_factor,
        scheduler_one_cycle_final_div_factor=scheduler_one_cycle_final_div_factor,
        scheduler_one_cycle_three_phase=scheduler_one_cycle_three_phase,
        scheduler_step_lr_step_size=scheduler_step_lr_step_size,
        scheduler_step_lr_gamma=scheduler_step_lr_gamma,
        hard_mining_enabled=hard_mining_enabled,
        hard_mining_strength=hard_mining_strength,
        hard_mining_ema_alpha=hard_mining_ema_alpha,
        hard_pixel_mining_enabled=hard_pixel_mining_enabled,
        hard_pixel_mining_ratio=hard_pixel_mining_ratio,
        log_update_frequency=log_update_frequency,
        skip_uniform_labels=skip_uniform_labels,
        rare_patch_oversampling_enabled=rare_patch_oversampling_enabled,
        rare_patch_oversampling_factor=rare_patch_oversampling_factor,
        use_multi_gpu=use_multi_gpu,
        multi_gpu_mode=multi_gpu_mode,
        torch_compile_enabled=torch_compile_enabled,
        show_batch_preview=show_batch_preview,
    )


def reset_settings_to_defaults(presenter) -> None:
    presenter.settings_state = SettingsState()
    apply_settings_to_panel(presenter)
    update_settings_window_state(presenter)
    presenter._set_max_shift()
    presenter._calculate_expected_samples()
    presenter._validate_start_button()


def restore_task_state_to_ui(
    presenter,
    main_state: MainWindowState,
    settings_state: SettingsState,
    *,
    log_message: str = 'Параметры задачи восстановлены в интерфейсе.',
) -> None:
    presenter.main_window_state = clone_main_window_state(main_state)
    presenter.settings_state = replace(settings_state)
    sample_folder = str(getattr(presenter.main_window_state, 'sample_folder', '') or '').strip()
    presenter.sample_calculator.set_path(Path(sample_folder) if sample_folder else None)

    presenter.view.restore_from_dataclass(presenter.main_window_state)
    apply_settings_to_panel(presenter)
    restore_work_mode_ui(presenter)
    presenter.view.apply_ui_mode(getattr(presenter.main_window_state, 'ui_mode', 'simple'))
    presenter.view.set_simple_workflow_profile(None)
    if getattr(presenter.main_window_state, 'ui_mode', 'simple') != 'simple' and hasattr(
        presenter.view,
        'show_settings_dock',
    ):
        presenter.view.show_settings_dock()

    update_main_window_state(presenter)
    update_settings_window_state(presenter)
    presenter._set_max_shift()
    presenter._calculate_expected_samples()
    presenter._validate_start_button()
    presenter.message_bus.publish('logging', log_message)
