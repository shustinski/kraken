"""main_presenter: update settings window state."""
from __future__ import annotations
from neuralimage.application.dto import (
    SettingsState,
)
from neuralimage.lib.data_interfaces import (
    normalize_multi_gpu_mode,
    normalize_validation_source,
)
from neuralimage.lib.loss_config import dominant_loss_function, resolve_loss_term_weights
import neuralimage.presenter.state_sync as state_sync


class UpdateSettingsWindowStateMixin:
    def _update_settings_window_state(self):
        return state_sync.update_settings_window_state(self)
        s = self.settings_panel

        h = s.horizontal_rotation.isChecked()
        v = s.vertical_rotation.isChecked()
        flip_x = s.flip_x.isChecked()
        flip_y = s.flip_y.isChecked()
        additional_augmentation = s.additional_augmentation_check_box.isChecked()
        augmentation_multiplier = float(s.augmentation_multiplier_spinbox.value())
        augmentation_brightness_strength = s.augmentation_brightness_spinbox.value()
        augmentation_contrast_strength = s.augmentation_contrast_spinbox.value()
        augmentation_gamma_strength = s.augmentation_gamma_spinbox.value()
        augmentation_noise_probability = s.augmentation_noise_probability_spinbox.value()
        augmentation_noise_sigma = s.augmentation_noise_sigma_spinbox.value()
        augmentation_blur_probability = s.augmentation_blur_probability_spinbox.value()
        augmentation_blur_radius = s.augmentation_blur_radius_spinbox.value()
        training_augmentation = s.get_training_augmentation_config()
        step = s.shift_spinbox.value()
        train_patch_size = (s.train_patch_x_size.value(), s.train_patch_y_size.value())
        random_patch_size_enabled = s.random_patch_size_check_box.isChecked()
        random_patch_min_size = (s.random_patch_min_x_size.value(), s.random_patch_min_y_size.value())
        random_patch_max_size = (s.random_patch_max_x_size.value(), s.random_patch_max_y_size.value())
        recognition_patch_size = (s.recognition_patch_x_size.value(), s.recognition_patch_y_size.value())
        model = s.get_selected_model()
        color_mode = s.get_color_mode_value() if hasattr(s, 'get_color_mode_value') else s.color_type.currentText()
        shuffle_frames = s.shuffle_frames_check_box.isChecked()
        shuffle_patches_in_frame = s.shuffle_patches_in_frame_check_box.isChecked()
        random_crop = s.random_crop_check_box.isChecked()
        crops_per_image = s.crops_per_image_spinbox.value()
        scale_augmentation = s.scale_augmentation_check_box.isChecked()
        scale_augmentation_strength = s.scale_augmentation_strength_spinbox.value()
        synthetic_defect_generator = (
            s.get_synthetic_defect_generator_config()
            if hasattr(s, 'get_synthetic_defect_generator_config')
            else {}
        )
        cutout_enabled = s.cutout_check_box.isChecked()
        cutout_probability = s.cutout_probability_spinbox.value()
        cutout_holes = s.cutout_holes_spinbox.value()
        cutout_size_ratio = s.cutout_size_ratio_spinbox.value()
        random_artifacts_enabled = s.random_artifacts_check_box.isChecked()
        random_artifacts_probability = s.random_artifacts_probability_spinbox.value()
        random_artifacts_count = s.random_artifacts_count_spinbox.value()
        random_artifacts_size_ratio = s.random_artifacts_size_ratio_spinbox.value()
        random_artifact_type_enabled = {
            artifact_name: checkbox.isChecked()
            for artifact_name, checkbox in getattr(s, 'random_artifact_type_checkboxes', {}).items()
        }
        mixup_enabled = s.mixup_check_box.isChecked()
        mixup_probability = s.mixup_probability_spinbox.value()
        mixup_alpha = s.mixup_alpha_spinbox.value()
        validation = s.validation_check_box.isChecked()
        validation_percent = s.validation_spinbox.value()
        validation_source = normalize_validation_source(s.get_validation_source_value())
        validation_image_folder = s.validation_image_path()
        validation_label_folder = s.validation_label_path()
        save_validation_binary_images = s.save_validation_binary_images_check_box.isChecked()
        cut_mode = self._update_cut_mode()
        train_batch_size = s.train_batch_spinbox.value()
        dataloader_num_workers = s.dataloader_num_workers_spinbox.value()
        recognition_batch_size = s.recognition_batch_spinbox.value()
        sync_patch_sizes = s.sync_patch_sizes_check_box.isChecked()
        if sync_patch_sizes:
            recognition_patch_size = train_patch_size
        overlap = s.overlap_spinbox.value()
        recognition_jpeg_quality = s.recognition_jpeg_quality_spinbox.value()
        recognition_multiprocessing_enabled = s.recognition_multiprocessing_check_box.isChecked()
        recognition_binarize_output = s.recognition_binarize_output_check_box.isChecked()
        recognition_use_auto_threshold = s.recognition_use_auto_threshold_check_box.isChecked()
        recognition_threshold = s.recognition_threshold_spinbox.value()
        recognition_tta_enabled = s.recognition_tta_check_box.isChecked()
        confidence_tta_enabled = (
            s.is_confidence_tta_enabled()
            if hasattr(s, 'is_confidence_tta_enabled')
            else False
        )
        recognition_postprocess = s.recognition_postprocess_check_box.isChecked()
        recognition_postprocess_kernel_size = s.recognition_postprocess_kernel_size_spinbox.value()
        confidence_save_mode = (
            s.get_confidence_save_mode_value()
            if hasattr(s, 'get_confidence_save_mode_value')
            else 'off'
        )
        log_update_frequency = s.log_update_frequency_spinbox.value()
        optimizer_name = s.optimizer_type.currentText()
        mixed_precision = s.mixed_precision_type.currentText()
        deep_supervision = s.deep_supervision_check_box.isChecked()
        topograph_enabled = s.topograph_enabled_check_box.isChecked()
        topograph_loss_weight = float(s.topograph_loss_weight_spinbox.value())
        topograph_debug_viz = s.topograph_debug_viz_check_box.isChecked()
        current_state = getattr(self, 'settings_state', SettingsState())
        loss_term_weights = (
            s.get_loss_term_weights()
            if hasattr(s, 'get_loss_term_weights')
            else resolve_loss_term_weights({}, fallback_loss_function='bce')
        )
        loss_function = dominant_loss_function(
            loss_term_weights,
            fallback=getattr(current_state, 'loss_function', 'bce'),
        )
        dice_loss_weight = float(getattr(current_state, 'dice_loss_weight', 0.5))
        iou_loss_weight = float(getattr(current_state, 'iou_loss_weight', 0.5))
        learning_rate = s.learning_rate_spinbox.value()
        weight_decay = s.weight_decay_spinbox.value()
        early_stopping_enabled = s.early_stopping_check_box.isChecked()
        warmup_enabled = s.warmup_check_box.isChecked()
        warmup_epochs = s.warmup_epochs_spinbox.value()
        warmup_start_factor = s.warmup_start_factor_spinbox.value()
        scheduler_name = s.get_scheduler_value()
        scheduler_plateau_factor = s.scheduler_plateau_factor_spinbox.value()
        scheduler_plateau_patience = s.scheduler_plateau_patience_spinbox.value()
        scheduler_plateau_threshold = s.scheduler_plateau_threshold_spinbox.value()
        scheduler_plateau_min_lr = s.scheduler_plateau_min_lr_spinbox.value()
        scheduler_plateau_cooldown = s.scheduler_plateau_cooldown_spinbox.value()
        scheduler_cosine_t_max = s.scheduler_cosine_t_max_spinbox.value()
        scheduler_cosine_eta_min = s.scheduler_cosine_eta_min_spinbox.value()
        scheduler_one_cycle_max_lr = s.scheduler_one_cycle_max_lr_spinbox.value()
        scheduler_one_cycle_pct_start = s.scheduler_one_cycle_pct_start_spinbox.value()
        scheduler_one_cycle_anneal_strategy = s.get_scheduler_one_cycle_anneal_strategy_value()
        scheduler_one_cycle_div_factor = s.scheduler_one_cycle_div_factor_spinbox.value()
        scheduler_one_cycle_final_div_factor = s.scheduler_one_cycle_final_div_factor_spinbox.value()
        scheduler_one_cycle_three_phase = s.scheduler_one_cycle_three_phase_check_box.isChecked()
        scheduler_step_lr_step_size = s.scheduler_step_lr_step_size_spinbox.value()
        scheduler_step_lr_gamma = s.scheduler_step_lr_gamma_spinbox.value()
        hard_mining_enabled = s.hard_mining_check_box.isChecked()
        hard_pixel_mining_enabled = s.hard_pixel_mining_check_box.isChecked()
        hard_pixel_mining_ratio = s.hard_pixel_mining_ratio_spinbox.value()
        sem_segmentation_config = (
            s.get_sem_segmentation_config()
            if hasattr(s, 'get_sem_segmentation_config')
            else dict(getattr(self.settings_state, 'sem_segmentation_config', {}))
        )
        skip_uniform_labels = s.skip_uniform_labels_check_box.isChecked()
        rare_patch_oversampling_enabled = s.rare_patch_oversampling_check_box.isChecked()
        rare_patch_oversampling_factor = s.rare_patch_oversampling_factor_spinbox.value()
        multi_gpu_mode = normalize_multi_gpu_mode(s.multi_gpu_mode_combo.currentText())
        use_multi_gpu = multi_gpu_mode != 'off'
        torch_compile_enabled = s.torch_compile_check_box.isChecked()
        show_batch_preview = self.view.is_batch_preview_enabled()
        crop_enabled = s.enable_crop_processing.isChecked()
        resize_enabled = s.enable_resize_processing.isChecked()
        edge_cut_size = s.cut_corner_spinbox.value()
        compression_factor = s.compression_factor_spinbox.value()
        if hasattr(self.view, 'is_recursive_file_search_enabled'):
            recursive_file_search = self.view.is_recursive_file_search_enabled()
        else:
            recursive_file_search = s.recursive_file_search_check_box.isChecked()

        state = SettingsState(step=step, vertical_rotation=v, horizontal_rotation=h,
                              flip_x=flip_x,
                              flip_y=flip_y,
                              additional_augmentation=additional_augmentation,
                              augmentation_multiplier=augmentation_multiplier,
                              augmentation_brightness_strength=augmentation_brightness_strength,
                              augmentation_contrast_strength=augmentation_contrast_strength,
                              augmentation_gamma_strength=augmentation_gamma_strength,
                              augmentation_noise_probability=augmentation_noise_probability,
                              augmentation_noise_sigma=augmentation_noise_sigma,
                              augmentation_blur_probability=augmentation_blur_probability,
                              augmentation_blur_radius=augmentation_blur_radius,
                              training_augmentation=training_augmentation,
                              sample_size=train_patch_size,
                              train_patch_size=train_patch_size,
                              random_patch_size_enabled=random_patch_size_enabled,
                              random_patch_min_size=random_patch_min_size,
                              random_patch_max_size=random_patch_max_size,
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
                              random_artifacts_flake_enabled=bool(
                                  random_artifact_type_enabled.get('flake', True)
                              ),
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
                              crop_enabled=crop_enabled, resize_enabled=resize_enabled, edge_cut_size=edge_cut_size,
                              compression_factor=compression_factor,
                              recursive_file_search=recursive_file_search,
                              optimizer_name=optimizer_name,
                              mixed_precision=mixed_precision,
                              deep_supervision=deep_supervision,
                              loss_function=loss_function,
                              loss_term_weights=loss_term_weights,
                              dice_loss_weight=dice_loss_weight,
                              iou_loss_weight=iou_loss_weight,
                              topograph_enabled=topograph_enabled,
                              topograph_loss_weight=topograph_loss_weight,
                              topograph_debug_viz=topograph_debug_viz,
                              learning_rate=learning_rate,
                              weight_decay=weight_decay,
                              early_stopping_enabled=early_stopping_enabled,
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
                              hard_pixel_mining_enabled=hard_pixel_mining_enabled,
                              hard_pixel_mining_ratio=hard_pixel_mining_ratio,
                              sem_segmentation_config=sem_segmentation_config,
                              log_update_frequency=log_update_frequency,
                              skip_uniform_labels=skip_uniform_labels,
                              rare_patch_oversampling_enabled=rare_patch_oversampling_enabled,
                              rare_patch_oversampling_factor=rare_patch_oversampling_factor,
                              use_multi_gpu=use_multi_gpu,
                              multi_gpu_mode=multi_gpu_mode,
                              torch_compile_enabled=torch_compile_enabled,
                              show_batch_preview=show_batch_preview)

        self.settings_state = state
