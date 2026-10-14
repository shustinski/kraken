"""main_presenter: apply settings panel."""
from __future__ import annotations
from neuralimage.lib.data_interfaces import (
    normalize_multi_gpu_mode,
    normalize_validation_source,
)
from neuralimage.lib.loss_config import resolve_loss_term_weights
import neuralimage.presenter.state_sync as state_sync


class ApplySettingsPanelMixin:
    def _apply_settings_to_panel(self) -> None:
        """
        Populate the widgets in ``self.settings_panel`` with the values stored in
        ``state`` (an instance of :class:`SettingsState`).

        The method mirrors the logic of ``_update_settings_window_state`` - every
        attribute that is read there is written back here.

        Parameters
        ----------
        state:
            An object that contains the same attribute names that
            ``_update_settings_window_state`` reads (e.g. ``step``, ``vertical_rotation``,
            ``horizontal_rotation`` ...). It can be a ``SettingsState`` dataclass,
            a ``namedtuple`` or any plain object with those attributes.

        Notes
        -----
        * Signals emitted by the widgets are **temporarily blocked** while the UI is
          being updated. This prevents spurious "value-changed" callbacks from
          running before the whole form is in a consistent state.
        * If a widget is optional (e.g. it may be ``None`` in some UI versions) we
          guard against ``AttributeError`` so the function is safe to reuse across
          different dialog layouts.
        """
        return state_sync.apply_settings_to_panel(self)
        # ------------------------------------------------------------------
        # 1. Берем короткую ссылку на панель для компактности кода.
        # ------------------------------------------------------------------
        s = self.settings_panel
        state = self.settings_state
        # --------------------------------------------------------------
        # 3. Записываем каждое поле состояния обратно в виджет.
        # --------------------------------------------------------------
        # 3.1 Check-box (bool)
        s.horizontal_rotation.setChecked(state.horizontal_rotation)
        s.vertical_rotation.setChecked(state.vertical_rotation)
        s.flip_x.setChecked(bool(getattr(state, 'flip_x', False)))
        s.flip_y.setChecked(bool(getattr(state, 'flip_y', False)))
        s.additional_augmentation_check_box.setChecked(state.additional_augmentation)
        s.augmentation_brightness_spinbox.setValue(state.augmentation_brightness_strength)
        s.augmentation_contrast_spinbox.setValue(state.augmentation_contrast_strength)
        s.augmentation_gamma_spinbox.setValue(float(getattr(state, 'augmentation_gamma_strength', 0.15)))
        s.augmentation_noise_probability_spinbox.setValue(state.augmentation_noise_probability)
        s.augmentation_noise_sigma_spinbox.setValue(state.augmentation_noise_sigma)
        s.augmentation_blur_probability_spinbox.setValue(float(getattr(state, 'augmentation_blur_probability', 0.25)))
        s.augmentation_blur_radius_spinbox.setValue(float(getattr(state, 'augmentation_blur_radius', 1.0)))
        s.set_training_augmentation_config(getattr(state, 'training_augmentation', {}))
        if hasattr(s, '_sync_augmentation_controls'):
            s._sync_augmentation_controls(state.additional_augmentation)

        # 3.2  Spin boxes (int/float)
        s.shift_spinbox.setValue(state.step)
        train_patch_size = tuple(getattr(state, 'train_patch_size', None) or state.sample_size)
        recognition_patch_size = tuple(getattr(state, 'recognition_patch_size', None) or state.sample_size)
        s.train_patch_size_widget.set_target_size(
            int(train_patch_size[0]),
            int(train_patch_size[1]),
        )
        s.random_patch_size_check_box.setChecked(bool(getattr(state, 'random_patch_size_enabled', False)))
        random_patch_min = tuple(getattr(state, 'random_patch_min_size', (128, 128)))
        random_patch_max = tuple(getattr(state, 'random_patch_max_size', (512, 512)))
        s.random_patch_min_size_widget.set_target_size(
            int(random_patch_min[0]),
            int(random_patch_min[1]),
        )
        s.random_patch_max_size_widget.set_target_size(
            int(random_patch_max[0]),
            int(random_patch_max[1]),
        )
        s.recognition_patch_size_widget.set_target_size(
            int(recognition_patch_size[0]),
            int(recognition_patch_size[1]),
        )

        # 3.3 Combo-box (str - используем setCurrentText, который выбирает
        #      entry if it exists, otherwise it will add a new entry.)
        s.set_model(state.model)
        if hasattr(s, 'set_color_mode_value'):
            s.set_color_mode_value(state.color_mode)
        else:
            s.color_type.setCurrentText(state.color_mode)
        s.shuffle_frames_check_box.setChecked(bool(getattr(state, 'shuffle', True)))
        s.shuffle_patches_in_frame_check_box.setChecked(
            bool(getattr(state, 'shuffle_patches_in_frame', getattr(state, 'shuffle', True)))
        )
        s.random_crop_check_box.setChecked(bool(getattr(state, 'random_crop', False)))
        s.crops_per_image_spinbox.setValue(int(getattr(state, 'crops_per_image', 64)))
        s.augmentation_multiplier_spinbox.setValue(float(getattr(state, 'augmentation_multiplier', 0.0)))
        s.scale_augmentation_check_box.setChecked(bool(getattr(state, 'scale_augmentation', False)))
        s.scale_augmentation_strength_spinbox.setValue(float(getattr(state, 'scale_augmentation_strength', 0.2)))
        if hasattr(s, 'set_synthetic_defect_generator_config'):
            s.set_synthetic_defect_generator_config(getattr(state, 'synthetic_defect_generator', {}))
        s.cutout_check_box.setChecked(bool(getattr(state, 'cutout_enabled', False)))
        s.cutout_probability_spinbox.setValue(float(getattr(state, 'cutout_probability', 1.0)))
        s.cutout_holes_spinbox.setValue(int(getattr(state, 'cutout_holes', 1)))
        s.cutout_size_ratio_spinbox.setValue(float(getattr(state, 'cutout_size_ratio', 0.25)))
        s.random_artifacts_check_box.setChecked(bool(getattr(state, 'random_artifacts_enabled', False)))
        s.random_artifacts_probability_spinbox.setValue(float(getattr(state, 'random_artifacts_probability', 1.0)))
        s.random_artifacts_count_spinbox.setValue(int(getattr(state, 'random_artifacts_count', 1)))
        s.random_artifacts_size_ratio_spinbox.setValue(
            float(getattr(state, 'random_artifacts_size_ratio', 0.25))
        )
        for artifact_name, checkbox in getattr(s, 'random_artifact_type_checkboxes', {}).items():
            checkbox.setChecked(bool(getattr(state, f'random_artifacts_{artifact_name}_enabled', True)))
        s.mixup_check_box.setChecked(bool(getattr(state, 'mixup_enabled', False)))
        s.mixup_probability_spinbox.setValue(float(getattr(state, 'mixup_probability', 1.0)))
        s.mixup_alpha_spinbox.setValue(float(getattr(state, 'mixup_alpha', 0.2)))

        # 3.4  Validation controls
        s.validation_check_box.setChecked(state.use_validation)
        s.validation_spinbox.setValue(state.validation_percent)
        s.set_validation_source_value(
            normalize_validation_source(getattr(state, 'validation_source', 'split'))
        )
        s.set_validation_image_path(str(getattr(state, 'validation_image_folder', '')))
        s.set_validation_label_path(str(getattr(state, 'validation_label_folder', '')))
        s.save_validation_binary_images_check_box.setChecked(
            bool(getattr(state, 'save_validation_binary_images', False))
        )

        # 3.5 Режим нарезки

        s.restore_cut_mode(state.sample_cut_mode)
        if hasattr(s, '_sync_augmentation_controls'):
            s._sync_augmentation_controls(state.additional_augmentation)
        if hasattr(s, '_sync_training_augmentation_controls'):
            s._sync_training_augmentation_controls()
        s.epochs_spinbox.setValue(int(getattr(self.main_window_state, 'epochs', 20)))

        # 3.6 Размер батча / overlap
        train_batch_size = int(getattr(state, 'train_batch_size', None) or state.batch_size)
        recognition_batch_size = int(getattr(state, 'recognition_batch_size', None) or state.batch_size)
        s.train_batch_spinbox.setValue(train_batch_size)
        s.dataloader_num_workers_spinbox.setValue(int(getattr(state, 'dataloader_num_workers', -1)))
        s.recognition_batch_spinbox.setValue(recognition_batch_size)
        s.overlap_spinbox.setValue(state.overlap)
        s.recognition_jpeg_quality_spinbox.setValue(int(getattr(state, 'recognition_jpeg_quality', 95)))
        s.recognition_multiprocessing_check_box.setChecked(
            bool(getattr(state, 'recognition_multiprocessing_enabled', True))
        )
        s.recognition_binarize_output_check_box.setChecked(bool(getattr(state, 'recognition_binarize_output', True)))
        s.recognition_use_auto_threshold_check_box.setChecked(
            bool(getattr(state, 'recognition_use_auto_threshold', True))
        )
        s.recognition_threshold_spinbox.setValue(float(getattr(state, 'recognition_threshold', 0.5)))
        s.recognition_tta_check_box.setChecked(bool(getattr(state, 'recognition_tta_enabled', False)))
        s.recognition_postprocess_check_box.setChecked(bool(getattr(state, 'recognition_postprocess', False)))
        s.recognition_postprocess_kernel_size_spinbox.setValue(
            int(getattr(state, 'recognition_postprocess_kernel_size', 3))
        )
        if hasattr(s, 'set_confidence_output_mode'):
            s.set_confidence_output_mode(
                str(getattr(state, 'confidence_save_mode', 'off')),
                bool(getattr(state, 'confidence_tta_enabled', False)),
            )
        elif hasattr(s, 'set_confidence_save_mode_value'):
            s.set_confidence_save_mode_value(str(getattr(state, 'confidence_save_mode', 'off')))
        s.log_update_frequency_spinbox.setValue(state.log_update_frequency)
        s.optimizer_type.setCurrentText(state.optimizer_name)
        s.mixed_precision_type.setCurrentText(state.mixed_precision)
        s.deep_supervision_check_box.setChecked(bool(getattr(state, 'deep_supervision', False)))
        s.topograph_enabled_check_box.setChecked(bool(getattr(state, 'topograph_enabled', False)))
        s.topograph_loss_weight_spinbox.setValue(float(getattr(state, 'topograph_loss_weight', 0.1)))
        s.topograph_debug_viz_check_box.setChecked(bool(getattr(state, 'topograph_debug_viz', False)))
        s._on_topograph_enabled_toggled(s.topograph_enabled_check_box.isChecked())
        if hasattr(s, 'set_loss_term_weights'):
            s.set_loss_term_weights(
                resolve_loss_term_weights(
                    getattr(state, 'loss_term_weights', None),
                    fallback_loss_function=state.loss_function,
                )
            )
        if hasattr(s, '_sync_loss_controls'):
            s._sync_loss_controls()
        s.learning_rate_spinbox.setValue(state.learning_rate)
        s.weight_decay_spinbox.setValue(state.weight_decay)
        s.early_stopping_check_box.setChecked(state.early_stopping_enabled)
        s.warmup_check_box.setChecked(state.warmup_enabled)
        s.warmup_epochs_spinbox.setValue(state.warmup_epochs)
        s.warmup_start_factor_spinbox.setValue(state.warmup_start_factor)
        s.set_scheduler_value(str(getattr(state, 'scheduler_name', 'off')))
        s.scheduler_plateau_factor_spinbox.setValue(float(getattr(state, 'scheduler_plateau_factor', 0.5)))
        s.scheduler_plateau_patience_spinbox.setValue(int(getattr(state, 'scheduler_plateau_patience', 3)))
        s.scheduler_plateau_threshold_spinbox.setValue(float(getattr(state, 'scheduler_plateau_threshold', 1e-4)))
        s.scheduler_plateau_min_lr_spinbox.setValue(float(getattr(state, 'scheduler_plateau_min_lr', 1e-6)))
        s.scheduler_plateau_cooldown_spinbox.setValue(int(getattr(state, 'scheduler_plateau_cooldown', 0)))
        s.scheduler_cosine_t_max_spinbox.setValue(int(getattr(state, 'scheduler_cosine_t_max', 10)))
        s.scheduler_cosine_eta_min_spinbox.setValue(float(getattr(state, 'scheduler_cosine_eta_min', 1e-6)))
        s.scheduler_one_cycle_max_lr_spinbox.setValue(float(getattr(state, 'scheduler_one_cycle_max_lr', 1e-3)))
        s.scheduler_one_cycle_pct_start_spinbox.setValue(
            float(getattr(state, 'scheduler_one_cycle_pct_start', 0.3))
        )
        s.set_scheduler_one_cycle_anneal_strategy_value(
            str(getattr(state, 'scheduler_one_cycle_anneal_strategy', 'cos'))
        )
        s.scheduler_one_cycle_div_factor_spinbox.setValue(
            float(getattr(state, 'scheduler_one_cycle_div_factor', 25.0))
        )
        s.scheduler_one_cycle_final_div_factor_spinbox.setValue(
            float(getattr(state, 'scheduler_one_cycle_final_div_factor', 10000.0))
        )
        s.scheduler_one_cycle_three_phase_check_box.setChecked(
            bool(getattr(state, 'scheduler_one_cycle_three_phase', False))
        )
        s.scheduler_step_lr_step_size_spinbox.setValue(int(getattr(state, 'scheduler_step_lr_step_size', 10)))
        s.scheduler_step_lr_gamma_spinbox.setValue(float(getattr(state, 'scheduler_step_lr_gamma', 0.1)))
        s.hard_mining_check_box.setChecked(state.hard_mining_enabled)
        s.hard_pixel_mining_check_box.setChecked(bool(getattr(state, 'hard_pixel_mining_enabled', False)))
        s.hard_pixel_mining_ratio_spinbox.setValue(float(getattr(state, 'hard_pixel_mining_ratio', 0.25)))
        if hasattr(s, 'set_sem_segmentation_config'):
            s.set_sem_segmentation_config(getattr(state, 'sem_segmentation_config', {}))
        s.skip_uniform_labels_check_box.setChecked(state.skip_uniform_labels)
        s.rare_patch_oversampling_check_box.setChecked(
            bool(getattr(state, 'rare_patch_oversampling_enabled', False))
        )
        s.rare_patch_oversampling_factor_spinbox.setValue(
            int(getattr(state, 'rare_patch_oversampling_factor', 2))
        )
        multi_gpu_mode = normalize_multi_gpu_mode(
            getattr(state, 'multi_gpu_mode', ''),
            use_multi_gpu_fallback=bool(getattr(state, 'use_multi_gpu', False)),
        )
        s.sync_patch_sizes_check_box.setChecked(bool(getattr(state, 'sync_patch_sizes', True)))
        if hasattr(self.view, 'set_recursive_file_search'):
            self.view.set_recursive_file_search(bool(getattr(state, 'recursive_file_search', False)))
        elif hasattr(s, 'recursive_file_search_check_box'):
            s.recursive_file_search_check_box.setChecked(bool(getattr(state, 'recursive_file_search', False)))
        s.multi_gpu_mode_combo.setCurrentText(multi_gpu_mode)
        s.torch_compile_check_box.setChecked(state.torch_compile_enabled)
        view = self.__dict__.get('view')
        if view is not None and hasattr(view, 'set_batch_preview_enabled'):
            view.set_batch_preview_enabled(state.show_batch_preview)

        # 3.7  Additional processing flags
        s.enable_crop_processing.setChecked(state.crop_enabled)
        s.enable_resize_processing.setChecked(state.resize_enabled)
        if hasattr(s, '_sync_preprocess_controls'):
            s._sync_preprocess_controls()
        if hasattr(s, '_sync_patch_size_controls'):
            s._sync_patch_size_controls()
        if hasattr(s, '_sync_rare_patch_oversampling_controls'):
            s._sync_rare_patch_oversampling_controls()
        if hasattr(s, '_sync_recognition_output_controls'):
            s._sync_recognition_output_controls()
        if hasattr(s, 'sync_business_logic_controls'):
            main_state = self.__dict__.get('main_window_state')
            work_mode = getattr(main_state, 'work_mode', '')
            s.sync_business_logic_controls(work_mode)

        # 3.8 Размер обрезки по краям
        s.cut_corner_spinbox.setValue(state.edge_cut_size)

        # 3.9 Compression factor
        s.compression_factor_spinbox.setValue(max(1, int(getattr(state, 'compression_factor', 1))))
