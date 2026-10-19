"""test_settings_optimizer_ui: test main presenter applies 2."""
from __future__ import annotations
import types
import pytest
from neuralimage.lib.data_interfaces import build_synthetic_defect_generator_parameters
from neuralimage.view.settings_panel import SettingsPanel
from neuralimage.view.window_dataclasses import SettingsState

from test_settings_optimizer_ui_parts.preprocessing_groupbox_controls import _import_main_presenter_with_stubs

def test_main_presenter_applies_and_reads_optimizer_settings(qapp):
    module = _import_main_presenter_with_stubs()
    presenter = module.MainPresenter.__new__(module.MainPresenter)
    from PyQt6.QtCore import QObject
    QObject.__init__(presenter)

    panel = SettingsPanel()
    panel.model_type_init(['MockNet'])
    presenter.settings_panel = panel
    presenter.view = types.SimpleNamespace(
        _batch_preview_enabled=True,
        set_batch_preview_enabled=lambda enabled: None,
        is_batch_preview_enabled=lambda: True,
    )

    def _set_batch_preview_enabled(enabled: bool):
        presenter.view._batch_preview_enabled = bool(enabled)

    def _is_batch_preview_enabled() -> bool:
        return bool(presenter.view._batch_preview_enabled)

    presenter.view.set_batch_preview_enabled = _set_batch_preview_enabled
    presenter.view.is_batch_preview_enabled = _is_batch_preview_enabled
    presenter.view.is_recursive_file_search_enabled = lambda: True
    presenter.main_window_state = module.MainWindowState(epochs=1)
    presenter.settings_state = SettingsState(
        model='MockNet',
        validation_source='external',
        validation_image_folder='C:/val_images',
        validation_label_folder='C:/val_labels',
        save_validation_binary_images=True,
        shuffle=False,
        shuffle_patches_in_frame=True,
        random_crop=True,
        crops_per_image=19,
        additional_augmentation=True,
        tech_aug={
            'enabled': True,
            'min_operations': 2,
            'max_operations': 3,
            'debug_return_pair': True,
            'max_changed_pixels_ratio': 0.18,
            'max_foreground_ratio_delta': 0.11,
            'global_width': {'probability': 0.4, 'kernel_size_range': [1, 2]},
            'boundary_aware': {'probability': 0.9, 'band_width_range': [1, 2]},
        },
        augmentation_brightness_strength=0.22,
        augmentation_contrast_strength=0.18,
        augmentation_gamma_strength=0.16,
        augmentation_noise_probability=0.4,
        augmentation_noise_sigma=0.012,
        augmentation_blur_probability=0.45,
        augmentation_blur_radius=1.3,
        optimizer_name='adamw',
        mixed_precision='bf16',
        loss_function='dice',
        loss_term_weights={'bce': 0.45, 'dice': 0.55},
        dice_loss_weight=0.55,
        iou_loss_weight=0.35,
        learning_rate=0.0007,
        weight_decay=0.02,
        deep_supervision=False,
        warmup_enabled=True,
        warmup_epochs=4,
        warmup_start_factor=0.3,
        scheduler_name='step_lr',
        scheduler_step_lr_step_size=5,
        scheduler_step_lr_gamma=0.25,
        hard_mining_enabled=True,
        hard_mining_strength=2.6,
        hard_mining_ema_alpha=0.45,
        hard_pixel_mining_enabled=True,
        hard_pixel_mining_ratio=0.2,
        cutout_enabled=True,
        cutout_probability=0.9,
        cutout_holes=2,
        cutout_size_ratio=0.3,
        random_artifacts_enabled=True,
        random_artifacts_probability=0.65,
        random_artifacts_count=3,
        random_artifacts_size_ratio=0.2,
        mixup_enabled=True,
        mixup_probability=0.8,
        mixup_alpha=0.35,
        synthetic_defect_generator={
            'enabled': True,
            'epoch_size_factor': 1.6,
            'trace_count_range': [6, 9],
            'segment_count_range': [3, 5],
            'trace_half_width_range': [2, 4],
            'background_noise_sigma_range': [0.01, 0.03],
            'trace_noise_sigma_range': [0.02, 0.04],
            'defects': {
                'enabled': True,
                'defect_probability': 0.7,
                'min_defects': 2,
                'max_defects': 4,
                'max_attempts_per_defect': 11,
                'use_input_mask': False,
                'use_defect_mask_as_label': False,
                'defect_probabilities': {
                    'break': 1.0,
                    'short': 1.0,
                    'missing_copper': 1.0,
                    'excess_copper': 1.0,
                    'pinhole': 1.0,
                    'spurious_copper': 1.0,
                    'via': 1.0,
                    'misalignment': 1.0,
                },
                'defect_severities': {
                    'break': 0.75,
                    'short': 0.6,
                    'missing_copper': 0.55,
                    'excess_copper': 0.45,
                    'pinhole': 0.8,
                    'spurious_copper': 0.9,
                    'via': 0.88,
                    'misalignment': 0.5,
                },
            },
        },
        pcb_defects={
            'enabled': True,
            'defect_probability': 0.7,
            'min_defects': 2,
            'max_defects': 4,
            'max_attempts_per_defect': 11,
            'use_input_mask': False,
            'use_defect_mask_as_label': True,
            'defect_probabilities': {
                'break': 1.0,
                'short': 1.0,
                'missing_copper': 1.0,
                'excess_copper': 1.0,
                'pinhole': 1.0,
                'spurious_copper': 1.0,
                'via': 1.0,
                'misalignment': 1.0,
            },
            'defect_severities': {
                'break': 0.75,
                'short': 0.6,
                'missing_copper': 0.55,
                'excess_copper': 0.45,
                'pinhole': 0.8,
                'spurious_copper': 0.9,
                'via': 0.88,
                'misalignment': 0.5,
            },
        },
        skip_uniform_labels=True,
        rare_patch_oversampling_enabled=True,
        rare_patch_oversampling_factor=7,
        recognition_jpeg_quality=88,
        recognition_multiprocessing_enabled=False,
        recognition_binarize_output=False,
        recognition_use_auto_threshold=False,
        recognition_threshold=0.67,
        recognition_tta_enabled=True,
        confidence_tta_enabled=False,
        recognition_postprocess=True,
        recognition_postprocess_kernel_size=5,
        confidence_save_mode='separate_grayscale',
        torch_compile_enabled=False,
        early_stopping_enabled=True,
        early_stopping_patience=6,
        early_stopping_min_delta=0.004,
        early_stopping_restore_best_weights=False,
        show_batch_preview=False,
        dataloader_num_workers=7,
    )

    module.MainPresenter._apply_settings_to_panel(presenter)
    assert panel.optimizer_type.currentText() == 'adamw'
    assert panel.epochs_spinbox.value() == 1
    assert panel.shuffle_frames_check_box.isChecked() is False
    assert panel.shuffle_patches_in_frame_check_box.isChecked() is True
    assert panel.random_crop_check_box.isChecked() is True
    assert panel.crops_per_image_spinbox.value() == 19
    assert panel.tech_augmentation_check_box.isChecked() is False
    assert panel.get_validation_source_value() == 'external'
    assert panel.validation_image_path() == 'C:/val_images'
    assert panel.validation_label_path() == 'C:/val_labels'
    assert panel.save_validation_binary_images_check_box.isChecked() is True
    assert panel.additional_augmentation_check_box.isChecked() is True
    assert panel.augmentation_brightness_spinbox.value() == pytest.approx(0.22)
    assert panel.augmentation_contrast_spinbox.value() == pytest.approx(0.18)
    assert panel.augmentation_gamma_spinbox.value() == pytest.approx(0.16)
    assert panel.augmentation_noise_probability_spinbox.value() == pytest.approx(0.4)
    assert panel.augmentation_noise_sigma_spinbox.value() == pytest.approx(0.012)
    assert panel.augmentation_blur_probability_spinbox.value() == pytest.approx(0.45)
    assert panel.augmentation_blur_radius_spinbox.value() == pytest.approx(1.3)
    assert panel.mixed_precision_type.currentText() == 'bf16'
    assert panel.get_loss_term_weights()['bce'] == pytest.approx(0.45)
    assert panel.get_loss_term_weights()['dice'] == pytest.approx(0.55)
    assert panel.learning_rate_spinbox.value() == pytest.approx(0.0007)
    assert panel.weight_decay_spinbox.value() == pytest.approx(0.02)
    assert panel.deep_supervision_check_box.isChecked() is False
    assert panel.dataloader_num_workers_spinbox.value() == 7
    assert panel.warmup_check_box.isChecked() is True
    assert panel.warmup_epochs_spinbox.value() == 4
    assert panel.warmup_start_factor_spinbox.value() == pytest.approx(0.3)
    assert panel.get_scheduler_value() == 'step_lr'
    assert panel.scheduler_step_lr_step_size_spinbox.value() == 5
    assert panel.scheduler_step_lr_gamma_spinbox.value() == pytest.approx(0.25)
    assert panel.hard_mining_check_box.isChecked() is True
    assert presenter.settings_state.hard_mining_strength == pytest.approx(2.6)
    assert presenter.settings_state.hard_mining_ema_alpha == pytest.approx(0.45)
    assert panel.hard_pixel_mining_check_box.isChecked() is True
    assert panel.hard_pixel_mining_ratio_spinbox.value() == pytest.approx(0.2)
    assert panel.cutout_check_box.isChecked() is True
    assert panel.cutout_probability_spinbox.value() == pytest.approx(0.9)
    assert panel.cutout_holes_spinbox.value() == 2
    assert panel.cutout_size_ratio_spinbox.value() == pytest.approx(0.3)
    assert panel.random_artifacts_check_box.isChecked() is True
    assert panel.random_artifacts_probability_spinbox.value() == pytest.approx(0.65)
    assert panel.random_artifacts_count_spinbox.value() == 3
    assert panel.random_artifacts_size_ratio_spinbox.value() == pytest.approx(0.2)
    assert panel.mixup_check_box.isChecked() is True
    assert panel.mixup_probability_spinbox.value() == pytest.approx(0.8)
    assert panel.mixup_alpha_spinbox.value() == pytest.approx(0.35)
    assert panel.synthetic_defect_generator_check_box.isChecked() is True
    assert panel.synthetic_dataset_factor_spinbox.value() == pytest.approx(1.6)
    assert panel.synthetic_trace_count_min_spinbox.value() == 6
    assert panel.synthetic_trace_count_max_spinbox.value() == 9
    assert panel.pcb_defects_probability_spinbox.value() == pytest.approx(0.7)
    assert panel.pcb_defects_min_count_spinbox.value() == 2
    assert panel.pcb_defects_max_count_spinbox.value() == 4
    assert panel.pcb_defect_type_spinboxes['break'].value() == 75
    assert panel.pcb_defect_type_spinboxes['via'].value() == 88
    assert panel.skip_uniform_labels_check_box.isChecked() is True
    assert panel.rare_patch_oversampling_check_box.isChecked() is True
    assert panel.rare_patch_oversampling_factor_spinbox.value() == 7
    assert panel.recognition_jpeg_quality_spinbox.value() == 88
    assert panel.recognition_multiprocessing_check_box.isChecked() is False
    assert panel.recognition_binarize_output_check_box.isChecked() is False
    assert panel.recognition_use_auto_threshold_check_box.isChecked() is False
    assert panel.recognition_threshold_spinbox.value() == pytest.approx(0.67)
    assert panel.recognition_tta_check_box.isChecked() is True
    assert panel.get_confidence_export_mode_value() == 'model_output'
    assert panel.recognition_postprocess_check_box.isChecked() is True
    assert panel.recognition_postprocess_kernel_size_spinbox.value() == 5
    assert panel.get_confidence_save_mode_value() == 'separate_grayscale'
    assert panel.torch_compile_check_box.isChecked() is False
    assert panel.early_stopping_check_box.isChecked() is True
    assert presenter.settings_state.early_stopping_patience == 6
    assert presenter.settings_state.early_stopping_min_delta == pytest.approx(0.004)
    assert presenter.settings_state.early_stopping_restore_best_weights is False
    assert presenter.view.is_batch_preview_enabled() is False

    panel.shuffle_frames_check_box.setChecked(True)
    panel.shuffle_patches_in_frame_check_box.setChecked(False)
    panel.random_crop_check_box.setChecked(False)
    panel.crops_per_image_spinbox.setValue(11)
    panel.additional_augmentation_check_box.setChecked(True)
    panel.tech_augmentation_check_box.setChecked(True)
    panel.tech_augmentation_debug_pair_check_box.setChecked(False)
    panel.tech_aug_min_operations_spinbox.setValue(1)
    panel.tech_aug_max_operations_spinbox.setValue(2)
    panel.tech_aug_max_changed_pixels_ratio_spinbox.setValue(0.16)
    panel.tech_aug_max_foreground_ratio_delta_spinbox.setValue(0.08)
    panel.tech_aug_global_width_probability_spinbox.setValue(0.55)
    panel.tech_aug_scale_rethreshold_probability_spinbox.setValue(0.45)
    panel.tech_aug_blur_threshold_probability_spinbox.setValue(0.25)
    panel.tech_aug_boundary_aware_probability_spinbox.setValue(0.65)
    panel.tech_aug_local_morphology_probability_spinbox.setValue(0.3)
    panel.tech_aug_gap_variation_probability_spinbox.setValue(0.2)
    panel.augmentation_brightness_spinbox.setValue(0.35)
    panel.augmentation_contrast_spinbox.setValue(0.27)
    panel.augmentation_gamma_spinbox.setValue(0.19)
    panel.augmentation_noise_probability_spinbox.setValue(0.55)
    panel.augmentation_noise_sigma_spinbox.setValue(0.02)
    panel.augmentation_blur_probability_spinbox.setValue(0.5)
    panel.augmentation_blur_radius_spinbox.setValue(1.6)
    panel.optimizer_type.setCurrentText('adamw_muon')
    panel.mixed_precision_type.setCurrentText('off')
    panel.set_loss_term_weights({'bce': 0.2, 'iou': 0.8})
    panel.learning_rate_spinbox.setValue(0.0003)
    panel.weight_decay_spinbox.setValue(0.015)
    panel.deep_supervision_check_box.setChecked(True)
    panel.dataloader_num_workers_spinbox.setValue(2)
    panel.warmup_check_box.setChecked(False)
    panel.warmup_epochs_spinbox.setValue(2)
    panel.warmup_start_factor_spinbox.setValue(0.15)
    panel.set_scheduler_value('one_cycle')
    panel.scheduler_one_cycle_max_lr_spinbox.setValue(0.002)
    panel.scheduler_one_cycle_pct_start_spinbox.setValue(0.4)
    panel.set_scheduler_one_cycle_anneal_strategy_value('linear')
    panel.scheduler_one_cycle_div_factor_spinbox.setValue(10.0)
    panel.scheduler_one_cycle_final_div_factor_spinbox.setValue(500.0)
    panel.scheduler_one_cycle_three_phase_check_box.setChecked(True)
    panel.hard_mining_check_box.setChecked(True)
    panel.hard_pixel_mining_check_box.setChecked(True)
    panel.hard_pixel_mining_ratio_spinbox.setValue(0.35)
    panel.cutout_check_box.setChecked(True)
    panel.cutout_probability_spinbox.setValue(0.75)
    panel.cutout_holes_spinbox.setValue(4)
    panel.cutout_size_ratio_spinbox.setValue(0.28)
    panel.random_artifacts_check_box.setChecked(True)
    panel.random_artifacts_probability_spinbox.setValue(0.55)
    panel.random_artifacts_count_spinbox.setValue(2)
    panel.random_artifacts_size_ratio_spinbox.setValue(0.18)
    panel.mixup_check_box.setChecked(True)
    panel.mixup_probability_spinbox.setValue(0.6)
    panel.mixup_alpha_spinbox.setValue(0.5)
    panel.pcb_defects_check_box.setChecked(True)
    panel.pcb_defects_probability_spinbox.setValue(0.45)
    panel.pcb_defects_min_count_spinbox.setValue(3)
    panel.pcb_defects_max_count_spinbox.setValue(2)
    panel.pcb_defects_use_input_mask_check_box.setChecked(True)
    panel.pcb_defects_use_defect_mask_as_label_check_box.setChecked(False)
    panel.pcb_defect_type_spinboxes['break'].setValue(100)
    panel.pcb_defect_type_spinboxes['short'].setValue(40)
    panel.pcb_defect_type_spinboxes['missing_copper'].setValue(100)
    panel.pcb_defect_type_spinboxes['excess_copper'].setValue(70)
    panel.pcb_defect_type_spinboxes['pinhole'].setValue(90)
    panel.pcb_defect_type_spinboxes['spurious_copper'].setValue(100)
    panel.pcb_defect_type_spinboxes['via'].setValue(100)
    panel.pcb_defect_type_spinboxes['misalignment'].setValue(30)
    panel.set_validation_source_value('external')
    panel.set_validation_image_path('D:/val_images')
    panel.set_validation_label_path('D:/val_labels')
    panel.save_validation_binary_images_check_box.setChecked(True)
    panel.skip_uniform_labels_check_box.setChecked(False)
    panel.rare_patch_oversampling_check_box.setChecked(True)
    panel.rare_patch_oversampling_factor_spinbox.setValue(4)
    panel.recognition_jpeg_quality_spinbox.setValue(82)
    panel.recognition_multiprocessing_check_box.setChecked(True)
    panel.recognition_binarize_output_check_box.setChecked(True)
    panel.recognition_use_auto_threshold_check_box.setChecked(False)
    panel.recognition_threshold_spinbox.setValue(0.59)
    panel.recognition_tta_check_box.setChecked(True)
    panel.recognition_postprocess_check_box.setChecked(True)
    panel.recognition_postprocess_kernel_size_spinbox.setValue(7)
    panel.set_confidence_export_mode_value('tta')
    panel.torch_compile_check_box.setChecked(True)
    panel.early_stopping_check_box.setChecked(False)
    presenter.view.set_batch_preview_enabled(True)

    module.MainPresenter._update_settings_window_state(presenter)
    assert presenter.settings_state.additional_augmentation is True
    assert presenter.settings_state.shuffle is True
    assert presenter.settings_state.shuffle_patches_in_frame is False
    assert presenter.settings_state.random_crop is False
    assert presenter.settings_state.crops_per_image == 11
    assert presenter.settings_state.tech_aug == {}
    assert presenter.settings_state.augmentation_brightness_strength == pytest.approx(0.35)
    assert presenter.settings_state.augmentation_contrast_strength == pytest.approx(0.27)
    assert presenter.settings_state.augmentation_gamma_strength == pytest.approx(0.19)
    assert presenter.settings_state.augmentation_noise_probability == pytest.approx(0.55)
    assert presenter.settings_state.augmentation_noise_sigma == pytest.approx(0.02)
    assert presenter.settings_state.augmentation_blur_probability == pytest.approx(0.5)
    assert presenter.settings_state.augmentation_blur_radius == pytest.approx(1.6)
    assert presenter.settings_state.optimizer_name == 'adamw_muon'
    assert presenter.settings_state.mixed_precision == 'off'
    assert presenter.settings_state.loss_function == 'iou'
    assert presenter.settings_state.loss_term_weights['bce'] == pytest.approx(0.2)
    assert presenter.settings_state.loss_term_weights['iou'] == pytest.approx(0.8)
    assert presenter.settings_state.dice_loss_weight == pytest.approx(0.55)
    assert presenter.settings_state.iou_loss_weight == pytest.approx(0.35)
    assert presenter.settings_state.learning_rate == pytest.approx(0.0003)
    assert presenter.settings_state.weight_decay == pytest.approx(0.015)
    assert presenter.settings_state.deep_supervision is True
    assert presenter.settings_state.dataloader_num_workers == 2
    assert presenter.settings_state.warmup_enabled is False
    assert presenter.settings_state.warmup_epochs == 2
    assert presenter.settings_state.warmup_start_factor == pytest.approx(0.15)
    assert presenter.settings_state.scheduler_name == 'one_cycle'
    assert presenter.settings_state.scheduler_one_cycle_max_lr == pytest.approx(0.002)
    assert presenter.settings_state.scheduler_one_cycle_pct_start == pytest.approx(0.4)
    assert presenter.settings_state.scheduler_one_cycle_anneal_strategy == 'linear'
    assert presenter.settings_state.scheduler_one_cycle_div_factor == pytest.approx(10.0)
    assert presenter.settings_state.scheduler_one_cycle_final_div_factor == pytest.approx(500.0)
    assert presenter.settings_state.scheduler_one_cycle_three_phase is True
    assert presenter.settings_state.hard_mining_enabled is True
    assert presenter.settings_state.hard_mining_strength == pytest.approx(2.6)
    assert presenter.settings_state.hard_mining_ema_alpha == pytest.approx(0.45)
    assert presenter.settings_state.hard_pixel_mining_enabled is True
    assert presenter.settings_state.hard_pixel_mining_ratio == pytest.approx(0.35)
    assert presenter.settings_state.cutout_enabled is True
    assert presenter.settings_state.cutout_probability == pytest.approx(0.75)
    assert presenter.settings_state.cutout_holes == 4
    assert presenter.settings_state.cutout_size_ratio == pytest.approx(0.28)
    assert presenter.settings_state.random_artifacts_enabled is True
    assert presenter.settings_state.random_artifacts_probability == pytest.approx(0.55)
    assert presenter.settings_state.random_artifacts_count == 2
    assert presenter.settings_state.random_artifacts_size_ratio == pytest.approx(0.18)
    assert presenter.settings_state.mixup_enabled is True
    assert presenter.settings_state.mixup_probability == pytest.approx(0.6)
    assert presenter.settings_state.mixup_alpha == pytest.approx(0.5)
    updated_synthetic_generator = build_synthetic_defect_generator_parameters(
        presenter.settings_state.synthetic_defect_generator
    )
    assert updated_synthetic_generator.enabled is True
    assert updated_synthetic_generator.defects.defect_probability == pytest.approx(0.45)
    assert updated_synthetic_generator.defects.min_defects == 2
    assert updated_synthetic_generator.defects.max_defects == 2
    assert updated_synthetic_generator.defects.use_input_mask is True
    assert updated_synthetic_generator.defects.use_defect_mask_as_label is False
    assert updated_synthetic_generator.defects.max_attempts_per_defect == 11
    assert updated_synthetic_generator.defects.defect_probabilities['break'] == pytest.approx(1.0)
    assert updated_synthetic_generator.defects.defect_probabilities['via'] == pytest.approx(1.0)
    assert updated_synthetic_generator.defects.defect_severities['break'] == pytest.approx(1.0)
    assert updated_synthetic_generator.defects.defect_severities['via'] == pytest.approx(1.0)
    assert presenter.settings_state.validation_source == 'external'
    assert presenter.settings_state.validation_image_folder == 'D:/val_images'
    assert presenter.settings_state.validation_label_folder == 'D:/val_labels'
    assert presenter.settings_state.save_validation_binary_images is True
    assert presenter.settings_state.skip_uniform_labels is False
    assert presenter.settings_state.rare_patch_oversampling_enabled is True
    assert presenter.settings_state.rare_patch_oversampling_factor == 4
    assert presenter.settings_state.recognition_jpeg_quality == 82
    assert presenter.settings_state.recognition_multiprocessing_enabled is True
    assert presenter.settings_state.recognition_binarize_output is True
    assert presenter.settings_state.recognition_use_auto_threshold is False
    assert presenter.settings_state.recognition_threshold == pytest.approx(0.59)
    assert presenter.settings_state.recognition_tta_enabled is True
    assert presenter.settings_state.confidence_tta_enabled is True
    assert presenter.settings_state.recognition_postprocess is True
    assert presenter.settings_state.recognition_postprocess_kernel_size == 7
    assert presenter.settings_state.confidence_save_mode == 'separate_grayscale'
    assert presenter.settings_state.torch_compile_enabled is True
    assert presenter.settings_state.early_stopping_enabled is False
    assert presenter.settings_state.early_stopping_patience == 6
    assert presenter.settings_state.early_stopping_min_delta == pytest.approx(0.004)
    assert presenter.settings_state.early_stopping_restore_best_weights is False
    assert presenter.settings_state.show_batch_preview is True
    assert presenter.settings_state.recursive_file_search is True
