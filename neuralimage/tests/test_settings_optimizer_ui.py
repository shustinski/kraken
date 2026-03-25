import importlib
import sys
import types

import pytest

pytest.importorskip('PyQt6')

from PyQt6.QtWidgets import QApplication

from lib.data_interfaces import WorkMode
from view.settings_panel import SettingsPanel
from view.window_dataclasses import SettingsState


@pytest.fixture(scope='module')
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _import_main_presenter_with_stubs():
    nn_stub = types.ModuleType('model.NeuralNetwork')
    nn_stub.get_registered_models = lambda: {}
    handler_stub = types.ModuleType('model.general_neural_handler')
    handler_stub.GeneralNeuralHandler = object
    images_stub = types.ModuleType('lib.images')
    images_stub.SampleWorker = object

    sys.modules['model.NeuralNetwork'] = nn_stub
    sys.modules['model.general_neural_handler'] = handler_stub
    sys.modules['lib.images'] = images_stub
    return importlib.import_module('presenter.main_presenter')


def test_settings_panel_emits_optimizer_settings_changed(qapp):
    panel = SettingsPanel()
    panel.connect_internal_signals()

    calls = {'count': 0}
    panel.optimizer_settings_changed.connect(lambda: calls.__setitem__('count', calls['count'] + 1))

    panel.optimizer_type.setCurrentText('adamw')
    panel.mixed_precision_type.setCurrentText('fp16')
    panel.learning_rate_spinbox.setValue(0.0002)
    panel.weight_decay_spinbox.setValue(0.01)

    assert calls['count'] >= 4


def test_settings_panel_localization_resources_cover_runtime_fallback_keys():
    import json
    from pathlib import Path

    required_keys = {
        'recognition_group',
        'save_validation_binary_images',
        'save_validation_binary_images_tip',
        'validation_source_label',
        'validation_source_split',
        'validation_source_external',
        'validation_source_tip',
        'validation_image_path_label',
        'validation_image_path_tip',
        'validation_label_path_label',
        'validation_label_path_tip',
        'scheduler_name_label',
        'scheduler_plateau_factor_tip',
        'scheduler_plateau_patience_tip',
        'scheduler_plateau_threshold_tip',
        'scheduler_plateau_min_lr_tip',
        'scheduler_plateau_cooldown_tip',
        'scheduler_cosine_t_max_tip',
        'scheduler_cosine_eta_min_tip',
        'scheduler_one_cycle_max_lr_tip',
        'scheduler_one_cycle_pct_start_tip',
        'scheduler_one_cycle_anneal_strategy_tip',
        'scheduler_one_cycle_div_factor_tip',
        'scheduler_one_cycle_final_div_factor_tip',
        'scheduler_one_cycle_three_phase_tip',
        'scheduler_step_lr_step_size_tip',
        'scheduler_step_lr_gamma_tip',
        'cutout_probability_tip',
        'cutout_holes_tip',
        'cutout_size_ratio_tip',
        'mixup_probability_tip',
        'mixup_alpha_tip',
        'pcb_defects_group',
        'pcb_defects_group_tip',
        'pcb_defects_probability_tip',
        'pcb_defects_min_count_tip',
        'pcb_defects_max_count_tip',
        'pcb_defects_use_input_mask',
        'pcb_defects_use_input_mask_tip',
        'pcb_defects_use_defect_mask_as_label',
        'pcb_defects_use_defect_mask_as_label_tip',
        'pcb_break_weight_tip',
        'pcb_short_weight_tip',
        'pcb_missing_copper_weight_tip',
        'pcb_excess_copper_weight_tip',
        'pcb_pinhole_weight_tip',
        'pcb_spurious_copper_weight_tip',
        'pcb_via_weight_tip',
        'pcb_misalignment_weight_tip',
        'loss_function',
        'sample_size_tip',
        'shift_tip',
    }

    for path in ('resources/ui_texts_ru.json', 'resources/ui_texts_en.json'):
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        settings_panel = data.get('settings_panel', {})
        missing = sorted(required_keys - set(settings_panel))
        assert not missing, f'{path} missing keys: {missing}'


def test_settings_panel_optimizer_presets_apply_values_and_highlight_active(qapp):
    panel = SettingsPanel()
    panel.connect_internal_signals()

    adam_btn, adamw_btn, muon_btn = panel.optimizer_preset_buttons

    adamw_btn.click()
    assert panel.optimizer_type.currentText() == 'adamw'
    assert panel.learning_rate_spinbox.value() == pytest.approx(0.0005)
    assert panel.weight_decay_spinbox.value() == pytest.approx(0.01)
    assert adamw_btn.isChecked() is True
    assert adam_btn.isChecked() is False
    assert muon_btn.isChecked() is False

    muon_btn.click()
    assert panel.optimizer_type.currentText() == 'adamw_muon'
    assert panel.learning_rate_spinbox.value() == pytest.approx(0.0003)
    assert panel.weight_decay_spinbox.value() == pytest.approx(0.02)
    assert muon_btn.isChecked() is True
    assert adam_btn.isChecked() is False
    assert adamw_btn.isChecked() is False

    panel.learning_rate_spinbox.setValue(0.00031)
    assert adam_btn.isChecked() is False
    assert adamw_btn.isChecked() is False
    assert muon_btn.isChecked() is False


def test_settings_panel_toggles_validation_spinbox(qapp):
    panel = SettingsPanel()

    panel.validation_check_box.setChecked(False)
    assert panel.validation_spinbox.isEnabled() is False

    panel.validation_check_box.setChecked(True)
    assert panel.validation_spinbox.isEnabled() is True


def test_settings_panel_shows_only_selected_scheduler_fields(qapp):
    panel = SettingsPanel()

    panel.set_scheduler_value('off')
    panel._sync_scheduler_controls()
    assert panel._field_rows[panel.scheduler_step_lr_step_size_spinbox].isHidden() is True
    assert panel._field_rows[panel.scheduler_one_cycle_max_lr_spinbox].isHidden() is True

    panel.set_scheduler_value('step_lr')
    panel._sync_scheduler_controls()
    assert panel._field_rows[panel.scheduler_step_lr_step_size_spinbox].isHidden() is False
    assert panel._field_rows[panel.scheduler_step_lr_gamma_spinbox].isHidden() is False
    assert panel._field_rows[panel.scheduler_plateau_factor_spinbox].isHidden() is True

    panel.set_scheduler_value('one_cycle')
    panel._sync_scheduler_controls()
    assert panel._field_rows[panel.scheduler_one_cycle_max_lr_spinbox].isHidden() is False
    assert panel._field_rows[panel.scheduler_one_cycle_three_phase_check_box].isHidden() is False
    assert panel._field_rows[panel.scheduler_step_lr_step_size_spinbox].isHidden() is True


def test_settings_panel_switches_between_split_and_external_validation_controls(qapp):
    panel = SettingsPanel()

    panel.validation_check_box.setChecked(True)
    panel.set_validation_source_value('split')
    assert panel._field_rows[panel.validation_spinbox].isEnabled() is True
    assert panel._field_rows[panel.validation_image_path_label].isEnabled() is False
    assert panel._field_rows[panel.validation_label_path_label].isEnabled() is False

    panel.set_validation_source_value('external')
    assert panel._field_rows[panel.validation_spinbox].isEnabled() is False
    assert panel._field_rows[panel.validation_image_path_label].isEnabled() is True
    assert panel._field_rows[panel.validation_label_path_label].isEnabled() is True


def test_settings_panel_disables_field_rows_and_descriptions(qapp):
    panel = SettingsPanel()

    panel.validation_check_box.setChecked(False)
    validation_row = panel._field_rows[panel.validation_spinbox]
    validation_desc = panel._desc_labels['validation_percent']
    assert validation_row.isEnabled() is False
    assert validation_desc.isEnabled() is False

    panel.validation_check_box.setChecked(True)
    assert validation_row.isEnabled() is True
    assert validation_desc.isEnabled() is True

    panel.additional_augmentation_check_box.setChecked(True)
    panel.additional_augmentation_check_box.setChecked(False)
    aug_row = panel._field_rows[panel.augmentation_brightness_spinbox]
    aug_desc = panel._desc_labels['augmentation_brightness_strength']
    assert aug_row.isEnabled() is False
    assert aug_desc.isEnabled() is False

    panel.enable_resize_processing.setChecked(False)
    target_row = panel._field_rows[panel.target_size_widget]
    assert target_row.isEnabled() is False
    assert panel.target_x_size.isEnabled() is False
    assert panel.target_y_size.isEnabled() is False

    panel.enable_resize_processing.setChecked(True)
    assert target_row.isEnabled() is True
    assert panel.target_x_size.isEnabled() is True
    assert panel.target_y_size.isEnabled() is True


def test_settings_panel_syncs_tech_aug_controls(qapp):
    panel = SettingsPanel()

    panel.tech_augmentation_check_box.setChecked(False)
    assert panel.tech_augmentation_debug_pair_check_box.isEnabled() is False
    assert panel._field_rows[panel.tech_aug_min_operations_spinbox].isEnabled() is False
    assert panel._field_rows[panel.tech_aug_boundary_aware_probability_spinbox].isEnabled() is False

    panel.tech_augmentation_check_box.setChecked(True)
    assert panel.tech_augmentation_debug_pair_check_box.isEnabled() is True
    assert panel._field_rows[panel.tech_aug_min_operations_spinbox].isEnabled() is True
    assert panel._field_rows[panel.tech_aug_boundary_aware_probability_spinbox].isEnabled() is True

    panel.sync_business_logic_controls(WorkMode.recognition_only.value)
    assert panel.tech_augmentation_check_box.isEnabled() is False
    assert panel._field_rows[panel.tech_aug_min_operations_spinbox].isEnabled() is False

    panel.sync_business_logic_controls(WorkMode.train_only.value)
    assert panel.tech_augmentation_check_box.isEnabled() is True


def test_settings_panel_syncs_recognition_output_controls(qapp):
    panel = SettingsPanel()

    panel.recognition_binarize_output_check_box.setChecked(True)
    panel.recognition_use_auto_threshold_check_box.setChecked(True)
    assert panel._field_rows[panel.recognition_threshold_spinbox].isEnabled() is False

    panel.recognition_use_auto_threshold_check_box.setChecked(False)
    assert panel._field_rows[panel.recognition_threshold_spinbox].isEnabled() is True

    panel.recognition_postprocess_check_box.setChecked(True)
    assert panel._field_rows[panel.recognition_postprocess_kernel_size_spinbox].isEnabled() is True

    panel.recognition_binarize_output_check_box.setChecked(False)
    assert panel._field_rows[panel.recognition_threshold_spinbox].isEnabled() is False
    assert panel._field_rows[panel.recognition_postprocess_kernel_size_spinbox].isEnabled() is False


def test_settings_panel_lists_new_loss_functions(qapp):
    panel = SettingsPanel()

    assert 'cldice' in panel.loss_term_checkboxes
    assert 'boundary' in panel.loss_term_checkboxes
    assert 'focal_tversky' in panel.loss_term_checkboxes
    assert 'bce_dice' not in panel.loss_term_checkboxes
    assert 'focal_dice' not in panel.loss_term_checkboxes
    assert panel.loss_terms_groupbox.title().strip()
    assert panel.loss_terms_widget not in panel._field_rows


def test_settings_panel_limits_loss_weight_sum_online(qapp):
    panel = SettingsPanel()

    panel.set_loss_term_weights({'bce': 0.8, 'dice': 0.2})
    panel.loss_term_checkboxes['iou'].setChecked(True)
    panel.loss_term_spinboxes['iou'].setValue(0.5)

    weights = panel.get_loss_term_weights()
    assert sum(weights.values()) <= 1.0 + 1e-6
    assert panel.loss_term_spinboxes['iou'].value() == pytest.approx(0.0)
    assert 'iou' not in weights


def test_settings_panel_uses_single_visible_label_inside_each_labeled_row(qapp):
    panel = SettingsPanel()

    cases = (
        (panel.general_form, 'sync_patch_sizes', panel.sync_patch_sizes_check_box),
        (panel.general_form, 'train_patch_size', panel.train_patch_size_widget),
        (panel.general_form, 'recognition_patch_size', panel.recognition_patch_size_widget),
        (panel.augmentation_form, 'scale_augmentation_strength', panel.scale_augmentation_strength_spinbox),
        (panel.augmentation_form, 'tech_aug_boundary_aware_probability', panel.tech_aug_boundary_aware_probability_spinbox),
        (panel.optimizer_form, 'train_batch_size', panel.train_batch_spinbox),
        (panel.optimizer_form, 'dataloader_num_workers', panel.dataloader_num_workers_spinbox),
        (panel.optimizer_form, 'recognition_batch_size', panel.recognition_batch_spinbox),
        (panel.runtime_form, 'multi_gpu', panel.multi_gpu_mode_combo),
        (panel.scheduler_form, 'scheduler_name', panel.scheduler_type_combo),
    )

    for form, key, field in cases:
        row_widget = panel._field_rows.get(field, field)
        desc_label = panel._desc_labels[key]
        assert desc_label.text().strip()
        assert form.labelForField(row_widget) is None


def test_settings_panel_keeps_color_mode_value_stable_when_language_changes(qapp):
    panel = SettingsPanel()
    panel.set_color_mode_value('ЧБ')

    panel.set_ui_language('en')
    assert panel.get_color_mode_value() == 'ЧБ'
    assert panel.settings_tabs.tabText(panel._page_indexes['base']) == 'Basic'
    assert panel.settings_tabs.tabText(panel._page_indexes['training']) == 'Training'
    assert panel.settings_tabs.tabText(panel._page_indexes['recognition']) == 'Recognition'
    assert panel.settings_tabs.tabText(panel._page_indexes['expert']) == 'Expert'

    panel.set_ui_language('ru')
    assert panel.get_color_mode_value() == 'ЧБ'
    assert panel.settings_tabs.tabText(panel._page_indexes['base']) == 'Базовые'
    assert panel.settings_tabs.tabText(panel._page_indexes['training']) == 'Обучение'
    assert panel.settings_tabs.tabText(panel._page_indexes['recognition']) == 'Распознавание'
    assert panel.settings_tabs.tabText(panel._page_indexes['expert']) == 'Эксперт'


def test_settings_panel_disables_irrelevant_controls_by_work_mode(qapp):
    panel = SettingsPanel()

    panel.sync_business_logic_controls(WorkMode.recognition_only.value)
    assert panel.sample_type_groupbox.isEnabled() is False
    assert panel.additional_augmentation_check_box.isEnabled() is False
    assert panel.tech_augmentation_check_box.isEnabled() is False
    assert panel.cutout_check_box.isEnabled() is False
    assert panel.random_artifacts_check_box.isEnabled() is False
    assert panel.mixup_check_box.isEnabled() is False
    assert panel.pcb_defects_check_box.isEnabled() is False
    assert panel.validation_check_box.isEnabled() is False
    assert panel._field_rows[panel.optimizer_type].isEnabled() is False
    assert panel._field_rows[panel.overlap_spinbox].isEnabled() is True
    assert panel._field_rows[panel.batch_spinbox].isEnabled() is True
    assert panel._field_rows[panel.nn_model_type].isEnabled() is False
    assert panel.multi_gpu_check_box.isEnabled() is False
    assert panel.torch_compile_check_box.isEnabled() is True
    assert panel.edit_rare_regions_button.isEnabled() is False

    panel.sync_business_logic_controls(WorkMode.train_only.value)
    assert panel.sample_type_groupbox.isEnabled() is True
    assert panel.additional_augmentation_check_box.isEnabled() is True
    assert panel.tech_augmentation_check_box.isEnabled() is True
    assert panel.cutout_check_box.isEnabled() is True
    assert panel.random_artifacts_check_box.isEnabled() is True
    assert panel.mixup_check_box.isEnabled() is True
    assert panel.pcb_defects_check_box.isEnabled() is True
    assert panel._field_rows[panel.optimizer_type].isEnabled() is True
    assert panel._field_rows[panel.overlap_spinbox].isEnabled() is False
    assert panel._field_rows[panel.nn_model_type].isEnabled() is True
    assert panel.edit_rare_regions_button.isEnabled() is True

    panel.sync_business_logic_controls(WorkMode.further_training.value)
    assert panel._field_rows[panel.nn_model_type].isEnabled() is False
    assert panel._field_rows[panel.overlap_spinbox].isEnabled() is True


def test_settings_panel_toggles_pcb_defect_controls(qapp):
    panel = SettingsPanel()
    panel.sync_business_logic_controls(WorkMode.train_only.value)

    panel.pcb_defects_check_box.setChecked(False)
    panel._sync_training_augmentation_controls()
    assert panel._field_rows[panel.pcb_defects_probability_spinbox].isEnabled() is False
    assert panel._field_rows[panel.pcb_defects_min_count_spinbox].isEnabled() is False
    assert panel._field_rows[panel.pcb_defect_type_spinboxes['break']].isEnabled() is False
    assert panel.pcb_defects_use_input_mask_check_box.isEnabled() is False

    panel.pcb_defects_check_box.setChecked(True)
    assert panel._field_rows[panel.pcb_defects_probability_spinbox].isEnabled() is True
    assert panel._field_rows[panel.pcb_defects_max_count_spinbox].isEnabled() is True
    assert panel._field_rows[panel.pcb_defect_type_spinboxes['via']].isEnabled() is True
    assert panel.pcb_defects_use_defect_mask_as_label_check_box.isEnabled() is True


def test_settings_panel_emits_validation_settings_changed(qapp):
    panel = SettingsPanel()
    panel.connect_internal_signals()
    calls = {'count': 0}
    panel.validation_settings_changed.connect(lambda: calls.__setitem__('count', calls['count'] + 1))

    panel.validation_check_box.setChecked(True)
    panel.validation_spinbox.setValue(25)

    assert calls['count'] >= 2


def test_settings_panel_emits_reset_defaults_requested(qapp):
    panel = SettingsPanel()
    panel.connect_internal_signals()
    calls = {'count': 0}
    panel.reset_defaults_requested.connect(lambda: calls.__setitem__('count', calls['count'] + 1))

    panel.reset_defaults_button.click()

    assert calls['count'] == 1


def test_main_presenter_applies_and_reads_optimizer_settings(qapp):
    module = _import_main_presenter_with_stubs()
    presenter = module.MainPresenter.__new__(module.MainPresenter)

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
        pcb_defects={
            'enabled': True,
            'defect_probability': 0.7,
            'min_defects': 2,
            'max_defects': 4,
            'max_attempts_per_defect': 11,
            'use_input_mask': False,
            'use_defect_mask_as_label': True,
            'defect_probabilities': {
                'break': 1.5,
                'short': 0.6,
                'missing_copper': 1.1,
                'excess_copper': 0.9,
                'pinhole': 0.8,
                'spurious_copper': 1.25,
                'via': 1.75,
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
        recognition_postprocess=True,
        recognition_postprocess_kernel_size=5,
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
    assert panel.shuffle_frames_check_box.isChecked() is False
    assert panel.shuffle_patches_in_frame_check_box.isChecked() is True
    assert panel.random_crop_check_box.isChecked() is True
    assert panel.crops_per_image_spinbox.value() == 19
    assert panel.tech_augmentation_check_box.isChecked() is True
    assert panel.tech_augmentation_debug_pair_check_box.isChecked() is True
    assert panel.tech_aug_min_operations_spinbox.value() == 2
    assert panel.tech_aug_max_operations_spinbox.value() == 3
    assert panel.tech_aug_max_changed_pixels_ratio_spinbox.value() == pytest.approx(0.18)
    assert panel.tech_aug_max_foreground_ratio_delta_spinbox.value() == pytest.approx(0.11)
    assert panel.tech_aug_global_width_probability_spinbox.value() == pytest.approx(0.4)
    assert panel.tech_aug_boundary_aware_probability_spinbox.value() == pytest.approx(0.9)
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
    assert panel.dataloader_num_workers_spinbox.value() == 7
    assert panel.warmup_check_box.isChecked() is True
    assert panel.warmup_epochs_spinbox.value() == 4
    assert panel.warmup_start_factor_spinbox.value() == pytest.approx(0.3)
    assert panel.get_scheduler_value() == 'step_lr'
    assert panel.scheduler_step_lr_step_size_spinbox.value() == 5
    assert panel.scheduler_step_lr_gamma_spinbox.value() == pytest.approx(0.25)
    assert panel.hard_mining_check_box.isChecked() is True
    assert panel.hard_mining_strength_spinbox.value() == pytest.approx(2.6)
    assert panel.hard_mining_ema_alpha_spinbox.value() == pytest.approx(0.45)
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
    assert panel.pcb_defects_check_box.isChecked() is True
    assert panel.pcb_defects_probability_spinbox.value() == pytest.approx(0.7)
    assert panel.pcb_defects_min_count_spinbox.value() == 2
    assert panel.pcb_defects_max_count_spinbox.value() == 4
    assert panel.pcb_defects_use_input_mask_check_box.isChecked() is False
    assert panel.pcb_defects_use_defect_mask_as_label_check_box.isChecked() is True
    assert panel.pcb_defect_type_spinboxes['break'].value() == pytest.approx(1.5)
    assert panel.pcb_defect_type_spinboxes['via'].value() == pytest.approx(1.75)
    assert panel.skip_uniform_labels_check_box.isChecked() is True
    assert panel.rare_patch_oversampling_check_box.isChecked() is True
    assert panel.rare_patch_oversampling_factor_spinbox.value() == 7
    assert panel.recognition_jpeg_quality_spinbox.value() == 88
    assert panel.recognition_multiprocessing_check_box.isChecked() is False
    assert panel.recognition_binarize_output_check_box.isChecked() is False
    assert panel.recognition_use_auto_threshold_check_box.isChecked() is False
    assert panel.recognition_threshold_spinbox.value() == pytest.approx(0.67)
    assert panel.recognition_postprocess_check_box.isChecked() is True
    assert panel.recognition_postprocess_kernel_size_spinbox.value() == 5
    assert panel.torch_compile_check_box.isChecked() is False
    assert panel.early_stopping_check_box.isChecked() is True
    assert panel.early_stopping_patience_spinbox.value() == 6
    assert panel.early_stopping_min_delta_spinbox.value() == pytest.approx(0.004)
    assert panel.restore_best_weights_check_box.isChecked() is False
    assert presenter.view.is_batch_preview_enabled() is False

    panel.cut_dataset_type.setChecked(True)
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
    panel.hard_mining_strength_spinbox.setValue(3.4)
    panel.hard_mining_ema_alpha_spinbox.setValue(0.25)
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
    panel.pcb_defect_type_spinboxes['break'].setValue(2.0)
    panel.pcb_defect_type_spinboxes['short'].setValue(0.4)
    panel.pcb_defect_type_spinboxes['missing_copper'].setValue(1.2)
    panel.pcb_defect_type_spinboxes['excess_copper'].setValue(0.7)
    panel.pcb_defect_type_spinboxes['pinhole'].setValue(0.9)
    panel.pcb_defect_type_spinboxes['spurious_copper'].setValue(1.4)
    panel.pcb_defect_type_spinboxes['via'].setValue(1.1)
    panel.pcb_defect_type_spinboxes['misalignment'].setValue(0.3)
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
    panel.recognition_postprocess_check_box.setChecked(True)
    panel.recognition_postprocess_kernel_size_spinbox.setValue(7)
    panel.torch_compile_check_box.setChecked(True)
    panel.early_stopping_check_box.setChecked(False)
    panel.early_stopping_patience_spinbox.setValue(2)
    panel.early_stopping_min_delta_spinbox.setValue(0.002)
    panel.restore_best_weights_check_box.setChecked(True)
    presenter.view.set_batch_preview_enabled(True)

    module.MainPresenter._update_settings_window_state(presenter)
    assert presenter.settings_state.additional_augmentation is True
    assert presenter.settings_state.shuffle is True
    assert presenter.settings_state.shuffle_patches_in_frame is False
    assert presenter.settings_state.random_crop is False
    assert presenter.settings_state.crops_per_image == 11
    assert presenter.settings_state.tech_aug['enabled'] is True
    assert presenter.settings_state.tech_aug.get('min_operations', 1) == 1
    assert presenter.settings_state.tech_aug['max_operations'] == 2
    assert presenter.settings_state.tech_aug.get('debug_return_pair', False) is False
    assert presenter.settings_state.tech_aug['max_changed_pixels_ratio'] == pytest.approx(0.16)
    assert presenter.settings_state.tech_aug['max_foreground_ratio_delta'] == pytest.approx(0.08)
    assert presenter.settings_state.tech_aug['global_width']['probability'] == pytest.approx(0.55)
    assert presenter.settings_state.tech_aug['global_width']['kernel_size_range'] == [1, 2]
    assert presenter.settings_state.tech_aug['scale_rethreshold']['probability'] == pytest.approx(0.45)
    assert presenter.settings_state.tech_aug['blur_threshold']['probability'] == pytest.approx(0.25)
    assert presenter.settings_state.tech_aug['boundary_aware']['probability'] == pytest.approx(0.65)
    assert presenter.settings_state.tech_aug['boundary_aware']['band_width_range'] == [1, 2]
    assert presenter.settings_state.tech_aug['local_morphology']['probability'] == pytest.approx(0.3)
    assert presenter.settings_state.tech_aug['gap_variation']['probability'] == pytest.approx(0.2)
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
    assert presenter.settings_state.hard_mining_strength == pytest.approx(3.4)
    assert presenter.settings_state.hard_mining_ema_alpha == pytest.approx(0.25)
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
    assert presenter.settings_state.pcb_defects['enabled'] is True
    assert presenter.settings_state.pcb_defects['defect_probability'] == pytest.approx(0.45)
    assert presenter.settings_state.pcb_defects['min_defects'] == 2
    assert presenter.settings_state.pcb_defects['max_defects'] == 2
    assert presenter.settings_state.pcb_defects.get('use_input_mask', True) is True
    assert presenter.settings_state.pcb_defects['use_defect_mask_as_label'] is False
    assert presenter.settings_state.pcb_defects['max_attempts_per_defect'] == 11
    assert presenter.settings_state.pcb_defects['defect_probabilities']['break'] == pytest.approx(2.0)
    assert presenter.settings_state.pcb_defects['defect_probabilities']['via'] == pytest.approx(1.1)
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
    assert presenter.settings_state.recognition_postprocess is True
    assert presenter.settings_state.recognition_postprocess_kernel_size == 7
    assert presenter.settings_state.torch_compile_enabled is True
    assert presenter.settings_state.early_stopping_enabled is False
    assert presenter.settings_state.early_stopping_patience == 2
    assert presenter.settings_state.early_stopping_min_delta == pytest.approx(0.002)
    assert presenter.settings_state.early_stopping_restore_best_weights is True
    assert presenter.settings_state.show_batch_preview is True
