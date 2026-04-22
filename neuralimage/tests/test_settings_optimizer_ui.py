import importlib
import sys
import types

import numpy as np
import pytest

pytest.importorskip('PyQt6')

from PIL import Image
from PyQt6.QtWidgets import QApplication

from lib.data_interfaces import WorkMode, build_synthetic_defect_generator_parameters
from model.NeuralNetwork.registrator import ModelType
from tests.helpers import make_test_dir
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
    nn_stub.get_registered_model_names_by_type = lambda: {}
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
    panel.mixed_precision_type.setCurrentText('off')
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
        'pcb_break_severity_tip',
        'pcb_short_severity_tip',
        'pcb_missing_copper_severity_tip',
        'pcb_excess_copper_severity_tip',
        'pcb_pinhole_severity_tip',
        'pcb_spurious_copper_severity_tip',
        'pcb_via_severity_tip',
        'pcb_misalignment_severity_tip',
        'loss_function',
        'sample_size_tip',
        'shift_tip',
        'augmentation_preview_button',
        'augmentation_preview_button_tip',
    }

    for path in ('resources/ui_texts_ru.json', 'resources/ui_texts_en.json'):
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        settings_panel = data.get('settings_panel', {})
        missing = sorted(required_keys - set(settings_panel))
        assert not missing, f'{path} missing keys: {missing}'


def test_settings_panel_optimizer_presets_apply_values_and_highlight_active(qapp):
    panel = SettingsPanel()
    panel.connect_internal_signals()

    adam_btn, adamw_btn = panel.optimizer_preset_buttons
    assert len(panel.optimizer_preset_buttons) == 2
    assert panel.optimizer_advanced_content_widget.isHidden() is True
    assert panel.loss_advanced_content_widget.isHidden() is True

    adamw_btn.click()
    assert panel.optimizer_type.currentText() == 'adamw'
    assert panel.learning_rate_spinbox.value() == pytest.approx(0.0005)
    assert panel.weight_decay_spinbox.value() == pytest.approx(0.01)
    assert adamw_btn.property('selectionRole') == 'mode'
    assert adamw_btn.isChecked() is True
    assert adam_btn.isChecked() is False

    panel.learning_rate_spinbox.setValue(0.00031)
    assert adam_btn.isChecked() is False
    assert adamw_btn.isChecked() is False


def test_settings_panel_toggles_validation_spinbox(qapp):
    panel = SettingsPanel()
    panel.expert_groupbox.setChecked(True)

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
    panel.expert_groupbox.setChecked(True)

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
    panel.expert_groupbox.setChecked(True)

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


def test_settings_panel_toggles_augmentation_groupboxes(qapp):
    panel = SettingsPanel()
    panel.sync_business_logic_controls(WorkMode.train_only.value)
    panel.expert_groupbox.setChecked(True)

    assert panel.spatial_groupbox.isHidden() is False

    panel.additional_augmentation_check_box.setChecked(False)
    panel._sync_augmentation_controls(panel.additional_augmentation_check_box.isChecked())
    assert panel.photometric_groupbox.isHidden() is False
    assert panel._field_rows[panel.augmentation_brightness_spinbox].isEnabled() is False

    panel.additional_augmentation_check_box.setChecked(True)
    panel._sync_augmentation_controls(panel.additional_augmentation_check_box.isChecked())
    assert panel._field_rows[panel.augmentation_brightness_spinbox].isEnabled() is True

    panel.cutout_check_box.setChecked(False)
    panel.random_artifacts_check_box.setChecked(False)
    panel.mixup_check_box.setChecked(False)
    panel._sync_training_augmentation_controls()
    assert panel.cutout_groupbox.isHidden() is False
    assert panel.random_artifacts_groupbox.isHidden() is False
    assert panel.mixup_groupbox.isHidden() is False
    assert panel._field_rows[panel.cutout_probability_spinbox].isEnabled() is False
    assert panel._field_rows[panel.random_artifacts_probability_spinbox].isEnabled() is False
    assert panel._field_rows[panel.mixup_probability_spinbox].isEnabled() is False

    panel.cutout_check_box.setChecked(True)
    panel.random_artifacts_check_box.setChecked(True)
    panel.mixup_check_box.setChecked(True)
    panel._sync_training_augmentation_controls()
    assert panel._field_rows[panel.cutout_probability_spinbox].isEnabled() is True
    assert panel._field_rows[panel.random_artifacts_probability_spinbox].isEnabled() is True
    assert panel._field_rows[panel.mixup_probability_spinbox].isEnabled() is True


def test_settings_panel_uses_photometric_and_spatial_labels_and_shared_sampling_row(qapp):
    panel = SettingsPanel()

    assert panel.additional_augmentation_check_box.text() == 'Фотометрическая аугументация'
    assert panel.spatial_groupbox.title() == 'Пространственные аугментации'
    assert panel.flip_x.text()
    assert panel.flip_y.text()
    assert panel._field_rows[panel.shift_spinbox].parentWidget() is panel.spatial_sampling_row_widget
    assert panel._field_rows[panel.crops_per_image_spinbox].parentWidget() is panel.spatial_sampling_row_widget


def test_settings_panel_places_mixed_precision_in_runtime_and_rare_patch_on_training_tab(qapp):
    panel = SettingsPanel()

    train_batch_row = panel._field_rows[panel.train_batch_spinbox]
    mixed_precision_row = panel._field_rows[panel.mixed_precision_type]
    log_update_frequency_row = panel._field_rows[panel.log_update_frequency_spinbox]
    rare_patch_factor_row = panel._field_rows[panel.rare_patch_oversampling_factor_spinbox]

    assert panel.runtime_groupbox.isAncestorOf(train_batch_row) is True
    assert panel.runtime_groupbox.isAncestorOf(mixed_precision_row) is True
    assert panel.precision_loss_groupbox.isAncestorOf(mixed_precision_row) is False
    assert panel.runtime_groupbox.isAncestorOf(log_update_frequency_row) is True
    assert panel.optimizer_groupbox.isAncestorOf(log_update_frequency_row) is False
    assert panel.training_page_layout.indexOf(panel.rare_patch_groupbox) != -1
    assert panel.training_page_layout.indexOf(panel.expert_groupbox) != -1
    assert panel.expert_groupbox.isCheckable() is True
    assert panel.expert_groupbox.isChecked() is False
    assert panel.expert_content_widget.isHidden() is True
    panel.expert_groupbox.setChecked(True)
    assert panel.expert_content_widget.isHidden() is False
    assert panel.expert_groupbox.isAncestorOf(panel.model_variants_groupbox) is True
    assert panel.shuffle_groupbox.isAncestorOf(panel.shuffle_frames_check_box) is True
    assert panel.shuffle_groupbox.isAncestorOf(panel.shuffle_patches_in_frame_check_box) is True
    assert panel.model_variants_groupbox.isAncestorOf(panel.deprecated_model_type) is True
    assert panel.model_variants_groupbox.isAncestorOf(panel.experimental_model_type) is True
    assert panel.rare_patch_groupbox.isAncestorOf(panel.rare_patch_oversampling_check_box) is True
    assert panel.rare_patch_groupbox.isAncestorOf(rare_patch_factor_row) is True
    assert panel.runtime_groupbox.isAncestorOf(panel.rare_patch_oversampling_check_box) is False
    assert panel.general_groupbox.isAncestorOf(panel.shuffle_frames_check_box) is False
    assert panel.general_groupbox.isAncestorOf(panel.deprecated_model_type) is False
    assert panel.expert_groupbox.isAncestorOf(panel.warmup_groupbox) is True


def test_settings_panel_syncs_tech_aug_controls(qapp):
    panel = SettingsPanel()
    panel.sync_business_logic_controls(WorkMode.train_only.value)
    min_row = panel._field_rows.get(panel.tech_aug_min_operations_spinbox, panel.tech_aug_min_operations_spinbox)
    boundary_row = panel._field_rows.get(
        panel.tech_aug_boundary_aware_probability_spinbox,
        panel.tech_aug_boundary_aware_probability_spinbox,
    )

    panel.tech_augmentation_check_box.setChecked(False)
    assert panel.tech_augmentation_debug_pair_check_box.isEnabled() is False
    assert min_row.isEnabled() is False
    assert boundary_row.isEnabled() is False

    panel.tech_augmentation_check_box.setChecked(True)
    panel._sync_tech_augmentation_controls()
    assert panel.tech_augmentation_debug_pair_check_box.isEnabled() is True
    assert min_row.isEnabled() is True
    assert boundary_row.isEnabled() is True

    panel.sync_business_logic_controls(WorkMode.recognition_only.value)
    assert panel.tech_augmentation_check_box.isEnabled() is False
    assert min_row.isEnabled() is False

    panel.sync_business_logic_controls(WorkMode.train_only.value)
    assert panel.tech_augmentation_check_box.isEnabled() is True


def test_settings_panel_syncs_recognition_output_controls(qapp):
    panel = SettingsPanel()

    panel.recognition_binarize_output_check_box.setChecked(True)
    panel.recognition_use_auto_threshold_check_box.setChecked(True)
    assert panel._field_rows[panel.recognition_threshold_spinbox].isEnabled() is False
    assert panel.recognition_tta_check_box.isEnabled() is True
    assert panel._field_rows[panel.confidence_save_mode_combo].isEnabled() is True

    panel.recognition_use_auto_threshold_check_box.setChecked(False)
    assert panel._field_rows[panel.recognition_threshold_spinbox].isEnabled() is True

    panel.recognition_postprocess_check_box.setChecked(True)
    assert panel._field_rows[panel.recognition_postprocess_kernel_size_spinbox].isEnabled() is True

    panel.recognition_binarize_output_check_box.setChecked(False)
    assert panel._field_rows[panel.recognition_threshold_spinbox].isEnabled() is False
    assert panel._field_rows[panel.recognition_postprocess_kernel_size_spinbox].isEnabled() is False
    assert panel.recognition_tta_check_box.isEnabled() is True
    assert panel._field_rows[panel.confidence_save_mode_combo].isEnabled() is True


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


def test_settings_panel_applies_loss_presets_for_conductors_and_contacts(qapp):
    panel = SettingsPanel()

    panel.loss_preset_buttons['conductors'].click()
    assert panel.get_loss_term_weights() == {'bce': 0.5, 'dice': 0.5}
    assert panel.loss_preset_buttons['conductors'].isChecked() is True
    assert panel.loss_preset_buttons['contacts'].isChecked() is False

    panel.loss_preset_buttons['contacts'].click()
    assert panel.get_loss_term_weights() == {'bce': 0.5, 'focal_tversky': 0.5}
    assert panel.loss_preset_buttons['contacts'].isChecked() is True
    assert panel.loss_preset_buttons['conductors'].isChecked() is False


def test_settings_panel_uses_single_visible_label_inside_each_labeled_row(qapp):
    panel = SettingsPanel()

    cases = (
        (panel.general_form, 'epochs', panel.epochs_spinbox),
        (panel.general_form, 'sync_patch_sizes', panel.sync_patch_sizes_check_box),
        (panel.general_form, 'train_patch_size', panel.train_patch_size_widget),
        (panel.general_form, 'recognition_patch_size', panel.recognition_patch_size_widget),
        (panel.spatial_form, 'scale_augmentation_strength', panel.scale_augmentation_strength_spinbox),
        (panel.runtime_form, 'train_batch_size', panel.train_batch_spinbox),
        (panel.runtime_form, 'dataloader_num_workers', panel.dataloader_num_workers_spinbox),
        (panel.recognition_form, 'recognition_batch_size', panel.recognition_batch_spinbox),
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
    assert panel.settings_tabs.tabText(panel._page_indexes['training']) == 'Training'
    assert panel.settings_tabs.tabText(panel._page_indexes['recognition']) == 'Recognition'
    assert len(panel._page_indexes) == 2
    assert panel.expert_groupbox.title() == 'Expert settings'

    panel.set_ui_language('ru')
    assert panel.get_color_mode_value() == 'ЧБ'
    assert panel.settings_tabs.tabText(panel._page_indexes['training']) == 'Обучение'
    assert panel.settings_tabs.tabText(panel._page_indexes['recognition']) == 'Распознавание'
    assert panel.expert_groupbox.title() == 'Экспертные настройки'


def test_settings_panel_uses_localized_validation_folder_placeholders(qapp):
    panel = SettingsPanel()

    assert 'Click to choose' not in panel.validation_image_path_label.text()
    assert 'Click to choose' not in panel.validation_label_path_label.text()
    assert panel.validation_image_path_label.text().strip()
    assert panel.validation_label_path_label.text().strip()


def test_settings_panel_disables_irrelevant_controls_by_work_mode(qapp):
    panel = SettingsPanel()

    panel.sync_business_logic_controls(WorkMode.recognition_only.value)
    panel.expert_groupbox.setChecked(True)
    assert panel.sample_type_groupbox.isEnabled() is False
    assert panel.additional_augmentation_check_box.isEnabled() is False
    assert panel.tech_augmentation_check_box.isEnabled() is False
    assert panel.cutout_check_box.isEnabled() is False
    assert panel.random_artifacts_check_box.isEnabled() is False
    assert panel.mixup_check_box.isEnabled() is False
    assert panel.pcb_defects_check_box.isEnabled() is False
    assert panel.validation_check_box.isEnabled() is False
    assert panel.optimizer_presets_widget.isEnabled() is False
    assert panel._field_rows[panel.overlap_spinbox].isEnabled() is True
    assert panel._field_rows[panel.batch_spinbox].isEnabled() is False
    assert panel._field_rows[panel.nn_model_type].isEnabled() is False
    assert panel._field_rows[panel.deprecated_model_type].isEnabled() is False
    assert panel._field_rows[panel.experimental_model_type].isEnabled() is False
    assert panel.multi_gpu_check_box.isEnabled() is False
    assert panel.torch_compile_check_box.isEnabled() is True
    assert panel.edit_rare_regions_button.isEnabled() is False

    panel.sync_business_logic_controls(WorkMode.train_only.value)
    panel.expert_groupbox.setChecked(True)
    assert panel.sample_type_groupbox.isEnabled() is True
    assert panel.additional_augmentation_check_box.isEnabled() is True
    assert panel.tech_augmentation_check_box.isEnabled() is True
    assert panel.cutout_check_box.isEnabled() is True
    assert panel.random_artifacts_check_box.isEnabled() is True
    assert panel.mixup_check_box.isEnabled() is True
    assert panel.pcb_defects_check_box.isEnabled() is True
    assert panel.optimizer_presets_widget.isEnabled() is True
    assert panel._field_rows[panel.overlap_spinbox].isEnabled() is False
    assert panel._field_rows[panel.nn_model_type].isEnabled() is True
    assert panel._field_rows[panel.deprecated_model_type].isEnabled() is True
    assert panel._field_rows[panel.experimental_model_type].isEnabled() is True
    assert panel._field_rows[panel.deprecated_model_type].isEnabled() is True
    assert panel._field_rows[panel.experimental_model_type].isEnabled() is True
    assert panel.edit_rare_regions_button.isEnabled() is True

    panel.sync_business_logic_controls(WorkMode.further_training.value)
    assert panel._field_rows[panel.nn_model_type].isEnabled() is False
    assert panel._field_rows[panel.deprecated_model_type].isEnabled() is False
    assert panel._field_rows[panel.experimental_model_type].isEnabled() is False
    assert panel._field_rows[panel.overlap_spinbox].isEnabled() is True


def test_settings_panel_splits_model_groups_into_three_comboboxes(qapp):
    panel = SettingsPanel()

    panel.model_type_init(
        {
            ModelType.stable: ['EfficientUNet'],
            ModelType.deprecated: ['S 660k', 'M 720k'],
            ModelType.experimental: ['FrameUnet', 'Transformer', 'UNET++'],
        }
    )

    assert [panel.nn_model_type.itemText(i) for i in range(panel.nn_model_type.count())] == [
        '-',
        'EfficientUNet',
    ]
    assert [panel.deprecated_model_type.itemText(i) for i in range(1, panel.deprecated_model_type.count())] == [
        'S 660k',
        'M 720k',
    ]
    assert [panel.experimental_model_type.itemText(i) for i in range(1, panel.experimental_model_type.count())] == [
        'FrameUnet',
        'Transformer',
        'UNET++',
    ]

    panel.set_model('M 720k')
    assert panel.get_selected_model() == 'M 720k'
    assert panel.deprecated_model_type.currentText() == 'M 720k'
    assert panel.nn_model_type.currentIndex() == 0
    assert panel.experimental_model_type.currentIndex() == 0

    panel.set_model('FrameUnet')
    assert panel.get_selected_model() == 'FrameUnet'
    assert panel.experimental_model_type.currentText() == 'FrameUnet'
    assert panel.nn_model_type.currentIndex() == 0
    assert panel.deprecated_model_type.currentIndex() == 0

    panel.set_model('UNET++')
    assert panel.get_selected_model() == 'UNET++'
    assert panel.experimental_model_type.currentText() == 'UNET++'
    assert panel.nn_model_type.currentIndex() == 0
    assert panel.deprecated_model_type.currentIndex() == 0


def test_settings_panel_toggles_pcb_defect_controls(qapp):
    panel = SettingsPanel()
    panel.sync_business_logic_controls(WorkMode.train_only.value)
    panel.expert_groupbox.setChecked(True)

    panel.synthetic_defect_generator_check_box.setChecked(False)
    panel._sync_synthetic_defect_generator_controls()
    assert panel.synthetic_defect_generator_groupbox.isHidden() is False
    assert panel._field_rows[panel.pcb_defects_probability_spinbox].isEnabled() is False
    assert panel._field_rows[panel.pcb_defects_min_count_spinbox].isEnabled() is False
    assert panel._field_rows[panel.pcb_defect_type_checkboxes['break']].isEnabled() is False
    assert panel._field_rows[panel.pcb_defect_type_spinboxes['break']].isEnabled() is False

    panel.synthetic_defect_generator_check_box.setChecked(True)
    panel._sync_synthetic_defect_generator_controls()
    assert panel.synthetic_defect_generator_groupbox.isHidden() is False
    assert panel._field_rows[panel.pcb_defects_probability_spinbox].isEnabled() is True
    assert panel._field_rows[panel.pcb_defects_max_count_spinbox].isEnabled() is True
    assert panel._field_rows[panel.pcb_defect_type_checkboxes['via']].isEnabled() is True
    assert panel._field_rows[panel.pcb_defect_type_spinboxes['via']].isEnabled() is True

    panel.pcb_defect_type_checkboxes['via'].setChecked(False)
    panel._sync_synthetic_defect_generator_controls()
    assert panel._field_rows[panel.pcb_defect_type_spinboxes['via']].isEnabled() is False


def test_settings_panel_shows_ic_synthetic_defect_controls(qapp):
    panel = SettingsPanel()
    panel.sync_business_logic_controls(WorkMode.train_only.value)
    panel.expert_groupbox.setChecked(True)
    panel.synthetic_defect_generator_check_box.setChecked(True)
    panel.set_synthetic_topology_domain_value('ic')
    panel._sync_synthetic_domain_controls()
    panel._sync_synthetic_defect_generator_controls()

    assert panel._field_rows[panel.ic_defect_type_checkboxes['line_break']].isHidden() is False
    assert panel._field_rows[panel.ic_defect_type_spinboxes['line_break']].isHidden() is False
    assert panel._field_rows[panel.pcb_defect_type_checkboxes['break']].isHidden() is True
    assert panel._field_rows[panel.ic_defect_type_checkboxes['line_break']].isEnabled() is True
    assert panel._field_rows[panel.ic_defect_type_spinboxes['line_break']].isEnabled() is True


def test_settings_panel_saves_ic_synthetic_defect_toggles(qapp):
    panel = SettingsPanel()
    panel.sync_business_logic_controls(WorkMode.train_only.value)
    panel.synthetic_defect_generator_check_box.setChecked(True)
    panel.set_synthetic_topology_domain_value('ic')
    panel.ic_defect_type_checkboxes['bridge'].setChecked(False)
    panel.ic_defect_type_checkboxes['via_open'].setChecked(True)
    panel.ic_defect_type_spinboxes['via_open'].setValue(90)

    config = build_synthetic_defect_generator_parameters(panel.get_synthetic_defect_generator_config())

    assert config.topology_domain == 'ic'
    assert config.ic_defects.defect_probabilities['bridge'] == pytest.approx(0.0)
    assert config.ic_defects.defect_probabilities['via_open'] == pytest.approx(1.0)
    assert config.ic_defects.defect_severities['via_open'] == pytest.approx(0.9)


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


def test_settings_panel_emits_augmentation_preview_requested(qapp):
    panel = SettingsPanel()
    panel.connect_internal_signals()
    panel.expert_groupbox.setChecked(True)
    calls = {'count': 0}
    panel.augmentation_preview_requested.connect(lambda: calls.__setitem__('count', calls['count'] + 1))

    panel.augmentation_preview_button.click()

    assert calls['count'] == 1


def test_main_presenter_opens_augmentation_preview_dialog(qapp):
    module = _import_main_presenter_with_stubs()
    presenter = module.MainPresenter.__new__(module.MainPresenter)

    sample_root = make_test_dir('augmentation_preview_presenter')
    sample_dir = sample_root / 'samples'
    label_dir = sample_root / 'labels'
    sample_dir.mkdir()
    label_dir.mkdir()

    image = np.zeros((32, 32), dtype=np.uint8)
    image[4:20, 6:12] = 255
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:24, 10:22] = 255
    Image.fromarray(image, mode='L').save(sample_dir / 'frame.png')
    Image.fromarray(mask, mode='L').save(label_dir / 'frame.png')

    opened: dict[str, object] = {}

    class _FakeDestroyed:
        def connect(self, _callback):
            opened['connected'] = True

    class _FakeSignal:
        def connect(self, _callback):
            opened['apply_connected'] = True

    class _FakeDialog:
        def __init__(self, training_parameters, parent):
            opened['training_parameters'] = training_parameters
            opened['parent'] = parent
            self.destroyed = _FakeDestroyed()
            self.apply_to_main_requested = _FakeSignal()

        def show(self):
            opened['show_called'] = True

        def raise_(self):
            opened['raise_called'] = True

        def activateWindow(self):
            opened['activate_called'] = True

    sys.modules['view.augmentation_preview_dialog'] = types.SimpleNamespace(
        AugmentationPreviewDialog=_FakeDialog
    )

    presenter.main_window_state = module.MainWindowState(
        work_mode=WorkMode.train_only.value,
        sample_folder=str(sample_dir),
        label_folder=str(label_dir),
        epochs=1,
    )
    presenter.settings_state = SettingsState()
    presenter.view = types.SimpleNamespace(
        show_warning=types.SimpleNamespace(emit=lambda message: opened.__setitem__('warning', message))
    )
    presenter._augmentation_preview_dialog = None
    presenter._update_main_window_state = lambda: None
    presenter._update_settings_window_state = lambda: None

    module.MainPresenter._open_augmentation_preview(presenter)

    training_parameters = opened['training_parameters']
    assert training_parameters.image_path == sample_dir
    assert training_parameters.label_path == label_dir
    assert opened.get('warning') is None
    assert opened.get('apply_connected') is True
    assert opened.get('show_called') is True
    assert opened.get('raise_called') is True
    assert opened.get('activate_called') is True
    sys.modules.pop('view.augmentation_preview_dialog', None)


def test_main_presenter_converts_cif_masks_for_augmentation_preview(qapp, monkeypatch):
    module = _import_main_presenter_with_stubs()
    presenter = module.MainPresenter.__new__(module.MainPresenter)

    sample_root = make_test_dir('augmentation_preview_presenter_cif')
    sample_dir = sample_root / 'samples'
    label_dir = sample_root / 'labels'
    raster_dir = sample_root / 'binary_cif'
    sample_dir.mkdir()
    label_dir.mkdir()
    raster_dir.mkdir()

    image = np.zeros((32, 32), dtype=np.uint8)
    image[6:22, 8:18] = 255
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[10:24, 12:26] = 255
    Image.fromarray(image, mode='L').save(sample_dir / 'frame.png')
    Image.fromarray(mask, mode='L').save(raster_dir / 'frame.png')
    (label_dir / 'frame.cif').write_text('dummy', encoding='utf-8')

    opened: dict[str, object] = {}

    class _FakeDestroyed:
        def connect(self, _callback):
            opened['connected'] = True

    class _FakeSignal:
        def connect(self, _callback):
            opened['apply_connected'] = True

    class _FakeDialog:
        def __init__(self, training_parameters, parent):
            opened['training_parameters'] = training_parameters
            opened['parent'] = parent
            self.destroyed = _FakeDestroyed()
            self.apply_to_main_requested = _FakeSignal()

        def show(self):
            opened['show_called'] = True

        def raise_(self):
            opened['raise_called'] = True

        def activateWindow(self):
            opened['activate_called'] = True

    sys.modules['view.augmentation_preview_dialog'] = types.SimpleNamespace(
        AugmentationPreviewDialog=_FakeDialog
    )

    rare_patch_module = types.ModuleType('lib.rare_patch_masks')

    def _prepare_label_folder_for_rare_patch_editor(folder, log_callback=None):
        opened['prepared_folder'] = folder
        return raster_dir, None

    rare_patch_module.prepare_label_folder_for_rare_patch_editor = _prepare_label_folder_for_rare_patch_editor
    original_rare_patch_module = sys.modules.get('lib.rare_patch_masks')
    sys.modules['lib.rare_patch_masks'] = rare_patch_module
    monkeypatch.setattr(module, 'filter_files', lambda folder, suffixes: [folder / 'frame.cif'])

    presenter.main_window_state = module.MainWindowState(
        work_mode=WorkMode.train_only.value,
        sample_folder=str(sample_dir),
        label_folder=str(label_dir),
        epochs=1,
    )
    presenter.settings_state = SettingsState()
    presenter.view = types.SimpleNamespace(
        show_warning=types.SimpleNamespace(emit=lambda message: opened.__setitem__('warning', message))
    )
    presenter._augmentation_preview_dialog = None
    presenter._update_main_window_state = lambda: None
    presenter._update_settings_window_state = lambda: None

    try:
        module.MainPresenter._open_augmentation_preview(presenter)
    finally:
        sys.modules.pop('view.augmentation_preview_dialog', None)
        if original_rare_patch_module is None:
            sys.modules.pop('lib.rare_patch_masks', None)
        else:
            sys.modules['lib.rare_patch_masks'] = original_rare_patch_module

    training_parameters = opened['training_parameters']
    assert opened['prepared_folder'] == label_dir
    assert training_parameters.image_path == sample_dir
    assert training_parameters.label_path == raster_dir
    assert opened.get('warning') is None
    assert opened.get('apply_connected') is True
    assert opened.get('show_called') is True


def test_main_presenter_applies_augmentation_preview_payload_to_settings_panel(qapp):
    module = _import_main_presenter_with_stubs()
    presenter = module.MainPresenter.__new__(module.MainPresenter)
    presenter.settings_panel = SettingsPanel()
    presenter.settings_panel.connect_internal_signals()
    presenter.settings_state = SettingsState()
    presenter.view = types.SimpleNamespace(is_batch_preview_enabled=lambda: False)

    payload = {
        'horizontal_rotation': True,
        'vertical_rotation': False,
        'flip_x': True,
        'flip_y': False,
        'random_crop': True,
        'crops_per_image': 23,
        'scale_augmentation': True,
        'scale_augmentation_strength': 0.37,
        'additional_augmentation': True,
        'augmentation_brightness_strength': 0.21,
        'augmentation_contrast_strength': 0.0,
        'augmentation_gamma_strength': 0.18,
        'augmentation_noise_probability': 0.44,
        'augmentation_noise_sigma': 0.013,
        'augmentation_blur_probability': 0.0,
        'augmentation_blur_radius': 0.0,
        'synthetic_defect_generator': {
            'enabled': True,
            'epoch_size_factor': 1.8,
            'trace_count_range': [8, 10],
            'segment_count_range': [3, 7],
            'trace_half_width_range': [2, 5],
            'background_noise_sigma_range': [0.01, 0.04],
            'trace_noise_sigma_range': [0.02, 0.05],
            'defects': {
                'enabled': True,
                'defect_probability': 0.52,
                'min_defects': 2,
                'max_defects': 5,
                'max_attempts_per_defect': 13,
                'use_input_mask': False,
                'use_defect_mask_as_label': False,
                'defect_probabilities': {
                    'break': 1.0,
                    'short': 0.0,
                    'missing_copper': 1.0,
                    'excess_copper': 0.0,
                    'pinhole': 1.0,
                    'spurious_copper': 0.0,
                    'via': 1.0,
                    'misalignment': 1.0,
                },
                'defect_severities': {
                    'break': 0.7,
                    'short': 0.0,
                    'missing_copper': 0.4,
                    'excess_copper': 0.0,
                    'pinhole': 0.6,
                    'spurious_copper': 0.0,
                    'via': 0.85,
                    'misalignment': 0.2,
                },
            },
        },
        'cutout_enabled': True,
        'cutout_probability': 0.81,
        'cutout_holes': 3,
        'cutout_size_ratio': 0.29,
        'random_artifacts_enabled': True,
        'random_artifacts_probability': 0.63,
        'random_artifacts_count': 2,
        'random_artifacts_size_ratio': 0.18,
        'random_artifacts_dust_enabled': True,
        'random_artifacts_resist_residue_enabled': False,
        'random_artifacts_etch_residue_enabled': True,
        'random_artifacts_particle_cluster_enabled': False,
        'random_artifacts_flake_enabled': True,
        'mixup_enabled': True,
        'mixup_probability': 0.58,
        'mixup_alpha': 0.47,
    }

    module.MainPresenter._apply_augmentation_preview_settings(presenter, payload)

    panel = presenter.settings_panel
    assert panel.horizontal_rotation.isChecked() is True
    assert panel.vertical_rotation.isChecked() is False
    assert panel.flip_x.isChecked() is True
    assert panel.flip_y.isChecked() is False
    assert panel.random_crop_check_box.isChecked() is True
    assert panel.crops_per_image_spinbox.value() == 23
    assert panel.scale_augmentation_check_box.isChecked() is True
    assert panel.scale_augmentation_strength_spinbox.value() == pytest.approx(0.37)
    assert panel.additional_augmentation_check_box.isChecked() is True
    assert panel.augmentation_brightness_spinbox.value() == pytest.approx(0.21)
    assert panel.augmentation_contrast_spinbox.value() == pytest.approx(0.0)
    assert panel.augmentation_gamma_spinbox.value() == pytest.approx(0.18)
    assert panel.augmentation_noise_probability_spinbox.value() == pytest.approx(0.44)
    assert panel.augmentation_noise_sigma_spinbox.value() == pytest.approx(0.013)
    assert panel.augmentation_blur_probability_spinbox.value() == pytest.approx(0.0)
    assert panel.cutout_check_box.isChecked() is True
    assert panel.cutout_holes_spinbox.value() == 3
    assert panel.random_artifacts_check_box.isChecked() is True
    assert panel.random_artifact_type_checkboxes['resist_residue'].isChecked() is False
    assert panel.random_artifact_type_checkboxes['flake'].isChecked() is True
    assert panel.mixup_check_box.isChecked() is True
    assert panel.mixup_alpha_spinbox.value() == pytest.approx(0.47)
    assert panel.synthetic_defect_generator_check_box.isChecked() is True
    assert panel.synthetic_dataset_factor_spinbox.value() == pytest.approx(1.8)
    assert panel.synthetic_trace_count_min_spinbox.value() == 8
    assert panel.synthetic_trace_count_max_spinbox.value() == 10
    assert panel.synthetic_segment_count_min_spinbox.value() == 3
    assert panel.synthetic_segment_count_max_spinbox.value() == 7
    assert panel.synthetic_trace_half_width_min_spinbox.value() == 2
    assert panel.synthetic_trace_half_width_max_spinbox.value() == 5
    assert panel.pcb_defects_probability_spinbox.value() == pytest.approx(0.52)
    assert panel.pcb_defect_type_checkboxes['short'].isChecked() is False
    assert panel.pcb_defect_type_spinboxes['short'].value() == 0
    assert panel.pcb_defect_type_checkboxes['via'].isChecked() is True
    assert panel.pcb_defect_type_spinboxes['via'].value() == 85

    synthetic_generator = build_synthetic_defect_generator_parameters(panel.get_synthetic_defect_generator_config())
    assert synthetic_generator.enabled is True
    assert synthetic_generator.epoch_size_factor == pytest.approx(1.8)
    assert synthetic_generator.trace_count_range == (8, 10)
    assert synthetic_generator.segment_count_range == (3, 7)
    assert synthetic_generator.trace_half_width_range == (2, 5)
    assert synthetic_generator.defects.defect_probability == pytest.approx(0.52)
    assert synthetic_generator.defects.max_attempts_per_defect == 13
    assert synthetic_generator.defects.use_input_mask is True
    assert synthetic_generator.defects.defect_probabilities['break'] == pytest.approx(1.0)
    assert synthetic_generator.defects.defect_probabilities['short'] == pytest.approx(0.0)
    assert synthetic_generator.defects.defect_severities['break'] == pytest.approx(0.7)
    assert synthetic_generator.defects.defect_severities['via'] == pytest.approx(0.85)

    assert presenter.settings_state.random_crop is True
    assert presenter.settings_state.flip_x is True
    assert presenter.settings_state.flip_y is False
    assert presenter.settings_state.crops_per_image == 23
    assert presenter.settings_state.scale_augmentation is True
    assert presenter.settings_state.additional_augmentation is True
    state_synthetic_generator = build_synthetic_defect_generator_parameters(
        presenter.settings_state.synthetic_defect_generator
    )
    assert state_synthetic_generator.defects.defect_probabilities['via'] == pytest.approx(1.0)
    assert state_synthetic_generator.defects.defect_severities['via'] == pytest.approx(0.85)


def test_main_presenter_updates_main_view_sample_count_when_calculated(qapp):
    module = _import_main_presenter_with_stubs()
    presenter = module.MainPresenter.__new__(module.MainPresenter)
    presenter._sample_count_worker_thread = None
    presenter._sample_count_cache_path = None
    presenter._sample_count_cache_sizes = None
    presenter._latest_sample_count_request_id = 7
    presenter._start_pending_sample_count_request_if_needed = lambda: None
    presenter.settings_panel = types.SimpleNamespace(set_samples_count=lambda value: setattr(presenter, '_panel_count', value))
    presenter.view = types.SimpleNamespace(set_samples_count=lambda value: setattr(presenter, '_view_count', value))

    module.MainPresenter._on_sample_count_calculated(presenter, 7, 'D:/sample', [], 123)

    assert presenter._panel_count == 123
    assert presenter._view_count == 123


def test_main_presenter_sample_count_includes_synthetic_frames(qapp):
    module = _import_main_presenter_with_stubs()
    presenter = module.MainPresenter.__new__(module.MainPresenter)
    root = make_test_dir('presenter_sample_count_with_synthetic')
    sample_dir = root / 'samples'
    sample_dir.mkdir()

    class _Signals:
        def __init__(self):
            self.payload = None
            self.calculated = types.SimpleNamespace(
                emit=lambda request_id, normalized_path, image_sizes, total_samples: setattr(
                    self,
                    'payload',
                    (request_id, normalized_path, image_sizes, total_samples),
                )
            )
            self.failed = types.SimpleNamespace(emit=lambda *_args: (_ for _ in ()).throw(AssertionError('unexpected failure')))

    class _SampleWorker:
        @staticmethod
        def collect_image_paths(path):
            return [path / 'frame_0.png', path / 'frame_1.png']

        @staticmethod
        def collect_image_sizes(_image_paths):
            return [(512, 512), (512, 512)]

        @staticmethod
        def calculate_total_samples(image_sizes, params):
            return sum(_SampleWorker.calculate_image_parts_for_settings(size, params) for size in image_sizes)

        @staticmethod
        def calculate_image_parts_for_settings(image_size, params):
            height, width = image_size
            return ((width - params.x_size) // params.step + 1) * ((height - params.y_size) // params.step + 1)

    presenter._sample_count_signals = _Signals()
    module.SampleWorker = _SampleWorker

    calculator_settings = types.SimpleNamespace(
        step=128,
        x_size=256,
        y_size=256,
        vertical_rotation=False,
        horizontal_rotation=False,
        flip_x=False,
        flip_y=False,
        additional_augmentation=False,
        scale_augmentation=False,
        random_crop=False,
        crops_per_image=64,
    )
    synthetic_config = {
        'enabled': True,
        'epoch_size_factor': 1.0,
        'image_size_xy': [1024, 1024],
    }

    module.MainPresenter._run_sample_count_request(
        presenter,
        3,
        str(sample_dir),
        calculator_settings,
        synthetic_config,
        None,
        None,
    )

    assert presenter._sample_count_signals.payload is not None
    assert presenter._sample_count_signals.payload[3] == 116


def test_main_presenter_logs_background_sample_indexing_start(qapp, monkeypatch):
    module = _import_main_presenter_with_stubs()
    presenter = module.MainPresenter.__new__(module.MainPresenter)
    presenter._sample_count_cache_path = None
    presenter._sample_count_cache_sizes = None

    logged_messages: list[str] = []
    presenter._publish_log_message = logged_messages.append

    captured: dict[str, object] = {}

    class _FakeThread:
        def __init__(self, *, target, args, daemon, name):
            captured['target'] = target
            captured['args'] = args
            captured['daemon'] = daemon
            captured['name'] = name

        def start(self):
            captured['started'] = True

    monkeypatch.setattr(module.threading, 'Thread', _FakeThread)

    module.MainPresenter._start_sample_count_request(
        presenter,
        (5, 'D:/sample_folder', types.SimpleNamespace(), None),
    )

    assert presenter._sample_count_worker_thread is not None
    assert captured['started'] is True
    assert any('отдельном потоке' in message for message in logged_messages)


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
    assert presenter.settings_state.early_stopping_patience == 2
    assert presenter.settings_state.early_stopping_min_delta == pytest.approx(0.002)
    assert presenter.settings_state.early_stopping_restore_best_weights is True
    assert presenter.settings_state.show_batch_preview is True
