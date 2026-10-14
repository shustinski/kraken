"""test_settings_optimizer_ui: test settings panel toggles."""
from __future__ import annotations
from neuralimage.lib.data_interfaces import WorkMode
from neuralimage.view.settings_panel import SettingsPanel

def test_settings_panel_switches_between_split_and_external_validation_controls(qapp):
    panel = SettingsPanel()
    panel.expert_groupbox.setChecked(True)

    panel.validation_check_box.setChecked(True)
    panel.set_validation_source_value('split')
    assert panel.validation_mode_combo.isHidden() is False
    assert panel.validation_spinbox.isHidden() is False
    assert panel.validation_image_path_label.isHidden() is True
    assert panel.validation_label_path_label.isHidden() is True
    assert panel._field_rows[panel.validation_spinbox].isEnabled() is True
    assert panel._field_rows[panel.validation_image_path_label].isEnabled() is False
    assert panel._field_rows[panel.validation_label_path_label].isEnabled() is False

    panel.set_validation_source_value('external')
    assert panel.validation_mode_combo.isHidden() is False
    assert panel.validation_spinbox.isHidden() is True
    assert panel.validation_image_path_label.isHidden() is False
    assert panel.validation_label_path_label.isHidden() is False
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

    assert panel.additional_augmentation_check_box.title() == 'Фотометрическая аугументация'
    assert panel.spatial_groupbox.title() == 'Пространственные аугментации'
    assert panel.flip_x.text()
    assert panel.flip_y.text()
    assert panel._field_rows[panel.shift_spinbox].parentWidget() is panel.spatial_sampling_row_widget
    assert panel._field_rows[panel.crops_per_image_spinbox].parentWidget() is panel.spatial_sampling_row_widget

def test_settings_panel_places_mixed_precision_in_runtime_and_rare_patch_on_training_tab(qapp):
    panel = SettingsPanel()
    panel.expert_mode_check_box.setChecked(False)

    train_batch_row = panel._field_rows[panel.train_batch_spinbox]
    mixed_precision_row = panel._field_rows[panel.mixed_precision_type]
    log_update_frequency_row = panel._field_rows[panel.log_update_frequency_spinbox]
    rare_patch_factor_row = panel._field_rows[panel.rare_patch_oversampling_factor_spinbox]

    assert panel.runtime_groupbox.isAncestorOf(train_batch_row) is True
    assert panel.runtime_groupbox.isAncestorOf(mixed_precision_row) is True
    assert panel.precision_loss_groupbox.isAncestorOf(mixed_precision_row) is False
    assert panel.runtime_groupbox.isAncestorOf(log_update_frequency_row) is True
    assert panel.optimizer_groupbox.isAncestorOf(log_update_frequency_row) is False
    assert panel._settings_cards['sampling'].isAncestorOf(panel.rare_patch_groupbox)
    assert panel.expert_mode_check_box.isChecked() is False
    assert panel.model_variants_groupbox.isHidden() is True
    panel.expert_groupbox.setChecked(True)
    assert panel.model_variants_groupbox.isHidden() is False
    assert panel.shuffle_groupbox.isAncestorOf(panel.shuffle_frames_check_box) is True
    assert panel.shuffle_groupbox.isAncestorOf(panel.shuffle_patches_in_frame_check_box) is True
    assert panel.model_variants_groupbox.isAncestorOf(panel.deprecated_model_type) is True
    assert panel.model_variants_groupbox.isAncestorOf(panel.experimental_model_type) is True
    assert panel.rare_patch_groupbox.isAncestorOf(panel.rare_patch_oversampling_check_box) is True
    assert panel.rare_patch_groupbox.isAncestorOf(rare_patch_factor_row) is True
    assert panel.runtime_groupbox.isAncestorOf(panel.rare_patch_oversampling_check_box) is False
    assert panel.general_groupbox.isAncestorOf(panel.shuffle_frames_check_box) is False
    assert panel.general_groupbox.isAncestorOf(panel.deprecated_model_type) is False
    assert panel._settings_cards['schedule'].isAncestorOf(panel.warmup_groupbox)

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
