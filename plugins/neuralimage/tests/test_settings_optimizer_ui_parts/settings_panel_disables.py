"""test_settings_optimizer_ui: test settings panel disables."""
from __future__ import annotations
import pytest
from neuralimage.lib.data_interfaces import WorkMode
from neuralimage.model.NeuralNetwork.registrator import ModelType
from neuralimage.view.settings_panel import SettingsPanel

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
    assert panel.loss_preset_buttons['conductors'].property('presetRole') == 'quickPreset'
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
        (None, 'scale_augmentation_strength', panel.scale_augmentation_strength_spinbox),
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
        if form is not None:
            assert form.labelForField(row_widget) is None

def test_settings_panel_keeps_color_mode_value_stable_when_language_changes(qapp):
    panel = SettingsPanel()
    panel.set_color_mode_value('ЧБ')

    panel.set_ui_language('en')
    assert panel.get_color_mode_value() == 'ЧБ'
    assert panel.settings_tabs.tabText(panel._page_indexes['data']) == 'Data'
    assert panel.settings_tabs.tabText(panel._page_indexes['training']) == 'Model and training'
    assert panel.settings_tabs.tabText(panel._page_indexes['recognition']) == 'Recognition'
    assert len(panel._page_indexes) == 3
    assert panel.expert_mode_check_box.text() == 'Expert mode'

    panel.set_ui_language('ru')
    assert panel.get_color_mode_value() == 'ЧБ'
    assert panel.settings_tabs.tabText(panel._page_indexes['data']) == 'Данные'
    assert panel.settings_tabs.tabText(panel._page_indexes['training']) == 'Модель и обучение'
    assert panel.settings_tabs.tabText(panel._page_indexes['recognition']) == 'Распознавание'
    assert panel.expert_mode_check_box.text() == 'Экспертный режим'

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
    assert not hasattr(panel, 'sample_type_groupbox')
    assert panel.preprocessing_groupbox.isEnabled() is False
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
    assert panel.torch_compile_check_box.isEnabled() is False
    assert panel.edit_rare_regions_button.isEnabled() is False

    panel.sync_business_logic_controls(WorkMode.train_only.value)
    panel.expert_groupbox.setChecked(True)
    assert panel.preprocessing_groupbox.isEnabled() is True
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
