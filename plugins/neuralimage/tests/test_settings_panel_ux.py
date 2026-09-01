from __future__ import annotations

import os

import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
pytest.importorskip('PyQt6')

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from neuralimage.lib.data_interfaces import WorkMode
from neuralimage.view.settings_panel import SettingsPanel


@pytest.fixture(scope='module')
def qapp():
    return QApplication.instance() or QApplication([])


def test_settings_panel_has_three_responsibility_pages(qapp):
    panel = SettingsPanel()

    assert panel.settings_tabs.count() == 3
    assert tuple(panel._page_indexes) == ('data', 'training', 'recognition')
    assert not hasattr(panel, 'sem_segmentation_preset_combo')
    assert not hasattr(panel, 'sem_segmentation_validate_button')
    assert set(panel._settings_cards) == {
        'data_source', 'preprocessing', 'augmentation', 'sampling', 'validation_data',
        'architecture', 'training', 'supervision', 'losses', 'schedule',
        'confidence_training', 'validation_metrics', 'runtime', 'recognition',
        'inference_uncertainty', 'active_learning',
    }


def test_work_mode_hides_whole_irrelevant_pages(qapp):
    panel = SettingsPanel()
    panel.sync_business_logic_controls(WorkMode.recognition_only.value)

    assert not panel.settings_tabs.isTabVisible(panel._page_indexes['data'])
    assert not panel.settings_tabs.isTabVisible(panel._page_indexes['training'])
    assert panel.settings_tabs.isTabVisible(panel._page_indexes['recognition'])

    panel.sync_business_logic_controls(WorkMode.train_only.value)
    assert panel.settings_tabs.isTabVisible(panel._page_indexes['data'])
    assert panel.settings_tabs.isTabVisible(panel._page_indexes['training'])
    assert not panel.settings_tabs.isTabVisible(panel._page_indexes['recognition'])


def test_augmentation_editor_is_four_column_hierarchy(qapp):
    panel = SettingsPanel()
    editor = panel.training_augmentation_editor

    assert editor.tree.columnCount() == 4
    assert editor.tree.topLevelItemCount() == 3
    assert [editor.tree.topLevelItem(index).text(1) for index in range(3)] == [
        'Пространственные', 'Фотометрические', 'SEM-аугментации'
    ]
    assert editor.tree.itemWidget(editor.row_items['brightness'], 2) is panel.augmentation_brightness_probability_spinbox
    assert editor.tree.itemWidget(editor.row_items['brightness'], 3) is panel.augmentation_brightness_spinbox


def test_augmentation_block_and_defaults_control_real_widgets(qapp):
    panel = SettingsPanel()
    editor = panel.training_augmentation_editor
    spatial = editor.block_items['spatial']
    spatial.setCheckState(0, Qt.CheckState.Unchecked)

    assert not panel.horizontal_rotation.isChecked()
    assert not panel.vertical_rotation.isChecked()
    panel.augmentation_brightness_probability_spinbox.setValue(0.17)
    editor.defaults_button.click()
    assert panel.augmentation_brightness_probability_spinbox.value() == pytest.approx(1.0)
    assert panel.horizontal_rotation.isChecked()


def test_patch_cutting_mode_uses_exclusive_radios(qapp):
    panel = SettingsPanel()
    panel.random_sampling_radio.setChecked(True)
    assert panel.random_crop_check_box.isChecked()
    assert panel.crops_per_image_spinbox.isEnabled()
    assert not panel.shift_spinbox.isEnabled()

    panel.grid_sampling_radio.setChecked(True)
    assert not panel.random_crop_check_box.isChecked()
    assert panel.shift_spinbox.isEnabled()
    assert not panel.crops_per_image_spinbox.isEnabled()


def test_expert_mode_hides_in_place_and_preserves_values(qapp):
    panel = SettingsPanel()
    panel.expert_mode_check_box.setChecked(False)
    panel.learning_rate_spinbox.setValue(0.0123)
    assert panel.model_variants_groupbox.isHidden()
    assert panel._settings_cards['supervision'].isHidden()

    panel.expert_mode_check_box.setChecked(True)
    assert not panel.model_variants_groupbox.isHidden()
    assert not panel._settings_cards['supervision'].isHidden()
    assert panel.learning_rate_spinbox.value() == pytest.approx(0.0123)


def test_expert_mode_is_persisted_independently(qapp, monkeypatch, tmp_path):
    monkeypatch.setenv('NEURALIMAGE_SETTINGS_DIR', str(tmp_path))
    first = SettingsPanel()
    first.expert_mode_check_box.setChecked(True)

    restored = SettingsPanel()
    assert restored.expert_mode_check_box.isChecked()


def test_card_reset_restores_only_its_own_defaults(qapp):
    panel = SettingsPanel()
    mode = panel.sem_segmentation_controls['pre_mode']
    panel.epochs_spinbox.setValue(77)
    panel.preprocessing_groupbox.setChecked(True)
    assert mode.currentData() == 'per_image_percentile'

    panel._settings_cards['preprocessing'].reset_button.click()

    assert mode.currentData() == 'none'
    assert panel.epochs_spinbox.value() == 77


def test_search_navigates_to_expert_setting(qapp):
    panel = SettingsPanel()
    panel.expert_mode_check_box.setChecked(False)

    panel.settings_search.setText('attention dimension')
    panel._activate_first_search_result()

    assert panel.expert_mode_check_box.isChecked()
    assert panel.settings_tabs.currentIndex() == panel._page_indexes['training']
    assert panel._settings_cards['architecture'].is_expanded()


def test_live_validation_marks_field_card_and_status(qapp):
    panel = SettingsPanel()
    errors = []
    panel.configuration_validity_changed.connect(lambda valid, messages: errors.append((valid, messages)))
    panel.sem_segmentation_controls['context_dim'].setValue(128)
    panel.sem_segmentation_controls['context_heads'].setValue(3)

    assert errors and errors[-1][0] is False
    assert panel.configuration_status_button.text() != 'OK'
    assert not panel.sem_segmentation_error_labels['context_dim'].isHidden()
    assert panel._settings_cards['architecture'].error_label.text()

    panel.sem_segmentation_controls['context_heads'].setValue(4)
    assert errors[-1][0] is True
    assert panel.configuration_status_button.text() == 'OK'


def test_uncertainty_method_hides_irrelevant_parameters(qapp):
    panel = SettingsPanel()
    panel.sync_business_logic_controls(WorkMode.recognition_only.value)
    panel.show_settings_page('recognition')
    panel._settings_cards['recognition'].set_expanded(False)
    panel._settings_cards['inference_uncertainty'].set_expanded(True)
    panel.show()
    qapp.processEvents()
    controls = panel.sem_segmentation_controls
    assert not panel.recognition_tta_check_box.isHidden()
    assert not panel._field_rows[panel.confidence_save_mode_combo].isVisible()
    assert controls['uncertainty_enabled'].isVisible()
    controls['uncertainty_enabled'].setChecked(True)
    controls['uncertainty_method'].setCurrentIndex(
        controls['uncertainty_method'].findData('mc_dropout')
    )

    assert not controls['uncertainty_samples'].parentWidget().isHidden()
    assert controls['uncertainty_tta_flips'].parentWidget().isHidden()

    controls['uncertainty_method'].setCurrentIndex(
        controls['uncertainty_method'].findData('tta_variance')
    )
    assert controls['uncertainty_samples'].parentWidget().isHidden()
    assert not controls['uncertainty_tta_flips'].parentWidget().isHidden()
    assert panel.is_confidence_tta_enabled()
    panel.close()


def test_mask_tta_does_not_control_uncertainty_settings(qapp):
    panel = SettingsPanel()
    panel.sync_business_logic_controls(WorkMode.recognition_only.value)
    controls = panel.sem_segmentation_controls
    controls['uncertainty_enabled'].setChecked(False)

    panel.recognition_tta_check_box.setChecked(True)

    assert panel.recognition_tta_check_box.isChecked()
    assert not controls['uncertainty_enabled'].isChecked()
    assert controls['uncertainty_method'].parentWidget().isHidden()

    controls['uncertainty_enabled'].setChecked(True)
    controls['uncertainty_method'].setCurrentIndex(
        controls['uncertainty_method'].findData('tta_variance')
    )
    controls['uncertainty_enabled'].setChecked(False)

    assert panel.recognition_tta_check_box.isChecked()


def test_legacy_confidence_state_migrates_into_the_single_uncertainty_card(qapp):
    panel = SettingsPanel()

    panel.set_confidence_output_mode('separate_grayscale', confidence_tta_enabled=True)
    panel.set_sem_segmentation_config({})

    controls = panel.sem_segmentation_controls
    assert controls['uncertainty_enabled'].isChecked()
    assert controls['uncertainty_method'].currentData() == 'tta_variance'
    assert controls['uncertainty_export'].isChecked()
    assert panel.get_confidence_save_mode_value() == 'separate_grayscale'
