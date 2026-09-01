from __future__ import annotations

import os

import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
pytest.importorskip('PyQt6')

from PyQt6.QtWidgets import QApplication

from neuralimage.view.settings_panel import SettingsPanel


def _field_is_hidden(panel: SettingsPanel, field) -> bool:
    return panel._field_rows.get(field, field).isHidden()


@pytest.fixture(scope='module')
def app():
    instance = QApplication.instance() or QApplication([])
    yield instance


def test_new_training_defaults_are_disabled(app):
    panel = SettingsPanel()

    assert panel.mixed_precision_type.currentText() == 'off'
    assert panel.multi_gpu_mode_combo.currentText() == 'off'
    assert not panel.torch_compile_check_box.isChecked()


def test_early_stopping_is_single_toggle_and_disables_epochs(app):
    panel = SettingsPanel()
    assert not hasattr(panel, 'early_stopping_patience_spinbox')
    assert not hasattr(panel, 'early_stopping_min_delta_spinbox')
    assert not hasattr(panel, 'restore_best_weights_check_box')
    assert panel._settings_cards['training'].isAncestorOf(panel.epochs_spinbox)
    assert panel._settings_cards['training'].isAncestorOf(panel.early_stopping_check_box)
    assert panel._settings_cards['validation_data'].isAncestorOf(panel.validation_check_box)

    panel.early_stopping_check_box.setChecked(True)

    assert not panel.epochs_spinbox.isEnabled()
    assert _field_is_hidden(panel, panel.epochs_spinbox)
    assert not panel.early_stopping_control_warning.isHidden()


def test_validation_controls_have_one_owner_on_data_page(app):
    panel = SettingsPanel()

    assert panel._settings_cards['validation_data'].isAncestorOf(panel.validation_check_box)
    assert not panel.validation_groupbox.isAncestorOf(panel.validation_check_box)
    assert panel.validation_groupbox.isAncestorOf(panel.validation_mode_combo)
    assert panel.validation_groupbox.isAncestorOf(panel.validation_spinbox)
    assert panel.validation_groupbox.isAncestorOf(panel.validation_image_path_label)
    assert panel.validation_groupbox.isAncestorOf(panel.validation_label_path_label)

    panel.validation_check_box.setChecked(True)
    assert panel.validation_spinbox.isEnabled()


def test_random_patch_size_is_available_for_online_dataset_and_hidden_until_enabled(app):
    panel = SettingsPanel()
    assert _field_is_hidden(panel, panel.random_patch_min_size_widget)
    panel.torch_compile_check_box.setChecked(True)
    panel.random_patch_size_check_box.setChecked(True)
    assert not _field_is_hidden(panel, panel.random_patch_min_size_widget)
    assert not _field_is_hidden(panel, panel.random_patch_max_size_widget)
    assert not panel.torch_compile_check_box.isEnabled()
    assert not panel.torch_compile_check_box.isChecked()

    assert not hasattr(panel, 'cut_dataset_type')
    assert panel.random_patch_size_check_box.isEnabled()


def test_dependent_augmentation_and_validation_fields_are_hidden(app):
    panel = SettingsPanel()

    assert _field_is_hidden(panel, panel.validation_spinbox)
    assert _field_is_hidden(panel, panel.augmentation_brightness_spinbox)
    assert _field_is_hidden(panel, panel.cutout_probability_spinbox)
    assert _field_is_hidden(panel, panel.random_artifacts_probability_spinbox)
    assert _field_is_hidden(panel, panel.synthetic_dataset_factor_spinbox)
