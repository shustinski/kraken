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
    epochs_row, _ = panel.general_form.getWidgetPosition(panel._field_rows[panel.epochs_spinbox])
    early_stopping_row, _ = panel.general_form.getWidgetPosition(panel.early_stopping_check_box)
    assert early_stopping_row == epochs_row + 1

    panel.early_stopping_check_box.setChecked(True)

    assert not panel.epochs_spinbox.isEnabled()
    assert _field_is_hidden(panel, panel.epochs_spinbox)
    assert not panel.early_stopping_control_warning.isHidden()


def test_random_patch_size_is_online_only_and_hidden_until_enabled(app):
    panel = SettingsPanel()
    assert _field_is_hidden(panel, panel.random_patch_min_size_widget)
    panel.random_patch_size_check_box.setChecked(True)
    assert not _field_is_hidden(panel, panel.random_patch_min_size_widget)
    assert not _field_is_hidden(panel, panel.random_patch_max_size_widget)

    panel.cut_dataset_type.setChecked(True)

    assert not panel.random_patch_size_check_box.isEnabled()
    assert _field_is_hidden(panel, panel.random_patch_min_size_widget)


def test_dependent_augmentation_and_validation_fields_are_hidden(app):
    panel = SettingsPanel()

    assert _field_is_hidden(panel, panel.validation_spinbox)
    assert _field_is_hidden(panel, panel.augmentation_brightness_spinbox)
    assert _field_is_hidden(panel, panel.cutout_probability_spinbox)
    assert _field_is_hidden(panel, panel.random_artifacts_probability_spinbox)
    assert _field_is_hidden(panel, panel.synthetic_dataset_factor_spinbox)
