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


def test_settings_panel_disables_irrelevant_controls_by_work_mode(qapp):
    panel = SettingsPanel()

    panel.sync_business_logic_controls(WorkMode.recognition_only.value)
    assert panel.sample_type_groupbox.isEnabled() is False
    assert panel.additional_augmentation_check_box.isEnabled() is False
    assert panel.validation_check_box.isEnabled() is False
    assert panel._field_rows[panel.optimizer_type].isEnabled() is False
    assert panel._field_rows[panel.overlap_spinbox].isEnabled() is True
    assert panel._field_rows[panel.batch_spinbox].isEnabled() is True
    assert panel._field_rows[panel.nn_model_type].isEnabled() is False
    assert panel.multi_gpu_check_box.isEnabled() is False
    assert panel.torch_compile_check_box.isEnabled() is True

    panel.sync_business_logic_controls(WorkMode.train_only.value)
    assert panel.sample_type_groupbox.isEnabled() is True
    assert panel.additional_augmentation_check_box.isEnabled() is True
    assert panel._field_rows[panel.optimizer_type].isEnabled() is True
    assert panel._field_rows[panel.overlap_spinbox].isEnabled() is False
    assert panel._field_rows[panel.nn_model_type].isEnabled() is True

    panel.sync_business_logic_controls(WorkMode.further_training.value)
    assert panel._field_rows[panel.nn_model_type].isEnabled() is False
    assert panel._field_rows[panel.overlap_spinbox].isEnabled() is True


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
        additional_augmentation=True,
        augmentation_brightness_strength=0.22,
        augmentation_contrast_strength=0.18,
        augmentation_noise_probability=0.4,
        augmentation_noise_sigma=0.012,
        optimizer_name='adamw',
        mixed_precision='bf16',
        loss_function='bce_dice',
        dice_loss_weight=0.55,
        iou_loss_weight=0.35,
        learning_rate=0.0007,
        weight_decay=0.02,
        warmup_enabled=True,
        warmup_epochs=4,
        warmup_start_factor=0.3,
        hard_mining_enabled=True,
        hard_mining_strength=2.6,
        hard_mining_ema_alpha=0.45,
        skip_uniform_labels=True,
        torch_compile_enabled=False,
        early_stopping_enabled=True,
        early_stopping_patience=6,
        early_stopping_min_delta=0.004,
        early_stopping_restore_best_weights=False,
        show_batch_preview=False,
    )

    module.MainPresenter._apply_settings_to_panel(presenter)
    assert panel.optimizer_type.currentText() == 'adamw'
    assert panel.additional_augmentation_check_box.isChecked() is True
    assert panel.augmentation_brightness_spinbox.value() == pytest.approx(0.22)
    assert panel.augmentation_contrast_spinbox.value() == pytest.approx(0.18)
    assert panel.augmentation_noise_probability_spinbox.value() == pytest.approx(0.4)
    assert panel.augmentation_noise_sigma_spinbox.value() == pytest.approx(0.012)
    assert panel.mixed_precision_type.currentText() == 'bf16'
    assert panel.loss_function_type.currentText() == 'bce_dice'
    assert panel.dice_loss_weight_spinbox.value() == pytest.approx(0.55)
    assert panel.iou_loss_weight_spinbox.value() == pytest.approx(0.35)
    assert panel.learning_rate_spinbox.value() == pytest.approx(0.0007)
    assert panel.weight_decay_spinbox.value() == pytest.approx(0.02)
    assert panel.warmup_check_box.isChecked() is True
    assert panel.warmup_epochs_spinbox.value() == 4
    assert panel.warmup_start_factor_spinbox.value() == pytest.approx(0.3)
    assert panel.hard_mining_check_box.isChecked() is True
    assert panel.hard_mining_strength_spinbox.value() == pytest.approx(2.6)
    assert panel.hard_mining_ema_alpha_spinbox.value() == pytest.approx(0.45)
    assert panel.skip_uniform_labels_check_box.isChecked() is True
    assert panel.torch_compile_check_box.isChecked() is False
    assert panel.early_stopping_check_box.isChecked() is True
    assert panel.early_stopping_patience_spinbox.value() == 6
    assert panel.early_stopping_min_delta_spinbox.value() == pytest.approx(0.004)
    assert panel.restore_best_weights_check_box.isChecked() is False
    assert presenter.view.is_batch_preview_enabled() is False

    panel.cut_dataset_type.setChecked(True)
    panel.additional_augmentation_check_box.setChecked(True)
    panel.augmentation_brightness_spinbox.setValue(0.35)
    panel.augmentation_contrast_spinbox.setValue(0.27)
    panel.augmentation_noise_probability_spinbox.setValue(0.55)
    panel.augmentation_noise_sigma_spinbox.setValue(0.02)
    panel.optimizer_type.setCurrentText('adamw_muon')
    panel.mixed_precision_type.setCurrentText('off')
    panel.loss_function_type.setCurrentText('bce_iou')
    panel.dice_loss_weight_spinbox.setValue(1.0)
    panel.iou_loss_weight_spinbox.setValue(0.8)
    panel.learning_rate_spinbox.setValue(0.0003)
    panel.weight_decay_spinbox.setValue(0.015)
    panel.warmup_check_box.setChecked(False)
    panel.warmup_epochs_spinbox.setValue(2)
    panel.warmup_start_factor_spinbox.setValue(0.15)
    panel.hard_mining_check_box.setChecked(True)
    panel.hard_mining_strength_spinbox.setValue(3.4)
    panel.hard_mining_ema_alpha_spinbox.setValue(0.25)
    panel.skip_uniform_labels_check_box.setChecked(False)
    panel.torch_compile_check_box.setChecked(True)
    panel.early_stopping_check_box.setChecked(False)
    panel.early_stopping_patience_spinbox.setValue(2)
    panel.early_stopping_min_delta_spinbox.setValue(0.002)
    panel.restore_best_weights_check_box.setChecked(True)
    presenter.view.set_batch_preview_enabled(True)

    module.MainPresenter._update_settings_window_state(presenter)
    assert presenter.settings_state.additional_augmentation is True
    assert presenter.settings_state.augmentation_brightness_strength == pytest.approx(0.35)
    assert presenter.settings_state.augmentation_contrast_strength == pytest.approx(0.27)
    assert presenter.settings_state.augmentation_noise_probability == pytest.approx(0.55)
    assert presenter.settings_state.augmentation_noise_sigma == pytest.approx(0.02)
    assert presenter.settings_state.optimizer_name == 'adamw_muon'
    assert presenter.settings_state.mixed_precision == 'off'
    assert presenter.settings_state.loss_function == 'bce_iou'
    assert presenter.settings_state.dice_loss_weight == pytest.approx(1.0)
    assert presenter.settings_state.iou_loss_weight == pytest.approx(0.8)
    assert presenter.settings_state.learning_rate == pytest.approx(0.0003)
    assert presenter.settings_state.weight_decay == pytest.approx(0.015)
    assert presenter.settings_state.warmup_enabled is False
    assert presenter.settings_state.warmup_epochs == 2
    assert presenter.settings_state.warmup_start_factor == pytest.approx(0.15)
    assert presenter.settings_state.hard_mining_enabled is True
    assert presenter.settings_state.hard_mining_strength == pytest.approx(3.4)
    assert presenter.settings_state.hard_mining_ema_alpha == pytest.approx(0.25)
    assert presenter.settings_state.skip_uniform_labels is False
    assert presenter.settings_state.torch_compile_enabled is True
    assert presenter.settings_state.early_stopping_enabled is False
    assert presenter.settings_state.early_stopping_patience == 2
    assert presenter.settings_state.early_stopping_min_delta == pytest.approx(0.002)
    assert presenter.settings_state.early_stopping_restore_best_weights is True
    assert presenter.settings_state.show_batch_preview is True
