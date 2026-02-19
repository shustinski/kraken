import pytest

pytest.importorskip('PyQt6')

from presenter.state_store import (
    load_main_window_state,
    load_settings_state,
    save_main_window_state,
    save_settings_state,
)
from view.window_dataclasses import MainWindowState, SettingsState
from tests.helpers import make_test_dir


def test_state_store_roundtrip_main_window(monkeypatch):
    settings_dir = make_test_dir("qsettings_main")
    monkeypatch.setenv("NEURALIMAGE_SETTINGS_DIR", str(settings_dir))
    state = MainWindowState(
        work_mode='train_only',
        source_folder='s',
        result_folder='r',
        model_path='m',
        label_folder='l',
        sample_folder='p',
        epochs=7,
    )
    save_main_window_state(state)
    loaded = load_main_window_state()
    assert loaded.work_mode == 'train_only'
    assert loaded.epochs == 7


def test_state_store_roundtrip_settings(monkeypatch):
    settings_dir = make_test_dir("qsettings_settings")
    monkeypatch.setenv("NEURALIMAGE_SETTINGS_DIR", str(settings_dir))
    state = SettingsState(
        step=77,
        sample_size=(11, 22),
        model='M 720k',
        optimizer_name='adamw',
        mixed_precision='fp16',
        loss_function='bce_dice',
        dice_loss_weight=0.6,
        iou_loss_weight=0.4,
        learning_rate=0.0002,
        weight_decay=0.01,
        warmup_enabled=True,
        warmup_epochs=5,
        warmup_start_factor=0.25,
        additional_augmentation=True,
        augmentation_brightness_strength=0.2,
        augmentation_contrast_strength=0.3,
        augmentation_noise_probability=0.6,
        augmentation_noise_sigma=0.015,
        crop_enabled=True,
        resize_enabled=False,
        hard_mining_enabled=True,
        hard_mining_strength=3.5,
        hard_mining_ema_alpha=0.4,
        skip_uniform_labels=True,
        torch_compile_enabled=False,
        early_stopping_enabled=True,
        early_stopping_patience=8,
        early_stopping_min_delta=0.003,
        early_stopping_restore_best_weights=False,
        show_batch_preview=False,
    )
    save_settings_state(state)
    loaded = load_settings_state()
    assert loaded.step == 77
    assert loaded.sample_size == (11, 22)
    assert loaded.optimizer_name == 'adamw'
    assert loaded.mixed_precision == 'fp16'
    assert loaded.loss_function == 'bce_dice'
    assert loaded.dice_loss_weight == 0.6
    assert loaded.iou_loss_weight == 0.4
    assert loaded.learning_rate == 0.0002
    assert loaded.weight_decay == 0.01
    assert loaded.warmup_enabled is True
    assert loaded.warmup_epochs == 5
    assert loaded.warmup_start_factor == 0.25
    assert loaded.additional_augmentation is True
    assert loaded.augmentation_brightness_strength == 0.2
    assert loaded.augmentation_contrast_strength == 0.3
    assert loaded.augmentation_noise_probability == 0.6
    assert loaded.augmentation_noise_sigma == 0.015
    assert loaded.crop_enabled is True
    assert loaded.resize_enabled is False
    assert loaded.hard_mining_enabled is True
    assert loaded.hard_mining_strength == 3.5
    assert loaded.hard_mining_ema_alpha == 0.4
    assert loaded.skip_uniform_labels is True
    assert loaded.torch_compile_enabled is False
    assert loaded.early_stopping_enabled is True
    assert loaded.early_stopping_patience == 8
    assert loaded.early_stopping_min_delta == 0.003
    assert loaded.early_stopping_restore_best_weights is False
    assert loaded.show_batch_preview is False


def test_state_store_roundtrip_main_window_ini_backend(monkeypatch):
    settings_dir = make_test_dir("ini_main")
    ini_path = settings_dir / "state.ini"
    monkeypatch.setenv("NEURALIMAGE_STATE_BACKEND", "ini")
    monkeypatch.setenv("NEURALIMAGE_INI_PATH", str(ini_path))

    state = MainWindowState(
        work_mode='train_only',
        source_folder='source',
        result_folder='result',
        model_path='model',
        label_folder='labels',
        sample_folder='samples',
        epochs=11,
    )
    save_main_window_state(state)
    loaded = load_main_window_state()

    assert loaded.work_mode == 'train_only'
    assert loaded.source_folder == 'source'
    assert loaded.epochs == 11


def test_state_store_roundtrip_settings_ini_backend(monkeypatch):
    settings_dir = make_test_dir("ini_settings")
    ini_path = settings_dir / "state.ini"
    monkeypatch.setenv("NEURALIMAGE_STATE_BACKEND", "ini")
    monkeypatch.setenv("NEURALIMAGE_INI_PATH", str(ini_path))

    state = SettingsState(
        step=55,
        sample_size=(64, 96),
        model='M 720k',
        use_validation=True,
        validation_percent=25,
        optimizer_name='adamw_muon',
        mixed_precision='off',
        loss_function='dice',
        dice_loss_weight=1.0,
        iou_loss_weight=0.3,
        learning_rate=0.0003,
        weight_decay=0.02,
        warmup_enabled=True,
        warmup_epochs=3,
        warmup_start_factor=0.1,
        additional_augmentation=False,
        augmentation_brightness_strength=0.12,
        augmentation_contrast_strength=0.08,
        augmentation_noise_probability=0.25,
        augmentation_noise_sigma=0.005,
        crop_enabled=False,
        resize_enabled=True,
        hard_mining_enabled=True,
        hard_mining_strength=2.8,
        hard_mining_ema_alpha=0.3,
        skip_uniform_labels=True,
        torch_compile_enabled=False,
        early_stopping_enabled=True,
        early_stopping_patience=4,
        early_stopping_min_delta=0.001,
        early_stopping_restore_best_weights=True,
        show_batch_preview=False,
    )
    save_settings_state(state)
    loaded = load_settings_state()

    assert loaded.step == 55
    assert loaded.sample_size == (64, 96)
    assert loaded.use_validation is True
    assert loaded.validation_percent == 25
    assert loaded.optimizer_name == 'adamw_muon'
    assert loaded.mixed_precision == 'off'
    assert loaded.loss_function == 'dice'
    assert loaded.dice_loss_weight == 1.0
    assert loaded.iou_loss_weight == 0.3
    assert loaded.learning_rate == 0.0003
    assert loaded.weight_decay == 0.02
    assert loaded.warmup_enabled is True
    assert loaded.warmup_epochs == 3
    assert loaded.warmup_start_factor == 0.1
    assert loaded.additional_augmentation is False
    assert loaded.augmentation_brightness_strength == 0.12
    assert loaded.augmentation_contrast_strength == 0.08
    assert loaded.augmentation_noise_probability == 0.25
    assert loaded.augmentation_noise_sigma == 0.005
    assert loaded.crop_enabled is False
    assert loaded.resize_enabled is True
    assert loaded.hard_mining_enabled is True
    assert loaded.hard_mining_strength == 2.8
    assert loaded.hard_mining_ema_alpha == 0.3
    assert loaded.skip_uniform_labels is True
    assert loaded.torch_compile_enabled is False
    assert loaded.early_stopping_enabled is True
    assert loaded.early_stopping_patience == 4
    assert loaded.early_stopping_min_delta == 0.001
    assert loaded.early_stopping_restore_best_weights is True
    assert loaded.show_batch_preview is False
