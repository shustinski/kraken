import os

import pytest

django = pytest.importorskip("django")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "webui_project.settings")
django.setup()

from webui.forms import MainWindowForm, SettingsForm, defaults_from_settings_state
from view.window_dataclasses import SettingsState
from tests.helpers import make_test_dir


def test_settings_form_to_state_maps_new_processing_and_augmentation_fields():
    data = {
        "step": "100",
        "vertical_rotation": "on",
        "horizontal_rotation": "on",
        "additional_augmentation": "on",
        "augmentation_brightness_strength": "0.3",
        "augmentation_contrast_strength": "0.2",
        "augmentation_noise_probability": "0.6",
        "augmentation_noise_sigma": "0.02",
        "sample_x": "256",
        "sample_y": "256",
        "model": "M 720k",
        "color_mode": "RGB",
        "use_validation": "on",
        "validation_percent": "20",
        "shuffle": "on",
        "sample_cut_mode": "online",
        "batch_size": "8",
        "overlap": "16",
        "log_update_frequency": "40",
        "crop_enabled": "on",
        "resize_enabled": "on",
        "edge_cut_size": "10",
        "target_x": "1024",
        "target_y": "768",
        "optimizer_name": "adamw",
        "mixed_precision": "bf16",
        "loss_function": "bce_dice",
        "dice_loss_weight": "0.7",
        "iou_loss_weight": "0.3",
        "learning_rate": "0.0005",
        "weight_decay": "0.01",
        "warmup_enabled": "on",
        "warmup_epochs": "3",
        "warmup_start_factor": "0.1",
        "hard_mining_enabled": "on",
        "hard_mining_strength": "2.5",
        "hard_mining_ema_alpha": "0.3",
        "skip_uniform_labels": "on",
        "early_stopping_enabled": "on",
        "early_stopping_patience": "7",
        "early_stopping_min_delta": "0.001",
        "early_stopping_restore_best_weights": "on",
        "torch_compile_enabled": "on",
        "show_batch_preview": "on",
        "use_multi_gpu": "on",
    }
    form = SettingsForm(data=data)
    assert form.is_valid(), form.errors

    state = form.to_state()
    assert state.additional_augmentation is True
    assert state.augmentation_brightness_strength == pytest.approx(0.3)
    assert state.augmentation_contrast_strength == pytest.approx(0.2)
    assert state.augmentation_noise_probability == pytest.approx(0.6)
    assert state.augmentation_noise_sigma == pytest.approx(0.02)
    assert state.crop_enabled is True
    assert state.resize_enabled is True
    assert state.log_update_frequency == 40
    assert state.torch_compile_enabled is True


def test_defaults_from_settings_state_exposes_new_keys():
    state = SettingsState(
        additional_augmentation=True,
        augmentation_brightness_strength=0.25,
        augmentation_contrast_strength=0.15,
        augmentation_noise_probability=0.4,
        augmentation_noise_sigma=0.01,
        crop_enabled=True,
        resize_enabled=False,
        log_update_frequency=10,
        torch_compile_enabled=True,
    )
    defaults = defaults_from_settings_state(state)

    assert defaults["additional_augmentation"] is True
    assert defaults["augmentation_brightness_strength"] == pytest.approx(0.25)
    assert defaults["augmentation_contrast_strength"] == pytest.approx(0.15)
    assert defaults["augmentation_noise_probability"] == pytest.approx(0.4)
    assert defaults["augmentation_noise_sigma"] == pytest.approx(0.01)
    assert defaults["crop_enabled"] is True
    assert defaults["resize_enabled"] is False
    assert defaults["log_update_frequency"] == 10
    assert defaults["torch_compile_enabled"] is True


def test_settings_form_defaults_optional_disabled_fields():
    # Simulate UI with disabled dependent fields: omitted values must fallback to SettingsState defaults.
    data = {
        "step": "100",
        "sample_x": "256",
        "sample_y": "256",
        "model": "M 720k",
        "color_mode": "RGB",
        "sample_cut_mode": "online",
        "batch_size": "8",
        "overlap": "16",
        "log_update_frequency": "0",
        "optimizer_name": "adam",
        "mixed_precision": "bf16",
        "loss_function": "bce",
        "learning_rate": "0.001",
        "weight_decay": "0.0",
        "warmup_epochs": "3",
        "warmup_start_factor": "0.1",
        "early_stopping_patience": "10",
        "early_stopping_min_delta": "0.0",
    }
    form = SettingsForm(data=data)
    assert form.is_valid(), form.errors

    state = form.to_state()
    defaults = SettingsState()
    assert state.validation_percent == defaults.validation_percent
    assert state.edge_cut_size == defaults.edge_cut_size
    assert state.target_size == defaults.target_size
    assert state.dice_loss_weight == defaults.dice_loss_weight
    assert state.iou_loss_weight == defaults.iou_loss_weight
    assert state.hard_mining_strength == defaults.hard_mining_strength
    assert state.hard_mining_ema_alpha == defaults.hard_mining_ema_alpha


def test_main_window_form_train_only_allows_empty_source_and_result():
    root = make_test_dir("web_form_train_only")
    sample = root / "sample"
    label = root / "label"
    sample.mkdir()
    label.mkdir()
    data = {
        "work_mode": "train_only",
        "source_folder": "",
        "result_folder": "",
        "sample_folder": str(sample),
        "label_folder": str(label),
        "model_path": "",
        "epochs": "5",
    }

    form = MainWindowForm(data=data)
    assert form.is_valid(), form.errors


def test_main_window_form_normalizes_legacy_work_mode_alias():
    root = make_test_dir("web_form_mode_alias")
    source = root / "source"
    result = root / "result"
    model = root / "model.pth"
    source.mkdir()
    result.mkdir()
    model.write_text("x", encoding="utf-8")
    data = {
        "work_mode": "recognintion_only",
        "source_folder": str(source),
        "result_folder": str(result),
        "sample_folder": "",
        "label_folder": "",
        "model_path": str(model),
        "epochs": "1",
    }

    form = MainWindowForm(data=data)
    assert form.is_valid(), form.errors
    state = form.to_state()
    assert state.work_mode == "recognition_only"
