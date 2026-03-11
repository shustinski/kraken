import os

import pytest

django = pytest.importorskip("django")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "webui_project.settings")
django.setup()

from webui.forms import MainWindowForm, SettingsForm, defaults_from_main_state, defaults_from_settings_state
from view.window_dataclasses import MainWindowState, SettingsState
from tests.helpers import make_test_dir


def _build_valid_main_window_form_data(work_mode: str) -> dict[str, str]:
    root = make_test_dir(f"web_form_valid_{work_mode}")
    source = root / "source"
    result = root / "result"
    sample = root / "sample"
    label = root / "label"
    model = root / "model.pth"
    source.mkdir()
    result.mkdir()
    sample.mkdir()
    label.mkdir()
    model.write_text("x", encoding="utf-8")
    return {
        "work_mode": work_mode,
        "source_folder": str(source),
        "result_folder": str(result),
        "sample_folder": str(sample),
        "label_folder": str(label),
        "model_path": str(model),
        "epochs": "5",
    }


def test_settings_form_to_state_maps_new_processing_and_augmentation_fields():
    data = {
        "step": "100",
        "vertical_rotation": "on",
        "horizontal_rotation": "on",
        "additional_augmentation": "on",
        "random_crop": "on",
        "crops_per_image": "23",
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
        "hard_pixel_mining_enabled": "on",
        "hard_pixel_mining_ratio": "0.2",
        "cutout_enabled": "on",
        "cutout_probability": "0.8",
        "cutout_holes": "3",
        "cutout_size_ratio": "0.35",
        "mixup_enabled": "on",
        "mixup_probability": "0.7",
        "mixup_alpha": "0.4",
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
    assert state.random_crop is True
    assert state.crops_per_image == 23
    assert state.augmentation_brightness_strength == pytest.approx(0.3)
    assert state.augmentation_contrast_strength == pytest.approx(0.2)
    assert state.augmentation_noise_probability == pytest.approx(0.6)
    assert state.augmentation_noise_sigma == pytest.approx(0.02)
    assert state.crop_enabled is True
    assert state.resize_enabled is True
    assert state.log_update_frequency == 40
    assert state.torch_compile_enabled is True
    assert state.hard_pixel_mining_enabled is True
    assert state.hard_pixel_mining_ratio == pytest.approx(0.2)
    assert state.cutout_enabled is True
    assert state.cutout_probability == pytest.approx(0.8)
    assert state.cutout_holes == 3
    assert state.cutout_size_ratio == pytest.approx(0.35)
    assert state.mixup_enabled is True
    assert state.mixup_probability == pytest.approx(0.7)
    assert state.mixup_alpha == pytest.approx(0.4)


def test_defaults_from_settings_state_exposes_new_keys():
    state = SettingsState(
        additional_augmentation=True,
        random_crop=True,
        crops_per_image=23,
        augmentation_brightness_strength=0.25,
        augmentation_contrast_strength=0.15,
        augmentation_noise_probability=0.4,
        augmentation_noise_sigma=0.01,
        crop_enabled=True,
        resize_enabled=False,
        log_update_frequency=10,
        torch_compile_enabled=True,
        hard_pixel_mining_enabled=True,
        hard_pixel_mining_ratio=0.3,
        cutout_enabled=True,
        cutout_probability=0.85,
        cutout_holes=2,
        cutout_size_ratio=0.4,
        mixup_enabled=True,
        mixup_probability=0.65,
        mixup_alpha=0.35,
    )
    defaults = defaults_from_settings_state(state)

    assert defaults["additional_augmentation"] is True
    assert defaults["random_crop"] is True
    assert defaults["crops_per_image"] == 23
    assert defaults["augmentation_brightness_strength"] == pytest.approx(0.25)
    assert defaults["augmentation_contrast_strength"] == pytest.approx(0.15)
    assert defaults["augmentation_noise_probability"] == pytest.approx(0.4)
    assert defaults["augmentation_noise_sigma"] == pytest.approx(0.01)
    assert defaults["crop_enabled"] is True
    assert defaults["resize_enabled"] is False
    assert defaults["log_update_frequency"] == 10
    assert defaults["torch_compile_enabled"] is True
    assert defaults["hard_pixel_mining_enabled"] is True
    assert defaults["hard_pixel_mining_ratio"] == pytest.approx(0.3)
    assert defaults["cutout_enabled"] is True
    assert defaults["cutout_probability"] == pytest.approx(0.85)
    assert defaults["cutout_holes"] == 2
    assert defaults["cutout_size_ratio"] == pytest.approx(0.4)
    assert defaults["mixup_enabled"] is True
    assert defaults["mixup_probability"] == pytest.approx(0.65)
    assert defaults["mixup_alpha"] == pytest.approx(0.35)


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
    assert state.hard_pixel_mining_ratio == defaults.hard_pixel_mining_ratio
    assert state.cutout_probability == defaults.cutout_probability
    assert state.cutout_holes == defaults.cutout_holes
    assert state.cutout_size_ratio == defaults.cutout_size_ratio
    assert state.mixup_probability == defaults.mixup_probability
    assert state.mixup_alpha == defaults.mixup_alpha


def test_settings_form_lists_new_loss_function_choices():
    form = SettingsForm()
    loss_choices = {value for value, _label in form.fields['loss_function'].choices}

    assert 'boundary' in loss_choices
    assert 'focal_tversky' in loss_choices


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

    form = MainWindowForm(data=data, language="en")
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

    form = MainWindowForm(data=data, language="en")
    assert form.is_valid(), form.errors
    state = form.to_state()
    assert state.work_mode == "recognition_only"


def test_main_window_form_normalizer_keeps_payload_when_data_missing():
    args, kwargs = MainWindowForm._normalize_legacy_work_mode_payload((), {})
    assert args == ()
    assert kwargs == {}


def test_main_window_form_normalizer_updates_positional_data_payload():
    original_data = {"work_mode": "recognintion_only"}
    args, kwargs = MainWindowForm._normalize_legacy_work_mode_payload((original_data,), {})

    assert kwargs == {}
    assert original_data["work_mode"] == "recognintion_only"
    assert args[0]["work_mode"] == "recognition_only"


def test_main_window_form_rejects_missing_source_folder_for_recognition_mode():
    data = _build_valid_main_window_form_data("recognition_only")
    data["source_folder"] = ""

    form = MainWindowForm(data=data, language="en")
    assert not form.is_valid()
    assert form.errors.as_data()["__all__"][0].message == 'The “Source files folder” field must point to an existing directory.'


def test_main_window_form_rejects_missing_result_folder_for_recognition_mode():
    data = _build_valid_main_window_form_data("recognition_only")
    data["result_folder"] = f"{data['result_folder']}_missing"

    form = MainWindowForm(data=data, language="en")
    assert not form.is_valid()
    assert form.errors.as_data()["__all__"][0].message == 'The “Output folder” field must point to an existing directory.'


def test_main_window_form_rejects_missing_sample_folder_for_training_mode():
    data = _build_valid_main_window_form_data("train_and_recognition")
    data["sample_folder"] = f"{data['sample_folder']}_missing"

    form = MainWindowForm(data=data, language="en")
    assert not form.is_valid()
    assert form.errors.as_data()["__all__"][0].message == "Training requires an existing folder with training images."


def test_main_window_form_rejects_missing_label_folder_for_training_mode():
    data = _build_valid_main_window_form_data("train_and_recognition")
    data["label_folder"] = f"{data['label_folder']}_missing"

    form = MainWindowForm(data=data, language="en")
    assert not form.is_valid()
    assert form.errors.as_data()["__all__"][0].message == "Training requires an existing folder with masks or labels."


def test_main_window_form_rejects_missing_model_path_for_recognition_mode():
    data = _build_valid_main_window_form_data("recognition_only")
    data["model_path"] = f"{data['model_path']}_missing"

    form = MainWindowForm(data=data, language="en")
    assert not form.is_valid()
    assert form.errors.as_data()["__all__"][0].message == "The selected mode requires a valid `.pth` model file path."


def test_defaults_from_main_state_returns_dataclass_dict():
    state = MainWindowState(
        work_mode="train_only",
        source_folder="D:/src",
        result_folder="D:/res",
        model_path="D:/models/m.pth",
        label_folder="D:/label",
        sample_folder="D:/sample",
        epochs=7,
    )

    assert defaults_from_main_state(state) == {
        "work_mode": "train_only",
        "source_folder": "D:/src",
        "result_folder": "D:/res",
        "model_path": "D:/models/m.pth",
        "label_folder": "D:/label",
        "sample_folder": "D:/sample",
        "epochs": 7,
    }


def test_main_window_form_localizes_labels_by_requested_language():
    ru_form = MainWindowForm(language="ru")
    en_form = MainWindowForm(language="en")

    assert ru_form.fields["source_folder"].label == "Папка с исходными файлами"
    assert en_form.fields["source_folder"].label == "Source files folder"


def test_settings_form_localizes_help_texts_by_requested_language():
    ru_form = SettingsForm(language="ru")
    en_form = SettingsForm(language="en")

    assert "скорость" in ru_form.fields["learning_rate"].help_text.lower()
    assert "learning rate" in en_form.fields["learning_rate"].help_text.lower()
