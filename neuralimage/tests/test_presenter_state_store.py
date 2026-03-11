import pytest

pytest.importorskip('PyQt6')

from application.dto import MainWindowState, SettingsState
from infrastructure.config.state_store import (
    IniStateStore,
    QSettingsStateStore,
    create_state_store,
    load_main_window_state,
    load_settings_state,
    save_main_window_state,
    save_settings_state,
)
from tests.helpers import make_test_dir


def test_create_state_store_defaults_to_qsettings(monkeypatch):
    monkeypatch.delenv("NEURALIMAGE_STATE_BACKEND", raising=False)

    store = create_state_store()

    assert isinstance(store, QSettingsStateStore)


def test_create_state_store_uses_requested_default_backend(monkeypatch):
    monkeypatch.delenv("NEURALIMAGE_STATE_BACKEND", raising=False)

    store = create_state_store(default_backend='ini')

    assert isinstance(store, IniStateStore)


def test_create_state_store_environment_override_wins(monkeypatch):
    monkeypatch.setenv("NEURALIMAGE_STATE_BACKEND", "ini")

    store = create_state_store(default_backend='qsettings')

    assert isinstance(store, IniStateStore)


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
        sync_patch_sizes=False,
        recognition_jpeg_quality=89,
        shuffle=False,
        shuffle_patches_in_frame=True,
        random_crop=True,
        crops_per_image=21,
        model='M 720k',
        optimizer_name='adamw',
        mixed_precision='fp16',
        loss_function='bce_dice',
        loss_term_weights={'bce': 0.4, 'dice': 0.6},
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
        augmentation_gamma_strength=0.18,
        augmentation_noise_probability=0.6,
        augmentation_noise_sigma=0.015,
        augmentation_blur_probability=0.4,
        augmentation_blur_radius=1.7,
        recognition_binarize_output=False,
        recognition_use_auto_threshold=False,
        recognition_threshold=0.63,
        recognition_postprocess=True,
        recognition_postprocess_kernel_size=5,
        crop_enabled=True,
        resize_enabled=False,
        hard_mining_enabled=True,
        hard_mining_strength=3.5,
        hard_mining_ema_alpha=0.4,
        hard_pixel_mining_enabled=True,
        hard_pixel_mining_ratio=0.2,
        cutout_enabled=True,
        cutout_probability=0.8,
        cutout_holes=2,
        cutout_size_ratio=0.35,
        mixup_enabled=True,
        mixup_probability=0.7,
        mixup_alpha=0.45,
        skip_uniform_labels=True,
        rare_patch_oversampling_enabled=True,
        rare_patch_oversampling_factor=5,
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
    assert loaded.sync_patch_sizes is False
    assert loaded.recognition_jpeg_quality == 89
    assert loaded.shuffle is False
    assert loaded.shuffle_patches_in_frame is True
    assert loaded.random_crop is True
    assert loaded.crops_per_image == 21
    assert loaded.optimizer_name == 'adamw'
    assert loaded.mixed_precision == 'fp16'
    assert loaded.loss_function == 'bce_dice'
    assert loaded.loss_term_weights == {'bce': 0.4, 'dice': 0.6}
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
    assert loaded.augmentation_gamma_strength == pytest.approx(0.18)
    assert loaded.augmentation_noise_probability == 0.6
    assert loaded.augmentation_noise_sigma == 0.015
    assert loaded.augmentation_blur_probability == pytest.approx(0.4)
    assert loaded.augmentation_blur_radius == pytest.approx(1.7)
    assert loaded.recognition_binarize_output is False
    assert loaded.recognition_use_auto_threshold is False
    assert loaded.recognition_threshold == pytest.approx(0.63)
    assert loaded.recognition_postprocess is True
    assert loaded.recognition_postprocess_kernel_size == 5
    assert loaded.crop_enabled is True
    assert loaded.resize_enabled is False
    assert loaded.hard_mining_enabled is True
    assert loaded.hard_mining_strength == 3.5
    assert loaded.hard_mining_ema_alpha == 0.4
    assert loaded.hard_pixel_mining_enabled is True
    assert loaded.hard_pixel_mining_ratio == pytest.approx(0.2)
    assert loaded.cutout_enabled is True
    assert loaded.cutout_probability == pytest.approx(0.8)
    assert loaded.cutout_holes == 2
    assert loaded.cutout_size_ratio == pytest.approx(0.35)
    assert loaded.mixup_enabled is True
    assert loaded.mixup_probability == pytest.approx(0.7)
    assert loaded.mixup_alpha == pytest.approx(0.45)
    assert loaded.skip_uniform_labels is True
    assert loaded.rare_patch_oversampling_enabled is True
    assert loaded.rare_patch_oversampling_factor == 5
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
        sync_patch_sizes=False,
        recognition_jpeg_quality=91,
        shuffle=True,
        shuffle_patches_in_frame=False,
        random_crop=False,
        crops_per_image=13,
        model='M 720k',
        use_validation=True,
        validation_percent=25,
        optimizer_name='adamw_muon',
        mixed_precision='off',
        loss_function='dice',
        loss_term_weights={'dice': 1.0},
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
        augmentation_gamma_strength=0.11,
        augmentation_noise_probability=0.25,
        augmentation_noise_sigma=0.005,
        augmentation_blur_probability=0.3,
        augmentation_blur_radius=1.2,
        recognition_binarize_output=True,
        recognition_use_auto_threshold=False,
        recognition_threshold=0.58,
        recognition_postprocess=True,
        recognition_postprocess_kernel_size=7,
        crop_enabled=False,
        resize_enabled=True,
        hard_mining_enabled=True,
        hard_mining_strength=2.8,
        hard_mining_ema_alpha=0.3,
        hard_pixel_mining_enabled=True,
        hard_pixel_mining_ratio=0.35,
        cutout_enabled=True,
        cutout_probability=0.9,
        cutout_holes=4,
        cutout_size_ratio=0.4,
        mixup_enabled=True,
        mixup_probability=0.55,
        mixup_alpha=0.3,
        skip_uniform_labels=True,
        rare_patch_oversampling_enabled=True,
        rare_patch_oversampling_factor=4,
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
    assert loaded.sync_patch_sizes is False
    assert loaded.recognition_jpeg_quality == 91
    assert loaded.shuffle is True
    assert loaded.shuffle_patches_in_frame is False
    assert loaded.random_crop is False
    assert loaded.crops_per_image == 13
    assert loaded.use_validation is True
    assert loaded.validation_percent == 25
    assert loaded.optimizer_name == 'adamw_muon'
    assert loaded.mixed_precision == 'off'
    assert loaded.loss_function == 'dice'
    assert loaded.loss_term_weights == {'dice': 1.0}
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
    assert loaded.augmentation_gamma_strength == pytest.approx(0.11)
    assert loaded.augmentation_noise_probability == 0.25
    assert loaded.augmentation_noise_sigma == 0.005
    assert loaded.augmentation_blur_probability == pytest.approx(0.3)
    assert loaded.augmentation_blur_radius == pytest.approx(1.2)
    assert loaded.recognition_binarize_output is True
    assert loaded.recognition_use_auto_threshold is False
    assert loaded.recognition_threshold == pytest.approx(0.58)
    assert loaded.recognition_postprocess is True
    assert loaded.recognition_postprocess_kernel_size == 7
    assert loaded.crop_enabled is False
    assert loaded.resize_enabled is True
    assert loaded.hard_mining_enabled is True
    assert loaded.hard_mining_strength == 2.8
    assert loaded.hard_mining_ema_alpha == 0.3
    assert loaded.hard_pixel_mining_enabled is True
    assert loaded.hard_pixel_mining_ratio == pytest.approx(0.35)
    assert loaded.cutout_enabled is True
    assert loaded.cutout_probability == pytest.approx(0.9)
    assert loaded.cutout_holes == 4
    assert loaded.cutout_size_ratio == pytest.approx(0.4)
    assert loaded.mixup_enabled is True
    assert loaded.mixup_probability == pytest.approx(0.55)
    assert loaded.mixup_alpha == pytest.approx(0.3)
    assert loaded.skip_uniform_labels is True
    assert loaded.rare_patch_oversampling_enabled is True
    assert loaded.rare_patch_oversampling_factor == 4
    assert loaded.torch_compile_enabled is False
    assert loaded.early_stopping_enabled is True
    assert loaded.early_stopping_patience == 4
    assert loaded.early_stopping_min_delta == 0.001
    assert loaded.early_stopping_restore_best_weights is True
    assert loaded.show_batch_preview is False
