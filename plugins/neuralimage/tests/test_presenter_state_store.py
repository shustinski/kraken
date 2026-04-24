import json

import pytest

pytest.importorskip('PyQt6')

from neuralimage.application.dto import MainWindowState, SettingsState
from neuralimage.infrastructure.config.state_store import (
    IniStateStore,
    QSettingsStateStore,
    WORKFLOW_SNAPSHOT_FILENAME,
    create_state_store,
    load_main_window_state,
    load_settings_state,
    load_workflow_snapshot,
    resolve_workflow_snapshot_path,
    save_main_window_state,
    save_settings_state,
    save_workflow_snapshot,
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
        ui_mode='simple',
        mode_state={
            'train_only': {
                'source_folder': '',
                'result_folder': '',
                'model_path': '',
                'label_folder': 'labels-train',
                'sample_folder': 'samples-train',
                'epochs': 7,
            },
            'recognition_only': {
                'source_folder': 'src-rec',
                'result_folder': 'res-rec',
                'model_path': 'model-rec.pth',
                'label_folder': '',
                'sample_folder': '',
                'epochs': 0,
            },
        },
    )
    save_main_window_state(state)
    loaded = load_main_window_state()
    assert loaded.work_mode == 'train_only'
    assert loaded.epochs == 7
    assert loaded.ui_mode == 'simple'
    assert loaded.mode_state['train_only']['sample_folder'] == 'samples-train'
    assert loaded.mode_state['recognition_only']['model_path'] == 'model-rec.pth'


def test_state_store_roundtrip_settings(monkeypatch):
    settings_dir = make_test_dir("qsettings_settings")
    monkeypatch.setenv("NEURALIMAGE_SETTINGS_DIR", str(settings_dir))
    state = SettingsState(
        step=77,
        sample_size=(11, 22),
        sync_patch_sizes=False,
        recognition_jpeg_quality=89,
        recognition_multiprocessing_enabled=False,
        shuffle=False,
        shuffle_patches_in_frame=True,
        random_crop=True,
        crops_per_image=21,
        model='M 720k',
        validation_source='external',
        validation_image_folder='val_images',
        validation_label_folder='val_labels',
        save_validation_binary_images=True,
        optimizer_name='adamw',
        mixed_precision='fp16',
        loss_function='bce_dice',
        loss_term_weights={'bce': 0.4, 'dice': 0.6},
        dice_loss_weight=0.6,
        iou_loss_weight=0.4,
        learning_rate=0.0002,
        weight_decay=0.01,
        deep_supervision=False,
        warmup_enabled=True,
        warmup_epochs=5,
        warmup_start_factor=0.25,
        scheduler_name='cosine_annealing',
        scheduler_cosine_t_max=12,
        scheduler_cosine_eta_min=2e-5,
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
        recognition_tta_enabled=True,
        confidence_tta_enabled=True,
        recognition_postprocess=True,
        recognition_postprocess_kernel_size=5,
        confidence_save_mode='separate_grayscale',
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
        random_artifacts_enabled=True,
        random_artifacts_probability=0.6,
        random_artifacts_count=3,
        random_artifacts_size_ratio=0.2,
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
        dataloader_num_workers=5,
        tech_aug={'enabled': True, 'boundary_aware': {'probability': 0.9}},
        pcb_defects={
            'enabled': True,
            'defect_probability': 0.75,
            'min_defects': 1,
            'max_defects': 2,
            'defect_probabilities': {'break': 1.0, 'short': 0.0},
        },
    )
    save_settings_state(state)
    loaded = load_settings_state()
    assert loaded.step == 77
    assert loaded.sample_size == (11, 22)
    assert loaded.sync_patch_sizes is False
    assert loaded.recognition_jpeg_quality == 89
    assert loaded.recognition_multiprocessing_enabled is False
    assert loaded.shuffle is False
    assert loaded.shuffle_patches_in_frame is True
    assert loaded.random_crop is True
    assert loaded.crops_per_image == 21
    assert loaded.validation_source == 'external'
    assert loaded.validation_image_folder == 'val_images'
    assert loaded.validation_label_folder == 'val_labels'
    assert loaded.save_validation_binary_images is True
    assert loaded.optimizer_name == 'adamw'
    assert loaded.mixed_precision == 'fp16'
    assert loaded.loss_function == 'bce_dice'
    assert loaded.loss_term_weights == {'bce': 0.4, 'dice': 0.6}
    assert loaded.dice_loss_weight == 0.6
    assert loaded.iou_loss_weight == 0.4
    assert loaded.learning_rate == 0.0002
    assert loaded.weight_decay == 0.01
    assert loaded.deep_supervision is False
    assert loaded.warmup_enabled is True
    assert loaded.warmup_epochs == 5
    assert loaded.warmup_start_factor == 0.25
    assert loaded.scheduler_name == 'cosine_annealing'
    assert loaded.scheduler_cosine_t_max == 12
    assert loaded.scheduler_cosine_eta_min == pytest.approx(2e-5)
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
    assert loaded.recognition_tta_enabled is True
    assert loaded.confidence_tta_enabled is True
    assert loaded.recognition_postprocess is True
    assert loaded.recognition_postprocess_kernel_size == 5
    assert loaded.confidence_save_mode == 'separate_grayscale'
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
    assert loaded.random_artifacts_enabled is True
    assert loaded.random_artifacts_probability == pytest.approx(0.6)
    assert loaded.random_artifacts_count == 3
    assert loaded.random_artifacts_size_ratio == pytest.approx(0.2)
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
    assert loaded.dataloader_num_workers == 5
    assert loaded.tech_aug == {'enabled': True, 'boundary_aware': {'probability': 0.9}}
    assert loaded.pcb_defects['enabled'] is True
    assert loaded.pcb_defects['defect_probability'] == pytest.approx(0.75)
    assert loaded.pcb_defects['defect_probabilities']['short'] == pytest.approx(0.0)


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
        ui_mode='simple',
    )
    save_main_window_state(state)
    loaded = load_main_window_state()

    assert loaded.work_mode == 'train_only'
    assert loaded.source_folder == 'source'
    assert loaded.epochs == 11
    assert loaded.ui_mode == 'simple'


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
        recognition_multiprocessing_enabled=False,
        shuffle=True,
        shuffle_patches_in_frame=False,
        random_crop=False,
        crops_per_image=13,
        model='M 720k',
        use_validation=True,
        validation_percent=25,
        validation_source='external',
        validation_image_folder='val_images_ini',
        validation_label_folder='val_labels_ini',
        save_validation_binary_images=True,
        optimizer_name='adamw_muon',
        mixed_precision='off',
        loss_function='dice',
        loss_term_weights={'dice': 1.0},
        dice_loss_weight=1.0,
        iou_loss_weight=0.3,
        learning_rate=0.0003,
        weight_decay=0.02,
        deep_supervision=False,
        warmup_enabled=True,
        warmup_epochs=3,
        warmup_start_factor=0.1,
        scheduler_name='step_lr',
        scheduler_step_lr_step_size=4,
        scheduler_step_lr_gamma=0.2,
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
        recognition_tta_enabled=True,
        confidence_tta_enabled=False,
        recognition_postprocess=True,
        recognition_postprocess_kernel_size=7,
        confidence_save_mode='separate_grayscale',
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
        random_artifacts_enabled=True,
        random_artifacts_probability=0.5,
        random_artifacts_count=2,
        random_artifacts_size_ratio=0.18,
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
        dataloader_num_workers=3,
        tech_aug={'enabled': True, 'global_width': {'probability': 0.75}},
        pcb_defects={
            'enabled': True,
            'defect_probability': 0.65,
            'min_defects': 1,
            'max_defects': 3,
        },
    )
    save_settings_state(state)
    loaded = load_settings_state()

    assert loaded.step == 55
    assert loaded.sample_size == (64, 96)
    assert loaded.sync_patch_sizes is False
    assert loaded.recognition_jpeg_quality == 91
    assert loaded.recognition_multiprocessing_enabled is False
    assert loaded.shuffle is True
    assert loaded.shuffle_patches_in_frame is False
    assert loaded.random_crop is False
    assert loaded.crops_per_image == 13
    assert loaded.use_validation is True
    assert loaded.validation_percent == 25
    assert loaded.validation_source == 'external'
    assert loaded.validation_image_folder == 'val_images_ini'
    assert loaded.validation_label_folder == 'val_labels_ini'
    assert loaded.save_validation_binary_images is True
    assert loaded.optimizer_name == 'adamw_muon'
    assert loaded.mixed_precision == 'off'
    assert loaded.loss_function == 'dice'
    assert loaded.loss_term_weights == {'dice': 1.0}
    assert loaded.dice_loss_weight == 1.0
    assert loaded.iou_loss_weight == 0.3
    assert loaded.learning_rate == 0.0003
    assert loaded.weight_decay == 0.02
    assert loaded.deep_supervision is False
    assert loaded.warmup_enabled is True
    assert loaded.warmup_epochs == 3
    assert loaded.warmup_start_factor == 0.1
    assert loaded.scheduler_name == 'step_lr'
    assert loaded.scheduler_step_lr_step_size == 4
    assert loaded.scheduler_step_lr_gamma == pytest.approx(0.2)
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
    assert loaded.recognition_tta_enabled is True
    assert loaded.confidence_tta_enabled is False
    assert loaded.recognition_postprocess is True
    assert loaded.recognition_postprocess_kernel_size == 7
    assert loaded.confidence_save_mode == 'separate_grayscale'
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
    assert loaded.random_artifacts_enabled is True
    assert loaded.random_artifacts_probability == pytest.approx(0.5)
    assert loaded.random_artifacts_count == 2
    assert loaded.random_artifacts_size_ratio == pytest.approx(0.18)
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
    assert loaded.dataloader_num_workers == 3
    assert loaded.tech_aug == {'enabled': True, 'global_width': {'probability': 0.75}}
    assert loaded.pcb_defects['enabled'] is True
    assert loaded.pcb_defects['defect_probability'] == pytest.approx(0.65)


def test_workflow_snapshot_roundtrip_and_payload():
    root = make_test_dir("workflow_snapshot")
    sample = root / "images"
    label = root / "labels"
    source = root / "source"
    result = root / "result"
    for path in (sample, label, source, result):
        path.mkdir(parents=True, exist_ok=True)

    main_state = MainWindowState(
        work_mode='train_and_recognition',
        source_folder=str(source),
        result_folder=str(result),
        sample_folder=str(sample),
        label_folder=str(label),
        epochs=12,
    )
    settings_state = SettingsState(
        sample_size=(128, 256),
        train_patch_size=(128, 256),
        recognition_patch_size=(64, 96),
        sync_patch_sizes=False,
        train_batch_size=9,
        recognition_batch_size=5,
        batch_size=9,
        overlap=14,
        recognition_jpeg_quality=88,
        recognition_tta_enabled=True,
        confidence_tta_enabled=True,
        confidence_save_mode='separate_grayscale',
        mixed_precision='fp16',
    )

    snapshot_path = root / "snapshot.json"
    saved_path = save_workflow_snapshot(main_state, settings_state, destination=snapshot_path)
    restored_main, restored_settings = load_workflow_snapshot(snapshot_path)
    payload = json.loads(snapshot_path.read_text(encoding='utf-8'))

    assert saved_path == snapshot_path
    assert restored_main.work_mode == 'train_and_recognition'
    assert restored_main.sample_folder == str(sample)
    assert restored_main.label_folder == str(label)
    assert restored_main.epochs == 12
    assert restored_settings.train_patch_size == (128, 256)
    assert restored_settings.recognition_patch_size == (64, 96)
    assert restored_settings.sync_patch_sizes is False
    assert restored_settings.train_batch_size == 9
    assert restored_settings.recognition_batch_size == 5
    assert restored_settings.recognition_jpeg_quality == 88
    assert restored_settings.recognition_tta_enabled is True
    assert restored_settings.confidence_tta_enabled is True
    assert restored_settings.confidence_save_mode == 'separate_grayscale'
    assert payload["format_version"] == 1
    assert payload["main_window_state"]["sample_path"] == str(sample)
    assert payload["settings_state"]["train_patch_x_size"] == 128
    assert payload["workflow"]["work_mode"] == "train_and_recognition"
    assert payload["workflow"]["training"]["image_path"] == str(sample)
    assert payload["workflow"]["recognition"]["result_folder"] == str(result)
    assert "source_files" not in payload["workflow"]["recognition"]


def test_resolve_workflow_snapshot_path_uses_common_parent():
    root = make_test_dir("workflow_snapshot_common")
    sample = root / "images"
    label = root / "labels"
    sample.mkdir(parents=True, exist_ok=True)
    label.mkdir(parents=True, exist_ok=True)

    main_state = MainWindowState(sample_folder=str(sample), label_folder=str(label))

    assert resolve_workflow_snapshot_path(main_state) == root / WORKFLOW_SNAPSHOT_FILENAME


def test_resolve_workflow_snapshot_path_falls_back_to_sample_parent_when_roots_differ():
    sample_root = make_test_dir("workflow_snapshot_sample_root")
    label_root = make_test_dir("workflow_snapshot_label_root")
    sample = sample_root / "images"
    label = label_root / "labels"
    sample.mkdir(parents=True, exist_ok=True)
    label.mkdir(parents=True, exist_ok=True)

    main_state = MainWindowState(sample_folder=str(sample), label_folder=str(label))

    assert resolve_workflow_snapshot_path(main_state) == sample.parent / WORKFLOW_SNAPSHOT_FILENAME


@pytest.mark.parametrize(
    "payload",
    [
        "not-json",
        {"format_version": 99},
        {"format_version": 1, "main_window_state": {}},
    ],
)
def test_load_workflow_snapshot_rejects_invalid_payload(payload):
    root = make_test_dir("workflow_snapshot_invalid")
    snapshot_path = root / "invalid.json"
    if isinstance(payload, str):
        snapshot_path.write_text(payload, encoding='utf-8')
    else:
        snapshot_path.write_text(json.dumps(payload), encoding='utf-8')

    with pytest.raises(ValueError):
        load_workflow_snapshot(snapshot_path)


def test_load_main_window_state_defaults_to_simple_ui_mode(monkeypatch):
    settings_dir = make_test_dir("qsettings_main_default_ui_mode")
    monkeypatch.setenv("NEURALIMAGE_SETTINGS_DIR", str(settings_dir))

    loaded = load_main_window_state()

    assert loaded.ui_mode == 'simple'
