from presenter.workflow_mapper import resolve_work_mode
from presenter.workflow_mapper import build_workflow_parameters
from view.window_dataclasses import MainWindowState, SettingsState
from tests.helpers import make_test_dir


def test_resolve_work_mode_known_value():
    mode = resolve_work_mode('train_only')
    assert mode is not None
    assert mode.value == 'train_only'


def test_resolve_work_mode_unknown_value():
    assert resolve_work_mode('unknown') is None


def test_resolve_work_mode_legacy_value_aliases():
    mode = resolve_work_mode('recognintion_only')
    assert mode is not None
    assert mode.value == 'recognition_only'


def test_build_workflow_parameters_falls_back_to_adam_for_unknown_optimizer():
    source = make_test_dir("workflow_source")
    result = make_test_dir("workflow_result")
    sample = make_test_dir("workflow_sample")
    label = make_test_dir("workflow_label")

    main = MainWindowState(
        work_mode='train_only',
        source_folder=str(source),
        result_folder=str(result),
        sample_folder=str(sample),
        label_folder=str(label),
        epochs=1,
    )
    settings = SettingsState(
        optimizer_name='invalid_optimizer_name',
        loss_function='bce_dice',
        dice_loss_weight=0.7,
        iou_loss_weight=0.2,
        warmup_enabled=True,
        warmup_epochs=4,
        warmup_start_factor=0.2,
        hard_mining_enabled=True,
        hard_mining_strength=3.0,
        hard_mining_ema_alpha=0.35,
        skip_uniform_labels=True,
        early_stopping_enabled=True,
        early_stopping_patience=7,
        early_stopping_min_delta=0.005,
        early_stopping_restore_best_weights=False,
    )

    _, training, _ = build_workflow_parameters(main, settings)

    assert training.optimizer.name.value == 'adam'
    assert training.loss_function == 'bce_dice'
    assert training.dice_loss_weight == 0.7
    assert training.iou_loss_weight == 0.2
    assert training.warmup.enabled is True
    assert training.warmup.epochs == 4
    assert training.warmup.start_factor == 0.2
    assert training.hard_mining.enabled is True
    assert training.hard_mining.strength == 3.0
    assert training.hard_mining.ema_alpha == 0.35
    assert training.skip_uniform_labels is True
    assert training.early_stopping.enabled is True
    assert training.early_stopping.patience == 7
    assert training.early_stopping.min_delta == 0.005
    assert training.early_stopping.restore_best_weights is False


def test_build_workflow_parameters_maps_separate_crop_and_resize_flags():
    source = make_test_dir("workflow_source_flags")
    result = make_test_dir("workflow_result_flags")
    sample = make_test_dir("workflow_sample_flags")
    label = make_test_dir("workflow_label_flags")

    main = MainWindowState(
        work_mode='train_only',
        source_folder=str(source),
        result_folder=str(result),
        sample_folder=str(sample),
        label_folder=str(label),
        epochs=1,
    )
    settings = SettingsState(
        crop_enabled=True,
        resize_enabled=False,
        additional_augmentation=True,
        augmentation_brightness_strength=0.2,
        augmentation_contrast_strength=0.15,
        augmentation_noise_probability=0.65,
        augmentation_noise_sigma=0.02,
        edge_cut_size=12,
        target_size=(1024, 768),
    )

    _, training, _ = build_workflow_parameters(main, settings)

    assert training.prepare.enable_crop is True
    assert training.prepare.enable_resize is False
    assert training.generation.additional_augmentation is True
    assert training.generation.augmentation_brightness_strength == 0.2
    assert training.generation.augmentation_contrast_strength == 0.15
    assert training.generation.augmentation_noise_probability == 0.65
    assert training.generation.augmentation_noise_sigma == 0.02
    assert training.prepare.edge_cut == (12, 12)
    assert training.prepare.target_size == (1024, 768)
