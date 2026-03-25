from application.dto import MainWindowState, SettingsState
from application.services.workflow_mapper import build_workflow_parameters, resolve_work_mode
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
        loss_term_weights={'bce': 0.35, 'dice': 0.65},
        dice_loss_weight=0.7,
        iou_loss_weight=0.2,
        validation_source='external',
        validation_image_folder=str(result / 'val_images'),
        validation_label_folder=str(result / 'val_labels'),
        save_validation_binary_images=True,
        warmup_enabled=True,
        warmup_epochs=4,
        warmup_start_factor=0.2,
        scheduler_name='reduce_on_plateau',
        scheduler_plateau_factor=0.4,
        scheduler_plateau_patience=5,
        scheduler_plateau_threshold=0.002,
        scheduler_plateau_min_lr=1e-5,
        scheduler_plateau_cooldown=2,
        hard_mining_enabled=True,
        hard_mining_strength=3.0,
        hard_mining_ema_alpha=0.35,
        hard_pixel_mining_enabled=True,
        hard_pixel_mining_ratio=0.2,
        cutout_enabled=True,
        cutout_probability=0.85,
        cutout_holes=3,
        cutout_size_ratio=0.3,
        random_artifacts_enabled=True,
        random_artifacts_probability=0.55,
        random_artifacts_count=2,
        random_artifacts_size_ratio=0.22,
        mixup_enabled=True,
        mixup_probability=0.65,
        mixup_alpha=0.4,
        skip_uniform_labels=True,
        rare_patch_oversampling_enabled=True,
        rare_patch_oversampling_factor=6,
        early_stopping_enabled=True,
        early_stopping_patience=7,
        early_stopping_min_delta=0.005,
        early_stopping_restore_best_weights=False,
    )

    _, training, _ = build_workflow_parameters(main, settings)

    assert training.optimizer.name.value == 'adam'
    assert training.loss_function == 'bce_dice'
    assert training.loss_term_weights == {'bce': 0.35, 'dice': 0.65}
    assert training.dice_loss_weight == 0.7
    assert training.iou_loss_weight == 0.2
    assert training.validation_source == 'external'
    assert training.validation_image_path == result / 'val_images'
    assert training.validation_label_path == result / 'val_labels'
    assert training.save_validation_binary_images is True
    assert training.warmup.enabled is True
    assert training.warmup.epochs == 4
    assert training.warmup.start_factor == 0.2
    assert training.scheduler.name.value == 'reduce_on_plateau'
    assert training.scheduler.plateau_factor == 0.4
    assert training.scheduler.plateau_patience == 5
    assert training.scheduler.plateau_threshold == 0.002
    assert training.scheduler.plateau_min_lr == 1e-5
    assert training.scheduler.plateau_cooldown == 2
    assert training.hard_mining.enabled is True
    assert training.hard_mining.strength == 3.0
    assert training.hard_mining.ema_alpha == 0.35
    assert training.hard_mining.pixel_enabled is True
    assert training.hard_mining.pixel_keep_ratio == 0.2
    assert training.cutout.enabled is True
    assert training.cutout.probability == 0.85
    assert training.cutout.holes == 3
    assert training.cutout.size_ratio == 0.3
    assert training.random_artifacts.enabled is True
    assert training.random_artifacts.probability == 0.55
    assert training.random_artifacts.count == 2
    assert training.random_artifacts.size_ratio == 0.22
    assert training.mixup.enabled is True
    assert training.mixup.probability == 0.65
    assert training.mixup.alpha == 0.4
    assert training.skip_uniform_labels is True
    assert training.rare_patch_oversampling_enabled is True
    assert training.rare_patch_oversampling_factor == 6
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
        augmentation_gamma_strength=0.17,
        augmentation_noise_probability=0.65,
        augmentation_noise_sigma=0.02,
        augmentation_blur_probability=0.35,
        augmentation_blur_radius=1.4,
        edge_cut_size=12,
        target_size=(1024, 768),
    )

    _, training, _ = build_workflow_parameters(main, settings)

    assert training.prepare.enable_crop is True
    assert training.prepare.enable_resize is False
    assert training.generation.additional_augmentation is True
    assert training.generation.augmentation_brightness_strength == 0.2
    assert training.generation.augmentation_contrast_strength == 0.15
    assert training.generation.augmentation_gamma_strength == 0.17
    assert training.generation.augmentation_noise_probability == 0.65
    assert training.generation.augmentation_noise_sigma == 0.02
    assert training.generation.augmentation_blur_probability == 0.35
    assert training.generation.augmentation_blur_radius == 1.4
    assert training.prepare.edge_cut == (12, 12)
    assert training.prepare.target_size == (1024, 768)


def test_build_workflow_parameters_maps_frame_and_patch_shuffle_flags_separately():
    source = make_test_dir("workflow_source_shuffle")
    result = make_test_dir("workflow_result_shuffle")
    sample = make_test_dir("workflow_sample_shuffle")
    label = make_test_dir("workflow_label_shuffle")

    main = MainWindowState(
        work_mode='train_only',
        source_folder=str(source),
        result_folder=str(result),
        sample_folder=str(sample),
        label_folder=str(label),
        epochs=1,
    )
    settings = SettingsState(
        shuffle=False,
        shuffle_patches_in_frame=True,
        random_crop=True,
        crops_per_image=17,
    )

    _, training, _ = build_workflow_parameters(main, settings)

    assert training.shuffle is False
    assert training.generation.shuffle_patches_in_frame is True
    assert training.generation.random_crop is True
    assert training.generation.crops_per_image == 17


def test_build_workflow_parameters_maps_recognition_output_parameters():
    source = make_test_dir("workflow_source_jpeg_quality")
    result = make_test_dir("workflow_result_jpeg_quality")
    sample = make_test_dir("workflow_sample_jpeg_quality")
    label = make_test_dir("workflow_label_jpeg_quality")

    main = MainWindowState(
        work_mode='recognition_only',
        source_folder=str(source),
        result_folder=str(result),
        model_path=str(result / "model.pth"),
        sample_folder=str(sample),
        label_folder=str(label),
        epochs=1,
    )
    settings = SettingsState(
        recognition_jpeg_quality=87,
        recognition_multiprocessing_enabled=False,
        recognition_binarize_output=False,
        recognition_use_auto_threshold=False,
        recognition_threshold=0.61,
        recognition_postprocess=True,
        recognition_postprocess_kernel_size=5,
    )

    _, _, recognition = build_workflow_parameters(main, settings)

    assert recognition.jpeg_quality == 87
    assert recognition.recognition_multiprocessing_enabled is False
    assert recognition.binarize_output is False
    assert recognition.use_auto_threshold is False
    assert recognition.threshold == 0.61
    assert recognition.postprocess_enabled is True
    assert recognition.postprocess_kernel_size == 5


def test_build_workflow_parameters_defers_recognition_source_indexing():
    source = make_test_dir("workflow_source_lazy_recognition")
    result = make_test_dir("workflow_result_lazy_recognition")
    sample = make_test_dir("workflow_sample_lazy_recognition")
    label = make_test_dir("workflow_label_lazy_recognition")
    (source / "frame_001.png").write_bytes(b"not-used")

    main = MainWindowState(
        work_mode='recognition_only',
        source_folder=str(source),
        result_folder=str(result),
        model_path=str(result / "model.pth"),
        sample_folder=str(sample),
        label_folder=str(label),
        epochs=1,
    )
    settings = SettingsState()

    _, _, recognition = build_workflow_parameters(main, settings)

    assert recognition.source_folder == source
    assert recognition.source_files == []


def test_build_workflow_parameters_maps_dataloader_num_workers():
    source = make_test_dir("workflow_source_dataloader_workers")
    result = make_test_dir("workflow_result_dataloader_workers")
    sample = make_test_dir("workflow_sample_dataloader_workers")
    label = make_test_dir("workflow_label_dataloader_workers")

    main = MainWindowState(
        work_mode='train_only',
        source_folder=str(source),
        result_folder=str(result),
        sample_folder=str(sample),
        label_folder=str(label),
        epochs=1,
    )
    settings = SettingsState(dataloader_num_workers=6)

    _, training, _ = build_workflow_parameters(main, settings)

    assert training.dataloader_num_workers == 6


def test_build_workflow_parameters_maps_one_cycle_scheduler():
    source = make_test_dir("workflow_source_scheduler")
    result = make_test_dir("workflow_result_scheduler")
    sample = make_test_dir("workflow_sample_scheduler")
    label = make_test_dir("workflow_label_scheduler")

    main = MainWindowState(
        work_mode='train_only',
        source_folder=str(source),
        result_folder=str(result),
        sample_folder=str(sample),
        label_folder=str(label),
        epochs=3,
    )
    settings = SettingsState(
        scheduler_name='one_cycle',
        scheduler_one_cycle_max_lr=0.002,
        scheduler_one_cycle_pct_start=0.4,
        scheduler_one_cycle_anneal_strategy='linear',
        scheduler_one_cycle_div_factor=10.0,
        scheduler_one_cycle_final_div_factor=500.0,
        scheduler_one_cycle_three_phase=True,
    )

    _, training, _ = build_workflow_parameters(main, settings)

    assert training.scheduler.name.value == 'one_cycle'
    assert training.scheduler.one_cycle_max_lr == 0.002
    assert training.scheduler.one_cycle_pct_start == 0.4
    assert training.scheduler.one_cycle_anneal_strategy == 'linear'
    assert training.scheduler.one_cycle_div_factor == 10.0
    assert training.scheduler.one_cycle_final_div_factor == 500.0
    assert training.scheduler.one_cycle_three_phase is True


def test_build_workflow_parameters_defaults_to_split_validation_source():
    source = make_test_dir("workflow_source_validation_default")
    result = make_test_dir("workflow_result_validation_default")
    sample = make_test_dir("workflow_sample_validation_default")
    label = make_test_dir("workflow_label_validation_default")

    main = MainWindowState(
        work_mode='train_only',
        source_folder=str(source),
        result_folder=str(result),
        sample_folder=str(sample),
        label_folder=str(label),
        epochs=1,
    )
    settings = SettingsState(use_validation=True, validation_percent=15)

    _, training, _ = build_workflow_parameters(main, settings)

    assert training.validation is True
    assert training.validation_percent == 15
    assert training.validation_source == 'split'
    assert training.validation_image_path is None
    assert training.validation_label_path is None
    assert training.save_validation_binary_images is False


def test_build_workflow_parameters_syncs_patch_sizes_when_enabled():
    source = make_test_dir("workflow_source_sync_patch")
    result = make_test_dir("workflow_result_sync_patch")
    sample = make_test_dir("workflow_sample_sync_patch")
    label = make_test_dir("workflow_label_sync_patch")

    main = MainWindowState(
        work_mode='train_and_recognition',
        source_folder=str(source),
        result_folder=str(result),
        sample_folder=str(sample),
        label_folder=str(label),
        epochs=1,
    )
    settings = SettingsState(
        train_patch_size=(192, 128),
        recognition_patch_size=(320, 224),
        sync_patch_sizes=True,
    )

    _, _, recognition = build_workflow_parameters(main, settings)

    assert recognition.part_size == (192, 128)


def test_build_workflow_parameters_keeps_recognition_patch_size_when_sync_disabled():
    source = make_test_dir("workflow_source_no_sync_patch")
    result = make_test_dir("workflow_result_no_sync_patch")
    sample = make_test_dir("workflow_sample_no_sync_patch")
    label = make_test_dir("workflow_label_no_sync_patch")

    main = MainWindowState(
        work_mode='train_and_recognition',
        source_folder=str(source),
        result_folder=str(result),
        sample_folder=str(sample),
        label_folder=str(label),
        epochs=1,
    )
    settings = SettingsState(
        train_patch_size=(192, 128),
        recognition_patch_size=(320, 224),
        sync_patch_sizes=False,
    )

    _, _, recognition = build_workflow_parameters(main, settings)

    assert recognition.part_size == (320, 224)


def test_build_workflow_parameters_maps_pcb_defects():
    source = make_test_dir("workflow_source_pcb_defects")
    result = make_test_dir("workflow_result_pcb_defects")
    sample = make_test_dir("workflow_sample_pcb_defects")
    label = make_test_dir("workflow_label_pcb_defects")

    main = MainWindowState(
        work_mode='train_only',
        source_folder=str(source),
        result_folder=str(result),
        sample_folder=str(sample),
        label_folder=str(label),
        epochs=1,
    )
    settings = SettingsState(
        pcb_defects={
            'enabled': True,
            'defect_probability': 0.8,
            'min_defects': 2,
            'max_defects': 4,
            'use_input_mask': True,
            'use_defect_mask_as_label': True,
            'defect_probabilities': {
                'break': 1.0,
                'short': 0.0,
            },
        }
    )

    _, training, _ = build_workflow_parameters(main, settings)

    assert training.pcb_defects.enabled is True
    assert training.pcb_defects.defect_probability == 0.8
    assert training.pcb_defects.min_defects == 2
    assert training.pcb_defects.max_defects == 4
    assert training.pcb_defects.use_input_mask is True
    assert training.pcb_defects.use_defect_mask_as_label is True
    assert training.pcb_defects.defect_probabilities['break'] == 1.0
    assert training.pcb_defects.defect_probabilities['short'] == 0.0
