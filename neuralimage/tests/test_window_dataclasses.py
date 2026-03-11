from view.window_dataclasses import MainWindowState, SettingsState


def test_main_window_state_defaults():
    state = MainWindowState()
    assert state.epochs == 20
    assert state.work_mode == ''


def test_settings_state_defaults():
    state = SettingsState()
    assert state.sample_size == (256, 256)
    assert state.sample_cut_mode == 'online'
    assert state.random_crop is False
    assert state.crops_per_image == 64
    assert state.augmentation_gamma_strength == 0.15
    assert state.augmentation_blur_probability == 0.25
    assert state.augmentation_blur_radius == 1.0
    assert state.loss_function == 'bce'
    assert state.loss_term_weights == {'bce': 1.0}
    assert state.recognition_binarize_output is True
    assert state.recognition_use_auto_threshold is True
    assert state.recognition_threshold == 0.5
    assert state.recognition_postprocess is False
    assert state.recognition_postprocess_kernel_size == 3
    assert state.dice_loss_weight == 0.5
    assert state.iou_loss_weight == 0.5
    assert state.warmup_enabled is False
    assert state.hard_mining_enabled is False
    assert state.hard_mining_strength == 2.0
    assert state.hard_mining_ema_alpha == 0.2
    assert state.hard_pixel_mining_enabled is False
    assert state.hard_pixel_mining_ratio == 0.25
    assert state.cutout_enabled is False
    assert state.cutout_probability == 1.0
    assert state.cutout_holes == 1
    assert state.cutout_size_ratio == 0.25
    assert state.mixup_enabled is False
    assert state.mixup_probability == 1.0
    assert state.mixup_alpha == 0.2
    assert state.skip_uniform_labels is False
    assert state.rare_patch_oversampling_enabled is False
    assert state.rare_patch_oversampling_factor == 2
    assert state.torch_compile_enabled is True
    assert state.early_stopping_enabled is False

