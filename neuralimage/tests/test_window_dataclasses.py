from view.window_dataclasses import MainWindowState, SettingsState


def test_main_window_state_defaults():
    state = MainWindowState()
    assert state.epochs == 20
    assert state.work_mode == ''


def test_settings_state_defaults():
    state = SettingsState()
    assert state.sample_size == (256, 256)
    assert state.sample_cut_mode == 'online'
    assert state.loss_function == 'bce'
    assert state.dice_loss_weight == 0.5
    assert state.iou_loss_weight == 0.5
    assert state.warmup_enabled is False
    assert state.hard_mining_enabled is False
    assert state.hard_mining_strength == 2.0
    assert state.hard_mining_ema_alpha == 0.2
    assert state.skip_uniform_labels is False
    assert state.torch_compile_enabled is True
    assert state.early_stopping_enabled is False

