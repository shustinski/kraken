from view.window_dataclasses import MainWindowState, SettingsState


def test_main_window_state_defaults():
    state = MainWindowState()
    assert state.epochs == 20
    assert state.work_mode == ''


def test_settings_state_defaults():
    state = SettingsState()
    assert state.sample_size == (256, 256)
    assert state.sample_cut_mode == 'online'
    assert state.warmup_enabled is False
    assert state.early_stopping_enabled is False

