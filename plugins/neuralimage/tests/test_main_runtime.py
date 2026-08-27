import main as main_module


def test_configure_multiprocessing_start_method_prefers_spawn_on_linux(monkeypatch):
    recorded: list[tuple[str, bool]] = []

    monkeypatch.setattr(main_module.sys, 'platform', 'linux')
    monkeypatch.delenv('NEURALIMAGE_MP_START_METHOD', raising=False)
    monkeypatch.setattr(main_module.mp, 'get_start_method', lambda allow_none=True: None)
    monkeypatch.setattr(
        main_module.mp,
        'set_start_method',
        lambda method, force=False: recorded.append((str(method), bool(force))),
    )

    resolved = main_module._configure_multiprocessing_start_method()

    assert resolved == 'spawn'
    assert recorded == [('spawn', False)]


def test_configure_multiprocessing_start_method_keeps_existing_mode(monkeypatch):
    recorded: list[tuple[str, bool]] = []

    monkeypatch.setattr(main_module.sys, 'platform', 'linux')
    monkeypatch.delenv('NEURALIMAGE_MP_START_METHOD', raising=False)
    monkeypatch.setattr(main_module.mp, 'get_start_method', lambda allow_none=True: 'forkserver')
    monkeypatch.setattr(
        main_module.mp,
        'set_start_method',
        lambda method, force=False: recorded.append((str(method), bool(force))),
    )

    resolved = main_module._configure_multiprocessing_start_method()

    assert resolved == 'forkserver'
    assert recorded == []


def test_parser_exposes_desktop_and_agent_modes_only():
    parser = main_module._build_parser()
    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
    }

    assert '--ui-only' in option_strings
    assert '--kraken-job-manifest' in option_strings
    assert '--web' not in option_strings
    assert '--host' not in option_strings
    assert '--port' not in option_strings
