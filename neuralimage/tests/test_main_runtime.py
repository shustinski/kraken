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
