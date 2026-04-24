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


def test_run_web_ui_applies_migrations_before_runserver(monkeypatch):
    calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    class _ManagementModule:
        @staticmethod
        def execute_from_command_line(args):
            calls.append(('execute_from_command_line', tuple(args), {}))

    monkeypatch.setattr(main_module.importlib, 'import_module', lambda name: _ManagementModule)
    monkeypatch.delenv('DJANGO_SETTINGS_MODULE', raising=False)

    main_module._run_web_ui('0.0.0.0', 8123)

    assert calls == [
        ('execute_from_command_line', ('manage.py', 'migrate', '--noinput'), {}),
        ('execute_from_command_line', ('manage.py', 'runserver', '0.0.0.0:8123', '--noreload'), {}),
    ]
