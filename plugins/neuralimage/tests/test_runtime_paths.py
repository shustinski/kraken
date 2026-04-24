import importlib.util
import sys
import types
from pathlib import Path

from neuralimage.lib import runtime_paths


def test_resolve_resource_path_uses_project_resources_in_source_mode(monkeypatch):
    monkeypatch.delattr(sys, 'frozen', raising=False)

    resolved = runtime_paths.resolve_resource_path('conductors_workflow.json')

    assert resolved == runtime_paths.project_root() / 'resources' / 'conductors_workflow.json'


def test_resolve_resource_path_uses_internal_bundle_in_frozen_mode(tmp_path, monkeypatch):
    executable_path = tmp_path / 'NeuralImage.exe'
    executable_path.write_text('', encoding='utf-8')
    bundled_internal = tmp_path / '_internal' / 'resources'
    bundled_internal.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(sys, 'frozen', True, raising=False)
    monkeypatch.setattr(sys, 'executable', str(executable_path))
    monkeypatch.delattr(sys, '_MEIPASS', raising=False)

    resolved = runtime_paths.resolve_resource_path('conductors_workflow.json')

    assert resolved == bundled_internal / 'conductors_workflow.json'


def test_main_module_does_not_import_controller_at_module_import_time(monkeypatch):
    module_name = '_test_main_lazy_import'
    main_path = Path(__file__).resolve().parent.parent / 'main.py'
    dummy_controller = types.ModuleType('controller')

    monkeypatch.setitem(sys.modules, 'controller', dummy_controller)
    sys.modules.pop(module_name, None)

    spec = importlib.util.spec_from_file_location(module_name, main_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert hasattr(module, 'main')


def test_main_module_restores_missing_standard_streams(monkeypatch):
    module_name = '_test_main_streams'
    main_path = Path(__file__).resolve().parent.parent / 'main.py'

    monkeypatch.setattr(sys, 'stdout', None, raising=False)
    monkeypatch.setattr(sys, 'stderr', None, raising=False)
    sys.modules.pop(module_name, None)

    spec = importlib.util.spec_from_file_location(module_name, main_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert sys.stdout is not None
    assert sys.stderr is not None
