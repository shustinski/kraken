import importlib
import sys
import types
from pathlib import Path

import pytest

pytest.importorskip('PyQt6')

from PyQt6.QtWidgets import QApplication, QWidget

from neuralimage.application.dto import MainWindowState, SettingsState
from tests.helpers import make_test_dir


class _FakeSampleWorker:
    def __init__(self):
        self.path = None
        self.settings = None

    def set_path(self, path):
        self.path = path

    def set_settings(self, settings):
        self.settings = settings

    def __len__(self):
        return 0


class _FakeGeneralNeuralHandler:
    def __init__(self, *args, **kwargs):
        pass

    def start(self):
        return None

    def stop_execution(self):
        return None


class _FakeStateStore:
    def __init__(self) -> None:
        self.saved_main_state: MainWindowState | None = None
        self.saved_settings_state: SettingsState | None = None

    def load_main_window_state(self) -> MainWindowState:
        return MainWindowState()

    def save_main_window_state(self, state: MainWindowState) -> None:
        self.saved_main_state = state

    def load_settings_state(self) -> SettingsState:
        return SettingsState()

    def save_settings_state(self, state: SettingsState) -> None:
        self.saved_settings_state = state


class _FakeSignal:
    def __init__(self) -> None:
        self._callbacks = []

    def connect(self, callback) -> None:
        self._callbacks.append(callback)


class _FakeHandlerThread:
    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs
        self.ask = _FakeSignal()
        self.finished = _FakeSignal()
        self.started = False

    def start(self) -> None:
        self.started = True


class _FakeRunningHandlerThread:
    def __init__(self) -> None:
        self.stop_called = False
        self.wait_calls: list[int] = []
        self._running = True

    def stop(self) -> None:
        self.stop_called = True

    def isRunning(self) -> bool:
        return self._running

    def wait(self, timeout_ms: int) -> bool:
        self.wait_calls.append(timeout_ms)
        self._running = False
        return True


@pytest.fixture(scope='module')
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _import_main_presenter_with_stubs(monkeypatch):
    nn_stub = types.ModuleType('model.NeuralNetwork')
    nn_stub.get_registered_models = lambda: {'M 720k': object()}
    nn_stub.get_registered_model_names_by_type = lambda: {'stable': ['M 720k']}
    handler_stub = types.ModuleType('model.general_neural_handler')
    handler_stub.GeneralNeuralHandler = _FakeGeneralNeuralHandler
    images_stub = types.ModuleType('lib.images')
    images_stub.SampleWorker = _FakeSampleWorker

    monkeypatch.setitem(sys.modules, 'model.NeuralNetwork', nn_stub)
    monkeypatch.setitem(sys.modules, 'model.general_neural_handler', handler_stub)
    monkeypatch.setitem(sys.modules, 'lib.images', images_stub)
    sys.modules.pop('presenter.main_presenter', None)
    return importlib.import_module('presenter.main_presenter')


def test_main_presenter_restores_task_state_to_main_window_and_settings(qapp, monkeypatch):
    module = _import_main_presenter_with_stubs(monkeypatch)
    presenter = module.MainPresenter(_FakeStateStore())

    restored_main = MainWindowState(
        work_mode='recognition_only',
        source_folder='source-dir',
        result_folder='result-dir',
        label_folder='label-dir',
        sample_folder='sample-dir',
        model_path='model-file.pth',
        epochs=9,
    )
    restored_settings = SettingsState(
        step=72,
        sample_cut_mode='disk',
        train_patch_size=(128, 256),
        recognition_patch_size=(64, 96),
        sync_patch_sizes=False,
        train_batch_size=17,
        recognition_batch_size=11,
        batch_size=17,
        overlap=12,
        recognition_jpeg_quality=91,
        recognition_multiprocessing_enabled=False,
        recognition_tta_enabled=True,
        confidence_tta_enabled=True,
        confidence_save_mode='separate_grayscale',
        deep_supervision=False,
        show_batch_preview=False,
        random_crop=True,
    )

    presenter._restore_task_state_to_ui(restored_main, restored_settings)
    qapp.processEvents()

    assert presenter.view.lbl_source.text() == 'source-dir'
    assert presenter.view.lbl_result.text() == 'result-dir'
    assert presenter.view.label_path.text() == 'label-dir'
    assert presenter.view.sample_path.text() == 'sample-dir'
    assert presenter.view.model_path.text() == 'model-file.pth'
    assert presenter.view.le_epochs.value() == 9
    assert presenter.settings_panel.epochs_spinbox.value() == 9
    assert presenter.view.rb_recognition.isChecked() is True
    assert presenter.settings_panel.shift_spinbox.value() == 72
    assert presenter.settings_panel.cut_dataset_type.isChecked() is True
    assert presenter.settings_panel.sync_patch_sizes_check_box.isChecked() is False
    assert presenter.settings_panel.train_patch_x_size.value() == 128
    assert presenter.settings_panel.train_patch_y_size.value() == 256
    assert presenter.settings_panel.recognition_patch_x_size.value() == 64
    assert presenter.settings_panel.recognition_patch_y_size.value() == 96
    assert presenter.settings_panel.train_batch_spinbox.value() == 17
    assert presenter.settings_panel.recognition_batch_spinbox.value() == 11
    assert presenter.settings_panel.overlap_spinbox.value() == 12
    assert presenter.settings_panel.recognition_jpeg_quality_spinbox.value() == 91
    assert presenter.settings_panel.recognition_multiprocessing_check_box.isChecked() is False
    assert presenter.settings_panel.recognition_tta_check_box.isChecked() is True
    assert presenter.settings_panel.get_confidence_export_mode_value() == 'tta'
    assert presenter.settings_panel.get_confidence_save_mode_value() == 'separate_grayscale'
    assert presenter.settings_panel.deep_supervision_check_box.isChecked() is False
    assert presenter.view.is_batch_preview_enabled() is False

    presenter.view.allow_close()
    presenter.view.close()


def test_main_presenter_can_restore_ui_from_workflow_snapshot_file(qapp, monkeypatch):
    module = _import_main_presenter_with_stubs(monkeypatch)
    state_store = _FakeStateStore()
    presenter = module.MainPresenter(state_store)

    restored_main = MainWindowState(
        work_mode='train_only',
        source_folder='restored-source',
        result_folder='restored-result',
        label_folder='restored-labels',
        sample_folder='restored-samples',
        epochs=13,
    )
    restored_settings = SettingsState(
        step=44,
        train_patch_size=(160, 224),
        recognition_patch_size=(96, 112),
        sync_patch_sizes=False,
        train_batch_size=7,
        recognition_batch_size=3,
        batch_size=7,
        recognition_threshold=0.62,
        recognition_use_auto_threshold=False,
        recognition_tta_enabled=True,
        confidence_tta_enabled=False,
        confidence_save_mode='separate_grayscale',
        deep_supervision=False,
    )
    monkeypatch.setattr(module, '_tk_filedialog', lambda *_args, **_kwargs: 'workflow.json')
    monkeypatch.setattr(module, 'load_workflow_snapshot', lambda *_args, **_kwargs: (restored_main, restored_settings))

    presenter._on_open_config_requested()
    qapp.processEvents()

    assert presenter.view.sample_path.text() == 'restored-samples'
    assert presenter.view.label_path.text() == 'restored-labels'
    assert presenter.view.le_epochs.value() == 13
    assert presenter.settings_panel.epochs_spinbox.value() == 13
    assert presenter.view.rb_train_only.isChecked() is True
    assert presenter.settings_panel.train_patch_x_size.value() == 160
    assert presenter.settings_panel.train_patch_y_size.value() == 224
    assert presenter.settings_panel.recognition_patch_x_size.value() == 96
    assert presenter.settings_panel.recognition_patch_y_size.value() == 112
    assert presenter.settings_panel.train_batch_spinbox.value() == 7
    assert presenter.settings_panel.recognition_batch_spinbox.value() == 3
    assert presenter.settings_panel.recognition_threshold_spinbox.value() == pytest.approx(0.62)
    assert presenter.settings_panel.recognition_use_auto_threshold_check_box.isChecked() is False
    assert presenter.settings_panel.recognition_tta_check_box.isChecked() is True
    assert presenter.settings_panel.get_confidence_export_mode_value() == 'model_output'
    assert presenter.settings_panel.get_confidence_save_mode_value() == 'separate_grayscale'
    assert presenter.settings_panel.deep_supervision_check_box.isChecked() is False
    assert state_store.saved_main_state is not None
    assert state_store.saved_settings_state is not None

    presenter.view.allow_close()
    presenter.view.close()


def test_main_presenter_shows_warning_for_invalid_workflow_snapshot(qapp, monkeypatch):
    module = _import_main_presenter_with_stubs(monkeypatch)
    presenter = module.MainPresenter(_FakeStateStore())
    warnings: list[str] = []
    presenter.view.show_warning.connect(warnings.append)
    presenter.view.set_jpg_path('before-open')

    monkeypatch.setattr(module, '_tk_filedialog', lambda *_args, **_kwargs: 'broken.json')
    monkeypatch.setattr(
        module,
        'load_workflow_snapshot',
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError('bad payload')),
    )

    presenter._on_open_config_requested()
    qapp.processEvents()

    assert warnings
    assert 'bad payload' in warnings[-1]
    assert presenter.view.sample_path.text() == 'before-open'

    presenter.view.allow_close()
    presenter.view.close()


@pytest.mark.parametrize(
    ("work_mode", "should_save_snapshot"),
    [
        ('train_only', True),
        ('recognition_only', False),
    ],
)
def test_main_presenter_start_task_saves_snapshot_only_for_training_modes(qapp, monkeypatch, work_mode, should_save_snapshot):
    module = _import_main_presenter_with_stubs(monkeypatch)
    presenter = module.MainPresenter(_FakeStateStore())
    monkeypatch.setattr(module, 'GeneralNeuralHandlerThread', _FakeHandlerThread)

    calls: list[tuple[MainWindowState, SettingsState, tuple[object, object, object]]] = []

    def _capture_snapshot(main_state, settings_state, workflow_snapshot=None, **_kwargs):
        calls.append((main_state, settings_state, workflow_snapshot))
        return Path("snapshot.json")

    monkeypatch.setattr(module, 'save_workflow_snapshot', _capture_snapshot)

    root = make_test_dir(f"presenter_start_task_{work_mode}")
    sample = root / "sample"
    label = root / "label"
    source = root / "source"
    result = root / "result"
    for path in (sample, label, source, result):
        path.mkdir(parents=True, exist_ok=True)

    task = types.SimpleNamespace(
        task_id=5,
        main_window_state=MainWindowState(
            work_mode=work_mode,
            sample_folder=str(sample),
            label_folder=str(label),
            source_folder=str(source),
            result_folder=str(result),
            model_path=str(result / "model.pth"),
            epochs=2,
        ),
        settings_state=SettingsState(),
    )

    presenter._start_task(task)
    qapp.processEvents()

    assert len(calls) == int(should_save_snapshot)
    if should_save_snapshot:
        assert calls[0][2][0].value == work_mode
    assert isinstance(presenter.neuaral_handler, _FakeHandlerThread)
    assert presenter.neuaral_handler.started is True

    presenter.view.allow_close()
    presenter.view.close()


def test_main_presenter_shutdown_stops_active_worker_and_owned_threads(qapp, monkeypatch):
    module = _import_main_presenter_with_stubs(monkeypatch)
    presenter = module.MainPresenter(_FakeStateStore())
    active_handler = _FakeRunningHandlerThread()
    update_thread = _FakeRunningHandlerThread()
    download_thread = _FakeRunningHandlerThread()
    rare_patch_thread = _FakeRunningHandlerThread()
    presenter.neuaral_handler = active_handler
    presenter._update_check_thread = update_thread
    presenter._update_download_thread = download_thread
    presenter._rare_patch_editor_prepare_thread = rare_patch_thread

    presenter.shutdown(wait_ms=25)
    presenter.shutdown(wait_ms=25)
    qapp.processEvents()

    assert active_handler.stop_called is True
    assert active_handler.wait_calls == [25]
    assert update_thread.stop_called is True
    assert update_thread.wait_calls == [25]
    assert download_thread.stop_called is True
    assert download_thread.wait_calls == [25]
    assert rare_patch_thread.stop_called is True
    assert rare_patch_thread.wait_calls == [25]
    assert presenter.neuaral_handler is None
    assert presenter._update_check_thread is None
    assert presenter._update_download_thread is None
    assert presenter._rare_patch_editor_prepare_thread is None
    assert presenter._shutdown_requested is True

    presenter.view.allow_close()
    presenter.view.close()


def test_main_presenter_keeps_selected_simple_profile_label_after_preset_load(qapp, monkeypatch):
    module = _import_main_presenter_with_stubs(monkeypatch)
    presenter = module.MainPresenter(_FakeStateStore())

    restored_main = MainWindowState(work_mode='recognition_only', ui_mode='simple')
    restored_settings = SettingsState()
    monkeypatch.setattr(module, 'load_workflow_snapshot', lambda *_args, **_kwargs: (restored_main, restored_settings))

    presenter._on_simple_workflow_requested('contacts')
    qapp.processEvents()

    assert presenter.view.btn_simple_contacts.text() in presenter.view.simple_workflow_label.text()

    presenter.view.allow_close()
    presenter.view.close()


class _FakeValidationGradientLitePlugin:
    instances: list["_FakeValidationGradientLitePlugin"] = []

    plugin_id = "validation_gradient_widget_lite"
    display_name = "Validation Gradient Widget Lite"

    def __init__(self) -> None:
        self.shutdown_called = False
        self.widget = None
        type(self).instances.append(self)

    def create_widget(self, host=None, parent=None):
        self.widget = QWidget(parent)
        return self.widget

    def shutdown(self) -> None:
        self.shutdown_called = True


def test_main_presenter_opens_validation_gradient_plugin_window(qapp, monkeypatch):
    module = _import_main_presenter_with_stubs(monkeypatch)
    import neuralimage.Validation_gradient_widget_lite as lite_pkg
    monkeypatch.setattr(lite_pkg, 'ValidationGradientLitePlugin', _FakeValidationGradientLitePlugin)
    presenter = module.MainPresenter(_FakeStateStore())

    presenter._on_open_validation_gradient_requested()
    qapp.processEvents()

    window = presenter._validation_gradient_window
    assert window is not None
    assert window.isVisible() is True
    assert isinstance(presenter._validation_gradient_plugin, _FakeValidationGradientLitePlugin)

    first_plugin = presenter._validation_gradient_plugin
    presenter._on_open_validation_gradient_requested()
    qapp.processEvents()

    assert presenter._validation_gradient_plugin is first_plugin

    window.close()
    qapp.processEvents()

    assert first_plugin.shutdown_called is True
    assert presenter._validation_gradient_window is None
    assert presenter._validation_gradient_plugin is None

    presenter.view.allow_close()
    presenter.view.close()
