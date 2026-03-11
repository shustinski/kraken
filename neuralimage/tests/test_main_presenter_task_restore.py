import importlib
import sys
import types

import pytest

pytest.importorskip('PyQt6')

from PyQt6.QtWidgets import QApplication

from application.dto import MainWindowState, SettingsState


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
    def load_main_window_state(self) -> MainWindowState:
        return MainWindowState()

    def save_main_window_state(self, state: MainWindowState) -> None:
        return None

    def load_settings_state(self) -> SettingsState:
        return SettingsState()

    def save_settings_state(self, state: SettingsState) -> None:
        return None


@pytest.fixture(scope='module')
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _import_main_presenter_with_stubs(monkeypatch):
    nn_stub = types.ModuleType('model.NeuralNetwork')
    nn_stub.get_registered_models = lambda: {'M 720k': object()}
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
    assert presenter.view.is_batch_preview_enabled() is False

    presenter.view.allow_close()
    presenter.view.close()
