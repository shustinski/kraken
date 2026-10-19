"""test_settings_optimizer_ui: test main presenter leaves."""
from __future__ import annotations
import sys
import types
import numpy as np
from PIL import Image
from neuralimage.lib.data_interfaces import WorkMode
from tests.helpers import make_test_dir
from neuralimage.view.window_dataclasses import SettingsState

from test_settings_optimizer_ui_parts.preprocessing_groupbox_controls import _import_main_presenter_with_stubs

def test_main_presenter_opens_augmentation_preview_dialog(qapp):
    module = _import_main_presenter_with_stubs()
    presenter = module.MainPresenter.__new__(module.MainPresenter)
    from PyQt6.QtCore import QObject
    QObject.__init__(presenter)
    presenter.settings_panel = object()

    sample_root = make_test_dir('augmentation_preview_presenter')
    sample_dir = sample_root / 'samples'
    label_dir = sample_root / 'labels'
    sample_dir.mkdir()
    label_dir.mkdir()

    image = np.zeros((32, 32), dtype=np.uint8)
    image[4:20, 6:12] = 255
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:24, 10:22] = 255
    Image.fromarray(image, mode='L').save(sample_dir / 'frame.png')
    Image.fromarray(mask, mode='L').save(label_dir / 'frame.png')

    opened: dict[str, object] = {}

    class _FakeDestroyed:
        def connect(self, _callback):
            opened['connected'] = True

    class _FakeSignal:
        def connect(self, _callback):
            opened['apply_connected'] = True

    class _FakeDialog:
        def __init__(self, training_parameters, parent):
            opened['training_parameters'] = training_parameters
            opened['parent'] = parent
            self.destroyed = _FakeDestroyed()
            self.apply_to_main_requested = _FakeSignal()

        def show(self):
            opened['show_called'] = True

        def raise_(self):
            opened['raise_called'] = True

        def activateWindow(self):
            opened['activate_called'] = True

    sys.modules['neuralimage.view.augmentation_preview_dialog'] = types.SimpleNamespace(
        AugmentationPreviewDialog=_FakeDialog
    )

    presenter.main_window_state = module.MainWindowState(
        work_mode=WorkMode.train_only.value,
        sample_folder=str(sample_dir),
        label_folder=str(label_dir),
        epochs=1,
    )
    presenter.settings_state = SettingsState()
    presenter.view = types.SimpleNamespace(
        show_warning=types.SimpleNamespace(emit=lambda message: opened.__setitem__('warning', message))
    )
    presenter._augmentation_preview_dialog = None
    presenter._update_main_window_state = lambda: None
    presenter._update_settings_window_state = lambda: None

    module.MainPresenter._open_augmentation_preview(presenter)

    training_parameters = opened['training_parameters']
    assert training_parameters.image_path == sample_dir
    assert training_parameters.label_path == label_dir
    assert opened.get('warning') is None
    assert opened.get('apply_connected') is True
    assert opened.get('show_called') is True
    assert opened.get('raise_called') is True
    assert opened.get('activate_called') is True
    sys.modules.pop('neuralimage.view.augmentation_preview_dialog', None)

def test_main_presenter_leaves_cif_masks_for_lazy_augmentation_preview_loading(qapp):
    module = _import_main_presenter_with_stubs()
    presenter = module.MainPresenter.__new__(module.MainPresenter)
    from PyQt6.QtCore import QObject
    QObject.__init__(presenter)
    presenter.settings_panel = object()

    sample_root = make_test_dir('augmentation_preview_presenter_cif')
    sample_dir = sample_root / 'samples'
    label_dir = sample_root / 'labels'
    raster_dir = sample_root / 'binary_cif'
    sample_dir.mkdir()
    label_dir.mkdir()
    raster_dir.mkdir()

    image = np.zeros((32, 32), dtype=np.uint8)
    image[6:22, 8:18] = 255
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[10:24, 12:26] = 255
    Image.fromarray(image, mode='L').save(sample_dir / 'frame.png')
    Image.fromarray(mask, mode='L').save(raster_dir / 'frame.png')
    (label_dir / 'frame.cif').write_text('dummy', encoding='utf-8')

    opened: dict[str, object] = {}

    class _FakeDestroyed:
        def connect(self, _callback):
            opened['connected'] = True

    class _FakeSignal:
        def connect(self, _callback):
            opened['apply_connected'] = True

    class _FakeDialog:
        def __init__(self, training_parameters, parent):
            opened['training_parameters'] = training_parameters
            opened['parent'] = parent
            self.destroyed = _FakeDestroyed()
            self.apply_to_main_requested = _FakeSignal()

        def show(self):
            opened['show_called'] = True

        def raise_(self):
            opened['raise_called'] = True

        def activateWindow(self):
            opened['activate_called'] = True

    sys.modules['neuralimage.view.augmentation_preview_dialog'] = types.SimpleNamespace(
        AugmentationPreviewDialog=_FakeDialog
    )

    presenter.main_window_state = module.MainWindowState(
        work_mode=WorkMode.train_only.value,
        sample_folder=str(sample_dir),
        label_folder=str(label_dir),
        epochs=1,
    )
    presenter.settings_state = SettingsState()
    presenter.view = types.SimpleNamespace(
        show_warning=types.SimpleNamespace(emit=lambda message: opened.__setitem__('warning', message))
    )
    presenter._augmentation_preview_dialog = None
    presenter._update_main_window_state = lambda: None
    presenter._update_settings_window_state = lambda: None

    try:
        module.MainPresenter._open_augmentation_preview(presenter)
    finally:
        sys.modules.pop('neuralimage.view.augmentation_preview_dialog', None)

    training_parameters = opened['training_parameters']
    assert training_parameters.image_path == sample_dir
    assert training_parameters.label_path == label_dir
    assert opened.get('warning') is None
    assert opened.get('apply_connected') is True
    assert opened.get('show_called') is True
