"""test_settings_optimizer_ui: test main presenter sample."""
from __future__ import annotations
import types
from tests.helpers import make_test_dir

from test_settings_optimizer_ui_parts.preprocessing_groupbox_controls import _import_main_presenter_with_stubs

def test_main_presenter_sample_count_includes_synthetic_frames(qapp):
    module = _import_main_presenter_with_stubs()
    presenter = module.MainPresenter.__new__(module.MainPresenter)
    from PyQt6.QtCore import QObject
    QObject.__init__(presenter)
    root = make_test_dir('presenter_sample_count_with_synthetic')
    sample_dir = root / 'samples'
    sample_dir.mkdir()

    class _Signals:
        def __init__(self):
            self.payload = None
            self.calculated = types.SimpleNamespace(
                emit=lambda request_id, normalized_path, image_sizes, total_samples: setattr(
                    self,
                    'payload',
                    (request_id, normalized_path, image_sizes, total_samples),
                )
            )
            self.failed = types.SimpleNamespace(emit=lambda *_args: (_ for _ in ()).throw(AssertionError('unexpected failure')))

    class _SampleWorker:
        @staticmethod
        def collect_image_paths(path):
            return [path / 'frame_0.png', path / 'frame_1.png']

        @staticmethod
        def collect_image_sizes(_image_paths):
            return [(512, 512), (512, 512)]

        @staticmethod
        def calculate_total_samples(image_sizes, params):
            return sum(_SampleWorker.calculate_image_parts_for_settings(size, params) for size in image_sizes)

        @staticmethod
        def calculate_image_parts_for_settings(image_size, params):
            height, width = image_size
            return ((width - params.x_size) // params.step + 1) * ((height - params.y_size) // params.step + 1)

    presenter._sample_count_signals = _Signals()
    module.SampleWorker = _SampleWorker

    calculator_settings = types.SimpleNamespace(
        step=128,
        x_size=256,
        y_size=256,
        vertical_rotation=False,
        horizontal_rotation=False,
        flip_x=False,
        flip_y=False,
        additional_augmentation=False,
        scale_augmentation=False,
        random_crop=False,
        crops_per_image=64,
    )
    synthetic_config = {
        'enabled': True,
        'epoch_size_factor': 1.0,
        'image_size_xy': [1024, 1024],
    }

    module.MainPresenter._run_sample_count_request(
        presenter,
        3,
        str(sample_dir),
        calculator_settings,
        synthetic_config,
        None,
        None,
    )

    assert presenter._sample_count_signals.payload is not None
    assert presenter._sample_count_signals.payload[3] == 116

def test_main_presenter_logs_background_sample_indexing_start(qapp, monkeypatch):
    module = _import_main_presenter_with_stubs()
    presenter = module.MainPresenter.__new__(module.MainPresenter)
    from PyQt6.QtCore import QObject
    QObject.__init__(presenter)
    presenter._sample_count_cache_path = None
    presenter._sample_count_cache_sizes = None

    logged_messages: list[str] = []
    presenter._publish_log_message = logged_messages.append

    captured: dict[str, object] = {}

    class _FakeThread:
        def __init__(self, *, target, args, daemon, name):
            captured['target'] = target
            captured['args'] = args
            captured['daemon'] = daemon
            captured['name'] = name

        def start(self):
            captured['started'] = True

    monkeypatch.setattr(module.threading, 'Thread', _FakeThread)

    module.MainPresenter._start_sample_count_request(
        presenter,
        (5, 'D:/sample_folder', types.SimpleNamespace(), None),
    )

    assert presenter._sample_count_worker_thread is not None
    assert captured['started'] is True
    assert any('отдельном потоке' in message for message in logged_messages)
