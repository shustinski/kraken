from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from neuralimage.lib.data_interfaces import RecognitionParameters
from neuralimage.model.NeuralNetwork.model_train_and_recognition import ModelRecognizer
from tests.helpers import make_test_dir


class _StubBus:
    def __init__(self):
        self.messages: list[tuple[str, object]] = []

    def publish(self, topic: str, payload):
        self.messages.append((topic, payload))


def _build_params(base_dir: Path) -> RecognitionParameters:
    return RecognitionParameters(
        source_files=[Path("frame_001.jpg")],
        result_folder=base_dir / "result",
        model="dummy_model_path.pth",
        part_size=(16, 16),
        batch_size=4,
        overlap=2,
    )


def test_model_recognizer_stop_kills_child_process(monkeypatch):
    import neuralimage.model.NeuralNetwork.model_train_and_recognition as target
    events = {"killed": 0}

    class _FakeQueue:
        def __init__(self):
            self._items: list[object] = []

        def put(self, item):
            self._items.append(item)

        def get(self):
            return self._items.pop(0)

        def empty(self):
            return len(self._items) == 0

        def close(self):
            return

    class _FakeEvent:
        def __init__(self):
            self._set = False

        def set(self):
            self._set = True

        def wait(self, timeout=None):
            return self._set

    class _FakeRecognizerProcess:
        def __init__(self, recognition_parameters, message_bus, stop_event, multithreading=False):
            self.exitcode = None
            self._alive = False

        def start(self):
            self._alive = True

        def is_alive(self):
            return self._alive

        def join(self, timeout=None):
            return

        def kill(self):
            events["killed"] += 1
            self._alive = False
            self.exitcode = -9

    monkeypatch.setattr(target.mp, "Queue", lambda: _FakeQueue())
    monkeypatch.setattr(target.mp, "Event", lambda: _FakeEvent())
    monkeypatch.setattr(target, "RecognizerProcess", _FakeRecognizerProcess)
    monkeypatch.setattr(target.time, "sleep", lambda *_: None)

    base_dir = make_test_dir("model_recognizer_stop")
    recognizer = ModelRecognizer(_build_params(base_dir), _StubBus())
    recognizer.start()
    recognizer.stop()
    recognizer.join(timeout=2)

    assert recognizer.is_alive() is False
    assert events["killed"] == 1
    assert recognizer.succeeded is False


def test_recognition_manifest_skips_completed_files_on_resume(tmp_path):
    first = tmp_path / 'first.png'
    second = tmp_path / 'second.png'
    first.touch()
    second.touch()
    manifest = tmp_path / '.neuralimage-recognition-progress.json'
    bus = _StubBus()

    initial = SimpleNamespace(
        result_folder=tmp_path,
        source_folder=tmp_path,
        source_files=[first, second],
        recursive_file_search=False,
        recognition_multiprocessing_enabled=False,
        resume_manifest_path=manifest,
        resume_from_manifest=False,
    )
    recognizer = ModelRecognizer(initial, bus)
    recognizer._load_or_reset_manifest()
    recognizer._record_completed_file(str(first), str(tmp_path / 'first.jpg'))

    resumed = SimpleNamespace(**vars(initial))
    resumed.source_files = [first, second]
    resumed.resume_from_manifest = True
    resumed_recognizer = ModelRecognizer(resumed, bus)
    resumed_recognizer._load_or_reset_manifest()

    assert resumed.source_files == [second]
    assert manifest.exists()
