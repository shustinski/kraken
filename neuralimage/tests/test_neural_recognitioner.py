from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")

from lib.data_interfaces import RecognitionParameters
from model.NeuralNetwork.model_train_and_recognition import NeuralRecognizer
from tests.helpers import make_test_dir


class _StubBus:
    def __init__(self):
        self.messages: list[tuple[str, object]] = []

    def publish(self, topic: str, payload):
        self.messages.append((topic, payload))


def _build_params(base_dir: Path, model) -> RecognitionParameters:
    return RecognitionParameters(
        source_files=[Path("frame_001.jpg"), Path("frame_002.jpg")],
        result_folder=base_dir / "result",
        model=model,
        part_size=(16, 16),
        batch_size=4,
        overlap=2,
    )


def test_run_uses_one_thread_on_small_workload(monkeypatch):
    base_dir = make_test_dir("neural_rec_small")
    bus = _StubBus()
    recognizer = NeuralRecognizer(_build_params(base_dir, model="dummy_model_path.pth"), bus)

    calls = {"one": 0, "multi": 0}
    recognizer.prepare_model = lambda: None
    recognizer.run_one_thread = lambda: calls.__setitem__("one", calls["one"] + 1)
    recognizer.run_multiprocessing = lambda: calls.__setitem__("multi", calls["multi"] + 1)
    monkeypatch.setattr("model.NeuralNetwork.model_train_and_recognition.mp.cpu_count", lambda: 64)
    monkeypatch.setattr("model.NeuralNetwork.model_train_and_recognition.torch.cuda.is_available", lambda: False)

    recognizer.run(multithreading=True)

    assert calls["one"] == 1
    assert calls["multi"] == 0


def test_stop_sets_both_stop_events():
    base_dir = make_test_dir("neural_rec_stop")
    bus = _StubBus()
    recognizer = NeuralRecognizer(_build_params(base_dir, model="dummy_model_path.pth"), bus)

    recognizer.stop()

    assert recognizer._thread_stop_event.is_set()
    assert recognizer.stop_event.is_set()


def test_run_falls_back_to_one_thread_for_in_memory_model(monkeypatch):
    base_dir = make_test_dir("neural_rec_mem")
    bus = _StubBus()
    recognizer = NeuralRecognizer(_build_params(base_dir, model=nn.Identity()), bus)

    calls = {"one": 0, "multi": 0}
    recognizer.prepare_model = lambda: None
    recognizer.run_one_thread = lambda: calls.__setitem__("one", calls["one"] + 1)
    recognizer.run_multiprocessing = lambda: calls.__setitem__("multi", calls["multi"] + 1)
    monkeypatch.setattr("model.NeuralNetwork.model_train_and_recognition.mp.cpu_count", lambda: 64)
    monkeypatch.setattr("model.NeuralNetwork.model_train_and_recognition.torch.cuda.is_available", lambda: False)

    recognizer.run(multithreading=True)

    assert calls["one"] == 1
    assert calls["multi"] == 0
