import pytest

pytest.importorskip("torch")

import neuralimage.model.NeuralNetwork.model_train_and_recognition as target
class _StubBus:
    def __init__(self):
        self.messages: list[tuple[str, object]] = []

    def publish(self, topic: str, payload):
        self.messages.append((topic, payload))


class _FakeQueue:
    def __init__(self, items: list[list[object]]):
        self._items = list(items)

    def get(self):
        return self._items.pop(0)

    def empty(self):
        return len(self._items) == 0


def test_training_elapsed_suffix_uses_process_start_time(monkeypatch):
    trainer = target.ModelTrainer.__new__(target.ModelTrainer)
    trainer._bus = _StubBus()
    trainer.message_queue = _FakeQueue(
        [
            ["training", "first"],
            ["training", "second"],
        ]
    )
    trainer.error_message = None

    ticks = iter([70.0, 130.0])
    monkeypatch.setattr(target.time, "perf_counter", lambda: next(ticks))

    trainer._drain_training_queue(append_elapsed_suffix=True, started_at=10.0)

    payloads = [str(message[1]) for message in trainer._bus.messages]
    assert payloads[0].endswith("00:01:00")
    assert payloads[1].endswith("00:02:00")


def test_recognition_elapsed_suffix_uses_process_start_time(monkeypatch):
    recognizer = target.ModelRecognizer.__new__(target.ModelRecognizer)
    recognizer._bus = _StubBus()
    recognizer.message_queue = _FakeQueue(
        [
            ["logging", "first"],
            ["logging", "second"],
            ["logging", "third"],
        ]
    )
    recognizer.error_message = None

    ticks = iter([15.0, 50.0, 125.0])
    monkeypatch.setattr(target.time, "perf_counter", lambda: next(ticks))

    recognizer._drain_recognition_queue(append_elapsed_suffix=True, started_at=5.0)

    payloads = [str(message[1]) for message in recognizer._bus.messages]
    assert payloads[0].endswith("00:00:10")
    assert payloads[1].endswith("00:00:45")
    assert payloads[2].endswith("00:02:00")
