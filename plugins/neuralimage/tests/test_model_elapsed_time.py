import pickle
from pathlib import Path
from types import SimpleNamespace

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


class _PicklableQueue:
    def put(self, _message):
        return


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


def test_model_trainer_passes_initialized_pause_event_to_training_process(monkeypatch):
    captured: dict[str, object] = {}

    class _FakeTrainingProcess:
        def __init__(self, *_args, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(target, 'TrainerProcess', _FakeTrainingProcess)
    trainer = target.ModelTrainer(
        train_dataloader=object(),
        val_dataloader=None,
        model=object(),
        save_path=Path('model.pth'),
        epochs=1,
        message_bus=_StubBus(),
    )

    try:
        trainer._create_training_process()
        assert captured['pause_event'] is trainer._pause_event
        assert not trainer._pause_event.is_set()
        trainer.pause()
        assert trainer._pause_event.is_set()
    finally:
        trainer.message_queue.close()


def test_training_compile_probe_reads_runtime_frozen_state(monkeypatch):
    messages: list[list[object]] = []
    trainer = target.TrainerProcess.__new__(target.TrainerProcess)
    trainer._model = target.nn.Identity()
    trainer._bus = type('_QueueBus', (), {'put': lambda _self, message: messages.append(message)})()
    monkeypatch.setenv('NEURALIMAGE_TORCH_COMPILE', '0')

    trainer._try_compile_model()

    assert trainer._torch_compile_active is False
    assert messages


def test_trainer_checkpoint_logger_is_picklable_for_windows_spawn(tmp_path):
    trainer = target.TrainerProcess(
        train_dataloader=object(),
        val_dataloader=None,
        model=target.nn.Identity(),
        save_path=tmp_path / 'model.pth',
        epochs=1,
        message_bus=_PicklableQueue(),
    )

    pickle.dumps(trainer._checkpoint_manager)


def test_epoch_progress_is_zero_while_first_epoch_is_running(monkeypatch):
    messages: list[list[object]] = []
    trainer = target.TrainerProcess.__new__(target.TrainerProcess)
    trainer._model = target.nn.Identity()
    trainer._train_dataloader = SimpleNamespace(sampler=None)
    trainer._epochs = 5
    trainer._bus = SimpleNamespace(put=messages.append)
    monkeypatch.setattr(target, '_collect_memory_metrics', lambda: None)
    run_context = SimpleNamespace(
        optimizer=SimpleNamespace(param_groups=[{'lr': 0.001}]),
        train_size=20,
    )

    trainer._publish_epoch_start(0, run_context, distributed=False)

    metric_payloads = [message[1] for message in messages if message[0] == 'metrics']
    assert {'type': 'train_epoch_progress', 'current': 0, 'total': 5} in metric_payloads
    assert {'type': 'train_batch_progress', 'current': 0, 'total': 20} in metric_payloads
    assert {'type': 'validation_progress', 'current': 0, 'total': 0} in metric_payloads


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
