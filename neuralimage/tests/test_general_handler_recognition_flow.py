from types import SimpleNamespace

from model.general_neural_handler import GeneralNeuralHandler


class _Bus:
    def __init__(self):
        self.messages: list[tuple[str, str]] = []

    def publish(self, topic: str, payload):
        self.messages.append((str(topic), str(payload)))


def _build_handler(*, callback=None, need_stop: bool = False) -> GeneralNeuralHandler:
    handler = GeneralNeuralHandler.__new__(GeneralNeuralHandler)
    handler.message_bus = _Bus()
    handler.recognition_parameters = SimpleNamespace()
    handler.callback = callback
    handler._need_stop = bool(need_stop)
    handler.current_thread = None
    handler.train_loader = None
    handler.val_loader = None
    return handler


def test_start_recognition_waits_for_model_recognizer_join(monkeypatch):
    import model.general_neural_handler as target

    flow: list[str] = []

    class _FakeRecognizer:
        def __init__(self, recognition_parameters, message_bus, callback=None):
            self.succeeded = True
            self.error_message = None

        def start(self):
            flow.append('start')

        def join(self):
            flow.append('join')

    monkeypatch.setattr(target, 'ModelRecognizer', _FakeRecognizer)

    handler = _build_handler(callback=lambda: flow.append('callback'))
    target.GeneralNeuralHandler._start_recognition(handler)

    assert flow == ['start', 'join', 'callback']
    assert handler.current_thread is None


def test_start_recognition_reports_generic_error_on_failed_recognizer_without_message(monkeypatch):
    import model.general_neural_handler as target

    class _FakeRecognizer:
        def __init__(self, recognition_parameters, message_bus, callback=None):
            self.succeeded = False
            self.error_message = None

        def start(self):
            return

        def join(self):
            return

    monkeypatch.setattr(target, 'ModelRecognizer', _FakeRecognizer)

    handler = _build_handler(callback=None, need_stop=False)
    target.GeneralNeuralHandler._start_recognition(handler)

    assert any(topic == 'error' for topic, _ in handler.message_bus.messages)
