from pathlib import Path
from types import SimpleNamespace

from neuralimage.model.general_neural_handler import GeneralNeuralHandler


class _Bus:
    def __init__(self):
        self.messages: list[tuple[str, str]] = []

    def publish(self, topic: str, payload):
        self.messages.append((str(topic), str(payload)))


def _build_handler() -> GeneralNeuralHandler:
    handler = GeneralNeuralHandler.__new__(GeneralNeuralHandler)
    handler.message_bus = _Bus()
    handler.train_loader = object()
    handler.val_loader = None
    handler.work_mode = 'train_only'
    handler._hard_mining_active = True
    handler._training_failed = False
    handler.current_thread = None
    handler._release_torch_memory = lambda: None
    handler._stop_training_callback = lambda: None
    handler.tranining_parameters = SimpleNamespace(
        multi_gpu_mode='distributeddataparallel',
        use_multi_gpu=True,
        epochs=1,
        optimizer=SimpleNamespace(),
        mixed_precision='bf16',
        loss_function='bce',
        loss_term_weights={},
        dice_loss_weight=0.5,
        iou_loss_weight=0.5,
        hard_mining=SimpleNamespace(),
        early_stopping=SimpleNamespace(),
        warmup=SimpleNamespace(),
        skip_uniform_labels=False,
        show_batch_preview=True,
        log_update_frequency=0,
    )
    return handler


def test_start_training_preserves_multi_gpu_via_dataparallel_when_hard_mining_is_enabled(monkeypatch):
    import neuralimage.model.general_neural_handler as target
    captured: dict[str, object] = {}

    class _FakeTrainer:
        def __init__(self, *_args, **kwargs):
            captured.update(kwargs)
            self.succeeded = True
            self.error_message = None

        def start(self):
            return

        def join(self):
            return

    monkeypatch.setattr(target, 'ModelTrainer', _FakeTrainer)

    handler = _build_handler()
    target.GeneralNeuralHandler._start_training(handler, model=object(), model_save_path=Path('model.pth'))

    assert captured['multi_gpu_mode'] == 'dataparallel'
    assert captured['use_multi_gpu'] is True
    assert any(
        'DistributedDataParallel заменен на nn.DataParallel' in payload
        for topic, payload in handler.message_bus.messages
        if topic == 'logging'
    )
