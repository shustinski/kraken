from pathlib import Path
from types import SimpleNamespace

from lib.data_interfaces import WorkMode
from model.general_neural_handler import GeneralNeuralHandler


class _Bus:
    def __init__(self):
        self.messages: list[tuple[str, str]] = []

    def publish(self, topic: str, payload):
        self.messages.append((str(topic), str(payload)))


def _build_handler(*, model_name: str, segment_size: tuple[int, int]) -> GeneralNeuralHandler:
    handler = GeneralNeuralHandler.__new__(GeneralNeuralHandler)
    handler.work_mode = WorkMode.train_only
    handler.message_bus = _Bus()
    handler.recognition_parameters = SimpleNamespace(model=model_name)
    handler.tranining_parameters = SimpleNamespace(
        colors=1,
        epochs=1,
        image_path=Path('tmp_train') / 'images',
        generation=SimpleNamespace(
            step=32,
            segment_size=segment_size,
            vertical_rotation=False,
            horizontal_rotation=False,
        ),
    )
    return handler


def test_resolve_training_model_passes_transformer_img_size_from_patch(monkeypatch):
    import model.general_neural_handler as target

    captured: dict[str, object] = {}

    class _DummyModel:
        pass

    def _fake_create_model(model_name, input_channels, **kwargs):
        captured['model_name'] = model_name
        captured['input_channels'] = input_channels
        captured['kwargs'] = kwargs
        return _DummyModel()

    monkeypatch.setattr(target, 'create_model', _fake_create_model)

    handler = _build_handler(model_name='Transformer', segment_size=(384, 384))
    model, save_path = target.GeneralNeuralHandler._resolve_training_model(handler)

    assert captured == {
        'model_name': 'Transformer',
        'input_channels': 1,
        'kwargs': {'img_size': 384},
    }
    assert getattr(model, '_neuralimage_model_name') == 'Transformer'
    assert int(getattr(model, '_neuralimage_input_channels')) == 1
    assert getattr(model, '_neuralimage_model_kwargs') == {'img_size': 384}
    assert save_path == Path('tmp_train') / 'Transformer_shift_32_epoch1.pth'
