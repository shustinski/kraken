from pathlib import Path
from types import SimpleNamespace

from lib.data_interfaces import WorkMode
from model.general_neural_handler import GeneralNeuralHandler


class _Bus:
    def __init__(self):
        self.messages: list[tuple[str, str]] = []

    def publish(self, topic: str, payload):
        self.messages.append((str(topic), str(payload)))


def _build_handler(
    *,
    model_name: str,
    segment_size: tuple[int, int],
    colors: int = 1,
    work_mode: WorkMode = WorkMode.train_only,
) -> GeneralNeuralHandler:
    handler = GeneralNeuralHandler.__new__(GeneralNeuralHandler)
    handler.work_mode = work_mode
    handler.message_bus = _Bus()
    handler._need_stop = False
    handler.recognition_parameters = SimpleNamespace(model=model_name)
    handler.tranining_parameters = SimpleNamespace(
        colors=colors,
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


def test_resolve_training_model_uses_selected_rgb_input_channels(monkeypatch):
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

    handler = _build_handler(model_name='S 660k', segment_size=(256, 256), colors=3)
    model, save_path = target.GeneralNeuralHandler._resolve_training_model(handler)

    assert captured == {
        'model_name': 'S 660k',
        'input_channels': 3,
        'kwargs': {},
    }
    assert int(getattr(model, '_neuralimage_input_channels')) == 3
    assert save_path == Path('tmp_train') / 'S 660k_shift_32_epoch1.pth'


def test_resolve_training_model_stops_on_loaded_model_channel_mismatch(monkeypatch):
    import model.general_neural_handler as target

    loaded_model = SimpleNamespace(_neuralimage_input_channels=1)
    checkpoint_path = Path('tmp_models') / 'grayscale_checkpoint.pth'

    monkeypatch.setattr(target, 'load_model_artifact', lambda *_args, **_kwargs: loaded_model)

    handler = _build_handler(
        model_name=str(checkpoint_path),
        segment_size=(256, 256),
        colors=3,
        work_mode=WorkMode.further_training,
    )
    model, save_path = target.GeneralNeuralHandler._resolve_training_model(handler)

    assert model is loaded_model
    assert save_path == checkpoint_path
    assert handler._need_stop is True
    assert any(
        topic == 'error'
        and 'selected RGB mode (3 channels)' in payload
        and 'expects 1 channel(s)' in payload
        for topic, payload in handler.message_bus.messages
    )


def test_start_stops_before_training_when_model_resolution_sets_need_stop():
    handler = GeneralNeuralHandler.__new__(GeneralNeuralHandler)
    handler.work_mode = WorkMode.train_only
    handler.message_bus = _Bus()
    handler.recognition_parameters = SimpleNamespace(model='S 660k')
    handler.tranining_parameters = SimpleNamespace()
    handler._need_stop = False
    handler._training_failed = False

    flow: list[str] = []

    handler._prepare_training_pipeline = lambda: flow.append('prepare')

    def _resolve_training_model():
        flow.append('resolve_model')
        handler._need_stop = True
        return object(), Path('tmp_train') / 'model.pth'

    handler._resolve_training_model = _resolve_training_model
    handler._start_training = lambda *_args, **_kwargs: flow.append('train')
    handler._start_recognition = lambda: flow.append('recognize')

    GeneralNeuralHandler.start(handler)

    assert flow == ['prepare', 'resolve_model']
