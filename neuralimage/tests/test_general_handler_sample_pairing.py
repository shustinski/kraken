from types import SimpleNamespace

from model.general_neural_handler import GeneralNeuralHandler
from tests.helpers import make_test_dir


class _Bus:
    def __init__(self):
        self.messages: list[tuple[str, str]] = []

    def publish(self, topic: str, payload):
        self.messages.append((str(topic), str(payload)))


def _build_handler(validation: bool = False) -> GeneralNeuralHandler:
    handler = GeneralNeuralHandler.__new__(GeneralNeuralHandler)
    handler.message_bus = _Bus()
    handler._need_stop = False
    handler.tranining_parameters = SimpleNamespace(validation=validation, validation_percent=20)
    return handler


def test_get_zipped_samples_sets_stop_on_image_label_mismatch():
    root = make_test_dir('pairing_mismatch')
    image_dir = root / 'images'
    label_dir = root / 'labels'
    image_dir.mkdir()
    label_dir.mkdir()

    (image_dir / 'a.jpg').write_bytes(b'x')
    (image_dir / 'b.jpg').write_bytes(b'x')
    (label_dir / 'a.jpg').write_bytes(b'x')

    handler = _build_handler(validation=False)
    train_samples, val_samples = GeneralNeuralHandler._get_zipped_samples(handler, image_dir, label_dir)

    assert handler._need_stop is True
    assert train_samples == []
    assert val_samples is None
    assert any(topic == 'error' for topic, _ in handler.message_bus.messages)


def test_get_zipped_samples_pairs_by_stem():
    root = make_test_dir('pairing_by_stem')
    image_dir = root / 'images'
    label_dir = root / 'labels'
    image_dir.mkdir()
    label_dir.mkdir()

    (image_dir / 'b.jpg').write_bytes(b'x')
    (image_dir / 'a.png').write_bytes(b'x')
    (label_dir / 'a.jpg').write_bytes(b'x')
    (label_dir / 'b.png').write_bytes(b'x')

    handler = _build_handler(validation=False)
    train_samples, val_samples = GeneralNeuralHandler._get_zipped_samples(handler, image_dir, label_dir)

    assert handler._need_stop is False
    assert val_samples is None
    assert [image.stem for image, _ in train_samples] == ['a', 'b']
    assert [label.stem for _, label in train_samples] == ['a', 'b']
