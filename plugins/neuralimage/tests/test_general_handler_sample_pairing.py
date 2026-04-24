from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('PIL')

from PIL import Image

from neuralimage.model.general_neural_handler import (
    GeneralNeuralHandler,
    _deterministic_validation_split,
    _estimate_label_foreground_ratio,
    _label_ratio_bucket,
)
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


def test_deterministic_validation_split_stratifies_by_label_coverage():
    root = make_test_dir('pairing_validation_split')
    image_dir = root / 'images'
    label_dir = root / 'labels'
    image_dir.mkdir()
    label_dir.mkdir()

    samples: list[tuple[object, object]] = []
    patterns = {
        'a': np.zeros((8, 8), dtype=np.uint8),
        'b': np.pad(np.ones((2, 2), dtype=np.uint8) * 255, 3),
        'c': np.full((8, 8), 255, dtype=np.uint8),
    }

    for prefix, pattern in patterns.items():
        for index in range(3):
            image_path = image_dir / f'{prefix}{index}.png'
            label_path = label_dir / f'{prefix}{index}.png'
            image_path.write_bytes(b'x')
            Image.fromarray(pattern, mode='L').save(label_path)
            samples.append((image_path, label_path))

    train_samples, val_samples = _deterministic_validation_split(samples, val_count=3)

    val_buckets = {
        _label_ratio_bucket(_estimate_label_foreground_ratio(label_path))
        for _image_path, label_path in val_samples
    }

    assert len(train_samples) == 6
    assert len(val_samples) == 3
    assert len(val_buckets) == 3
    assert {image.stem for image, _ in val_samples} != {'c0', 'c1', 'c2'}
