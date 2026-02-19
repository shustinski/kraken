from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip('torch')

from model.NeuralNetwork.recognition_pipeline import _to_channel_first


def test_to_channel_first_expands_grayscale_to_requested_channels() -> None:
    image = np.arange(16, dtype=np.float32).reshape(4, 4)
    converted = _to_channel_first(image, channels=3)

    assert converted.shape == (3, 4, 4)
    assert np.array_equal(converted[0], converted[1])
    assert np.array_equal(converted[1], converted[2])


def test_to_channel_first_reduces_multi_channel_to_single_channel() -> None:
    image = np.zeros((2, 2, 3), dtype=np.float32)
    image[:, :, 1] = 3.0
    converted = _to_channel_first(image, channels=1)

    assert converted.shape == (1, 2, 2)
    assert np.allclose(converted[0], 1.0)
