import numpy as np
import pytest

torch = pytest.importorskip('torch')

from model.NeuralNetwork.recognition_pipeline import gpu_predict
from model.NeuralNetwork import recognition_pipeline as recognition_pipeline_module


class _NaNModel(torch.nn.Module):
    def forward(self, x):
        batch, _, height, width = x.shape
        return torch.full((batch, 1, height, width), float('nan'), device=x.device)


def test_gpu_predict_sanitizes_non_finite_outputs():
    payload = {
        'cutted_image': np.zeros((2, 1, 8, 8), dtype=np.float32),
    }
    model = _NaNModel()

    predicted = gpu_predict(payload, model, torch.device('cpu'), batch_size=1)

    output = predicted['predicted_image']
    assert np.isfinite(output).all()
    assert output.shape == (2, 1, 8, 8)
    assert np.allclose(output, 0.5, atol=1e-6)

    stats = predicted.get('_prediction_stats')
    assert isinstance(stats, dict)
    assert int(stats.get('non_finite', 0)) > 0


def test_shared_memory_payload_roundtrip_for_cut_batches():
    if not recognition_pipeline_module._shared_memory_available():
        pytest.skip('shared_memory is unavailable on this platform/runtime')

    payload = {
        'cutted_image': np.arange(2 * 1 * 4 * 4, dtype=np.float32).reshape(2, 1, 4, 4),
    }

    recognition_pipeline_module._store_payload_array_in_shared_memory(payload, 'cutted_image')
    restored_payload, handles = recognition_pipeline_module._restore_payload_arrays_from_shared_memory(
        payload,
        ('cutted_image',),
    )
    try:
        restored = restored_payload['cutted_image']
        assert isinstance(restored, np.ndarray)
        assert restored.shape == (2, 1, 4, 4)
        assert restored.dtype == np.float32
        assert np.array_equal(restored, np.arange(2 * 1 * 4 * 4, dtype=np.float32).reshape(2, 1, 4, 4))
    finally:
        recognition_pipeline_module._cleanup_shared_handles(handles, unlink=True)
