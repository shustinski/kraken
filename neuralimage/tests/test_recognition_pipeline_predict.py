import numpy as np
import pytest

torch = pytest.importorskip('torch')

from model.NeuralNetwork.recognition_pipeline import gpu_predict


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
