import numpy as np

from neuralimage.active_learning.config import ActiveLearningConfig
from neuralimage.active_learning.export import ActiveLearningExporter
from neuralimage.active_learning.scoring import score_prediction_uncertainty


def test_score_prediction_uncertainty_detects_ambiguous_region():
    probabilities = np.full((16, 16), 0.5, dtype=np.float32)
    payload = score_prediction_uncertainty(probabilities, low_confidence_threshold=0.6)
    assert bool(payload['high_entropy'].any())
    assert float(payload['mean_entropy']) > 0.0


def test_active_learning_exporter_writes_needs_annotation(tmp_path):
    exporter = ActiveLearningExporter(ActiveLearningConfig(enabled=True, max_exports_per_run=4))
    image = np.full((16, 16), 128, dtype=np.uint8)
    probabilities = np.full((16, 16), 0.5, dtype=np.float32)
    record = exporter.export_sample(
        export_root=tmp_path / 'NeedsAnnotation',
        sample_id='patch_001',
        image=image,
        probabilities=probabilities,
    )
    assert record is not None
    assert (tmp_path / 'NeedsAnnotation' / 'images' / 'patch_001.png').exists()
