from concurrent.futures import ThreadPoolExecutor

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
    assert len(record.rois) == 1
    assert (tmp_path / 'NeedsAnnotation' / 'manifest.jsonl').exists()
    assert (tmp_path / 'NeedsAnnotation' / 'manifest.csv').exists()

    # Export is resumable and does not duplicate a previously selected frame.
    duplicate = exporter.export_sample(
        export_root=tmp_path / 'NeedsAnnotation',
        sample_id='patch_001',
        image=image,
        probabilities=probabilities,
    )
    assert duplicate is None


def test_active_learning_detects_source_disagreement(tmp_path):
    config = ActiveLearningConfig(
        enabled=True,
        low_confidence_threshold=0.0,
        high_entropy_threshold=1.0,
        disagreement_threshold=0.2,
        min_roi_area=4,
    )
    exporter = ActiveLearningExporter(config)
    probabilities = np.full((16, 16), 0.9, dtype=np.float32)
    disagreement = np.zeros_like(probabilities)
    disagreement[4:12, 4:12] = 0.5
    record = exporter.export_sample(
        export_root=tmp_path / 'NeedsAnnotation',
        sample_id='frame',
        image=np.zeros((16, 16), dtype=np.uint8),
        probabilities=probabilities,
        disagreement=disagreement,
        metadata={'model_hash': 'abc', 'config_hash': 'def'},
    )
    assert record is not None
    assert 'source_disagreement' in record.reasons


def test_active_learning_manifest_updates_are_concurrency_safe(tmp_path):
    exporter = ActiveLearningExporter(
        ActiveLearningConfig(enabled=True, max_exports_per_run=4, min_roi_area=1)
    )
    export_root = tmp_path / 'NeedsAnnotation'
    image = np.zeros((8, 8), dtype=np.uint8)
    probabilities = np.full((8, 8), 0.5, dtype=np.float32)

    def export(index: int):
        return exporter.export_sample(
            export_root=export_root,
            sample_id=f'frame_{index}',
            image=image,
            probabilities=probabilities,
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(export, range(8)))

    rows = (export_root / 'manifest.jsonl').read_text(encoding='utf-8').splitlines()
    assert len(rows) == 4
    assert len({row.split('"sample_id":"', 1)[1].split('"', 1)[0] for row in rows}) == 4
    assert not (export_root / '.manifest.lock').exists()


def test_active_learning_run_limit_does_not_cap_resumed_dataset(tmp_path):
    config = ActiveLearningConfig(enabled=True, max_exports_per_run=1, min_roi_area=1)
    export_root = tmp_path / 'NeedsAnnotation'
    inputs = {
        'export_root': export_root,
        'image': np.zeros((8, 8), dtype=np.uint8),
        'probabilities': np.full((8, 8), 0.5, dtype=np.float32),
    }

    assert ActiveLearningExporter(config).export_sample(sample_id='first', **inputs) is not None
    assert ActiveLearningExporter(config).export_sample(sample_id='second', **inputs) is not None

    rows = (export_root / 'manifest.jsonl').read_text(encoding='utf-8').splitlines()
    assert len(rows) == 2
