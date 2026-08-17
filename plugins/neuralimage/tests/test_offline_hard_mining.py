import json

import numpy as np

from neuralimage.training.hard_mining import (
    OfflineHardDatasetBuilder,
    compute_geometry_difficulty_features,
)


def test_geometry_features_prioritize_thin_dense_routing():
    simple = np.zeros((64, 64), dtype=np.float32)
    simple[20:44, 20:44] = 1.0
    routed = np.zeros_like(simple)
    routed[8:56:6, 4:60] = 1.0
    assert compute_geometry_difficulty_features(routed).score > compute_geometry_difficulty_features(simple).score


def test_offline_manifest_contains_geometry_loss_roi_and_ranking(tmp_path):
    mask = np.zeros((32, 32), dtype=np.float32)
    mask[15, 4:28] = 1.0
    output = OfflineHardDatasetBuilder(tmp_path / 'hard.jsonl').build(
        [('sample-a', mask)],
        historical_losses={'sample-a': 0.7},
        rois={'sample-a': (4, 5, 20, 21)},
        frame_ids={'sample-a': 'frame-1'},
    )
    payload = json.loads(output.read_text(encoding='utf-8').strip())
    assert payload['rank'] == 1
    assert payload['frame'] == 'frame-1'
    assert payload['roi'] == [4, 5, 20, 21]
    assert payload['historical_loss'] == 0.7
    assert (tmp_path / 'hard.csv').exists()
