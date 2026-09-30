"""Packed cell outlines and features read as the same pairs and stay small."""
from __future__ import annotations

import copy
import dataclasses
import pickle
import sys

import cv2
import numpy as np

from karakal.core.grid_anomaly import GridCellAnalysisResult, GridDamageAnalysisConfig, detect_grid_cell_anomalies
from karakal.core.grid_packed import PackedFeatures, PackedPoints

POINTS = ((10, 20), (300, 20), (300, 470), (10, 470))
FEATURES = (("width_ratio", 1.25), ("area_ratio", 0.5), ("solidity", 0.9))


def _cell(**overrides) -> GridCellAnalysisResult:
    values = dict(
        row=0, col=0, bbox=(10, 20, 291, 451), centroid=(155.0, 245.0), contour_id=1,
        status="normal", score=0.0, feature_snapshot=FEATURES, outline=POINTS,
    )
    values.update(overrides)
    return GridCellAnalysisResult(**values)


def test_packed_values_read_as_the_original_pairs() -> None:
    cell = _cell()
    assert isinstance(cell.outline, PackedPoints) and isinstance(cell.feature_snapshot, PackedFeatures)
    assert tuple(cell.outline) == POINTS and cell.outline == POINTS and len(cell.outline) == 4
    assert cell.outline[1] == (300, 20) and tuple(cell.outline[1:3]) == POINTS[1:3]
    assert np.array_equal(np.asarray(cell.outline, dtype=np.int32), np.asarray(POINTS, dtype=np.int32))
    assert dict(cell.feature_snapshot) == dict(FEATURES) and tuple(cell.feature_snapshot) == FEATURES
    assert cell.feature_snapshot == FEATURES


def test_packed_cells_survive_pickle_replace_copy_and_asdict() -> None:
    cell = _cell()
    assert pickle.loads(pickle.dumps(cell)) == cell
    changed = dataclasses.replace(cell, status="broken", reasons=("broken_geometry",))
    assert changed.outline is cell.outline and changed.feature_snapshot is cell.feature_snapshot
    assert copy.deepcopy(cell) == cell
    payload = dataclasses.asdict(cell)
    assert tuple(payload["outline"]) == POINTS and dict(payload["feature_snapshot"]) == dict(FEATURES)
    assert _cell(outline=(), feature_snapshot=()).outline == ()


def test_equal_features_share_their_names() -> None:
    first = PackedFeatures(FEATURES)
    second = PackedFeatures(tuple((key, value + 1.0) for key, value in FEATURES))
    assert first._keys is second._keys


def _deep_size(value, seen: set[int]) -> int:
    if id(value) in seen:
        return 0
    seen.add(id(value))
    size = sys.getsizeof(value)
    if isinstance(value, (tuple, list)):
        size += sum(_deep_size(item, seen) for item in value)
    elif hasattr(value, "__slots__") and not isinstance(value, (str, bytes)):
        size += sum(_deep_size(getattr(value, name), seen) for name in value.__slots__ if hasattr(value, name))
    return size


def test_round_cell_frame_stays_within_memory_budget() -> None:
    # Tuples of points cost ~5 KB per round cell; a 20k-frame run then filled the RAM.
    image = np.zeros((700, 900), dtype=np.uint8)
    for y in range(40, 660, 60):
        for x in range(40, 860, 60):
            cv2.circle(image, (x, y), 22, 255, -1)
    result = detect_grid_cell_anomalies(
        image, config=GridDamageAnalysisConfig(cell_representation="binary", scoring_mode="calibrated")
    )
    cells = result.per_cell_results
    assert len(cells) > 100 and all(len(cell.outline) >= 20 for cell in cells)
    seen: set[int] = set()
    per_cell = sum(_deep_size(cell.outline, seen) + _deep_size(cell.feature_snapshot, seen) for cell in cells) / len(cells)
    assert per_cell < 1200, per_cell
