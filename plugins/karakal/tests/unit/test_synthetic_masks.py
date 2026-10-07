"""Synthetic frames with known answers agree with the ground-truth object graph."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from karakal.core import error_classes as ec
from karakal.core.grid_ground_truth import match_instances
from karakal.core.synthetic_masks import (
    DEFECT_CLASSES,
    SyntheticLayout,
    add_defect,
    normal_frame,
    synthetic_frame,
)


def _object_classes(mask: np.ndarray, class_map: np.ndarray) -> dict[int, int]:
    count, labels = cv2.connectedComponents((mask > 0).astype(np.uint8), connectivity=8)
    classes = {}
    for label in range(1, count):
        values = class_map[labels == label]
        classes[label] = int(np.bincount(values).argmax())
    return classes


def test_normal_frame_has_only_normal_cells_some_cut_by_the_edge() -> None:
    frame = normal_frame(seed=4)
    assert np.array_equal(frame.mask, frame.truth)
    assert not frame.ignore.any()
    assert set(np.unique(frame.expected_class_map())) == {ec.OK}
    height, width = frame.mask.shape
    assert any(x < 0 or y < 0 or x + w > width or y + h > height for x, y, w, h in frame.cells)


@pytest.mark.parametrize("kind", sorted(DEFECT_CLASSES))
def test_each_defect_kind_is_drawn_with_its_class(kind: str) -> None:
    frame = add_defect(normal_frame(seed=1), kind, seed=7)
    assert len(frame.defects) == 1
    defect = frame.defects[0]
    assert defect.error_class == DEFECT_CLASSES[kind]
    expected = frame.expected_class_map()
    if defect.error_class == ec.OK:
        assert set(np.unique(expected)) == {ec.OK}
    else:
        assert defect.error_class in set(np.unique(expected))


@pytest.mark.parametrize("kind", sorted(DEFECT_CLASSES))
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_generator_answer_matches_the_ground_truth_graph(kind: str, seed: int) -> None:
    frame = add_defect(normal_frame(seed=seed), kind, seed=seed * 31)
    graph = match_instances(frame.mask, frame.truth, ignore=frame.ignore).expected_class_map()
    expected = frame.expected_class_map()
    assert _object_classes(frame.mask, graph) == _object_classes(frame.mask, expected)
    assert np.array_equal(graph == ec.MISSED, expected == ec.MISSED)


def test_jpeg_copy_keeps_the_answer() -> None:
    frame = synthetic_frame({"bite": 2, "bridge": 1, "split": 1, "debris": 2}, seed=5)
    lossless = match_instances(frame.mask, frame.truth).class_counts()
    compressed = match_instances(frame.jpeg(75).mask, frame.truth).class_counts()
    assert compressed == lossless


def test_scatter_layout_has_no_rows_and_no_overlaps() -> None:
    layout = SyntheticLayout(arrangement="scatter")
    frame = normal_frame(layout, seed=2)
    xs = sorted({x for x, _y, _w, _h in frame.cells})
    assert len(xs) > 0.7 * len(frame.cells)
    count = cv2.connectedComponents((frame.mask > 0).astype(np.uint8))[0] - 1
    assert count == len(frame.cells)


def test_mixed_frame_counts() -> None:
    frame = synthetic_frame({"cut_corner": 2, "short": 1, "solid_merge": 1, "removed": 2, "conductor": 1}, seed=9)
    counts = match_instances(frame.mask, frame.truth, ignore=frame.ignore).class_counts()
    assert counts.get("bad_geometry") == 3
    assert counts.get("merge") == 1
    assert counts.get("missed") == 2
    assert counts.get("debris", 0) == 0
