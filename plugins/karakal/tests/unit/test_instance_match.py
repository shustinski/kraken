"""Object graph between a network mask and its ground truth, and the validation metrics."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from karakal.core import error_classes as ec
from karakal.core.grid_ground_truth import match_instances
from karakal.core.segmentation_metrics import (
    ObjectConfusion,
    boundary_iou,
    brier_score,
    expected_calibration_error,
    mask_iou,
    mean_over_frames,
    split_merge_scores,
)

CELL_W, CELL_H = 18, 40


def _truth(cells: int = 8) -> tuple[np.ndarray, list[tuple[int, int]]]:
    image = np.zeros((200, 60 * cells + 40), dtype=np.uint8)
    origins = [(20 + 60 * index, 60) for index in range(cells)]
    for x, y in origins:
        cv2.rectangle(image, (x, y), (x + CELL_W - 1, y + CELL_H - 1), 255, -1)
    return image, origins


def _class_of_object_at(match, x: int, y: int) -> int:
    return match.prediction_class(int(match.prediction_labels[y, x]))


def test_each_relation_of_the_graph_gets_its_class() -> None:
    truth, origins = _truth()
    network = truth.copy()
    # 0: normal; 1: bitten corner -> geometry; 2-3: bridged -> merge; 4: cut in two -> split;
    # 5: missing -> miss; 6: normal; 7: normal; plus a crumb away from every cell -> debris.
    x, y = origins[1]
    network[y : y + 22, x : x + 12] = 0
    x2, y2 = origins[2]
    network[y2 + 15 : y2 + 25, x2 + CELL_W : origins[3][0]] = 255
    x4, y4 = origins[4]
    network[y4 + 18 : y4 + 22, x4 : x4 + CELL_W] = 0
    x5, y5 = origins[5]
    network[y5 : y5 + CELL_H, x5 : x5 + CELL_W] = 0
    network[150:156, 100:106] = 255

    match = match_instances(network, truth)

    assert _class_of_object_at(match, origins[0][0] + 5, origins[0][1] + 5) == ec.OK
    assert _class_of_object_at(match, x + 15, y + 35) == ec.BAD_GEOMETRY
    assert _class_of_object_at(match, x2 + 5, y2 + 5) == ec.MERGE
    assert _class_of_object_at(match, x4 + 5, y4 + 5) == ec.SPLIT
    assert _class_of_object_at(match, x4 + 5, y4 + 35) == ec.SPLIT
    assert _class_of_object_at(match, 102, 152) == ec.DEBRIS
    assert sum(1 for item in match.truths if item.missed) == 1
    counts = match.class_counts()
    assert counts["merge"] == 1 and counts["split"] == 2 and counts["missed"] == 1


def test_expected_class_map_paints_whole_objects_misses_and_ignore() -> None:
    truth, origins = _truth(3)
    network = truth.copy()
    x, y = origins[2]
    network[y : y + CELL_H, x : x + CELL_W] = 0
    ignore = np.zeros(truth.shape, dtype=bool)
    ignore[:, :10] = True

    class_map = match_instances(network, truth, ignore=ignore).expected_class_map()
    assert class_map.dtype == np.uint8
    assert class_map.shape == truth.shape
    assert set(np.unique(class_map[y : y + CELL_H, x : x + CELL_W])) == {ec.MISSED}
    assert np.all(class_map[:, :10] == ec.IGNORE)
    assert np.all(class_map[origins[0][1] : origins[0][1] + CELL_H, origins[0][0] : origins[0][0] + CELL_W] == ec.OK)


def test_objects_in_the_ignore_mask_are_neither_debris_nor_misses() -> None:
    truth, origins = _truth(6)
    truth[:, 280:] = 0
    network = truth.copy()
    rng = np.random.default_rng(1)
    for _ in range(30):
        px, py = rng.integers(0, 15, size=2)
        network[160 + py, 300 + px] = 255
    truth[60:100, 300:318] = 255
    ignore = np.zeros(truth.shape, dtype=bool)
    ignore[50:200, 290:340] = True

    match = match_instances(network, truth, ignore=ignore)
    assert not any(item.missed for item in match.truths)
    assert all(item.error_class in {ec.OK, ec.IGNORE} for item in match.predictions)


def test_a_crumb_touching_a_cell_is_debris_not_a_split() -> None:
    truth, origins = _truth(1)
    network = truth.copy()
    x, y = origins[0]
    network[y + 10 : y + 13, x + CELL_W + 1 : x + CELL_W + 4] = 255
    truth[y + 10 : y + 13, x + CELL_W : x + CELL_W + 2] = 255  # truth blob slightly wider
    match = match_instances(network, truth)
    classes = sorted(item.error_class for item in match.predictions)
    assert classes == [ec.OK, ec.DEBRIS]


def test_jpeg_copies_give_the_same_graph() -> None:
    truth, origins = _truth()
    network = truth.copy()
    x2, y2 = origins[2]
    network[y2 + 15 : y2 + 25, x2 + CELL_W : origins[3][0]] = 255

    def jpeg(image: np.ndarray) -> np.ndarray:
        return cv2.imdecode(cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 75])[1], cv2.IMREAD_GRAYSCALE)

    assert match_instances(jpeg(network), jpeg(truth)).class_counts() == match_instances(network, truth).class_counts()


def test_boundary_iou_reacts_to_a_contour_error_mask_iou_hides() -> None:
    truth = np.zeros((120, 120), dtype=bool)
    truth[10:110, 10:110] = True
    shifted = np.zeros_like(truth)
    shifted[12:112, 12:112] = True
    assert mask_iou(truth, shifted) > 0.9
    assert boundary_iou(truth, shifted, width=2) < 0.6
    assert boundary_iou(truth, truth) == 1.0


def test_split_and_merge_scores_separate_the_two_errors() -> None:
    truth = np.zeros((40, 80), dtype=np.int32)
    truth[5:35, 5:35] = 1
    truth[5:35, 45:75] = 2
    perfect = split_merge_scores(truth, truth)
    assert perfect.rand_split == pytest.approx(1.0) and perfect.rand_merge == pytest.approx(1.0)
    assert perfect.vi_split == pytest.approx(0.0, abs=1e-9)

    merged = truth.copy()
    merged[merged == 2] = 1
    scores = split_merge_scores(merged, truth)
    assert scores.rand_split == pytest.approx(1.0)
    assert scores.rand_merge < 0.6
    assert scores.vi_merge > 0.5 and scores.vi_split == pytest.approx(0.0, abs=1e-9)

    split = truth.copy()
    split[5:35, 20:35][split[5:35, 20:35] == 1] = 3
    scores = split_merge_scores(split, truth)
    assert scores.rand_merge == pytest.approx(1.0)
    assert scores.rand_split < 0.8
    assert scores.vi_split > 0.3 and scores.vi_merge == pytest.approx(0.0, abs=1e-9)


def test_object_confusion_scores_classes_and_false_alarms() -> None:
    confusion = ObjectConfusion(classes=(ec.OK, ec.BAD_GEOMETRY, ec.DEBRIS, ec.MERGE))
    confusion.add([ec.OK, ec.OK, ec.MERGE, ec.DEBRIS], [ec.OK, ec.DEBRIS, ec.MERGE, ec.DEBRIS])
    confusion.add([ec.BAD_GEOMETRY, ec.OK], [ec.OK, ec.OK])
    per_class = confusion.per_class()
    assert per_class["merge"]["f1"] == 1.0
    assert per_class["debris"]["precision"] == 0.5
    assert per_class["bad_geometry"]["recall"] == 0.0
    detection = confusion.detection()
    assert detection["false_alarms_per_frame"] == 0.5
    assert confusion.matrix().sum() == 6


def test_calibration_scores_and_frame_averaging() -> None:
    assert brier_score([1.0, 0.0], [True, False]) == 0.0
    assert expected_calibration_error([0.9, 0.9, 0.1, 0.1], [True, True, False, False]) == pytest.approx(0.1)
    assert mean_over_frames([{"a": 1.0, "b": 2.0}, {"a": 3.0}]) == {"a": 2.0, "b": 2.0}
