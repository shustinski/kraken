"""Ground-truth matching for calibrated grid inspection."""

from __future__ import annotations

import cv2
import numpy as np

from karakal.core.grid_calibration import GridCalibration
from karakal.core.grid_ground_truth import detection_f1, fit_markup, match_masks


def _outline(fill: tuple[tuple[int, int], ...] = (), skip: tuple[tuple[int, int], ...] = ()) -> np.ndarray:
    image = np.zeros((220, 280), dtype=np.uint8)
    for row in range(3):
        for col in range(4):
            if (row, col) in skip:
                continue
            x, y = 16 + col * 66, 16 + row * 66
            cv2.rectangle(image, (x, y), (x + 40, y + 40), 255, 2)
            if (row, col) in fill:
                cv2.rectangle(image, (x + 4, y + 4), (x + 36, y + 36), 255, -1)
    return image


def test_each_error_type_is_matched_and_a_missing_cell_is_only_a_count() -> None:
    truth = _outline()
    network = _outline(fill=((0, 0),), skip=((2, 0),))
    cv2.rectangle(network, (86, 20), (118, 52), 255, -1)
    cv2.rectangle(network, (148, 16), (220, 56), 255, 2)
    cv2.rectangle(network, (4, 4), (10, 10), 255, -1)
    cv2.rectangle(network, (150, 82), (190, 124), 0, -1)

    matched = match_masks(network, truth)
    labels = {item["label"] for item in matched["components"]}
    assert "filled_cell" in labels or "partial_filled_cell" in labels
    assert "merged_contour" in labels
    assert "small_artifact" in labels
    assert "broken_geometry" in labels
    assert "normal" in labels
    assert matched["missed"] >= 1


def test_markup_fit_reaches_the_f1_bar_and_empty_markup_is_safe() -> None:
    truth = _outline()
    network = _outline(fill=((0, 0), (1, 1)))
    cv2.rectangle(network, (250, 8), (258, 16), 255, -1)
    frames = (match_masks(network, truth), match_masks(network, truth))
    fitted = fit_markup(frames)
    assert fitted["f1"] >= 0.8
    assert detection_f1(fitted["metrics"] and frames[0]["components"], fitted["sliders"]) >= 0.8
    empty = fit_markup(())
    assert empty["sliders"] == {}
    calibration = GridCalibration()
    assert calibration.examples == ()


def test_mask_size_mismatch_is_an_error() -> None:
    try:
        match_masks(np.zeros((4, 4), dtype=np.uint8), np.zeros((5, 5), dtype=np.uint8))
    except ValueError:
        return
    raise AssertionError("size mismatch must be reported")
