"""Core-based merged_contour detection."""
from __future__ import annotations

import cv2
import numpy as np

from karakal.core.grid_anomaly import GridDamageAnalysisConfig, detect_grid_cell_anomalies
from karakal.core.grid_merge_cores import (
    analyze_merge_cores,
    find_cell_cores,
    merge_core_thresholds,
    score_merge_cores,
)


def _solid_cell(image: np.ndarray, x: int, y: int, w: int = 22, h: int = 32) -> None:
    cv2.rectangle(image, (x, y), (x + w - 1, y + h - 1), 255, -1)


def _lattice(rows: int = 5, cols: int = 7, cell_w: int = 22, cell_h: int = 32, pitch_x: int = 36, pitch_y: int = 48) -> np.ndarray:
    image = np.zeros((16 + rows * pitch_y, 16 + cols * pitch_x), dtype=np.uint8)
    for row in range(rows):
        for col in range(cols):
            _solid_cell(image, 16 + col * pitch_x, 16 + row * pitch_y, cell_w, cell_h)
    return image


def _calibrated(**sliders: int) -> GridDamageAnalysisConfig:
    return GridDamageAnalysisConfig(
        cell_representation="binary",
        scoring_mode="calibrated",
        blur_radius=0,
        morphology_open=0,
        morphology_close=0,
        min_contour_area=40.0,
        min_cell_size=6,
        fill_sensitivity=int(sliders.get("fill", 60)),
        debris_sensitivity=int(sliders.get("debris", 75)),
        geometry_sensitivity=int(sliders.get("geometry", 40)),
        merge_sensitivity=int(sliders.get("merge", 35)),
        min_grid_candidate_cells=4,
    )


def test_merge_thresholds_loosen_with_sensitivity() -> None:
    low_core, low_tol = merge_core_thresholds(0)
    high_core, high_tol = merge_core_thresholds(100)
    assert high_core < low_core
    assert high_tol > low_tol


def test_pair_with_neck_is_merged() -> None:
    image = _lattice()
    cv2.rectangle(image, (16 + 22, 16 + 12), (16 + 36, 16 + 20), 255, -1)
    result = detect_grid_cell_anomalies(image, config=_calibrated())
    merged = [cell for cell in result.cells if "merged_contour" in cell.reasons]
    assert len(merged) == 1
    assert merged[0].bbox[2] > 30


def test_six_cell_chain_is_one_merge() -> None:
    image = _lattice(rows=4, cols=8)
    y2 = 16 + 48
    for col in range(6):
        _solid_cell(image, 16 + col * 36, y2)
        if col:
            cv2.rectangle(image, (16 + (col - 1) * 36 + 22, y2 + 10), (16 + col * 36, y2 + 18), 255, -1)
    result = detect_grid_cell_anomalies(image, config=_calibrated())
    merged = [cell for cell in result.cells if "merged_contour" in cell.reasons]
    assert len(merged) == 1
    assert merged[0].bbox[2] >= 5 * 22


def test_solid_block_without_neck_still_merges() -> None:
    image = _lattice()
    cv2.rectangle(image, (16, 16), (16 + 36 + 22 - 1, 16 + 32 - 1), 255, -1)
    result = detect_grid_cell_anomalies(image, config=_calibrated())
    merged = [cell for cell in result.cells if "merged_contour" in cell.reasons]
    assert len(merged) == 1


def test_elongated_single_cell_is_not_merge() -> None:
    analysis = analyze_merge_cores(
        np.array([[[16, 16]], [[50, 16]], [[50, 47]], [[16, 47]]], dtype=np.int32),
        (16, 16, 35, 32),
        area=35 * 32,
        cell_width=22,
        cell_height=32,
        merge_sensitivity=35,
    )
    assert analysis.score == 0.0


def test_grain_blob_is_not_merge() -> None:
    rng = np.random.default_rng(7)
    image = _lattice()
    for _ in range(80):
        x = int(rng.integers(20, 260))
        y = int(rng.integers(280, 320))
        w = int(rng.integers(4, 18))
        h = int(rng.integers(4, 14))
        cv2.rectangle(image, (x, y), (x + w, y + h), 255, -1)
    result = detect_grid_cell_anomalies(image, config=_calibrated())
    grain_merges = [
        cell for cell in result.cells if "merged_contour" in cell.reasons and cell.centroid[1] > 270
    ]
    assert grain_merges == []


def test_jpeg_pair_still_merges() -> None:
    image = _lattice()
    cv2.rectangle(image, (16 + 22, 16 + 12), (16 + 36, 16 + 20), 255, -1)
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 55])
    assert ok
    decoded = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(decoded, 127, 255, cv2.THRESH_BINARY)
    result = detect_grid_cell_anomalies(binary, config=_calibrated())
    assert any("merged_contour" in cell.reasons for cell in result.cells)


def test_find_cores_on_two_filled_disks() -> None:
    roi = np.zeros((80, 120), dtype=np.uint8)
    cv2.rectangle(roi, (10, 20), (10 + 27, 20 + 27), 255, -1)
    cv2.rectangle(roi, (10 + 36, 20), (10 + 36 + 27, 20 + 27), 255, -1)
    cv2.rectangle(roi, (10 + 27, 30), (10 + 36, 38), 255, -1)
    cores = find_cell_cores(roi, cell_width=28, cell_height=28, min_core_frac=0.35)
    assert len(cores) >= 2
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    assert contours
    x, y, w, h = cv2.boundingRect(contours[0])
    score = score_merge_cores(
        contours[0],
        (x, y, w, h),
        area=float(cv2.contourArea(contours[0])),
        cell_width=28,
        cell_height=28,
    )
    assert score > 0.02
