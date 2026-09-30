"""Confidence fill is scored inside binary cell interiors, not from confidence contours."""
from __future__ import annotations

import cv2
import numpy as np

from karakal.core.grid_anomaly import (
    GridCellAnalysisResult,
    GridDamageAnalysisConfig,
    GridFrameAnalysisResult,
    classify_confidence_map_kind,
    score_confidence_fill_on_binary_result,
)


def _cell(bbox, outline, *, row=0) -> GridCellAnalysisResult:
    x, y, w, h = bbox
    return GridCellAnalysisResult(
        row=row,
        col=0,
        bbox=bbox,
        centroid=(x + w / 2.0, y + h / 2.0),
        contour_id=row,
        status="normal",
        score=0.0,
        reasons=(),
        outline=tuple(outline),
    )


def _square_outline(x: int, y: int, w: int, h: int) -> list[tuple[int, int]]:
    return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]


def _binary_frame(cells: list[GridCellAnalysisResult], shape=(120, 120)) -> GridFrameAnalysisResult:
    return GridFrameAnalysisResult(
        frame_id="t",
        frame_path="",
        image_width=shape[1],
        image_height=shape[0],
        grid_rows=1,
        grid_cols=len(cells),
        total_expected_cells=len(cells),
        detected_cells=len(cells),
        normal_cells=len(cells),
        suspicious_cells=0,
        broken_cells=0,
        missing_cells=0,
        artifact_cells=0,
        damage_score=0.0,
        severity_level="OK",
        grid_detected=True,
        per_cell_results=tuple(cells),
        component_count=len(cells),
        cell_width=24,
        cell_height=24,
    )


def test_white_interior_with_dark_ring_is_normal() -> None:
    cells = [
        _cell((10, 10, 30, 30), _square_outline(10, 10, 30, 30), row=0),
        _cell((50, 10, 30, 30), _square_outline(50, 10, 30, 30), row=1),
        _cell((10, 50, 30, 30), _square_outline(10, 50, 30, 30), row=2),
        _cell((50, 50, 30, 30), _square_outline(50, 50, 30, 30), row=3),
        _cell((90, 50, 24, 24), _square_outline(90, 50, 24, 24), row=4),
    ]
    conf = np.full((120, 120), 253, dtype=np.uint8)
    # Dark ring on every cell — should be ignored after interior erosion.
    for cell in cells:
        x, y, w, h = cell.bbox
        conf[y : y + 2, x : x + w] = 180
        conf[y + h - 2 : y + h, x : x + w] = 180
        conf[y : y + h, x : x + 2] = 180
        conf[y : y + h, x + w - 2 : x + w] = 180
    result = score_confidence_fill_on_binary_result(_binary_frame(cells), conf)
    assert not any(cell.reasons for cell in result.per_cell_results)


def test_dark_interior_is_filled_cell() -> None:
    cells = [
        _cell((10, 10, 30, 30), _square_outline(10, 10, 30, 30), row=i) for i in range(6)
    ]
    # Spread cells so packing works.
    cells = [
        _cell((10 + (i % 3) * 36, 10 + (i // 3) * 40, 30, 30), _square_outline(10 + (i % 3) * 36, 10 + (i // 3) * 40, 30, 30), row=i)
        for i in range(6)
    ]
    conf = np.full((120, 120), 253, dtype=np.uint8)
    x, y, w, h = cells[0].bbox
    conf[y + 4 : y + h - 4, x + 4 : x + w - 4] = 200  # ~53 levels down
    result = score_confidence_fill_on_binary_result(_binary_frame(cells), conf)
    hit = next(cell for cell in result.per_cell_results if cell.row == 0)
    assert "filled_cell" in hit.reasons


def test_coherent_blotch_is_partial_fill() -> None:
    cells = [
        _cell((10 + (i % 3) * 36, 10 + (i // 3) * 40, 30, 30), _square_outline(10 + (i % 3) * 36, 10 + (i // 3) * 40, 30, 30), row=i)
        for i in range(6)
    ]
    conf = np.full((120, 120), 253, dtype=np.uint8)
    x, y, w, h = cells[1].bbox
    # ~25 % of interior as one blotch, clearly below the 8-level drop floor.
    conf[y + 6 : y + 16, x + 6 : x + 20] = 200
    result = score_confidence_fill_on_binary_result(_binary_frame(cells), conf)
    hit = next(cell for cell in result.per_cell_results if cell.row == 1)
    assert "partial_filled_cell" in hit.reasons
    assert "filled_cell" not in hit.reasons


def test_scattered_noise_three_levels_is_normal() -> None:
    cells = [
        _cell((10 + (i % 3) * 36, 10 + (i // 3) * 40, 30, 30), _square_outline(10 + (i % 3) * 36, 10 + (i // 3) * 40, 30, 30), row=i)
        for i in range(6)
    ]
    conf = np.full((120, 120), 253, dtype=np.uint8)
    rng = np.random.default_rng(0)
    noise = rng.integers(-3, 4, size=conf.shape)
    conf = np.clip(conf.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    result = score_confidence_fill_on_binary_result(_binary_frame(cells), conf)
    assert not any(cell.reasons for cell in result.per_cell_results)


def test_map_kind_bimodal_vs_bright() -> None:
    bright = np.full((64, 64), 250, dtype=np.uint8)
    bright[10:14, 10:50] = 230  # thin dips
    assert classify_confidence_map_kind(bright) == "confidence"
    bimodal = np.zeros((64, 64), dtype=np.uint8)
    bimodal[0:32, :] = 255
    assert classify_confidence_map_kind(bimodal) == "class_probability"


def _full_frame_interior_mask(outline, shape, erode_px=3):
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.asarray(outline, dtype=np.int32).reshape(-1, 1, 2)], 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px * 2 + 1, erode_px * 2 + 1))
    eroded = cv2.erode(mask, kernel, iterations=1)
    return eroded if int(np.count_nonzero(eroded)) >= 8 else mask


def test_cell_interior_crop_matches_full_frame_mask() -> None:
    # Per-cell full-frame masks cost cells x frame pixels; the crop must keep identical pixels.
    from karakal.core.grid_anomaly import _cell_interior_mask_from_outline

    shape = (90, 140)
    outlines = [
        _square_outline(10, 10, 30, 30),
        [(0, 0), (20, 0), (20, 15), (0, 15)],  # touches the frame corner
        [(100, 60), (139, 62), (135, 89), (104, 85)],  # touches the far edges
        [(60, 20), (66, 20), (66, 24), (60, 24)],  # too small to survive erosion
        [(70, 30), (95, 35), (88, 55), (72, 50), (80, 42)],  # concave
    ]
    for outline in outlines:
        crop = _cell_interior_mask_from_outline(outline, shape, erode_px=3)
        assert crop is not None
        x0, y0, mask = crop
        assert mask.size < shape[0] * shape[1]
        rebuilt = np.zeros(shape, dtype=np.uint8)
        rebuilt[y0 : y0 + mask.shape[0], x0 : x0 + mask.shape[1]] = mask
        np.testing.assert_array_equal(rebuilt, _full_frame_interior_mask(outline, shape))
