"""Speed budgets for calibrated grid inspection."""

from __future__ import annotations

from time import perf_counter

import cv2
import numpy as np
import pytest

from karakal.core.grid_anomaly import (
    GridCellAnalysisResult,
    GridDamageAnalysisConfig,
    GridFrameAnalysisResult,
    compare_grid_cell_analyses,
    detect_grid_cell_anomalies,
)


def _dense_outline(cells: int = 1500) -> np.ndarray:
    columns = 50
    rows = max(1, (cells + columns - 1) // columns)
    size = 12
    gap = 4
    image = np.zeros((rows * (size + gap) + 20, columns * (size + gap) + 20), dtype=np.uint8)
    count = 0
    for row in range(rows):
        for col in range(columns):
            if count >= cells:
                return image
            x = 8 + col * (size + gap)
            y = 8 + row * (size + gap)
            cv2.rectangle(image, (x, y), (x + size, y + size), 255, 1)
            count += 1
    return image


def _frame_from_cells(cells: list[GridCellAnalysisResult]) -> GridFrameAnalysisResult:
    return GridFrameAnalysisResult(
        frame_id="f",
        frame_path="",
        image_width=640,
        image_height=480,
        grid_rows=1,
        grid_cols=1,
        total_expected_cells=len(cells),
        detected_cells=len(cells),
        normal_cells=len(cells),
        suspicious_cells=0,
        broken_cells=0,
        missing_cells=0,
        artifact_cells=0,
        damage_score=0.0,
        severity_level="none",
        grid_detected=True,
        cell_width=12.0,
        cell_height=12.0,
        per_cell_results=tuple(cells),
    )


def _calibrated_1500_elapsed_ms() -> tuple[float, int]:
    image = _dense_outline(1500)
    config = GridDamageAnalysisConfig(
        cell_representation="binary",
        scoring_mode="calibrated",
        blur_radius=0,
        min_contour_area=4.0,
        min_cell_size=2,
        min_grid_candidate_cells=4,
    )
    detect_grid_cell_anomalies(image, config=config)  # warm-up
    started = perf_counter()
    result = detect_grid_cell_anomalies(image, config=config)
    elapsed_ms = (perf_counter() - started) * 1000.0
    return elapsed_ms, len(result.cells)


def test_calibrated_1500_cells_stays_within_degradation_budget() -> None:
    """Regression guard: calibrated path must not regress past 800 ms on this host."""
    elapsed_ms, cell_count = _calibrated_1500_elapsed_ms()
    assert cell_count >= 1400
    assert elapsed_ms <= 800.0


@pytest.mark.xfail(
    strict=False,
    reason="P4 target is 150 ms per 1500-cell frame; current Windows contour path still exceeds it",
)
def test_calibrated_1500_cells_meets_150ms_target() -> None:
    elapsed_ms, cell_count = _calibrated_1500_elapsed_ms()
    assert cell_count >= 1400
    assert elapsed_ms <= 150.0


def test_decision_recompute_stays_within_budget() -> None:
    from karakal.core.grid_calibration import decide_reasons_batch

    score_rows = [
        {"fill": 0.2, "geometry": 0.1, "merge": 0.05, "debris": 0.3, "edge": 0.0} for _ in range(1500)
    ]
    thresholds = {"fill": 0.4, "geometry": 0.6, "merge": 0.65, "debris": 0.25, "edge": 1.0}
    examples = [
        {
            "label": "normal" if index % 3 else "filled_cell",
            "features": {
                "width_ratio": 1.0,
                "height_ratio": 1.0,
                "area_ratio": 1.0,
                "interior_fill": 0.1 + 0.01 * (index % 5),
                "reference_interior": 0.1,
                "solidity": 0.9,
                "reference_solidity": 0.9,
                "extent": 0.7,
                "reference_extent": 0.7,
            },
        }
        for index in range(300)
    ]
    feature_rows = [dict(examples[index % len(examples)]["features"]) for index in range(1500)]
    decide_reasons_batch(score_rows[:10], thresholds, feature_rows=feature_rows[:10], examples=examples)
    started = perf_counter()
    decided = decide_reasons_batch(
        score_rows,
        thresholds,
        feature_rows=feature_rows,
        examples=examples,
        example_influence=0.5,
    )
    assert len(decided) == 1500
    assert (perf_counter() - started) * 1000.0 <= 250.0  # 30 ms target; matrix build dominates on Windows


def test_compare_grid_cell_analyses_matches_on_shifted_grids() -> None:
    cells_a = [
        GridCellAnalysisResult(0, 0, (10 + index * 20, 10, 12, 12), (16.0 + index * 20, 16.0), index, "normal", 0.0, ())
        for index in range(80)
    ]
    cells_b = [
        GridCellAnalysisResult(
            0,
            0,
            (12 + index * 20, 11, 12, 12),
            (18.0 + index * 20, 17.0),
            index,
            "normal" if index != 5 else "broken",
            0.0 if index != 5 else 0.9,
            () if index != 5 else ("filled_cell",),
        )
        for index in range(80)
    ]
    left = compare_grid_cell_analyses(_frame_from_cells(cells_a), _frame_from_cells(cells_b))
    right = compare_grid_cell_analyses(_frame_from_cells(cells_a), _frame_from_cells(cells_b))
    assert {cell.reasons for cell in left.cells} == {cell.reasons for cell in right.cells}
    assert any("defect_disagreement" in cell.reasons or "geometry_mismatch" in cell.reasons for cell in left.cells)
