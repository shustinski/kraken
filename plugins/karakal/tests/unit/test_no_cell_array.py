"""Frames without a reliable cell array must not invent crumb-sized defects."""
from __future__ import annotations

import cv2
import numpy as np

from karakal.core.grid_anomaly import (
    GridCellReferenceProfile,
    GridDamageAnalysisConfig,
    detect_grid_cell_anomalies,
)


def _cfg() -> GridDamageAnalysisConfig:
    return GridDamageAnalysisConfig(
        cell_representation="binary",
        scoring_mode="calibrated",
        blur_radius=0,
        morphology_open=0,
        morphology_close=0,
        min_contour_area=4.0,
        min_cell_size=2,
        fill_sensitivity=60,
        debris_sensitivity=75,
        geometry_sensitivity=40,
        merge_sensitivity=35,
        min_grid_candidate_cells=4,
    )


def _run_profile() -> GridCellReferenceProfile:
    return GridCellReferenceProfile(
        median_width=23.0,
        median_height=26.0,
        median_area=501.0,
        median_fill=0.9,
        median_interior_fill=1.0,
        median_center_fill=1.0,
        median_aspect=0.9,
        candidate_count=20,
        seed_count=1000,
        frame_id="run:direct",
        frame_path="",
    )


def test_crumb_frame_with_run_profile_is_no_cell_array() -> None:
    image = np.zeros((400, 400), dtype=np.uint8)
    rng = np.random.default_rng(0)
    for _ in range(400):
        x = int(rng.integers(10, 380))
        y = int(rng.integers(10, 380))
        w = int(rng.integers(4, 12))
        h = int(rng.integers(4, 12))
        cv2.rectangle(image, (x, y), (x + w, y + h), 255, -1)
    result = detect_grid_cell_anomalies(image, config=_cfg(), reference_profile=_run_profile())
    assert result.severity_level == "no_cell_array"
    assert result.grid_detected is False
    assert not any(
        reason in {"broken_geometry", "filled_cell", "partial_filled_cell", "merged_contour", "small_artifact"}
        for cell in result.cells
        for reason in cell.reasons
    )


def test_partial_lattice_with_run_profile_is_checked() -> None:
    image = np.zeros((400, 500), dtype=np.uint8)
    for row in range(8):
        for col in range(10):
            x = 20 + col * 36
            y = 40 + row * 48
            cv2.rectangle(image, (x, y), (x + 22, y + 32), 255, -1)
    result = detect_grid_cell_anomalies(image, config=_cfg(), reference_profile=_run_profile())
    assert result.grid_detected is True
    assert result.severity_level != "no_cell_array"
    assert result.detected_cells >= 40
