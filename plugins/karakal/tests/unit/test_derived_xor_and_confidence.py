"""Confidence-aware broken geometry and the class-conflict error type (conflicts: test_pair_conflict.py)."""

from __future__ import annotations

import numpy as np
import pytest

from karakal.core import grid_anomaly
from karakal.ui.i18n import Translator


def _candidate(
    contour_id: int,
    bbox: tuple[int, int, int, int],
    *,
    area: float | None = None,
    fill: float = 0.12,
    interior: float = 0.04,
    center: float = 0.02,
    solidity: float = 0.94,
    extent: float = 0.74,
    vertices: int = 4,
    hole: float = 0.04,
    child_count: int = 1,
    touches_border: bool = False,
    outline_min: float = 0.72,
    outline_imbalance: float = 0.08,
):
    x, y, width, height = bbox
    bbox_area = float(width * height)
    return grid_anomaly._ContourCandidate(
        contour_id=int(contour_id),
        contour=None,
        bbox=(int(x), int(y), int(width), int(height)),
        centroid=(float(x + width / 2.0), float(y + height / 2.0)),
        area=float(bbox_area * 0.70 if area is None else area),
        bbox_area=bbox_area,
        aspect_ratio=float(width / max(1, height)),
        extent=float(extent),
        solidity=float(solidity),
        perimeter=float(width * 2 + height * 2),
        approx_vertices=int(vertices),
        fill_ratio=float(fill),
        interior_fill_ratio=float(interior),
        center_fill_ratio=float(center),
        outline_min_side_coverage=float(outline_min),
        outline_mean_side_coverage=max(float(outline_min), 0.78),
        outline_side_imbalance=float(outline_imbalance),
        inner_hole_ratio=float(hole),
        child_count=int(child_count),
        touches_border=bool(touches_border),
    )


def test_confidence_boost_flags_ragged_low_confidence_cell() -> None:
    candidate = _candidate(
        1,
        (20, 20, 10, 10),
        area=70.0,
        solidity=0.86,
        extent=0.74,
        vertices=5,
        fill=0.12,
        interior=0.04,
        child_count=0,
    )
    confidence = np.full((64, 64), 0.85, dtype=np.float32)
    confidence[20:30, 20:30] = 0.28

    score, reasons = grid_anomaly._classify_detected_cell(
        candidate,
        median_width=10.0,
        median_height=10.0,
        median_area=70.0,
        median_fill=0.12,
        median_interior_fill=0.04,
        median_center_fill=0.02,
        median_aspect=1.0,
        config=grid_anomaly.GridDamageAnalysisConfig(cell_representation="confidence"),
        confidence_map=confidence,
    )

    assert "broken_geometry" in reasons
    assert score >= 0.78


def test_confidence_boost_keeps_clean_high_confidence_cell_normal() -> None:
    candidate = _candidate(
        2,
        (20, 20, 10, 10),
        area=70.0,
        solidity=0.96,
        extent=0.74,
        vertices=4,
        fill=0.12,
        interior=0.04,
        child_count=1,
    )
    confidence = np.full((64, 64), 0.90, dtype=np.float32)

    score, reasons = grid_anomaly._classify_detected_cell(
        candidate,
        median_width=10.0,
        median_height=10.0,
        median_area=70.0,
        median_fill=0.12,
        median_interior_fill=0.04,
        median_center_fill=0.02,
        median_aspect=1.0,
        config=grid_anomaly.GridDamageAnalysisConfig(cell_representation="confidence"),
        confidence_map=confidence,
    )

    assert reasons == ()
    assert score == 0.0


def test_confidence_boost_is_local_to_candidate_bbox() -> None:
    candidate = _candidate(
        3,
        (20, 20, 10, 10),
        area=70.0,
        solidity=0.88,
        extent=0.74,
        vertices=5,
        fill=0.12,
        interior=0.04,
        child_count=0,
    )
    confidence = np.full((64, 64), 0.12, dtype=np.float32)
    confidence[20:30, 20:30] = 0.88

    score, reasons = grid_anomaly._apply_confidence_geometry_boost(
        candidate,
        0.0,
        [],
        median_width=10.0,
        median_height=10.0,
        median_area=70.0,
        confidence_map=confidence,
    )

    assert reasons == []
    assert score == 0.0


def test_class_conflict_i18n_label_exists() -> None:
    assert Translator().tr("grid_error.class_conflict")


def test_class_conflict_is_selectable_error_type() -> None:
    from karakal.ui.ui_constants import GRID_INSPECTION_ERROR_TYPE_OPTIONS, GRID_INSPECTION_ERROR_TYPE_COLORS

    values = {error_type for _label, error_type in GRID_INSPECTION_ERROR_TYPE_OPTIONS}
    assert "class_conflict" in values
    assert "class_conflict" in GRID_INSPECTION_ERROR_TYPE_COLORS
    assert "class_conflict" in grid_anomaly.GRID_DAMAGE_REASON_TYPES
