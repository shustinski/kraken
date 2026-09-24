"""Unit tests for ones∩zeros class conflict review and confidence-aware broken geometry."""

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


@pytest.mark.skipif(grid_anomaly.cv2 is None, reason="OpenCV required for class-conflict CCs")
def test_class_conflict_no_overlap_is_clean() -> None:
    ones = np.zeros((64, 64), dtype=bool)
    zeros = np.zeros((64, 64), dtype=bool)
    ones[8:28, 8:28] = True
    zeros[36:56, 36:56] = True

    result = grid_anomaly.analyze_class_conflict_masks(ones, zeros, frame_id="disjoint")

    assert result.damage_score == 0.0
    assert result.per_cell_results == ()
    assert result.grid_detected is True


@pytest.mark.skipif(grid_anomaly.cv2 is None, reason="OpenCV required for class-conflict CCs")
def test_class_conflict_large_overlap_needs_review() -> None:
    ones = np.zeros((64, 64), dtype=bool)
    zeros = np.zeros((64, 64), dtype=bool)
    ones[10:40, 10:40] = True
    zeros[20:35, 20:35] = True  # 15x15 = 225 px overlap

    result = grid_anomaly.analyze_class_conflict_masks(ones, zeros, frame_id="overlap")

    assert len(result.per_cell_results) == 1
    cell = result.per_cell_results[0]
    assert cell.reasons == ("class_conflict",)
    assert cell.status == "suspicious"
    assert result.damage_score > 0.0


@pytest.mark.skipif(grid_anomaly.cv2 is None, reason="OpenCV required for class-conflict CCs")
def test_class_conflict_identical_masks_are_conflict() -> None:
    """Same silhouette on ones and zeros is a full class clash."""

    mask = np.zeros((48, 48), dtype=bool)
    mask[8:40, 8:40] = True

    result = grid_anomaly.analyze_class_conflict_masks(mask, mask, frame_id="identical")

    assert len(result.per_cell_results) >= 1
    assert result.per_cell_results[0].reasons == ("class_conflict",)
    assert result.damage_score > 0.0


@pytest.mark.skipif(grid_anomaly.cv2 is None, reason="OpenCV required for class-conflict CCs")
def test_class_conflict_tiny_dirt_is_ignored() -> None:
    ones = np.zeros((64, 64), dtype=bool)
    zeros = np.zeros((64, 64), dtype=bool)
    ones[10:40, 10:40] = True
    zeros[50:52, 50:52] = True
    ones[50:52, 50:52] = True  # 2x2 overlap dirt
    zeros[55, 55:60] = True
    ones[55, 55:60] = True  # 5 px line overlap

    result = grid_anomaly.analyze_class_conflict_masks(ones, zeros, frame_id="dirt")

    assert result.per_cell_results == ()
    assert result.damage_score == 0.0


@pytest.mark.skipif(grid_anomaly.cv2 is None, reason="OpenCV required for class-conflict CCs")
def test_class_conflict_thin_contour_fringe_is_ignored() -> None:
    ones = np.zeros((64, 64), dtype=bool)
    zeros = np.zeros((64, 64), dtype=bool)
    ones[10:40, 10:40] = True
    zeros[10:40, 39:41] = True  # overlap = column x=39 → 1×30 strip

    result = grid_anomaly.analyze_class_conflict_masks(ones, zeros, frame_id="fringe")

    assert result.per_cell_results == ()
    assert result.damage_score == 0.0


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


def test_derived_conflict_i18n_labels_exist() -> None:
    translator = Translator()
    assert translator.tr("grid_layer.derived_conflict")
    assert translator.tr("grid_error.class_conflict")


def test_class_conflict_is_selectable_error_type() -> None:
    from karakal.ui.ui_constants import GRID_INSPECTION_ERROR_TYPE_OPTIONS, GRID_INSPECTION_ERROR_TYPE_COLORS

    values = {error_type for _label, error_type in GRID_INSPECTION_ERROR_TYPE_OPTIONS}
    assert "class_conflict" in values
    assert "class_conflict" in GRID_INSPECTION_ERROR_TYPE_COLORS
    assert "class_conflict" in grid_anomaly.GRID_DAMAGE_REASON_TYPES
