"""Apply narrower defect settings to a finished frame result without analyzing the image again.

A run stores every cell it kept. Raising the minimum debris size or switching a defect type off
only removes reasons from those cells, so the matrix can follow such changes at once and match
a fresh run. Settings that could add defects (a smaller debris size, a type switched back on,
other thresholds, calibration) need a new run; see ``can_narrow``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

from .grid_anomaly import (
    CONTOUR_AREA_FEATURE,
    DEBRIS_MIN_AREA_PX,
    LEFTOVER_DEBRIS_FEATURE,
    GridDamageSeverityThresholds,
    GridFrameAnalysisResult,
    _damage_score,
    _status_for_reasons,
)


@dataclass(frozen=True, slots=True)
class GridNarrowing:
    """What the matrix shows on top of the stored run."""

    debris_min_area_px: int
    # None keeps every reason the run produced.
    enabled_reason_types: frozenset[str] | None = None


def can_narrow(
    *,
    run_debris_min_area_px: int,
    run_reason_types: Iterable[str] | None,
    debris_min_area_px: int,
    enabled_reason_types: Iterable[str] | None,
) -> bool:
    """True when the live settings only hide defects of the run, never add any."""

    if int(debris_min_area_px) < int(run_debris_min_area_px):
        return False
    if enabled_reason_types is None:
        return run_reason_types is None
    if run_reason_types is None:
        return True
    return set(enabled_reason_types) <= set(run_reason_types)


def _cell_like(features: dict[str, float]) -> bool:
    """Same size test the analysis uses to keep a defect-free piece as a normal cell."""

    if "width_ratio" not in features or "height_ratio" not in features or "area_ratio" not in features:
        # No relative features (older or template runs): keep the piece as a normal cell.
        return True
    width_ratio = float(features["width_ratio"])
    height_ratio = float(features["height_ratio"])
    area_ratio = float(features["area_ratio"])
    solidity = float(features.get("solidity", 1.0))
    return (
        0.50 <= min(width_ratio, height_ratio)
        and max(width_ratio, height_ratio) <= 1.85
        and 0.35 <= area_ratio <= 2.40
        and solidity >= 0.45
    )


def narrow_grid_frame_result(
    result: GridFrameAnalysisResult | None,
    narrowing: GridNarrowing,
) -> GridFrameAnalysisResult | None:
    """Drop the defects the narrower settings hide and recount the frame score.

    Returns the same object when nothing changes, so unchanged frames keep their exact run values.
    """

    if not isinstance(result, GridFrameAnalysisResult):
        return result
    enabled = narrowing.enabled_reason_types
    min_area = float(narrowing.debris_min_area_px)
    changed = False
    cells = []
    for cell in result.per_cell_results or ():
        reasons = tuple(str(reason) for reason in (cell.reasons or ()))
        if not reasons:
            cells.append(cell)
            continue
        kept = reasons
        if enabled is not None:
            kept = tuple(reason for reason in kept if reason in enabled or reason == "conductor_zone")
        features = dict(cell.feature_snapshot or ())
        if "small_artifact" in kept and float(features.get(CONTOUR_AREA_FEATURE, min_area)) < min_area:
            kept = tuple(reason for reason in kept if reason != "small_artifact")
        if kept == reasons:
            cells.append(cell)
            continue
        changed = True
        if kept:
            cells.append(replace(cell, reasons=kept, status=_status_for_reasons(kept)))
            continue
        # The run skips a defect-free piece unless it is cell-sized.
        if LEFTOVER_DEBRIS_FEATURE in features or not _cell_like(features):
            continue
        cells.append(replace(cell, reasons=(), status="normal", score=0.0))
    if not changed:
        return result
    cell_rows = [cell for cell in cells if "conductor_zone" not in (cell.reasons or ())]
    damage = _damage_score(cell_rows, max(1, len(cell_rows))) if cell_rows else 0.0
    bad = [cell for cell in cell_rows if cell.status != "normal"]
    return replace(
        result,
        per_cell_results=tuple(cells),
        detected_cells=len(cell_rows),
        total_expected_cells=len(cell_rows),
        normal_cells=len(cell_rows) - len(bad),
        suspicious_cells=sum(1 for cell in bad if cell.status == "suspicious"),
        broken_cells=sum(1 for cell in bad if cell.status == "broken"),
        missing_cells=sum(1 for cell in bad if cell.status == "missing"),
        artifact_cells=sum(1 for cell in bad if cell.status == "artifact"),
        damage_score=float(damage),
        severity_level=str(GridDamageSeverityThresholds().level_for_score(damage)),
    )


def default_run_debris_min_area(config_payload: dict | None) -> int:
    """Smallest debris a stored run kept.

    Runs since algorithm 93 keep debris down to the analysis floor whatever the operator
    set; older runs dropped pieces under the operator's size (or the fixed 24 px).
    """

    payload = config_payload or {}
    try:
        if "debris_analysis_floor_px" in payload:
            return int(float(payload["debris_analysis_floor_px"]))
        return int(payload.get("debris_min_area_px", DEBRIS_MIN_AREA_PX))
    except (TypeError, ValueError):
        return int(DEBRIS_MIN_AREA_PX)
