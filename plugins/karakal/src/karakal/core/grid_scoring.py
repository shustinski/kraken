"""Continuous defect scores for calibrated grid inspection.

Legacy classification stays in grid_anomaly. This module is Qt-free and picklable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def slider_threshold(slider: int) -> float:
    """Map a 0..100 sensitivity slider to a defect threshold. Higher slider, lower threshold."""

    unit = max(0.0, min(100.0, float(slider))) / 100.0
    return 1.0 - unit


def _logistic(value: float) -> float:
    clipped = max(-12.0, min(12.0, float(value)))
    return 1.0 / (1.0 + math.exp(-clipped))


def score_fill(interior_fill: float, reference_interior: float) -> float:
    """How much denser this cell interior is than a normal cell."""

    return _logistic((float(interior_fill) - float(reference_interior) - 0.24) / 0.06)


PARTIAL_FILL_GAP = 0.15
NORMAL_SCORE_CEILING = 0.02
# Cells this much shorter or narrower than their neighbours are natural spread, not broken.
SIZE_SPREAD_RATIO = 0.85
# Rounded cells empty their corners as much as a slanted end does, so the corner counts
# at a fraction: a balanced preset marks only cut-off corners, a strict one also slants.
CORNER_WEIGHT = 0.4


def geometry_agrees_with_neighbors(
    solidity: float,
    extent: float,
    neighbor_solidity: float,
    neighbor_extent: float,
    *,
    neighbors_unmarked: bool,
    tolerance: float = 0.08,
    window_fill: float | None = None,
    neighbor_window_fill: float | None = None,
    notch_depth: float | None = None,
    neighbor_notch_depth: float | None = None,
    corner_fill: float | None = None,
    neighbor_corner_fill: float | None = None,
) -> bool:
    """A shared local shape is not broken geometry. Fill is never affected.

    A notch in a long cell barely moves solidity and extent, so the local measures
    must agree with the neighbors too.
    """

    if not neighbors_unmarked:
        return False
    if window_fill is not None and neighbor_window_fill is not None:
        if abs(float(window_fill) - float(neighbor_window_fill)) > tolerance:
            return False
    if notch_depth is not None and neighbor_notch_depth is not None:
        if abs(float(notch_depth) - float(neighbor_notch_depth)) > tolerance:
            return False
    if corner_fill is not None and neighbor_corner_fill is not None:
        if abs(float(corner_fill) - float(neighbor_corner_fill)) > tolerance:
            return False
    return (
        abs(float(solidity) - float(neighbor_solidity)) <= tolerance
        and abs(float(extent) - float(neighbor_extent)) <= tolerance
    )


def score_geometry(
    solidity: float,
    extent: float,
    reference_solidity: float,
    reference_extent: float,
    *,
    window_fill: float | None = None,
    reference_window_fill: float | None = None,
    notch_depth: float | None = None,
    reference_notch_depth: float | None = None,
    corner_fill: float | None = None,
    reference_corner_fill: float | None = None,
    size_ratio: float | None = None,
) -> float:
    """How far the cell shape is from the normal-cell reference.

    Solidity and extent cover the whole cell, so a notch the size of the short side
    weighs less the longer the cell is. Two local measures keep the same notch at the
    same score for any cell shape: the fill of the worst square window (side = short
    side) and the deepest notch in short sides. For a square cell the window is the
    whole cell.

    A cut or slanted end shows in the emptiest corner (a square of half the short side).
    A cell drawn only in part keeps one side and loses the other: ``size_ratio`` is the
    smaller of width and length over the neighbouring cells; 15% short is natural spread.
    """

    deviation = max(float(reference_solidity) - float(solidity), abs(float(extent) - float(reference_extent)))
    if window_fill is not None and reference_window_fill is not None:
        deviation = max(deviation, float(reference_window_fill) - float(window_fill))
    if notch_depth is not None and reference_notch_depth is not None:
        deviation = max(deviation, float(notch_depth) - float(reference_notch_depth))
    if corner_fill is not None and reference_corner_fill is not None:
        deviation = max(deviation, CORNER_WEIGHT * (float(reference_corner_fill) - float(corner_fill)))
    if size_ratio is not None:
        deviation = max(deviation, SIZE_SPREAD_RATIO - float(size_ratio))
    return _logistic((deviation - 0.16) / 0.04)


def score_size_shortfall(size_ratio: float) -> float:
    """Geometry score from size alone: a cell shorter or narrower than its neighbours."""

    return _logistic((SIZE_SPREAD_RATIO - float(size_ratio) - 0.16) / 0.04)


def score_merge(largest_axis_ratio: float, area_ratio: float) -> float:
    """How far the contour extends past a single cell slot.

    A pair or triple of slots scores high. A field-sized mass is not stuck cells.
    """

    excess = max(float(largest_axis_ratio) - 1.0, float(area_ratio) - 1.0)
    if excess >= 2.8:
        return 0.0
    return _logistic((excess - 0.80) / 0.20)


def score_debris(area_ratio: float, largest_axis_ratio: float) -> float:
    """How much a component looks like debris instead of a full cell."""

    small = (0.45 - float(area_ratio)) / 0.12
    not_a_slot = (1.15 - float(largest_axis_ratio)) / 0.20
    return _logistic(min(small, not_a_slot))


def score_edge(touches_border: bool, smallest_axis_ratio: float) -> float:
    """How much a border cell is cropped compared with a full slot."""

    if not touches_border:
        return 0.0
    return _logistic((0.82 - float(smallest_axis_ratio)) / 0.10)


@dataclass(frozen=True, slots=True)
class CalibratedScores:
    fill: float
    geometry: float
    merge: float
    debris: float
    edge: float

    def reasons(
        self,
        *,
        fill: float,
        geometry: float,
        merge: float,
        debris: float,
        edge: float,
        edge_enabled: bool,
    ) -> tuple[str, ...]:
        found: list[str] = []
        fill_threshold = max(float(fill), NORMAL_SCORE_CEILING)
        partial_threshold = max(fill_threshold - PARTIAL_FILL_GAP, NORMAL_SCORE_CEILING)
        if self.fill > NORMAL_SCORE_CEILING and self.fill >= fill_threshold:
            found.append("filled_cell")
        elif self.fill > NORMAL_SCORE_CEILING and partial_threshold < fill_threshold and self.fill >= partial_threshold:
            found.append("partial_filled_cell")
        if self.geometry > NORMAL_SCORE_CEILING and self.geometry >= max(float(geometry), NORMAL_SCORE_CEILING):
            found.append("broken_geometry")
        if self.merge > NORMAL_SCORE_CEILING and self.merge >= max(float(merge), NORMAL_SCORE_CEILING):
            found.append("merged_contour")
        if self.debris > NORMAL_SCORE_CEILING and self.debris >= max(float(debris), NORMAL_SCORE_CEILING) and "merged_contour" not in found:
            found.append("small_artifact")
        if edge_enabled and self.edge >= edge and "broken_geometry" in found:
            found = [reason for reason in found if reason != "broken_geometry"]
            found.append("edge_clipped_cell")
        return tuple(found)
