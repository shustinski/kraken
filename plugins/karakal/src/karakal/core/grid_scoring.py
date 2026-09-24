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


def geometry_agrees_with_neighbors(
    solidity: float,
    extent: float,
    neighbor_solidity: float,
    neighbor_extent: float,
    *,
    neighbors_unmarked: bool,
    tolerance: float = 0.08,
) -> bool:
    """A shared local shape is not broken geometry. Fill is never affected."""

    if not neighbors_unmarked:
        return False
    return (
        abs(float(solidity) - float(neighbor_solidity)) <= tolerance
        and abs(float(extent) - float(neighbor_extent)) <= tolerance
    )


def score_geometry(solidity: float, extent: float, reference_solidity: float, reference_extent: float) -> float:
    """How far solidity and extent are from the normal-cell reference."""

    solidity_drop = float(reference_solidity) - float(solidity)
    extent_shift = abs(float(extent) - float(reference_extent))
    return _logistic((max(solidity_drop, extent_shift) - 0.16) / 0.04)


def score_merge(largest_axis_ratio: float, area_ratio: float) -> float:
    """How far the contour extends past a single cell slot."""

    return _logistic((max(float(largest_axis_ratio) - 1.0, float(area_ratio) - 1.0) - 0.80) / 0.20)


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
