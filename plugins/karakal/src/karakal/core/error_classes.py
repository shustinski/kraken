"""Canonical error classes of the cell-defect analysis.

The number is the value written to ``class_map``; names and colors are only for people.
The old reason strings of the grid analysis map onto these classes; touching the
frame border is a flag, not a class.
"""

from __future__ import annotations

OK = 0
BAD_GEOMETRY = 1
DEBRIS = 2
MERGE = 3
SPLIT = 4
MISSED = 5
UNKNOWN = 6
IGNORE = 255

ERROR_CLASS_NAMES: dict[int, str] = {
    OK: "ok",
    BAD_GEOMETRY: "bad_geometry",
    DEBRIS: "debris",
    MERGE: "merge",
    SPLIT: "split",
    MISSED: "missed",
    UNKNOWN: "unknown_anomaly",
    IGNORE: "ignore",
}

# Error classes the runtime analysis can emit; MISSED needs ground truth of the same frame.
RUNTIME_ERROR_CLASSES: tuple[int, ...] = (BAD_GEOMETRY, DEBRIS, MERGE, SPLIT, UNKNOWN)
ERROR_CLASSES: tuple[int, ...] = (BAD_GEOMETRY, DEBRIS, MERGE, SPLIT, MISSED, UNKNOWN)

# When one object carries several findings, the class map shows the first of this order;
# the others stay in the object's reasons.
CLASS_PRIORITY: tuple[int, ...] = (IGNORE, SPLIT, MERGE, BAD_GEOMETRY, DEBRIS, UNKNOWN, OK)

# Geometry, debris and merge keep the colors of the old analysis; the rest is new.
ERROR_CLASS_COLORS: dict[int, str] = {
    OK: "#000000",
    BAD_GEOMETRY: "#38bdf8",
    DEBRIS: "#ec4899",
    MERGE: "#a855f7",
    SPLIT: "#f97316",
    MISSED: "#22c55e",
    UNKNOWN: "#eab308",
    IGNORE: "#64748b",
}

LEGACY_REASON_CLASSES: dict[str, int] = {
    "broken_geometry": BAD_GEOMETRY,
    "filled_cell": BAD_GEOMETRY,
    "partial_filled_cell": BAD_GEOMETRY,
    "small_artifact": DEBRIS,
    "merged_contour": MERGE,
    "conductor_zone": IGNORE,
    "class_conflict": UNKNOWN,
}
# Reasons that are flags on an object, never its class.
LEGACY_FLAG_REASONS = frozenset({"edge_clipped_cell"})


def dominant_class(classes) -> int:
    """The class the class map shows for an object with these findings."""

    present = {int(value) for value in classes}
    for value in CLASS_PRIORITY:
        if value in present:
            return value
    return OK


def class_from_legacy_reasons(reasons) -> int:
    return dominant_class(LEGACY_REASON_CLASSES[str(reason)] for reason in reasons if str(reason) in LEGACY_REASON_CLASSES)
