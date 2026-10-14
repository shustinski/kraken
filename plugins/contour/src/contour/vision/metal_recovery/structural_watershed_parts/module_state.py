"""structural_watershed: module state."""
from __future__ import annotations

STRUCTURAL_WATERSHED_STRATEGY = "structural_watershed"

STRUCTURAL_VARIANTS = (
    "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "s12", "s13", "s14",
)

INSTANCE_IDENTITY_VARIANTS = frozenset(
    {"s5", "s6", "s7", "s8", "s9", "s10", "s11", "s12", "s13", "s14"}
)

GEODESIC_COMPETITION_VARIANTS = frozenset(
    {"s7", "s8", "s9", "s10", "s11", "s12", "s13", "s14"}
)

RIDGE_LINK_VARIANTS = frozenset({"s8", "s10", "s12", "s14"})

WIDE_CONSOLIDATE_VARIANTS = frozenset({"s9", "s10", "s11", "s14"})

BAND_GROUP_VARIANTS = frozenset({"s13", "s14"})

ORIENTATION_VETO_VARIANTS = frozenset({"s12", "s14"})

SEPARATOR_COMBINE_VARIANTS = frozenset({"s11", "s14"})

_FRANGI_BETA = 0.5

_RIDGE_SCALE_STEP = 1.6

_DENSE_RIDGE_SCALE_STEP = 1.35

_RIBBON_HALF_WIDTHS = (2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 14.0)

_NMS_DIRECTIONS = (
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
)
