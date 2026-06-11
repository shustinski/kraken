"""Comparison profile selection."""
from __future__ import annotations

from .models import ComparisonProfile

PIXEL_GROUP = "pixel"
SOFT_GROUP = "soft_confidence"
GEOMETRY_GROUP = "geometry"
COMPONENTS_GROUP = "components"
SKELETON_GROUP = "skeleton"
TOPOLOGY_GROUP = "topology"
POINT_GROUP = "point_matching"
EVIDENCE_GROUP = "evidence"
STABILITY_GROUP = "stability"
ENSEMBLE_GROUP = "ensemble"

PROFILES: dict[str, ComparisonProfile] = {
    "polygon": ComparisonProfile(
        "polygon",
        (PIXEL_GROUP, SOFT_GROUP, GEOMETRY_GROUP, COMPONENTS_GROUP, EVIDENCE_GROUP, STABILITY_GROUP),
        "Polygon and mask-oriented comparison.",
    ),
    "point": ComparisonProfile(
        "point",
        (PIXEL_GROUP, SOFT_GROUP, POINT_GROUP, COMPONENTS_GROUP, EVIDENCE_GROUP, STABILITY_GROUP),
        "Point object comparison.",
    ),
    "line_network": ComparisonProfile(
        "line_network",
        (PIXEL_GROUP, SOFT_GROUP, COMPONENTS_GROUP, SKELETON_GROUP, TOPOLOGY_GROUP, EVIDENCE_GROUP, STABILITY_GROUP),
        "Line-network comparison with skeleton and topology metrics.",
    ),
    "mixed": ComparisonProfile(
        "mixed",
        (
            PIXEL_GROUP,
            SOFT_GROUP,
            GEOMETRY_GROUP,
            COMPONENTS_GROUP,
            SKELETON_GROUP,
            TOPOLOGY_GROUP,
            POINT_GROUP,
            EVIDENCE_GROUP,
            STABILITY_GROUP,
            ENSEMBLE_GROUP,
        ),
        "Run every available comparison group.",
    ),
}


def resolve_profile(profile: ComparisonProfile | str | None, metadata: dict[str, object] | None = None) -> ComparisonProfile:
    if isinstance(profile, ComparisonProfile):
        return profile
    key = str(profile or "auto").strip().lower()
    if key == "auto":
        metadata = metadata or {}
        hinted = str(metadata.get("comparison_profile") or metadata.get("geometry_mode") or "").strip().lower()
        if hinted in PROFILES:
            return PROFILES[hinted]
        frame_type = str(metadata.get("frame_type") or "").strip().lower()
        if frame_type in {"line", "line_network", "skeleton"}:
            return PROFILES["line_network"]
        if frame_type == "point":
            return PROFILES["point"]
        return PROFILES["polygon"]
    return PROFILES.get(key, PROFILES["polygon"])
