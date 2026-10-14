"""_core: filter vias by conductor."""
from __future__ import annotations
from dataclasses import replace
import cv2
import numpy as np
from contour.domain import PolygonData
from contour.utils import ensure_binary_mask
from contour.application.processing import (
    VIA_SEARCH_MODE_TEMPLATE,
    ContourDebugCandidate,
    ContourExtractionSettings,
    normalize_via_search_mode,
)

def _debug_candidates_from_via_candidates(
    candidates: list[_ViaCandidate],
    *,
    accepted: bool,
    reason: str | None = None,
) -> list[ContourDebugCandidate]:
    debug: list[ContourDebugCandidate] = []
    for index, candidate in enumerate(candidates):
        bbox = _candidate_bbox(candidate)
        debug.append(
            ContourDebugCandidate(
                contour_index=index,
                bbox=bbox,
                area=float(candidate.width * candidate.height),
                perimeter=float(2.0 * (candidate.width + candidate.height)),
                roundness=float(candidate.roundness),
                accepted=bool(accepted),
                reason=reason
                or candidate.reason
                or (f"accepted:{candidate.source}" if accepted else f"rejected:{candidate.source}"),
                source=str(candidate.source),
                score=float(candidate.score),
            )
        )
    return debug

def _candidate_bbox(candidate: _ViaCandidate) -> tuple[int, int, int, int]:
    left = round(candidate.center_x - candidate.width / 2.0)
    top = round(candidate.center_y - candidate.height / 2.0)
    return left, top, max(1, round(candidate.width)), max(1, round(candidate.height))

def _should_run_sem_dual_branch(settings: ContourExtractionSettings) -> bool:
    """Return ``True`` when legacy conductor flow should also merge template vias.

    The SEM vision backend uses a separate entry point; dual-branch merging only
    applies on the legacy path so template vias can be combined with conductor masks.
    """

    if str(getattr(settings, "algorithm_backend", "legacy")).lower() == "sem":
        return False
    if settings.object_type == "via" or settings.output_mode == "box":
        return False
    if settings.extraction_profile != "conductors":
        return False
    if normalize_via_search_mode(settings.via_search_mode) != VIA_SEARCH_MODE_TEMPLATE:
        return False
    return bool(settings.via_template_images)

def _build_via_settings_for_dual_branch(settings: ContourExtractionSettings) -> ContourExtractionSettings:
    return replace(
        settings,
        extraction_profile="vias",
        object_type="via",
        output_mode="box",
    )

def _component_map(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    binary = ensure_binary_mask(mask)
    if cv2.countNonZero(binary) == 0:
        return None
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats((binary > 0).astype(np.uint8), connectivity=8)
    if count <= 1:
        return None
    return labels, stats

def _filter_vias_by_conductor_linearity(vias: list[PolygonData], conductor_mask: np.ndarray) -> list[PolygonData]:
    """Reject likely false vias sitting on long, straight conductor segments."""

    component_payload = _component_map(conductor_mask)
    if component_payload is None:
        return vias
    labels, stats = component_payload
    accepted: list[PolygonData] = []
    for polygon in vias:
        x_coord, y_coord, width, height = polygon.bbox
        center_x = round(x_coord + width / 2.0)
        center_y = round(y_coord + height / 2.0)
        if center_x < 0 or center_y < 0 or center_y >= labels.shape[0] or center_x >= labels.shape[1]:
            accepted.append(polygon)
            continue
        label = int(labels[center_y, center_x])
        if label <= 0 or label >= stats.shape[0]:
            accepted.append(polygon)
            continue
        comp_width = max(1, int(stats[label, cv2.CC_STAT_WIDTH]))
        comp_height = max(1, int(stats[label, cv2.CC_STAT_HEIGHT]))
        comp_area = max(1, int(stats[label, cv2.CC_STAT_AREA]))
        elongation = float(max(comp_width, comp_height)) / float(min(comp_width, comp_height))
        via_area = max(1.0, float(width * height))
        comp_to_via_area = float(comp_area) / via_area
        # If the candidate center lies on a very elongated component and that
        # component is much larger than via size, this is usually a trace tip /
        # edge artifact rather than a real contact.
        if elongation > 7.0 and comp_to_via_area > 16.0:
            continue
        accepted.append(polygon)
    return accepted

def _merge_dual_branch_polygons(
    conductor_polygons: list[PolygonData], via_polygons: list[PolygonData]
) -> list[PolygonData]:
    merged: list[PolygonData] = []
    for polygon in conductor_polygons:
        merged.append(polygon.clone())
    next_id = max((polygon.id for polygon in merged), default=0) + 1
    for polygon in via_polygons:
        clone = polygon.clone()
        clone.id = next_id
        clone.parent_id = None
        merged.append(clone)
        next_id += 1
    return merged

def _build_metalization_mask(gray: np.ndarray, settings: ContourExtractionSettings) -> np.ndarray:
    if gray.size == 0:
        return np.zeros_like(gray, dtype=np.uint8)
    from .....vision.metal_recovery import metal_recovery_config_from_settings
    from .....vision.metal_recovery.detector import build_metal_extraction_mask

    cfg = metal_recovery_config_from_settings(settings)
    mask, _dbg = build_metal_extraction_mask(gray, cfg)
    return mask
