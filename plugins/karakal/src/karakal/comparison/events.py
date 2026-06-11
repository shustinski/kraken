"""Event builders for local comparison differences."""
from __future__ import annotations

import numpy as np

from .components import Component, component_overlap_matrix, label_components
from .models import ComparisonEvent

try:
    from scipy import ndimage as ndi
except Exception:  # pragma: no cover
    ndi = None

LARGE_FALLBACK_LABEL_PIXEL_LIMIT = 262_144


def component_events(mask_a: np.ndarray, mask_b: np.ndarray, *, frame_id: str, model_ids: tuple[str, str], connectivity: int = 8) -> tuple[ComparisonEvent, ...]:
    labels_a, comps_a = label_components(mask_a, connectivity=connectivity)
    labels_b, comps_b = label_components(mask_b, connectivity=connectivity)
    return component_events_from_labels(labels_a, comps_a, labels_b, comps_b, frame_id=frame_id, model_ids=model_ids)


def component_events_from_labels(
    labels_a: np.ndarray,
    comps_a: tuple[Component, ...],
    labels_b: np.ndarray,
    comps_b: tuple[Component, ...],
    *,
    frame_id: str,
    model_ids: tuple[str, str],
) -> tuple[ComparisonEvent, ...]:
    overlaps = component_overlap_matrix(labels_a, len(comps_a), labels_b, len(comps_b))
    events: list[ComparisonEvent] = []
    for index_a, component in enumerate(comps_a):
        matched_b = np.nonzero(overlaps[index_a] > 0)[0].tolist() if overlaps.size else []
        if len(matched_b) >= 2:
            events.append(_component_event(frame_id, "COMPONENT_SPLIT", component.bbox, model_ids, [component.component_id], [item + 1 for item in matched_b]))
        if len(matched_b) == 0:
            events.append(_component_event(frame_id, "A_ONLY_COMPONENT", component.bbox, model_ids, [component.component_id], []))
    if overlaps.size:
        for index_b, component in enumerate(comps_b):
            matched_a = np.nonzero(overlaps[:, index_b] > 0)[0].tolist()
            if len(matched_a) >= 2:
                events.append(_component_event(frame_id, "COMPONENT_MERGE", component.bbox, model_ids, [item + 1 for item in matched_a], [component.component_id]))
            if len(matched_a) == 0:
                events.append(_component_event(frame_id, "B_ONLY_COMPONENT", component.bbox, model_ids, [], [component.component_id]))
    elif comps_b:
        for component in comps_b:
            events.append(_component_event(frame_id, "B_ONLY_COMPONENT", component.bbox, model_ids, [], [component.component_id]))
    return tuple(events)


def xor_hotspot_events(mask: np.ndarray, *, frame_id: str, model_ids: tuple[str, ...], connectivity: int = 8, min_area: int = 4) -> tuple[ComparisonEvent, ...]:
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2 or binary.size == 0 or not np.any(binary):
        return ()
    if ndi is None and binary.size > LARGE_FALLBACK_LABEL_PIXEL_LIMIT:
        bbox = _bbox_from_mask(binary)
        area = int(np.count_nonzero(binary))
        total = max(1, int(binary.size))
        return (
            ComparisonEvent(
                event_id=f"{frame_id}:DISAGREEMENT_HOTSPOT:all",
                event_type="DISAGREEMENT_HOTSPOT",
                risk=float(np.clip(area / max(1.0, total * 0.02), 0.10, 1.0)),
                bbox=bbox,
                point=(bbox[0] + bbox[2] / 2.0, bbox[1] + bbox[3] / 2.0) if bbox[2] > 0 and bbox[3] > 0 else None,
                object_ids=["all"],
                model_ids=list(model_ids),
                description="Local disagreement hotspot",
                recommended_layers=["mask_xor", "mask_a_only", "mask_b_only"],
                metrics={"area": area, "area_fraction": float(area / total)},
            ),
        )
    _labels, components = label_components(binary, connectivity=connectivity)
    events = []
    total = max(1, int(np.asarray(mask, dtype=bool).size))
    for component in components:
        if component.area < int(min_area):
            continue
        risk = float(np.clip(component.area / max(1.0, total * 0.02), 0.10, 1.0))
        events.append(
            ComparisonEvent(
                event_id=f"{frame_id}:DISAGREEMENT_HOTSPOT:{component.component_id}",
                event_type="DISAGREEMENT_HOTSPOT",
                risk=risk,
                bbox=component.bbox,
                point=component.centroid,
                object_ids=[str(component.component_id)],
                model_ids=list(model_ids),
                description="Local disagreement hotspot",
                recommended_layers=["mask_xor", "mask_a_only", "mask_b_only"],
                metrics={"area": int(component.area), "area_fraction": float(component.area / total)},
            )
        )
    return tuple(events)


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(np.asarray(mask, dtype=bool))
    if ys.size <= 0:
        return (0, 0, 0, 0)
    x0 = int(xs.min())
    y0 = int(ys.min())
    return (x0, y0, int(xs.max()) + 1 - x0, int(ys.max()) + 1 - y0)


def _component_event(
    frame_id: str,
    event_type: str,
    bbox: tuple[int, int, int, int],
    model_ids: tuple[str, str],
    object_ids_a: list[int],
    object_ids_b: list[int],
) -> ComparisonEvent:
    count_delta = abs(len(object_ids_a) - len(object_ids_b))
    risk = float(np.clip(0.35 + 0.15 * count_delta, 0.35, 1.0))
    return ComparisonEvent(
        event_id=f"{frame_id}:{event_type}:{bbox[0]}:{bbox[1]}",
        event_type=event_type,
        risk=risk,
        bbox=bbox,
        point=(bbox[0] + bbox[2] / 2.0, bbox[1] + bbox[3] / 2.0) if bbox[2] > 0 and bbox[3] > 0 else None,
        object_ids=[*(f"A:{value}" for value in object_ids_a), *(f"B:{value}" for value in object_ids_b)],
        model_ids=list(model_ids),
        description=event_type.replace("_", " ").title(),
        recommended_layers=["mask_xor", "components"],
        metrics={"a_count": len(object_ids_a), "b_count": len(object_ids_b)},
    )
