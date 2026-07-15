"""Topology metrics and local topology events."""
from __future__ import annotations

import numpy as np

from .components import label_components
from .models import ComparisonEvent
from .skeleton import skeleton_stats

try:
    from scipy import ndimage as ndi
except Exception:  # pragma: no cover
    ndi = None


def topology_metrics(mask: np.ndarray, *, connectivity: int = 8) -> dict[str, float]:
    binary = np.asarray(mask, dtype=bool)
    _labels, components = label_components(binary, connectivity=connectivity)
    beta0 = len(components)
    holes = _hole_count(binary, connectivity=connectivity)
    stats = skeleton_stats(binary)
    return {
        "beta0": float(beta0),
        "beta1": float(holes),
        "euler": float(beta0 - holes),
        "endpoint_count": float(stats.endpoint_count),
        "junction_count": float(stats.junction_count),
        "branch_count": float(stats.branch_count),
        "skeleton_length": float(stats.length),
    }


def topology_delta_events(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    *,
    frame_id: str,
    model_ids: tuple[str, str],
    connectivity: int = 8,
) -> tuple[ComparisonEvent, ...]:
    metrics_a = topology_metrics(mask_a, connectivity=connectivity)
    metrics_b = topology_metrics(mask_b, connectivity=connectivity)
    xor = np.logical_xor(np.asarray(mask_a, dtype=bool), np.asarray(mask_b, dtype=bool))
    bbox = _bbox_from_mask(xor)
    events: list[ComparisonEvent] = []
    if metrics_b["beta0"] > metrics_a["beta0"]:
        events.append(_event(frame_id, "PROBABLE_BREAK", metrics_b["beta0"] - metrics_a["beta0"], bbox, model_ids))
    if metrics_b["beta0"] < metrics_a["beta0"] or metrics_b["beta1"] > metrics_a["beta1"]:
        events.append(_event(frame_id, "PROBABLE_FALSE_BRIDGE", abs(metrics_b["beta0"] - metrics_a["beta0"]) + abs(metrics_b["beta1"] - metrics_a["beta1"]), bbox, model_ids))
    if metrics_b["endpoint_count"] > metrics_a["endpoint_count"] + 1:
        events.append(_event(frame_id, "PROBABLE_SPUR", metrics_b["endpoint_count"] - metrics_a["endpoint_count"], bbox, model_ids))
    return tuple(events)


def _hole_count(mask: np.ndarray, *, connectivity: int) -> int:
    if mask.ndim != 2 or mask.size == 0 or not np.any(mask):
        return 0
    if ndi is None or not hasattr(ndi, "binary_fill_holes"):
        return 0
    filled = ndi.binary_fill_holes(mask)
    holes = np.asarray(filled & ~mask, dtype=bool)
    _labels, components = label_components(holes, connectivity=connectivity)
    return len(components)


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(np.asarray(mask, dtype=bool))
    if ys.size <= 0:
        return (0, 0, 0, 0)
    x0 = int(xs.min())
    y0 = int(ys.min())
    return (x0, y0, int(xs.max()) + 1 - x0, int(ys.max()) + 1 - y0)


def _event(frame_id: str, event_type: str, delta: float, bbox: tuple[int, int, int, int], model_ids: tuple[str, str]) -> ComparisonEvent:
    risk = float(np.clip(abs(float(delta)) / 4.0, 0.25, 1.0))
    return ComparisonEvent(
        event_id=f"{frame_id}:{event_type}:{bbox[0]}:{bbox[1]}",
        event_type=event_type,
        risk=risk,
        bbox=bbox,
        point=(bbox[0] + bbox[2] / 2.0, bbox[1] + bbox[3] / 2.0) if bbox[2] > 0 and bbox[3] > 0 else None,
        object_ids=[],
        model_ids=list(model_ids),
        description=event_type.replace("_", " ").title(),
        recommended_layers=["mask_xor", "skeleton_xor"],
        metrics={"delta": float(delta)},
    )
