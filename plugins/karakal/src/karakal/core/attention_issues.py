"""Typed attention markers for uncertain regions in model output / confidence maps."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .backend_constants import MODEL_RISK_UNCERTAINTY_THRESHOLD, POLYGON_SUPPORT_THRESHOLD
from .confidence_maps import build_model_uncertainty, normalize_algorithmic_confidence, normalize_unit_map
from .confidence_analysis import _internal_confidence_probability_map
from .mask_primitives import _binary_dilate, _boundary_mask, _label_components

ATTENTION_COMPUTE_LIGHTWEIGHT = "lightweight"
ATTENTION_COMPUTE_STANDARD = "standard"
ATTENTION_COMPUTE_MODES = (ATTENTION_COMPUTE_LIGHTWEIGHT, ATTENTION_COMPUTE_STANDARD)
DEFAULT_ATTENTION_COMPUTE_MODE = ATTENTION_COMPUTE_LIGHTWEIGHT

ATTENTION_ISSUE_BREAK = "break"
ATTENTION_ISSUE_MERGE = "merge"
ATTENTION_ISSUE_UNCERTAIN_BOUNDARY = "uncertain_boundary"
ATTENTION_ISSUE_UNCERTAIN_FILL = "uncertain_fill"
ATTENTION_ISSUE_ARTIFACT = "artifact"
ATTENTION_ISSUE_ORIGINAL_MISMATCH = "original_mismatch"

ATTENTION_ISSUE_TYPES = (
    ATTENTION_ISSUE_BREAK,
    ATTENTION_ISSUE_MERGE,
    ATTENTION_ISSUE_UNCERTAIN_BOUNDARY,
    ATTENTION_ISSUE_UNCERTAIN_FILL,
    ATTENTION_ISSUE_ARTIFACT,
    ATTENTION_ISSUE_ORIGINAL_MISMATCH,
)

_ISSUE_LABEL_KEYS = {
    ATTENTION_ISSUE_BREAK: "attention.issue.break",
    ATTENTION_ISSUE_MERGE: "attention.issue.merge",
    ATTENTION_ISSUE_UNCERTAIN_BOUNDARY: "attention.issue.uncertain_boundary",
    ATTENTION_ISSUE_UNCERTAIN_FILL: "attention.issue.uncertain_fill",
    ATTENTION_ISSUE_ARTIFACT: "attention.issue.artifact",
    ATTENTION_ISSUE_ORIGINAL_MISMATCH: "attention.issue.original_mismatch",
}


@dataclass(frozen=True, slots=True)
class AttentionIssue:
    issue_id: str
    issue_type: str
    severity: str
    score: float
    bbox: tuple[float, float, float, float]
    label_key: str
    detail: str = ""
    model_id: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, object] | object) -> AttentionIssue | None:
        if not isinstance(payload, dict):
            return None
        bbox_value = payload.get("bbox")
        if not isinstance(bbox_value, (tuple, list)) or len(bbox_value) < 4:
            return None
        try:
            bbox = (
                float(bbox_value[0]),
                float(bbox_value[1]),
                float(bbox_value[2]),
                float(bbox_value[3]),
            )
        except (TypeError, ValueError):
            return None
        if bbox[2] <= 0.0 or bbox[3] <= 0.0:
            return None
        issue_type = str(payload.get("issue_type") or ATTENTION_ISSUE_UNCERTAIN_FILL)
        if issue_type not in ATTENTION_ISSUE_TYPES:
            issue_type = ATTENTION_ISSUE_UNCERTAIN_FILL
        score = float(payload.get("score") or 0.0)
        return cls(
            issue_id=str(payload.get("issue_id") or ""),
            issue_type=issue_type,
            severity=str(payload.get("severity") or _severity(score)),
            score=float(np.clip(score, 0.0, 1.0)),
            bbox=bbox,
            label_key=str(payload.get("label_key") or _ISSUE_LABEL_KEYS[issue_type]),
            detail=str(payload.get("detail") or ""),
            model_id=str(payload.get("model_id") or ""),
        )


def normalize_attention_compute_mode(value: str | None) -> str:
    text = str(value or "").strip().lower()
    if text == ATTENTION_COMPUTE_STANDARD:
        return ATTENTION_COMPUTE_STANDARD
    return ATTENTION_COMPUTE_LIGHTWEIGHT


def attention_issues_to_payload(issues: tuple[AttentionIssue, ...] | list[AttentionIssue]) -> list[dict[str, object]]:
    return [issue.to_dict() for issue in issues]


def attention_issues_from_payload(payload: object) -> tuple[AttentionIssue, ...]:
    if not isinstance(payload, (list, tuple)):
        return ()
    issues: list[AttentionIssue] = []
    for item in payload:
        issue = AttentionIssue.from_dict(item)
        if issue is not None:
            issues.append(issue)
    return tuple(issues)


def build_attention_issues(
    values: np.ndarray | object,
    mask: np.ndarray | object | None = None,
    *,
    original: np.ndarray | object | None = None,
    source: str = "output",
    model_id: str = "",
    limit: int = 8,
    uncertainty_threshold: float = MODEL_RISK_UNCERTAINTY_THRESHOLD,
    compute_mode: str | None = None,
    probability_proxy: np.ndarray | object | None = None,
) -> tuple[AttentionIssue, ...]:
    """Build ranked typed markers for uncertain regions.

    ``source``:
    - ``output``: derive uncertainty from model output probabilities / grayscale.
    - ``confidence``: treat values as a confidence map (white = confident).

    ``compute_mode``:
    - ``lightweight``: boundary/fill/artifact only (no merge/break/original mismatch).
    - ``standard``: full classifier including topology and original mismatch.
    """

    mode = normalize_attention_compute_mode(compute_mode)
    map_values = normalize_unit_map(values)
    if map_values.size == 0:
        return ()
    mask_bool = (
        np.asarray(mask, dtype=bool)
        if mask is not None
        else np.asarray(map_values >= 0.5, dtype=bool)
    )
    if mask_bool.shape != map_values.shape:
        mask_bool = np.asarray(map_values >= 0.5, dtype=bool)

    if str(source) == "confidence":
        uncertainty = build_model_uncertainty(map_values)
        support = np.ones_like(uncertainty, dtype=bool)
    else:
        # Binary masks (0/1) have zero algorithmic uncertainty; reuse the same
        # distance-transform proxy that powers "output by itself" frame scores.
        if probability_proxy is not None:
            probability = normalize_unit_map(probability_proxy)
            if probability.shape != map_values.shape:
                probability = _internal_confidence_probability_map(
                    map_values,
                    support_mask=mask_bool,
                    allow_binary_proxy=True,
                )
        else:
            probability = _internal_confidence_probability_map(
                map_values,
                support_mask=mask_bool,
                allow_binary_proxy=True,
            )
        confidence = normalize_algorithmic_confidence(probability)
        uncertainty = np.clip(1.0 - confidence, 0.0, 1.0).astype(np.float32, copy=False)
        support = mask_bool | (probability >= float(POLYGON_SUPPORT_THRESHOLD))
    if not np.any(support):
        support = np.ones_like(uncertainty, dtype=bool)

    uncertain = support & np.isfinite(uncertainty) & (uncertainty > float(uncertainty_threshold))
    if not np.any(uncertain):
        return ()

    labels, count = _label_components(uncertain)
    if count <= 0:
        return ()
    areas = np.bincount(labels.ravel(), minlength=count + 1)
    frame_area = max(1, int(uncertainty.size))
    object_area = max(1, int(np.count_nonzero(mask_bool)))
    boundary = _boundary_mask(mask_bool) if np.any(mask_bool) else np.zeros_like(mask_bool, dtype=bool)
    boundary_band = np.asarray(_binary_dilate(boundary, radius=2), dtype=bool) if np.any(boundary) else boundary
    interior = mask_bool & ~boundary_band

    original_array = None if original is None else np.asarray(original)
    original_edge_support = None
    if mode == ATTENTION_COMPUTE_STANDARD and original_array is not None:
        original_edge_support = _original_edge_support(original_array)

    ranked: list[tuple[float, int, float]] = []
    for label_id in range(1, count + 1):
        area = int(areas[label_id]) if label_id < areas.size else 0
        if area <= 0:
            continue
        region = labels == int(label_id)
        mean_u = float(np.mean(uncertainty[region], dtype=np.float64))
        score = float(np.clip((area / frame_area) * 4.0 + mean_u * 0.65, 0.0, 1.0))
        ranked.append((score, label_id, mean_u))
    ranked.sort(key=lambda item: item[0], reverse=True)

    issues: list[AttentionIssue] = []
    for index, (score, label_id, mean_u) in enumerate(ranked[: max(1, int(limit))]):
        region = labels == int(label_id)
        bbox = _bbox_for_label(labels, label_id)
        if bbox is None:
            continue
        issue_type = _classify_region(
            region,
            mask_bool=mask_bool,
            boundary_band=boundary_band,
            interior=interior,
            uncertainty=uncertainty,
            area=int(areas[label_id]),
            frame_area=frame_area,
            object_area=object_area,
            original=original_array,
            original_edge_support=original_edge_support,
            compute_mode=mode,
        )
        issues.append(
            AttentionIssue(
                issue_id=f"{model_id or 'model'}:{issue_type}:{index + 1}",
                issue_type=issue_type,
                severity=_severity(score),
                score=float(score),
                bbox=bbox,
                label_key=_ISSUE_LABEL_KEYS[issue_type],
                detail=f"mean_uncertainty={mean_u:.3f}",
                model_id=str(model_id or ""),
            )
        )
    return tuple(issues)


def _severity(value: float) -> str:
    if value < 0.35:
        return "low"
    if value < 0.55:
        return "medium"
    if value < 0.75:
        return "high"
    return "critical"


def _bbox_for_label(labels: np.ndarray, label_id: int) -> tuple[float, float, float, float] | None:
    yy, xx = np.nonzero(labels == int(label_id))
    if yy.size == 0:
        return None
    height, width = labels.shape
    x0, x1 = int(np.min(xx)), int(np.max(xx)) + 1
    y0, y1 = int(np.min(yy)), int(np.max(yy)) + 1
    return (
        x0 / max(1, width),
        y0 / max(1, height),
        (x1 - x0) / max(1, width),
        (y1 - y0) / max(1, height),
    )


def _classify_region(
    region: np.ndarray,
    *,
    mask_bool: np.ndarray,
    boundary_band: np.ndarray,
    interior: np.ndarray,
    uncertainty: np.ndarray,
    area: int,
    frame_area: int,
    object_area: int,
    original: np.ndarray | None,
    original_edge_support: np.ndarray | None = None,
    compute_mode: str = ATTENTION_COMPUTE_LIGHTWEIGHT,
) -> str:
    region_area_frac = float(area / max(1, frame_area))
    if region_area_frac < 0.0015 or area < max(8, int(0.002 * object_area)):
        return ATTENTION_ISSUE_ARTIFACT

    region_pixels = max(1, int(np.count_nonzero(region)))
    boundary_overlap = float(np.count_nonzero(region & boundary_band) / region_pixels)
    interior_overlap = float(np.count_nonzero(region & interior) / region_pixels)

    if compute_mode == ATTENTION_COMPUTE_STANDARD:
        edge_support = original_edge_support
        if edge_support is None and original is not None:
            edge_support = _original_edge_support(original)
        if edge_support is not None and np.any(region & boundary_band):
            unsupported = (region & boundary_band) & ~edge_support
            mismatch = float(
                np.count_nonzero(unsupported) / max(1, np.count_nonzero(region & boundary_band))
            )
            if mismatch >= 0.55:
                return ATTENTION_ISSUE_ORIGINAL_MISMATCH
        if _looks_like_merge(region, mask_bool):
            return ATTENTION_ISSUE_MERGE
        if _looks_like_break(region, mask_bool):
            return ATTENTION_ISSUE_BREAK

    if boundary_overlap >= 0.45 and boundary_overlap >= interior_overlap:
        return ATTENTION_ISSUE_UNCERTAIN_BOUNDARY
    if interior_overlap >= 0.35:
        return ATTENTION_ISSUE_UNCERTAIN_FILL
    mean_u = float(np.mean(uncertainty[region], dtype=np.float64))
    if mean_u >= 0.55 and boundary_overlap >= 0.25:
        return ATTENTION_ISSUE_UNCERTAIN_BOUNDARY
    return ATTENTION_ISSUE_UNCERTAIN_FILL


def _looks_like_merge(region: np.ndarray, mask_bool: np.ndarray) -> bool:
    if not np.any(mask_bool):
        return False
    # Thin uncertain bridges often sit between two object lobes: dilating the
    # bridge and removing it should split the local object into more components.
    bridge = np.asarray(region, dtype=bool)
    local = np.asarray(_binary_dilate(bridge, radius=3), dtype=bool)
    local_object = mask_bool & local
    if not np.any(local_object):
        return False
    before_labels, before_count = _label_components(local_object)
    if before_count <= 0:
        return False
    without_bridge = local_object & ~np.asarray(_binary_dilate(bridge, radius=1), dtype=bool)
    _after_labels, after_count = _label_components(without_bridge)
    del before_labels
    yy, xx = np.nonzero(bridge)
    if yy.size == 0:
        return False
    height = int(np.max(yy) - np.min(yy) + 1)
    width = int(np.max(xx) - np.min(xx) + 1)
    thin = min(height, width) <= max(3, int(0.35 * max(height, width)))
    return bool(thin and after_count > before_count)


def _looks_like_break(region: np.ndarray, mask_bool: np.ndarray) -> bool:
    if not np.any(mask_bool):
        return False
    # Uncertain gaps that separate object parts: filling the gap should reduce
    # the number of local components.
    gap = np.asarray(region, dtype=bool)
    local = np.asarray(_binary_dilate(gap, radius=3), dtype=bool)
    local_object = mask_bool & local
    if not np.any(local_object):
        return False
    _before_labels, before_count = _label_components(local_object)
    filled = local_object | gap
    _after_labels, after_count = _label_components(filled)
    holes_before = _interior_hole_count(local_object)
    holes_after = _interior_hole_count(filled)
    return bool(before_count > after_count or holes_before > holes_after)


def _interior_hole_count(mask: np.ndarray) -> int:
    inverse_labels, count = _label_components(~np.asarray(mask, dtype=bool))
    if count <= 0:
        return 0
    border_ids = set(np.unique(inverse_labels[0, :]).tolist())
    border_ids.update(np.unique(inverse_labels[-1, :]).tolist())
    border_ids.update(np.unique(inverse_labels[:, 0]).tolist())
    border_ids.update(np.unique(inverse_labels[:, -1]).tolist())
    return sum(1 for label_id in range(1, count + 1) if label_id not in border_ids)


def _original_edge_support(original: np.ndarray) -> np.ndarray:
    values = np.asarray(original, dtype=np.float32)
    if values.size == 0:
        return np.zeros_like(values, dtype=bool)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=bool)
    low, high = (float(item) for item in np.percentile(finite, (5.0, 95.0)))
    normalized = np.zeros_like(values, dtype=np.float32)
    if high > low + 1e-6:
        normalized = np.clip((values - low) / (high - low), 0.0, 1.0)
    grad_x = np.zeros_like(normalized)
    grad_y = np.zeros_like(normalized)
    grad_x[:, :-1] = np.abs(normalized[:, 1:] - normalized[:, :-1])
    grad_y[:-1, :] = np.abs(normalized[1:, :] - normalized[:-1, :])
    gradient = np.hypot(grad_x, grad_y).astype(np.float32)
    positive = gradient[gradient > 0.0]
    if positive.size:
        scale = max(float(np.percentile(positive, 95.0)), 1e-6)
        gradient = np.clip(gradient / scale, 0.0, 1.0)
    return np.asarray(_binary_dilate(gradient >= 0.35, radius=2), dtype=bool)


def _original_mismatch_score(boundary_region: np.ndarray, original: np.ndarray) -> float:
    values = np.asarray(original, dtype=np.float32)
    if values.shape != boundary_region.shape:
        return 0.0
    if not np.any(boundary_region):
        return 0.0
    support = _original_edge_support(values)
    if support.shape != boundary_region.shape:
        return 0.0
    unsupported = boundary_region & ~support
    return float(np.count_nonzero(unsupported) / max(1, np.count_nonzero(boundary_region)))


__all__ = [
    "ATTENTION_COMPUTE_LIGHTWEIGHT",
    "ATTENTION_COMPUTE_MODES",
    "ATTENTION_COMPUTE_STANDARD",
    "ATTENTION_ISSUE_ARTIFACT",
    "ATTENTION_ISSUE_BREAK",
    "ATTENTION_ISSUE_MERGE",
    "ATTENTION_ISSUE_ORIGINAL_MISMATCH",
    "ATTENTION_ISSUE_TYPES",
    "ATTENTION_ISSUE_UNCERTAIN_BOUNDARY",
    "ATTENTION_ISSUE_UNCERTAIN_FILL",
    "AttentionIssue",
    "DEFAULT_ATTENTION_COMPUTE_MODE",
    "attention_issues_from_payload",
    "attention_issues_to_payload",
    "build_attention_issues",
    "normalize_attention_compute_mode",
]
