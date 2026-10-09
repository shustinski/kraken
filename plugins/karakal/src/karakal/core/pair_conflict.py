"""Conflict of two networks that paint mutually exclusive cell classes («ones» and «zeros»).

One network paints the ones cells of a frame, the other its zero cells. A place both paint is an
error. Every overlap blob is measured against the two cells it lies in: its share is the blob area
over the smaller of those cells. Two neighbours touching by their edges give a thin blob with a tiny
share; one cell painted by both networks gives a share near 1.

The run keeps every blob down to a small floor, with its area and share. The operator's minimum
area and share only hide blobs afterwards (:func:`filter_class_conflicts`), so changing them needs
no new run. Both masks are binarised by the same policy as the cell analysis of a layer.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from . import grid_anomaly
from .cache_utils import atomic_pickle_dump
from .grid_anomaly import GridCellAnalysisResult, GridDamageSeverityThresholds, GridFrameAnalysisResult, _damage_score
from .mask_normalization import binarize_mask

try:
    import cv2
except Exception:  # pragma: no cover - OpenCV is optional at runtime
    cv2 = None

_LOGGER = logging.getLogger(__name__)

CONFLICT_REASON = "class_conflict"
# Result layer of a pair in the worker output and in the stored payloads.
PAIR_CONFLICT_LAYER = "conflict"
CONFLICT_ALGORITHM_VERSION = "pair_class_conflict.v1"
# Display defaults: a conflict is at least this many pixels and this share of the smaller cell.
CONFLICT_MIN_AREA_PX = 6
CONFLICT_MIN_CELL_SHARE = 0.10
# The run keeps blobs down to this size; the display thresholds never go below it.
CONFLICT_ANALYSIS_FLOOR_PX = 2
CONFLICT_AREA_FEATURE = "conflict_area_px"
CONFLICT_SHARE_FEATURE = "conflict_cell_share"
# Boxes of the two cells the blob lies in (x, y, width, height of the ones and the zeros cell).
CONFLICT_CELL_BOX_FEATURES = (
    ("ones_cell_x", "ones_cell_y", "ones_cell_w", "ones_cell_h"),
    ("zeros_cell_x", "zeros_cell_y", "zeros_cell_w", "zeros_cell_h"),
)

_PAIR_KEY_PREFIX = "pair::"


def pair_key(ones_model_id: str, zeros_model_id: str) -> str:
    """Matrix selection key of a ones/zeros pair."""

    return f"{_PAIR_KEY_PREFIX}{ones_model_id}::{zeros_model_id}"


def parse_pair_key(value: object) -> tuple[str, str] | None:
    """(ones model id, zeros model id) of a pair key, None for anything else."""

    text = str(value or "")
    if not text.startswith(_PAIR_KEY_PREFIX):
        return None
    parts = text[len(_PAIR_KEY_PREFIX) :].split("::")
    if len(parts) != 2 or not all(parts):
        return None
    return parts[0], parts[1]


@dataclass(frozen=True, slots=True)
class ConflictThresholds:
    min_area_px: int = CONFLICT_MIN_AREA_PX
    min_cell_share: float = CONFLICT_MIN_CELL_SHARE

    def normalized(self) -> "ConflictThresholds":
        return ConflictThresholds(
            min_area_px=max(CONFLICT_ANALYSIS_FLOOR_PX, int(self.min_area_px)),
            min_cell_share=float(np.clip(float(self.min_cell_share), 0.0, 1.0)),
        )


def _empty_result(frame_id: str, frame_path: str, width: int, height: int, cells: int = 0) -> GridFrameAnalysisResult:
    return GridFrameAnalysisResult(
        frame_id=str(frame_id or ""),
        frame_path=str(frame_path or ""),
        image_width=int(width),
        image_height=int(height),
        grid_rows=0,
        grid_cols=0,
        total_expected_cells=int(cells),
        detected_cells=int(cells),
        normal_cells=int(cells),
        suspicious_cells=0,
        broken_cells=0,
        missing_cells=0,
        artifact_cells=0,
        damage_score=0.0,
        severity_level="OK",
        grid_detected=True,
        per_cell_results=(),
        component_count=int(cells),
    )


def _is_contour_jitter(
    conflict_roi: np.ndarray,
    union_boundary_roi: np.ndarray,
    *,
    area: int,
    width: int,
    height: int,
) -> bool:
    """A hairline fringe along the edge of the two masks is contour jitter, not a class clash."""

    if area <= 0 or width <= 0 or height <= 0:
        return True
    min_side = min(int(width), int(height))
    max_side = max(int(width), int(height))
    # Strips 1–2 px thick are jitter even when they fill their box.
    if min_side <= 2 and max_side >= 6:
        return True
    if min_side <= 3 and max_side >= 8 and float(area) <= 0.45 * float(width * height):
        return True
    if min_side <= 2 and area <= 24:
        return True
    boundary = np.asarray(union_boundary_roi, dtype=bool)
    conflict = np.asarray(conflict_roi, dtype=bool)
    if not np.any(conflict):
        return True
    if np.any(boundary):
        overlap = float(np.count_nonzero(conflict & boundary)) / float(max(1, area))
        if overlap >= 0.62 and min_side <= 5:
            return True
        if overlap >= 0.78 and area <= 40:
            return True
    return False


def _owner_cell(labels: np.ndarray, stats: np.ndarray, component: np.ndarray, x: int, y: int) -> tuple[int, tuple[int, int, int, int]]:
    """Area and box of the cell holding most of the blob pixels."""

    height, width = component.shape
    owners = labels[y : y + height, x : x + width][component]
    owners = owners[owners > 0]
    if owners.size == 0:
        return 0, (x, y, width, height)
    index = int(np.bincount(owners).argmax())
    return (
        int(stats[index, cv2.CC_STAT_AREA]),
        (
            int(stats[index, cv2.CC_STAT_LEFT]),
            int(stats[index, cv2.CC_STAT_TOP]),
            int(stats[index, cv2.CC_STAT_WIDTH]),
            int(stats[index, cv2.CC_STAT_HEIGHT]),
        ),
    )


def _blob_outline(component: np.ndarray, x: int, y: int) -> tuple[tuple[int, int], ...]:
    contours, _hierarchy = cv2.findContours(
        np.asarray(component, dtype=np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return ()
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
    return tuple((int(px) + x, int(py) + y) for px, py in contour)


def analyze_class_conflict_masks(
    ones_mask: np.ndarray,
    zeros_mask: np.ndarray,
    *,
    frame_id: str = "",
    frame_path: str = "",
) -> GridFrameAnalysisResult:
    """Every blob painted by both networks, with its area and its share of the smaller cell."""

    ones = np.asarray(ones_mask, dtype=bool)
    zeros = np.asarray(zeros_mask, dtype=bool)
    if ones.shape != zeros.shape:
        raise ValueError(f"Conflict masks have different shapes: {ones.shape} vs {zeros.shape}")
    height, width = ones.shape
    conflict = ones & zeros
    if cv2 is None or not np.any(conflict):
        return _empty_result(frame_id, frame_path, width, height)

    ones_count, ones_labels, ones_stats, _ = cv2.connectedComponentsWithStats(ones.astype(np.uint8), connectivity=8)
    zeros_count, zeros_labels, zeros_stats, _ = cv2.connectedComponentsWithStats(zeros.astype(np.uint8), connectivity=8)
    cell_count = max(0, int(ones_count) - 1) + max(0, int(zeros_count) - 1)

    union = (ones | zeros).astype(np.uint8)
    union_boundary = union.astype(bool) & ~cv2.erode(union, np.ones((3, 3), dtype=np.uint8), iterations=1).astype(bool)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(conflict.astype(np.uint8), connectivity=8)
    findings: list[GridCellAnalysisResult] = []
    for index in range(1, int(count)):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < CONFLICT_ANALYSIS_FLOOR_PX:
            continue
        x = int(stats[index, cv2.CC_STAT_LEFT])
        y = int(stats[index, cv2.CC_STAT_TOP])
        bw = int(stats[index, cv2.CC_STAT_WIDTH])
        bh = int(stats[index, cv2.CC_STAT_HEIGHT])
        component = labels[y : y + bh, x : x + bw] == index
        if _is_contour_jitter(component, union_boundary[y : y + bh, x : x + bw], area=area, width=bw, height=bh):
            continue
        ones_area, ones_box = _owner_cell(ones_labels, ones_stats, component, x, y)
        zeros_area, zeros_box = _owner_cell(zeros_labels, zeros_stats, component, x, y)
        share = float(area) / float(max(1, min(ones_area or area, zeros_area or area)))
        features = [(CONFLICT_AREA_FEATURE, float(area)), (CONFLICT_SHARE_FEATURE, float(min(1.0, share)))]
        for names, box in zip(CONFLICT_CELL_BOX_FEATURES, (ones_box, zeros_box)):
            features.extend((name, float(value)) for name, value in zip(names, box))
        findings.append(
            GridCellAnalysisResult(
                row=len(findings),
                col=0,
                bbox=(x, y, bw, bh),
                centroid=(float(centroids[index][0]), float(centroids[index][1])),
                contour_id=int(index),
                status="broken",
                score=float(min(1.0, 0.7 + 0.3 * min(1.0, share))),
                reasons=(CONFLICT_REASON,),
                feature_snapshot=tuple(features),
                outline=_blob_outline(component, x, y),
            )
        )
    if not findings:
        return _empty_result(frame_id, frame_path, width, height, cell_count)
    return _recount(_empty_result(frame_id, frame_path, width, height, cell_count), findings)


def _recount(result: GridFrameAnalysisResult, cells: list[GridCellAnalysisResult]) -> GridFrameAnalysisResult:
    total = max(1, int(result.component_count or 0), len(cells))
    damage = _damage_score(cells, total) if cells else 0.0
    return replace(
        result,
        per_cell_results=tuple(cells),
        total_expected_cells=total,
        detected_cells=total,
        normal_cells=total - len(cells),
        suspicious_cells=0,
        broken_cells=len(cells),
        damage_score=float(damage),
        severity_level=GridDamageSeverityThresholds().level_for_score(damage),
    )


def conflict_features(cell: object) -> dict[str, float]:
    return {str(key): float(value) for key, value in (getattr(cell, "feature_snapshot", ()) or ())}


def filter_class_conflicts(
    result: GridFrameAnalysisResult | None,
    thresholds: ConflictThresholds,
    *,
    label: str = "",
) -> GridFrameAnalysisResult | None:
    """Keep the conflicts at or above the thresholds and recount the frame score.

    ``label`` names the other network on every kept conflict (shown in lists and tooltips).
    """

    if not isinstance(result, GridFrameAnalysisResult):
        return result
    limits = thresholds.normalized()
    kept: list[GridCellAnalysisResult] = []
    for cell in result.per_cell_results or ():
        features = conflict_features(cell)
        area = float(features.get(CONFLICT_AREA_FEATURE, 0.0))
        share = float(features.get(CONFLICT_SHARE_FEATURE, 0.0))
        if area < limits.min_area_px or share < limits.min_cell_share:
            continue
        kept.append(replace(cell, feature_cluster_label=label) if label else cell)
    if not label and len(kept) == len(result.per_cell_results or ()):
        return result
    return _recount(result, kept)


def _cache_key(path_a: Path, path_b: Path, frame_id: str) -> tuple[object, ...]:
    return (
        CONFLICT_ALGORITHM_VERSION,
        str(frame_id or ""),
        grid_anomaly._grid_cache_identity(path_a),
        grid_anomaly._grid_cache_identity(path_b),
    )


def analyze_class_conflict_frame_paths(
    ones_path: Path | str,
    zeros_path: Path | str,
    *,
    frame_id: str = "",
    use_cache: bool = True,
) -> GridFrameAnalysisResult | None:
    """Load the two masks of a frame, binarise them like the cell analysis, and find conflicts."""

    first_path = Path(ones_path)
    second_path = Path(zeros_path)
    cache_path = grid_anomaly._grid_damage_cache_path(_cache_key(first_path, second_path, frame_id))
    if use_cache and cache_path.is_file():
        try:
            with cache_path.open("rb") as handle:
                payload = pickle.load(handle)
            if isinstance(payload, GridFrameAnalysisResult):
                return payload
        except Exception as error:
            _LOGGER.warning("Ignoring corrupt class-conflict cache entry %s: %s", cache_path, error)
    image_a = grid_anomaly._load_cv2_grayscale_image(first_path)
    image_b = grid_anomaly._load_cv2_grayscale_image(second_path)
    if image_a is None or image_b is None:
        return None
    if image_a.shape != image_b.shape:
        image_b = cv2.resize(image_b, (image_a.shape[1], image_a.shape[0]), interpolation=cv2.INTER_NEAREST)
    result = analyze_class_conflict_masks(
        binarize_mask(image_a),
        binarize_mask(image_b),
        frame_id=frame_id,
        frame_path=f"{first_path.name}|{second_path.name}",
    )
    if use_cache:
        try:
            grid_anomaly.GRID_DAMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            atomic_pickle_dump(cache_path, result)
        except (OSError, pickle.PickleError, TypeError, ValueError) as error:
            _LOGGER.warning("Could not store class-conflict cache for %s: %s", frame_id, error)
    return result


def analyze_class_conflict_chunk(
    entries: tuple[tuple[str, str, str], ...],
    *,
    use_cache: bool = True,
) -> tuple[dict[str, dict[str, GridFrameAnalysisResult]], dict[str, str]]:
    """Conflicts of (frame key, ones mask path, zeros mask path) entries."""

    payloads: dict[str, dict[str, GridFrameAnalysisResult]] = {}
    errors: dict[str, str] = {}
    for key, ones_path, zeros_path in entries:
        try:
            result = analyze_class_conflict_frame_paths(ones_path, zeros_path, frame_id=key, use_cache=use_cache)
        except Exception as error:
            errors[str(key)] = f"{type(error).__name__}: {error}"
            continue
        if result is None:
            errors[str(key)] = "decode_error"
            continue
        payloads[str(key)] = {PAIR_CONFLICT_LAYER: result}
    return payloads, errors
