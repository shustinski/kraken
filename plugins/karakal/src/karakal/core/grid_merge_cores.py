"""Merged-cell detection from distance-transform cores inside a blob.

No lattice / conductor knowledge: a blob is merged when it contains two or more
cell-sized cores and either a neck between them or an area near N cell areas.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .grid_scoring import _logistic, slider_threshold


@dataclass(frozen=True, slots=True)
class MergeCoreHit:
    x: float
    y: float
    radius: float


@dataclass(frozen=True, slots=True)
class MergeCoreAnalysis:
    cores: tuple[MergeCoreHit, ...]
    core_count: int
    has_neck: bool
    area_multiple: float
    solid_block: bool
    score: float
    uncertain_block: bool = False


def merge_core_thresholds(merge_sensitivity: int) -> tuple[float, float]:
    """Return (min_core_frac_of_min_side, area_multiplicity_tolerance)."""

    unit = max(0.0, min(1.0, float(merge_sensitivity) / 100.0))
    # Higher sensitivity → accept slightly smaller cores and looser area match.
    min_core_frac = 0.48 - 0.10 * unit  # 0.48 .. 0.38
    area_tol = 0.18 + 0.14 * unit  # 0.18 .. 0.32
    return float(min_core_frac), float(area_tol)


def filled_contour_roi(
    contour,
    bbox: tuple[int, int, int, int],
    *,
    pad: int = 2,
) -> tuple[np.ndarray, int, int]:
    """Binary ROI with the contour filled; returns (roi, origin_x, origin_y)."""

    left, top, width, height = (int(value) for value in bbox[:4])
    margin = max(1, int(pad))
    origin_x = max(0, left - margin)
    origin_y = max(0, top - margin)
    roi_w = max(1, width + 2 * margin)
    roi_h = max(1, height + 2 * margin)
    roi = np.zeros((roi_h, roi_w), dtype=np.uint8)
    if contour is None:
        return roi, origin_x, origin_y
    shifted = np.asarray(contour, dtype=np.int32).reshape(-1, 1, 2).copy()
    shifted[:, 0, 0] -= origin_x
    shifted[:, 0, 1] -= origin_y
    cv2.drawContours(roi, [shifted], -1, 255, thickness=-1)
    return roi, origin_x, origin_y


def _contour_solidity(contour, area: float) -> float:
    if contour is None or float(area) <= 1.0:
        return 0.0
    points = np.asarray(contour, dtype=np.int32).reshape(-1, 1, 2)
    if len(points) < 3:
        return 0.0
    hull = cv2.convexHull(points)
    hull_area = float(cv2.contourArea(hull))
    if hull_area <= 1.0:
        return 0.0
    return float(area) / hull_area


def find_cell_cores(
    binary_roi: np.ndarray,
    cell_width: float,
    cell_height: float,
    *,
    min_core_frac: float = 0.40,
    nms_frac: float = 0.50,
    distance: np.ndarray | None = None,
) -> tuple[MergeCoreHit, ...]:
    """Local maxima of the distance transform that are cell-sized cores."""

    if binary_roi is None or binary_roi.size == 0:
        return ()
    mask = (binary_roi > 0).astype(np.uint8)
    if int(mask.sum()) < 8:
        return ()
    cell_w = max(2.0, float(cell_width))
    cell_h = max(2.0, float(cell_height))
    min_side = min(cell_w, cell_h)
    min_radius = max(1.5, float(min_core_frac) * min_side)
    # Inscribed radius of a filled cell follows the larger side halfway; do not
    # clip peaks that sit in a normal-height 1×N painted block.
    max_radius = max(min_radius + 0.5, 0.55 * max(cell_w, cell_h))
    if distance is None:
        distance = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    # Suppress weak ridges: kernels about half a cell.
    block = max(3, int(round(0.50 * min_side)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (block, block))
    local_max = cv2.dilate(distance, kernel)
    peaks = (distance >= local_max - 1e-3) & (distance >= min_radius) & (distance <= max_radius) & (mask > 0)
    ys, xs = np.where(peaks)
    if xs.size == 0:
        return ()
    order = np.argsort(distance[ys, xs])[::-1]
    # Half-cell NMS so bumps inside one cell collapse to one core.
    nms_radius = max(min_radius * 1.25, float(nms_frac) * min_side, 0.48 * 0.5 * (cell_w + cell_h))
    kept: list[MergeCoreHit] = []
    for index in order:
        x = float(xs[index])
        y = float(ys[index])
        radius = float(distance[ys[index], xs[index]])
        if any((x - hit.x) ** 2 + (y - hit.y) ** 2 <= nms_radius**2 for hit in kept):
            continue
        kept.append(MergeCoreHit(x=x, y=y, radius=radius))
        if len(kept) >= 24:
            break
    return tuple(kept)


def _cores_at_cell_pitch(
    cores: tuple[MergeCoreHit, ...],
    cell_width: float,
    cell_height: float,
) -> bool:
    """True when at least one neighbour pair sits near a cell pitch (or two)."""

    if len(cores) < 2:
        return False
    pitch_x = max(2.0, float(cell_width))
    pitch_y = max(2.0, float(cell_height))
    # Allow 1–2 pitches so a thick bridge between two cells still counts.
    for index, left in enumerate(cores):
        for right in cores[index + 1 :]:
            dx = abs(right.x - left.x)
            dy = abs(right.y - left.y)
            gap = float(np.hypot(dx, dy))
            along_x = 0.85 * pitch_x <= dx <= 1.70 * pitch_x and dy <= 0.55 * pitch_y
            along_y = 0.85 * pitch_y <= dy <= 1.70 * pitch_y and dx <= 0.55 * pitch_x
            near_diag = (
                0.85 * min(pitch_x, pitch_y) <= gap <= 1.70 * max(pitch_x, pitch_y)
                and min(dx, dy) <= 0.45 * max(pitch_x, pitch_y)
            )
            if along_x or along_y or near_diag:
                return True
    return False


def _has_neck_between_cores(distance: np.ndarray, cores: tuple[MergeCoreHit, ...]) -> bool:
    if len(cores) < 2:
        return False
    height, width = distance.shape[:2]
    for index, left in enumerate(cores):
        for right in cores[index + 1 :]:
            gap = float(np.hypot(right.x - left.x, right.y - left.y))
            if gap < 1.0:
                continue
            # Only neighbour-ish pairs (within ~2.2 cell pitches of their radii).
            if gap > 2.4 * (left.radius + right.radius):
                continue
            samples = max(6, int(round(gap)))
            xs = np.linspace(left.x, right.x, samples)
            ys = np.linspace(left.y, right.y, samples)
            values = []
            for sample_x, sample_y in zip(xs, ys):
                ix = int(round(sample_x))
                iy = int(round(sample_y))
                if ix < 0 or iy < 0 or ix >= width or iy >= height:
                    continue
                values.append(float(distance[iy, ix]))
            if len(values) < 3:
                continue
            neck = min(values)
            core_floor = 0.42 * min(left.radius, right.radius)
            if neck <= core_floor:
                return True
    return False


def _convexity_has_opposing_dents(contour, *, min_depth: float) -> bool:
    if contour is None:
        return False
    points = np.asarray(contour, dtype=np.int32).reshape(-1, 1, 2)
    if len(points) < 6:
        return False
    hull = cv2.convexHull(points, returnPoints=False)
    if hull is None or len(hull) < 3:
        return False
    try:
        defects = cv2.convexityDefects(points, hull)
    except cv2.error:
        return False
    if defects is None or len(defects) == 0:
        return False
    # OpenCV 4 returns (N, 1, 4), OpenCV 5 returns (N, 4); reading row[0][3] failed silently on 5.
    depths = np.asarray(defects, dtype=np.float64).reshape(-1, 4)[:, 3] / 256.0
    return int(np.count_nonzero(depths >= float(min_depth))) >= 2


def _area_matches_multiple(area: float, core_count: int, cell_area: float, tolerance: float) -> bool:
    if core_count < 2 or cell_area <= 1.0:
        return False
    expected = float(core_count) * float(cell_area)
    return abs(float(area) - expected) <= float(tolerance) * expected


def _span_matches_cells(length: float, count: int, cell: float) -> bool:
    """Bounding length covers ~N cells, allowing pitch gaps between bridged cells."""

    cell = max(1.0, float(cell))
    count = int(count)
    if count < 2:
        return False
    lo = 0.80 * count * cell
    hi = 1.20 * count * cell + 0.90 * (count - 1) * cell
    return lo <= float(length) <= hi


def _pack_geometry_ok(
    bbox: tuple[int, int, int, int],
    count: int,
    cell_width: float,
    cell_height: float,
) -> bool:
    """True when the bbox is a 1×N / N×1 pack of cell-sized slots."""

    width = float(bbox[2])
    height = float(bbox[3])
    cell_w = max(1.0, float(cell_width))
    cell_h = max(1.0, float(cell_height))
    horizontal = abs(height / cell_h - 1.0) <= 0.22 and _span_matches_cells(width, count, cell_w)
    vertical = abs(width / cell_w - 1.0) <= 0.22 and _span_matches_cells(height, count, cell_h)
    return horizontal or vertical


def _solid_block_packing(
    bbox: tuple[int, int, int, int],
    area: float,
    cell_width: float,
    cell_height: float,
    core_count: int,
    *,
    area_tol: float,
    solidity: float,
) -> bool:
    """Filled rectangle covering N slots along one axis without a visible neck."""

    if core_count < 2:
        return False
    # Grain strips between zone and array often match 1×N height by accident.
    # True painted blocks are near-rectangles with high solidity / extent.
    if float(solidity) < 0.92:
        return False
    width = float(bbox[2])
    height = float(bbox[3])
    cell_w = max(1.0, float(cell_width))
    cell_h = max(1.0, float(cell_height))
    cell_area = cell_w * cell_h
    extent = float(area) / max(1.0, width * height)
    if extent < 0.88:
        return False
    if not _area_matches_multiple(area, core_count, cell_area, min(area_tol, 0.18)):
        return False
    along_w = abs(height / cell_h - 1.0) <= 0.22 and _span_matches_cells(width, core_count, cell_w)
    along_h = abs(width / cell_w - 1.0) <= 0.22 and _span_matches_cells(height, core_count, cell_h)
    return along_w or along_h


def _estimate_pack_count(
    bbox: tuple[int, int, int, int],
    area: float,
    cell_width: float,
    cell_height: float,
    *,
    area_tol: float,
    solidity: float,
) -> int:
    """How many cell slots a solid 1×N strip covers (no DT cores required)."""

    if float(solidity) < 0.92:
        return 1
    width = float(bbox[2])
    height = float(bbox[3])
    cell_w = max(1.0, float(cell_width))
    cell_h = max(1.0, float(cell_height))
    cell_area = cell_w * cell_h
    extent = float(area) / max(1.0, width * height)
    if extent < 0.88 or float(area) < 1.7 * cell_area:
        return 1
    # Only 1×N / N×1 strips: the short side must match one cell tightly.
    horizontal = abs(height / cell_h - 1.0) <= 0.22 and width / cell_w >= 1.7
    vertical = abs(width / cell_w - 1.0) <= 0.22 and height / cell_h >= 1.7
    if not horizontal and not vertical:
        return 1
    if horizontal:
        count = int(round(float(area) / cell_area))
        if count < 2:
            count = int(round(width / cell_w))
        if count >= 2 and _span_matches_cells(width, count, cell_w):
            if _area_matches_multiple(area, count, cell_area, min(area_tol, 0.18) + 0.08):
                return count
    if vertical:
        count = int(round(float(area) / cell_area))
        if count < 2:
            count = int(round(height / cell_h))
        if count >= 2 and _span_matches_cells(height, count, cell_h):
            if _area_matches_multiple(area, count, cell_area, min(area_tol, 0.18) + 0.08):
                return count
    return 1


def analyze_merge_cores(
    contour,
    bbox: tuple[int, int, int, int],
    *,
    area: float,
    cell_width: float,
    cell_height: float,
    merge_sensitivity: int = 35,
    confidence_roi: np.ndarray | None = None,
) -> MergeCoreAnalysis:
    """Score whether a filled contour is several stuck cells."""

    min_core_frac, area_tol = merge_core_thresholds(merge_sensitivity)
    cell_w = max(2.0, float(cell_width))
    cell_h = max(2.0, float(cell_height))
    cell_area = max(4.0, cell_w * cell_h)
    empty = MergeCoreAnalysis((), 0, False, 0.0, False, 0.0, False)
    if float(area) < 1.4 * cell_area:
        return empty

    solidity = _contour_solidity(contour, float(area))
    pack_count = _estimate_pack_count(
        bbox, float(area), cell_w, cell_h, area_tol=area_tol, solidity=solidity
    )
    width_ratio = float(bbox[2]) / cell_w
    height_ratio = float(bbox[3]) / cell_h
    # Typical single lattice cell: skip the distance transform.
    if (
        pack_count < 2
        and float(area) < 1.85 * cell_area
        and max(width_ratio, height_ratio) < 1.55
        and min(width_ratio, height_ratio) > 0.55
    ):
        return empty

    roi, _ox, _oy = filled_contour_roi(contour, bbox)
    mask = (roi > 0).astype(np.uint8)
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    cores = find_cell_cores(
        roi,
        cell_w,
        cell_h,
        min_core_frac=min_core_frac,
        distance=distance,
    )
    pitched = _cores_at_cell_pitch(cores, cell_w, cell_h)
    if len(cores) < 2 and pack_count < 2:
        return empty
    # Amorphous grain / holes: multiple DT bumps without cell pitch → not a merge.
    if len(cores) >= 2 and not pitched and pack_count < 2:
        return MergeCoreAnalysis(cores, len(cores), False, float(area) / cell_area, False, 0.0, False)

    has_neck = _has_neck_between_cores(distance, cores) if len(cores) >= 2 else False
    if not has_neck and len(cores) >= 2:
        has_neck = _convexity_has_opposing_dents(contour, min_depth=0.22 * min(cell_w, cell_h))
    if not has_neck and confidence_roi is not None and confidence_roi.size and len(cores) >= 2:
        try:
            conf = confidence_roi.astype(np.float32)
            if conf.shape[:2] != roi.shape[:2]:
                conf = cv2.resize(conf, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_LINEAR)
            left, right = cores[0], cores[1]
            samples = max(6, int(round(np.hypot(right.x - left.x, right.y - left.y))))
            xs = np.linspace(left.x, right.x, samples).astype(np.int32)
            ys = np.linspace(left.y, right.y, samples).astype(np.int32)
            xs = np.clip(xs, 0, conf.shape[1] - 1)
            ys = np.clip(ys, 0, conf.shape[0] - 1)
            line = conf[ys, xs]
            if line.size >= 3 and float(np.min(line)) <= 0.85 * float(np.median(line)):
                has_neck = True
        except Exception:
            pass

    count = max(len(cores) if pitched or pack_count < 2 else 0, pack_count)
    if pitched:
        count = max(len(cores), pack_count)
    elif pack_count >= 2:
        count = pack_count
    else:
        count = len(cores)
    multiple = float(area) / cell_area
    area_ok = _area_matches_multiple(area, count, cell_area, area_tol)
    solid_block = _solid_block_packing(
        bbox, area, cell_w, cell_h, count, area_tol=area_tol, solidity=solidity
    )
    if count >= 2 and pack_count >= 2 and not has_neck and solidity >= 0.92:
        solid_block = True
    uncertain = bool(solid_block and not has_neck and len(cores) < 2)

    if count < 2:
        return MergeCoreAnalysis(cores, len(cores), has_neck, multiple, False, 0.0, False)

    extent = float(area) / max(1.0, float(bbox[2]) * float(bbox[3]))
    # Sparse / jagged grain inside a large bbox is not a pack of cells.
    if extent < 0.62 and not solid_block:
        return MergeCoreAnalysis(cores, count, has_neck, multiple, False, 0.0, False)

    geometry_ok = _pack_geometry_ok(bbox, count, cell_w, cell_h)
    thin_row = abs(float(bbox[3]) / cell_h - 1.0) <= 0.25 and float(bbox[2]) / cell_w >= 1.6
    thin_col = abs(float(bbox[2]) / cell_w - 1.0) <= 0.25 and float(bbox[3]) / cell_h >= 1.6
    # Thin 1×N strips are where conductor grain mimics merges. Demand a near-filled
    # rectangle (solid_block) or a very tight area match with a clear neck.
    if thin_row or thin_col:
        evidence_ok = bool(
            solid_block
            or (
                area_ok
                and pitched
                and has_neck
                and geometry_ok
                and solidity >= 0.80
                and extent >= 0.75
                and abs(multiple - float(count)) <= 0.20
            )
            or (
                area_ok
                and pitched
                and has_neck
                and geometry_ok
                and len(cores) >= 5
                and extent >= 0.69
                and (len(cores) >= 6 or solidity >= 0.82)
                and abs(multiple - float(count)) <= 0.45
            )
        )
    else:
        evidence_ok = bool(
            solid_block
            or (
                area_ok
                and pitched
                and has_neck
                and geometry_ok
                and extent >= 0.65
                and solidity >= (0.72 if len(cores) >= 4 else 0.78)
            )
        )
    if not evidence_ok:
        return MergeCoreAnalysis(cores, count, has_neck, multiple, solid_block, 0.0, False)

    threshold = slider_threshold(merge_sensitivity)
    evidence = float(count - 1) + (0.35 if has_neck else 0.0) + (0.25 if area_ok or solid_block else 0.0)
    raw = _logistic((evidence - 0.55) / 0.35)
    score = float(raw) if raw >= max(threshold - 0.15, 0.02) else float(raw) * 0.5
    if score < max(threshold, 0.02) and evidence_ok:
        score = max(score, threshold + 0.05)
    return MergeCoreAnalysis(
        cores=cores,
        core_count=count,
        has_neck=has_neck,
        area_multiple=multiple,
        solid_block=solid_block,
        score=float(min(1.0, score)),
        uncertain_block=uncertain,
    )


def score_merge_cores(
    contour,
    bbox: tuple[int, int, int, int],
    *,
    area: float,
    cell_width: float,
    cell_height: float,
    merge_sensitivity: int = 35,
    confidence_roi: np.ndarray | None = None,
) -> float:
    return float(
        analyze_merge_cores(
            contour,
            bbox,
            area=area,
            cell_width=cell_width,
            cell_height=cell_height,
            merge_sensitivity=merge_sensitivity,
            confidence_roi=confidence_roi,
        ).score
    )
