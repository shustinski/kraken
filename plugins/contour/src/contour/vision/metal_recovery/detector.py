from __future__ import annotations

from dataclasses import dataclass, field
from math import atan2, degrees
from typing import Any

import cv2
import numpy as np

from ...application.preview_cancellation import raise_if_preview_cancelled
from ...contour_extractor import estimate_effective_polygon_width_px
from ...domain import PolygonData, compute_polygon_metrics
from ...domain.polygon_ring import is_valid_closed_polygon_ring
from ...utils import ensure_binary_mask, ensure_uint8

_TOPOLOGY_CHECK_MAX_VERTICES = 192
_TOPOLOGY_REPAIR_MIN_FILL_IOU = 0.98

_CANONICAL_90 = {0.0, 90.0, 180.0}
_CANONICAL_45 = {0.0, 45.0, 90.0, 135.0, 180.0}


def effective_conductor_width_px(width_from_dt: float, rw: float, rh: float) -> float:
    """Blend distance-transform width with minAreaRect minor axis.

    Binary masks and distance transforms often underestimate visible SEM trace width;
    the oriented bounding box narrow side is a practical lower bound on thickness.
    """
    span_minor = min(rw, rh) if rw > 0 and rh > 0 else 0.0
    span_major = max(rw, rh)
    if span_minor <= 1.0:
        return float(width_from_dt)
    blended = float(max(float(width_from_dt), 0.92 * span_minor))
    return float(min(blended, span_major))


from .segmentation import (
    MetalSegmentationConfig,
    build_metal_segmentation_mask,
    normalize_metal_segmentation_strategy,
)


def _normalize_metal_extraction_mode(value: str) -> str:
    """Backward-compatible alias for segmentation strategy normalization."""
    return normalize_metal_segmentation_strategy(value)


def _normalize_metal_segmentation_method(value: str) -> str:
    """Deprecated: maps old threshold leg names."""
    strat = normalize_metal_segmentation_strategy(value)
    if strat in {"local_adaptive", "sauvola", "auto"}:
        return "adaptive"
    return "otsu"


def _normalize_metal_sensitivity_token(value: str) -> str:
    t = str(value or "").strip().lower()
    if t in {"low", "низкая", "низк"}:
        return "low"
    if t in {"high", "высокая", "высок"}:
        return "high"
    return "medium"


@dataclass(slots=True)
class MetalRecoveryConfig:
    noise_suppression: int = 20
    contrast_bias: float = 0.0
    segmentation_strategy: str = "auto"
    gap_bridge_px: int = 2
    speckle_removal_px: int = 0
    contour_smooth_px: float = 0.0
    morph_close_radius: int = 2
    morph_open_radius: int = 0
    min_width_px: float = 8.0
    max_width_px: float | None = None
    min_length_px: float = 8.0
    min_area: float = 60.0
    max_area: float | None = None
    min_perimeter: float = 32.0
    max_perimeter: float | None = None
    epsilon_simplify: float = 2.0
    min_points: int = 4
    min_polygon_angle_deg: float = 30.0
    approximation_enabled: bool = True
    retrieval_external_only: bool = False
    allowed_angles: str = "free"
    angle_tolerance_deg: float = 7.0
    min_straightness: float = 0.2
    allow_t_junction: bool = True
    border_mode: str = "mark"
    check_contour_validity: bool = True
    min_inner_hole_area: float = 100.0
    preset_name: str = "standard"
    use_wide_conductor_gradient: bool = False
    wide_gradient_profile_radius_px: int = 8
    wide_gradient_min_direction_confidence: float = 0.15
    wide_gradient_min_pair_length_px: float = 24.0
    wide_gradient_parallel_tolerance_deg: float = 10.0
    wide_gradient_max_edge_gap_px: int = 5
    wide_gradient_min_overlap_ratio: float = 0.5
    # Grayscale / edge mode: limit morph closing so narrow gaps between adjacent SEM traces stay open.
    edge_close_cap_px: int = 9
    # Split incorrectly merged blobs using distance-transform seeds + watershed (topology on noisy SEM).
    edge_watershed_split: bool = True
    edge_watershed_dist_peak_frac: float = 0.38
    # Skip watershed on very large images (expensive); None/0 = always run.
    edge_watershed_max_pixels: int | None = 3_000_000

    def to_snapshot(self) -> dict[str, Any]:
        return {
            field.name: getattr(self, field.name)
            for field in MetalRecoveryConfig.__dataclass_fields__.values()
        }


@dataclass(slots=True)
class MetalPolygonRecord:
    polygon: PolygonData
    area: float
    perimeter: float
    width_px: float
    length_px: float
    straightness: float
    max_angle_deviation: float
    contour_validity: bool
    border_touch: bool
    reject_reason: str = ""


def format_metal_reject_detail_ru(
    case: str,
    *,
    config: MetalRecoveryConfig,
    width_px: float,
    length_px: float,
    area: float,
    perimeter: float,
    straightness: float,
    max_angle_deviation: float,
    vertex_count: int,
    t_branch_score: int = 0,
) -> str:
    """Human-readable Russian explanation: parameter, actual value, configured bound."""
    c = str(case or "").strip()
    tol = float(config.angle_tolerance_deg)
    mode = str(config.allowed_angles).strip()
    if c in {"мало_вершин"}:
        return (
            f"Топология: у контура {int(vertex_count)} вершин(ы), требуется не менее 3 "
            f"(параметр «мин. число точек» в упрощении: {int(config.min_points)})"
        )
    if c in {"самопересечение_или_топология"} or c.startswith("самопересечение"):
        return (
            "Топология: контур самопересекается, касается сам себя или имеет недопустимую конфигурацию рёбер "
            f"(проверка включена: check_contour_validity={bool(config.check_contour_validity)})"
        )
    if c in {"некорректный_контур", ""}:
        return (
            "Топология: контур не прошёл проверку целостности "
            f"(вершин: {int(vertex_count)}, проверка: {bool(config.check_contour_validity)})"
        )
    if c in {"ширина"}:
        return (
            "Оценка ширины (маска + minAreaRect): "
            f"у полигона {width_px:.2f} px, в настройках минимум {float(config.min_width_px):.2f} px"
        )
    if c in {"ширина_макс"}:
        cap_s = f"{float(config.max_width_px):.2f}" if config.max_width_px is not None else "—"
        return (
            "Оценка ширины (маска + minAreaRect): "
            f"у полигона {width_px:.2f} px, в настройках максимум {cap_s} px"
        )
    if c in {"длина"}:
        return (
            "Длина по minAreaRect (большая сторона): "
            f"у полигона {length_px:.2f} px, в настройках минимум {float(config.min_length_px):.2f} px"
        )
    if c in {"площадь"}:
        return f"Площадь контура: у полигона {area:.1f} px², в настройках минимум {float(config.min_area):.1f} px²"
    if c in {"площадь_макс"}:
        cap_s = f"{float(config.max_area):.1f}" if config.max_area is not None else "—"
        return f"Площадь контура: у полигона {area:.1f} px², в настройках максимум {cap_s} px²"
    if c in {"периметр"}:
        return (
            "Периметр контура: "
            f"у полигона {perimeter:.1f} px, в настройках минимум {float(config.min_perimeter):.1f} px"
        )
    if c in {"периметр_макс"}:
        cap_s = f"{float(config.max_perimeter):.1f}" if config.max_perimeter is not None else "—"
        return (
            "Периметр контура: "
            f"у полигона {perimeter:.1f} px, в настройках максимум {cap_s} px"
        )
    if c in {"прямолинейность"}:
        return (
            "Прямолинейность (2·max(стороны minAreaRect) / периметр): "
            f"у полигона {straightness:.3f}, в настройках минимум {float(config.min_straightness):.3f}"
        )
    if c in {"углы"}:
        return (
            f"Углы полигона: режим «{mode}», допуск ±{tol:.1f}°; "
            f"максимальное отклонение от допустимых направлений {max_angle_deviation:.2f}° "
            f"(ожидалось не больше {tol:.1f}°)"
        )
    if c in {"т_соединение_запрещено"}:
        return (
            "Т-образное ветвление (выпуклые дефекты контура): "
            f"оценка {int(t_branch_score)} > 2 при выключенном параметре «разрешить Т-соединения» "
            f"(allow_t_junction={bool(config.allow_t_junction)})"
        )
    if c in {"граница"}:
        return (
            "Касание границы кадра: контур касается края изображения; "
            f"режим обработки таких контуров: «{config.border_mode}» (ожидалось не попадать на край при режиме ignore)"
        )
    return f"Отклонение: {c}"


def format_metal_suspicion_detail_ru(
    *,
    config: MetalRecoveryConfig,
    max_angle_deviation: float,
    straightness: float,
    angle_soft: bool,
    straight_soft: bool,
) -> str:
    parts: list[str] = []
    tol = float(config.angle_tolerance_deg)
    soft_ang = tol * 0.65
    min_s = float(config.min_straightness)
    soft_s = min_s * 0.85
    if angle_soft:
        parts.append(
            f"углы близки к порогу: отклонение {max_angle_deviation:.2f}° при жёстком допуске ±{tol:.1f}° "
            f"(мягкое предупреждение уже от {soft_ang:.2f}°)"
        )
    if straight_soft:
        parts.append(
            f"прямолинейность близка к минимуму: {straightness:.3f} при пороге {min_s:.3f} "
            f"(мягкое предупреждение от {soft_s:.3f})"
        )
    if not parts:
        return "Сомнительный контур: сработали мягкие эвристики без детализации."
    return "Сомнительно: " + "; ".join(parts)


@dataclass(slots=True)
class MetalDetectionResult:
    accepted: list[PolygonData] = field(default_factory=list)
    rejected: list[MetalPolygonRecord] = field(default_factory=list)
    suspicious: list[MetalPolygonRecord] = field(default_factory=list)
    border: list[MetalPolygonRecord] = field(default_factory=list)
    debug_images: dict[str, np.ndarray] = field(default_factory=dict)
    params_snapshot: dict[str, Any] = field(default_factory=dict)
    wide_gradient_overlays: dict[str, list[PolygonData]] = field(default_factory=dict)


def _segmentation_config_from_recovery(config: MetalRecoveryConfig) -> MetalSegmentationConfig:
    gap = max(0, int(getattr(config, "gap_bridge_px", config.morph_close_radius) or 0))
    speckle = max(0, int(getattr(config, "speckle_removal_px", config.morph_open_radius) or 0))
    return MetalSegmentationConfig(
        noise_suppression=max(0, min(100, int(config.noise_suppression))),
        contrast_bias=float(config.contrast_bias),
        segmentation_strategy=normalize_metal_segmentation_strategy(config.segmentation_strategy),
        gap_bridge_px=gap,
        speckle_removal_px=speckle,
        min_width_px=float(config.min_width_px),
        edge_close_cap_px=int(config.edge_close_cap_px),
        edge_watershed_split=bool(config.edge_watershed_split),
        edge_watershed_dist_peak_frac=float(config.edge_watershed_dist_peak_frac),
        edge_watershed_max_pixels=config.edge_watershed_max_pixels,
    )


def build_metal_extraction_mask(
    gray: np.ndarray, config: MetalRecoveryConfig
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Build the working binary mask for conductor contour extraction."""
    raise_if_preview_cancelled()
    seg_cfg = _segmentation_config_from_recovery(config)
    result = build_metal_segmentation_mask(gray, seg_cfg)
    extra = dict(result.debug_images)
    if result.strategy == "legacy_otsu":
        extra["metal_threshold_mask"] = result.raw_segmentation
    return ensure_binary_mask(result.mask), extra


def _border_touch(
    contour: np.ndarray,
    *,
    width: int,
    height: int,
    margin_px: int = 1,
) -> bool:
    m = max(0, int(margin_px))
    x, y, w, h = cv2.boundingRect(contour)
    return x <= m or y <= m or x + w >= width - m or y + h >= height - m


def _straightness_metric(rect_w: float, rect_h: float, perimeter: float) -> float:
    major = max(float(rect_w), float(rect_h))
    if perimeter <= 1e-6:
        return 0.0
    return float(min(1.0, (2.0 * major) / perimeter))


def _angle_deviation_for_mode(
    points: list[tuple[float, float]],
    *,
    allowed: str,
    tolerance_deg: float,
) -> tuple[float, bool]:
    if len(points) < 3:
        return 0.0, True
    mode = str(allowed).strip().lower()
    if mode in {"free", "arbitrary", "произвольные", "any"}:
        return 0.0, True

    if mode in {"90_only", "90", "ortho", "только_90"}:
        canonical = _CANONICAL_90
    else:
        canonical = _CANONICAL_45

    max_dev = 0.0
    n = len(points)
    if points[0] == points[-1]:
        pts = points[:-1]
        n = len(pts)
    else:
        pts = points
    if n < 3:
        return 0.0, True

    for i in range(n):
        p0 = pts[(i - 1) % n]
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        a1 = atan2(p0[1] - p1[1], p0[0] - p1[0])
        a2 = atan2(p2[1] - p1[1], p2[0] - p1[0])
        ang = degrees(abs(a2 - a1))
        if ang > 180.0:
            ang = 360.0 - ang
        interior = 180.0 - ang
        if interior < 0.0:
            interior = 0.0
        best = min(abs(interior - c) for c in canonical)
        dev = best
        max_dev = max(max_dev, dev)
    ok = max_dev <= tolerance_deg + 1e-4
    return float(max_dev), bool(ok)


def _convex_branching_score(contour: np.ndarray) -> int:
    hull = cv2.convexHull(contour, returnPoints=False)
    if hull is None or len(hull) < 3:
        return 0
    defects = cv2.convexityDefects(contour, hull)
    if defects is None:
        return 0
    deep = 0
    for i in range(defects.shape[0]):
        _s, _e, _f, d = defects[i, 0]
        if float(d) / 256.0 > 4.0:
            deep += 1
    return deep


def _vectorize_contour(contour: np.ndarray, config: MetalRecoveryConfig) -> list[tuple[float, float]]:
    return _contour_to_polygon(
        contour,
        epsilon=config.epsilon_simplify,
        approx_enabled=config.approximation_enabled,
        min_angle_deg=config.min_polygon_angle_deg,
        smooth_px=float(getattr(config, "contour_smooth_px", 0.0) or 0.0),
        allowed_angles=config.allowed_angles,
        angle_tolerance_deg=config.angle_tolerance_deg,
    )


def _snap_rectilinear(
    points: list[tuple[float, float]],
    *,
    allowed_angles: str,
    tolerance_deg: float,
) -> list[tuple[float, float]]:
    mode = str(allowed_angles or "free").strip().lower()
    if mode == "free" or len(points) < 3:
        return points
    canon = sorted(_CANONICAL_45 if mode == "45_90" else _CANONICAL_90)
    tol = max(0.5, float(tolerance_deg))
    snapped = list(points)
    n = len(snapped)
    for i in range(n):
        x0, y0 = snapped[i]
        x1, y1 = snapped[(i + 1) % n]
        dx, dy = x1 - x0, y1 - y0
        length = float(np.hypot(dx, dy))
        if length < 1e-3:
            continue
        ang = degrees(atan2(dy, dx)) % 180.0
        best = min(canon, key=lambda c: min(abs(ang - c), abs(ang - c - 180.0), abs(ang - c + 180.0)))
        dev = min(abs(ang - best), abs(ang - best - 180.0), abs(ang - best + 180.0))
        if dev <= tol:
            rad = np.deg2rad(best)
            nx = x0 + length * float(np.cos(rad))
            ny = y0 + length * float(np.sin(rad))
            snapped[(i + 1) % n] = (nx, ny)
    return snapped


def _contour_to_polygon(
    contour: np.ndarray,
    *,
    epsilon: float,
    approx_enabled: bool,
    min_angle_deg: float,
    smooth_px: float = 0.0,
    allowed_angles: str = "free",
    angle_tolerance_deg: float = 7.0,
) -> list[tuple[float, float]]:
    work = contour
    if smooth_px > 0.05 and len(contour) >= 4:
        peri = float(cv2.arcLength(contour, True))
        if peri > 1.0:
            smooth_eps = max(0.1, min(float(smooth_px), 0.02 * peri))
            work = cv2.approxPolyDP(contour, smooth_eps, True)
    if not approx_enabled or epsilon <= 0:
        pts = [(float(p[0][0]), float(p[0][1])) for p in work]
    else:
        eps = max(0.1, float(epsilon))
        simplified = cv2.approxPolyDP(work, eps, True)
        pts = [(float(p[0][0]), float(p[0][1])) for p in simplified]
    if len(pts) >= 3 and min_angle_deg > 1e-6:
        pts = _remove_vertices_below_angle(pts, float(min_angle_deg))
    if len(pts) >= 3 and str(allowed_angles) != "free":
        pts = _snap_rectilinear(pts, allowed_angles=allowed_angles, tolerance_deg=angle_tolerance_deg)
    return pts


def _vertex_angle_deg(
    prev_point: tuple[float, float],
    current_point: tuple[float, float],
    next_point: tuple[float, float],
) -> float:
    a1 = atan2(prev_point[1] - current_point[1], prev_point[0] - current_point[0])
    a2 = atan2(next_point[1] - current_point[1], next_point[0] - current_point[0])
    angle = degrees(abs(a2 - a1))
    if angle > 180.0:
        angle = 360.0 - angle
    return float(angle)


def _remove_vertices_below_angle(
    points: list[tuple[float, float]],
    min_angle_deg: float,
) -> list[tuple[float, float]]:
    if len(points) < 3 or min_angle_deg <= 0.0:
        return points
    limit = max(0.0, min(180.0, float(min_angle_deg)))
    cleaned = list(points)
    changed = True
    while changed and len(cleaned) >= 4:
        changed = False
        for index, current_point in enumerate(cleaned):
            prev_point = cleaned[(index - 1) % len(cleaned)]
            next_point = cleaned[(index + 1) % len(cleaned)]
            if _vertex_angle_deg(prev_point, current_point, next_point) < limit - 1e-3:
                del cleaned[index]
                changed = True
                break
    return cleaned


def _filled_contour_iou(
    contour: np.ndarray,
    points: list[tuple[float, float]],
    shape_hw: tuple[int, int],
) -> float:
    h, w = shape_hw
    if h <= 0 or w <= 0 or contour is None or len(contour) < 3 or len(points) < 3:
        return 0.0
    contour_mask = np.zeros((h, w), dtype=np.uint8)
    polygon_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(contour_mask, [np.round(contour).astype(np.int32)], 0, 1, thickness=-1)
    polygon = np.array([(round(x), round(y)) for x, y in points], dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(polygon_mask, [polygon], 1)
    inter = int(np.logical_and(contour_mask, polygon_mask).sum())
    union = int(np.logical_or(contour_mask, polygon_mask).sum())
    if union <= 0:
        return 0.0
    return float(inter / union)


def _points_to_contour(points: list[tuple[float, float]]) -> np.ndarray:
    return np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)


def _filled_points_iou(
    left: list[tuple[float, float]],
    right: list[tuple[float, float]],
    shape_hw: tuple[int, int],
) -> float:
    return _filled_contour_iou(_points_to_contour(left), right, shape_hw)


def _repair_polygon_from_raster_fill(
    points: list[tuple[float, float]],
    config: MetalRecoveryConfig,
    shape_hw: tuple[int, int],
) -> tuple[list[tuple[float, float]], bool, str]:
    h, w = shape_hw
    if h <= 0 or w <= 0 or len(points) < 3:
        return points, False, "мало_вершин"
    mask = np.zeros((h, w), dtype=np.uint8)
    arr = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [arr], 255)
    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return points, False, "самопересечение_или_топология"
    contour = max(contours, key=lambda item: abs(float(cv2.contourArea(item))))
    candidate = _vectorize_contour(contour, config)
    valid, reason = _valid_topology(candidate, enabled=True)
    if not valid:
        return candidate, False, reason
    if _filled_points_iou(points, candidate, shape_hw) < _TOPOLOGY_REPAIR_MIN_FILL_IOU:
        return candidate, False, reason or "самопересечение_или_топология"
    return candidate, True, ""


def _repair_contour_polygon_for_topology(
    contour: np.ndarray,
    points: list[tuple[float, float]],
    config: MetalRecoveryConfig,
    shape_hw: tuple[int, int],
) -> tuple[list[tuple[float, float]], bool, str]:
    valid, reason = _valid_topology(points, enabled=True)
    if valid:
        return points, True, ""

    base_epsilon = max(0.1, float(config.epsilon_simplify))
    multipliers = (0.6, 0.8, 1.0, 1.15, 1.3, 1.5, 1.7, 2.0, 2.4, 2.8, 3.2, 3.8, 4.5, 5.5)
    for multiplier in multipliers:
        candidate = _contour_to_polygon(
            contour,
            epsilon=base_epsilon * multiplier,
            approx_enabled=True,
            min_angle_deg=float(config.min_polygon_angle_deg),
            smooth_px=float(getattr(config, "contour_smooth_px", 0.0) or 0.0),
            allowed_angles=config.allowed_angles,
            angle_tolerance_deg=config.angle_tolerance_deg,
        )
        candidate_valid, candidate_reason = _valid_topology(candidate, enabled=True)
        if not candidate_valid:
            reason = candidate_reason or reason
            continue
        if _filled_contour_iou(contour, candidate, shape_hw) < _TOPOLOGY_REPAIR_MIN_FILL_IOU:
            continue
        return candidate, True, ""

    raster_pts, raster_ok, raster_reason = _repair_polygon_from_raster_fill(points, config, shape_hw)
    if raster_ok:
        return raster_pts, True, ""
    return points, False, raster_reason or reason


def _finalize_accepted_polygon_topology(
    accepted: list[PolygonData],
    shape_hw: tuple[int, int],
    config: MetalRecoveryConfig,
) -> list[PolygonData]:
    kept: list[PolygonData] = []
    for poly in accepted:
        contour = _points_to_contour(poly.points)
        repaired, valid, _reason = _repair_contour_polygon_for_topology(
            contour,
            poly.points,
            config,
            shape_hw,
        )
        if not valid or len(repaired) < 3:
            continue
        out = poly if repaired == poly.points else poly.clone()
        if repaired != poly.points:
            out.points = repaired
            out.area, out.perimeter, out.bbox = compute_polygon_metrics(repaired)
        kept.append(out)
    return kept


def _polygon_raster_iou(a: PolygonData, b: PolygonData, shape_hw: tuple[int, int]) -> float:
    h, w = shape_hw
    if h <= 0 or w <= 0:
        return 0.0
    ax, ay, aw, ah = a.bbox
    bx, by, bw, bh = b.bbox
    x0 = max(0, min(ax, bx))
    y0 = max(0, min(ay, by))
    x1 = min(w, max(ax + aw, bx + bw))
    y1 = min(h, max(ay + ah, by + bh))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    if ax >= bx + bw or bx >= ax + aw or ay >= by + bh or by >= ay + ah:
        return 0.0
    roi_h, roi_w = y1 - y0, x1 - x0
    m1 = np.zeros((roi_h, roi_w), dtype=np.uint8)
    m2 = np.zeros((roi_h, roi_w), dtype=np.uint8)
    ca = np.array([(int(x) - x0, int(y) - y0) for x, y in a.points], dtype=np.int32).reshape(-1, 1, 2)
    cb = np.array([(int(x) - x0, int(y) - y0) for x, y in b.points], dtype=np.int32).reshape(-1, 1, 2)
    if ca.shape[0] < 3 or cb.shape[0] < 3:
        return 0.0
    cv2.fillPoly(m1, [ca], 1)
    cv2.fillPoly(m2, [cb], 1)
    inter = int(np.logical_and(m1, m2).sum())
    union = int(np.logical_or(m1, m2).sum())
    if union <= 0:
        return 0.0
    return float(inter / union)


def _nms_polygons_by_mask_iou(polygons: list[PolygonData], shape_hw: tuple[int, int], *, iou_threshold: float) -> list[PolygonData]:
    ordered = sorted(polygons, key=lambda p: float(p.area), reverse=True)
    kept: list[PolygonData] = []
    for p in ordered:
        px, py, pw, ph = p.bbox
        if any(
            not (px >= k.bbox[0] + k.bbox[2] or k.bbox[0] >= px + pw or py >= k.bbox[1] + k.bbox[3] or k.bbox[1] >= py + ph)
            and _polygon_raster_iou(p, k, shape_hw) >= iou_threshold
            for k in kept
        ):
            continue
        kept.append(p)
    return kept


def _merge_base_and_wide_metal(
    base: list[PolygonData],
    wide: list[PolygonData],
    shape_hw: tuple[int, int],
    *,
    iou_threshold: float = 0.35,
) -> list[PolygonData]:
    wide_dedup = _nms_polygons_by_mask_iou(wide, shape_hw, iou_threshold=0.45)
    merged = [p.clone() for p in base]
    for wp in wide_dedup:
        wpc = wp.clone()
        best_j = -1
        best_iou = 0.0
        for j, bp in enumerate(merged):
            wx, wy, ww, wh = wpc.bbox
            bx, by, bw, bh = bp.bbox
            if wx >= bx + bw or bx >= wx + ww or wy >= by + bh or by >= wy + wh:
                continue
            iou = _polygon_raster_iou(wpc, bp, shape_hw)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j >= 0 and best_iou >= iou_threshold:
            base_area = float(merged[best_j].area)
            wide_area = float(wpc.area)
            score_b = base_area
            score_w = wide_area * 0.95
            if score_w > score_b:
                merged[best_j] = wpc
        else:
            merged.append(wpc)
    return merged


def _valid_topology(points: list[tuple[float, float]], *, enabled: bool) -> tuple[bool, str]:
    if not enabled:
        return True, ""
    if len(points) < 3:
        return False, "мало_вершин"
    sample: list[tuple[float, float]] = list(points)
    # Dense chains (SEM / CHAIN_APPROX): uniform subsampling [::step] can introduce
    # spurious segment crossings that are not present on the true boundary; use
    # polyline simplification instead so the checked ring stays geometrically faithful.
    if len(sample) > _TOPOLOGY_CHECK_MAX_VERTICES:
        cnt = np.asarray(sample, dtype=np.float32).reshape(-1, 1, 2)
        peri = float(cv2.arcLength(cnt, True))
        if peri <= 1e-6:
            return False, "самопересечение_или_топология"
        eps = max(0.45, 0.001 * peri)
        simplified = cnt
        for _ in range(18):
            simplified = cv2.approxPolyDP(cnt, eps, True)
            if len(simplified) <= _TOPOLOGY_CHECK_MAX_VERTICES:
                break
            eps *= 1.28
        if len(simplified) < 3:
            return False, "мало_вершин"
        sample = [(float(simplified[i][0][0]), float(simplified[i][0][1])) for i in range(len(simplified))]
    if not is_valid_closed_polygon_ring(sample):
        return False, "самопересечение_или_топология"
    return True, ""


def _contour_depth(contour_index: int, hierarchy: np.ndarray, cache: dict[int, int]) -> int:
    if contour_index in cache:
        return cache[contour_index]
    parent_index = int(hierarchy[contour_index][3])
    if parent_index < 0:
        cache[contour_index] = 0
        return 0
    depth = _contour_depth(parent_index, hierarchy, cache) + 1
    cache[contour_index] = depth
    return depth


def _hierarchy_polygon_from_contour(
    contour: np.ndarray,
    *,
    polygon_id: int,
    parent_id: int,
    is_hole: bool,
    config: MetalRecoveryConfig,
    image_shape: tuple[int, int],
) -> PolygonData | None:
    raw_pts = [(float(contour[i][0][0]), float(contour[i][0][1])) for i in range(len(contour))]
    points = _vectorize_contour(contour, config)
    topo_pts = points if len(points) >= 3 else raw_pts
    topo_pts, valid, topo_reason = _repair_contour_polygon_for_topology(contour, topo_pts, config, image_shape)
    if not valid or len(topo_pts) < 3:
        return None
    points = topo_pts
    use_pts = points if len(points) >= 3 else raw_pts
    if len(use_pts) < 3:
        return None

    polygon = PolygonData(
        id=polygon_id,
        points=use_pts,
        is_hole=is_hole,
        parent_id=parent_id,
        category="conductor",
        shape_hint="polygon",
    )
    polygon.area, polygon.perimeter, polygon.bbox = compute_polygon_metrics(polygon.points)
    return polygon


def _append_hierarchy_descendants(
    accepted: list[PolygonData],
    accepted_mask: np.ndarray,
    raw_contours: tuple[np.ndarray, ...],
    hierarchy: np.ndarray,
    contour_to_polygon_id: dict[int, int],
    config: MetalRecoveryConfig,
) -> int:
    if config.retrieval_external_only or not hierarchy.size:
        return 0
    h, w = accepted_mask.shape[:2]
    depth_cache: dict[int, int] = {}
    children_by_parent: dict[int, list[int]] = {}
    for idx in range(min(len(raw_contours), hierarchy.shape[0])):
        parent_index = int(hierarchy[idx][3])
        if parent_index >= 0:
            children_by_parent.setdefault(parent_index, []).append(idx)

    added = 0
    next_id = max((polygon.id for polygon in accepted), default=0) + 1

    def visit_children(parent_contour_index: int, parent_polygon_id: int) -> None:
        nonlocal added, next_id
        for idx in children_by_parent.get(parent_contour_index, []):
            contour = raw_contours[idx]
            if contour is None or len(contour) < 3:
                continue
            depth = _contour_depth(idx, hierarchy, depth_cache)
            is_hole = bool(depth % 2)
            polygon = _hierarchy_polygon_from_contour(
                contour,
                polygon_id=next_id,
                parent_id=parent_polygon_id,
                is_hole=is_hole,
                config=config,
                image_shape=(h, w),
            )
            accepted_current = False
            if polygon is not None:
                if is_hole:
                    accepted_current = abs(float(polygon.area)) >= float(config.min_inner_hole_area)
                else:
                    accepted_current = (
                        abs(float(polygon.area)) + 1e-3 >= float(config.min_area)
                        and float(polygon.perimeter) + 1e-3 >= float(config.min_perimeter)
                    )
                    if accepted_current:
                        width_px, _wm = estimate_effective_polygon_width_px(accepted_mask, contour)
                        accepted_current = width_px + 1e-3 >= float(config.min_width_px)

            if accepted_current and polygon is not None:
                accepted.append(polygon)
                contour_to_polygon_id[idx] = polygon.id
                cv2.drawContours(accepted_mask, [contour], 0, 0 if is_hole else 255, thickness=-1)
                child_parent_polygon_id = polygon.id
                next_id += 1
                added += 1
            else:
                child_parent_polygon_id = parent_polygon_id
            visit_children(idx, child_parent_polygon_id)

    for root_contour_index, root_polygon_id in list(contour_to_polygon_id.items()):
        visit_children(root_contour_index, root_polygon_id)
    return added


def _append_hierarchy_holes(
    accepted: list[PolygonData],
    accepted_mask: np.ndarray,
    raw_contours: tuple[np.ndarray, ...],
    hierarchy: np.ndarray,
    contour_to_polygon_id: dict[int, int],
    config: MetalRecoveryConfig,
) -> int:
    """Backward-compatible wrapper for the full RETR_TREE descendant import."""
    return _append_hierarchy_descendants(
        accepted,
        accepted_mask,
        raw_contours,
        hierarchy,
        contour_to_polygon_id,
        config,
    )


def _renumber_polygons_preserving_parents(polygons: list[PolygonData]) -> None:
    old_to_new: dict[int, int] = {}
    for new_id, poly in enumerate(polygons, start=1):
        old_to_new[int(poly.id)] = new_id
    for poly in polygons:
        poly.id = old_to_new[int(poly.id)]
    for poly in polygons:
        if poly.parent_id is not None:
            poly.parent_id = old_to_new.get(int(poly.parent_id), poly.parent_id)


def detect_metalization(image: np.ndarray, config: MetalRecoveryConfig) -> MetalDetectionResult:
    if image.ndim == 3:
        gray = cv2.cvtColor(ensure_uint8(image), cv2.COLOR_BGR2GRAY)
    else:
        gray = ensure_uint8(image)
    if gray.size == 0:
        return MetalDetectionResult(params_snapshot=config.to_snapshot())

    mask, pre_dbg = build_metal_extraction_mask(gray, config)
    raise_if_preview_cancelled()
    h, w = mask.shape[:2]
    retr = cv2.RETR_EXTERNAL if config.retrieval_external_only else cv2.RETR_TREE
    raw_contours, hierarchy = cv2.findContours(mask, retr, cv2.CHAIN_APPROX_SIMPLE)
    raise_if_preview_cancelled()
    if hierarchy is None:
        hierarchy = np.empty((1, 0, 4), dtype=np.int32)

    accepted: list[PolygonData] = []
    rejected: list[MetalPolygonRecord] = []
    suspicious: list[MetalPolygonRecord] = []
    border: list[MetalPolygonRecord] = []
    accepted_mask = np.zeros_like(mask)

    next_id = 1
    contour_to_polygon_id: dict[int, int] = {}
    hierarchy_array = hierarchy[0] if hierarchy.size else np.empty((0, 4), dtype=np.int32)
    for idx, contour in enumerate(raw_contours):
        raise_if_preview_cancelled()
        if contour is None or len(contour) < 3:
            continue
        parent = int(hierarchy_array[idx][3]) if idx < hierarchy_array.shape[0] else -1
        if not config.retrieval_external_only and parent != -1:
            continue

        area = abs(float(cv2.contourArea(contour)))
        perimeter = float(cv2.arcLength(contour, True))
        if area < 1.0:
            continue

        rect = cv2.minAreaRect(contour)
        rw, rh = float(rect[1][0]), float(rect[1][1])
        width_px = min(rw, rh) if rw > 0.0 and rh > 0.0 else 0.0
        length_px = float(max(rw, rh))
        straight = _straightness_metric(rw, rh, perimeter)
        b_touch = _border_touch(contour, width=w, height=h)

        reject_case = ""
        t_branch_score = 0
        valid = True
        topo_reason = ""
        dev, ang_ok = 0.0, True
        n_vertices = int(len(contour))
        raw_pts = [(float(contour[i][0][0]), float(contour[i][0][1])) for i in range(len(contour))]
        points = raw_pts
        topo_pts = raw_pts
        if length_px + 1e-3 < config.min_length_px:
            reject_case = "length"
        elif False and n_vertices < max(3, int(config.min_points)):
            reject_case = "мало_вершин"
        elif False and width_px + 1e-3 < config.min_width_px:
            reject_case = "ширина"
        elif False and config.max_width_px is not None and width_px > float(config.max_width_px) + 1e-3:
            reject_case = "ширина_макс"
        elif False and length_px + 1e-3 < config.min_length_px:
            reject_case = "длина"
        elif area + 1e-3 < config.min_area:
            reject_case = "площадь"
        elif config.max_area is not None and area > float(config.max_area) + 1e-3:
            reject_case = "площадь_макс"
        elif perimeter + 1e-3 < config.min_perimeter:
            reject_case = "периметр"
        elif config.max_perimeter is not None and perimeter > float(config.max_perimeter) + 1e-3:
            reject_case = "периметр_макс"
        elif False and not config.allow_t_junction:
            t_branch_score = _convex_branching_score(contour)
            if t_branch_score > 2:
                reject_case = "т_соединение_запрещено"
        if not reject_case:
            width_dt, _wm = estimate_effective_polygon_width_px(mask, contour)
            width_px = effective_conductor_width_px(width_dt, rw, rh)
            if width_px + 1e-3 < config.min_width_px:
                reject_case = "width"
            elif config.max_width_px is not None and width_px > float(config.max_width_px) + 1e-3:
                reject_case = "width_max"

        if not reject_case:
            points = _vectorize_contour(contour, config)
            topo_pts = points if len(points) >= 3 else raw_pts
            topo_pts, valid, topo_reason = _repair_contour_polygon_for_topology(contour, topo_pts, config, (h, w))
            if valid and len(topo_pts) >= 3:
                points = topo_pts
            if len(points) < 3:
                dev, ang_ok = 0.0, True
            else:
                dev, ang_ok = _angle_deviation_for_mode(
                    points,
                    allowed=config.allowed_angles,
                    tolerance_deg=config.angle_tolerance_deg,
                )

            n_vertices = len(topo_pts)
            if n_vertices >= 2 and topo_pts[0] == topo_pts[-1]:
                n_vertices = max(0, n_vertices - 1)

            if not valid:
                reject_case = topo_reason or "самопересечение_или_топология"
            elif n_vertices < max(3, int(config.min_points)):
                reject_case = "min_vertices"
            elif not config.allow_t_junction:
                t_branch_score = _convex_branching_score(contour)
                if t_branch_score > 2:
                    reject_case = "t_junction_forbidden"

        angle_soft = bool(dev > config.angle_tolerance_deg * 0.65)
        straight_soft = bool(straight < config.min_straightness - 1e-4)
        soft_suspicious = angle_soft or straight_soft

        if not reject_case and config.border_mode == "ignore" and b_touch:
            reject_case = "граница"

        reject_reason = ""
        if reject_case:
            reject_reason = format_metal_reject_detail_ru(
                reject_case,
                config=config,
                width_px=float(width_px),
                length_px=length_px,
                area=area,
                perimeter=perimeter,
                straightness=straight,
                max_angle_deviation=float(dev),
                vertex_count=n_vertices,
                t_branch_score=int(t_branch_score),
            )

        use_pts = points if len(points) >= 3 else raw_pts
        poly = PolygonData(
            id=next_id,
            points=use_pts,
            is_hole=False,
            parent_id=None,
            category="conductor",
            shape_hint="polygon",
        )
        poly.area, poly.perimeter, poly.bbox = compute_polygon_metrics(poly.points)

        rec = MetalPolygonRecord(
            polygon=poly,
            area=area,
            perimeter=perimeter,
            width_px=float(width_px),
            length_px=length_px,
            straightness=straight,
            max_angle_deviation=float(dev),
            contour_validity=valid,
            border_touch=b_touch,
            reject_reason=reject_reason,
        )

        if reject_reason:
            rejected.append(rec)
            continue

        if config.border_mode == "mark" and b_touch:
            border_poly = poly.clone()
            border_poly.category = "metal_border"
            border.append(
                MetalPolygonRecord(
                    polygon=border_poly,
                    area=area,
                    perimeter=perimeter,
                    width_px=float(width_px),
                    length_px=length_px,
                    straightness=straight,
                    max_angle_deviation=float(dev),
                    contour_validity=valid,
                    border_touch=True,
                    reject_reason="",
                )
            )
            accept_poly = poly.clone()
            accept_poly.category = "metal_border"
        else:
            accept_poly = poly.clone()
            accept_poly.category = "conductor"

        accept_poly.id = next_id
        next_id += 1
        accepted.append(accept_poly)
        contour_to_polygon_id[idx] = accept_poly.id
        cv2.drawContours(accepted_mask, [contour], 0, 255, thickness=-1)

        if soft_suspicious:
            suspicious.append(
                MetalPolygonRecord(
                    polygon=accept_poly.clone(),
                    area=area,
                    perimeter=perimeter,
                    width_px=float(width_px),
                    length_px=length_px,
                    straightness=straight,
                    max_angle_deviation=float(dev),
                    contour_validity=valid,
                    border_touch=b_touch,
                    reject_reason=format_metal_suspicion_detail_ru(
                        config=config,
                        max_angle_deviation=float(dev),
                        straightness=straight,
                        angle_soft=angle_soft,
                        straight_soft=straight_soft,
                    ),
                )
            )

    dbg: dict[str, np.ndarray] = {
        "metal_source_gray": gray,
        "metal_binary_mask": mask,
        "metal_filtered_mask": accepted_mask,
    }
    for key, value in pre_dbg.items():
        if isinstance(value, np.ndarray):
            dbg[key] = value
    raise_if_preview_cancelled()
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(vis, raw_contours, -1, (0, 255, 0), 1)
    dbg["metal_contours_raw"] = vis
    dbg["metal_width_check"] = cv2.cvtColor(accepted_mask, cv2.COLOR_GRAY2BGR)

    wide_ov: dict[str, list[PolygonData]] = {}
    if config.use_wide_conductor_gradient:
        raise_if_preview_cancelled()
        from .wide_gradient import recover_wide_conductors_by_gradient

        wide_polys, w_dbg, wide_ov = recover_wide_conductors_by_gradient(gray, config)
        dbg.update(w_dbg)
        accepted = _merge_base_and_wide_metal(accepted, wide_polys, (h, w))

    _append_hierarchy_holes(
        accepted,
        accepted_mask,
        raw_contours,
        hierarchy_array,
        contour_to_polygon_id,
        config,
    )

    accepted = _finalize_accepted_polygon_topology(accepted, (h, w), config)

    _renumber_polygons_preserving_parents(accepted)

    return MetalDetectionResult(
        accepted=accepted,
        rejected=rejected,
        suspicious=suspicious,
        border=border,
        debug_images=dbg,
        params_snapshot=config.to_snapshot(),
        wide_gradient_overlays=wide_ov,
    )
