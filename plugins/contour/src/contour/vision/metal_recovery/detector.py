from __future__ import annotations

from dataclasses import dataclass, field, replace
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


def _normalize_metal_extraction_mode(value: str) -> str:
    """User-facing extraction mode: none | otsu | adaptive | hybrid."""
    t = str(value or "").strip().lower()
    if t in {
        "none",
        "off",
        "disabled",
        "без",
        "без_сегментации",
        "без сегментации",
        "grayscale",
        "edges",
        "edge",
        "no_segmentation",
        "no-segmentation",
    }:
        return "none"
    if t in {"hybrid", "гибрид", "both", "комбинированный"}:
        return "hybrid"
    if t in {"adaptive", "адаптив", "адаптивная"}:
        return "adaptive"
    if t in {"otsu"}:
        return "otsu"
    return "none"


def _normalize_metal_segmentation_method(value: str) -> str:
    """Threshold leg: Otsu vs Adaptive (only used inside `_segment_bright_metal`)."""
    return "adaptive" if str(value).strip().lower() in {"adaptive", "адаптив", "адаптивная"} else "otsu"


def _normalize_metal_sensitivity_token(value: str) -> str:
    t = str(value or "").strip().lower()
    if t in {"low", "низкая", "низк"}:
        return "low"
    if t in {"high", "высокая", "высок"}:
        return "high"
    return "medium"


def _sensitivity_offsets(sensitivity_0_100: int, token: str) -> tuple[float, int]:
    """Return (adaptive_c_offset, otsu_morph_delta) from unified sensitivity."""
    s = max(0, min(100, int(sensitivity_0_100)))
    mid = {"low": 35, "medium": 50, "high": 65}[_normalize_metal_sensitivity_token(token)]
    blend = 0.35 * mid + 0.65 * s
    # Higher blend -> more aggressive foreground (more detections)
    adaptive_c = -6.0 + (blend / 100.0) * 12.0
    morph_delta = int(round((blend - 50.0) / 25.0))  # -2..+2 typical
    return float(adaptive_c), morph_delta


@dataclass(slots=True)
class MetalRecoveryConfig:
    segmentation_method: str = "none"
    sensitivity_0_100: int = 50
    sensitivity_token: str = "medium"
    morph_close_radius: int = 1
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
    if c == "мало_вершин":
        return (
            f"Топология: у контура {int(vertex_count)} вершин(ы), требуется не менее 3 "
            f"(параметр «мин. число точек» в упрощении: {int(config.min_points)})"
        )
    if c == "самопересечение_или_топология" or c.startswith("самопересечение"):
        return (
            "Топология: контур самопересекается, касается сам себя или имеет недопустимую конфигурацию рёбер "
            f"(проверка включена: check_contour_validity={bool(config.check_contour_validity)})"
        )
    if c == "некорректный_контур" or c == "":
        return (
            "Топология: контур не прошёл проверку целостности "
            f"(вершин: {int(vertex_count)}, проверка: {bool(config.check_contour_validity)})"
        )
    if c == "ширина":
        return (
            "Оценка ширины (маска + minAreaRect): "
            f"у полигона {width_px:.2f} px, в настройках минимум {float(config.min_width_px):.2f} px"
        )
    if c == "ширина_макс":
        cap_s = f"{float(config.max_width_px):.2f}" if config.max_width_px is not None else "—"
        return (
            "Оценка ширины (маска + minAreaRect): "
            f"у полигона {width_px:.2f} px, в настройках максимум {cap_s} px"
        )
    if c == "длина":
        return (
            "Длина по minAreaRect (большая сторона): "
            f"у полигона {length_px:.2f} px, в настройках минимум {float(config.min_length_px):.2f} px"
        )
    if c == "площадь":
        return f"Площадь контура: у полигона {area:.1f} px², в настройках минимум {float(config.min_area):.1f} px²"
    if c == "площадь_макс":
        cap_s = f"{float(config.max_area):.1f}" if config.max_area is not None else "—"
        return f"Площадь контура: у полигона {area:.1f} px², в настройках максимум {cap_s} px²"
    if c == "периметр":
        return (
            "Периметр контура: "
            f"у полигона {perimeter:.1f} px, в настройках минимум {float(config.min_perimeter):.1f} px"
        )
    if c == "периметр_макс":
        cap_s = f"{float(config.max_perimeter):.1f}" if config.max_perimeter is not None else "—"
        return (
            "Периметр контура: "
            f"у полигона {perimeter:.1f} px, в настройках максимум {cap_s} px"
        )
    if c == "прямолинейность":
        return (
            "Прямолинейность (2·max(стороны minAreaRect) / периметр): "
            f"у полигона {straightness:.3f}, в настройках минимум {float(config.min_straightness):.3f}"
        )
    if c == "углы":
        return (
            f"Углы полигона: режим «{mode}», допуск ±{tol:.1f}°; "
            f"максимальное отклонение от допустимых направлений {max_angle_deviation:.2f}° "
            f"(ожидалось не больше {tol:.1f}°)"
        )
    if c == "т_соединение_запрещено":
        return (
            "Т-образное ветвление (выпуклые дефекты контура): "
            f"оценка {int(t_branch_score)} > 2 при выключенном параметре «разрешить Т-соединения» "
            f"(allow_t_junction={bool(config.allow_t_junction)})"
        )
    if c == "граница":
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


def _segment_bright_metal(gray: np.ndarray, config: MetalRecoveryConfig) -> np.ndarray:
    if gray.size == 0:
        return np.zeros_like(gray, dtype=np.uint8)
    method = _normalize_metal_segmentation_method(config.segmentation_method)
    ac_off, morph_delta = _sensitivity_offsets(config.sensitivity_0_100, config.sensitivity_token)
    close_r = max(1, int(config.morph_close_radius) + morph_delta)
    open_r = max(0, int(config.morph_open_radius))

    if method == "adaptive":
        block = max(11, min(99, int(max(gray.shape) // 12) * 2 + 1))
        mask = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, int(round(ac_off))
        )
    else:
        _t, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        tok = _normalize_metal_sensitivity_token(config.sensitivity_token)
        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        if tok == "low":
            mask = cv2.erode(mask, k3)
        elif tok == "high":
            mask = cv2.dilate(mask, k3)
        adj = morph_delta
        if adj > 0:
            mask = cv2.dilate(mask, k3, iterations=min(2, adj))
        elif adj < 0:
            mask = cv2.erode(mask, k3, iterations=min(2, -adj))

    ks = max(3, close_r * 2 + 1)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
    if open_r > 0:
        ko = max(3, open_r * 2 + 1)
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ko, ko))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)
    return ensure_binary_mask(mask)


def _count_binary_components(mask_u8: np.ndarray) -> int:
    if mask_u8 is None or mask_u8.size == 0:
        return 0
    m = (np.asarray(mask_u8) > 0).astype(np.uint8)
    n, _, _, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    return max(0, int(n) - 1)


def _watershed_split_touching_conductors(
    ribbon_u8: np.ndarray,
    guide_bgr: np.ndarray,
    *,
    dist_peak_frac: float,
) -> np.ndarray:
    """Break weak bridges between separate conductors using DT seeds + watershed."""
    m = (np.asarray(ribbon_u8) > 0).astype(np.uint8) * 255
    if int(cv2.countNonZero(m)) < 80:
        return ribbon_u8
    before_cc = _count_binary_components(m)
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    dmax = float(np.max(dist))
    if dmax < 2.2:
        return ribbon_u8
    frac = max(0.22, min(0.55, float(dist_peak_frac)))
    _, sure_fg = cv2.threshold(dist, frac * dmax, 255, cv2.THRESH_BINARY)
    sure_fg = sure_fg.astype(np.uint8)
    if int(cv2.countNonZero(sure_fg)) < 16:
        return ribbon_u8
    unknown = cv2.subtract(m, sure_fg)
    n_mark, markers = cv2.connectedComponents(sure_fg)
    if n_mark < 3:
        return ribbon_u8
    markers = markers.astype(np.int32) + 1
    markers[unknown == 255] = 0
    markers[m == 0] = 0
    img = np.asarray(guide_bgr)
    if img.ndim == 2:
        img3 = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img3 = img[:, :, :3].astype(np.uint8)
    else:
        img3 = img.astype(np.uint8)
    ws = markers.copy()
    raise_if_preview_cancelled()
    cv2.watershed(img3, ws)
    raise_if_preview_cancelled()
    out = np.zeros_like(m, dtype=np.uint8)
    for lbl in np.unique(ws):
        li = int(lbl)
        if li <= 1 or li == -1:
            continue
        out[ws == li] = 255
    if int(cv2.countNonZero(out)) < int(0.22 * float(cv2.countNonZero(m))):
        return ribbon_u8
    after_cc = _count_binary_components(out)
    if after_cc <= before_cc:
        return ribbon_u8
    if after_cc > max(before_cc + 8, before_cc * 3):
        return ribbon_u8
    return ensure_binary_mask(out)


def _grayscale_edge_conductor_mask(gray: np.ndarray, config: MetalRecoveryConfig) -> tuple[np.ndarray, np.ndarray]:
    """Closed regions from grayscale edges + local morphology (no global intensity threshold).

    Pipeline: Gaussian blur → white-hat emphasis → Canny(L2) → dilate → morph close/open.
    Returns ``(filled_region_mask_uint8, canny_edges_uint8)``.
    """
    if gray.size == 0:
        z = np.zeros_like(gray)
        return z, z
    gh, gw = int(gray.shape[0]), int(gray.shape[1])
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    ksz = max(5, min(21, int(2.25 * max(2.0, float(config.min_width_px))) | 1))
    k_th = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
    tophat = cv2.subtract(blurred, cv2.morphologyEx(blurred, cv2.MORPH_OPEN, k_th))
    enhanced = cv2.addWeighted(blurred, 0.58, tophat, 0.42, 0)
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)

    med = float(np.median(enhanced))
    if med < 1.0:
        med = float(np.mean(enhanced)) + 1.0
    sigma_use = 0.33
    lower = float((1.0 - sigma_use) * med)
    upper = float((1.0 + sigma_use) * med)
    adj = (float(config.sensitivity_0_100) - 50.0) / 50.0
    lower *= max(0.35, 1.0 - 0.28 * adj)
    upper *= max(0.55, 1.0 - 0.18 * adj)
    tok = _normalize_metal_sensitivity_token(config.sensitivity_token)
    if tok == "high":
        lower *= 0.88
        upper *= 0.92
    elif tok == "low":
        lower *= 1.12
        upper = min(255.0, upper * 1.08)

    lo = int(max(1, min(254, round(lower))))
    hi = int(max(lo + 4, min(255, round(upper))))
    edges = cv2.Canny(enhanced, lo, hi, L2gradient=True)
    raise_if_preview_cancelled()

    d3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thick = cv2.dilate(edges, d3, iterations=1)
    _, morph_delta = _sensitivity_offsets(config.sensitivity_0_100, config.sensitivity_token)
    close_r = max(1, int(config.morph_close_radius) + morph_delta)
    rk = max(3, min(25, close_r * 2 + 1))
    rw = max(rk, min(25, int(max(3, round(float(config.min_width_px)))) | 1))
    # Merge only the inner/outer Canny pair of one trace; cap so we do not close real gaps between neighbours.
    inner_merge = int(max(5, min(15, 2 * int(max(2, round(0.42 * float(config.min_width_px)))) + 1)))
    cap = int(getattr(config, "edge_close_cap_px", 9) or 9)
    cap = max(5, min(21, cap | 1))
    rw = min(rw, inner_merge, cap)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rw, rw))
    ribbon = cv2.morphologyEx(thick, cv2.MORPH_CLOSE, k_close)
    raise_if_preview_cancelled()
    open_r = max(0, int(config.morph_open_radius))
    if open_r > 0:
        ko = max(3, open_r * 2 + 1)
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ko, ko))
        ribbon = cv2.morphologyEx(ribbon, cv2.MORPH_OPEN, k_open)
    if bool(getattr(config, "edge_watershed_split", True)):
        cap = getattr(config, "edge_watershed_max_pixels", None)
        run_ws = cap is None or int(cap) <= 0 or gh * gw <= int(cap)
        if run_ws:
            guide = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            ribbon = _watershed_split_touching_conductors(
                ribbon,
                guide,
                dist_peak_frac=float(getattr(config, "edge_watershed_dist_peak_frac", 0.38) or 0.38),
            )
    return ensure_binary_mask(ribbon), edges


def build_metal_extraction_mask(
    gray: np.ndarray, config: MetalRecoveryConfig
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Build the working binary mask for conductor contour extraction (all segmentation modes)."""
    raise_if_preview_cancelled()
    mode = _normalize_metal_extraction_mode(config.segmentation_method)
    extra: dict[str, np.ndarray] = {}
    if mode == "none":
        mask, e = _grayscale_edge_conductor_mask(gray, config)
        extra["metal_edge_canny"] = e
        return mask, extra
    if mode == "hybrid":
        cfg_thr = replace(config, segmentation_method="otsu")
        mask_bin = _segment_bright_metal(gray, cfg_thr)
        raise_if_preview_cancelled()
        mask_edge, e = _grayscale_edge_conductor_mask(gray, config)
        extra["metal_edge_canny"] = e
        extra["metal_threshold_mask"] = mask_bin
        mask = cv2.bitwise_or(mask_bin, mask_edge)
        return ensure_binary_mask(mask), extra
    mask = _segment_bright_metal(gray, config)
    return mask, extra


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


def _contour_to_polygon(
    contour: np.ndarray,
    *,
    epsilon: float,
    approx_enabled: bool,
    min_angle_deg: float,
) -> list[tuple[float, float]]:
    if not approx_enabled or epsilon <= 0:
        pts = [(float(p[0][0]), float(p[0][1])) for p in contour]
    else:
        arc = cv2.arcLength(contour, True)
        eps = max(0.1, float(epsilon))
        simplified = cv2.approxPolyDP(contour, eps, True)
        pts = [(float(p[0][0]), float(p[0][1])) for p in simplified]
    if len(pts) >= 3 and min_angle_deg > 1e-6:
        # drop vertices with too sharp turns (noise spikes)
        filtered: list[tuple[float, float]] = []
        n = len(pts)
        for i in range(n):
            p0 = pts[(i - 1) % n]
            p1 = pts[i]
            p2 = pts[(i + 1) % n]
            a1 = atan2(p0[1] - p1[1], p0[0] - p1[0])
            a2 = atan2(p2[1] - p1[1], p2[0] - p1[0])
            ang = degrees(abs(a2 - a1))
            if ang > 180.0:
                ang = 360.0 - ang
            interior = abs(180.0 - ang)
            if interior >= float(min_angle_deg) - 1e-3:
                filtered.append(p1)
        if len(filtered) >= 3:
            pts = filtered
    return pts


def _polygon_raster_iou(a: PolygonData, b: PolygonData, shape_hw: tuple[int, int]) -> float:
    h, w = shape_hw
    if h <= 0 or w <= 0:
        return 0.0
    m1 = np.zeros((h, w), dtype=np.uint8)
    m2 = np.zeros((h, w), dtype=np.uint8)
    ca = np.array([(int(x), int(y)) for x, y in a.points], dtype=np.int32).reshape(-1, 1, 2)
    cb = np.array([(int(x), int(y)) for x, y in b.points], dtype=np.int32).reshape(-1, 1, 2)
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
        if any(_polygon_raster_iou(p, k, shape_hw) >= iou_threshold for k in kept):
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
    for idx, contour in enumerate(raw_contours):
        raise_if_preview_cancelled()
        if contour is None or len(contour) < 3:
            continue
        if not config.retrieval_external_only and hierarchy.size and idx < hierarchy.shape[1]:
            parent = int(hierarchy[0][idx][3])
            if parent != -1:
                continue

        area = abs(float(cv2.contourArea(contour)))
        perimeter = float(cv2.arcLength(contour, True))
        if area < 1.0:
            continue

        width_px, _wm = estimate_effective_polygon_width_px(mask, contour)
        rect = cv2.minAreaRect(contour)
        rw, rh = float(rect[1][0]), float(rect[1][1])
        width_px = effective_conductor_width_px(width_px, rw, rh)
        length_px = float(max(rw, rh))
        straight = _straightness_metric(rw, rh, perimeter)
        b_touch = _border_touch(contour, width=w, height=h)

        raw_pts = [(float(contour[i][0][0]), float(contour[i][0][1])) for i in range(len(contour))]
        points = _contour_to_polygon(
            contour,
            epsilon=config.epsilon_simplify,
            approx_enabled=config.approximation_enabled,
            min_angle_deg=config.min_polygon_angle_deg,
        )
        topo_pts = points if len(points) >= 3 else raw_pts
        valid, topo_reason = _valid_topology(topo_pts, enabled=config.check_contour_validity)
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

        angle_soft = bool(dev > config.angle_tolerance_deg * 0.65)
        straight_soft = bool(straight < config.min_straightness - 1e-4)
        soft_suspicious = angle_soft or straight_soft

        reject_case = ""
        t_branch_score = 0
        if not valid:
            reject_case = topo_reason or "некорректный_контур"
        elif n_vertices < max(3, int(config.min_points)):
            reject_case = "мало_вершин"
        elif width_px + 1e-3 < config.min_width_px:
            reject_case = "ширина"
        elif config.max_width_px is not None and width_px > float(config.max_width_px) + 1e-3:
            reject_case = "ширина_макс"
        elif length_px + 1e-3 < config.min_length_px:
            reject_case = "длина"
        elif area + 1e-3 < config.min_area:
            reject_case = "площадь"
        elif config.max_area is not None and area > float(config.max_area) + 1e-3:
            reject_case = "площадь_макс"
        elif perimeter + 1e-3 < config.min_perimeter:
            reject_case = "периметр"
        elif config.max_perimeter is not None and perimeter > float(config.max_perimeter) + 1e-3:
            reject_case = "периметр_макс"
        elif not config.allow_t_junction:
            t_branch_score = _convex_branching_score(contour)
            if t_branch_score > 2:
                reject_case = "т_соединение_запрещено"
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
    dbg.update(pre_dbg)
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

    for new_id, poly in enumerate(accepted, start=1):
        poly.id = new_id

    return MetalDetectionResult(
        accepted=accepted,
        rejected=rejected,
        suspicious=suspicious,
        border=border,
        debug_images=dbg,
        params_snapshot=config.to_snapshot(),
        wide_gradient_overlays=wide_ov,
    )
