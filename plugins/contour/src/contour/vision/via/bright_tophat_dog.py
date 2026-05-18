"""Bright via detector: two-stage (candidates + scoring), classical OpenCV/NumPy only.

Independent from Qt; tune and test without UI changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import pi
from typing import Any

import cv2
import numpy as np

from ...application.preview_cancellation import raise_if_preview_cancelled
from ..io_normalize import to_gray_u8

# BGR for overlay: green = accepted, yellow = soft, red = hard
_COLOR_ACCEPT = (0, 220, 0)
_COLOR_SOFT = (0, 220, 255)
_COLOR_HARD = (0, 0, 255)

# Flat or noisy response plateaus can mark almost every pixel as a "local maximum";
# combined with a slow all-vs-all distance check, that stalls the UI. Cap and index.
_MAX_LOCAL_MAXIMA_PEAKS = 8192
_MAX_STAGE1_RAW_CANDIDATES = 10_000


def _clip01(value: float) -> float:
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return float(value)


class _CenterDistanceIndex:
    """Bucketing for O(1) amortized "any center within min_dist" checks on a 2D grid."""

    __slots__ = ("_buckets", "_cell", "_min_dist")

    def __init__(self, min_dist: float) -> None:
        self._min_dist = float(max(0.0, min_dist))
        self._cell = max(1.0, self._min_dist)
        self._buckets: dict[tuple[int, int], list[tuple[float, float]]] = {}

    def is_close(self, cx: float, cy: float) -> bool:
        if self._min_dist <= 0.0:
            return False
        c = self._cell
        ix, iy = int(cx // c), int(cy // c)
        md = self._min_dist
        for dxi in (-1, 0, 1):
            for dyi in (-1, 0, 1):
                for ox, oy in self._buckets.get((ix + dxi, iy + dyi), ()):
                    if float(np.hypot(cx - ox, cy - oy)) < md:
                        return True
        return False

    def add(self, cx: float, cy: float) -> None:
        c = self._cell
        self._buckets.setdefault((int(cx // c), int(cy // c)), []).append((cx, cy))


@dataclass(frozen=True, slots=True)
class BrightViaDetectorConfig:
    diameter_min: int = 6
    diameter_max: int = 8
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: int = 8
    median_blur_kernel: int = 3
    tophat_kernel_size: int = 11
    dog_sigma_small: float = 0.8
    dog_sigma_large: float = 2.0
    threshold_percentile: float = 99.0
    mask_combine_mode: str = "OR"
    min_area_factor: float = 0.45
    max_area_factor: float = 1.8
    min_circularity: float = 0.30
    min_aspect: float = 0.45
    max_aspect: float = 2.2
    bright_center_min_score: float = 6.0
    metal_constraint_mode: str = "soft"
    use_metal_mask: bool = True
    metal_fraction_min: float = 0.3
    max_radial_asymmetry: float = 18.0
    max_edge_likeness: float = 35.0
    max_line_likeness: float = 65.0
    nms_distance: int = 5
    min_final_score: float = 45.0
    show_rejected_candidates: bool = True
    hard_reject_on_asymmetry: bool = False
    hard_reject_on_edge: bool = False
    hard_reject_on_line: bool = False

    @classmethod
    def from_legacy_settings(cls, settings: Any) -> BrightViaDetectorConfig:
        return cls(
            diameter_min=int(getattr(settings, "bright_via_diameter_min", cls.diameter_min)),
            diameter_max=int(getattr(settings, "bright_via_diameter_max", cls.diameter_max)),
            clahe_clip_limit=float(getattr(settings, "bright_via_clahe_clip_limit", cls.clahe_clip_limit)),
            clahe_tile_grid_size=int(getattr(settings, "bright_via_clahe_tile_grid_size", cls.clahe_tile_grid_size)),
            median_blur_kernel=int(getattr(settings, "bright_via_median_blur_kernel", cls.median_blur_kernel)),
            tophat_kernel_size=int(getattr(settings, "bright_via_tophat_kernel_size", cls.tophat_kernel_size)),
            dog_sigma_small=float(getattr(settings, "bright_via_dog_sigma_small", cls.dog_sigma_small)),
            dog_sigma_large=float(getattr(settings, "bright_via_dog_sigma_large", cls.dog_sigma_large)),
            threshold_percentile=float(
                getattr(settings, "bright_via_threshold_percentile", cls.threshold_percentile)
            ),
            mask_combine_mode=str(getattr(settings, "bright_via_mask_combine_mode", cls.mask_combine_mode)),
            min_area_factor=float(getattr(settings, "bright_via_min_area_factor", cls.min_area_factor)),
            max_area_factor=float(getattr(settings, "bright_via_max_area_factor", cls.max_area_factor)),
            min_circularity=float(getattr(settings, "bright_via_min_circularity", cls.min_circularity)),
            min_aspect=float(getattr(settings, "bright_via_min_aspect", cls.min_aspect)),
            max_aspect=float(getattr(settings, "bright_via_max_aspect", cls.max_aspect)),
            bright_center_min_score=float(
                getattr(settings, "bright_via_bright_center_min_score", cls.bright_center_min_score)
            ),
            metal_constraint_mode=_normalize_metal_constraint_mode(
                getattr(
                    settings,
                    "bright_via_metal_constraint_mode",
                    "soft" if bool(getattr(settings, "bright_via_use_metal_mask", cls.use_metal_mask)) else "disabled",
                )
            ),
            use_metal_mask=bool(getattr(settings, "bright_via_use_metal_mask", cls.use_metal_mask)),
            metal_fraction_min=float(getattr(settings, "bright_via_metal_fraction_min", cls.metal_fraction_min)),
            max_radial_asymmetry=float(
                getattr(settings, "bright_via_max_radial_asymmetry", cls.max_radial_asymmetry)
            ),
            max_edge_likeness=float(getattr(settings, "bright_via_max_edge_likeness", cls.max_edge_likeness)),
            max_line_likeness=float(getattr(settings, "bright_via_max_line_likeness", cls.max_line_likeness)),
            nms_distance=int(getattr(settings, "bright_via_nms_distance", cls.nms_distance)),
            min_final_score=float(getattr(settings, "bright_via_min_final_score", cls.min_final_score)),
            show_rejected_candidates=bool(
                getattr(settings, "bright_via_show_rejected", cls.show_rejected_candidates)
            ),
            hard_reject_on_asymmetry=bool(
                getattr(settings, "bright_via_hard_reject_on_asymmetry", cls.hard_reject_on_asymmetry)
            ),
            hard_reject_on_edge=bool(
                getattr(settings, "bright_via_hard_reject_on_edge", cls.hard_reject_on_edge)
            ),
            hard_reject_on_line=bool(
                getattr(settings, "bright_via_hard_reject_on_line", cls.hard_reject_on_line)
            ),
        ).validated()

    def validated(self) -> BrightViaDetectorConfig:
        errors: list[str] = []
        if self.diameter_min <= 0 or self.diameter_max <= 0:
            errors.append("diameters must be positive")
        if self.diameter_min > self.diameter_max:
            errors.append("diameter_min must be <= diameter_max")
        if self.median_blur_kernel < 1 or self.median_blur_kernel % 2 == 0:
            errors.append("median_blur_kernel must be odd and >= 1")
        if self.tophat_kernel_size < 3 or self.tophat_kernel_size % 2 == 0:
            errors.append("tophat_kernel_size must be odd and >= 3")
        if self.clahe_tile_grid_size < 1:
            errors.append("clahe_tile_grid_size must be >= 1")
        if self.clahe_clip_limit <= 0.0:
            errors.append("clahe_clip_limit must be positive")
        if self.dog_sigma_small <= 0.0 or self.dog_sigma_large <= 0.0:
            errors.append("DoG sigmas must be positive")
        if self.dog_sigma_small >= self.dog_sigma_large:
            errors.append("dogSigmaSmall must be < dogSigmaLarge")
        if not 90.0 <= self.threshold_percentile <= 99.9:
            errors.append("thresholdPercentile must be in range 90-99.9")
        if str(self.mask_combine_mode).upper() not in {"OR", "AND"}:
            errors.append("maskCombineMode must be OR or AND")
        if _normalize_metal_constraint_mode(self.metal_constraint_mode) not in {"disabled", "soft", "strict"}:
            errors.append("metalConstraintMode must be disabled, soft, or strict")
        for name, value in (
            ("min_area_factor", self.min_area_factor),
            ("max_area_factor", self.max_area_factor),
            ("min_circularity", self.min_circularity),
            ("min_aspect", self.min_aspect),
            ("max_aspect", self.max_aspect),
            ("metal_fraction_min", self.metal_fraction_min),
            ("max_radial_asymmetry", self.max_radial_asymmetry),
            ("max_edge_likeness", self.max_edge_likeness),
            ("max_line_likeness", self.max_line_likeness),
        ):
            if float(value) < 0.0:
                errors.append(f"{name} must be non-negative")
        if not 0.0 <= float(self.min_final_score) <= 100.0:
            errors.append("minFinalScore must be 0-100")
        if self.min_area_factor <= 0.0 or self.max_area_factor <= 0.0:
            errors.append("area factors must be positive")
        if self.min_aspect <= 0.0 or self.max_aspect <= 0.0:
            errors.append("aspect thresholds must be positive")
        if self.min_aspect > self.max_aspect:
            errors.append("minAspect must be <= maxAspect")
        if self.nms_distance < 0:
            errors.append("nmsDistance must be >= 0")
        if errors:
            raise ValueError("; ".join(errors))
        return self


@dataclass(frozen=True, slots=True)
class BrightViaDetection:
    center: tuple[float, float]
    bbox: tuple[int, int, int, int]
    area: float
    circularity: float
    aspect: float
    brightness_score: float
    local_peak_score: float
    tophat_response: float
    dog_response: float
    metal_fraction: float
    final_score: float
    radial_asymmetry: float = 0.0
    edge_likeness: float = 0.0
    line_likeness: float = 0.0
    distance_to_edge: float = 0.0
    status: str = "accepted"
    hard_reason: str = ""


@dataclass(slots=True)
class BrightViaDetectionResult:
    """detections: accepted only (exported / mask). candidates: all after NMS + scoring."""

    detections: list[BrightViaDetection] = field(default_factory=list)
    candidates: list[BrightViaDetection] = field(default_factory=list)
    debug_images: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass(slots=True)
class BrightViaPreparedCandidates:
    gray: np.ndarray
    processed: np.ndarray
    tophat: np.ndarray
    dog: np.ndarray
    tophat_mask: np.ndarray
    dog_mask: np.ndarray
    via_mask: np.ndarray
    candidate_mask: np.ndarray
    metal_mask: np.ndarray
    edge_distance: np.ndarray
    raw_candidates: list["_RawStage1"]
    min_area: float
    max_area: float
    nominal_diameter: float
    nominal_radius: float
    metal_mode: str


@dataclass(slots=True)
class _CandidateDebugMaps:
    radial_symmetry: np.ndarray
    edge_likeness: np.ndarray
    line_likeness: np.ndarray

    @classmethod
    def empty(cls, shape: tuple[int, ...]) -> _CandidateDebugMaps:
        return cls(
            radial_symmetry=np.zeros(shape, dtype=np.uint8),
            edge_likeness=np.zeros(shape, dtype=np.uint8),
            line_likeness=np.zeros(shape, dtype=np.uint8),
        )


@dataclass(slots=True)
class _RawStage1:
    center: tuple[float, float]
    contour: np.ndarray | None
    bbox: tuple[int, int, int, int]
    area: float
    circularity: float
    aspect: float
    prelim: float
    source: str


def detect_bright_vias(image: np.ndarray, config: BrightViaDetectorConfig) -> BrightViaDetectionResult:
    cfg = config.validated()
    prepared = prepare_bright_via_candidates(image, cfg)
    return score_bright_via_candidates(prepared, cfg)


def prepare_bright_via_candidates(image: np.ndarray, config: BrightViaDetectorConfig) -> BrightViaPreparedCandidates:
    cfg = config.validated()
    gray = _normalize_u8(to_gray_u8(image))
    if gray.size == 0:
        return BrightViaPreparedCandidates(
            gray=gray,
            processed=gray,
            tophat=gray,
            dog=gray,
            tophat_mask=gray,
            dog_mask=gray,
            via_mask=gray,
            candidate_mask=gray,
            metal_mask=gray,
            edge_distance=gray.astype(np.float32),
            raw_candidates=[],
            min_area=0.0,
            max_area=0.0,
            nominal_diameter=0.0,
            nominal_radius=0.0,
            metal_mode="disabled",
        )
    raise_if_preview_cancelled()
    processed = _preprocess(gray, cfg)
    raise_if_preview_cancelled()
    tophat = _white_tophat(processed, cfg.tophat_kernel_size)
    raise_if_preview_cancelled()
    dog = _dog_response(processed, cfg.dog_sigma_small, cfg.dog_sigma_large)
    raise_if_preview_cancelled()
    tophat_norm = _normalize_response(tophat)
    dog_norm = _normalize_response(dog)
    tophat_mask = _percentile_mask(tophat_norm, cfg.threshold_percentile)
    dog_mask = _percentile_mask(dog_norm, cfg.threshold_percentile)
    absolute_bright_mask = _percentile_mask(gray, 98.8)
    if cfg.mask_combine_mode.upper() == "AND":
        via_mask = cv2.bitwise_and(tophat_mask, dog_mask)
    else:
        via_mask = cv2.bitwise_or(tophat_mask, dog_mask)
    via_mask = cv2.bitwise_or(via_mask, absolute_bright_mask)
    raise_if_preview_cancelled()
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    via_mask = cv2.morphologyEx(via_mask, cv2.MORPH_OPEN, open_kernel)
    candidate_soup = cv2.bitwise_or(via_mask, cv2.bitwise_or(tophat_mask, dog_mask))
    metal_mode = "disabled" if not cfg.use_metal_mask else _normalize_metal_constraint_mode(cfg.metal_constraint_mode)
    metal_mask = _metal_mask(processed) if metal_mode != "disabled" else np.zeros_like(gray, dtype=np.uint8)
    edge_distance = _edge_distance_map(processed)
    min_area = pi * (float(cfg.diameter_min) * 0.5) ** 2 * float(cfg.min_area_factor)
    max_area = pi * (float(cfg.diameter_max) * 0.5) ** 2 * float(cfg.max_area_factor)
    nominal_diameter = (float(cfg.diameter_min) + float(cfg.diameter_max)) * 0.5
    nominal_radius = max(1.0, nominal_diameter * 0.5)
    raise_if_preview_cancelled()
    raws = _stage1_raw_candidates(
        tophat_u8=tophat_norm,
        dog_u8=dog_norm,
        tophat_mask=tophat_mask,
        dog_mask=dog_mask,
        via_mask=via_mask,
        cfg=cfg,
        min_area=min_area,
        max_area=max_area,
    )
    raise_if_preview_cancelled()
    raws = _raw_nms(raws, float(cfg.nms_distance))
    return BrightViaPreparedCandidates(
        gray=gray,
        processed=processed,
        tophat=tophat_norm,
        dog=dog_norm,
        tophat_mask=tophat_mask,
        dog_mask=dog_mask,
        via_mask=via_mask,
        candidate_mask=candidate_soup,
        metal_mask=metal_mask,
        edge_distance=edge_distance,
        raw_candidates=raws,
        min_area=min_area,
        max_area=max_area,
        nominal_diameter=nominal_diameter,
        nominal_radius=nominal_radius,
        metal_mode=metal_mode,
    )


def score_bright_via_candidates(
    prepared: BrightViaPreparedCandidates, config: BrightViaDetectorConfig
) -> BrightViaDetectionResult:
    cfg = config.validated()
    if prepared.gray.size == 0:
        return BrightViaDetectionResult()
    debug = _CandidateDebugMaps.empty(prepared.gray.shape[:2])
    gray_f32 = prepared.gray.astype(np.float32, copy=False)
    processed_f32 = prepared.processed.astype(np.float32, copy=False)
    tophat_f32 = prepared.tophat.astype(np.float32, copy=False)
    dog_f32 = prepared.dog.astype(np.float32, copy=False)
    raise_if_preview_cancelled()

    candidates_scored: list[BrightViaDetection] = []
    for i, raw in enumerate(prepared.raw_candidates):
        if i & 31 == 0:
            raise_if_preview_cancelled()
        det = _score_one_candidate(
            raw=raw,
            gray=prepared.gray,
            gray_f32=gray_f32,
            processed=prepared.processed,
            processed_f32=processed_f32,
            tophat=prepared.tophat,
            tophat_f32=tophat_f32,
            dog=prepared.dog,
            dog_f32=dog_f32,
            metal_mask=prepared.metal_mask,
            edge_distance=prepared.edge_distance,
            cfg=cfg,
            metal_mode=prepared.metal_mode,
            nominal_diameter=prepared.nominal_diameter,
            nominal_radius=prepared.nominal_radius,
            min_area=prepared.min_area,
            max_area=prepared.max_area,
            debug=debug,
        )
        if det is not None:
            candidates_scored.append(det)
    raise_if_preview_cancelled()
    candidates_scored = suppress_close_points(candidates_scored, cfg.nms_distance)
    accepted = [c for c in candidates_scored if c.status == "accepted"]
    rejected = [c for c in candidates_scored if c.status != "accepted"]
    if len(rejected) > 1000:
        rejected = sorted(rejected, key=lambda item: item.final_score, reverse=True)[:1000]
    compact_candidates = accepted + rejected
    show_rejected = bool(cfg.show_rejected_candidates)
    if prepared.gray.shape[0] * prepared.gray.shape[1] >= 1_800_000:
        show_rejected = False
    overlay = _draw_overlay(prepared.gray, compact_candidates, cfg, show_rejected=show_rejected)

    return BrightViaDetectionResult(
        detections=accepted,
        candidates=compact_candidates,
        debug_images={
            "raw_gray": prepared.gray,
            "processed": prepared.processed,
            "tophat": prepared.tophat,
            "dog": prepared.dog,
            "tophat_mask": prepared.tophat_mask,
            "dog_mask": prepared.dog_mask,
            "via_mask": prepared.via_mask,
            "candidate_mask": prepared.candidate_mask,
            "metal_mask": prepared.metal_mask,
            "radial_symmetry": debug.radial_symmetry,
            "edge_likeness": debug.edge_likeness,
            "line_likeness": debug.line_likeness,
            "distance_to_edge": _normalize_response(prepared.edge_distance),
            "final_overlay": overlay,
        },
    )


def _stage1_raw_candidates(
    *,
    tophat_u8: np.ndarray,
    dog_u8: np.ndarray,
    tophat_mask: np.ndarray,
    dog_mask: np.ndarray,
    via_mask: np.ndarray,
    cfg: BrightViaDetectorConfig,
    min_area: float,
    max_area: float,
) -> list[_RawStage1]:
    raws: list[_RawStage1] = []
    pi_v = float(pi)
    tophat_f32 = tophat_u8.astype(np.float32, copy=False)
    dog_f32 = dog_u8.astype(np.float32, copy=False)
    for contour, tag in _contours_from_mask(via_mask, min_area, max_area):
        det = _raw_from_contour(contour, tophat_f32, dog_f32, tag, pi_v)
        if det is not None:
            raws.append(det)
    if len(raws) > _MAX_STAGE1_RAW_CANDIDATES:
        raws = sorted(raws, key=lambda z: z.prelim, reverse=True)[:_MAX_STAGE1_RAW_CANDIDATES]
    nms = max(2.0, float(cfg.nms_distance) * 0.5, float(cfg.diameter_min) * 0.35)
    near = _CenterDistanceIndex(nms)
    for r in raws:
        near.add(r.center[0], r.center[1])
    existing = {(_round2(r.center[0]), _round2(r.center[1])) for r in raws}
    for label, tmask, response in (
        ("tophat_peak", tophat_mask, tophat_u8),
        ("dog_peak", dog_mask, dog_u8),
    ):
        for cx, cy, val in _local_maxima_points(
            response, tmask, max_points=_MAX_LOCAL_MAXIMA_PEAKS
        ):
            key = (_round2(cx), _round2(cy))
            if key in existing:
                continue
            if near.is_close(cx, cy):
                continue
            raw = _raw_from_peak(cx, cy, val, response, label, cfg, min_area, max_area, pi_v)
            if raw is not None:
                raws.append(raw)
                near.add(cx, cy)
                existing.add(key)
    if len(raws) > _MAX_STAGE1_RAW_CANDIDATES:
        raws = sorted(raws, key=lambda z: z.prelim, reverse=True)[:_MAX_STAGE1_RAW_CANDIDATES]
    return raws


def _round2(x: float) -> float:
    return round(x * 2.0) * 0.5


def _contours_from_mask(
    via_mask: np.ndarray, min_area: float, max_area: float
) -> list[tuple[np.ndarray, str]]:
    contours, _h = cv2.findContours(via_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: list[tuple[np.ndarray, str]] = []
    for c in contours:
        a = float(cv2.contourArea(c))
        if a > max_area * 1.2 or a < 2.0:
            continue
        out.append((c, "contour"))
    return out


def _raw_from_contour(
    contour: np.ndarray,
    tophat_f32: np.ndarray,
    dog_f32: np.ndarray,
    tag: str,
    pi_v: float,
) -> _RawStage1 | None:
    area = float(cv2.contourArea(contour))
    per = float(cv2.arcLength(contour, True))
    if per <= 1e-6 or area <= 1.0:
        return None
    circ = float(4.0 * pi_v * area / (per * per))
    x, y, w, h = cv2.boundingRect(contour)
    aspect = float(w) / float(max(1, h))
    center = _contour_center_from_rect(contour, x, y, w, h)
    top = _disk_mean_f32(tophat_f32, center, max(1.0, (w + h) * 0.2))
    dog = _disk_mean_f32(dog_f32, center, max(1.0, (w + h) * 0.2))
    prelim = max(top, dog) * max(0.2, min(1.0, circ * 1.2))
    return _RawStage1(
        center=center,
        contour=contour,
        bbox=(int(x), int(y), int(w), int(h)),
        area=area,
        circularity=circ,
        aspect=aspect,
        prelim=float(prelim),
        source=tag,
    )


def _raw_from_peak(
    cx: float,
    cy: float,
    val: float,
    _response: np.ndarray,
    label: str,
    cfg: BrightViaDetectorConfig,
    min_area: float,
    max_area: float,
    pi_v: float,
) -> _RawStage1 | None:
    d = (float(cfg.diameter_min) + float(cfg.diameter_max)) * 0.5
    r = max(1.0, d * 0.5)
    side = max(2, int(round(d)))
    ix = int(round(cx)) - side // 2
    iy = int(round(cy)) - side // 2
    area = pi_v * r * r
    if not (min_area * 0.5 <= area <= max_area * 1.2):
        return None
    return _RawStage1(
        center=(float(cx), float(cy)),
        contour=None,
        bbox=(ix, iy, side, side),
        area=float(area),
        circularity=1.0,
        aspect=1.0,
        prelim=float(val),
        source=label,
    )


def _contour_center_from_rect(
    contour: np.ndarray, x_coord: int, y_coord: int, width: int, height: int
) -> tuple[float, float]:
    moments = cv2.moments(contour)
    if abs(float(moments.get("m00", 0.0))) > 1e-6:
        return float(moments["m10"] / moments["m00"]), float(moments["m01"] / moments["m00"])
    return float(x_coord + width * 0.5), float(y_coord + height * 0.5)


def _local_maxima_points(
    response_u8: np.ndarray, support: np.ndarray, *, max_points: int = _MAX_LOCAL_MAXIMA_PEAKS
) -> list[tuple[float, float, float]]:
    if response_u8.size == 0 or max_points <= 0:
        return []
    work = response_u8.astype(np.float32)
    work[support == 0] = 0.0
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dil = cv2.dilate(work, kernel)
    lm = (work >= dil) & (work > 0) & (support > 0)
    ys, xs = np.where(lm)
    if ys.size == 0:
        return []
    if ys.size > max_points:
        vals = work[ys, xs]
        pick = np.argpartition(-vals, max_points - 1)[:max_points]
        ys, xs = ys[pick], xs[pick]
    return [(float(x), float(y), float(work[y, x])) for y, x in zip(ys, xs, strict=True)]


def _raw_nms(raws: list[_RawStage1], min_dist: float) -> list[_RawStage1]:
    if not raws or min_dist <= 0:
        return raws
    kept: list[_RawStage1] = []
    idx = _CenterDistanceIndex(float(min_dist))
    for r in sorted(raws, key=lambda z: z.prelim, reverse=True):
        cx, cy = r.center
        if idx.is_close(cx, cy):
            continue
        kept.append(r)
        idx.add(cx, cy)
    return kept


def _score_one_candidate(
    *,
    raw: _RawStage1,
    gray: np.ndarray,
    gray_f32: np.ndarray,
    processed: np.ndarray,
    processed_f32: np.ndarray,
    tophat: np.ndarray,
    tophat_f32: np.ndarray,
    dog: np.ndarray,
    dog_f32: np.ndarray,
    metal_mask: np.ndarray,
    edge_distance: np.ndarray,
    cfg: BrightViaDetectorConfig,
    metal_mode: str,
    nominal_diameter: float,
    nominal_radius: float,
    min_area: float,
    max_area: float,
    debug: _CandidateDebugMaps,
) -> BrightViaDetection | None:
    x, y, w, h = raw.bbox
    center = raw.center
    area = raw.area
    aspect = float(w) / float(max(1, h)) if h > 0 else 999.0
    if raw.contour is not None:
        perimeter = float(cv2.arcLength(raw.contour, True))
        circularity = float(4.0 * pi * area / (perimeter * perimeter + 1e-6))
    else:
        circularity = 1.0
    if area < min_area * 0.99 or area > max_area * 1.01:
        return _hard(
            center,
            (x, y, w, h),
            area,
            circularity,
            aspect,
            "size",
        )
    if aspect < float(cfg.min_aspect) * 0.999 or aspect > float(cfg.max_aspect) * 1.001:
        return _hard(center, (x, y, w, h), area, circularity, aspect, "aspect")

    eff_d = max(nominal_diameter, float(np.sqrt(max(area, 1.0) * 4.0 / pi)))
    eff_r = max(1.0, eff_d * 0.5)
    center_mean = _center_mean_score_f32(gray_f32, center, eff_d)
    local_contrast = _bright_center_score_f32(processed_f32, center, eff_d)
    if center_mean < float(cfg.bright_center_min_score) * 0.999:
        return _hard(
            center,
            (x, y, w, h),
            area,
            circularity,
            aspect,
            f"low_center (raw={center_mean:.1f} min={cfg.bright_center_min_score})",
        )

    radial = _radial_symmetry_score_f32(processed_f32, center[0], center[1], eff_r)
    edge = _edge_likeness_score_f32(processed_f32, center[0], center[1], max(nominal_radius, eff_r))
    line = _line_likeness_score_f32(processed_f32, center[0], center[1], max(nominal_radius, eff_r))
    d_edge = _sample_distance(edge_distance, center)

    if cfg.hard_reject_on_asymmetry and float(cfg.max_radial_asymmetry) > 0 and radial > float(
        cfg.max_radial_asymmetry
    ):
        return _hard(center, (x, y, w, h), area, circularity, aspect, f"asymmetry>{cfg.max_radial_asymmetry}")
    if cfg.hard_reject_on_edge and float(cfg.max_edge_likeness) > 0 and edge > float(cfg.max_edge_likeness):
        return _hard(center, (x, y, w, h), area, circularity, aspect, f"edge>{cfg.max_edge_likeness}")
    if cfg.hard_reject_on_line and float(cfg.max_line_likeness) > 0 and line > float(cfg.max_line_likeness):
        return _hard(center, (x, y, w, h), area, circularity, aspect, f"line>{cfg.max_line_likeness}")

    metal = 1.0
    if metal_mode != "disabled":
        metal = mask_fraction(metal_mask, _expanded_bbox(center, nominal_diameter * 1.8, processed.shape))
    if metal_mode == "strict" and metal < float(cfg.metal_fraction_min) - 1e-4:
        return _hard(
            center,
            (x, y, w, h),
            area,
            circularity,
            aspect,
            f"metal<{cfg.metal_fraction_min}",
        )

    if debug is not None:
        _paint_score_disk(debug.radial_symmetry, center, max(2.0, eff_r), radial, 60.0)
        _paint_score_disk(debug.edge_likeness, center, max(2.0, eff_r), edge, 80.0)
        _paint_score_disk(debug.line_likeness, center, max(2.0, eff_r), line, 100.0)

    top = _disk_mean_f32(tophat_f32, center, max(1.0, nominal_diameter * 0.35))
    dogm = _disk_mean_f32(dog_f32, center, max(1.0, nominal_diameter * 0.35))
    local_peak = 0.5 * (float(top) + float(dogm))
    fscore = _composite_final_0_100(
        center_mean=center_mean,
        local_contrast=local_contrast,
        local_peak=local_peak,
        area=area,
        min_area=min_area,
        max_area=max_area,
        circularity=circularity,
        min_circularity=float(cfg.min_circularity),
        radial_asymmetry=radial,
        edge_likeness=edge,
        line_likeness=line,
        max_asym=cfg.max_radial_asymmetry,
        max_edge=cfg.max_edge_likeness,
        max_line=cfg.max_line_likeness,
        metal_fraction=metal,
        metal_mode=metal_mode,
        distance_to_edge=d_edge,
    )
    st = "accepted" if fscore >= float(cfg.min_final_score) else "soft_reject"
    return BrightViaDetection(
        center=(float(center[0]), float(center[1])),
        bbox=(int(x), int(y), int(w), int(h)),
        area=float(area),
        circularity=float(circularity),
        aspect=float(aspect if aspect < 1000 else 1.0),
        brightness_score=float(center_mean),
        local_peak_score=float(local_peak),
        tophat_response=float(top),
        dog_response=float(dogm),
        metal_fraction=float(metal),
        final_score=float(fscore),
        radial_asymmetry=float(radial),
        edge_likeness=float(edge),
        line_likeness=float(line),
        distance_to_edge=float(d_edge),
        status=st,
        hard_reason="",
    )


def _hard(
    center: tuple[float, float],
    bbox: tuple[int, int, int, int],
    area: float,
    circularity: float,
    aspect: float,
    reason: str,
) -> BrightViaDetection:
    return BrightViaDetection(
        center=(float(center[0]), float(center[1])),
        bbox=bbox,
        area=float(area),
        circularity=float(circularity),
        aspect=float(aspect),
        brightness_score=0.0,
        local_peak_score=0.0,
        tophat_response=0.0,
        dog_response=0.0,
        metal_fraction=0.0,
        final_score=0.0,
        status="hard_reject",
        hard_reason=reason,
    )


def _composite_final_0_100(
    *,
    center_mean: float,
    local_contrast: float,
    local_peak: float,
    area: float,
    min_area: float,
    max_area: float,
    circularity: float,
    min_circularity: float,
    radial_asymmetry: float,
    edge_likeness: float,
    line_likeness: float,
    max_asym: float,
    max_edge: float,
    max_line: float,
    metal_fraction: float,
    metal_mode: str,
    distance_to_edge: float,
) -> float:
    bright_abs_norm = _clip01(center_mean / 255.0)
    bright_contrast_norm = _clip01(local_contrast / 40.0)
    lp_norm = _clip01(local_peak / 255.0)
    mid = 0.5 * (min_area + max_area)
    span = max(max_area - min_area, 1.0)
    area_norm = 1.0 - float(min(1.0, abs(area - mid) / (span * 0.5 + 1e-6)))
    area_norm = _clip01(area_norm)
    circ_ref = max(min_circularity, 0.01)
    circ_norm = _clip01(circularity / circ_ref)
    m_as = max(float(max_asym), 1e-3)
    sym_norm = _clip01(1.0 - radial_asymmetry / m_as)
    m_e = max(float(max_edge), 1e-3)
    m_l = max(float(max_line), 1e-3)
    not_edge = _clip01(1.0 - edge_likeness / m_e)
    not_line = _clip01(1.0 - line_likeness / m_l)
    mbonus = 0.0
    if metal_mode == "soft":
        mbonus = 5.0 * _clip01(metal_fraction)
    elif metal_mode == "strict":
        mbonus = 3.0 * _clip01(metal_fraction)
    distb = min(5.0, max(0.0, float(distance_to_edge) * 0.25))
    core = (
        18.0 * bright_abs_norm
        + 10.0 * bright_contrast_norm
        + 20.0 * lp_norm
        + 10.0 * area_norm
        + 12.0 * circ_norm
        + 10.0 * sym_norm
        + 10.0 * not_edge
        + 5.0 * not_line
    )
    total = core + mbonus + distb
    if total <= 0.0:
        return 0.0
    if total >= 100.0:
        return 100.0
    return float(total)


def radial_symmetry_score(gray: np.ndarray, cx: float, cy: float, radius: float) -> float:
    data = _as_gray_f32(gray)
    return _radial_symmetry_score_f32(data, cx, cy, radius)


def _radial_symmetry_score_f32(data: np.ndarray, cx: float, cy: float, radius: float) -> float:
    if data.size == 0:
        return 0.0
    radius = max(1.0, float(radius))
    samples = [
        _bilinear_sample(data, cx + radius * dx, cy + radius * dy)
        for dx, dy in (
            (0.0, -1.0),
            (0.70710678, -0.70710678),
            (1.0, 0.0),
            (0.70710678, 0.70710678),
            (0.0, 1.0),
            (-0.70710678, 0.70710678),
            (-1.0, 0.0),
            (-0.70710678, -0.70710678),
        )
    ]
    return float(np.std(np.asarray(samples, dtype=np.float32)))


def edge_likeness_score(gray: np.ndarray, cx: float, cy: float, radius: float) -> float:
    data = _as_gray_f32(gray)
    return _edge_likeness_score_f32(data, cx, cy, radius)


def _edge_likeness_score_f32(data: np.ndarray, cx: float, cy: float, radius: float) -> float:
    if data.size == 0:
        return 0.0
    radius = max(1.0, float(radius))
    offsets = (
        ((0.0, -1.0), (0.0, 1.0)),
        ((1.0, 0.0), (-1.0, 0.0)),
        ((0.70710678, -0.70710678), (-0.70710678, 0.70710678)),
        ((0.70710678, 0.70710678), (-0.70710678, -0.70710678)),
    )
    differences = [
        abs(
            _bilinear_sample(data, cx + radius * first[0], cy + radius * first[1])
            - _bilinear_sample(data, cx + radius * second[0], cy + radius * second[1])
        )
        for first, second in offsets
    ]
    return float(max(differences, default=0.0))


def line_likeness_score(gray: np.ndarray, cx: float, cy: float, radius: float) -> float:
    data = _as_gray_f32(gray)
    return _line_likeness_score_f32(data, cx, cy, radius)


def _line_likeness_score_f32(data: np.ndarray, cx: float, cy: float, radius: float) -> float:
    patch = _local_patch(data, cx, cy, max(2.0, float(radius) * 1.8))
    if patch.size <= 4:
        return 0.0
    gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
    jxx = float(np.mean(gx * gx))
    jyy = float(np.mean(gy * gy))
    jxy = float(np.mean(gx * gy))
    trace = jxx + jyy
    if trace <= 1e-6:
        return 0.0
    delta = float(np.sqrt((jxx - jyy) * (jxx - jyy) + 4.0 * jxy * jxy))
    return float(np.clip(100.0 * delta / (trace + 1e-6), 0.0, 100.0))


def bright_center_score(gray: np.ndarray, center: tuple[float, float], diameter: float) -> float:
    data = _as_gray_f32(gray)
    return _bright_center_score_f32(data, center, diameter)


def _bright_center_score_f32(data: np.ndarray, center: tuple[float, float], diameter: float) -> float:
    if data.size == 0:
        return 0.0
    cx, cy = float(center[0]), float(center[1])
    radius = max(1.0, float(diameter) * 0.5)
    pad = max(2, round(radius * 2.2))
    left = max(0, round(cx) - pad)
    right = min(data.shape[1], round(cx) + pad + 1)
    top = max(0, round(cy) - pad)
    bottom = min(data.shape[0], round(cy) + pad + 1)
    if right <= left or bottom <= top:
        return 0.0
    patch = data[top:bottom, left:right]
    yy = (np.arange(top, bottom, dtype=np.float32) - cy).reshape(-1, 1)
    xx = (np.arange(left, right, dtype=np.float32) - cx).reshape(1, -1)
    dist2 = xx * xx + yy * yy
    center_mask = dist2 <= max(1.0, radius * 0.45) ** 2
    ring_mask = (dist2 >= max(1.0, radius * 0.85) ** 2) & (dist2 <= max(2.0, radius * 1.65) ** 2)
    if not np.any(center_mask) or not np.any(ring_mask):
        return 0.0
    return float(patch[center_mask].mean() - patch[ring_mask].mean())


def center_mean_score(gray: np.ndarray, center: tuple[float, float], diameter: float) -> float:
    data = _as_gray_f32(gray)
    return _center_mean_score_f32(data, center, diameter)


def _center_mean_score_f32(data: np.ndarray, center: tuple[float, float], diameter: float) -> float:
    if data.size == 0:
        return 0.0
    radius = max(1.0, float(diameter) * 0.22)
    return _disk_mean_f32(data, center, radius)


def mask_fraction(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> float:
    data = to_gray_u8(mask)
    if data.size == 0:
        return 0.0
    x_coord, y_coord, width, height = bbox
    left = max(0, int(x_coord))
    top = max(0, int(y_coord))
    right = min(data.shape[1], left + max(0, int(width)))
    bottom = min(data.shape[0], top + max(0, int(height)))
    if right <= left or bottom <= top:
        return 0.0
    patch = data[top:bottom, left:right]
    return float(cv2.countNonZero(patch)) / float(patch.size)


def suppress_close_points(
    detections: list[BrightViaDetection], distance: int
) -> list[BrightViaDetection]:
    if distance <= 0 or not detections:
        return sorted(detections, key=lambda item: (item.center[1], item.center[0]))
    kept: list[BrightViaDetection] = []
    min_dist = float(distance)
    idx = _CenterDistanceIndex(min_dist)
    for candidate in sorted(detections, key=lambda item: item.final_score, reverse=True):
        cx, cy = candidate.center
        if idx.is_close(cx, cy):
            continue
        kept.append(candidate)
        idx.add(cx, cy)
    return sorted(kept, key=lambda item: (item.center[1], item.center[0]))


def _preprocess(gray: np.ndarray, cfg: BrightViaDetectorConfig) -> np.ndarray:
    tile = max(1, int(cfg.clahe_tile_grid_size))
    processed = cv2.createCLAHE(clipLimit=float(cfg.clahe_clip_limit), tileGridSize=(tile, tile)).apply(gray)
    if cfg.median_blur_kernel > 1:
        processed = cv2.medianBlur(processed, int(cfg.median_blur_kernel))
    return processed


def _white_tophat(gray: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(kernel_size), int(kernel_size)))
    return cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)


def _dog_response(gray: np.ndarray, sigma_small: float, sigma_large: float) -> np.ndarray:
    data = gray.astype(np.float32)
    small = cv2.GaussianBlur(data, (0, 0), float(sigma_small))
    large = cv2.GaussianBlur(data, (0, 0), float(sigma_large))
    return np.maximum(small - large, 0.0)


def _percentile_mask(response: np.ndarray, percentile: float) -> np.ndarray:
    if response.size == 0:
        return np.zeros_like(response, dtype=np.uint8)
    threshold = float(np.percentile(response, float(percentile)))
    return np.where(response >= threshold, 255, 0).astype(np.uint8)


def _metal_mask(gray: np.ndarray) -> np.ndarray:
    if gray.size == 0:
        return np.zeros_like(gray, dtype=np.uint8)
    _t, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def _edge_distance_map(gray: np.ndarray) -> np.ndarray:
    if gray.size == 0:
        return np.zeros_like(gray, dtype=np.float32)
    blur = cv2.GaussianBlur(to_gray_u8(gray), (3, 3), 0.0)
    median = float(np.median(blur))
    lower = int(max(0, 0.66 * median))
    upper = int(min(255, max(lower + 1, 1.33 * median)))
    edges = cv2.Canny(blur, lower, upper)
    return cv2.distanceTransform(cv2.bitwise_not(edges), cv2.DIST_L2, 3)


def _expanded_bbox(center: tuple[float, float], size: float, shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    half = max(1, round(float(size) * 0.5))
    cx, cy = round(center[0]), round(center[1])
    left = max(0, cx - half)
    top = max(0, cy - half)
    right = min(int(shape[1]), cx + half + 1)
    bottom = min(int(shape[0]), cy + half + 1)
    return left, top, max(0, right - left), max(0, bottom - top)


def _disk_mean(image: np.ndarray, center: tuple[float, float], radius: float) -> float:
    data = _as_gray_f32(image)
    return _disk_mean_f32(data, center, radius)


def _disk_mean_f32(data: np.ndarray, center: tuple[float, float], radius: float) -> float:
    if data.size == 0:
        return 0.0
    bbox = _expanded_bbox(center, float(radius) * 2.0, data.shape)
    x_coord, y_coord, width, height = bbox
    if width <= 0 or height <= 0:
        return 0.0
    patch = data[y_coord : y_coord + height, x_coord : x_coord + width]
    yy = (np.arange(y_coord, y_coord + height, dtype=np.float32) - float(center[1])).reshape(-1, 1)
    xx = (np.arange(x_coord, x_coord + width, dtype=np.float32) - float(center[0])).reshape(1, -1)
    disk = xx * xx + yy * yy <= max(1.0, float(radius)) ** 2
    if not np.any(disk):
        return 0.0
    return float(patch[disk].mean())


def _local_patch(data: np.ndarray, cx: float, cy: float, radius: float) -> np.ndarray:
    if data.size == 0:
        return data
    pad = max(1, round(float(radius)))
    left = max(0, round(cx) - pad)
    right = min(data.shape[1], round(cx) + pad + 1)
    top = max(0, round(cy) - pad)
    bottom = min(data.shape[0], round(cy) + pad + 1)
    if right <= left or bottom <= top:
        return np.zeros((0, 0), dtype=data.dtype)
    return data[top:bottom, left:right]


def _bilinear_sample(data: np.ndarray, x_coord: float, y_coord: float) -> float:
    if data.size == 0:
        return 0.0
    x_max = max(0.0, float(data.shape[1] - 1.0))
    y_max = max(0.0, float(data.shape[0] - 1.0))
    if x_coord < 0.0:
        x_coord = 0.0
    elif x_coord > x_max:
        x_coord = x_max
    if y_coord < 0.0:
        y_coord = 0.0
    elif y_coord > y_max:
        y_coord = y_max
    x0 = int(np.floor(x_coord))
    y0 = int(np.floor(y_coord))
    x1 = min(data.shape[1] - 1, x0 + 1)
    y1 = min(data.shape[0] - 1, y0 + 1)
    dx = x_coord - x0
    dy = y_coord - y0
    top = float(data[y0, x0]) * (1.0 - dx) + float(data[y0, x1]) * dx
    bottom = float(data[y1, x0]) * (1.0 - dx) + float(data[y1, x1]) * dx
    return float(top * (1.0 - dy) + bottom * dy)


def _sample_distance(distance_map: np.ndarray, center: tuple[float, float]) -> float:
    if distance_map.size == 0:
        return 0.0
    return _bilinear_sample(distance_map, center[0], center[1])


def _as_gray_f32(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2 and image.dtype == np.float32:
        return image
    return to_gray_u8(image).astype(np.float32)


def _paint_score_disk(
    image: np.ndarray,
    center: tuple[float, float],
    radius: float,
    value: float,
    scale_max: float,
) -> None:
    if image.size == 0:
        return
    ratio = float(value) * 255.0 / max(1.0, float(scale_max))
    if ratio <= 0.0:
        encoded = 0
    elif ratio >= 255.0:
        encoded = 255
    else:
        encoded = round(ratio)
    cv2.circle(image, (round(center[0]), round(center[1])), max(1, round(radius)), encoded, thickness=-1)


def _normalize_u8(gray: np.ndarray) -> np.ndarray:
    data = to_gray_u8(gray)
    if data.size == 0:
        return data
    min_value = float(data.min())
    max_value = float(data.max())
    if max_value - min_value <= 1e-6:
        return np.zeros_like(data, dtype=np.uint8)
    return np.clip((data.astype(np.float32) - min_value) * (255.0 / (max_value - min_value)), 0, 255).astype(
        np.uint8
    )


def _normalize_response(response: np.ndarray) -> np.ndarray:
    data = np.nan_to_num(response.astype(np.float32), copy=False)
    max_value = float(data.max()) if data.size else 0.0
    if max_value <= 1e-6:
        return np.zeros(data.shape, dtype=np.uint8)
    return np.clip(data * (255.0 / max_value), 0, 255).astype(np.uint8)


def _normalize_metal_constraint_mode(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"disabled", "off", "none", "false", "0"}:
        return "disabled"
    if text in {"strict", "hard"}:
        return "strict"
    return "soft"


def _draw_overlay(
    gray: np.ndarray,
    candidates: list[BrightViaDetection],
    cfg: BrightViaDetectorConfig,
    *,
    show_rejected: bool | None = None,
) -> np.ndarray:
    base = cv2.cvtColor(to_gray_u8(gray), cv2.COLOR_GRAY2BGR)
    display_rejected = cfg.show_rejected_candidates if show_rejected is None else bool(show_rejected)
    for det in candidates:
        if not display_rejected and det.status != "accepted":
            continue
        color = _COLOR_ACCEPT
        if det.status == "soft_reject":
            color = _COLOR_SOFT
        elif det.status == "hard_reject":
            color = _COLOR_HARD
        x, y, w, h = det.bbox
        cv2.rectangle(base, (x, y), (x + w, y + h), color, 1)
        cx, cy = int(round(det.center[0])), int(round(det.center[1]))
        r = max(2, int(round(0.5 * max(w, h))))
        cv2.circle(base, (cx, cy), r, color, 1)
    return base
