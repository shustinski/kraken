"""Heuristic check of chip frames: how clean the etch of metallization and vias is, without a network.

The frames are photos of one chip layer (lines, pours, vias). There is no lattice, as in a memory
matrix, and no ground truth: every frame is read on its own and then against the whole run.

- Capture: a frame with too little contrast, too much clipped light, or a blur (narrow lines no
  longer reaching the brightness of wide metal, or a sharpness far below the run's) is not rated;
  it keeps only the reason.
- Metal: Otsu on the smoothed frame, metal brighter than the substrate unless said otherwise. The
  frame is first scaled down until a typical line is about 8 px wide.
- Width and gap: the distance transform along the skeleton of the metal and of the gaps. The
  narrow width W and the narrow gap G of a frame are their 20th percentiles. Metal narrower than
  half of W is thin (a neck or a bridge strand); a gap narrower than half of G nearly shorts.
- Topology: two line ends facing each other closer than the narrow gap, or tapering before the
  gap, are an open.
- Contamination: mid-tone pixels away from the edges (a residue film), spots far from their
  class (particles), a slow drift of brightness inside one class (stains), metal crumbs and
  pinholes smaller than half a W x W square.
- Geometry: chip edges run at 0, 45 and 90 degrees; the share of edge strength off those angles.
- Run: the features against the run's median and MAD, as in the single-result risk. The width
  risk is the narrow width and gap drifting from the run: over- or under-etch.

The risk is 0-100, higher is worse. The verdict names the leading problem.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

import numpy as np

from .domain import BuildResult, FrameAnalysisSummary, FrameRecord
from .frame_naming import DEFAULT_FRAME_NAMING, FrameNaming
from .image_formats import SUPPORTED_IMAGE_EXTENSION_SET
from .image_io import load_grayscale_image
from .mask_primitives import _boundary_mask, _distance_transform, _label_components, _neighbor_count, skeletonize
from .metric_keys import compute_metric_percentiles
from .repository_shared import BuildCancelledError, cv2
from .single_result_risk import (
    MIN_BASELINE_FRAMES,
    SENSITIVITY_MULTIPLIERS,
    RiskReason,
    _batch_risk,
    _clip01,
    _extract_rows,
    _robust_baseline,
    _severity,
)

HEURISTIC_CHECK_VERSION = "heuristic_check.v1"

HEURISTIC_CHECK_SCORE = "heuristic_check_score"
HEURISTIC_TOPOLOGY_RISK = "heuristic_topology_risk"
HEURISTIC_CONTAMINATION_RISK = "heuristic_contamination_risk"
HEURISTIC_WIDTH_RISK = "heuristic_width_risk"
HEURISTIC_GEOMETRY_RISK = "heuristic_geometry_risk"
HEURISTIC_BATCH_RISK = "heuristic_batch_risk"
HEURISTIC_CHECK_METRICS = (
    HEURISTIC_CHECK_SCORE,
    HEURISTIC_TOPOLOGY_RISK,
    HEURISTIC_CONTAMINATION_RISK,
    HEURISTIC_WIDTH_RISK,
    HEURISTIC_GEOMETRY_RISK,
    HEURISTIC_BATCH_RISK,
)

VERDICT_NORMAL = "normal"
VERDICT_TOPOLOGY = "topology"
VERDICT_DIRT = "dirt"
VERDICT_ETCH_DEVIATION = "etch_deviation"
VERDICT_UNREADABLE = "unreadable"
VERDICTS = (VERDICT_NORMAL, VERDICT_TOPOLOGY, VERDICT_DIRT, VERDICT_ETCH_DEVIATION, VERDICT_UNREADABLE)

# Reason codes by the group the details window draws them in.
TOPOLOGY_REASONS = ("open", "thin_metal", "narrow_gap")
CONTAMINATION_REASONS = ("residue", "particle", "stain", "debris", "pinhole")
GEOMETRY_REASONS = ("off_axis_edges",)
WIDTH_REASONS = ("etch_over", "etch_under", "width_spread")
CAPTURE_REASONS = ("low_contrast", "clipped", "blur", "no_pattern")

COMPONENT_WEIGHTS = {
    HEURISTIC_TOPOLOGY_RISK: 0.30,
    HEURISTIC_CONTAMINATION_RISK: 0.30,
    HEURISTIC_WIDTH_RISK: 0.15,
    HEURISTIC_GEOMETRY_RISK: 0.10,
    HEURISTIC_BATCH_RISK: 0.15,
}
VERDICT_LEVEL = 0.5
REASONS_PER_CODE = 4
REASONS_KEPT = 12
# Features of the run baseline. Capture values stay out: they rate the photo, not the etch.
BASELINE_FEATURES = (
    "metal_fraction",
    "width_narrow",
    "width_median",
    "width_spread",
    "gap_narrow",
    "gap_median",
    "off_axis_share",
    "residue_fraction",
    "particle_fraction",
    "stain_fraction",
)
WIDTH_FEATURES = ("width_narrow", "gap_narrow", "width_spread")


@dataclass(frozen=True, slots=True)
class HeuristicCheckConfig:
    metal_brighter: bool = True
    smooth_sigma: float = 1.0
    # Frames are scaled down until the typical line is about this wide (px); finer is only slower.
    analysis_width: float = 8.0
    # Share of 0..255 between the 1st and 99th percentiles below which a frame is not rated.
    min_contrast: float = 0.08
    max_clipped_fraction: float = 0.30
    # Metal share outside this range: no pattern to rate.
    min_metal_fraction: float = 0.02
    max_metal_fraction: float = 0.98
    # Sharpness below this share of the run's median is a blur.
    blur_ratio: float = 0.30
    # Narrow lines (gaps) reaching less than this share of the wide metal (substrate) level: a blur.
    min_line_fill: float = 0.75
    # Metal or gap narrower than this share of the narrow width (gap) is thin (narrow).
    thin_ratio: float = 0.5
    # Facing line ends closer than this share of the narrow gap are an open.
    open_gap_ratio: float = 0.8
    # A line end narrower than this share of the line behind it tapers.
    taper_ratio: float = 0.6
    # Crumbs and holes smaller than this many W x W squares are debris and pinholes.
    debris_area_ratio: float = 2.0
    residue_low: float = 0.3
    residue_high: float = 0.7
    particle_mad: float = 6.0
    particle_min_area: int = 4
    stain_level: float = 0.12
    off_axis_degrees: float = 10.0


@dataclass(frozen=True, slots=True)
class HeuristicCheckSummary:
    """One frame's result. Risks are 0-100 (None when not rated); reasons carry frame-relative boxes."""

    total_risk: float | None
    verdict: str
    topology_risk: float | None
    contamination_risk: float | None
    width_risk: float | None
    geometry_risk: float | None
    batch_outlier_risk: float | None
    sensitivity: str
    capture_issues: tuple[str, ...]
    reasons: tuple[RiskReason, ...]
    component_weights: dict[str, float]
    component_contributions: dict[str, float]
    feature_values: dict[str, float]


@dataclass(slots=True)
class FrameFindings:
    """What one frame shows by itself, before the run is known."""

    features: dict[str, float]
    extra: dict[str, float]
    capture_issues: tuple[str, ...]
    topology_raw: float = 0.0
    contamination_raw: float = 0.0
    geometry_raw: float = 0.0
    reasons: list[RiskReason] = field(default_factory=list)


@dataclass(slots=True)
class _HeuristicRow:
    record: FrameRecord
    findings: FrameFindings | None
    error: str = ""

    @property
    def features(self) -> dict[str, float]:
        return {} if self.findings is None else self.findings.features


# --- Photos or masks -----------------------------------------------------------------------------


def looks_like_photo(gray: np.ndarray) -> bool:
    """A photo spreads over many gray levels; a mask, even a soft one, sits mostly at 0 and 255."""

    values = np.asarray(gray)
    if values.ndim != 2 or values.size == 0:
        return False
    step = max(1, int(np.sqrt(values.size / 250_000.0)))
    sample = np.asarray(values[::step, ::step], dtype=np.uint8)
    extreme = float(np.mean((sample <= 10) | (sample >= 245)))
    histogram = np.bincount(sample.ravel(), minlength=256)
    levels = int(np.count_nonzero(histogram > sample.size * 1.0e-4))
    return extreme < 0.5 and levels >= 32


_PHOTO_FOLDER_CACHE: dict[tuple[str, int], bool] = {}


def folder_holds_photos(folder: Path | str | None, naming: FrameNaming | None = None, *, samples: int = 3) -> bool:
    """Whether a source folder holds chip photos rather than network masks (a few frames vote)."""

    if folder is None:
        return False
    path = Path(folder)
    try:
        stamp = int(path.stat().st_mtime_ns)
    except OSError:
        return False
    cache_key = (str(path), stamp)
    cached = _PHOTO_FOLDER_CACHE.get(cache_key)
    if cached is not None:
        return cached
    naming = (naming or DEFAULT_FRAME_NAMING).normalized()
    try:
        candidates = sorted(
            entry
            for entry in path.iterdir()
            if entry.is_file()
            and entry.suffix.lower() in SUPPORTED_IMAGE_EXTENSION_SET
            and not naming.is_confidence_file(entry)
        )
    except OSError:
        return False
    votes: list[bool] = []
    for candidate in candidates[: max(1, int(samples))]:
        try:
            votes.append(looks_like_photo(load_grayscale_image(candidate)))
        except Exception:
            continue
    result = bool(votes) and sum(votes) * 2 > len(votes)
    _PHOTO_FOLDER_CACHE[cache_key] = result
    return result


# --- One frame -----------------------------------------------------------------------------------


def _bbox(y0: float, x0: float, y1: float, x1: float, shape: tuple[int, int]) -> tuple[float, float, float, float]:
    height, width = shape
    y0 = max(0.0, min(float(height), y0))
    y1 = max(y0 + 1.0, min(float(height), y1))
    x0 = max(0.0, min(float(width), x0))
    x1 = max(x0 + 1.0, min(float(width), x1))
    return (x0 / max(1, width), y0 / max(1, height), (x1 - x0) / max(1, width), (y1 - y0) / max(1, height))


def _component_boxes(
    mask: np.ndarray, *, min_area: int = 1, max_area: float | None = None, skip_border: bool = False, margin: float = 0.0
) -> list[tuple[int, tuple[float, float, float, float]]]:
    """(area, box) of the components of ``mask`` within the size limits, largest first."""

    if not np.any(mask):
        return []
    if cv2 is not None:
        count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
            np.asarray(mask, dtype=np.uint8), connectivity=8
        )
        rows = [
            (int(stats[i, cv2.CC_STAT_AREA]), int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
             int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]))
            for i in range(1, count)
        ]
    else:
        labels, count = _label_components(np.asarray(mask, dtype=bool))
        rows = []
        for label_id in range(1, count + 1):
            yy, xx = np.nonzero(labels == label_id)
            rows.append((int(yy.size), int(xx.min()), int(yy.min()), int(xx.max() - xx.min() + 1), int(yy.max() - yy.min() + 1)))
    height, width = mask.shape
    boxes: list[tuple[int, tuple[float, float, float, float]]] = []
    for area, left, top, box_width, box_height in rows:
        if area < min_area or (max_area is not None and area >= max_area):
            continue
        if skip_border and (left == 0 or top == 0 or left + box_width >= width or top + box_height >= height):
            continue
        boxes.append(
            (
                area,
                _bbox(top - margin, left - margin, top + box_height + margin, left + box_width + margin, mask.shape),
            )
        )
    boxes.sort(key=lambda item: item[0], reverse=True)
    return boxes


def _open3(mask: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), dtype=np.uint8)
    return cv2.morphologyEx(np.asarray(mask, dtype=np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)


def _component_mask(mask: np.ndarray, *, min_area: int) -> np.ndarray:
    """Only the components of at least ``min_area`` pixels."""

    if not np.any(mask):
        return np.asarray(mask, dtype=bool)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        np.asarray(mask, dtype=np.uint8), connectivity=8
    )
    keep = np.zeros(count, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= max(1, int(min_area))
    return keep[labels]


def _pixel_box(box: tuple[float, float, float, float], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    height, width = shape
    x, y, w, h = box
    x0, y0 = int(round(x * width)), int(round(y * height))
    return y0, x0, max(y0 + 1, int(round((y + h) * height))), max(x0 + 1, int(round((x + w) * width)))


def _fill_box(mask: np.ndarray, box: tuple[float, float, float, float]) -> None:
    y0, x0, y1, x1 = _pixel_box(box, mask.shape)
    mask[y0:y1, x0:x1] = True


def _box_center_in(mask: np.ndarray, box: tuple[float, float, float, float]) -> bool:
    y0, x0, y1, x1 = _pixel_box(box, mask.shape)
    cy = min(mask.shape[0] - 1, max(0, (y0 + y1) // 2))
    cx = min(mask.shape[1] - 1, max(0, (x0 + x1) // 2))
    return bool(mask[cy, cx])


def _rough_width(mask: np.ndarray, distance: np.ndarray) -> float:
    """Typical width from the ridge of the distance transform (no skeleton needed)."""

    if not np.any(mask):
        return 0.0
    ridge = mask & (distance >= 1.0) & (distance >= cv2.dilate(distance, np.ones((3, 3), dtype=np.uint8)))
    values = distance[ridge]
    return float(max(1.0, 2.0 * np.median(values) - 1.0)) if values.size else 1.0


def _bounded_skeleton(mask: np.ndarray, distance: np.ndarray, cap: float) -> np.ndarray:
    """Skeleton of the mask with its deep interior hollowed out, so thinning stops after ~cap/2 passes.

    A wide pour becomes a ring, whose skeleton has no ends and the pour's width cap; lines keep
    their own skeleton and widths.
    """

    return skeletonize(np.asarray(mask & (distance <= cap), dtype=np.uint8))


def _prune(skeleton: np.ndarray, steps: int) -> np.ndarray:
    """Skeleton without spurs up to ``steps`` long: the forks thinning leaves at a square line end.

    Every line end gets shorter too; ``_metal_tip`` finds where the line really ends.
    """

    pruned = np.asarray(skeleton, dtype=bool).copy()
    for _step in range(max(0, int(steps))):
        neighbors = _neighbor_count(pruned)
        tips = pruned & (neighbors <= 1)
        if not np.any(tips):
            break
        pruned &= ~tips
    return pruned


@dataclass(slots=True)
class _Profile:
    skeleton: np.ndarray
    neighbors: np.ndarray
    core: np.ndarray
    narrow: float
    median: float
    spread: float


def _width_profile(mask: np.ndarray, distance: np.ndarray) -> _Profile:
    rough = _rough_width(mask, distance)
    skeleton = _prune(_bounded_skeleton(mask, distance, max(6.0, 3.0 * rough)), max(1, int(round(rough / 2.0))))
    neighbors = _neighbor_count(skeleton)
    if not np.any(skeleton):
        empty = np.zeros_like(mask, dtype=bool)
        return _Profile(skeleton, neighbors, empty, rough, rough, 0.0)
    radius = max(2, int(round(rough)))
    joints = (skeleton & (neighbors == 1)) | (skeleton & (neighbors >= 3))
    near_joint = cv2.dilate(joints.astype(np.uint8), np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)) > 0
    core = skeleton & ~near_joint
    core[:radius, :] = False
    core[-radius:, :] = False
    core[:, :radius] = False
    core[:, -radius:] = False
    widths = 2.0 * distance[core if np.any(core) else skeleton] - 1.0
    p20, p25, p50, p75 = (float(value) for value in np.percentile(widths, (20.0, 25.0, 50.0, 75.0)))
    return _Profile(skeleton, neighbors, core, max(1.0, p20), max(1.0, p50), (p75 - p25) / max(1.0, p50))


_NEIGHBOR_STEPS = tuple((dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx)


def _walk(skeleton: np.ndarray, neighbors: np.ndarray, start: tuple[int, int], limit: int) -> tuple[list[tuple[int, int]], bool]:
    """Path from a line end inward, up to ``limit`` steps; True when it ran into a branch point."""

    height, width = skeleton.shape
    path = [start]
    previous: tuple[int, int] | None = None
    current = start
    for _step in range(limit):
        cy, cx = current
        candidates = [
            (cy + dy, cx + dx)
            for dy, dx in _NEIGHBOR_STEPS
            if 0 <= cy + dy < height and 0 <= cx + dx < width and skeleton[cy + dy, cx + dx]
            and (cy + dy, cx + dx) != previous and (cy + dy, cx + dx) not in path[-3:]
        ]
        if len(candidates) != 1:
            return path, len(candidates) > 1
        previous, current = current, candidates[0]
        path.append(current)
        if int(neighbors[current]) >= 3:
            return path, True
    return path, False


def _metal_tip(metal: np.ndarray, start: tuple[int, int], direction: tuple[float, float], limit: float) -> tuple[float, float]:
    """Where the metal really ends: from a skeleton end straight on until the metal stops.

    Thinning and pruning leave the skeleton end about a line width short of the line's end.
    """

    height, width = metal.shape
    y, x = float(start[0]), float(start[1])
    tip = (y, x)
    for step in np.arange(0.5, limit, 0.5):
        cy, cx = start[0] + direction[0] * step, start[1] + direction[1] * step
        iy, ix = int(round(cy)), int(round(cx))
        if not (0 <= iy < height and 0 <= ix < width) or not metal[iy, ix]:
            break
        tip = (cy, cx)
    return tip


def _opens(
    profile: _Profile,
    distance: np.ndarray,
    metal: np.ndarray,
    gap_narrow: float,
    config: HeuristicCheckConfig,
) -> list[tuple[tuple[int, int], tuple[int, int], float]]:
    """Facing line ends of the metal that look broken: (tip a, tip b, gap between the tips)."""

    skeleton, neighbors = profile.skeleton, profile.neighbors
    width = profile.narrow
    height, frame_width = skeleton.shape
    margin = int(round(width)) + 2
    ends_y, ends_x = np.nonzero(skeleton & (neighbors == 1))
    limit = max(4, int(round(3.0 * width)))
    points: list[tuple[int, int]] = []
    directions: list[tuple[float, float]] = []
    tapered: list[bool] = []
    for y, x in zip(ends_y.tolist(), ends_x.tolist()):
        if y < margin or x < margin or y >= height - margin or x >= frame_width - margin:
            continue
        path, branched = _walk(skeleton, neighbors, (int(y), int(x)), limit)
        if len(path) < 3 or (branched and len(path) <= max(2, int(round(width / 2.0)))):
            continue  # a spur of a rough edge or a corner, not a line end
        tail_y, tail_x = path[-1]
        vy, vx = float(y - tail_y), float(x - tail_x)
        norm = float(np.hypot(vy, vx))
        if norm < 1e-6:
            continue
        near_width = 2.0 * float(distance[path[0]]) - 1.0
        far = path[min(len(path) - 1, int(round(width))) :]
        far_width = 2.0 * float(np.median([distance[point] for point in far])) - 1.0 if far else near_width
        unit_direction = (vy / norm, vx / norm)
        tip_y, tip_x = _metal_tip(metal, (int(y), int(x)), unit_direction, 2.0 * width + 4.0)
        points.append((int(round(tip_y)), int(round(tip_x))))
        directions.append(unit_direction)
        tapered.append(near_width < config.taper_ratio * far_width)
    if len(points) < 2:
        return []
    coords = np.asarray(points, dtype=np.float64)
    vectors = np.asarray(directions, dtype=np.float64)
    reach = 2.5 * max(width, gap_narrow, 2.0)
    pairs: list[tuple[float, int, int]] = []
    for i in range(len(points)):
        delta = coords - coords[i]
        gaps = np.hypot(delta[:, 0], delta[:, 1])
        for j in np.flatnonzero((gaps > 0.0) & (gaps <= reach)).tolist():
            if j <= i:
                continue
            unit = delta[j] / gaps[j]
            if float(vectors[i] @ unit) < 0.7 or float(vectors[j] @ -unit) < 0.7:
                continue
            if gaps[j] < config.open_gap_ratio * gap_narrow or tapered[i] or tapered[j]:
                pairs.append((float(gaps[j]), i, j))
    pairs.sort()
    used: set[int] = set()
    found: list[tuple[tuple[int, int], tuple[int, int], float]] = []
    for gap, i, j in pairs:
        if i in used or j in used:
            continue
        used.update((i, j))
        found.append((points[i], points[j], gap))
    return found


def _class_drift(values: np.ndarray, weight: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Local mean of ``values`` over ``weight`` pixels and where it is defined."""

    weight_f = np.asarray(weight, dtype=np.float32)
    blurred_weight = cv2.GaussianBlur(weight_f, (0, 0), sigma)
    blurred = cv2.GaussianBlur(values * weight_f, (0, 0), sigma)
    valid = blurred_weight > 0.25
    return np.where(valid, blurred / np.maximum(blurred_weight, 1.0e-6), 0.0), valid


def _add_reasons(
    reasons: list[RiskReason], code: str, contribution: float, boxes: list[tuple[int, tuple[float, float, float, float]]]
) -> None:
    for _area, box in boxes[:REASONS_PER_CODE]:
        reasons.append(RiskReason(code, _severity(contribution), contribution, box, detail=str(len(boxes))))


def analyze_frame(gray: np.ndarray, config: HeuristicCheckConfig | None = None) -> FrameFindings:
    """Features, raw risks and reasons of one chip photo by itself."""

    if cv2 is None:  # pragma: no cover - OpenCV is a hard dependency of the app
        raise RuntimeError("OpenCV is required for the heuristic check")
    cfg = config or HeuristicCheckConfig()
    image = np.asarray(gray, dtype=np.uint8)
    values = image.astype(np.float32)
    low, high = (float(value) for value in np.percentile(values, (1.0, 99.0)))
    contrast = (high - low) / 255.0
    clipped = float(np.mean((image <= 2) | (image >= 253)))
    normalized = np.clip((values - low) / max(high - low, 1.0e-6), 0.0, 1.0)
    sharpness = float(np.var(cv2.Laplacian(normalized, cv2.CV_32F, ksize=3)))
    extra = {"contrast": contrast, "clipped_fraction": clipped, "sharpness": sharpness}
    issues: list[str] = []
    if contrast < cfg.min_contrast:
        issues.append("low_contrast")
    if clipped > cfg.max_clipped_fraction:
        issues.append("clipped")
    if "low_contrast" in issues:
        return FrameFindings({}, extra, tuple(issues))

    def metal_of(plane: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # No morphology: an opening would erase the very strands and crumbs looked for.
        smoothed = cv2.GaussianBlur(plane, (0, 0), float(cfg.smooth_sigma))
        smoothed_u8 = np.asarray(np.round(smoothed * 255.0), dtype=np.uint8)
        level, _binary = cv2.threshold(smoothed_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        found = smoothed_u8 > level if cfg.metal_brighter else smoothed_u8 <= level
        return smoothed, np.asarray(found, dtype=bool)

    smooth, metal = metal_of(normalized)
    metal_fraction = float(np.mean(metal))
    extra["metal_fraction"] = metal_fraction
    if not (cfg.min_metal_fraction <= metal_fraction <= cfg.max_metal_fraction):
        return FrameFindings({}, extra, tuple(issues + ["no_pattern"]))
    # Work where the typical line is about analysis_width wide: thinning and walking cost grow
    # with the width, the findings do not.
    scale = 1.0
    rough = _rough_width(metal, _distance_transform(metal))
    if rough > 1.25 * cfg.analysis_width:
        target_width = max(16, int(round(image.shape[1] * cfg.analysis_width / rough)))
        target_height = max(16, int(round(image.shape[0] * cfg.analysis_width / rough)))
        scale = target_width / float(image.shape[1])
        smooth, metal = metal_of(cv2.resize(normalized, (target_width, target_height), interpolation=cv2.INTER_AREA))
    extra["analysis_scale"] = scale
    gap = ~metal
    shape = metal.shape

    # Width and gap.
    metal_distance = _distance_transform(metal)
    gap_distance = _distance_transform(gap)
    metal_profile = _width_profile(metal, metal_distance)
    gap_profile = _width_profile(gap, gap_distance)
    width_narrow, gap_narrow = metal_profile.narrow, gap_profile.narrow
    reasons: list[RiskReason] = []

    def narrow_boxes(profile: _Profile, distance: np.ndarray) -> list[tuple[int, tuple[float, float, float, float]]]:
        if profile.narrow < 4.0:
            return []  # one or two pixel wide lines cannot neck any further
        # Joints only make the width larger; only the ends and the frame border make it smaller.
        ends = profile.skeleton & (profile.neighbors <= 1)
        near_end = cv2.dilate(ends.astype(np.uint8), np.ones((5, 5), dtype=np.uint8)) > 0
        candidates = profile.skeleton & ~near_end
        border = max(2, int(round(profile.narrow)))
        candidates[:border, :] = False
        candidates[-border:, :] = False
        candidates[:, :border] = False
        candidates[:, -border:] = False
        narrow = candidates & (2.0 * distance - 1.0 < cfg.thin_ratio * profile.narrow)
        grown = cv2.dilate(narrow.astype(np.uint8), np.ones((3, 3), dtype=np.uint8)) > 0
        return _component_boxes(grown, min_area=4, margin=profile.narrow)

    thin = narrow_boxes(metal_profile, metal_distance)
    narrow_gaps = narrow_boxes(gap_profile, gap_distance)

    # Opens.
    opens = _opens(metal_profile, metal_distance, metal, gap_narrow, cfg)
    open_boxes = [
        (
            int(round(gap_size)),
            _bbox(
                min(a[0], b[0]) - width_narrow,
                min(a[1], b[1]) - width_narrow,
                max(a[0], b[0]) + width_narrow + 1,
                max(a[1], b[1]) + width_narrow + 1,
                shape,
            ),
        )
        for a, b, gap_size in opens
    ]

    # Crumbs of metal and holes in it.
    small_area = max(4.0, cfg.debris_area_ratio * width_narrow * width_narrow)
    debris = _component_boxes(metal, min_area=3, max_area=small_area, skip_border=True, margin=2.0)
    pinholes = _component_boxes(gap, min_area=3, max_area=small_area, skip_border=True, margin=2.0)

    # Contamination by brightness. The class levels come from pixels well inside their class;
    # the edge band is as wide as the frame's own blur: mid-tones per edge pixel near the edges.
    inner_metal = metal & (metal_distance > 2.0)
    inner_gap = gap & (gap_distance > 2.0)
    metal_level = float(np.median(smooth[inner_metal])) if np.any(inner_metal) else float(np.median(smooth[metal]))
    gap_level = float(np.median(smooth[inner_gap])) if np.any(inner_gap) else float(np.median(smooth[gap]))
    separation = metal_level - gap_level
    extra["class_separation"] = abs(separation)
    if abs(separation) < 0.05:
        return FrameFindings({}, extra, tuple(issues + ["low_contrast"]))
    position = (smooth - gap_level) / separation
    mid_tone = (position > cfg.residue_low) & (position < cfg.residue_high)
    edge_distance = np.where(metal, metal_distance, gap_distance)
    near_edge = edge_distance <= max(3.0, 0.5 * width_narrow)
    boundary_length = int(np.count_nonzero(_boundary_mask(metal)))
    transition = float(np.count_nonzero(mid_tone & near_edge)) / float(max(1, boundary_length))
    band = max(2.0, 0.25 * width_narrow, transition + 1.0)
    extra["edge_band"] = band
    far_metal = metal & (metal_distance > band)
    far_gap = gap & (gap_distance > band)
    if np.any(far_metal):
        metal_level = float(np.median(smooth[far_metal]))
    if np.any(far_gap):
        gap_level = float(np.median(smooth[far_gap]))
    separation = metal_level - gap_level if abs(metal_level - gap_level) >= 0.05 else separation
    position = (smooth - gap_level) / separation
    # A soft focus: the narrow lines no longer reach the level of wide metal, nor the narrow gaps
    # that of open substrate. Measured on the skeletons, against the wide areas when there are any.
    fills: list[float] = []
    for profile, mask_of_class, distance, own, other in (
        (metal_profile, metal, metal_distance, metal_level, gap_level),
        (gap_profile, gap, gap_distance, gap_level, metal_level),
    ):
        if not np.any(profile.core):
            continue
        deep = mask_of_class & (distance > max(band, 0.5 * profile.narrow + 1.5))
        reference = float(np.median(smooth[deep])) if np.count_nonzero(deep) >= 32 else own
        if abs(reference - other) >= 0.05:
            fills.append((float(np.median(smooth[profile.core])) - other) / (reference - other))
    line_fill = min(fills, default=1.0)
    extra["line_fill"] = line_fill
    if line_fill < cfg.min_line_fill:
        return FrameFindings({}, extra, tuple(issues + ["blur"]))
    # Gradient: the residue film is flat, the edges are not; geometry reads its directions.
    grad_x = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(grad_x, grad_y)
    edge = magnitude > max(float(np.percentile(magnitude, 90.0)), 0.05)
    boundary = _boundary_mask(metal)
    edge_level = float(np.median(magnitude[boundary])) if np.any(boundary) else float(np.max(magnitude))
    flat = magnitude < 0.25 * edge_level

    # A residue film: a flat mid-tone patch, wherever the mask happened to put it.
    residue = _open3((position > cfg.residue_low) & (position < cfg.residue_high) & flat)
    residue = _component_mask(residue, min_area=int(round(width_narrow * width_narrow)))
    # Particles: small spots far from their class level; bigger ones are stains.
    deviating = np.zeros(shape, dtype=bool)
    for region, level in ((far_metal, metal_level), (far_gap, gap_level)):
        if not np.any(region):
            continue
        deviation = np.abs(smooth - level)
        mad = float(np.median(deviation[region]))
        limit = max(cfg.particle_mad * 1.4826 * mad, 0.25 * abs(separation))
        deviating |= region & (deviation > limit)
    deviating &= ~residue
    particle_limit = 4.0 * width_narrow * width_narrow
    particle_boxes = _component_boxes(
        deviating, min_area=int(cfg.particle_min_area), max_area=particle_limit, margin=2.0
    )
    particle_pixels = sum(area for area, _box in particle_boxes)
    sigma = max(8.0, min(shape) / 48.0)
    stain = _component_mask(deviating, min_area=int(particle_limit))
    # Stains are read inside wide areas only, against their own level: a narrow line under a
    # soft focus never reaches the brightness of a pour, and that is not a stain.
    for mask_of_class, distance, narrow in ((metal, metal_distance, width_narrow), (gap, gap_distance, gap_narrow)):
        deep = mask_of_class & (distance > max(band, 0.5 * narrow + 1.5))
        if not np.any(deep):
            continue
        drift, valid = _class_drift(smooth, deep, sigma)
        level = float(np.median(smooth[deep]))
        stain |= valid & deep & (np.abs(drift - level) > cfg.stain_level * abs(separation))
    stain &= ~residue
    residue_fraction = float(np.mean(residue))
    particle_fraction = float(particle_pixels) / float(max(1, metal.size))
    stain_fraction = float(np.mean(stain))

    # Dirt hides the topology: thin metal, narrow gaps and opens inside a dirty patch are the
    # dirt's own shapes, so they count as dirt there, not as broken lines.
    dirty = residue | stain
    for _area, box in particle_boxes + debris + pinholes:
        _fill_box(dirty, box)
    dirty = (
        cv2.dilate(
            dirty.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=max(1, int(round(width_narrow)))
        )
        > 0
    )

    def clean_of_dirt(
        boxes: list[tuple[int, tuple[float, float, float, float]]],
    ) -> list[tuple[int, tuple[float, float, float, float]]]:
        return [item for item in boxes if not _box_center_in(dirty, item[1])]

    thin = clean_of_dirt(thin)
    narrow_gaps = clean_of_dirt(narrow_gaps)
    open_boxes = clean_of_dirt(open_boxes)

    # Geometry: edges off 0/45/90 degrees.
    angle = np.mod(np.degrees(np.arctan2(grad_y, grad_x)), 45.0)
    off_axis = edge & (np.minimum(angle, 45.0 - angle) > cfg.off_axis_degrees)
    edge_strength = float(np.sum(magnitude[edge], dtype=np.float64))
    off_axis_share = float(np.sum(magnitude[off_axis], dtype=np.float64)) / max(edge_strength, 1.0e-9)
    off_density, off_valid = _class_drift(off_axis.astype(np.float32), edge, sigma)
    off_regions = off_valid & (off_density > 0.6) & (
        cv2.GaussianBlur(edge.astype(np.float32), (0, 0), sigma) > 0.02
    )

    topology_raw = _clip01((len(open_boxes) + 0.5 * len(thin) + 0.5 * len(narrow_gaps)) / 2.0)
    contamination_raw = _clip01(
        max(
            residue_fraction / 0.02,
            particle_fraction / 0.005,
            stain_fraction / 0.03,
            (len(debris) + len(pinholes)) / 10.0,
        )
    )
    geometry_raw = _clip01((off_axis_share - 0.25) / 0.35)

    _add_reasons(reasons, "open", _clip01(len(open_boxes) / 2.0), open_boxes)
    _add_reasons(reasons, "thin_metal", _clip01(len(thin) / 4.0), thin)
    _add_reasons(reasons, "narrow_gap", _clip01(len(narrow_gaps) / 4.0), narrow_gaps)
    _add_reasons(reasons, "residue", _clip01(residue_fraction / 0.02), _component_boxes(residue, min_area=9, margin=2.0))
    _add_reasons(reasons, "particle", _clip01(particle_fraction / 0.005), particle_boxes)
    _add_reasons(reasons, "stain", _clip01(stain_fraction / 0.03), _component_boxes(stain, min_area=16))
    _add_reasons(reasons, "debris", _clip01(len(debris) / 10.0), debris)
    _add_reasons(reasons, "pinhole", _clip01(len(pinholes) / 10.0), pinholes)
    if geometry_raw >= 0.35:
        _add_reasons(reasons, "off_axis_edges", geometry_raw, _component_boxes(off_regions, min_area=16))

    features = {
        "metal_fraction": metal_fraction,
        # Widths in pixels of the frame itself, whatever scale the analysis ran at.
        "width_narrow": width_narrow / scale,
        "width_median": metal_profile.median / scale,
        "width_spread": metal_profile.spread,
        "gap_narrow": gap_narrow / scale,
        "gap_median": gap_profile.median / scale,
        "off_axis_share": off_axis_share,
        "residue_fraction": residue_fraction,
        "particle_fraction": particle_fraction,
        "stain_fraction": stain_fraction,
    }
    extra.update(
        {
            "open_count": float(len(open_boxes)),
            "thin_metal_count": float(len(thin)),
            "narrow_gap_count": float(len(narrow_gaps)),
            "debris_count": float(len(debris)),
            "pinhole_count": float(len(pinholes)),
        }
    )
    return FrameFindings(features, extra, tuple(issues), topology_raw, contamination_raw, geometry_raw, reasons)


# --- The run -------------------------------------------------------------------------------------


def _frame_path(record: FrameRecord, model_id: str) -> str:
    return str((record.model_mask_paths or {}).get(model_id) or record.original_path or "")


def _extract(config: HeuristicCheckConfig) -> Callable[[FrameRecord, str], _HeuristicRow]:
    def extract(record: FrameRecord, model_id: str) -> _HeuristicRow:
        path_text = _frame_path(record, model_id)
        if not path_text:
            return _HeuristicRow(record, None, "missing_frame")
        try:
            gray = load_grayscale_image(Path(path_text))
        except Exception:
            return _HeuristicRow(record, None, "decode_error")
        return _HeuristicRow(record, analyze_frame(gray, config))

    return extract


def _scaled(value: float, multiplier: float) -> float:
    return _clip01(float(value) / max(0.1, multiplier))


def _width_risk(
    features: dict[str, float], baseline: dict[str, tuple[float, float]], multiplier: float
) -> tuple[float | None, list[RiskReason]]:
    """Narrow width and gap against the run: thinner metal with wider gaps is over-etch."""

    if not all(key in baseline for key in ("width_narrow", "gap_narrow")):
        return None, []
    deviations: dict[str, float] = {}
    for key in WIDTH_FEATURES:
        if key not in baseline or key not in features:
            continue
        median, scale = baseline[key]
        deviations[key] = (float(features[key]) - median) / max(scale * 1.4826, 1.0e-9)
    risk = _clip01(max((abs(value) for value in deviations.values()), default=0.0) / max(0.1, 3.0 * multiplier))
    reasons: list[RiskReason] = []
    if risk >= 0.35:
        width_dev, gap_dev = deviations.get("width_narrow", 0.0), deviations.get("gap_narrow", 0.0)
        width_median = baseline["width_narrow"][0]
        detail = f"{(float(features['width_narrow']) / max(width_median, 1.0e-9) - 1.0) * 100.0:+.0f}%"
        if width_dev <= 0.0 <= gap_dev and max(abs(width_dev), abs(gap_dev)) >= 1.0:
            code = "etch_over"
        elif gap_dev <= 0.0 <= width_dev and max(abs(width_dev), abs(gap_dev)) >= 1.0:
            code = "etch_under"
        else:
            code = "width_spread"
        reasons.append(RiskReason(code, _severity(risk), risk, None, detail=detail))
    return risk, reasons


def _width_baseline(rows: list[_HeuristicRow]) -> dict[str, tuple[float, float]]:
    """Median and spread of the width features over the run.

    Widths are often the same whole number of pixels on every clean frame; then the spread is
    floored at a quarter pixel or 3 % of the width, so one frame a pixel thinner still shows.
    """

    if len(rows) < MIN_BASELINE_FRAMES:
        return {}
    baseline: dict[str, tuple[float, float]] = {}
    for key in WIDTH_FEATURES:
        values = np.asarray([row.features[key] for row in rows if key in row.features], dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size < MIN_BASELINE_FRAMES:
            continue
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        q25, q75 = (float(value) for value in np.percentile(values, (25.0, 75.0)))
        floor = 0.03 * abs(median) if key == "width_spread" else max(0.25, 0.03 * abs(median))
        baseline[key] = (median, max(mad, (q75 - q25) / 1.349 / 1.4826, floor / 1.4826, 1.0e-6))
    return baseline


def _verdict(topology: float, contamination: float, width: float | None) -> str:
    if topology >= VERDICT_LEVEL and topology >= contamination:
        return VERDICT_TOPOLOGY
    if contamination >= VERDICT_LEVEL:
        return VERDICT_DIRT
    if width is not None and width >= VERDICT_LEVEL:
        return VERDICT_ETCH_DEVIATION
    return VERDICT_NORMAL


def _unrated_summary(row: _HeuristicRow, issues: tuple[str, ...], sensitivity: str) -> HeuristicCheckSummary:
    findings = row.findings
    values = {} if findings is None else {**findings.extra, **findings.features}
    return HeuristicCheckSummary(
        total_risk=None,
        verdict=VERDICT_UNREADABLE,
        topology_risk=None,
        contamination_risk=None,
        width_risk=None,
        geometry_risk=None,
        batch_outlier_risk=None,
        sensitivity=sensitivity,
        capture_issues=issues,
        reasons=tuple(RiskReason(code, "high", 1.0, None) for code in issues),
        component_weights={},
        component_contributions={},
        feature_values=values,
    )


def compute_heuristic_check(
    build_result: BuildResult,
    *,
    sensitivity: str = "balanced",
    config: HeuristicCheckConfig | None = None,
    excluded_record_keys: set[str] | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
    state_callback: Callable[[str, str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> BuildResult:
    """Rate every frame of one photo source: risk 0-100, verdict and reasons."""

    if len(build_result.model_specs) != 1:
        raise ValueError("The heuristic check reads exactly one photo source")
    cfg = config or HeuristicCheckConfig()
    sensitivity_key = str(sensitivity or "balanced").strip().lower()
    sensitivity_key = sensitivity_key if sensitivity_key in SENSITIVITY_MULTIPLIERS else "balanced"
    multiplier = float(SENSITIVITY_MULTIPLIERS[sensitivity_key])
    excluded = {str(key) for key in (excluded_record_keys or set())}
    model_id = str(build_result.model_specs[0].model_id)
    source_records = [record for record in build_result.records if str(record.key) not in excluded]
    total_work = max(1, len(source_records) * 2)
    max_workers = int(getattr(build_result.options, "max_workers", 0) or (os.cpu_count() or 4))
    rows: list[_HeuristicRow] = _extract_rows(
        source_records,
        model_id,
        max_workers=max_workers,
        total_work=total_work,
        progress_callback=progress_callback,
        state_callback=state_callback,
        cancel_check=cancel_check,
        extract=_extract(cfg),
    )

    rated = [row for row in rows if not row.error and row.findings is not None and not row.findings.capture_issues]
    sharpness = [float(row.findings.extra["sharpness"]) for row in rated]
    blur_limit = (
        cfg.blur_ratio * float(np.median(sharpness)) if len(sharpness) >= MIN_BASELINE_FRAMES else None
    )
    if blur_limit is not None:
        rated = [row for row in rated if float(row.findings.extra["sharpness"]) >= blur_limit]
    baseline = {
        key: value for key, value in _robust_baseline(rated).items() if key in BASELINE_FEATURES
    }
    width_baseline = _width_baseline(rated)

    updated: dict[str, FrameRecord] = {}
    for offset, row in enumerate(rows, start=1):
        if cancel_check is not None and cancel_check():
            raise BuildCancelledError("Build cancelled")
        record = row.record
        key = str(record.key)
        if row.error or row.findings is None:
            summary = FrameAnalysisSummary(0.0, 0.0, 0.0, 0.0, notes=(row.error or "decode_error",), frame_type="polygon")
            updated[key] = replace(record, summary=summary, score_ready=False, absolute_score=None)
            if state_callback is not None:
                state_callback(key, "error")
            continue
        findings = row.findings
        issues = tuple(findings.capture_issues)
        if blur_limit is not None and not issues and float(findings.extra["sharpness"]) < blur_limit:
            issues = ("blur",)
        if issues:
            risk_summary = _unrated_summary(row, issues, sensitivity_key)
            summary = FrameAnalysisSummary(
                0.0, 0.0, 0.0, 0.0, notes=(VERDICT_UNREADABLE, *issues), frame_type="polygon", heuristic_check=risk_summary
            )
            updated[key] = replace(record, summary=summary, score_ready=False, absolute_score=None)
            if state_callback is not None:
                state_callback(key, "ready")
            continue

        batch_raw, batch_reasons = _batch_risk(row, baseline, multiplier)
        batch_reasons = [reason for reason in batch_reasons if reason.code != "insufficient_baseline"]
        width_raw, width_reasons = _width_risk(findings.features, width_baseline, multiplier)
        topology = _scaled(findings.topology_raw, multiplier)
        contamination = _scaled(findings.contamination_raw, multiplier)
        geometry = _scaled(findings.geometry_raw, multiplier)
        components = {
            HEURISTIC_TOPOLOGY_RISK: topology,
            HEURISTIC_CONTAMINATION_RISK: contamination,
            HEURISTIC_WIDTH_RISK: width_raw,
            HEURISTIC_GEOMETRY_RISK: geometry,
            HEURISTIC_BATCH_RISK: batch_raw,
        }
        available = {name: value for name, value in components.items() if value is not None}
        weight_sum = sum(COMPONENT_WEIGHTS[name] for name in available)
        weights = {name: COMPONENT_WEIGHTS[name] / max(weight_sum, 1.0e-12) for name in available}
        contributions = {name: 100.0 * weights[name] * float(value) for name, value in available.items()}
        # The worst problem must not drown in clean components: a frame with a clear open is at
        # least as risky as that open.
        total = max(sum(contributions.values()) / 100.0, 0.8 * max(topology, contamination, width_raw or 0.0))
        reasons = [
            replace(reason, contribution=_scaled(reason.contribution, multiplier), severity=_severity(_scaled(reason.contribution, multiplier)))
            for reason in findings.reasons
        ]
        reasons.extend(width_reasons)
        reasons.extend(batch_reasons)
        reasons.sort(key=lambda item: item.contribution, reverse=True)
        verdict = _verdict(topology, contamination, width_raw)
        risk_summary = HeuristicCheckSummary(
            total_risk=100.0 * _clip01(total),
            verdict=verdict,
            topology_risk=100.0 * topology,
            contamination_risk=100.0 * contamination,
            width_risk=None if width_raw is None else 100.0 * width_raw,
            geometry_risk=100.0 * geometry,
            batch_outlier_risk=None if batch_raw is None else 100.0 * batch_raw,
            sensitivity=sensitivity_key,
            capture_issues=(),
            reasons=tuple(reasons[:REASONS_KEPT]),
            component_weights=weights,
            component_contributions=contributions,
            feature_values={**findings.extra, **findings.features},
        )
        metric_values = {name: 100.0 * float(value) for name, value in available.items()}
        metric_values[HEURISTIC_CHECK_SCORE] = float(risk_summary.total_risk)
        summary = FrameAnalysisSummary(
            disagreement_score=0.0,
            temporal_instability=0.0,
            structural_anomaly=topology,
            export_priority_score=risk_summary.total_risk / 100.0,
            metric_values=metric_values,
            notes=(verdict, *(reason.code for reason in risk_summary.reasons)),
            frame_type="polygon",
            heuristic_check=risk_summary,
        )
        updated[key] = replace(
            record,
            summary=summary,
            absolute_score=risk_summary.total_risk,
            score=1.0 - risk_summary.total_risk / 100.0,
            relative_score=risk_summary.total_risk / 100.0,
            score_ready=True,
        )
        if state_callback is not None:
            state_callback(key, "ready")
        if progress_callback is not None:
            progress_callback(len(source_records) + offset, total_work, key)

    records = tuple(updated.get(str(record.key), replace(record, score_ready=False)) for record in build_result.records)
    percentile_map = compute_metric_percentiles(records, HEURISTIC_CHECK_SCORE)
    records = tuple(
        replace(record, score_percentile=float(percentile_map.get(record.key, 0.0))) if record.score_ready else record
        for record in records
    )
    scored = [record for record in records if record.score_ready and record.absolute_score is not None]
    available_metrics = [
        key
        for key in HEURISTIC_CHECK_METRICS
        if key == HEURISTIC_CHECK_SCORE
        or any(record.summary is not None and key in record.summary.metric_values for record in records)
    ]
    return replace(
        build_result,
        records=records,
        min_score=min((record.score for record in scored), default=0.0),
        max_score=max((record.score for record in scored), default=0.0),
        scores_computed=True,
        best_match_key=min(scored, key=lambda item: float(item.absolute_score)).key if scored else None,
        min_absolute_score=min((float(record.absolute_score) for record in scored), default=None),
        max_absolute_score=max((float(record.absolute_score) for record in scored), default=None),
        selected_metric_key=HEURISTIC_CHECK_SCORE,
        available_metric_keys=tuple(available_metrics),
    )


def verdict_counts(build_result: BuildResult) -> dict[str, int]:
    """How many frames of the run got each verdict."""

    counts = {verdict: 0 for verdict in VERDICTS}
    for record in build_result.records:
        summary = None if record.summary is None else getattr(record.summary, "heuristic_check", None)
        if summary is not None:
            counts[str(summary.verdict)] = counts.get(str(summary.verdict), 0) + 1
    return counts


__all__ = [
    "CAPTURE_REASONS",
    "CONTAMINATION_REASONS",
    "GEOMETRY_REASONS",
    "HEURISTIC_BATCH_RISK",
    "HEURISTIC_CHECK_METRICS",
    "HEURISTIC_CHECK_SCORE",
    "HEURISTIC_CHECK_VERSION",
    "HEURISTIC_CONTAMINATION_RISK",
    "HEURISTIC_GEOMETRY_RISK",
    "HEURISTIC_TOPOLOGY_RISK",
    "HEURISTIC_WIDTH_RISK",
    "TOPOLOGY_REASONS",
    "VERDICTS",
    "VERDICT_DIRT",
    "VERDICT_ETCH_DEVIATION",
    "VERDICT_NORMAL",
    "VERDICT_TOPOLOGY",
    "VERDICT_UNREADABLE",
    "WIDTH_REASONS",
    "FrameFindings",
    "HeuristicCheckConfig",
    "HeuristicCheckSummary",
    "analyze_frame",
    "compute_heuristic_check",
    "folder_holds_photos",
    "looks_like_photo",
    "verdict_counts",
]
