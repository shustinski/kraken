"""Cell defects of one frame, decided from the mask alone and without a grid.

Every mask component is judged by its own look against the normal-cell bank: no rows,
columns, neighbours or expected cell count. An empty place is never an error; a missed
cell can only be found against ground truth of the same frame (grid_ground_truth).

Order of decisions on one frame:

1. binarize and split into components (mask_normalization, cell_components);
2. conductor fields: IGNORE, with cells next to them protected (conductor_ignore);
3. per component: merge (distance-transform cores, grid_merge_cores), cell-likeness,
   geometry (size per side against the bank's limits, shape against the bank's
   examples, holes), pieces touching the frame border (a flag, decision Н5);
4. pieces that together make one normal cell are a split, before any of them can be
   called debris; what is left and does not look like a cell is debris;
5. the class map shows one class per object in the order IGNORE, SPLIT, MERGE,
   BAD_GEOMETRY, DEBRIS, UNKNOWN (decision Н6); other findings stay in the reasons.

The optional confidence map only adds features and moves the score; the class of
every object is the same with and without it (decision Н2).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - OpenCV is a hard dependency of the app
    cv2 = None

from . import error_classes as ec
from .cell_components import CellComponent, ComponentSet, extract_components
from .conductor_ignore import ConductorConfig, final_ignore
from .grid_merge_cores import (
    _convexity_has_opposing_dents,
    filled_contour_roi,
    find_cell_cores,
    merge_core_thresholds,
)
from .grid_template import CellTemplateMatcher, geometry_thresholds
from .mask_normalization import MaskNormalizationConfig, grayscale_levels
from .normal_bank import SOURCE_REFERENCE, NormalBank, mask_descriptor, shape_descriptor

CELL_ERROR_ALGORITHM_VERSION = "cell_errors.v1"
RESULT_SCHEMA_VERSION = "karakal.cell-errors.v1"

MODE_MASK = "MASK"
MODE_MASK_CONF = "MASK_CONF"
MODE_MASK_REF = "MASK_REF"
MODE_MASK_CONF_REF = "MASK_CONF_REF"

@dataclass(frozen=True, slots=True)
class CellErrorConfig:
    geometry_sensitivity: int = 40
    merge_sensitivity: int = 35
    # The analysis keeps debris down to this size; the operator's larger minimum only hides marks.
    debris_min_area_px: int = 4
    # Shape distance (in units of normal variation) above which a cell's outline is broken:
    # 1.4 (strict slider) .. 2.4 (soft); 2.0 at the default 40. On real masks 2.0 marks 0.1-0.4% of
    # clean cells; synthetic bites and cut corners start at 2.85.
    shape_limit: float = 1.4
    shape_limit_loose: float = 1.0
    # A piece is cell-like when its shape is within this many limits of the bank (a bitten or
    # cut cell is still a cell; synthetic bites and corners lie at 2-3 limits) ...
    cell_like_shape_factor: float = 3.5
    # ... and it is cell-sized: most of a cell, or a solid part of one that keeps one full side.
    full_min_area: float = 0.60
    partial_min_side: float = 0.30
    partial_min_area: float = 0.25
    partial_min_solidity: float = 0.85
    # ... or the normal shape at a reduced size, down to this share of the cell area.
    scaled_min_area: float = 0.35
    # Merges are looked for in pieces of at least this many (largest normal) cell areas.
    merge_min_area: float = 1.40
    # A piece one cell wide and this many cells long is cells in a row.
    row_min_length: float = 1.60
    row_min_solidity: float = 0.80
    # A non-cell piece larger than this many cell areas is an unknown anomaly, not debris.
    unknown_area: float = 2.40
    # Split pieces: gap between them (share of the cell's short side) and shape slack of their union.
    split_gap: float = 0.35
    split_shape_slack: float = 1.5
    # A hole larger than this share of the cell (and this many pixels) breaks it.
    hole_share: float = 0.03
    hole_min_px: int = 24
    # Clipped by the frame: a strip along the edge from this share to this many cells long.
    edge_min_along: float = 0.25
    edge_max_along: float = 1.25
    edge_min_solidity: float = 0.80
    # How far uncertainty moves the score; never across the error threshold.
    confidence_weight: float = 0.25
    conductor: ConductorConfig = field(default_factory=ConductorConfig)
    normalization: MaskNormalizationConfig = field(default_factory=MaskNormalizationConfig)


@dataclass(frozen=True, slots=True)
class ErrorRegion:
    instance_id: int
    error_class: int
    score: float
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    pixel_area: int
    component_ids: tuple[int, ...]
    reasons: tuple[str, ...]
    evidence_sources: tuple[str, ...]
    feature_snapshot: tuple[tuple[str, float], ...]
    touches_border: bool = False
    # Pixels of the object inside its box, packed (np.packbits) to keep runs compact.
    mask_bits: bytes = b""

    def mask(self) -> np.ndarray:
        x, y, w, h = self.bbox
        bits = np.unpackbits(np.frombuffer(self.mask_bits, dtype=np.uint8), count=int(w) * int(h))
        return bits.reshape(int(h), int(w)).astype(bool)

    @property
    def class_name(self) -> str:
        return ec.ERROR_CLASS_NAMES[self.error_class]


@dataclass(frozen=True, slots=True)
class ErrorAnalysisResult:
    """Analysis of one frame. Maps are drawn from the regions on request (see error_maps)."""

    frame_id: str
    frame_path: str
    mode: str
    algorithm_version: str
    normal_model_id: str
    normal_model_status: str
    input_shape: tuple[int, int]
    binarization: dict[str, Any]
    inputs: dict[str, bool]
    regions: tuple[ErrorRegion, ...]
    ignore_regions: tuple[ErrorRegion, ...]
    diagnostics: dict[str, Any]
    schema_version: str = RESULT_SCHEMA_VERSION
    # Normal cells (class OK), kept so a frame's share of defects and the matrix counts match
    # what the operator sees. Not painted on the class map.
    normal_regions: tuple[ErrorRegion, ...] = ()

    @property
    def binarization_threshold(self) -> float:
        return float(self.binarization.get("threshold", 0.0))

    def class_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for region in self.regions:
            counts[region.class_name] = counts.get(region.class_name, 0) + 1
        return counts

    def class_map(self) -> np.ndarray:
        from .error_maps import render_class_map

        return render_class_map(self)

    def score_map(self) -> np.ndarray:
        from .error_maps import render_score_map

        return render_score_map(self)

    def instance_map(self) -> np.ndarray:
        from .error_maps import render_instance_map

        return render_instance_map(self)


def detect_mode(*, has_confidence: bool, bank: NormalBank | None) -> str:
    reference = bank is not None and bank.source == SOURCE_REFERENCE
    if has_confidence:
        return MODE_MASK_CONF_REF if reference else MODE_MASK_CONF
    return MODE_MASK_REF if reference else MODE_MASK


def _logistic(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, value))))


def _pack(piece: np.ndarray) -> bytes:
    return np.packbits(np.asarray(piece, dtype=bool).reshape(-1)).tobytes()


def confidence_uncertainty(confidence: np.ndarray, shape: tuple[int, int], foreground: np.ndarray) -> tuple[np.ndarray, str]:
    """0..1 uncertainty map normalized by the map itself (decision Н4: no fixed 0.35-0.65 band)."""

    from .grid_anomaly import classify_confidence_map_kind

    gray = grayscale_levels(confidence)
    if gray.shape != tuple(shape):
        gray = cv2.resize(gray, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    kind = classify_confidence_map_kind(gray)
    values = gray.astype(np.float32)
    if kind == "class_probability":
        return 1.0 - np.abs(2.0 * values / 255.0 - 1.0), kind
    inside = values[foreground] if np.any(foreground) else values.reshape(-1)
    norm = float(np.median(inside))
    mad = float(np.median(np.abs(inside - norm)))
    span = max(8.0, 3.0 * mad)
    return np.clip((norm - values) / (2.0 * span), 0.0, 1.0), kind


def _confidence_features(uncertainty: np.ndarray, piece: np.ndarray, x0: int, y0: int) -> dict[str, float]:
    region = uncertainty[y0 : y0 + piece.shape[0], x0 : x0 + piece.shape[1]]
    values = region[piece]
    if values.size == 0:
        return {}
    eroded = cv2.erode(piece.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    border = piece & ~eroded
    border_values = region[border] if np.any(border) else values
    return {
        "confidence_uncertainty_mean": float(values.mean()),
        "confidence_uncertainty_p90": float(np.quantile(values, 0.90)),
        "confidence_uncertain_share": float(np.mean(values >= 0.5)),
        "confidence_border_uncertainty": float(border_values.mean()),
    }


@dataclass(slots=True)
class _Verdict:
    component: CellComponent
    error_class: int = ec.OK
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)
    features: dict[str, float] = field(default_factory=dict)
    split_group: int = -1


class _FrameJudge:
    """Bank limits and cached shape distances for one frame."""

    def __init__(self, components: ComponentSet, bank: NormalBank, cfg: CellErrorConfig) -> None:
        self.components = components
        self.bank = bank
        self.cfg = cfg
        looseness = 1.0 - max(0, min(100, int(cfg.geometry_sensitivity))) / 100.0
        self.shape_limit = float(cfg.shape_limit) + float(cfg.shape_limit_loose) * looseness
        (self.w_lo, self.w_hi), (self.h_lo, self.h_hi) = bank.side_limits(cfg.geometry_sensitivity)
        width_scale = bank.width_range[1] / max(1.0, bank.cell_width) if bank.width_variable else 1.0
        height_scale = bank.height_range[1] / max(1.0, bank.cell_height) if bank.height_variable else 1.0
        self.full_area = float(bank.cell_area) * width_scale * height_scale
        self.matcher = CellTemplateMatcher(bank.template) if bank.template is not None else None
        self.depth_limit = geometry_thresholds(bank.template, cfg.geometry_sensitivity)[3] if bank.template else 0.0
        self._distance: dict[int, float] = {}

    def distances(self, items: list[CellComponent]) -> None:
        todo = [item for item in items if item.label not in self._distance]
        if not todo:
            return
        descriptors = np.stack([shape_descriptor(item, self.components.labels) for item in todo])
        for item, value in zip(todo, self.bank.normal_distance(descriptors)):
            self._distance[item.label] = float(value)

    def distance(self, item: CellComponent) -> float:
        if item.label not in self._distance:
            self.distances([item])
        return self._distance[item.label]

    def full_size(self, item: CellComponent) -> bool:
        return self.w_lo <= item.width <= self.w_hi and self.h_lo <= item.height <= self.h_hi

    def partial_size(self, item: CellComponent) -> bool:
        cfg, bank = self.cfg, self.bank
        if item.solidity < cfg.partial_min_solidity or item.area < cfg.partial_min_area * bank.cell_area:
            return False
        keeps_width = self.w_lo <= item.width <= self.w_hi and item.height >= cfg.partial_min_side * bank.cell_height
        keeps_height = self.h_lo <= item.height <= self.h_hi and item.width >= cfg.partial_min_side * bank.cell_width
        return keeps_width or keeps_height

    def cell_like(self, item: CellComponent) -> bool:
        """A cell, perhaps a broken one (decision Н6): cell-sized and loosely the normal shape."""

        sized = (
            self.cfg.full_min_area * self.bank.cell_area <= item.area <= self.cfg.unknown_area * self.full_area
        ) or self.partial_size(item)
        if sized and self.distance(item) <= self.cfg.cell_like_shape_factor * self.shape_limit:
            return True
        # A cell drawn too small: the normal shape at a reduced size is a wrong-size cell, not debris.
        return (
            item.area >= self.cfg.scaled_min_area * self.bank.cell_area
            and item.area <= self.cfg.unknown_area * self.full_area
            and self.distance(item) <= self.shape_limit
        )

    def protected_from_conductor(self, item: CellComponent) -> bool:
        """Confidently a normal cell (shape within the normal limit, crisp edges), or a cell cut by
        the frame. Only these are cut out of a conductor field; a cell-sized lump of grain is not."""

        if self.clipped_by_edge(item):
            return True
        crisp = item.edge_softness < self.cfg.conductor.soft_edge
        return crisp and self.full_size(item) and self.distance(item) <= self.shape_limit

    def clipped_by_edge(self, item: CellComponent) -> bool:
        """A strip along the frame edge that a cell cut by the frame leaves (decision Н5)."""

        if not item.touches_border or item.solidity < self.cfg.edge_min_solidity:
            return False
        sides = set(item.border_sides)
        horizontal = bool(sides & {"left", "right"})
        vertical = bool(sides & {"top", "bottom"})
        width_cells = item.width / max(1.0, self.bank.cell_width)
        height_cells = item.height / max(1.0, self.h_hi / 1.12 if self.bank.height_variable else self.bank.cell_height)
        if horizontal and vertical:
            return width_cells <= 1.0 + 0.1 and height_cells <= 1.0 + 0.1
        if horizontal:
            return width_cells <= 1.1 and self.cfg.edge_min_along <= height_cells <= self.cfg.edge_max_along
        return height_cells <= 1.1 and self.cfg.edge_min_along <= width_cells <= self.cfg.edge_max_along

    def size_deviation(self, item: CellComponent) -> float:
        def outside(value: float, low: float, high: float) -> float:
            return max(0.0, (low - value) / max(1.0, low), (value - high) / max(1.0, high))

        return max(outside(item.width, self.w_lo, self.w_hi), outside(item.height, self.h_lo, self.h_hi))

    def template_deep(self, item: CellComponent) -> tuple[float, int]:
        if self.matcher is None:
            return 0.0, 0
        return self.matcher.measure_points(item.contour, self.depth_limit)


def _merged_parts(piece: np.ndarray, judge: _FrameJudge, in_a_row: bool, along: int) -> tuple[list[np.ndarray], bool]:
    """Cells inside a merged piece, and whether a neck or bridge separated them.

    Opening at 0.7 of the cell's short side cuts bridges and necks thinner than that and keeps cells.
    A solid block with no neck is cut into equal cells along its long side. (A watershed from
    the cores puts the cut anywhere along a flat bridge, the plateau case of Roerdink & Meijster.)
    """

    bank = judge.bank
    side = max(3, int(round(0.7 * min(bank.cell_width, bank.cell_height))) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (side, side))
    opened = cv2.morphologyEx(np.pad(piece.astype(np.uint8), side), cv2.MORPH_OPEN, kernel)[side:-side, side:-side]
    count, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    big = [index for index in range(1, count) if stats[index, cv2.CC_STAT_AREA] >= 0.5 * bank.cell_area]
    if len(big) >= 2:
        return [labels == index for index in big], True
    if not in_a_row:
        return [], False
    rows, cols = piece.shape
    parts = []
    count_along = max(2, int(along))
    if cols >= rows:
        edges = np.linspace(0, cols, count_along + 1).astype(int)
        for left, right in zip(edges[:-1], edges[1:]):
            part = np.zeros_like(piece)
            part[:, left:right] = piece[:, left:right]
            parts.append(part)
    else:
        edges = np.linspace(0, rows, count_along + 1).astype(int)
        for top, bottom in zip(edges[:-1], edges[1:]):
            part = np.zeros_like(piece)
            part[top:bottom, :] = piece[top:bottom, :]
            parts.append(part)
    return parts, False


def _cell_like_parts(judge: _FrameJudge, parts: list[np.ndarray]) -> int:
    bank = judge.bank
    good = 0
    for part in parts:
        ys, xs = np.nonzero(part)
        if ys.size < judge.cfg.full_min_area * bank.cell_area:
            continue
        crop = part[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
        distance = float(bank.normal_distance(mask_descriptor(crop)[None, :])[0])
        if distance <= judge.cfg.cell_like_shape_factor * judge.shape_limit:
            good += 1
    return good


def _merge_verdict(judge: _FrameJudge, item: CellComponent, cfg: CellErrorConfig) -> tuple[float, list[str], dict[str, float]]:
    """Several cells in one piece, confirmed by the cells themselves.

    The piece must be clearly larger than one cell. It is then cut at its necks and bridges
    (or, a solid block one cell wide, into equal cells), and at least two parts must look like
    normal cells: a ragged lump of conductor grain does not. Distance-transform cores and
    opposing dents (grid_merge_cores) add to the score.
    """

    bank = judge.bank
    reasons: list[str] = []
    features: dict[str, float] = {}
    if item.area < cfg.merge_min_area * judge.full_area:
        return 0.0, reasons, features
    min_core_frac, _area_tol = merge_core_thresholds(int(cfg.merge_sensitivity))
    roi, _ox, _oy = filled_contour_roi(item.contour, item.bbox)
    piece = roi > 0
    distance = cv2.distanceTransform(piece.astype(np.uint8), cv2.DIST_L2, 3)
    cores = find_cell_cores(roi, bank.cell_width, bank.cell_height, min_core_frac=min_core_frac, distance=distance)
    dents = _convexity_has_opposing_dents(item.contour, min_depth=0.22 * min(bank.cell_width, bank.cell_height))
    long_width = judge.w_lo <= item.width <= judge.w_hi and item.height >= cfg.row_min_length * judge.h_hi / 1.12
    long_height = judge.h_lo <= item.height <= judge.h_hi and item.width >= cfg.row_min_length * judge.w_hi / 1.12
    in_a_row = (long_width or long_height) and item.solidity >= cfg.row_min_solidity
    along = round(max(item.width / max(1.0, bank.cell_width), item.height / max(1.0, judge.h_hi / 1.12)))
    parts, necked = _merged_parts(piece, judge, in_a_row, along)
    good = _cell_like_parts(judge, parts)
    multiple = float(item.area) / max(1.0, judge.full_area)
    features.update(
        {
            "merge_cores": float(len(cores)),
            "merge_dents": float(dents),
            "merge_neck": float(necked),
            "merge_area_cells": multiple,
            "merge_cell_parts": float(good),
        }
    )
    if good < 2:
        return 0.0, reasons, features
    evidence = float(good - 1) + (0.35 if necked or dents else 0.0) + (0.25 if len(cores) >= 2 else 0.0)
    score = max(0.5, 1.0 / (1.0 + math.exp(-(evidence - 0.55) / 0.35)))
    features["merge_score"] = score
    reasons.extend(["merge_cells", "merge_neck"] if necked else ["cells_in_a_row"])
    return score, reasons, features


def _find_splits(judge: _FrameJudge, verdicts: dict[int, _Verdict], candidates: list[CellComponent], cfg: CellErrorConfig) -> int:
    """Group pieces whose union is one normal cell. Returns the number of groups."""

    if len(candidates) < 2:
        return 0
    labels = judge.components.labels
    bank = judge.bank
    gap_limit = max(2.0, cfg.split_gap * min(bank.cell_width, bank.cell_height))
    side = 2 * int(math.ceil(gap_limit / 2.0)) + 1
    close_kernel = np.ones((side, side), dtype=np.uint8)
    parent = {item.label: item.label for item in candidates}

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union_box(boxes) -> tuple[int, int, int, int]:
        x0 = min(b[0] for b in boxes)
        y0 = min(b[1] for b in boxes)
        x1 = max(b[0] + b[2] for b in boxes)
        y1 = max(b[1] + b[3] for b in boxes)
        return x0, y0, x1 - x0, y1 - y0

    def union_matches(items: list[CellComponent]) -> bool:
        x, y, w, h = union_box([item.bbox for item in items])
        if not (judge.w_lo <= w <= judge.w_hi and judge.h_lo <= h <= judge.h_hi):
            return False
        crop = labels[y : y + h, x : x + w]
        piece = np.isin(crop, [item.label for item in items])
        if int(np.count_nonzero(piece)) < cfg.full_min_area * bank.cell_area * (h / max(1.0, bank.cell_height) if bank.height_variable else 1.0):
            return False
        # The cut itself is expected: close it before comparing the union with normal cells.
        pad = close_kernel.shape[0]
        closed = cv2.morphologyEx(np.pad(piece.astype(np.uint8), pad), cv2.MORPH_CLOSE, close_kernel)[pad:-pad, pad:-pad]
        distance = float(bank.normal_distance(mask_descriptor(closed > 0)[None, :])[0])
        return distance <= cfg.split_shape_slack * judge.shape_limit

    centers = np.asarray([[item.bbox[0] + item.bbox[2] / 2.0, item.bbox[1] + item.bbox[3] / 2.0] for item in candidates])
    reach = float(max(judge.w_hi, judge.h_hi))
    for i, first in enumerate(candidates):
        close = np.flatnonzero(np.abs(centers[:, 0] - centers[i, 0]) + np.abs(centers[:, 1] - centers[i, 1]) <= 1.5 * reach)
        for j in close:
            if j <= i:
                continue
            second = candidates[int(j)]
            a, b = first.bbox, second.bbox
            gap_x = max(0, max(a[0], b[0]) - min(a[0] + a[2], b[0] + b[2]))
            gap_y = max(0, max(a[1], b[1]) - min(a[1] + a[3], b[1] + b[3]))
            if math.hypot(gap_x, gap_y) > gap_limit:
                continue
            root_a, root_b = find(first.label), find(second.label)
            if root_a == root_b:
                continue
            members = [item for item in candidates if find(item.label) in (root_a, root_b)]
            if union_matches(members):
                parent[root_b] = root_a
    groups: dict[int, list[CellComponent]] = {}
    for item in candidates:
        groups.setdefault(find(item.label), []).append(item)
    count = 0
    for members in groups.values():
        if len(members) < 2:
            continue
        count += 1
        for item in members:
            verdict = verdicts[item.label]
            verdict.split_group = count
    return count


def analyze_cell_errors(
    mask_image: np.ndarray,
    bank: NormalBank,
    *,
    confidence: np.ndarray | None = None,
    config: CellErrorConfig | None = None,
    frame_id: str = "",
    frame_path: str = "",
) -> ErrorAnalysisResult:
    """Classes, scores and regions of one frame. The bank is fixed for the whole run."""

    cfg = config or CellErrorConfig()
    started = perf_counter()
    components = extract_components(mask_image, normalization=cfg.normalization)
    shape = components.shape
    mode = detect_mode(has_confidence=confidence is not None, bank=bank)
    sources = ["mask", "normal_bank"] + (["confidence"] if confidence is not None else [])
    diagnostics: dict[str, Any] = {
        "components": len(components.components),
        "normal_model_reason": bank.reason,
        "cell_width": round(float(bank.cell_width), 2),
        "cell_height": round(float(bank.cell_height), 2),
    }

    uncertainty = None
    if confidence is not None:
        uncertainty, kind = confidence_uncertainty(confidence, shape, components.normalized.binary)
        diagnostics["confidence_kind"] = kind

    if not bank.ready or components.normalized.status in ("empty", "full"):
        # Without a reliable normal cell nothing can be called broken; conductors cannot be told
        # from cells either. The frame is reported, not guessed (cold start safeguard).
        diagnostics["conservative"] = True
        diagnostics["elapsed_ms"] = round((perf_counter() - started) * 1000.0, 1)
        return ErrorAnalysisResult(
            frame_id=str(frame_id),
            frame_path=str(frame_path),
            mode=mode,
            algorithm_version=CELL_ERROR_ALGORITHM_VERSION,
            normal_model_id=bank.bank_id if bank.exemplars.size else "",
            normal_model_status=bank.status,
            input_shape=shape,
            binarization=components.normalized.metadata(),
            inputs={"mask": True, "confidence": confidence is not None, "reference": bank.source == SOURCE_REFERENCE},
            regions=(),
            ignore_regions=(),
            diagnostics=diagnostics,
        )

    judge = _FrameJudge(components, bank, cfg)
    sized = [item for item in components.components if item.area >= max(8, int(0.15 * bank.cell_area))]
    judge.distances(sized)

    zones = final_ignore(
        components,
        cell_area=bank.cell_area,
        is_cell_like=judge.protected_from_conductor,
        config=cfg.conductor,
    )
    ignored = set(zones.ignored_labels)
    near = set(zones.near_labels) | set(zones.protected_labels)
    diagnostics["conductor_window"] = zones.window

    verdicts: dict[int, _Verdict] = {}
    split_candidates: list[CellComponent] = []
    for item in components.components:
        verdict = _Verdict(item)
        verdicts[item.label] = verdict
        if item.label in ignored:
            verdict.error_class = ec.IGNORE
            continue
        if item.area < cfg.debris_min_area_px:
            continue
        if item.label in near:
            verdict.reasons.append("near_conductor")
        if item.touches_border:
            verdict.reasons.append("touches_frame_border")
        distance = judge.distance(item) if item.area >= max(8, int(0.15 * bank.cell_area)) else float("inf")
        verdict.features.update(
            {
                "width_ratio": item.width / max(1.0, bank.cell_width),
                "height_ratio": item.height / max(1.0, bank.cell_height),
                "area_ratio": item.area / max(1.0, bank.cell_area),
                "normal_distance": float(min(distance, 99.0)),
                "solidity": item.solidity,
                "extent": item.extent,
                "compactness": item.compactness,
                "contour_area_px": float(item.area),
            }
        )
        merge_score, merge_reasons, merge_features = _merge_verdict(judge, item, cfg)
        verdict.features.update(merge_features)
        if merge_reasons:
            verdict.error_class = ec.MERGE
            verdict.score = merge_score
            verdict.reasons.extend(merge_reasons)
            continue
        if item.touches_border:
            # A cell cut by the frame is not judged by size or outline (decision Н5).
            if judge.clipped_by_edge(item) or judge.cell_like(item):
                continue
            if item.area > cfg.unknown_area * judge.full_area:
                verdict.error_class = ec.UNKNOWN
                verdict.score = 0.6
                verdict.reasons.append("large_irregular")
            elif item.area < cfg.full_min_area * bank.cell_area:
                verdict.error_class = ec.DEBRIS
                verdict.score = 0.75
                verdict.reasons.append("not_cell_like")
            continue
        cell_like = judge.cell_like(item)
        verdict.features["cell_like"] = float(cell_like)
        if cell_like:
            size_off = not judge.full_size(item)
            size_dev = judge.size_deviation(item)
            shape_ratio = distance / judge.shape_limit
            deviation, deep = judge.template_deep(item)
            verdict.features.update(
                {"size_deviation": size_dev, "shape_ratio": shape_ratio, "template_deviation": deviation, "template_deep_pixels": float(deep)}
            )
            reasons = []
            if size_off:
                if not judge.w_lo <= item.width <= judge.w_hi:
                    reasons.append("size_width")
                if not judge.h_lo <= item.height <= judge.h_hi:
                    reasons.append("size_height")
            # The mean-template depth test only describes the cell here: on real masks it calls the
            # network's wavy edges broken, which the bank of normal examples already accepts.
            if shape_ratio > 1.0:
                reasons.append("shape")
            if item.hole_area >= max(cfg.hole_min_px, cfg.hole_share * item.filled_area):
                reasons.append("hole")
            closeness = max(shape_ratio, 1.0 + 5.0 * size_dev if size_off else 0.0)
            score = _logistic((closeness - 1.0) / 0.15)
            if reasons:
                verdict.error_class = ec.BAD_GEOMETRY
                verdict.score = max(0.5, score)
                verdict.reasons.extend(reasons)
            else:
                verdict.score = min(0.49, score)
            # A partial piece may still be one half of a split cell.
            if not judge.full_size(item):
                split_candidates.append(item)
            continue
        if item.area <= cfg.merge_min_area * judge.full_area:
            split_candidates.append(item)
        if item.area > cfg.unknown_area * judge.full_area:
            verdict.error_class = ec.UNKNOWN
            verdict.score = 0.6
            verdict.reasons.append("large_irregular")
        else:
            verdict.error_class = ec.DEBRIS
            verdict.score = 0.75 + 0.2 * max(0.0, 1.0 - item.area / max(1.0, bank.cell_area))
            verdict.reasons.append("not_cell_like")

    split_groups = _find_splits(judge, verdicts, split_candidates, cfg)
    diagnostics["split_groups"] = split_groups

    if uncertainty is not None:
        for verdict in verdicts.values():
            if verdict.error_class == ec.IGNORE or verdict.component.area < cfg.debris_min_area_px:
                continue
            piece, x0, y0 = verdict.component.crop(components.labels)
            verdict.features.update(_confidence_features(uncertainty, piece, x0, y0))

    regions: list[ErrorRegion] = []
    grouped: dict[int, list[_Verdict]] = {}
    for verdict in verdicts.values():
        if verdict.split_group >= 0:
            grouped.setdefault(verdict.split_group, []).append(verdict)
    labels = components.labels

    def confidence_adjusted(score: float, features: dict[str, float], is_error: bool) -> tuple[float, list[str]]:
        share = features.get("confidence_uncertain_share")
        if share is None:
            return score, []
        moved = float(score) + cfg.confidence_weight * (float(share) - 0.1)
        moved = min(1.0, max(0.5, moved)) if is_error else max(0.0, min(0.49, moved))
        return moved, (["low_confidence"] if share >= 0.3 else [])

    normal_regions: list[ErrorRegion] = []

    def add_region(
        members: list[_Verdict],
        error_class: int,
        score: float,
        reasons: list[str],
        features: dict[str, float],
        target: list[ErrorRegion] | None = None,
    ) -> None:
        bucket = regions if target is None else target
        boxes = [item.component.bbox for item in members]
        x0 = min(b[0] for b in boxes)
        y0 = min(b[1] for b in boxes)
        x1 = max(b[0] + b[2] for b in boxes)
        y1 = max(b[1] + b[3] for b in boxes)
        ids = [item.component.label for item in members]
        piece = np.isin(labels[y0:y1, x0:x1], ids)
        area = int(np.count_nonzero(piece))
        ys, xs = np.nonzero(piece)
        score, extra = confidence_adjusted(score, features, error_class != ec.OK)
        bucket.append(
            ErrorRegion(
                instance_id=len(bucket) + 1 if error_class != ec.OK else 0,
                error_class=int(error_class),
                score=float(score),
                bbox=(int(x0), int(y0), int(x1 - x0), int(y1 - y0)),
                centroid=(float(x0 + xs.mean()), float(y0 + ys.mean())),
                pixel_area=area,
                component_ids=tuple(sorted(int(value) for value in ids)),
                reasons=tuple(dict.fromkeys(reasons + extra)),
                evidence_sources=tuple(sources),
                feature_snapshot=tuple((key, float(value)) for key, value in sorted(features.items())),
                touches_border=any(item.component.touches_border for item in members),
                mask_bits=_pack(piece),
            )
        )

    done: set[int] = set()
    for group, members in sorted(grouped.items()):
        reasons = ["split_fragments"]
        for item in members:
            # Other findings on a fragment stay as reasons (decision Н6).
            reasons.extend(f"fragment_{reason}" for reason in item.reasons if reason not in ("touches_frame_border",))
        features = {"fragment_count": float(len(members))}
        for item in members:
            for key, value in item.features.items():
                if key.startswith("confidence_"):
                    features[key] = max(features.get(key, 0.0), value)
        add_region(members, ec.SPLIT, 0.8, reasons, features)
        done.update(item.component.label for item in members)
    for verdict in sorted(verdicts.values(), key=lambda item: (item.component.bbox[1], item.component.bbox[0])):
        if verdict.component.label in done or verdict.error_class == ec.IGNORE:
            continue
        if verdict.error_class == ec.OK:
            item = verdict.component
            # Normal cells and cells cut by the frame; crumbs below the analysis floor are no objects.
            if item.area >= cfg.debris_min_area_px and (judge.cell_like(item) or judge.clipped_by_edge(item)):
                add_region([verdict], ec.OK, verdict.score, list(verdict.reasons), dict(verdict.features), normal_regions)
            continue
        add_region([verdict], verdict.error_class, verdict.score, list(verdict.reasons), dict(verdict.features))

    ignore_regions: list[ErrorRegion] = []
    if zones.found:
        count, zone_labels, stats, _centroids = cv2.connectedComponentsWithStats(zones.zone.astype(np.uint8), connectivity=8)
        for index in range(1, int(count)):
            x, y, w, h, area = (int(value) for value in stats[index, :5])
            piece = zone_labels[y : y + h, x : x + w] == index
            ys, xs = np.nonzero(piece)
            inside = tuple(sorted(int(value) for value in np.unique(labels[y : y + h, x : x + w][piece]) if value in ignored))
            ignore_regions.append(
                ErrorRegion(
                    instance_id=index,
                    error_class=ec.IGNORE,
                    score=0.0,
                    bbox=(x, y, w, h),
                    centroid=(float(x + xs.mean()), float(y + ys.mean())),
                    pixel_area=area,
                    component_ids=inside,
                    reasons=("conductor_zone",),
                    evidence_sources=("mask",),
                    feature_snapshot=(),
                    mask_bits=_pack(piece),
                )
            )

    ok_scores = [verdict.score for verdict in verdicts.values() if verdict.error_class == ec.OK and verdict.score > 0]
    diagnostics.update(
        {
            "objects_ok": len(normal_regions),
            "objects_ignored": len(ignored),
            "near_threshold_ok": sum(1 for value in ok_scores if value >= 0.25),
            "elapsed_ms": round((perf_counter() - started) * 1000.0, 1),
        }
    )
    return ErrorAnalysisResult(
        frame_id=str(frame_id),
        frame_path=str(frame_path),
        mode=mode,
        algorithm_version=CELL_ERROR_ALGORITHM_VERSION,
        normal_model_id=bank.bank_id,
        normal_model_status=bank.status,
        input_shape=shape,
        binarization=components.normalized.metadata(),
        inputs={"mask": True, "confidence": confidence is not None, "reference": bank.source == SOURCE_REFERENCE},
        regions=tuple(regions),
        ignore_regions=tuple(ignore_regions),
        diagnostics=diagnostics,
        normal_regions=tuple(normal_regions),
    )
