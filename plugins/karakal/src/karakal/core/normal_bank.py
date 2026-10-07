"""Bank of normal cells: one cell type, several normal shape examples.

A layer has one kind of cell with one size range (decisions В4, Н1). The bank keeps
many normal examples of that cell instead of one mean picture, so the network's honest
variation (rounder or sharper corners, slightly uneven sides) stays normal while a bite,
a cut corner or a hole does not. Shape is compared with the size taken out; size is
checked on its own, per side. A side that normal cells keep constant is checked strictly
(another size is a geometry error). A side that varies by design (cells drawn at several
lengths) is a range, and only a clear fragment of it is an error.

A shape is judged by its distance to the k-th nearest example (k = 4 by default), not
the nearest one: a defective cell that slipped into the bank then cannot excuse its
look-alikes. The bank is built with outlier trimming and capped by greedy k-center
selection (the PatchCore coreset), which keeps rare normal shapes that random sampling
would drop.

A session bank comes from the frames of the current run, a reference bank from operator
masks. They are separate objects; nothing in a run changes a saved reference bank.
"""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - OpenCV is a hard dependency of the app
    cv2 = None

from .cell_components import CellComponent, ComponentSet
from .conductor_ignore import ConductorConfig, excluded_from_bank, preliminary_zones, typical_cell_area
from .grid_template import CellShapeTemplate, build_cell_shape_template

NORMAL_BANK_VERSION = "normal_bank.v1"
STATUS_OK = "ok"
STATUS_INSUFFICIENT = "insufficient_normal_model"
SOURCE_SESSION = "session"
SOURCE_REFERENCE = "reference"

DESCRIPTOR_GRID = 16
# Weight of each scalar shape feature next to the 16x16 silhouette.
SCALAR_WEIGHT = 8.0


@dataclass(frozen=True, slots=True)
class NormalBankConfig:
    min_seed_area: int = 40
    seeds_per_frame: int = 200
    max_pool: int = 3000
    max_exemplars: int = 500
    knn_k: int = 4
    trim_share: float = 0.10
    trim_rounds: int = 2
    min_exemplars: int = 12
    # Share of the cell-sized pieces that must fall in the dominant cluster (cold start assumption).
    min_dominance: float = 0.5
    # A modal size below this is crumbs, not cells.
    min_cell_area: float = 36.0
    # Pieces with a clean outline take part in finding the cell size.
    min_clean_solidity: float = 0.85
    min_clean_extent: float = 0.60
    # Pieces drawn with a wide gray fringe (conductor grain) never enter the bank.
    max_edge_softness: float = 0.15
    # Members of the dominant cluster: each fixed side within this share of its mode.
    side_window: float = 0.25
    # A side whose relative spread (MAD / median) among the cluster exceeds this varies by design.
    variable_side_spread: float = 0.10
    # On a varying side, pieces shorter than this share of the long normal cells are fragments.
    variable_side_min_fraction: float = 0.60
    # Quantiles that bound a varying side.
    side_quantiles: tuple[float, float] = (0.05, 0.98)
    # Quantile of normal examples' distances that defines distance 1.0.
    norm_quantile: float = 0.95
    # Floor of that unit: about one pixel of edge shift. Identical examples (synthetic masks)
    # would otherwise make every difference look huge.
    min_distance_norm: float = 1.5


def shape_descriptor(component: CellComponent, labels: np.ndarray) -> np.ndarray:
    """Size-free silhouette (16x16 of the box, area-averaged) plus four scalar shape features."""

    piece, _x0, _y0 = component.crop(labels)
    grid = cv2.resize(piece.astype(np.float32), (DESCRIPTOR_GRID, DESCRIPTOR_GRID), interpolation=cv2.INTER_AREA)
    hole_share = float(component.hole_area) / float(max(1, component.filled_area))
    scalars = np.asarray(
        [component.solidity, component.extent, component.compactness, hole_share], dtype=np.float32
    ) * np.float32(SCALAR_WEIGHT)
    return np.concatenate([grid.reshape(-1), scalars]).astype(np.float32)


def mask_descriptor(piece: np.ndarray) -> np.ndarray:
    """The same descriptor for any pixel mask cropped to its box, e.g. the union of a split cell's pieces."""

    mask = np.asarray(piece, dtype=bool)
    area = int(np.count_nonzero(mask))
    rows, cols = mask.shape
    grid = cv2.resize(mask.astype(np.float32), (DESCRIPTOR_GRID, DESCRIPTOR_GRID), interpolation=cv2.INTER_AREA)
    padded = np.pad(mask.astype(np.uint8), 1)
    found = cv2.findContours(padded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = found[0] if len(found) == 2 else found[1]
    perimeter = float(sum(cv2.arcLength(contour, True) for contour in contours)) or float(2 * (rows + cols))
    filled = np.zeros_like(padded)
    cv2.drawContours(filled, list(contours), -1, 1, thickness=-1)
    points = np.concatenate([contour.reshape(-1, 2) for contour in contours]) if contours else np.zeros((0, 2), np.int32)
    hull_area = float(cv2.contourArea(cv2.convexHull(points))) if len(points) >= 3 else 0.0
    contour_area = float(sum(cv2.contourArea(contour) for contour in contours))
    filled_area = max(area, int(np.count_nonzero(filled)))
    scalars = np.asarray(
        [
            min(1.0, contour_area / hull_area) if hull_area > 0.0 else 1.0,
            float(area) / float(max(1, rows * cols)),
            float(4.0 * np.pi * area / max(perimeter * perimeter, 1.0)),
            float(filled_area - area) / float(max(1, filled_area)),
        ],
        dtype=np.float32,
    ) * np.float32(SCALAR_WEIGHT)
    return np.concatenate([grid.reshape(-1), scalars]).astype(np.float32)


def _pairwise(queries: np.ndarray, bank: np.ndarray) -> np.ndarray:
    q = queries.astype(np.float64)
    b = bank.astype(np.float64)
    squared = (q * q).sum(axis=1)[:, None] + (b * b).sum(axis=1)[None, :] - 2.0 * q @ b.T
    return np.sqrt(np.maximum(squared, 0.0))


def _kth(distances: np.ndarray, k: int) -> np.ndarray:
    k = max(1, min(int(k), distances.shape[1]))
    return np.partition(distances, k - 1, axis=1)[:, k - 1]


def _greedy_coreset(points: np.ndarray, size: int) -> np.ndarray:
    """Indexes of a k-center subset: each new point is the one farthest from those taken."""

    if len(points) <= size:
        return np.arange(len(points))
    center = points.mean(axis=0, keepdims=True)
    first = int(np.argmin(_pairwise(points, center)[:, 0]))
    chosen = [first]
    nearest = _pairwise(points, points[first : first + 1])[:, 0]
    for _ in range(size - 1):
        index = int(np.argmax(nearest))
        chosen.append(index)
        nearest = np.minimum(nearest, _pairwise(points, points[index : index + 1])[:, 0])
    return np.asarray(sorted(chosen))


def _spread(values: np.ndarray) -> float:
    median = float(np.median(values))
    return float(1.4826 * np.median(np.abs(values - median)) / max(1.0, median))


@dataclass(frozen=True, slots=True)
class NormalBank:
    source: str
    status: str
    cell_width: float = 0.0
    cell_height: float = 0.0
    cell_area: float = 0.0
    width_spread: float = 0.0
    height_spread: float = 0.0
    area_spread: float = 0.0
    # Per side: does it vary by design, and the (low, high) quantiles of normal cells.
    width_variable: bool = False
    height_variable: bool = False
    width_range: tuple[float, float] = (0.0, 0.0)
    height_range: tuple[float, float] = (0.0, 0.0)
    exemplars: np.ndarray = field(default_factory=lambda: np.zeros((0, DESCRIPTOR_GRID**2 + 4), dtype=np.float32))
    distance_norm: float = 1.0
    knn_k: int = 4
    template: CellShapeTemplate | None = None
    frames_used: int = 0
    candidates: int = 0
    dominance: float = 0.0
    reason: str = ""
    version: str = NORMAL_BANK_VERSION

    @property
    def ready(self) -> bool:
        return self.status == STATUS_OK

    @property
    def bank_id(self) -> str:
        digest = hashlib.sha1()
        digest.update(json.dumps(self._header(), sort_keys=True).encode("utf-8"))
        digest.update(np.ascontiguousarray(self.exemplars, dtype=np.float16).tobytes())
        if self.template is not None:
            digest.update(self.template.mask)
        return f"{self.source}:{digest.hexdigest()[:16]}"

    def size_ratios(self, component: CellComponent) -> tuple[float, float, float]:
        return (
            float(component.width) / max(1.0, self.cell_width),
            float(component.height) / max(1.0, self.cell_height),
            float(component.area) / max(1.0, self.cell_area),
        )

    def side_limits(self, geometry_sensitivity: int = 40) -> tuple[tuple[float, float], tuple[float, float]]:
        """(low, high) pixel limits of width and height for one geometry slider position."""

        unit = max(0.0, min(100.0, float(geometry_sensitivity))) / 100.0
        looseness = 1.0 - unit

        def limits(median: float, spread: float, variable: bool, bounds: tuple[float, float]) -> tuple[float, float]:
            tolerance = max(0.12, 3.0 * float(spread)) + 0.10 * looseness
            if not variable:
                return median * (1.0 - tolerance), median * (1.0 + tolerance)
            # A strict slider calls shorter pieces fragments sooner: 0.50 .. 0.70 of the long cells.
            fraction = 0.50 + 0.20 * unit
            return fraction * float(bounds[1]), float(bounds[1]) * (1.12 + 0.10 * looseness)

        return (
            limits(self.cell_width, self.width_spread, self.width_variable, self.width_range),
            limits(self.cell_height, self.height_spread, self.height_variable, self.height_range),
        )

    def normal_distance(self, descriptors: np.ndarray) -> np.ndarray:
        """Distance to the k-th nearest normal example, in units of normal variation (1.0 = edge of normal)."""

        values = np.asarray(descriptors, dtype=np.float32)
        if values.size == 0 or len(self.exemplars) == 0:
            return np.full(len(values), np.inf)
        return _kth(_pairwise(values, self.exemplars), self.knn_k) / max(self.distance_norm, 1.0e-6)

    def _header(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "source": self.source,
            "status": self.status,
            "cell_width": round(float(self.cell_width), 4),
            "cell_height": round(float(self.cell_height), 4),
            "cell_area": round(float(self.cell_area), 4),
            "width_spread": round(float(self.width_spread), 6),
            "height_spread": round(float(self.height_spread), 6),
            "area_spread": round(float(self.area_spread), 6),
            "width_variable": bool(self.width_variable),
            "height_variable": bool(self.height_variable),
            "width_range": [round(float(value), 4) for value in self.width_range],
            "height_range": [round(float(value), 4) for value in self.height_range],
            "distance_norm": round(float(self.distance_norm), 6),
            "knn_k": int(self.knn_k),
            "frames_used": int(self.frames_used),
            "candidates": int(self.candidates),
            "dominance": round(float(self.dominance), 6),
            "reason": self.reason,
        }

    def to_payload(self) -> dict[str, Any]:
        payload = self._header()
        payload["bank_id"] = self.bank_id
        payload["exemplars"] = {
            "shape": list(self.exemplars.shape),
            "float16_b64": base64.b64encode(np.ascontiguousarray(self.exemplars, dtype=np.float16).tobytes()).decode("ascii"),
        }
        if self.template is not None:
            t = self.template
            payload["template"] = {
                "canvas": [int(t.canvas_width), int(t.canvas_height)],
                "mask_b64": base64.b64encode(t.mask).decode("ascii"),
                "band": float(t.band),
                "cell": [float(t.cell_width), float(t.cell_height), float(t.cell_area)],
                "deviation_norm": float(t.deviation_norm),
                "spread": [float(t.width_spread), float(t.height_spread)],
                "seed_count": int(t.seed_count),
            }
        return payload

    @staticmethod
    def from_payload(payload: dict[str, Any]) -> "NormalBank":
        if str(payload.get("version") or "") != NORMAL_BANK_VERSION:
            raise ValueError(f"unsupported normal bank version: {payload.get('version')}")
        raw = payload.get("exemplars") or {}
        shape = tuple(int(value) for value in raw.get("shape") or (0, DESCRIPTOR_GRID**2 + 4))
        data = base64.b64decode(str(raw.get("float16_b64") or ""))
        exemplars = np.frombuffer(data, dtype=np.float16).astype(np.float32).reshape(shape)
        template = None
        if isinstance(payload.get("template"), dict):
            t = payload["template"]
            template = CellShapeTemplate(
                canvas_width=int(t["canvas"][0]),
                canvas_height=int(t["canvas"][1]),
                mask=base64.b64decode(t["mask_b64"]),
                band=float(t["band"]),
                cell_width=float(t["cell"][0]),
                cell_height=float(t["cell"][1]),
                cell_area=float(t["cell"][2]),
                deviation_norm=float(t["deviation_norm"]),
                width_spread=float(t["spread"][0]),
                height_spread=float(t["spread"][1]),
                seed_count=int(t["seed_count"]),
            )
        return NormalBank(
            source=str(payload["source"]),
            status=str(payload["status"]),
            cell_width=float(payload["cell_width"]),
            cell_height=float(payload["cell_height"]),
            cell_area=float(payload["cell_area"]),
            width_spread=float(payload["width_spread"]),
            height_spread=float(payload["height_spread"]),
            area_spread=float(payload["area_spread"]),
            width_variable=bool(payload.get("width_variable", False)),
            height_variable=bool(payload.get("height_variable", False)),
            width_range=tuple(float(value) for value in payload.get("width_range") or (0.0, 0.0)),
            height_range=tuple(float(value) for value in payload.get("height_range") or (0.0, 0.0)),
            exemplars=exemplars,
            distance_norm=float(payload["distance_norm"]),
            knn_k=int(payload["knn_k"]),
            template=template,
            frames_used=int(payload.get("frames_used", 0)),
            candidates=int(payload.get("candidates", 0)),
            dominance=float(payload.get("dominance", 0.0)),
            reason=str(payload.get("reason") or ""),
        )

    def save(self, path: Path | str) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_payload(), ensure_ascii=False), encoding="utf-8")
        return target

    @staticmethod
    def load(path: Path | str) -> "NormalBank":
        return NormalBank.from_payload(json.loads(Path(path).read_text(encoding="utf-8")))


def _insufficient(source: str, reason: str, **extra: Any) -> NormalBank:
    return NormalBank(source=source, status=STATUS_INSUFFICIENT, reason=reason, **extra)


def build_normal_bank(
    component_sets: Sequence[ComponentSet],
    *,
    source: str = SOURCE_SESSION,
    config: NormalBankConfig | None = None,
    conductor_config: ConductorConfig | None = None,
) -> NormalBank:
    """Bank from several frames: dominant cell-sized cluster, trimmed, capped by a coreset."""

    cfg = config or NormalBankConfig()
    pool: list[tuple[CellComponent, np.ndarray]] = []
    frames_used = 0
    for components in component_sets:
        frame_area = typical_cell_area(components)
        zones = preliminary_zones(components, cell_area=frame_area or None, config=conductor_config)
        excluded = excluded_from_bank(components, zones)
        seeds = [
            item
            for item in components.components
            if not item.touches_border
            and item.area >= cfg.min_seed_area
            and item.label not in excluded
            and item.edge_softness < cfg.max_edge_softness
        ]
        if not seeds:
            continue
        frames_used += 1
        if len(seeds) > cfg.seeds_per_frame:
            step = len(seeds) / float(cfg.seeds_per_frame)
            seeds = [seeds[int(index * step)] for index in range(cfg.seeds_per_frame)]
        pool.extend((item, components.labels) for item in seeds)
    if len(pool) < cfg.min_exemplars:
        return _insufficient(source, "too_few_pieces", frames_used=frames_used, candidates=len(pool))

    areas = np.asarray([item.area for item, _labels in pool], dtype=np.float64)
    widths = np.asarray([item.width for item, _labels in pool], dtype=np.float64)
    heights = np.asarray([item.height for item, _labels in pool], dtype=np.float64)
    clean = np.asarray(
        [item.solidity >= cfg.min_clean_solidity and item.extent >= cfg.min_clean_extent for item, _labels in pool]
    )
    if np.count_nonzero(clean) < cfg.min_exemplars:
        return _insufficient(source, "no_clean_cells", frames_used=frames_used, candidates=len(pool))

    def mode(values: np.ndarray) -> float:
        bins = np.floor(np.log(values) / 0.10).astype(int)
        uniques, counts = np.unique(bins, return_counts=True)
        best = uniques[int(np.argmax(counts))]
        return float(np.median(values[np.abs(bins - best) <= 1]))

    mode_w, mode_h = mode(widths[clean]), mode(heights[clean])
    near_w = clean & (np.abs(widths / mode_w - 1.0) <= cfg.side_window)
    near_h = clean & (np.abs(heights / mode_h - 1.0) <= cfg.side_window)
    # A side varies by design when it spreads widely among the pieces that match the other side.
    width_variable = bool(near_h.any() and _spread(widths[near_h]) > cfg.variable_side_spread)
    height_variable = bool(near_w.any() and _spread(heights[near_w]) > cfg.variable_side_spread)
    if width_variable and height_variable:
        return _insufficient(source, "no_fixed_side", frames_used=frames_used, candidates=len(pool))
    if height_variable:
        top = float(np.quantile(heights[near_w], cfg.side_quantiles[1]))
        cluster = near_w & (heights >= cfg.variable_side_min_fraction * top)
    elif width_variable:
        top = float(np.quantile(widths[near_h], cfg.side_quantiles[1]))
        cluster = near_h & (widths >= cfg.variable_side_min_fraction * top)
    else:
        cluster = near_w & near_h
    if np.count_nonzero(cluster) < cfg.min_exemplars:
        return _insufficient(source, "too_few_normal_cells", frames_used=frames_used, candidates=len(pool))
    median_area = float(np.median(areas[cluster]))
    if median_area < cfg.min_cell_area:
        return _insufficient(source, "crumb_sized_cluster", frames_used=frames_used, candidates=len(pool))
    cluster_areas = areas[cluster]
    cell_sized = (areas >= 0.5 * float(cluster_areas.min())) & (areas <= 2.0 * float(cluster_areas.max()))
    dominance = float(np.count_nonzero(cluster)) / float(max(1, np.count_nonzero(cell_sized)))
    if dominance < cfg.min_dominance:
        return _insufficient(
            source, "no_dominant_cell", frames_used=frames_used, candidates=len(pool), dominance=dominance
        )
    members = [pool[index] for index in np.flatnonzero(cluster)]
    if len(members) > cfg.max_pool:
        step = len(members) / float(cfg.max_pool)
        members = [members[int(index * step)] for index in range(cfg.max_pool)]
    descriptors = np.stack([shape_descriptor(item, labels) for item, labels in members])
    keep = np.ones(len(members), dtype=bool)
    for _ in range(max(0, int(cfg.trim_rounds))):
        kept = np.flatnonzero(keep)
        if len(kept) <= cfg.min_exemplars:
            break
        distances = _pairwise(descriptors[kept], descriptors[kept])
        np.fill_diagonal(distances, np.inf)
        score = _kth(distances, cfg.knn_k)
        cut = np.quantile(score, 1.0 - cfg.trim_share)
        keep[kept[score > cut]] = False
    kept = np.flatnonzero(keep)
    if len(kept) < cfg.min_exemplars:
        return _insufficient(source, "too_few_normal_cells", frames_used=frames_used, candidates=len(pool), dominance=dominance)
    pool_descriptors = descriptors[kept]
    chosen = _greedy_coreset(pool_descriptors, cfg.max_exemplars)
    exemplars = pool_descriptors[chosen]

    distances = _pairwise(pool_descriptors, exemplars)
    distances[chosen, np.arange(len(chosen))] = np.inf  # a bank member is not its own neighbour
    k = max(1, min(cfg.knn_k, len(chosen) - 1))
    norm = float(np.quantile(_kth(distances, k), cfg.norm_quantile))

    kept_items = [members[index][0] for index in kept]
    kept_w = np.asarray([item.width for item in kept_items], dtype=np.float64)
    kept_h = np.asarray([item.height for item in kept_items], dtype=np.float64)
    kept_a = np.asarray([item.area for item in kept_items], dtype=np.float64)
    low_q, high_q = cfg.side_quantiles
    template = build_cell_shape_template(
        [(item.contour, float(item.width), float(item.height), float(item.area)) for item in kept_items]
    )
    return NormalBank(
        source=source,
        status=STATUS_OK,
        cell_width=float(np.median(kept_w)),
        cell_height=float(np.median(kept_h)),
        cell_area=float(np.median(kept_a)),
        width_spread=_spread(kept_w),
        height_spread=_spread(kept_h),
        area_spread=_spread(kept_a),
        width_variable=width_variable,
        height_variable=height_variable,
        width_range=(float(np.quantile(kept_w, low_q)), float(np.quantile(kept_w, high_q))),
        height_range=(float(np.quantile(kept_h, low_q)), float(np.quantile(kept_h, high_q))),
        exemplars=exemplars.astype(np.float32),
        distance_norm=max(norm, float(cfg.min_distance_norm)),
        knn_k=k,
        template=template,
        frames_used=frames_used,
        candidates=len(pool),
        dominance=dominance,
    )
