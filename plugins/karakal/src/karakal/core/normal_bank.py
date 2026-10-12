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

Operator masks are truth: every cell drawn there is normal, so nothing is trimmed from them,
a shape is judged by the nearest drawn cell, and the drawn sizes are inside the size limits.
Only cells cut by the frame edge are left out: they are not the whole cell shape.

Cells the operator adds by hand (ManualCell) join the operator cells on top of any bank: their
shape is normal, and their size too when the operator says so. They are kept with the layer,
outside the bank, so a rebuilt bank gets them again.
"""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass, field, replace
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
    # Operator cells kept as examples; beyond this a k-center subset that still covers every one.
    max_operator_exemplars: int = 8000


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
    # _pairwise to one point, with the float64 copy and the squared norms made once (the same
    # arithmetic, so the same subset): thousands of operator cells take seconds, not minutes.
    q = points.astype(np.float64)
    squared = (q * q).sum(axis=1)

    def to_point(index: int) -> np.ndarray:
        b = q[index : index + 1]
        return np.sqrt(np.maximum(squared[:, None] + (b * b).sum(axis=1)[None, :] - 2.0 * q @ b.T, 0.0))[:, 0]

    chosen = [first]
    nearest = to_point(first)
    for _ in range(size - 1):
        index = int(np.argmax(nearest))
        chosen.append(index)
        nearest = np.minimum(nearest, to_point(index))
    return np.asarray(sorted(chosen))


def _spread(values: np.ndarray) -> float:
    median = float(np.median(values))
    return float(1.4826 * np.median(np.abs(values - median)) / max(1.0, median))


def _empty_exemplars() -> np.ndarray:
    return np.zeros((0, DESCRIPTOR_GRID**2 + 4), dtype=np.float32)


@dataclass(frozen=True, slots=True)
class ExemplarView:
    """What the bank window shows of one example: its pixels, size and frame. Not part of the bank id."""

    frame: str
    width: int
    height: int
    area: int
    mask_bits: bytes

    def mask(self) -> np.ndarray:
        bits = np.unpackbits(np.frombuffer(self.mask_bits, dtype=np.uint8), count=self.width * self.height)
        return bits.reshape(self.height, self.width).astype(bool)


def exemplar_view(item: CellComponent, labels: np.ndarray, frame: str) -> ExemplarView:
    piece, _x0, _y0 = item.crop(labels)
    return ExemplarView(
        frame=str(frame),
        width=int(piece.shape[1]),
        height=int(piece.shape[0]),
        area=int(item.area),
        mask_bits=np.packbits(piece.reshape(-1)).tobytes(),
    )


def _frame_names(component_sets: Sequence[ComponentSet], frame_names: Sequence[str] | None) -> dict[int, str]:
    names = list(frame_names or ())
    return {id(components.labels): (names[index] if index < len(names) else "") for index, components in enumerate(component_sets)}


def _pack_exemplars(values: np.ndarray) -> dict[str, Any]:
    return {
        "shape": list(values.shape),
        "float16_b64": base64.b64encode(np.ascontiguousarray(values, dtype=np.float16).tobytes()).decode("ascii"),
    }


def _unpack_exemplars(raw: Any) -> np.ndarray:
    if not isinstance(raw, dict):
        return _empty_exemplars()
    shape = tuple(int(value) for value in raw.get("shape") or (0, DESCRIPTOR_GRID**2 + 4))
    data = base64.b64decode(str(raw.get("float16_b64") or ""))
    return np.frombuffer(data, dtype=np.float16).astype(np.float32).reshape(shape)


def _nearest(queries: np.ndarray, bank: np.ndarray, *, skip_self: bool = False, chunk: int = 512) -> np.ndarray:
    """Distance from each query to its nearest bank row, in chunks (thousands of operator cells)."""

    out = np.full(len(queries), np.inf)
    if len(bank) == 0:
        return out
    for start in range(0, len(queries), chunk):
        distances = _pairwise(queries[start : start + chunk], bank)
        if skip_self:
            rows = np.arange(distances.shape[0])
            distances[rows, start + rows] = np.inf
        out[start : start + chunk] = distances.min(axis=1)
    return out


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
    exemplars: np.ndarray = field(default_factory=_empty_exemplars)
    distance_norm: float = 1.0
    knn_k: int = 4
    template: CellShapeTemplate | None = None
    frames_used: int = 0
    candidates: int = 0
    dominance: float = 0.0
    reason: str = ""
    version: str = NORMAL_BANK_VERSION
    # Cells the operator drew. Each one is normal on its own, so a shape is judged by the
    # nearest of them (not the k-th: no network mistake hides among them), and the sizes
    # they span stay inside the size limits.
    operator_exemplars: np.ndarray = field(default_factory=_empty_exemplars)
    operator_width_range: tuple[float, float] = (0.0, 0.0)
    operator_height_range: tuple[float, float] = (0.0, 0.0)
    # For the bank window only (not in the id or the payload): the pixels of each example, in
    # the order of its descriptors, and width/height of every normal cell the bank was built from.
    exemplar_views: tuple[ExemplarView, ...] = ()
    operator_views: tuple[ExemplarView, ...] = ()
    normal_sizes: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float32))
    # Ids of the cells added by hand: the last of the operator examples, in this order.
    manual_ids: tuple[str, ...] = ()

    @property
    def ready(self) -> bool:
        return self.status == STATUS_OK

    @property
    def exemplar_count(self) -> int:
        return int(len(self.exemplars) + len(self.operator_exemplars))

    @property
    def bank_id(self) -> str:
        digest = hashlib.sha1()
        digest.update(json.dumps(self._header(), sort_keys=True).encode("utf-8"))
        digest.update(np.ascontiguousarray(self.exemplars, dtype=np.float16).tobytes())
        if len(self.operator_exemplars):
            digest.update(np.ascontiguousarray(self.operator_exemplars, dtype=np.float16).tobytes())
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

        def with_operator(pair: tuple[float, float], drawn: tuple[float, float]) -> tuple[float, float]:
            # A size the operator drew is normal at any slider position.
            if drawn[1] <= 0.0:
                return pair
            return min(pair[0], float(drawn[0])), max(pair[1], float(drawn[1]))

        return (
            with_operator(
                limits(self.cell_width, self.width_spread, self.width_variable, self.width_range),
                self.operator_width_range,
            ),
            with_operator(
                limits(self.cell_height, self.height_spread, self.height_variable, self.height_range),
                self.operator_height_range,
            ),
        )

    def normal_distance(self, descriptors: np.ndarray) -> np.ndarray:
        """Distance to the normal examples, in units of normal variation (1.0 = edge of normal):
        the k-th nearest example of the run, or the nearest operator cell, whichever is closer."""

        values = np.asarray(descriptors, dtype=np.float32)
        best = np.full(len(values), np.inf)
        if values.size == 0:
            return best
        if len(self.exemplars):
            best = _kth(_pairwise(values, self.exemplars), self.knn_k)
        if len(self.operator_exemplars):
            best = np.minimum(best, _nearest(values, self.operator_exemplars))
        return best / max(self.distance_norm, 1.0e-6)

    def _header(self) -> dict[str, Any]:
        header = {
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
        # Only a bank with operator cells carries these keys: a run bank keeps its old id.
        if len(self.operator_exemplars) or self.operator_width_range[1] > 0.0 or self.operator_height_range[1] > 0.0:
            header["operator_width_range"] = [round(float(value), 4) for value in self.operator_width_range]
            header["operator_height_range"] = [round(float(value), 4) for value in self.operator_height_range]
        return header

    def to_payload(self) -> dict[str, Any]:
        payload = self._header()
        payload["bank_id"] = self.bank_id
        payload["exemplars"] = _pack_exemplars(self.exemplars)
        if len(self.operator_exemplars):
            payload["operator_exemplars"] = _pack_exemplars(self.operator_exemplars)
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
        exemplars = _unpack_exemplars(payload.get("exemplars"))
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
            operator_exemplars=_unpack_exemplars(payload.get("operator_exemplars")),
            operator_width_range=tuple(float(value) for value in payload.get("operator_width_range") or (0.0, 0.0)),
            operator_height_range=tuple(float(value) for value in payload.get("operator_height_range") or (0.0, 0.0)),
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
    frame_names: Sequence[str] | None = None,
) -> NormalBank:
    """Bank from several frames: dominant cell-sized cluster, trimmed, capped by a coreset."""

    cfg = config or NormalBankConfig()
    names = _frame_names(component_sets, frame_names)
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
    views = tuple(
        exemplar_view(members[kept[index]][0], members[kept[index]][1], names.get(id(members[kept[index]][1]), ""))
        for index in chosen
    )
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
        exemplar_views=views,
        normal_sizes=np.stack([kept_w, kept_h], axis=1).astype(np.float32),
    )


def build_markup_bank(
    component_sets: Sequence[ComponentSet],
    *,
    source: str = SOURCE_REFERENCE,
    config: NormalBankConfig | None = None,
    frame_names: Sequence[str] | None = None,
) -> NormalBank:
    """Bank from operator masks: every drawn cell is a normal example, none is trimmed.

    Cells touching the frame edge are left out (a cut cell would make a half cell normal), and
    so are crumbs below ``min_seed_area`` (JPEG specks, not drawn cells). The unit of distance is
    how far a drawn cell usually is from its nearest other drawn cell, raised so that every drawn
    cell stays within 1.0 of the kept examples.
    """

    cfg = config or NormalBankConfig()
    names = _frame_names(component_sets, frame_names)
    items: list[tuple[CellComponent, np.ndarray]] = []
    frames_used = 0
    for components in component_sets:
        drawn = [
            (item, components.labels)
            for item in components.components
            if not item.touches_border and item.area >= cfg.min_seed_area
        ]
        if drawn:
            frames_used += 1
            items.extend(drawn)
    if len(items) < cfg.min_exemplars:
        return _insufficient(source, "too_few_markup_cells", frames_used=frames_used, candidates=len(items))

    widths = np.asarray([item.width for item, _labels in items], dtype=np.float64)
    heights = np.asarray([item.height for item, _labels in items], dtype=np.float64)
    areas = np.asarray([item.area for item, _labels in items], dtype=np.float64)
    width_spread, height_spread = _spread(widths), _spread(heights)
    width_variable = width_spread > cfg.variable_side_spread
    height_variable = height_spread > cfg.variable_side_spread
    if width_variable and height_variable:
        # One side of a cell is fixed (decision В4); the wider spread is the side drawn at several lengths.
        width_variable, height_variable = width_spread > height_spread, height_spread >= width_spread

    descriptors = np.stack([shape_descriptor(item, labels) for item, labels in items])
    chosen = _greedy_coreset(descriptors, max(1, int(cfg.max_operator_exemplars)))
    examples = descriptors[chosen]
    usual = float(np.quantile(_nearest(descriptors, descriptors, skip_self=True), cfg.norm_quantile))
    rest = np.setdiff1d(np.arange(len(descriptors)), chosen)
    cover = float(_nearest(descriptors[rest], examples).max()) if len(rest) else 0.0
    norm = max(usual, cover, float(cfg.min_distance_norm))

    low_q, high_q = cfg.side_quantiles
    template = build_cell_shape_template(
        [(item.contour, float(item.width), float(item.height), float(item.area)) for item, _labels in items]
    )
    return NormalBank(
        source=source,
        status=STATUS_OK,
        cell_width=float(np.median(widths)),
        cell_height=float(np.median(heights)),
        cell_area=float(np.median(areas)),
        width_spread=width_spread,
        height_spread=height_spread,
        area_spread=_spread(areas),
        width_variable=bool(width_variable),
        height_variable=bool(height_variable),
        width_range=(float(np.quantile(widths, low_q)), float(np.quantile(widths, high_q))),
        height_range=(float(np.quantile(heights, low_q)), float(np.quantile(heights, high_q))),
        distance_norm=norm,
        knn_k=int(cfg.knn_k),
        template=template,
        frames_used=frames_used,
        candidates=len(items),
        dominance=1.0,
        operator_exemplars=examples.astype(np.float32),
        operator_width_range=(float(widths.min()), float(widths.max())),
        operator_height_range=(float(heights.min()), float(heights.max())),
        operator_views=tuple(exemplar_view(items[index][0], items[index][1], names.get(id(items[index][1]), "")) for index in chosen),
        normal_sizes=np.stack([widths, heights], axis=1).astype(np.float32),
    )


ORIGIN_OPERATOR = "operator"
ORIGIN_RUN = "run"


def exemplar_unusualness(bank: NormalBank) -> dict[str, np.ndarray]:
    """How far each example is from the rest, in units of normal variation, the way the bank
    judges: a run example by its k-th nearest other run example, an operator cell by its
    nearest other operator cell or the run examples, whichever is closer (a cell added by hand
    to a run bank shows how far it was from that bank). The bank window lists the odd ones first."""

    norm = max(float(bank.distance_norm), 1.0e-6)
    out = {ORIGIN_RUN: np.zeros(len(bank.exemplars)), ORIGIN_OPERATOR: np.zeros(len(bank.operator_exemplars))}
    if len(bank.exemplars) > 1:
        distances = _pairwise(bank.exemplars, bank.exemplars)
        np.fill_diagonal(distances, np.inf)
        out[ORIGIN_RUN] = _kth(distances, min(bank.knn_k, len(bank.exemplars) - 1)) / norm
    if len(bank.operator_exemplars):
        values = _nearest(bank.operator_exemplars, bank.operator_exemplars, skip_self=True)
        if len(bank.exemplars):
            values = np.minimum(values, _kth(_pairwise(bank.operator_exemplars, bank.exemplars), bank.knn_k))
        out[ORIGIN_OPERATOR] = np.where(np.isfinite(values), values, 0.0) / norm
    return out


def nearest_exemplars(bank: NormalBank, descriptor: np.ndarray, count: int = 8) -> list[tuple[str, int, float]]:
    """The examples closest to one cell: (origin, index in that origin's examples, distance in units)."""

    norm = max(float(bank.distance_norm), 1.0e-6)
    query = np.asarray(descriptor, dtype=np.float32).reshape(1, -1)
    found: list[tuple[str, int, float]] = []
    for origin, pool in ((ORIGIN_RUN, bank.exemplars), (ORIGIN_OPERATOR, bank.operator_exemplars)):
        if len(pool):
            distances = _pairwise(query, pool)[0] / norm
            for index in np.argsort(distances)[:count]:
                found.append((origin, int(index), float(distances[index])))
    found.sort(key=lambda entry: entry[2])
    return found[:count]


def cell_at(components: ComponentSet, bbox: tuple[int, int, int, int]) -> CellComponent | None:
    """The mask piece a box on the frame points at: the one under its centre, else the one
    with most pixels inside it."""

    x, y, w, h = (int(value) for value in bbox)
    labels = components.labels
    rows, cols = labels.shape
    cx, cy = min(cols - 1, max(0, x + w // 2)), min(rows - 1, max(0, y + h // 2))
    label = int(labels[cy, cx])
    if label <= 0:
        window = labels[max(0, y) : max(0, y + h), max(0, x) : max(0, x + w)]
        values, counts = np.unique(window[window > 0], return_counts=True)
        if not len(values):
            return None
        label = int(values[int(np.argmax(counts))])
    return next((item for item in components.components if item.label == label), None)


@dataclass(frozen=True, slots=True)
class ManualCell:
    """A cell the operator called normal by hand, kept with its layer."""

    cell_id: str
    view: ExemplarView
    descriptor: np.ndarray
    # Its size is normal too: the size limits widen to take it in.
    size_normal: bool = False
    # What the bank said about it when it was added ("" when it was not marked): a count of
    # the marks the operator overruled.
    flagged: str = ""
    added: str = ""

    def to_payload(self) -> dict[str, Any]:
        return {
            "id": self.cell_id,
            "frame": self.view.frame,
            "width": int(self.view.width),
            "height": int(self.view.height),
            "area": int(self.view.area),
            "mask_b64": base64.b64encode(self.view.mask_bits).decode("ascii"),
            "descriptor_b64": base64.b64encode(np.asarray(self.descriptor, dtype=np.float16).tobytes()).decode("ascii"),
            "size_normal": bool(self.size_normal),
            "flagged": str(self.flagged or ""),
            "added": str(self.added or ""),
        }

    @staticmethod
    def from_payload(payload: Any) -> "ManualCell | None":
        try:
            view = ExemplarView(
                frame=str(payload.get("frame") or ""),
                width=int(payload["width"]),
                height=int(payload["height"]),
                area=int(payload["area"]),
                mask_bits=base64.b64decode(str(payload["mask_b64"])),
            )
            descriptor = np.frombuffer(base64.b64decode(str(payload["descriptor_b64"])), dtype=np.float16).astype(np.float32)
            if descriptor.shape != (DESCRIPTOR_GRID**2 + 4,) or view.width <= 0 or view.height <= 0:
                return None
            return ManualCell(
                cell_id=str(payload["id"]),
                view=view,
                descriptor=descriptor,
                size_normal=bool(payload.get("size_normal", False)),
                flagged=str(payload.get("flagged") or ""),
                added=str(payload.get("added") or ""),
            )
        except (KeyError, TypeError, ValueError, AttributeError):
            return None


def apply_manual_cells(bank: NormalBank, cells: Sequence[ManualCell]) -> NormalBank:
    """The bank with hand-added cells among the operator examples (judged by the nearest one).

    A bank without a reliable normal cell stays as it is: a few hand-picked cells cannot tell
    the cell size of a layer.
    """

    if not cells or not bank.ready:
        return bank
    descriptors = np.stack([np.asarray(cell.descriptor, dtype=np.float32) for cell in cells])
    width_range, height_range = bank.operator_width_range, bank.operator_height_range
    for cell in cells:
        if not cell.size_normal:
            continue

        def widen(pair: tuple[float, float], value: float) -> tuple[float, float]:
            return (value, value) if pair[1] <= 0.0 else (min(pair[0], value), max(pair[1], value))

        width_range = widen(width_range, float(cell.view.width))
        height_range = widen(height_range, float(cell.view.height))
    return replace(
        bank,
        operator_exemplars=np.concatenate([bank.operator_exemplars, descriptors]).astype(np.float32),
        operator_views=tuple(bank.operator_views) + tuple(cell.view for cell in cells),
        operator_width_range=width_range,
        operator_height_range=height_range,
        manual_ids=tuple(cell.cell_id for cell in cells),
    )
