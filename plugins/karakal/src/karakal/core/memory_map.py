"""Memory map: what is written in every cell of the memory, read from ones/zeros masks. Qt-free.

Each memory cell sits on a regular lattice. The ones network paints cells holding 1, the zeros
network cells holding 0, so a lattice site has one of these states:

- ONE / ZERO: painted by one network;
- CONFLICT: painted by both;
- EMPTY: painted by none. With both networks it is a missing cell; with one network it is only
  "not painted by it" (a 0 for the ones network, or a missing cell: the two cannot be told apart);
- DEFECT: a merged, broken or fragmentary cell (or a defect the cell analysis found there);
- PARTIAL: cut by the frame edge, unknown in this frame (another frame sees it whole);
- OUTSIDE: not a memory site (gaps between blocks, conductors, beyond the memory).

The work goes in three steps.

1. Sites of a frame (read_frame_sites). The lattice pitch is one for the whole memory; its phase
   is not: blocks may be shifted by a fraction of a pitch. Clean cells are grouped by phase, each
   group gives lattice sites wherever its cells are (holes up to a few sites are closed), and the
   masks are read at every site centre.
2. Seams (match_frames). Frames come row by row and overlap. For two neighbours every lattice
   translation in a plausible window is tried, and the one where the overlapping sites carry the
   same states wins. Random data make the true one unique; a seam whose best translation is not
   clearly better than the next, or has too few sites to compare, is not trusted.
3. Field (assemble_memory_map). Frames are placed through trusted seams first; frames without one
   (empty, periphery) take the typical offset of the trusted seams. Every trusted seam outside that
   placement is checked against it: a seam that disagrees is reported. Sites seen by several frames
   take the state of the frame that sees them farthest from its edge; disagreements are counted.
   Lines of sites with nothing painted at all are gaps between blocks; the rest splits into blocks.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - OpenCV is a hard dependency of the app
    cv2 = None

from .mask_normalization import binarize_mask

OUTSIDE = 0
EMPTY = 1
ONE = 2
ZERO = 3
CONFLICT = 4
DEFECT = 5
PARTIAL = 6
STATE_NAMES = {
    OUTSIDE: "outside",
    EMPTY: "empty",
    ONE: "one",
    ZERO: "zero",
    CONFLICT: "conflict",
    DEFECT: "defect",
    PARTIAL: "partial",
}
PAINTED = (ONE, ZERO, CONFLICT, DEFECT)

MODE_PAIR = "pair"
MODE_ONES = "ones"
MODE_ZEROS = "zeros"

SEAM_BITS = "bits"
SEAM_AMBIGUOUS = "ambiguous"
SEAM_FAILED = "failed"
SEAM_EMPTY = "empty"
SEAM_INCONSISTENT = "inconsistent"

PLACED_BITS = "bits"
PLACED_ESTIMATED = "estimated"


@dataclass(frozen=True, slots=True)
class Pitch:
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class MemoryMapConfig:
    # Clean cell: area within this share of the typical cell area.
    clean_area: tuple[float, float] = (0.6, 1.5)
    # A painted piece smaller than this share of a cell, on a site, is a fragment (DEFECT).
    fragment_area: float = 0.5
    # Cells of one phase group: phase within this share of the pitch on each axis.
    phase_tolerance: float = 0.2
    # A phase group needs at least this many clean cells (and this share of them).
    min_group_cells: int = 6
    min_group_share: float = 0.03
    # More phases than this in one frame is noise, not blocks.
    max_groups: int = 8
    # A cell is centred on a site when its centroid is within this share of the pitch.
    centre_tolerance: float = 0.25
    # Holes in a block up to this many sites are closed into it.
    close_sites: int = 5
    # A block of a frame needs this many clean cells on its sites, and this share of its box:
    # stray cell-sized specks in the periphery do not make memory.
    min_block_cells: int = 24
    min_edge_block_cells: int = 6
    min_block_density: float = 0.2
    # Overlapping phase groups: a place goes to the group with more cells within this many sites.
    owner_sites: int = 2
    # An edge line of a block with fewer cells than this share of the block's density is cut.
    edge_trim: float = 0.3
    edge_trim_sites: int = 16
    # A painted piece larger than this many cells is a conductor or periphery, not memory: its
    # sites are outside the memory (merged cells up to this size stay defects).
    conductor_cells: float = 6.0
    # An empty site this close to the frame edge (share of the pitch) may hold a cut cell.
    edge_margin: float = 0.4
    # Seams: sites to compare, agreement to trust, lead over the next translation. 24 random bits
    # agree by chance about once in a million; a memory strip of a few rows still gets compared.
    min_compared: int = 24
    min_agreement: float = 0.98
    min_lead: float = 0.05
    # Overlap window, in sites: up to this many columns (rows) and this much shift across.
    max_overlap_sites: int = 10
    max_shift_sites: int = 4
    # A trusted seam off the placement by more than this share of the pitch is inconsistent.
    seam_tolerance: float = 0.3
    # A line of at least this many sites with this little painted is a gap between blocks.
    gap_min_sites: int = 8
    gap_max_painted: float = 0.02
    # Blocks of the field smaller than this are dropped (strays of the periphery at frame edges).
    min_block_sites: int = 256
    # A block with almost nothing painted is not memory (a regular periphery structure).
    min_block_painted: float = 0.05
    # A block smaller than this share of the largest one is a service block.
    service_share: float = 0.02


# --- Pitch --------------------------------------------------------------------------------------


def _components(binary: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    return labels, stats, centroids


def _typical_area(stats_list: Iterable[np.ndarray]) -> float:
    areas = np.concatenate([stats[1:, cv2.CC_STAT_AREA] for stats in stats_list] or [np.zeros(0)])
    areas = areas[areas >= 40]
    if areas.size == 0:
        return 0.0
    # The most common piece size: cells outnumber crumbs and merges.
    logs = np.log(areas.astype(np.float64))
    bins = np.floor(logs / 0.1).astype(int)
    values, counts = np.unique(bins, return_counts=True)
    best = values[int(np.argmax(counts))]
    return float(np.median(areas[np.abs(bins - best) <= 1]))


def _clean_centroids(stats: np.ndarray, centroids: np.ndarray, area: float, cfg: MemoryMapConfig, shape) -> np.ndarray:
    if area <= 0.0 or len(stats) <= 1:
        return np.zeros((0, 2))
    s = stats[1:]
    a = s[:, cv2.CC_STAT_AREA]
    rows, cols = shape
    touches = (s[:, 0] <= 0) | (s[:, 1] <= 0) | (s[:, 0] + s[:, 2] >= cols) | (s[:, 1] + s[:, 3] >= rows)
    keep = (a >= cfg.clean_area[0] * area) & (a <= cfg.clean_area[1] * area) & ~touches
    return centroids[1:][keep]


def estimate_pitch(masks: Iterable[np.ndarray], config: MemoryMapConfig | None = None) -> Pitch | None:
    """Lattice pitch from a few masks: the usual distance to the next cell in a row and in a column,
    refined by a line fit of cell positions against their site numbers."""

    cfg = config or MemoryMapConfig()
    pts_all: list[np.ndarray] = []
    dx_all: list[np.ndarray] = []
    dy_all: list[np.ndarray] = []
    for mask in masks:
        binary = binarize_mask(mask)
        _labels, stats, centroids = _components(binary)
        area = _typical_area([stats])
        pts = _clean_centroids(stats, centroids, area, cfg, binary.shape)
        if len(pts) < 8:
            continue
        pts_all.append(pts)
        from scipy.spatial import cKDTree

        tree = cKDTree(pts)
        _d, idx = tree.query(pts, k=min(9, len(pts)))
        offsets = pts[idx[:, 1:]] - pts[:, None, :]
        ox, oy = np.abs(offsets[..., 0]).ravel(), np.abs(offsets[..., 1]).ravel()
        dx_all.append(ox[(oy < 0.25 * np.maximum(ox, 1)) & (ox > 2)])
        dy_all.append(oy[(ox < 0.25 * np.maximum(oy, 1)) & (oy > 2)])
    if not dx_all:
        return None
    dx, dy = np.concatenate(dx_all), np.concatenate(dy_all)
    if dx.size < 8 or dy.size < 8:
        return None

    def smallest_mode(values: np.ndarray) -> float:
        low = np.percentile(values, 5)
        near = values[values <= 1.35 * low] if np.any(values <= 1.35 * low) else values
        return float(np.median(near))

    px, py = smallest_mode(dx), smallest_mode(dy)
    # Refine: positions against site numbers along each axis (slope = pitch).
    refined = []
    for pitch, axis in ((px, 0), (py, 1)):
        slopes = []
        for pts in pts_all:
            v = pts[:, axis]
            phase = _circular_mean(v, pitch)
            index = np.round((v - phase) / pitch)
            if np.ptp(index) >= 3:
                slope, _intercept = np.polyfit(index, v, 1)
                if abs(slope / pitch - 1.0) < 0.05:
                    slopes.append(slope)
        refined.append(float(np.median(slopes)) if slopes else pitch)
    return Pitch(refined[0], refined[1])


def _circular_mean(values: np.ndarray, period: float) -> float:
    angles = 2.0 * np.pi * np.asarray(values, dtype=np.float64) / period
    return float((np.angle(np.mean(np.exp(1j * angles))) % (2.0 * np.pi)) * period / (2.0 * np.pi))


def _circular_diff(a: np.ndarray | float, b: float, period: float) -> np.ndarray:
    d = (np.asarray(a, dtype=np.float64) - b) % period
    return np.minimum(d, period - d)


def _phase_groups(points: np.ndarray, pitch: Pitch, cfg: MemoryMapConfig) -> list[tuple[float, float]]:
    """Lattice phases present among the cells: one per block shift, the biggest first."""

    if len(points) == 0:
        return []
    remaining = np.ones(len(points), dtype=bool)
    groups: list[tuple[float, float]] = []
    tol_x, tol_y = cfg.phase_tolerance * pitch.x, cfg.phase_tolerance * pitch.y
    minimum = max(cfg.min_group_cells, int(cfg.min_group_share * len(points)))
    px_all, py_all = points[:, 0] % pitch.x, points[:, 1] % pitch.y
    while np.count_nonzero(remaining) >= minimum:
        rest = np.flatnonzero(remaining)
        # Densest phase among the rest: the cell with the most others within the tolerance.
        sample = rest if len(rest) <= 400 else rest[np.linspace(0, len(rest) - 1, 400).astype(int)]
        best, best_count = None, 0
        for index in sample:
            near = (_circular_diff(px_all[rest], px_all[index], pitch.x) <= tol_x) & (
                _circular_diff(py_all[rest], py_all[index], pitch.y) <= tol_y
            )
            count = int(np.count_nonzero(near))
            if count > best_count:
                best, best_count = index, count
        if best is None or best_count < minimum:
            break
        phase_x, phase_y = px_all[best], py_all[best]
        seed = rest[
            (_circular_diff(px_all[rest], phase_x, pitch.x) <= tol_x) & (_circular_diff(py_all[rest], phase_y, pitch.y) <= tol_y)
        ]
        members = seed
        for _ in range(2):
            phase_x, phase_y = _circular_mean(points[members, 0], pitch.x), _circular_mean(points[members, 1], pitch.y)
            refined = rest[
                (_circular_diff(px_all[rest], phase_x, pitch.x) <= tol_x)
                & (_circular_diff(py_all[rest], phase_y, pitch.y) <= tol_y)
            ]
            if refined.size:
                members = refined
        # Every round takes at least the cells found first, so the loop always ends.
        remaining[seed] = False
        remaining[members] = False
        groups.append((phase_x, phase_y))
        if len(groups) >= cfg.max_groups:
            break
    return groups


# --- Sites of a frame ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FrameSites:
    """Lattice sites of one frame: centres in frame pixels, their state and phase group."""

    frame_id: str
    shape: tuple[int, int]
    x: np.ndarray
    y: np.ndarray
    state: np.ndarray
    group: np.ndarray
    phases: tuple[tuple[float, float], ...] = ()

    @property
    def count(self) -> int:
        return int(len(self.x))

    def edge_distance(self) -> np.ndarray:
        rows, cols = self.shape
        return np.minimum.reduce([self.x, self.y, cols - 1 - self.x, rows - 1 - self.y])


def _empty_sites(frame_id: str, shape: tuple[int, int]) -> FrameSites:
    zeros = np.zeros(0)
    return FrameSites(frame_id, shape, zeros, zeros, np.zeros(0, np.uint8), np.zeros(0, np.int16))


def read_frame_sites(
    ones: np.ndarray | None,
    zeros: np.ndarray | None,
    pitch: Pitch,
    *,
    frame_id: str = "",
    defect_boxes: Sequence[tuple[int, int, int, int]] = (),
    config: MemoryMapConfig | None = None,
) -> FrameSites:
    """Sites of one frame from one or two masks (ones, zeros; either may be missing)."""

    cfg = config or MemoryMapConfig()
    layers = []
    shape = None
    for mask in (ones, zeros):
        if mask is None:
            layers.append(None)
            continue
        binary = binarize_mask(mask)
        shape = binary.shape
        layers.append((binary, *_components(binary)))
    if shape is None:
        raise ValueError("a frame needs at least one mask")
    rows, cols = shape
    area = _typical_area([layer[2] for layer in layers if layer is not None])
    clean = [
        _clean_centroids(layer[2], layer[3], area, cfg, shape) for layer in layers if layer is not None
    ]
    points = np.concatenate(clean) if clean else np.zeros((0, 2))
    phases = _phase_groups(points, pitch, cfg)
    if not phases:
        return _empty_sites(frame_id, shape)

    xs, ys, gs = [], [], []
    supports: list[tuple[float, float, np.ndarray]] = []
    totals: list[int] = []
    tol = cfg.centre_tolerance
    for gi, (phase_x, phase_y) in enumerate(phases):
        n_cols = int(np.floor((cols - 1 - phase_x) / pitch.x)) + 1
        n_rows = int(np.floor((rows - 1 - phase_y) / pitch.y)) + 1
        occupied = np.zeros((n_rows, n_cols), np.uint8)
        for pts in clean:
            if not len(pts):
                continue
            fc, fr = (pts[:, 0] - phase_x) / pitch.x, (pts[:, 1] - phase_y) / pitch.y
            c, r = np.round(fc).astype(int), np.round(fr).astype(int)
            ok = (np.abs(fc - c) <= tol) & (np.abs(fr - r) <= tol) & (c >= 0) & (c < n_cols) & (r >= 0) & (r < n_rows)
            occupied[r[ok], c[ok]] = 1
        if not occupied.any():
            continue
        pad = cfg.close_sites
        kernel = np.ones((cfg.close_sites, cfg.close_sites), np.uint8)
        padded = np.pad(occupied, pad)
        closed = cv2.morphologyEx(padded, cv2.MORPH_CLOSE, kernel)
        # Holes enclosed by the block are sites too (a long run of one value with one network).
        outside = closed.copy()
        flood = np.zeros((outside.shape[0] + 2, outside.shape[1] + 2), np.uint8)
        cv2.floodFill(outside, flood, (0, 0), 2)
        closed = ((closed == 1) | (outside == 0)).astype(np.uint8)[pad:-pad, pad:-pad]
        # Memory blocks are rectangular: a block fills its box. With one network the zeros on a
        # block's edge have no cell to show them; the box brings them back. A part with few clean
        # cells, or cells spread thin over its box, is periphery and is dropped.
        n_parts, part_labels, part_stats, _c = cv2.connectedComponentsWithStats(closed, connectivity=8)
        blocks = np.zeros_like(closed)
        for part in range(1, n_parts):
            x0, y0 = part_stats[part, cv2.CC_STAT_LEFT], part_stats[part, cv2.CC_STAT_TOP]
            w, h = part_stats[part, cv2.CC_STAT_WIDTH], part_stats[part, cv2.CC_STAT_HEIGHT]
            cells = int(np.count_nonzero(occupied[y0 : y0 + h, x0 : x0 + w][part_labels[y0 : y0 + h, x0 : x0 + w] == part]))
            # A block cut by the frame edge may be a thin strip here and whole in the next frame.
            cut = x0 == 0 or y0 == 0 or x0 + w >= closed.shape[1] or y0 + h >= closed.shape[0]
            least = cfg.min_edge_block_cells if cut else cfg.min_block_cells
            if cells < least or cells < cfg.min_block_density * w * h:
                continue
            # A stray cell next to the block (periphery) stretches the box by a line; edge lines
            # with far fewer cells than the block are cut off.
            while w > 1 and h > 1:
                box = occupied[y0 : y0 + h, x0 : x0 + w]
                limit = cfg.edge_trim * float(box.mean())
                # A short line says too little about itself to be cut.
                rows_ok, cols_ok = w >= cfg.edge_trim_sites, h >= cfg.edge_trim_sites
                if rows_ok and box[0].mean() < limit:
                    y0, h = y0 + 1, h - 1
                elif rows_ok and box[-1].mean() < limit:
                    h -= 1
                elif cols_ok and box[:, 0].mean() < limit:
                    x0, w = x0 + 1, w - 1
                elif cols_ok and box[:, -1].mean() < limit:
                    w -= 1
                else:
                    break
            blocks[y0 : y0 + h, x0 : x0 + w] = 1
        closed = blocks
        r_idx, c_idx = np.nonzero(closed)
        xs.append(phase_x + c_idx * pitch.x)
        ys.append(phase_y + r_idx * pitch.y)
        gs.append(np.full(len(r_idx), gi, np.int16))
        # Clean cells of this group around each of its sites (a window of a few sites).
        win = 2 * cfg.owner_sites + 1
        supports.append((phase_x, phase_y, cv2.boxFilter(occupied.astype(np.float32), -1, (win, win), normalize=False)))
        totals.append(int(occupied.sum()))
    if not xs:
        return _empty_sites(frame_id, shape)
    x, y, group = np.concatenate(xs), np.concatenate(ys), np.concatenate(gs)
    if len(supports) > 1:
        # Where boxes of two phase groups overlap, a place belongs to the group whose cells are
        # around it (more of them; the bigger group on a tie): the other group's sites there fall
        # between real cells.
        support = np.zeros((len(supports), len(x)), np.float32)
        for h, (phase_x, phase_y, grid) in enumerate(supports):
            c = np.round((x - phase_x) / pitch.x).astype(int)
            r = np.round((y - phase_y) / pitch.y).astype(int)
            inside = (r >= 0) & (r < grid.shape[0]) & (c >= 0) & (c < grid.shape[1])
            support[h, inside] = grid[r[inside], c[inside]]
        support += np.asarray(totals, np.float32)[:, None] * 1.0e-6
        keep = support.argmax(axis=0) == group
        x, y, group = x[keep], y[keep], group[keep]
    xi = np.clip(np.round(x).astype(int), 0, cols - 1)
    yi = np.clip(np.round(y).astype(int), 0, rows - 1)

    painted = []
    defect = np.zeros(len(x), bool)
    partial_painted = np.zeros(len(x), bool)
    conductor = np.zeros(len(x), bool)
    for layer in layers:
        if layer is None:
            painted.append(None)
            continue
        _binary, labels, stats, _centroids = layer
        label = labels[yi, xi]
        hit = label > 0
        painted.append(hit)
        if not hit.any():
            continue
        # A piece over several sites is merged cells; a small one on a site is a fragment.
        sites_per_label = np.bincount(label[hit], minlength=len(stats))
        s = stats[label]
        touches = (s[:, 0] <= 0) | (s[:, 1] <= 0) | (s[:, 0] + s[:, 2] >= cols) | (s[:, 1] + s[:, 3] >= rows)
        partial_painted |= hit & touches
        conductor |= hit & (s[:, cv2.CC_STAT_AREA] > cfg.conductor_cells * max(area, 1.0))
        merged = hit & (sites_per_label[label] > 1) & ~touches
        fragment = hit & (s[:, cv2.CC_STAT_AREA] < cfg.fragment_area * area) & ~touches
        defect |= merged | fragment
    p_one = painted[0] if painted[0] is not None else np.zeros(len(x), bool)
    p_zero = painted[1] if painted[1] is not None else np.zeros(len(x), bool)
    state = np.full(len(x), EMPTY, np.uint8)
    state[p_one & ~p_zero] = ONE
    state[p_zero & ~p_one] = ZERO
    state[p_one & p_zero] = CONFLICT
    for bx, by, bw, bh in defect_boxes:
        defect |= (x >= bx) & (x < bx + bw) & (y >= by) & (y < by + bh) & (p_one | p_zero)
    state[defect] = DEFECT
    # Cut by the frame: a painted cell touching the edge, or an empty site close to it.
    near_edge = np.minimum.reduce([x, y, cols - 1 - x, rows - 1 - y]) < cfg.edge_margin * min(pitch.x, pitch.y)
    state[(partial_painted & ~conductor) | ((state == EMPTY) & near_edge)] = PARTIAL
    # Sites on a conductor are not memory sites at all.
    keep = ~conductor
    return FrameSites(frame_id, shape, x[keep], y[keep], state[keep], group[keep], tuple(phases))


# --- Seams --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Seam:
    """Two neighbour frames: how B lies relative to A (B's pixel = A's pixel - offset), and how sure."""

    a: int
    b: int
    kind: str  # "right" or "down"
    status: str
    offset: tuple[float, float] = (0.0, 0.0)
    agreement: float = 0.0
    compared: int = 0
    lead: float = 0.0


def match_frames(a: FrameSites, b: FrameSites, kind: str, pitch: Pitch, config: MemoryMapConfig | None = None, *, ia: int = 0, ib: int = 1) -> Seam:
    """Best lattice translation of B onto A at a right or bottom seam, by the states of shared sites."""

    from scipy.spatial import cKDTree

    cfg = config or MemoryMapConfig()
    # A defect mark is a note over the value, made frame by frame (a neighbour may see the same cell
    # whole and unmarked): like cells cut by the edge, such sites are not compared.
    known_a = (a.state != PARTIAL) & (a.state != DEFECT)
    known_b = (b.state != PARTIAL) & (b.state != DEFECT)
    if np.count_nonzero(known_a) < cfg.min_compared or np.count_nonzero(known_b) < cfg.min_compared:
        return Seam(ia, ib, kind, SEAM_EMPTY)
    rows, cols = a.shape
    over = cfg.max_overlap_sites
    shift = cfg.max_shift_sites
    if kind == "right":
        lo = np.array([cols - over * pitch.x, -shift * pitch.y])
        hi = np.array([cols + 0.5 * pitch.x, shift * pitch.y])
    else:
        lo = np.array([-shift * pitch.x, rows - over * pitch.y])
        hi = np.array([shift * pitch.x, rows + 0.5 * pitch.y])
    # Only the strips that can overlap take part: A's edge and B's opposite edge.
    reach_x = (over + 0.5) * pitch.x
    reach_y = (over + 0.5) * pitch.y
    if kind == "right":
        band_a = known_a & (a.x >= cols - reach_x)
        band_b = known_b & (b.x < reach_x)
    else:
        band_a = known_a & (a.y >= rows - reach_y)
        band_b = known_b & (b.y < reach_y)
    if np.count_nonzero(band_a) < cfg.min_compared or np.count_nonzero(band_b) < cfg.min_compared:
        return Seam(ia, ib, kind, SEAM_EMPTY)

    def phases_in(sites: FrameSites, band: np.ndarray) -> list[tuple[float, float]]:
        # Phases with sites in the strip, the most populated first (noise groups drop out).
        counts = np.bincount(sites.group[band].astype(np.int64), minlength=len(sites.phases))
        order = [g for g in np.argsort(-counts) if counts[g] >= max(4, 0.1 * counts.max())]
        return [sites.phases[g] for g in order[:3]]

    # Lattice translations that put a phase of B on a phase of A, inside the window; the same
    # translation reached through two phase pairs is tried once.
    candidates: list[tuple[float, float]] = []
    for pa in phases_in(a, band_a):
        for pb in phases_in(b, band_b):
            base = np.array([(pa[0] - pb[0]) % pitch.x, (pa[1] - pb[1]) % pitch.y])
            m = np.arange(np.ceil((lo[0] - base[0]) / pitch.x), np.floor((hi[0] - base[0]) / pitch.x) + 1)
            n = np.arange(np.ceil((lo[1] - base[1]) / pitch.y), np.floor((hi[1] - base[1]) / pitch.y) + 1)
            for mi in m:
                for ni in n:
                    t = (float(base[0] + mi * pitch.x), float(base[1] + ni * pitch.y))
                    if all(abs(t[0] - c[0]) > 2.0 or abs(t[1] - c[1]) > 2.0 for c in candidates):
                        candidates.append(t)
    if not candidates:
        return Seam(ia, ib, kind, SEAM_EMPTY)
    tree = cKDTree(np.stack([a.x[band_a], a.y[band_a]], axis=1))
    state_a = a.state[band_a]
    bx, by, state_b = b.x[band_b], b.y[band_b], b.state[band_b]
    radius = cfg.centre_tolerance * min(pitch.x, pitch.y)
    scored = []
    for tx, ty in candidates:
        # B's pixel (x, y) is A's pixel (x + tx, y + ty).
        moved = np.stack([bx + tx, by + ty], axis=1)
        inside = (moved[:, 0] >= 0) & (moved[:, 0] < cols) & (moved[:, 1] >= 0) & (moved[:, 1] < rows)
        if not inside.any():
            continue
        dist, idx = tree.query(moved[inside], distance_upper_bound=radius)
        found = np.isfinite(dist)
        compared = int(np.count_nonzero(found))
        if compared < cfg.min_compared:
            continue
        agree = float(np.mean(state_a[idx[found]] == state_b[inside][found]))
        scored.append((agree, compared, (tx, ty)))
    if not scored:
        return Seam(ia, ib, kind, SEAM_EMPTY)
    # Agreement decides: a wrong translation compares as many sites or more (a wider overlap),
    # but matches only by chance (about p^2 + (1-p)^2 of them).
    scored.sort(key=lambda entry: (entry[0], entry[1]), reverse=True)
    agree, compared, (tx, ty) = scored[0]
    # The runner-up is another translation, not the same one reached through another phase pair.
    apart = 0.5 * min(pitch.x, pitch.y)
    second = next(
        (entry[0] for entry in scored[1:] if np.hypot(entry[2][0] - tx, entry[2][1] - ty) > apart), 0.0
    )
    lead = agree - second
    if agree < cfg.min_agreement:
        status = SEAM_FAILED
    elif lead < cfg.min_lead:
        status = SEAM_AMBIGUOUS
    else:
        status = SEAM_BITS
    return Seam(ia, ib, kind, status, (tx, ty), agree, compared, lead)


def detect_frames_per_row(
    sites: Sequence[FrameSites],
    pitch: Pitch,
    config: MemoryMapConfig | None = None,
    *,
    samples: int = 5,
    max_per_row: int | None = None,
    progress=None,
) -> int | None:
    """Frames per row from the bits: for a few frames with memory, the frame that continues each
    one downwards is the one whose top rows match its bottom rows. The number found for at least
    two frames wins; None when the frames do not overlap downwards (or nothing matched)."""

    cfg = config or MemoryMapConfig()
    count = len(sites)
    limit = min(count - 1, int(max_per_row) if max_per_row else count - 1)
    if limit < 1:
        return None
    rich = [i for i, s in enumerate(sites) if s.count >= 4 * cfg.min_compared and i + 1 < count]
    if not rich:
        return None
    # Spread the samples over the run, avoiding its very start (periphery) where possible.
    picks = [rich[int(k * (len(rich) - 1) / max(1, samples - 1))] for k in range(min(samples, len(rich)))]
    votes: dict[int, int] = {}
    for done, index in enumerate(picks):
        for step in range(1, limit + 1):
            other = index + step
            if other >= count:
                break
            if sites[other].count < cfg.min_compared:
                continue
            seam = match_frames(sites[index], sites[other], "down", pitch, cfg, ia=index, ib=other)
            if seam.status == SEAM_BITS:
                votes[step] = votes.get(step, 0) + 1
                break
        if progress is not None:
            progress(done + 1, len(picks))
        best = max(votes.items(), key=lambda item: item[1], default=None)
        if best is not None and best[1] >= 2 and best[1] > sum(votes.values()) - best[1]:
            return best[0]
    best = max(votes.items(), key=lambda item: item[1], default=None)
    if best is None:
        return None
    if best[1] >= 2 or (len(picks) == 1 and best[1] == 1):
        return best[0]
    return None


def frame_neighbours(count: int, per_row: int) -> list[tuple[int, int, str]]:
    """Seams of frames laid row by row: (a, b, kind) for the right and the lower neighbour."""

    pairs = []
    for index in range(count):
        if (index + 1) % per_row and index + 1 < count:
            pairs.append((index, index + 1, "right"))
        if index + per_row < count:
            pairs.append((index, index + per_row, "down"))
    return pairs


# --- Field --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FramePlacement:
    frame_id: str
    x: float
    y: float
    method: str
    width: int = 0
    height: int = 0


@dataclass(frozen=True, slots=True)
class MapBlock:
    """One block in field coordinates: a box of the field (sites outside the block are OUTSIDE)."""

    index: int
    row: int
    col: int
    rows: int
    cols: int
    sites: int
    # Pixel offset of this block's sites from the field lattice (blocks shifted by part of a pitch).
    offset: tuple[float, float] = (0.0, 0.0)
    # A small block apart from the memory arrays (service rows, redundancy, test structures).
    # Found by size; the operator may make it a main block, or a main block a service one.
    service: bool = False


@dataclass(frozen=True, slots=True)
class MemoryMap:
    mode: str
    pitch: Pitch
    field: np.ndarray
    source: np.ndarray
    blocks: tuple[MapBlock, ...]
    frames: tuple[FramePlacement, ...]
    seams: tuple[Seam, ...]
    # Field (0, 0) is this pixel of the frame plane (frame positions are in that plane too).
    origin: tuple[float, float] = (0.0, 0.0)
    disagreements: int = 0
    collisions: int = 0
    block_index: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), np.int32))
    # A map turned or mirrored for export keeps where each site was on the built field
    # (row, column grids; empty for the built map itself).
    orig_rows: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), np.int32))
    orig_cols: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), np.int32))
    # Frames per row the map was laid out with (found from the bits when asked for «auto»).
    per_row: int = 0

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.field.shape)  # type: ignore[return-value]

    def counts(self) -> dict[str, int]:
        values, counts = np.unique(self.field, return_counts=True)
        return {STATE_NAMES[int(v)]: int(c) for v, c in zip(values, counts)}

    def seam_counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for seam in self.seams:
            out[seam.status] = out.get(seam.status, 0) + 1
        return out

    def frame_point(self, row: int, col: int) -> tuple[int, float, float] | None:
        """The frame that saw a field site best, and the site's pixel in that frame."""

        index = int(self.source[row, col])
        if index < 0:
            return None
        frame = self.frames[index]
        block = int(self.block_index[row, col]) if self.block_index.size else -1
        if self.orig_rows.size:
            row, col = int(self.orig_rows[row, col]), int(self.orig_cols[row, col])
        dx, dy = self.blocks[block].offset if 0 <= block < len(self.blocks) else (0.0, 0.0)
        px = self.origin[0] + col * self.pitch.x + dx - frame.x
        py = self.origin[1] + row * self.pitch.y + dy - frame.y
        return index, float(px), float(py)


def _place_frames(
    sites: Sequence[FrameSites], seams: Sequence[Seam], per_row: int, cfg: MemoryMapConfig, pitch: Pitch
) -> tuple[list[tuple[float, float]], list[str], list[Seam]]:
    count = len(sites)
    trusted = [seam for seam in seams if seam.status == SEAM_BITS]

    def typical(kind: str, fallback: tuple[float, float]) -> tuple[float, float]:
        offsets = np.array([seam.offset for seam in trusted if seam.kind == kind])
        return (float(np.median(offsets[:, 0])), float(np.median(offsets[:, 1]))) if len(offsets) else fallback

    rows, cols = sites[0].shape if sites else (0, 0)
    prior = {
        "right": typical("right", (float(cols), 0.0)),
        "down": typical("down", (0.0, float(rows))),
    }
    # Placement tree: trusted seams with the most evidence first (a maximum spanning forest), so
    # a weaker trusted seam that disagrees with stronger ones stays out of it and is reported.
    parent = list(range(count))

    def root(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    tree_edges: set[tuple[int, int]] = set()
    for seam in sorted(trusted, key=lambda item: (item.compared, item.lead), reverse=True):
        ra, rb = root(seam.a), root(seam.b)
        if ra != rb:
            parent[ra] = rb
            tree_edges.add((min(seam.a, seam.b), max(seam.a, seam.b)))
    # Links: a tree seam carries its own offset; any other seam the typical one for its kind.
    links: dict[int, list[tuple[int, tuple[float, float], bool]]] = {index: [] for index in range(count)}
    for seam in seams:
        in_tree = (min(seam.a, seam.b), max(seam.a, seam.b)) in tree_edges
        offset = seam.offset if in_tree else prior[seam.kind]
        links[seam.a].append((seam.b, offset, in_tree))
        links[seam.b].append((seam.a, (-offset[0], -offset[1]), in_tree))
    position: list[tuple[float, float] | None] = [None] * count
    method = [PLACED_ESTIMATED] * count
    if count:
        start = max(range(count), key=lambda i: sum(1 for _o, _off, tree in links[i] if tree))
        # 0-1 BFS: a frame is placed when taken from the queue, by the fewest estimated links.
        queue: deque[tuple[int, tuple[float, float], str]] = deque([(start, (0.0, 0.0), PLACED_BITS)])
        while queue:
            node, where, how = queue.popleft()
            if position[node] is not None:
                continue
            position[node], method[node] = where, how
            for other, offset, in_tree in links[node]:
                if position[other] is not None:
                    continue
                item = (other, (where[0] + offset[0], where[1] + offset[1]), PLACED_BITS if in_tree else PLACED_ESTIMATED)
                (queue.appendleft if in_tree else queue.append)(item)
        # Frames with no seams at all: their place in the row-by-row grid.
        for index in range(count):
            if position[index] is None:
                position[index] = (
                    (index % per_row) * prior["right"][0] + (index // per_row) * prior["down"][0],
                    (index % per_row) * prior["right"][1] + (index // per_row) * prior["down"][1],
                )
    checked = []
    limit = cfg.seam_tolerance * min(pitch.x, pitch.y)
    for seam in seams:
        if seam.status == SEAM_BITS and (min(seam.a, seam.b), max(seam.a, seam.b)) not in tree_edges:
            pa, pb = position[seam.a], position[seam.b]
            residual = np.hypot(pb[0] - pa[0] - seam.offset[0], pb[1] - pa[1] - seam.offset[1])
            if residual > limit:
                seam = Seam(seam.a, seam.b, seam.kind, SEAM_INCONSISTENT, seam.offset, seam.agreement, seam.compared, seam.lead)
        checked.append(seam)
    return [p for p in position], method, checked  # type: ignore[misc]


def assemble_memory_map(
    sites: Sequence[FrameSites],
    pitch: Pitch,
    *,
    per_row: int,
    mode: str = MODE_PAIR,
    seams: Sequence[Seam] | None = None,
    config: MemoryMapConfig | None = None,
) -> MemoryMap:
    """The whole memory from the sites of every frame (frames in row-by-row order)."""

    cfg = config or MemoryMapConfig()
    per_row = max(1, int(per_row))
    if seams is None:
        seams = [
            match_frames(sites[a], sites[b], kind, pitch, cfg, ia=a, ib=b)
            for a, b, kind in frame_neighbours(len(sites), per_row)
        ]
    positions, methods, seams = _place_frames(sites, seams, per_row, cfg, pitch)
    frames = tuple(
        FramePlacement(s.frame_id, float(p[0]), float(p[1]), m, int(s.shape[1]), int(s.shape[0]))
        for s, p, m in zip(sites, positions, methods)
    )
    # Every site in the frame plane: (X, Y) = frame position + site pixel.
    gx, gy, st, edge, src, grp = [], [], [], [], [], []
    # Phases of the frames' site groups in the frame plane, merged into the field's groups.
    clusters: list[list[float]] = []  # [phase x, phase y, weight]
    tol_x, tol_y = cfg.phase_tolerance * pitch.x, cfg.phase_tolerance * pitch.y
    local_to_global: dict[tuple[int, int], int] = {}
    entries = []
    for index, (s, p) in enumerate(zip(sites, positions)):
        for local, phase in enumerate(s.phases):
            weight = int(np.count_nonzero(s.group == local))
            if weight:
                entries.append((weight, index, local, (phase[0] + p[0]) % pitch.x, (phase[1] + p[1]) % pitch.y))
    for weight, index, local, phase_x, phase_y in sorted(entries, reverse=True):
        found = next(
            (
                ci
                for ci, (cx, cy, _w) in enumerate(clusters)
                if _circular_diff(phase_x, cx, pitch.x) <= tol_x and _circular_diff(phase_y, cy, pitch.y) <= tol_y
            ),
            None,
        )
        if found is None:
            clusters.append([phase_x, phase_y, float(weight)])
            found = len(clusters) - 1
        local_to_global[(index, local)] = found
    for index, (s, p) in enumerate(zip(sites, positions)):
        if not s.count:
            continue
        gx.append(s.x + p[0])
        gy.append(s.y + p[1])
        st.append(s.state)
        edge.append(s.edge_distance())
        src.append(np.full(s.count, index, np.int32))
        lookup = np.array([local_to_global.get((index, local), 0) for local in range(len(s.phases))] or [0])
        grp.append(lookup[s.group])
    if not gx:
        empty = np.zeros((0, 0), np.uint8)
        return MemoryMap(mode, pitch, empty, np.zeros((0, 0), np.int32), (), frames, tuple(seams))
    X, Y = np.concatenate(gx), np.concatenate(gy)
    state, edge, source = np.concatenate(st), np.concatenate(edge), np.concatenate(src)
    group = np.concatenate(grp)
    # One lattice for the field: the phase of the biggest group. A group shifted by part of a
    # pitch snaps to it by one whole number of sites, chosen once for the group: rounding site by
    # site would send a half-pitch block up in one frame and down in the next.
    phase = np.array([[c[0], c[1]] for c in clusters])
    main = int(np.argmax([c[2] for c in clusters]))
    ox, oy = float(phase[main, 0]), float(phase[main, 1])
    snap_x = np.round((phase[:, 0] - ox) / pitch.x).astype(np.int64)
    snap_y = np.round((phase[:, 1] - oy) / pitch.y).astype(np.int64)
    col = np.round((X - phase[group, 0]) / pitch.x).astype(np.int64) + snap_x[group]
    row = np.round((Y - phase[group, 1]) / pitch.y).astype(np.int64) + snap_y[group]
    # Where a group's sites sit relative to the field lattice, in pixels (for the frame point).
    group_offset = np.stack([phase[:, 0] - ox - snap_x * pitch.x, phase[:, 1] - oy - snap_y * pitch.y], axis=1)
    col0, row0 = int(col.min()), int(row.min())
    col -= col0
    row -= row0
    n_rows, n_cols = int(row.max()) + 1, int(col.max()) + 1
    key = row * n_cols + col
    # Best view of a site: known (not partial) first, then farthest from its frame edge.
    rank = np.lexsort((-edge, state == PARTIAL, key))
    key_sorted = key[rank]
    first = np.ones(len(rank), bool)
    first[1:] = key_sorted[1:] != key_sorted[:-1]
    best = rank[first]
    field_state = np.full(n_rows * n_cols, OUTSIDE, np.uint8)
    field_source = np.full(n_rows * n_cols, -1, np.int32)
    field_group = np.full(n_rows * n_cols, -1, np.int32)
    field_state[key[best]] = state[best]
    field_source[key[best]] = source[best]
    field_group[key[best]] = group[best]
    # Disagreements: known views of one site with different states.
    known = rank[state[rank] != PARTIAL]
    k_key = key[known]
    k_state = state[known]
    order = np.lexsort((k_state, k_key))
    k_key, k_state = k_key[order], k_state[order]
    starts = np.ones(len(k_key), bool)
    starts[1:] = k_key[1:] != k_key[:-1]
    changes = np.zeros(len(k_key), bool)
    changes[1:] = (k_key[1:] == k_key[:-1]) & (k_state[1:] != k_state[:-1])
    disagreements = int(len(np.unique(k_key[changes]))) if changes.any() else 0
    # Collisions: one field site got views of physically different sites (blocks shifted by part
    # of a pitch next to each other), told by views more than a third of a pitch apart.
    by_key = np.argsort(key, kind="stable")
    sk = key[by_key]
    starts_at = np.flatnonzero(np.r_[True, sk[1:] != sk[:-1]])
    spread_x = np.maximum.reduceat(X[by_key], starts_at) - np.minimum.reduceat(X[by_key], starts_at)
    spread_y = np.maximum.reduceat(Y[by_key], starts_at) - np.minimum.reduceat(Y[by_key], starts_at)
    collisions = int(np.count_nonzero(np.hypot(spread_x, spread_y) > 0.34 * min(pitch.x, pitch.y)))

    grid = field_state.reshape(n_rows, n_cols)
    source_grid = field_source.reshape(n_rows, n_cols)
    group_grid = field_group.reshape(n_rows, n_cols)
    # With one network an all-zero (or all-one) line inside a block looks exactly like a gap
    # between blocks; only with both networks is «nothing painted» a gap for sure.
    grid, block_index, blocks = _blocks(grid, cfg, find_gaps=mode == MODE_PAIR)
    # Each block keeps the pixel offset of the group most of its sites come from.
    with_offset = []
    for b in blocks:
        ids = group_grid[b.row : b.row + b.rows, b.col : b.col + b.cols][block_index[b.row : b.row + b.rows, b.col : b.col + b.cols] == b.index]
        ids = ids[ids >= 0]
        g = int(np.bincount(ids).argmax()) if ids.size else main
        with_offset.append(MapBlock(b.index, b.row, b.col, b.rows, b.cols, b.sites, (float(group_offset[g, 0]), float(group_offset[g, 1]))))
    blocks = with_offset
    source_grid = np.where(grid == OUTSIDE, -1, source_grid)
    # Trim the field to its blocks.
    if blocks:
        r0 = min(b.row for b in blocks)
        c0 = min(b.col for b in blocks)
        r1 = max(b.row + b.rows for b in blocks)
        c1 = max(b.col + b.cols for b in blocks)
        grid, source_grid, block_index = grid[r0:r1, c0:c1], source_grid[r0:r1, c0:c1], block_index[r0:r1, c0:c1]
        blocks = [MapBlock(b.index, b.row - r0, b.col - c0, b.rows, b.cols, b.sites, b.offset) for b in blocks]
        largest = max(b.sites for b in blocks)
        blocks = [replace(b, service=b.sites < cfg.service_share * largest) for b in blocks]
        col0 += c0
        row0 += r0
    origin = (ox + col0 * pitch.x, oy + row0 * pitch.y)
    return MemoryMap(
        mode=mode,
        pitch=pitch,
        field=np.ascontiguousarray(grid),
        source=np.ascontiguousarray(source_grid),
        blocks=tuple(blocks),
        frames=frames,
        seams=tuple(seams),
        origin=origin,
        disagreements=disagreements,
        collisions=collisions,
        block_index=np.ascontiguousarray(block_index),
        per_row=per_row,
    )


def _blocks(grid: np.ndarray, cfg: MemoryMapConfig, *, find_gaps: bool = True) -> tuple[np.ndarray, np.ndarray, list[MapBlock]]:
    """Gap lines (nothing painted along a long run) become OUTSIDE; the rest splits into blocks."""

    grid = grid.copy()
    exists = grid != OUTSIDE
    painted = np.isin(grid, PAINTED)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(exists.astype(np.uint8), connectivity=4)
    for label in range(1, count if find_gaps else 1):
        c0, r0 = int(stats[label, cv2.CC_STAT_LEFT]), int(stats[label, cv2.CC_STAT_TOP])
        r1, c1 = r0 + int(stats[label, cv2.CC_STAT_HEIGHT]), c0 + int(stats[label, cv2.CC_STAT_WIDTH])
        sub_exists = labels[r0:r1, c0:c1] == label
        sub_painted = painted[r0:r1, c0:c1] & sub_exists
        for axis in (0, 1):
            n_sites = sub_exists.sum(axis=axis)
            n_painted = sub_painted.sum(axis=axis)
            gap = (n_sites >= cfg.gap_min_sites) & (n_painted <= cfg.gap_max_painted * np.maximum(n_sites, 1))
            if not gap.any():
                continue
            region = grid[r0:r1, c0:c1]
            if axis == 0:
                region[:, gap & True] = np.where(sub_exists[:, gap], OUTSIDE, region[:, gap])
            else:
                region[gap, :] = np.where(sub_exists[gap, :], OUTSIDE, region[gap, :])
    exists = grid != OUTSIDE
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(exists.astype(np.uint8), connectivity=4)
    blocks: list[MapBlock] = []
    block_index = np.full(grid.shape, -1, np.int32)
    order = sorted(range(1, count), key=lambda label: (stats[label, cv2.CC_STAT_TOP], stats[label, cv2.CC_STAT_LEFT]))
    painted = np.isin(grid, PAINTED)
    for label in order:
        sites = int(stats[label, cv2.CC_STAT_AREA])
        c0, r0 = int(stats[label, cv2.CC_STAT_LEFT]), int(stats[label, cv2.CC_STAT_TOP])
        box = (slice(r0, r0 + int(stats[label, cv2.CC_STAT_HEIGHT])), slice(c0, c0 + int(stats[label, cv2.CC_STAT_WIDTH])))
        mask = labels[box] == label
        if sites < cfg.min_block_sites or np.count_nonzero(painted[box][mask]) < cfg.min_block_painted * sites:
            grid[box][mask] = OUTSIDE
            continue
        index = len(blocks)
        block_index[box][mask] = index
        blocks.append(
            MapBlock(
                index=index,
                row=int(stats[label, cv2.CC_STAT_TOP]),
                col=int(stats[label, cv2.CC_STAT_LEFT]),
                rows=int(stats[label, cv2.CC_STAT_HEIGHT]),
                cols=int(stats[label, cv2.CC_STAT_WIDTH]),
                sites=sites,
            )
        )
    return grid, block_index, blocks


# --- Whole run ----------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MapFrame:
    """One frame of the run: its id and the mask files of the two networks (either may be empty)."""

    frame_id: str
    ones_path: str = ""
    zeros_path: str = ""
    # Boxes (x, y, w, h) in frame pixels where the cell analysis found defects.
    defect_boxes: tuple[tuple[int, int, int, int], ...] = ()


class BuildCancelled(Exception):
    pass


def _read_mask(path: str) -> np.ndarray | None:
    if not path:
        return None
    try:
        data = np.fromfile(path, np.uint8)
    except OSError:
        return None
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE) if data.size else None


def map_mode(frames: Sequence[MapFrame]) -> str:
    has_ones = any(frame.ones_path for frame in frames)
    has_zeros = any(frame.zeros_path for frame in frames)
    if has_ones and has_zeros:
        return MODE_PAIR
    return MODE_ZEROS if has_zeros else MODE_ONES


MEMORY_MAP_VERSION = "memory_map.v3"
READ_SETTINGS = (
    "clean_area",
    "fragment_area",
    "phase_tolerance",
    "min_group_cells",
    "min_group_share",
    "max_groups",
    "centre_tolerance",
    "close_sites",
    "min_block_cells",
    "min_edge_block_cells",
    "min_block_density",
    "owner_sites",
    "edge_trim",
    "edge_trim_sites",
    "conductor_cells",
    "edge_margin",
)
SITE_CACHE_MAX_BYTES = 512 * 1024 * 1024


def _file_stamp(path: str) -> list:
    if not path:
        return []
    try:
        stat = Path(path).stat()
    except OSError:
        return [str(path)]
    return [str(Path(path).resolve()), int(stat.st_size), int(stat.st_mtime_ns)]


def site_cache_key(frame: MapFrame, pitch: Pitch, config: MemoryMapConfig) -> str:
    """A frame's sites change only with its masks (path, size, time), the pitch or the settings."""

    import hashlib
    import json
    from dataclasses import asdict

    payload = {
        "version": MEMORY_MAP_VERSION,
        "pitch": [round(pitch.x, 4), round(pitch.y, 4)],
        # Only what reading a frame depends on: seam and field settings do not re-read frames.
        "config": {key: value for key, value in asdict(config).items() if key in READ_SETTINGS},
        "ones": _file_stamp(frame.ones_path),
        "zeros": _file_stamp(frame.zeros_path),
        "defects": [list(box) for box in frame.defect_boxes],
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _compact(sites: FrameSites) -> FrameSites:
    return FrameSites(
        sites.frame_id,
        sites.shape,
        sites.x.astype(np.float32),
        sites.y.astype(np.float32),
        sites.state,
        sites.group,
        sites.phases,
    )


def _read_one(frame: MapFrame, pitch: Pitch, config: MemoryMapConfig) -> FrameSites:
    ones = _read_mask(frame.ones_path)
    zeros = _read_mask(frame.zeros_path)
    if ones is None and zeros is None:
        return _empty_sites(frame.frame_id, (0, 0))
    return _compact(
        read_frame_sites(ones, zeros, pitch, frame_id=frame.frame_id, defect_boxes=frame.defect_boxes, config=config)
    )


def _read_batch(batch: list[tuple[int, MapFrame]], pitch: Pitch, config: MemoryMapConfig) -> list[tuple[int, FrameSites]]:
    """Process-pool job: a few frames at once (one pickling round trip for several frames)."""

    return [(index, _read_one(frame, pitch, config)) for index, frame in batch]


def build_memory_map(
    frames: Sequence[MapFrame],
    *,
    per_row: int,
    pitch: Pitch | None = None,
    config: MemoryMapConfig | None = None,
    progress=None,
    cancelled=None,
    workers: int | None = None,
    pitch_sample: int = 24,
    use_processes: bool = True,
    cache_dir: Path | str | None = "default",
    startup_timeout: float = 120.0,
) -> MemoryMap:
    """Read every frame, sew the seams, lay out the field. progress(stage, done, total) is called
    as it goes («pitch», «frames», «per_row», «seams», «field»); cancelled() returning True stops it.
    per_row <= 0 finds the frames per row from the bits.

    Frames are read in worker processes (in threads when processes cannot start or are off), and
    each frame's sites are kept in a cache: a second build of the same masks reads none of them.
    """

    import os
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

    cfg = config or MemoryMapConfig()
    mode = map_mode(frames)
    total = len(frames)
    worker_count = max(1, int(workers or (os.cpu_count() or 4)))

    def report(stage: str, done: int, count: int) -> None:
        if progress is not None:
            progress(stage, done, count)
        if cancelled is not None and cancelled():
            raise BuildCancelled()

    if pitch is None:
        step = max(1, total // max(1, pitch_sample))
        sample = []
        for frame in frames[::step][:pitch_sample]:
            mask = _read_mask(frame.ones_path) if frame.ones_path else _read_mask(frame.zeros_path)
            if mask is not None:
                sample.append(mask)
        report("pitch", 1, 1)
        pitch = estimate_pitch(sample, cfg)
        if pitch is None:
            raise ValueError("no cell lattice found in the masks")

    # Cached frames first.
    if cache_dir == "default":
        from .backend_constants import CACHE_DIR

        cache_dir = CACHE_DIR / "memory_map_sites"
    cache_path = Path(cache_dir) if cache_dir else None
    results: list[FrameSites | None] = [None] * total
    keys: list[str] = [""] * total
    if cache_path is not None:
        import pickle

        for index, frame in enumerate(frames):
            keys[index] = site_cache_key(frame, pitch, cfg)
            entry = cache_path / f"{keys[index]}.pickle"
            if entry.is_file():
                try:
                    with entry.open("rb") as handle:
                        cached = pickle.load(handle)
                    if isinstance(cached, FrameSites):
                        results[index] = cached
                except Exception:
                    results[index] = None
    todo = [(index, frame) for index, frame in enumerate(frames) if results[index] is None]
    done = total - len(todo)
    report("frames", done, total)

    def store(index: int, sites: FrameSites) -> None:
        results[index] = sites
        if cache_path is not None and keys[index]:
            from .cache_utils import atomic_pickle_dump

            try:
                atomic_pickle_dump(cache_path / f"{keys[index]}.pickle", sites)
            except OSError:
                pass

    if todo:
        size = max(2, min(32, -(-len(todo) // (worker_count * 4))))
        batches = [todo[i : i + size] for i in range(0, len(todo), size)]
        remaining = list(batches)
        if use_processes and worker_count > 1 and len(todo) > size:
            # Processes skip the interpreter lock; a pool that cannot start (or breaks) hands the
            # frames it did not read to the threads below.
            import multiprocessing
            from concurrent.futures import ProcessPoolExecutor

            pool = None
            try:
                pool = ProcessPoolExecutor(max_workers=worker_count, mp_context=multiprocessing.get_context("spawn"))
                futures = {pool.submit(_read_batch, batch, pitch, cfg): batch for batch in batches}
                pending = set(futures)
                first = True
                while pending:
                    finished, pending = wait(pending, timeout=startup_timeout if first else None, return_when=FIRST_COMPLETED)
                    if not finished:
                        raise TimeoutError("frame readers did not start")
                    first = False
                    for future in finished:
                        for index, sites in future.result():
                            store(index, sites)
                        remaining.remove(futures[future])
                        done += len(futures[future])
                        report("frames", done, total)
            except BuildCancelled:
                if pool is not None:
                    pool.shutdown(wait=False, cancel_futures=True)
                raise
            except Exception:
                pass
            finally:
                if pool is not None:
                    pool.shutdown(wait=False, cancel_futures=True)
        if remaining:
            with ThreadPoolExecutor(max_workers=worker_count) as threads:
                for batch_result in threads.map(lambda batch: _read_batch(batch, pitch, cfg), remaining):
                    for index, sites in batch_result:
                        store(index, sites)
                        done += 1
                    report("frames", done, total)
        if cache_path is not None:
            from .cache_utils import trim_directory_by_bytes

            trim_directory_by_bytes(cache_path, max_bytes=SITE_CACHE_MAX_BYTES)

    # Frames that could not be read take the size of the others.
    shape = next((r.shape for r in results if r is not None and r.shape != (0, 0)), (0, 0))
    sites = [
        r if r is not None and r.shape != (0, 0) else _empty_sites(frames[i].frame_id, shape) for i, r in enumerate(results)
    ]
    if int(per_row) <= 0:
        # «Auto»: frames per row from the bits.
        report("per_row", 0, 1)
        found = detect_frames_per_row(sites, pitch, cfg, progress=lambda done, n: report("per_row", done, n))
        if found is None:
            raise ValueError("frames per row could not be found from the masks; set it by hand")
        per_row = found
    pairs = frame_neighbours(total, per_row)
    seams = []
    for index, (a, b, kind) in enumerate(pairs):
        seams.append(match_frames(sites[a], sites[b], kind, pitch, cfg, ia=a, ib=b))
        if index % 50 == 0 or index + 1 == len(pairs):
            report("seams", index + 1, len(pairs))
    report("field", 0, 1)
    mmap = assemble_memory_map(sites, pitch, per_row=per_row, mode=mode, seams=seams, config=cfg)
    report("field", 1, 1)
    return mmap


def with_block_roles(mmap: MemoryMap, service: Iterable[int]) -> MemoryMap:
    """The same map with these blocks marked service and the rest main (the operator's choice)."""

    chosen = {int(index) for index in service}
    return replace(mmap, blocks=tuple(replace(b, service=b.index in chosen) for b in mmap.blocks))
