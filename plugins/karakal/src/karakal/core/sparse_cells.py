"""Sparse cells: cells with few sides that have a neighbour.

On a frame the network painted almost empty, a lone cell is suspicious: dirt or a reflection the
network took for a cell, or a stray piece of another structure. Such a cell may also be of a wrong
shape or carry other findings.

Around every cell an area is searched: X lattice pitches to the left and to the right, Y pitches up
and down. Every other cell in the area takes the sides of the cell it lies beyond: a neighbour in the
row takes the left or the right side, one in the column the top or the bottom, a diagonal one two
sides (top and right, say). A cell is sparse when at least N of its four sides are free (N is a
setting, 3 by default; 4: no neighbour in the area at all).

Near the frame edge the area goes on in the neighbour frames of the snapshot, the diagonal ones
too. Frames overlap; where a neighbour frame lies is found by the bits of the frames
(memory_map.match_frames) on frames rich enough to compare, and the usual offset of the frame row
(else of the whole run) serves the frames with too few cells for that.

A sparse cell that is otherwise normal is a finding of its own (``sparse_cell``, status
suspicious); a cell with another finding keeps it and gets ``sparse_cell`` as an extra reason.
Debris, unknown pieces, conductor zones and conflicts are not cells: they get no mark and are no
neighbours.

The step needs every frame of a layer, so it runs on the results once the frames of the layer are
analysed (workers); the analysis of one frame does not know about it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Mapping, Sequence

import numpy as np

SPARSE_REASON = "sparse_cell"
# No other cell within this many lattice pitches along the row (X) and the column (Y). «Auto» is this
# reach on both axes, of the pitch found from the run's own cells; the operator may set X and Y apart.
DEFAULT_SPARSE_REACH_SITES = 4
SPARSE_REACH_RANGE = (0.5, 20.0)
# A cell is sparse when this many of its four sides have no neighbour in the area.
DEFAULT_SPARSE_FREE_SIDES = 3
SPARSE_FREE_SIDES_RANGE = (2, 4)
SPARSE_SCORE = 0.6
# Reasons of objects that are cells (perhaps broken ones). Normal cells have no reason.
CELL_REASONS = frozenset(
    {
        "broken_geometry",
        "merged_contour",
        "split_cell",
        "edge_clipped_cell",
        "filled_cell",
        "partial_filled_cell",
        SPARSE_REASON,
    }
)
# Features of a marked cell: the search window it was found sparse in (pitches) and the run's pitch (px).
SPARSE_REACH_X_FEATURE = "sparse_reach_x_sites"
SPARSE_REACH_Y_FEATURE = "sparse_reach_y_sites"
SPARSE_PITCH_X_FEATURE = "sparse_pitch_x_px"
SPARSE_PITCH_Y_FEATURE = "sparse_pitch_y_px"
# Free sides of the marked cell (0-4) and how many the run asked for.
SPARSE_FREE_FEATURE = "sparse_free_sides"
SPARSE_MIN_FREE_FEATURE = "sparse_min_free_sides"
SPARSE_FEATURES = (
    SPARSE_FREE_FEATURE,
    SPARSE_MIN_FREE_FEATURE,
    f"reason:{SPARSE_REASON}",
    SPARSE_REACH_X_FEATURE,
    SPARSE_REACH_Y_FEATURE,
    SPARSE_PITCH_X_FEATURE,
    SPARSE_PITCH_Y_FEATURE,
    "sparse_reach_sites",  # 0.4.9803-0.4.9804: one reach for both axes
)


def normalize_reach(value) -> tuple[float, float]:
    """(X, Y) reach in pitches from one number (both axes) or a pair, kept within the range."""

    low, high = SPARSE_REACH_RANGE
    if isinstance(value, (tuple, list)) and len(value) >= 2:
        pair = (float(value[0]), float(value[1]))
    else:
        pair = (float(value), float(value))
    return tuple(max(low, min(high, item)) for item in pair)  # type: ignore[return-value]


def normalize_free_sides(value) -> int:
    low, high = SPARSE_FREE_SIDES_RANGE
    try:
        return int(max(low, min(high, int(value))))
    except (TypeError, ValueError):
        return DEFAULT_SPARSE_FREE_SIDES


@dataclass(frozen=True, slots=True)
class LayerFrame:
    """A frame of the layer: its result key, its mask (for the seams) and its place in the matrix."""

    key: str
    path: str
    col: int
    row: int


def layer_frames(keys_and_paths: Sequence[tuple[str, str]], per_row: int, tiles: Mapping[str, tuple[int, int]] | None = None) -> list[LayerFrame]:
    """Frames of a layer laid row by row as the matrix lays them: by their tile when the name gives
    one, else ``per_row`` frames to a row in their order. Without a row length (0) frames have no
    neighbours: each is looked at on its own."""

    columns = int(per_row)
    frames = []
    for index, (key, path) in enumerate(keys_and_paths):
        tile = (tiles or {}).get(str(key))
        if tile is not None:
            col, row = int(tile[0]), int(tile[1])
        elif columns > 0:
            col, row = index % columns, index // columns
        else:
            col, row = 2 * index, 0
        frames.append(LayerFrame(str(key), str(path or ""), col, row))
    return frames


def is_cell(cell) -> bool:
    reasons = {str(reason) for reason in (getattr(cell, "reasons", ()) or ())}
    if "conductor_zone" in reasons:
        return False
    if not reasons:
        # A normal cell (the lattice analysis of old also lists missing places without a reason).
        return str(getattr(cell, "status", "normal")) == "normal"
    return bool(reasons & CELL_REASONS)


def _boxes(result) -> tuple[list[int], np.ndarray]:
    """Indexes of the cells among the result's records and their boxes (x0, y0, x1, y1)."""

    indexes: list[int] = []
    boxes: list[tuple[float, float, float, float]] = []
    for index, cell in enumerate(getattr(result, "per_cell_results", ()) or ()):
        if not is_cell(cell):
            continue
        x, y, w, h = (float(value) for value in tuple(cell.bbox)[:4])
        indexes.append(index)
        boxes.append((x, y, x + w, y + h))
    return indexes, np.asarray(boxes, np.float64).reshape(-1, 4)


def _sides(first: np.ndarray, second: np.ndarray, reach_x: float, reach_y: float) -> np.ndarray:
    """Sides (left, right, top, bottom) of the ``first`` boxes the ``second`` boxes take: a box in the
    area (gaps within reach_x and reach_y) takes every side its centre lies beyond."""

    gap_x = np.maximum(second[..., 0] - first[..., 2], first[..., 0] - second[..., 2])
    gap_y = np.maximum(second[..., 1] - first[..., 3], first[..., 1] - second[..., 3])
    inside = (gap_x <= reach_x) & (gap_y <= reach_y)
    centre_x = (second[..., 0] + second[..., 2]) / 2.0
    centre_y = (second[..., 1] + second[..., 3]) / 2.0
    return np.stack(
        [
            inside & (centre_x < first[..., 0]),
            inside & (centre_x > first[..., 2]),
            inside & (centre_y < first[..., 1]),
            inside & (centre_y > first[..., 3]),
        ],
        axis=-1,
    )


def free_sides(boxes: np.ndarray, own: int, reach_x: float, reach_y: float, *, enough: int = 5) -> np.ndarray:
    """How many of the four sides of each of the first ``own`` boxes have no other box in the area.

    Boxes are (x0, y0, x1, y1); those after the first ``own`` are only neighbours (cells of the next
    frames). Reaches are gaps between boxes in pixels. ``enough``: a cell with fewer free sides than
    this is settled (its count may stay too high), so a filled frame costs little.
    """

    boxes = np.asarray(boxes, np.float64).reshape(-1, 4)
    own = int(own)
    count = len(boxes)
    if own <= 0:
        return np.zeros(0, np.int64)
    if count < 2:
        return np.full(own, 4, np.int64)
    from scipy.spatial import cKDTree

    centres = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2.0, (boxes[:, 1] + boxes[:, 3]) / 2.0], axis=1)
    k = min(9, count)
    _distance, nearest = cKDTree(centres).query(centres[:own], k=k)
    nearest = np.asarray(nearest).reshape(own, k)
    taken = _sides(boxes[:own, None, :], boxes[nearest], reach_x, reach_y)
    taken &= (nearest != np.arange(own)[:, None])[..., None]
    free = 4 - taken.any(axis=1).sum(axis=1)
    # In a filled frame the nearest boxes settle almost every cell; the rest is checked against all.
    rest = np.flatnonzero(free >= enough)
    for start in range(0, len(rest), 256):
        chunk = rest[start : start + 256]
        taken = _sides(boxes[chunk, None, :], boxes[None, :, :], reach_x, reach_y)
        taken[np.arange(len(chunk)), chunk, :] = False
        free[chunk] = 4 - taken.any(axis=1).sum(axis=1)
    return free


def lone_cells(boxes: np.ndarray, own: int, reach_x: float, reach_y: float, min_free: int = 4) -> np.ndarray:
    """Which of the first ``own`` boxes have at least ``min_free`` free sides (see free_sides)."""

    return free_sides(boxes, own, reach_x, reach_y, enough=int(min_free)) >= int(min_free)


def pitch_from_boxes(boxes_list: Sequence[np.ndarray]) -> tuple[float, float] | None:
    """Lattice pitch from cell boxes of rich frames: the usual distance to the next cell in a row and
    in a column (the smallest common one: a missing bit makes a gap of two pitches)."""

    from scipy.spatial import cKDTree

    dx_all: list[np.ndarray] = []
    dy_all: list[np.ndarray] = []
    for boxes in boxes_list:
        if len(boxes) < 8:
            continue
        pts = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2.0, (boxes[:, 1] + boxes[:, 3]) / 2.0], axis=1)
        _d, idx = cKDTree(pts).query(pts, k=min(9, len(pts)))
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
        near = values[values <= 1.35 * low]
        return float(np.median(near if near.size else values))

    return smallest_mode(dx), smallest_mode(dy)


class _Seams:
    """Where the next frame lies (its pixel = this frame's pixel - offset), found by bits.

    Asked only for frames with a lone cell near the edge: their own seam is used when both frames
    have enough cells, else the median of trusted seams of the frame row, else of the run.
    """

    ROW_TRIES = 4
    RUN_TRIES = 8

    def __init__(
        self,
        frames: Sequence[LayerFrame],
        counts: Mapping[str, int],
        read_mask: Callable[[str], np.ndarray | None],
        pitch: tuple[float, float],
    ) -> None:
        from .memory_map import MemoryMapConfig, Pitch

        self._by_place = {(frame.col, frame.row): frame for frame in frames}
        self._frames = list(frames)
        self._counts = counts
        self._read = read_mask
        self._pitch = Pitch(float(pitch[0]), float(pitch[1]))
        self._cfg = MemoryMapConfig()
        self._sites: dict[str, object] = {}
        self._matched: dict[tuple[str, str, str], np.ndarray | None] = {}
        self._usual: dict[tuple[str, int | None], np.ndarray | None] = {}
        self.trusted = 0

    def _frame_sites(self, frame: LayerFrame):
        if frame.key not in self._sites:
            from .memory_map import read_frame_sites

            sites = None
            mask = self._read(frame.path) if frame.path else None
            if mask is not None:
                try:
                    sites = read_frame_sites(mask, None, self._pitch, frame_id=frame.key, config=self._cfg)
                except (ValueError, IndexError):
                    sites = None
            self._sites[frame.key] = sites
        return self._sites[frame.key]

    def _rich(self, frame: LayerFrame) -> bool:
        return int(self._counts.get(frame.key, 0)) >= self._cfg.min_compared

    def _match(self, kind: str, a: LayerFrame, b: LayerFrame) -> np.ndarray | None:
        key = (kind, a.key, b.key)
        if key not in self._matched:
            found = None
            if self._rich(a) and self._rich(b):
                from .memory_map import SEAM_BITS, match_frames

                first, second = self._frame_sites(a), self._frame_sites(b)
                if first is not None and second is not None:
                    seam = match_frames(first, second, kind, self._pitch, self._cfg)
                    if seam.status == SEAM_BITS:
                        found = np.asarray(seam.offset, np.float64)
                        self.trusted += 1
            self._matched[key] = found
        return self._matched[key]

    def _pairs(self, kind: str, row: int | None) -> list[tuple[LayerFrame, LayerFrame]]:
        step = (1, 0) if kind == "right" else (0, 1)
        pairs = []
        for frame in self._frames:
            if row is not None and frame.row != row:
                continue
            other = self._by_place.get((frame.col + step[0], frame.row + step[1]))
            if other is not None and self._rich(frame) and self._rich(other):
                pairs.append((frame, other))
        pairs.sort(key=lambda pair: -min(self._counts.get(pair[0].key, 0), self._counts.get(pair[1].key, 0)))
        if row is None and len(pairs) > self.RUN_TRIES:
            # Spread over the run rather than the richest corner of it.
            picks = np.linspace(0, len(pairs) - 1, 3 * self.RUN_TRIES).astype(int)
            pairs = [pairs[int(index)] for index in dict.fromkeys(picks)]
        return pairs

    def _usual_offset(self, kind: str, row: int | None) -> np.ndarray | None:
        key = (kind, row)
        if key not in self._usual:
            tries = self.ROW_TRIES if row is not None else self.RUN_TRIES
            found: list[np.ndarray] = []
            for attempt, (a, b) in enumerate(self._pairs(kind, row)):
                if len(found) >= 3 or attempt >= 3 * tries:
                    break
                offset = self._match(kind, a, b)
                if offset is not None:
                    found.append(offset)
            self._usual[key] = np.median(np.stack(found), axis=0) if found else None
        return self._usual[key]

    def offset(self, kind: str, a: LayerFrame, b: LayerFrame) -> np.ndarray | None:
        own = self._match(kind, a, b)
        if own is not None:
            return own
        usual = self._usual_offset(kind, a.row)
        return usual if usual is not None else self._usual_offset(kind, None)


def _translation(seams: "_Seams", by_place: dict, frame: LayerFrame, dc: int, dr: int) -> np.ndarray | None:
    """Shift from the frame ``dc`` columns and ``dr`` rows away into ``frame``'s pixels (its pixel =
    the other's pixel + shift); a diagonal frame through the frame beside or below, whichever is there."""

    def step(current: LayerFrame, step_col: int, step_row: int):
        nxt = by_place.get((current.col + step_col, current.row + step_row))
        if nxt is None:
            return None, None
        if step_col == 1:
            offset = seams.offset("right", current, nxt)
            sign = 1.0
        elif step_col == -1:
            offset = seams.offset("right", nxt, current)
            sign = -1.0
        elif step_row == 1:
            offset = seams.offset("down", current, nxt)
            sign = 1.0
        else:
            offset = seams.offset("down", nxt, current)
            sign = -1.0
        if offset is None:
            return None, None
        return nxt, sign * np.asarray(offset, np.float64)

    paths = [((dc, 0), (0, dr)), ((0, dr), (dc, 0))] if dc and dr else [((dc, dr),)]
    for path in paths:
        current, total = frame, np.zeros(2)
        for step_col, step_row in path:
            current, shift = step(current, step_col, step_row)
            if current is None:
                break
            total = total + shift
        else:
            return total
    return None


def _read_mask(path: str) -> np.ndarray | None:
    from .memory_map import _read_mask as read

    return read(path)


def _strip_marks(result):
    """The result without sparse marks of an earlier pass (the step runs again on stored results)."""

    cells = list(getattr(result, "per_cell_results", ()) or ())
    changed = False
    for index, cell in enumerate(cells):
        reasons = tuple(str(reason) for reason in (cell.reasons or ()))
        if SPARSE_REASON not in reasons:
            continue
        changed = True
        kept = tuple(reason for reason in reasons if reason != SPARSE_REASON)
        features = {key: value for key, value in (cell.feature_snapshot or ()) if key not in SPARSE_FEATURES}
        if kept:
            cells[index] = replace(cell, reasons=kept, feature_snapshot=tuple(sorted(features.items())))
        else:
            cells[index] = replace(cell, reasons=(), status="normal", score=0.0, feature_snapshot=tuple(sorted(features.items())))
    if not changed:
        return result
    from .grid_narrowing import with_cells

    return with_cells(result, cells)


def _marked(result, lone: Sequence[int], marks: dict[str, float]):
    """``result`` with the cells at ``lone`` marked sparse; ``marks``: the window and pitch features."""

    from .grid_narrowing import with_cells

    cells = list(result.per_cell_results)
    for index in lone:
        cell = cells[index]
        reasons = tuple(str(reason) for reason in (cell.reasons or ()))
        features = dict(cell.feature_snapshot or ())
        features[f"reason:{SPARSE_REASON}"] = 1.0
        features.update(marks)
        snapshot = tuple(sorted(features.items()))
        if not reasons:
            cells[index] = replace(cell, reasons=(SPARSE_REASON,), status="suspicious", score=SPARSE_SCORE, feature_snapshot=snapshot)
        else:
            cells[index] = replace(cell, reasons=reasons + (SPARSE_REASON,), feature_snapshot=snapshot)
    return with_cells(result, cells)


def copy_sparse_marks(result, stored):
    """``result`` of a frame analysed again (frame window) with the sparse marks of the run's
    ``stored`` result of the same frame: the same cells by their boxes. One frame alone cannot tell
    them, its neighbour frames take part."""

    from .grid_anomaly import GridFrameAnalysisResult

    if not isinstance(result, GridFrameAnalysisResult) or not isinstance(stored, GridFrameAnalysisResult):
        return result
    marks = {
        tuple(int(value) for value in tuple(cell.bbox)[:4]): {
            key: value for key, value in (cell.feature_snapshot or ()) if key in SPARSE_FEATURES and not key.startswith("reason:")
        }
        for cell in stored.per_cell_results or ()
        if SPARSE_REASON in (cell.reasons or ())
    }
    clean = _strip_marks(result)
    if not marks:
        return clean
    lone = [
        index
        for index, cell in enumerate(clean.per_cell_results or ())
        if is_cell(cell) and tuple(int(value) for value in tuple(cell.bbox)[:4]) in marks
    ]
    if not lone:
        return clean
    return _marked(clean, lone, next(iter(marks.values())))


def run_pitch(results: Mapping[str, object]) -> tuple[float, float] | None:
    """Lattice pitch (px) of a run: what its sparse marks were found with, else from its cells
    (a project keeps only its defects: then the marks are the only source)."""

    rich: list[np.ndarray] = []
    # A few hundred frames tell the pitch; a run of ten thousand is not walked through for a hint.
    for result in list(results.values())[:400]:
        for cell in getattr(result, "per_cell_results", ()) or ():
            if SPARSE_REASON in (cell.reasons or ()):
                features = dict(cell.feature_snapshot or ())
                if SPARSE_PITCH_X_FEATURE in features and SPARSE_PITCH_Y_FEATURE in features:
                    return float(features[SPARSE_PITCH_X_FEATURE]), float(features[SPARSE_PITCH_Y_FEATURE])
        _indexes, boxes = _boxes(result)
        if len(boxes) >= 8:
            rich.append(boxes)
    rich.sort(key=len, reverse=True)
    return pitch_from_boxes(rich[:24])


def mark_sparse_cells(
    results: Mapping[str, object],
    frames: Sequence[LayerFrame],
    *,
    reach_sites: float | tuple[float, float] = DEFAULT_SPARSE_REACH_SITES,
    min_free_sides: int = DEFAULT_SPARSE_FREE_SIDES,
    read_mask: Callable[[str], np.ndarray | None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Results of the layer's frames that changed, and what the step found (pitch, seams, counts).

    ``reach_sites``: how far the area reaches along the row (X) and the column (Y), in lattice
    pitches; one number for both axes or an (X, Y) pair. ``min_free_sides``: how many of the four
    sides must be free for a cell to be sparse.

    ``results`` are GridFrameAnalysisResult by frame key; frames without a result are skipped.
    """

    from .grid_anomaly import GridFrameAnalysisResult

    reader = read_mask or _read_mask
    stripped: dict[str, object] = {}
    cells: dict[str, tuple[list[int], np.ndarray]] = {}
    for frame in frames:
        result = results.get(frame.key)
        if not isinstance(result, GridFrameAnalysisResult):
            continue
        clean = _strip_marks(result)
        stripped[frame.key] = clean
        cells[frame.key] = _boxes(clean)
    reach_x_sites, reach_y_sites = normalize_reach(reach_sites)
    min_free = normalize_free_sides(min_free_sides)
    info: dict[str, object] = {
        "frames": len(cells),
        "sparse_cells": 0,
        "reach_sites": (reach_x_sites, reach_y_sites),
        "min_free_sides": min_free,
    }
    if not cells:
        return {}, info
    rich = sorted(cells.values(), key=lambda item: -len(item[0]))[:24]
    pitch = pitch_from_boxes([boxes for _indexes, boxes in rich])
    all_boxes = np.concatenate([boxes for _indexes, boxes in cells.values()]) if cells else np.zeros((0, 4))
    if len(all_boxes):
        cell_w = float(np.median(all_boxes[:, 2] - all_boxes[:, 0]))
        cell_h = float(np.median(all_boxes[:, 3] - all_boxes[:, 1]))
    else:
        cell_w = cell_h = 1.0
    if pitch is None:
        # Too few cells in the whole run for a lattice: cell size plus the usual gap.
        pitch = (1.7 * cell_w, 1.25 * cell_h)
        info["pitch_from"] = "cell_size"
    else:
        info["pitch_from"] = "cells"
    info["pitch"] = (round(pitch[0], 2), round(pitch[1], 2))
    # Gaps between boxes: N pitches less one cell, with a quarter pitch to spare.
    reach_x = max(0.0, reach_x_sites * pitch[0] - cell_w + 0.25 * pitch[0])
    reach_y = max(0.0, reach_y_sites * pitch[1] - cell_h + 0.25 * pitch[1])
    marks = {
        SPARSE_REACH_X_FEATURE: float(reach_x_sites),
        SPARSE_REACH_Y_FEATURE: float(reach_y_sites),
        SPARSE_PITCH_X_FEATURE: float(pitch[0]),
        SPARSE_PITCH_Y_FEATURE: float(pitch[1]),
        SPARSE_MIN_FREE_FEATURE: float(min_free),
    }

    by_place = {(frame.col, frame.row): frame for frame in frames if frame.key in cells}
    counts = {key: len(indexes) for key, (indexes, _boxes) in cells.items()}
    seams = _Seams([frame for frame in frames if frame.key in cells], counts, reader, pitch)
    changed: dict[str, object] = {}
    extended = 0
    for frame in frames:
        if cancelled is not None and cancelled():
            break
        if frame.key not in cells:
            continue
        indexes, boxes = cells[frame.key]
        result = stripped[frame.key]
        free = free_sides(boxes, len(boxes), reach_x, reach_y, enough=min_free)
        lone = free >= min_free
        if np.any(lone):
            width, height = float(result.image_width), float(result.image_height)
            edge = boxes[lone]
            # Where the area of a sparse cell goes past the frame edge: the neighbour frames there.
            horizontal = [step for step, crossed in ((-1, np.any(edge[:, 0] <= reach_x)), (1, np.any(width - edge[:, 2] <= reach_x))) if crossed]
            vertical = [step for step, crossed in ((-1, np.any(edge[:, 1] <= reach_y)), (1, np.any(height - edge[:, 3] <= reach_y))) if crossed]
            directions = [(dc, 0) for dc in horizontal] + [(0, dr) for dr in vertical]
            directions += [(dc, dr) for dc in horizontal for dr in vertical]
            extra: list[np.ndarray] = []
            for dc, dr in directions:
                other = by_place.get((frame.col + dc, frame.row + dr))
                if other is None or not len(cells[other.key][1]):
                    continue
                shift = _translation(seams, by_place, frame, dc, dr)
                if shift is None:
                    continue
                moved = cells[other.key][1] + np.array([shift[0], shift[1], shift[0], shift[1]])
                centre_x = (moved[:, 0] + moved[:, 2]) / 2.0
                centre_y = (moved[:, 1] + moved[:, 3]) / 2.0
                # What lies inside this frame this frame sees itself.
                outside = (centre_x < 0) | (centre_x >= width) | (centre_y < 0) | (centre_y >= height)
                if np.any(outside):
                    extra.append(moved[outside])
            if extra:
                extended += 1
                free = free_sides(np.concatenate([boxes, *extra]), len(boxes), reach_x, reach_y, enough=min_free)
                lone = free >= min_free
        marked_rows = np.flatnonzero(lone)
        if len(marked_rows):
            info["sparse_cells"] = int(info["sparse_cells"]) + len(marked_rows)
            updated = result
            for sides in sorted({int(free[row]) for row in marked_rows}):
                rows = [indexes[row] for row in marked_rows if int(free[row]) == sides]
                updated = _marked(updated, rows, {**marks, SPARSE_FREE_FEATURE: float(sides)})
        else:
            updated = result
        original = results.get(frame.key)
        # A stored result marked by an earlier pass comes out equal, not the same object.
        if updated is not original and updated != original:
            changed[frame.key] = updated
    info["frames_extended"] = extended
    info["trusted_seams"] = seams.trusted
    return changed, info
