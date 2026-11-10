"""Sparse cells: cells with no neighbour along their row and their column.

On a frame the network painted almost empty, a lone cell is suspicious: dirt or a reflection the
network took for a cell, or a stray piece of another structure. Such a cell may also be of a wrong
shape or carry other findings. A cell is sparse when no other cell lies within a few lattice
pitches along three axes:

- X: along its row (boxes that overlap it vertically), to the left and to the right;
- Y: along its column (boxes that overlap it horizontally), up and down;
- neighbour frames: near the frame edge the row and the column go on in the next frame of the
  snapshot. Frames overlap; where the next frame lies is found by the bits of the frames
  (memory_map.match_frames) on frames rich enough to compare, and the usual offset of the frame
  row (else of the whole run) serves the frames with too few cells for that.

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
# No other cell within this many lattice pitches along the row and the column.
DEFAULT_SPARSE_REACH_SITES = 4
SPARSE_REACH_RANGE = (1, 20)
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
# Feature of a marked cell: the reach (in pitches) it was found sparse at.
SPARSE_REACH_FEATURE = "sparse_reach_sites"
# Neighbour sides of a frame: kind of the seam and the step to the neighbour in (column, row).
_SIDES = (
    ("right", "right", (1, 0)),
    ("left", "right", (-1, 0)),
    ("down", "down", (0, 1)),
    ("up", "down", (0, -1)),
)


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


def _neighbours(first: np.ndarray, second: np.ndarray, reach_x: float, reach_y: float) -> np.ndarray:
    """Whether boxes lie in one row within reach_x of each other, or in one column within reach_y."""

    over_y = np.minimum(first[..., 3], second[..., 3]) - np.maximum(first[..., 1], second[..., 1]) > 0
    over_x = np.minimum(first[..., 2], second[..., 2]) - np.maximum(first[..., 0], second[..., 0]) > 0
    gap_x = np.maximum(second[..., 0] - first[..., 2], first[..., 0] - second[..., 2])
    gap_y = np.maximum(second[..., 1] - first[..., 3], first[..., 1] - second[..., 3])
    return (over_y & (gap_x <= reach_x)) | (over_x & (gap_y <= reach_y))


def lone_cells(boxes: np.ndarray, own: int, reach_x: float, reach_y: float) -> np.ndarray:
    """Which of the first ``own`` boxes have no other box within reach along their row or column.

    Boxes are (x0, y0, x1, y1); those after the first ``own`` are only neighbours (cells of the next
    frames). Reaches are gaps between boxes in pixels.
    """

    boxes = np.asarray(boxes, np.float64).reshape(-1, 4)
    own = int(own)
    count = len(boxes)
    if own <= 0:
        return np.zeros(0, bool)
    if count < 2:
        return np.ones(own, bool)
    from scipy.spatial import cKDTree

    centres = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2.0, (boxes[:, 1] + boxes[:, 3]) / 2.0], axis=1)
    k = min(9, count)
    _distance, nearest = cKDTree(centres).query(centres[:own], k=k)
    nearest = np.asarray(nearest).reshape(own, k)
    near = _neighbours(boxes[:own, None, :], boxes[nearest], reach_x, reach_y)
    near &= nearest != np.arange(own)[:, None]
    found = near.any(axis=1)
    # In a filled frame the nearest boxes settle almost every cell; the rest is checked against all.
    rest = np.flatnonzero(~found)
    for start in range(0, len(rest), 256):
        chunk = rest[start : start + 256]
        near = _neighbours(boxes[chunk, None, :], boxes[None, :, :], reach_x, reach_y)
        near[np.arange(len(chunk)), chunk] = False
        found[chunk] = near.any(axis=1)
    return ~found


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
        features = {
            key: value
            for key, value in (cell.feature_snapshot or ())
            if key not in (f"reason:{SPARSE_REASON}", SPARSE_REACH_FEATURE)
        }
        if kept:
            cells[index] = replace(cell, reasons=kept, feature_snapshot=tuple(sorted(features.items())))
        else:
            cells[index] = replace(cell, reasons=(), status="normal", score=0.0, feature_snapshot=tuple(sorted(features.items())))
    if not changed:
        return result
    from .grid_narrowing import with_cells

    return with_cells(result, cells)


def _marked(result, lone: Sequence[int], reach_sites: float):
    from .grid_narrowing import with_cells

    cells = list(result.per_cell_results)
    for index in lone:
        cell = cells[index]
        reasons = tuple(str(reason) for reason in (cell.reasons or ()))
        features = dict(cell.feature_snapshot or ())
        features[f"reason:{SPARSE_REASON}"] = 1.0
        features[SPARSE_REACH_FEATURE] = float(reach_sites)
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
    reach = {
        tuple(int(value) for value in tuple(cell.bbox)[:4]): float(dict(cell.feature_snapshot or ()).get(SPARSE_REACH_FEATURE, DEFAULT_SPARSE_REACH_SITES))
        for cell in stored.per_cell_results or ()
        if SPARSE_REASON in (cell.reasons or ())
    }
    clean = _strip_marks(result)
    if not reach:
        return clean
    lone = [
        index
        for index, cell in enumerate(clean.per_cell_results or ())
        if is_cell(cell) and tuple(int(value) for value in tuple(cell.bbox)[:4]) in reach
    ]
    if not lone:
        return clean
    return _marked(clean, lone, next(iter(reach.values())))


def mark_sparse_cells(
    results: Mapping[str, object],
    frames: Sequence[LayerFrame],
    *,
    reach_sites: float = DEFAULT_SPARSE_REACH_SITES,
    read_mask: Callable[[str], np.ndarray | None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Results of the layer's frames that changed, and what the step found (pitch, seams, counts).

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
    info: dict[str, object] = {"frames": len(cells), "sparse_cells": 0, "reach_sites": float(reach_sites)}
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
    reach = max(1.0, float(reach_sites))
    reach_x = reach * pitch[0] - cell_w + 0.25 * pitch[0]
    reach_y = reach * pitch[1] - cell_h + 0.25 * pitch[1]

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
        lone = lone_cells(boxes, len(boxes), reach_x, reach_y)
        if np.any(lone):
            width, height = float(result.image_width), float(result.image_height)
            edge = boxes[lone]
            open_sides = {
                "right": bool(np.any(width - edge[:, 2] <= reach_x)),
                "left": bool(np.any(edge[:, 0] <= reach_x)),
                "down": bool(np.any(height - edge[:, 3] <= reach_y)),
                "up": bool(np.any(edge[:, 1] <= reach_y)),
            }
            extra: list[np.ndarray] = []
            for side, kind, (step_col, step_row) in _SIDES:
                if not open_sides[side]:
                    continue
                other = by_place.get((frame.col + step_col, frame.row + step_row))
                if other is None:
                    continue
                if side in ("right", "down"):
                    offset = seams.offset(kind, frame, other)
                    sign = 1.0
                else:
                    offset = seams.offset(kind, other, frame)
                    sign = -1.0
                if offset is None:
                    continue
                theirs = cells[other.key][1]
                if not len(theirs):
                    continue
                shift = sign * np.array([offset[0], offset[1], offset[0], offset[1]])
                moved = theirs + shift
                centre_x = (moved[:, 0] + moved[:, 2]) / 2.0
                centre_y = (moved[:, 1] + moved[:, 3]) / 2.0
                # What lies inside this frame this frame sees itself.
                outside = (centre_x < 0) | (centre_x >= width) | (centre_y < 0) | (centre_y >= height)
                if np.any(outside):
                    extra.append(moved[outside])
            if extra:
                extended += 1
                lone = lone_cells(np.concatenate([boxes, *extra]), len(boxes), reach_x, reach_y)
        marked = [indexes[i] for i in np.flatnonzero(lone)]
        if marked:
            info["sparse_cells"] = int(info["sparse_cells"]) + len(marked)
            updated = _marked(result, marked, reach)
        else:
            updated = result
        original = results.get(frame.key)
        # A stored result marked by an earlier pass comes out equal, not the same object.
        if updated is not original and updated != original:
            changed[frame.key] = updated
    info["frames_extended"] = extended
    info["trusted_seams"] = seams.trusted
    return changed, info
