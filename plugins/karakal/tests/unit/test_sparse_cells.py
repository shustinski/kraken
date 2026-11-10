"""Sparse cells: no other cell along the row and the column, in the frame and in its neighbours."""

from __future__ import annotations

import cv2
import numpy as np

from karakal.core.grid_anomaly import GridCellAnalysisResult, GridFrameAnalysisResult, _status_for_reasons
from karakal.core.grid_narrowing import GridNarrowing, narrow_grid_frame_result
from karakal.core.sparse_cells import SPARSE_REASON, layer_frames, lone_cells, mark_sparse_cells

PX, PY = 37.75, 62.9
CELL_W, CELL_H = 22, 26
FRAME_W, FRAME_H = 760, 700
ORIGIN = (75.0, 18.0)


def _box(x: float, y: float, w: float = CELL_W, h: float = CELL_H) -> tuple[float, float, float, float]:
    return (x, y, x + w, y + h)


def test_row_or_column_neighbour_within_reach_is_enough():
    boxes = np.array([_box(100, 100), _box(100 + 3 * PX, 100), _box(400, 400), _box(400 + PX, 400 + PY)])
    reach_x, reach_y = 4 * PX - CELL_W, 4 * PY - CELL_H
    lone = lone_cells(boxes, 4, reach_x, reach_y)
    # Two in one row three pitches apart; the last two are diagonal neighbours only.
    assert lone.tolist() == [False, False, True, True]


def test_neighbour_boxes_count_but_are_not_judged():
    boxes = np.array([_box(100, 100), _box(100, 100 + 2 * PY)])
    lone = lone_cells(boxes, 1, 4 * PX - CELL_W, 4 * PY - CELL_H)
    assert lone.tolist() == [False]
    assert lone_cells(boxes[:1], 1, 4 * PX - CELL_W, 4 * PY - CELL_H).tolist() == [True]


class _Plane:
    """Cells on a lattice; frames overlap by 4 columns and 3 rows, odd frame rows drift by a site."""

    def __init__(self, rows: int = 20, cols: int = 38, seed: int = 3) -> None:
        rng = np.random.default_rng(seed)
        self.bits = rng.random((rows, cols)) < 0.67

    def clear(self, rows: range, cols: range, keep: set[tuple[int, int]]) -> None:
        for r in rows:
            for c in cols:
                self.bits[r, c] = (r, c) in keep

    @staticmethod
    def centre(r: int, c: int) -> tuple[float, float]:
        return ORIGIN[0] + c * PX, ORIGIN[1] + r * PY

    def frames(self, per_row: int = 2, frame_rows: int = 2) -> list[tuple[float, float]]:
        step_x, step_y = FRAME_W - 4 * PX, FRAME_H - 3 * PY
        return [
            (fc * step_x + (PX if fr % 2 else 0.0) - 10.0, fr * step_y - 5.0)
            for fr in range(frame_rows)
            for fc in range(per_row)
        ]

    def render(self, fx: float, fy: float) -> np.ndarray:
        image = np.zeros((FRAME_H, FRAME_W), np.uint8)
        for r, c in zip(*np.nonzero(self.bits)):
            x, y = self.centre(int(r), int(c))
            x0, y0 = int(round(x - fx - CELL_W / 2)), int(round(y - fy - CELL_H / 2))
            image[max(0, y0) : max(0, y0 + CELL_H), max(0, x0) : max(0, x0 + CELL_W)] = 255
        return image


def _result(key: str, mask: np.ndarray) -> GridFrameAnalysisResult:
    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cells = tuple(
        GridCellAnalysisResult(
            row=index,
            col=0,
            bbox=tuple(int(value) for value in stats[label, :4]),
            centroid=(float(centroids[label][0]), float(centroids[label][1])),
            contour_id=index + 1,
            status="normal",
            score=0.0,
        )
        for index, label in enumerate(range(1, count))
    )
    return GridFrameAnalysisResult(
        frame_id=key,
        frame_path="",
        image_width=FRAME_W,
        image_height=FRAME_H,
        grid_rows=0,
        grid_cols=0,
        total_expected_cells=len(cells),
        detected_cells=len(cells),
        normal_cells=len(cells),
        suspicious_cells=0,
        broken_cells=0,
        missing_cells=0,
        artifact_cells=0,
        damage_score=0.0,
        severity_level="OK",
        grid_detected=True,
        per_cell_results=cells,
    )


def _sparse_centres(result: GridFrameAnalysisResult, offset: tuple[float, float]) -> set[tuple[int, int]]:
    """Lattice sites (row, col) of the marked cells of a frame at ``offset`` in the plane."""

    sites = set()
    for cell in result.per_cell_results:
        if SPARSE_REASON in cell.reasons:
            x = cell.bbox[0] + cell.bbox[2] / 2 + offset[0]
            y = cell.bbox[1] + cell.bbox[3] / 2 + offset[1]
            sites.add((int(round((y - ORIGIN[1]) / PY)), int(round((x - ORIGIN[0]) / PX))))
    return sites


def _run(tmp_path, plane: _Plane, per_row: int = 2):
    positions = plane.frames()
    results, keys_paths = {}, []
    for index, (fx, fy) in enumerate(positions):
        key = f"F{index}"
        mask = plane.render(fx, fy)
        path = tmp_path / f"{key}.png"
        cv2.imwrite(str(path), mask)
        results[key] = _result(key, mask)
        keys_paths.append((key, str(path)))
    changed, info = mark_sparse_cells(results, layer_frames(keys_paths, per_row), reach_sites=4)
    merged = {key: changed.get(key, value) for key, value in results.items()}
    return merged, positions, info


def _plane_with_lone_cells() -> _Plane:
    plane = _Plane()
    # Near the right edge of frame 0 (in the overlap): its neighbour three sites to the right is seen
    # only by frame 1. Far from everything in frame 2: alone for real.
    plane.clear(range(0, 10), range(11, 27), keep={(4, 17), (4, 20)})
    plane.clear(range(10, 19), range(0, 13), keep={(14, 5)})
    return plane


def test_lone_cell_is_marked_and_the_next_frame_gives_neighbours(tmp_path):
    results, positions, info = _run(tmp_path, _plane_with_lone_cells())
    assert info["trusted_seams"] >= 1
    marked = {key: _sparse_centres(result, positions[int(key[1:])]) for key, result in results.items()}
    # The lone cell in frame 2 (and frame 0 sees nothing of it: it lies below frame 0).
    assert (14, 5) in marked["F2"]
    # The cell at the seam has its neighbour in frame 1: no mark in either frame.
    assert (4, 17) not in marked["F0"]
    assert (4, 17) not in marked["F1"] and (4, 20) not in marked["F1"]
    cell = next(c for c in results["F2"].per_cell_results if SPARSE_REASON in c.reasons)
    assert cell.status == "suspicious"
    assert results["F2"].suspicious_cells >= 1


def test_without_frame_layout_the_seam_cell_stays_marked(tmp_path):
    results, positions, _info = _run(tmp_path, _plane_with_lone_cells(), per_row=0)
    assert (4, 17) in _sparse_centres(results["F0"], positions[0])


def test_mark_is_an_extra_reason_and_narrows_away():
    assert _status_for_reasons((SPARSE_REASON,)) == "suspicious"
    assert _status_for_reasons(("broken_geometry", SPARSE_REASON)) == "broken"
    mask = np.zeros((FRAME_H, FRAME_W), np.uint8)
    mask[300:326, 300:322] = 255
    result = _result("F", mask)
    broken = (result.per_cell_results[0].__class__)(**{
        **{name: getattr(result.per_cell_results[0], name) for name in ("row", "col", "bbox", "centroid", "contour_id")},
        "status": "broken",
        "score": 0.8,
        "reasons": ("broken_geometry",),
    })
    from dataclasses import replace

    result = replace(result, per_cell_results=(broken,))
    changed, _info = mark_sparse_cells({"F": result}, layer_frames([("F", "")], 0))
    cell = changed["F"].per_cell_results[0]
    assert cell.reasons == ("broken_geometry", SPARSE_REASON) and cell.status == "broken"
    hidden = narrow_grid_frame_result(changed["F"], GridNarrowing(debris_min_area_px=0, enabled_reason_types=frozenset({"broken_geometry"})))
    assert hidden.per_cell_results[0].reasons == ("broken_geometry",)
    # Run again on the stored result: the old mark goes first, nothing is doubled.
    again, _info = mark_sparse_cells({"F": changed["F"]}, layer_frames([("F", "")], 0))
    assert again == {}
