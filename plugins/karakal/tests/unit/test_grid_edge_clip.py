"""Cells cut by the frame are edge clips every time, and they do not make a frame look damaged."""
from __future__ import annotations

import cv2
import numpy as np

from karakal.core.grid_anomaly import GridDamageAnalysisConfig, detect_grid_cell_anomalies

H, W = 700, 900
PX, PY, CW, CH = 50, 60, 24, 36
CONFIG = GridDamageAnalysisConfig(cell_representation="binary", scoring_mode="calibrated")


def _lattice(x0: int, *, sliver_rows: tuple[int, ...] = ()) -> np.ndarray:
    image = np.zeros((H, W), dtype=np.uint8)
    for row in range(10):
        y = 40 + row * PY
        for col in range(18):
            x = x0 + col * PX
            if x + CW <= 0 or x >= W:
                continue
            height = CH
            if col == 0 and row in sliver_rows:
                # Mostly outside the frame and drawn only in part.
                height = 14
            cv2.rectangle(image, (max(0, x), y), (min(W - 1, x + CW - 1), y + height - 1), 255, -1)
    return image


def _labels_at_left_edge(result) -> list[tuple[str, ...]]:
    return [tuple(cell.reasons) for cell in result.per_cell_results if cell.bbox[0] <= 1]


def test_clean_clipped_cells_are_edge_clips_not_normal() -> None:
    # First column cut by the left edge: 14 of 24 px visible, otherwise clean.
    result = detect_grid_cell_anomalies(_lattice(-10), config=CONFIG)
    labels = _labels_at_left_edge(result)
    assert labels and all(reasons == ("edge_clipped_cell",) for reasons in labels)


def test_partly_drawn_sliver_at_the_edge_is_an_edge_clip_not_debris() -> None:
    # 4 px of the cell are in the frame and the network drew 14 of 36 px of it.
    result = detect_grid_cell_anomalies(_lattice(-20, sliver_rows=(3, 6)), config=CONFIG)
    labels = _labels_at_left_edge(result)
    assert labels and all(reasons == ("edge_clipped_cell",) for reasons in labels)


def test_edge_clips_do_not_raise_the_frame_damage() -> None:
    clipped = detect_grid_cell_anomalies(_lattice(-10), config=CONFIG)
    assert clipped.severity_level == "OK"
    assert clipped.damage_score == 0.0
