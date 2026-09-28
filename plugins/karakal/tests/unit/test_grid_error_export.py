"""Filled-rectangle JPEG export uses only the selected defect types."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from karakal.core.grid_anomaly import GridCellAnalysisResult, GridFrameAnalysisResult
from karakal.core.grid_error_export import (
    GridErrorExportFrame,
    export_grid_cell_defect_frames,
    render_grid_error_image,
)
from karakal.ui.ui_constants import GRID_INSPECTION_ERROR_TYPE_COLORS


def _cell(reasons: tuple[str, ...], bbox: tuple[int, int, int, int], status: str = "broken") -> GridCellAnalysisResult:
    return GridCellAnalysisResult(0, 0, bbox, (bbox[0] + 1.0, bbox[1] + 1.0), 1, status, 0.9, reasons)


def _frame(cells: tuple[GridCellAnalysisResult, ...], size: int = 32) -> GridFrameAnalysisResult:
    return GridFrameAnalysisResult(
        frame_id="frame",
        frame_path="TEST_DIRECT_0013.jpg",
        image_width=size,
        image_height=size,
        grid_rows=1,
        grid_cols=1,
        total_expected_cells=1,
        detected_cells=len(cells),
        normal_cells=0,
        suspicious_cells=0,
        broken_cells=len(cells),
        missing_cells=0,
        artifact_cells=0,
        damage_score=0.9,
        severity_level="high",
        grid_detected=True,
        per_cell_results=cells,
    )


def _hex_bgr(name: str) -> tuple[int, int, int]:
    value = GRID_INSPECTION_ERROR_TYPE_COLORS[name].lstrip("#")
    red, green, blue = int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)
    return blue, green, red


def test_geometry_only_color_ignores_other_reasons() -> None:
    image = render_grid_error_image(
        _frame(
            (
                _cell(("broken_geometry",), (2, 2, 4, 4)),
                _cell(("merged_contour", "broken_geometry"), (10, 10, 4, 4)),
                _cell(("filled_cell",), (20, 20, 4, 4)),
                _cell(("normal",), (0, 0, 2, 2), status="normal"),
            )
        ),
        ("broken_geometry",),
        color_mode="color",
    )
    geometry = np.array(_hex_bgr("broken_geometry"), dtype=np.uint8)
    assert np.all(image[3, 3] == geometry)
    assert np.all(image[11, 11] == geometry)
    assert np.all(image[21, 21] == 0)
    assert np.all(image[0, 0] == 0)
    unique = {tuple(int(channel) for channel in pixel) for pixel in image.reshape(-1, 3)}
    assert unique <= {(0, 0, 0), tuple(int(channel) for channel in geometry)}


def test_bw_export_matches_the_error_mask_and_skips_zones(tmp_path: Path) -> None:
    frame = _frame(
        (
            _cell(("filled_cell",), (4, 4, 6, 6)),
            _cell(("conductor_zone",), (16, 16, 6, 6), status="zone"),
            _cell(("normal",), (0, 0, 3, 3), status="normal"),
        ),
        size=40,
    )
    image = render_grid_error_image(frame, ("filled_cell", "small_artifact", "broken_geometry"), color_mode="bw")
    assert image.ndim == 2
    assert image.shape == (40, 40)
    assert image[6, 6] == 255
    assert image[18, 18] == 0
    assert image[1, 1] == 0
    binary = np.where(image >= 127, 255, 0).astype(np.uint8)
    assert np.array_equal(binary, image)

    cyrillic = tmp_path / "экспорт"
    report = export_grid_cell_defect_frames(
        (
            GridErrorExportFrame("TEST_DIRECT_0013.jpg", "1", "confidence", frame),
            GridErrorExportFrame("TEST_DIRECT_0014.jpg", "1", "confidence", _frame((), size=40)),
            GridErrorExportFrame("TEST_DIRECT_0015.jpg", "1", "confidence", None),
        ),
        cyrillic / "grid_errors_all_bw",
        ("filled_cell",),
        color_mode="bw",
        skip_empty=False,
        all_types=True,
        algorithm_version="grid_damage_v79_small_artifact_broken",
    )
    written = cyrillic / "grid_errors_all_bw" / "1" / "confidence" / "TEST_DIRECT_0013.jpg"
    empty = cyrillic / "grid_errors_all_bw" / "1" / "confidence" / "TEST_DIRECT_0014.jpg"
    assert written in report.written
    assert empty in report.written
    assert any("0015" in item for item in report.missing_results)
    loaded = cv2.imdecode(np.fromfile(written, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    assert loaded is not None
    assert loaded.shape == (40, 40)
    assert (cyrillic / "grid_errors_all_bw" / "legend.json").is_file()
    assert (cyrillic / "grid_errors_all_bw" / "summary.csv").is_file()
    summary = (cyrillic / "grid_errors_all_bw" / "summary.csv").read_text(encoding="utf-8")
    assert "suspicious" not in summary

    skipped = export_grid_cell_defect_frames(
        (GridErrorExportFrame("TEST_DIRECT_0014.jpg", "1", "binary", _frame((), size=40)),),
        cyrillic / "grid_errors_all_bw_skip",
        ("filled_cell",),
        color_mode="bw",
        skip_empty=True,
    )
    assert skipped.written == ()
    assert not (cyrillic / "grid_errors_all_bw_skip" / "1" / "binary" / "TEST_DIRECT_0014.jpg").exists()
