"""Wizard export layout and filter source."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

from karakal.core.grid_anomaly import GridCellAnalysisResult, GridFrameAnalysisResult
from karakal.core.grid_error_export import (
    GridErrorExportFrame,
    GridExportLayerSpec,
    color_mode_folder_name,
    export_grid_error_wizard_run,
    render_grid_error_image,
    safe_export_folder_name,
    unique_run_dir,
)
from karakal.ui.grid_export_wizard import ExportLayerOffer, GridErrorExportWizard
from karakal.ui.ui_constants import GRID_INSPECTION_ERROR_TYPE_COLORS


def _cell(reasons: tuple[str, ...], bbox: tuple[int, int, int, int]) -> GridCellAnalysisResult:
    return GridCellAnalysisResult(0, 0, bbox, (bbox[0] + 1.0, bbox[1] + 1.0), 1, "broken", 0.9, reasons)


def _frame(cells: tuple[GridCellAnalysisResult, ...], size: int = 32) -> GridFrameAnalysisResult:
    return GridFrameAnalysisResult(
        frame_id="frame",
        frame_path="TEST_DIRECT_0009.jpg",
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


def _t(key: str, **kwargs) -> str:
    text = key
    for name, value in kwargs.items():
        text = text.replace("{" + name + "}", str(value))
    return text


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    return app


def test_safe_folder_keeps_cyrillic_and_spaces() -> None:
    assert safe_export_folder_name("Test по единицам") == "Test по единицам"
    assert ":" not in safe_export_folder_name('a:b/c')


def test_unique_run_dir_does_not_overwrite(tmp_path: Path) -> None:
    first = unique_run_dir(tmp_path)
    first.mkdir()
    second = unique_run_dir(tmp_path)
    assert first != second
    assert second.name.endswith("_2")


def test_single_white_is_grayscale_mask() -> None:
    image = render_grid_error_image(
        _frame((_cell(("broken_geometry",), (2, 2, 4, 4)),)),
        ("broken_geometry",),
        color_mode="single",
        single_color=(255, 255, 255),
    )
    assert image.ndim == 2
    assert image[3, 3] == 255
    assert image[0, 0] == 0


def test_filter_colors_use_palette_only() -> None:
    image = render_grid_error_image(
        _frame(
            (
                _cell(("broken_geometry",), (2, 2, 4, 4)),
                _cell(("small_artifact",), (10, 10, 4, 4)),
            )
        ),
        ("broken_geometry",),
        color_mode="filter",
    )
    value = GRID_INSPECTION_ERROR_TYPE_COLORS["broken_geometry"].lstrip("#")
    bgr = (int(value[4:6], 16), int(value[2:4], 16), int(value[0:2], 16))
    assert tuple(int(channel) for channel in image[3, 3]) == bgr
    assert np.all(image[11, 11] == 0)


def test_wizard_run_writes_two_model_folders(tmp_path: Path) -> None:
    cyrillic = tmp_path / "экспорт слои"
    frame = _frame((_cell(("broken_geometry",), (4, 4, 6, 6)),), size=40)
    empty = _frame((), size=40)
    report = export_grid_error_wizard_run(
        (
            GridErrorExportFrame("TEST_DIRECT_0009.jpg", "1", "binary", frame),
            GridErrorExportFrame("TEST_DIRECT_0009.jpg", "0", "binary", empty),
            GridErrorExportFrame("TEST_DIRECT_0012.jpg", "1", "binary", frame),
            GridErrorExportFrame("TEST_DIRECT_0012.jpg", "0", "binary", frame),
        ),
        cyrillic,
        ("broken_geometry", "small_artifact"),
        (
            GridExportLayerSpec("1", "Test по единицам", "Test по единицам"),
            GridExportLayerSpec("0", "Test по нулям", "Test по нулям"),
        ),
        image_format="png",
        color_mode="filter",
        skip_empty=True,
    )
    assert report.run_dir is not None
    mode = color_mode_folder_name("png", "filter")
    ones = report.run_dir / mode / "Test по единицам"
    zeros = report.run_dir / mode / "Test по нулям"
    assert (ones / "TEST_DIRECT_0009.jpg").with_suffix(".png").exists() or list(ones.glob("TEST_DIRECT_0009.*"))
    written_names = {path.name for path in report.written}
    assert any(name.startswith("TEST_DIRECT_0009") for name in written_names)
    assert any(name.startswith("TEST_DIRECT_0012") for name in written_names)
    # Empty zeros frame 9 skipped; frame 12 written for both.
    assert not list((zeros).glob("TEST_DIRECT_0009.*"))
    assert list((zeros).glob("TEST_DIRECT_0012.*"))
    assert (report.run_dir / "legend.json").is_file()
    assert (report.run_dir / "summary.csv").is_file()
    second = export_grid_error_wizard_run(
        (GridErrorExportFrame("TEST_DIRECT_0039.jpg", "1", "binary", frame),),
        cyrillic,
        ("broken_geometry",),
        (GridExportLayerSpec("1", "Test по единицам", "Test по единицам"),),
        image_format="png",
        color_mode="filter",
    )
    assert second.run_dir != report.run_dir


def test_wizard_blocks_empty_types_and_layers(qapp) -> None:
    empty_types = GridErrorExportWizard(
        _t,
        layers=(ExportLayerOffer("1", "ones", 3, 2),),
        selected_types=(),
        has_selection=False,
        output_dir=Path("."),
    )
    assert empty_types._layers_page.isComplete() is False
    assert empty_types.choices() is None

    wizard = GridErrorExportWizard(
        _t,
        layers=(ExportLayerOffer("1", "ones", 3, 2),),
        selected_types=("broken_geometry",),
        has_selection=False,
        output_dir=Path("."),
    )
    wizard._layers_page._set_all(False)
    assert wizard._layers_page.isComplete() is False
    wizard._layers_page._set_all(True)
    assert wizard._layers_page.isComplete() is True
    wizard.apply_saved_choices(
        {
            "image_format": "png",
            "color_mode": "filter",
            "single_color": [255, 0, 0],
            "layer_keys": ["1"],
            "frame_scope": "all",
            "skip_empty": True,
            "output_dir": str(Path("D:/tmp")),
        }
    )
    choices = wizard.choices()
    assert choices is not None
    assert choices.image_format == "png"
    assert choices.color_mode == "filter"
    assert choices.skip_empty is True
    assert choices.selected_types == ("broken_geometry",)
