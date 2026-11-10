from types import SimpleNamespace

from karakal.ui.details_dialog import ExtendFrameDetailsDialog
from karakal.ui.ui_constants import (
    GRID_INSPECTION_ERROR_TYPE_COLORS,
    GRID_INSPECTION_ERROR_TYPE_OPTIONS,
    grid_inspection_error_type_color,
)


def test_grid_error_palette_covers_every_selectable_error_type() -> None:
    error_types = {error_type for _label_key, error_type in GRID_INSPECTION_ERROR_TYPE_OPTIONS}

    assert set(GRID_INSPECTION_ERROR_TYPE_COLORS) == error_types
    assert len(set(GRID_INSPECTION_ERROR_TYPE_COLORS.values())) == len(error_types)
    assert all(grid_inspection_error_type_color(error_type).isValid() for error_type in error_types)


def test_grid_detail_overlay_uses_the_selectable_error_palette() -> None:
    for _label_key, error_type in GRID_INSPECTION_ERROR_TYPE_OPTIONS:
        cell = SimpleNamespace(reasons=(error_type,), status="broken")

        actual = ExtendFrameDetailsDialog._grid_cell_color(None, cell)

        assert actual.name() == grid_inspection_error_type_color(error_type).name()


def test_edge_clip_is_a_flag_not_a_selectable_defect_type() -> None:
    # Decision Н5: a cell cut by the frame edge is not a network error.
    assert "edge_clipped_cell" not in {error_type for _label_key, error_type in GRID_INSPECTION_ERROR_TYPE_OPTIONS}
    cell = SimpleNamespace(reasons=("broken_geometry", "edge_clipped_cell"), status="broken")
    actual = ExtendFrameDetailsDialog._grid_cell_color(None, cell)
    assert actual.name() == grid_inspection_error_type_color("broken_geometry").name()


def test_a_lone_broken_cell_is_shown_as_sparse() -> None:
    from karakal.core.exports import _grid_defect_reason_rgb
    from karakal.ui.ui_constants import grid_display_error_type

    reasons = ("broken_geometry", "sparse_cell")
    assert grid_display_error_type(reasons) == "sparse_cell"
    # The sparse type switched off: the cell is its other finding.
    assert grid_display_error_type(reasons, {"broken_geometry"}) == "broken_geometry"
    cell = SimpleNamespace(reasons=reasons, status="broken")
    assert ExtendFrameDetailsDialog._grid_cell_color(None, cell).name() == grid_inspection_error_type_color("sparse_cell").name()
    lime = grid_inspection_error_type_color("sparse_cell")
    expected = (lime.red(), lime.green(), lime.blue())
    assert _grid_defect_reason_rgb(reasons) == expected
    assert _grid_defect_reason_rgb(reasons, selected_types={"broken_geometry", "sparse_cell"}) == expected
    blue = grid_inspection_error_type_color("broken_geometry")
    assert _grid_defect_reason_rgb(reasons, selected_types={"broken_geometry"}) == (blue.red(), blue.green(), blue.blue())


def test_a_sparse_cell_is_a_thick_frame_over_the_fill_of_its_other_finding() -> None:
    from karakal.core.grid_anomaly import GridCellAnalysisResult, GridFrameAnalysisResult
    from karakal.core.grid_error_export import render_grid_error_image
    from karakal.ui.ui_constants import sparse_frame_width

    def cell(index, bbox, reasons):
        return GridCellAnalysisResult(
            row=index, col=0, bbox=bbox, centroid=(0.0, 0.0), contour_id=index + 1, status="broken", score=0.8, reasons=reasons
        )

    both = (10, 10, 22, 48)
    alone = (60, 10, 22, 48)
    result = GridFrameAnalysisResult(
        frame_id="f", frame_path="", image_width=100, image_height=70, grid_rows=0, grid_cols=0,
        total_expected_cells=2, detected_cells=2, normal_cells=0, suspicious_cells=0, broken_cells=2,
        missing_cells=0, artifact_cells=0, damage_score=0.5, severity_level="HIGH",
        per_cell_results=(cell(0, both, ("broken_geometry", "sparse_cell")), cell(1, alone, ("sparse_cell",))),
    )

    def bgr(error_type):
        color = grid_inspection_error_type_color(error_type)
        return [color.blue(), color.green(), color.red()]

    image = render_grid_error_image(result, ("broken_geometry", "sparse_cell"))
    side = sparse_frame_width(22, 48)
    assert side >= 3
    x, y, w, h = both
    # The pixels of both colours in one cell: blue inside, the lime frame around.
    assert image[y + h // 2, x + w // 2].tolist() == bgr("broken_geometry")
    assert image[y + h // 2, x].tolist() == bgr("sparse_cell") and image[y, x + w // 2].tolist() == bgr("sparse_cell")
    assert image[y + h // 2, x + side].tolist() == bgr("broken_geometry")
    # A sparse cell alone: only the frame.
    x, y, w, h = alone
    assert image[y + h // 2, x + w // 2].tolist() == [0, 0, 0]
    assert image[y + h // 2, x + w - 1].tolist() == bgr("sparse_cell")
    # The sparse type left out of the export: the cell is its geometry fill only.
    plain = render_grid_error_image(result, ("broken_geometry",))
    x, y, w, h = both
    assert plain[y + h // 2, x].tolist() == bgr("broken_geometry")
    assert not plain[alone[1] : alone[1] + alone[3], alone[0] : alone[0] + alone[2]].any()
