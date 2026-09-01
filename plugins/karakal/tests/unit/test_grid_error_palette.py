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
