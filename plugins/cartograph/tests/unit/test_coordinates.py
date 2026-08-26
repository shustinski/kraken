from __future__ import annotations

import pytest

from cartograph.domain.coordinates import (
    GridCoordinate,
    NominalPlacementMode,
    PixelSize,
    StageCoordinate,
    regular_grid_position,
    stage_to_pixels,
)
from cartograph.application.nominal import PlacementSettings, compute_nominal_placement
from cartograph.domain.errors import PlacementError
from cartograph.infrastructure.kraken.coordinates import from_kraken_xy, to_kraken_xy

from .helpers import grid_from_tiles, tile_at


def test_regular_grid_matches_specified_formula() -> None:
    coord = GridCoordinate(2, 3)
    position = regular_grid_position(coord, tile_width=100, tile_height=80, overlap_x=0.1, overlap_y=0.2)
    assert position.x == pytest.approx(3 * 100 * 0.9)
    assert position.y == pytest.approx(2 * 80 * 0.8)


def test_kraken_xy_round_trip() -> None:
    coord = GridCoordinate(4, 7)
    x, y = to_kraken_xy(coord)
    assert (x, y) == (8, 5)
    assert from_kraken_xy(x, y) == coord


def test_sem_stage_uses_microscope_coordinates() -> None:
    tiles = [
        tile_at(0, 0, 50, 50, stage=StageCoordinate(10.0, 20.0)),
        tile_at(0, 1, 50, 50, stage=StageCoordinate(55.0, 20.0)),
    ]
    grid = grid_from_tiles(tiles, overlap_x=0.1, overlap_y=0.1)
    placed = compute_nominal_placement(grid, PlacementSettings(mode=NominalPlacementMode.SEM_STAGE, pixel_size=PixelSize(1.0, 1.0)))
    assert placed[GridCoordinate(0, 0)].x == pytest.approx(0.0)
    assert placed[GridCoordinate(0, 1)].x == pytest.approx(45.0)


def test_sem_stage_rejects_missing_coordinates() -> None:
    tiles = [tile_at(0, 0, 50, 50, stage=StageCoordinate(0.0, 0.0)), tile_at(0, 1, 50, 50)]
    grid = grid_from_tiles(tiles)
    with pytest.raises(PlacementError, match="stage coordinates"):
        compute_nominal_placement(grid, PlacementSettings(mode=NominalPlacementMode.SEM_STAGE, pixel_size=PixelSize(1.0, 1.0)))


def test_hybrid_falls_back_when_stage_missing() -> None:
    tiles = [
        tile_at(0, 0, 100, 100, stage=StageCoordinate(0.0, 0.0)),
        tile_at(0, 1, 100, 100),
    ]
    grid = grid_from_tiles(tiles, overlap_x=0.1, overlap_y=0.1)
    placed = compute_nominal_placement(grid, PlacementSettings(mode=NominalPlacementMode.HYBRID, pixel_size=PixelSize(1.0, 1.0)))
    assert placed[GridCoordinate(0, 1)].x == pytest.approx(90.0)


def test_hybrid_rejects_outlier_stage_displacement() -> None:
    tiles = [
        tile_at(0, 0, 100, 100, stage=StageCoordinate(0.0, 0.0)),
        tile_at(0, 1, 100, 100, stage=StageCoordinate(900.0, 0.0)),
    ]
    grid = grid_from_tiles(tiles, overlap_x=0.1, overlap_y=0.1)
    placed = compute_nominal_placement(
        grid,
        PlacementSettings(mode=NominalPlacementMode.HYBRID, pixel_size=PixelSize(1.0, 1.0), stage_outlier_factor=2.0),
    )
    assert placed[GridCoordinate(0, 1)].x == pytest.approx(90.0)


def test_stage_to_pixels_uses_anisotropic_pixel_size() -> None:
    pixels = stage_to_pixels(StageCoordinate(10.0, 20.0), PixelSize(2.0, 4.0))
    assert pixels.x == pytest.approx(5.0)
    assert pixels.y == pytest.approx(5.0)
