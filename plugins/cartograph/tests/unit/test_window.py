from __future__ import annotations

from cartograph.domain.coordinates import GridCoordinate
from cartograph.domain.topology import four_neighborhood_edges, select_window, unit_square_cycles

from .helpers import grid_from_tiles, tile_at


def test_full_window_has_twelve_primary_edges() -> None:
    tiles = [tile_at(r, c, 32, 32) for r in range(3) for c in range(3)]
    grid = grid_from_tiles(tiles)
    window = select_window(grid, GridCoordinate(1, 1))
    assert len(window.tiles) == 9
    edges = four_neighborhood_edges(set(window.tiles))
    assert len(edges) == 12
    assert len(unit_square_cycles(set(window.tiles))) == 4


def test_window_slides_and_clips_at_origin() -> None:
    tiles = [tile_at(r, c, 32, 32) for r in range(4) for c in range(4)]
    grid = grid_from_tiles(tiles)
    corner = select_window(grid, GridCoordinate(0, 0))
    assert sorted(corner.tiles) == [GridCoordinate(0, 0), GridCoordinate(0, 1), GridCoordinate(1, 0), GridCoordinate(1, 1)]
    interior = select_window(grid, GridCoordinate(2, 2))
    assert GridCoordinate(1, 1) in interior.tiles
    assert GridCoordinate(3, 3) in interior.tiles
