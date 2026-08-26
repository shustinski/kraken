"""Nominal placement for navigation. Stage coordinates are a prior, not absolute truth."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from cartograph.domain.coordinates import (
    GridCoordinate,
    NominalCoordinate,
    NominalPlacementMode,
    PixelSize,
    Translation2D,
    regular_grid_position,
    stage_to_pixels,
)
from cartograph.domain.errors import PlacementError
from cartograph.domain.tiles import Tile, TileGrid
from cartograph.domain.topology import four_neighborhood_edges


@dataclass(frozen=True, slots=True)
class PlacementSettings:
    mode: NominalPlacementMode = NominalPlacementMode.REGULAR_GRID
    pixel_size: PixelSize | None = None
    stage_outlier_factor: float = 3.0
    min_stride_px: float = 8.0


def expected_pair_displacement(
    source: Tile,
    target: Tile,
    placement: Mapping[GridCoordinate, NominalCoordinate],
) -> Translation2D:
    origin = placement[source.coord]
    dest = placement[target.coord]
    return Translation2D(dest.x - origin.x, dest.y - origin.y)


def compute_nominal_placement(grid: TileGrid, settings: PlacementSettings) -> dict[GridCoordinate, NominalCoordinate]:
    regular = {
        coord: regular_grid_position(coord, tile.width, tile.height, grid.overlap_x, grid.overlap_y)
        for coord, tile in grid.tiles.items()
    }
    if settings.mode is NominalPlacementMode.REGULAR_GRID:
        return regular
    if settings.pixel_size is None:
        raise PlacementError("SEM_STAGE and HYBRID placement require pixel_size")
    if settings.mode is NominalPlacementMode.SEM_STAGE:
        return _sem_stage_positions(grid, settings.pixel_size, regular)
    return _hybrid_positions(grid, settings, regular)


def _sem_stage_positions(
    grid: TileGrid,
    pixel_size: PixelSize,
    regular: Mapping[GridCoordinate, NominalCoordinate],
) -> dict[GridCoordinate, NominalCoordinate]:
    missing = [coord for coord, tile in grid.tiles.items() if tile.stage is None]
    if missing:
        raise PlacementError(
            "SEM_STAGE placement requires stage coordinates on every tile; "
            f"missing {len(missing)} tile(s), e.g. {missing[0]}"
        )
    origin_coord = min(grid.tiles)
    origin_stage = grid.require(origin_coord).stage
    assert origin_stage is not None
    origin_px = stage_to_pixels(origin_stage, pixel_size)
    positions: dict[GridCoordinate, NominalCoordinate] = {}
    for coord, tile in grid.tiles.items():
        assert tile.stage is not None
        current = stage_to_pixels(tile.stage, pixel_size)
        positions[coord] = NominalCoordinate(x=current.x - origin_px.x, y=current.y - origin_px.y)
    return positions if positions else dict(regular)


def _hybrid_positions(
    grid: TileGrid,
    settings: PlacementSettings,
    regular: Mapping[GridCoordinate, NominalCoordinate],
) -> dict[GridCoordinate, NominalCoordinate]:
    """Neighborhood from GridCoordinate; edge lengths from stage when they are not outliers."""

    pixel_size = settings.pixel_size
    assert pixel_size is not None
    coords = set(grid.tiles)
    origin = min(coords)
    positions: dict[GridCoordinate, NominalCoordinate] = {origin: NominalCoordinate(0.0, 0.0)}
    pending = [origin]
    while pending:
        current = pending.pop(0)
        for source, target in four_neighborhood_edges(coords):
            if source != current and target != current:
                continue
            neighbor = target if source == current else source
            if neighbor in positions:
                continue
            delta = _hybrid_edge_delta(grid.require(source), grid.require(target), settings, regular)
            if source == current:
                base = positions[current]
                positions[neighbor] = NominalCoordinate(base.x + delta.dx, base.y + delta.dy)
            else:
                base = positions[current]
                positions[neighbor] = NominalCoordinate(base.x - delta.dx, base.y - delta.dy)
            pending.append(neighbor)
    for coord in coords:
        if coord not in positions:
            ref = regular[origin]
            item = regular[coord]
            positions[coord] = NominalCoordinate(item.x - ref.x, item.y - ref.y)
    return positions


def _hybrid_edge_delta(
    source: Tile,
    target: Tile,
    settings: PlacementSettings,
    regular: Mapping[GridCoordinate, NominalCoordinate],
) -> Translation2D:
    regular_delta = Translation2D(
        regular[target.coord].x - regular[source.coord].x,
        regular[target.coord].y - regular[source.coord].y,
    )
    if source.stage is None or target.stage is None or settings.pixel_size is None:
        return regular_delta
    source_px = stage_to_pixels(source.stage, settings.pixel_size)
    target_px = stage_to_pixels(target.stage, settings.pixel_size)
    stage_delta = Translation2D(target_px.x - source_px.x, target_px.y - source_px.y)
    stride = max(abs(regular_delta.dx), abs(regular_delta.dy), settings.min_stride_px)
    if stage_delta.magnitude > settings.stage_outlier_factor * stride:
        return regular_delta
    if regular_delta.magnitude > 0.0:
        # Reject stage vectors that point against grid topology.
        dot = stage_delta.dx * regular_delta.dx + stage_delta.dy * regular_delta.dy
        if dot <= 0.0:
            return regular_delta
    return stage_delta
