from __future__ import annotations

import numpy as np
import pytest

from cartograph.application.local_registration import LocalRegistrationRequest, RegisterLocalWindow
from cartograph.application.nominal import PlacementSettings, compute_nominal_placement
from cartograph.domain.coordinates import GridCoordinate
from cartograph.domain.registration import RegistrationParameters, RegistrationStatus
from cartograph.infrastructure.opencv.registrar import PythonPairRegistrar

from .helpers import buffer, crop_at, grid_from_tiles, tile_at, unique_scene


def test_empty_center_tile_skips_high_precision_pairs() -> None:
    blank = buffer(np.full((64, 64), 12.0, dtype=np.float32))
    textured = buffer(unique_scene(64, 64, seed=9))
    tiles = [tile_at(r, c, 64, 64) for r in range(3) for c in range(3)]
    grid = grid_from_tiles(tiles)
    images = {tile.coord: textured for tile in tiles}
    images[GridCoordinate(1, 1)] = blank
    images[GridCoordinate(1, 0)] = blank
    placement = compute_nominal_placement(grid, PlacementSettings())
    outcome = RegisterLocalWindow(PythonPairRegistrar(RegistrationParameters())).execute(
        LocalRegistrationRequest(grid=grid, images=images, center=GridCoordinate(1, 1), placement=placement)
    )
    statuses = {edge.result.status for edge in outcome.solution.graph.edges if GridCoordinate(1, 1) in {edge.source, edge.target}}
    assert RegistrationStatus.EMPTY_TILE in statuses
    assert all(edge.result.status is not RegistrationStatus.OK or edge.result.confidence > 0 for edge in outcome.solution.graph.edges)


def test_full_3x3_unique_scene_recovers_nominal_stride() -> None:
    scene = unique_scene(320, 320, seed=4, sigma=1.2)
    tile = 96
    stride = 72
    tiles = []
    images = {}
    for row in range(3):
        for col in range(3):
            item = tile_at(row, col, tile, tile)
            tiles.append(item)
            images[item.coord] = buffer(crop_at(scene, col * stride, row * stride, tile, tile))
    grid = grid_from_tiles(tiles, overlap_x=1.0 - stride / tile, overlap_y=1.0 - stride / tile)
    placement = compute_nominal_placement(grid, PlacementSettings())
    outcome = RegisterLocalWindow(PythonPairRegistrar(RegistrationParameters(search_radius_px=8.0))).execute(
        LocalRegistrationRequest(
            grid=grid,
            images=images,
            center=GridCoordinate(1, 1),
            placement=placement,
            parameters=RegistrationParameters(search_radius_px=8.0),
        )
    )
    poses = outcome.solution.poses
    assert poses[GridCoordinate(1, 1)].dx == 0.0
    assert poses[GridCoordinate(1, 2)].dx == pytest.approx(stride, abs=1.5)
    assert poses[GridCoordinate(2, 1)].dy == pytest.approx(stride, abs=1.5)
    assert len(outcome.solution.graph.edges) == 12
