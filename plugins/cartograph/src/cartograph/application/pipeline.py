"""Headless vertical-slice pipeline used by CLI and the diagnostic UI."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from cartograph.application.load_grid import LoadGridRequest, LoadTileGrid
from cartograph.application.local_registration import (
    LocalRegistrationOutcome,
    LocalRegistrationRequest,
    RegisterLocalWindow,
)
from cartograph.application.nominal import PlacementSettings, compute_nominal_placement
from cartograph.application.rendering import RenderLocalMosaic, RenderLocalMosaicRequest
from cartograph.domain.coordinates import GridCoordinate, NominalPlacementMode
from cartograph.domain.registration import RegistrationParameters
from cartograph.domain.tiles import ImageBuffer, TileGrid
from cartograph.domain.topology import LocalBlockSolution
from cartograph.infrastructure.image_io import OpenCvTileImageLoader
from cartograph.infrastructure.opencv import PythonRegistrationBackend
from cartograph.infrastructure.persistence import InMemoryRegistrationCache, JsonLocalBlockStore
from cartograph.infrastructure.render import BlendMode


@dataclass(frozen=True, slots=True)
class VerticalSliceRequest:
    path: Path
    center: GridCoordinate
    overlap_x: float = 0.1
    overlap_y: float = 0.1
    placement: PlacementSettings = PlacementSettings()
    parameters: RegistrationParameters = RegistrationParameters()
    blend: BlendMode = BlendMode.FEATHERED
    cache_dir: Path | None = None


@dataclass(frozen=True, slots=True)
class VerticalSliceResult:
    grid: TileGrid
    images: Mapping[GridCoordinate, ImageBuffer]
    placement: Mapping[GridCoordinate, object]
    outcome: LocalRegistrationOutcome
    mosaic: ImageBuffer

    @property
    def solution(self) -> LocalBlockSolution:
        return self.outcome.solution


class RunLocalVerticalSlice:
    def __init__(self) -> None:
        self._loader = LoadTileGrid()
        self._image_loader = OpenCvTileImageLoader()
        self._renderer = RenderLocalMosaic()

    def execute(self, request: VerticalSliceRequest) -> VerticalSliceResult:
        grid = self._loader.execute(LoadGridRequest(request.path, request.overlap_x, request.overlap_y))
        placement = compute_nominal_placement(grid, request.placement)
        window_coords = _window_coords(request.center)
        images = {
            coord: self._image_loader.load(grid.require(coord))
            for coord in window_coords
            if grid.contains(coord)
        }
        cache = InMemoryRegistrationCache()
        store = JsonLocalBlockStore(_cache_root(request))
        use_case = RegisterLocalWindow(
            PythonRegistrationBackend(request.parameters),
            cache=cache,
            store=store,
        )
        outcome = use_case.execute(
            LocalRegistrationRequest(
                grid=grid,
                images=images,
                center=request.center,
                placement=placement,
                parameters=request.parameters,
            )
        )
        mosaic = self._renderer.execute(
            RenderLocalMosaicRequest(
                tiles=outcome.window.tiles,
                images=images,
                poses=outcome.solution.poses,
                blend=request.blend,
            )
        )
        return VerticalSliceResult(
            grid=grid,
            images=images,
            placement=placement,
            outcome=outcome,
            mosaic=mosaic,
        )


def _window_coords(center: GridCoordinate) -> list[GridCoordinate]:
    coords: list[GridCoordinate] = []
    for d_row in (-1, 0, 1):
        for d_col in (-1, 0, 1):
            row = center.row + d_row
            col = center.col + d_col
            if row >= 0 and col >= 0:
                coords.append(GridCoordinate(row, col))
    return coords


def _cache_root(request: VerticalSliceRequest) -> Path:
    if request.cache_dir is not None:
        return request.cache_dir
    path = request.path
    root = path.parent if path.is_file() else path
    return root / ".cartograph" / "local-blocks"


def default_placement_mode(value: str) -> PlacementSettings:
    return PlacementSettings(mode=NominalPlacementMode(value))
