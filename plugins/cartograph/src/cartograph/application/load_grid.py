"""Load a SEM tile grid through a filesystem/manifest port."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cartograph.domain.tiles import TileGrid
from cartograph.infrastructure.grid_loader import load_tile_grid


@dataclass(frozen=True, slots=True)
class LoadGridRequest:
    path: Path
    overlap_x: float = 0.1
    overlap_y: float = 0.1


class LoadTileGrid:
    def execute(self, request: LoadGridRequest) -> TileGrid:
        return load_tile_grid(request.path, overlap_x=request.overlap_x, overlap_y=request.overlap_y)
