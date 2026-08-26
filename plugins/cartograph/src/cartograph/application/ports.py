"""Application ports. Cache and I/O stay replaceable; registration is coarse-grained."""

from __future__ import annotations

from typing import Mapping, Protocol

from cartograph.domain.coordinates import GridCoordinate, NominalCoordinate
from cartograph.domain.tiles import ImageBuffer, Tile, TileGrid
from cartograph.domain.topology import LocalBlockSolution


class TileImageLoader(Protocol):
    def load(self, tile: Tile) -> ImageBuffer:
        """Decode one tile into a domain grayscale buffer."""


class LocalRegistrationCache(Protocol):
    def get(self, key: str) -> LocalBlockSolution | None:
        """Return a previously computed local block, if present."""

    def put(self, key: str, solution: LocalBlockSolution) -> None:
        """Store a verified local block keyed by a parameter hash."""


class ThumbnailCache(Protocol):
    """Reserved. LOD-0 thumbnails are not implemented in v1."""

    def get(self, key: str) -> ImageBuffer | None: ...

    def put(self, key: str, image: ImageBuffer) -> None: ...


class DecodedTileCache(Protocol):
    """Reserved. Decoded-tile cache can wrap TileImageLoader later."""

    def get(self, source_id: str) -> ImageBuffer | None: ...

    def put(self, source_id: str, image: ImageBuffer) -> None: ...


class ViewportRenderCache(Protocol):
    """Reserved. Viewport mosaic cache is not implemented in v1."""

    def get(self, key: str) -> ImageBuffer | None: ...

    def put(self, key: str, image: ImageBuffer) -> None: ...


class LocalBlockStore(Protocol):
    def load(self, key: str) -> LocalBlockSolution | None:
        """Load a persisted local block."""

    def save(self, key: str, solution: LocalBlockSolution) -> None:
        """Persist transforms and registration diagnostics."""


class NominalPlacement(Protocol):
    def place(self, grid: TileGrid) -> Mapping[GridCoordinate, NominalCoordinate]:
        """Compute navigation positions without high-precision stitching."""
