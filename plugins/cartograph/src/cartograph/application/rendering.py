"""Render a local mosaic from original tiles and local-block poses."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from cartograph.domain.coordinates import GridCoordinate, Translation2D
from cartograph.domain.tiles import ImageBuffer, Tile
from cartograph.infrastructure.render import BlendMode, render_local_mosaic


@dataclass(frozen=True, slots=True)
class RenderLocalMosaicRequest:
    tiles: Mapping[GridCoordinate, Tile]
    images: Mapping[GridCoordinate, ImageBuffer]
    poses: Mapping[GridCoordinate, Translation2D]
    blend: BlendMode = BlendMode.FEATHERED
    feather_px: int = 16


class RenderLocalMosaic:
    def execute(self, request: RenderLocalMosaicRequest) -> ImageBuffer:
        return render_local_mosaic(
            request.tiles,
            request.images,
            request.poses,
            blend=request.blend,
            feather_px=request.feather_px,
        )
