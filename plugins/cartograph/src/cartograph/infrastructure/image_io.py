"""Decode tiles into domain ImageBuffer objects. OpenCV stays here."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from cartograph.domain.errors import GridLoadError
from cartograph.domain.tiles import ImageBuffer, Tile


class OpenCvTileImageLoader:
    def load(self, tile: Tile) -> ImageBuffer:
        if tile.path is None:
            raise GridLoadError(f"tile {tile.source_id} has no filesystem path")
        return load_grayscale(tile.path)


class MemoryTileImageLoader:
    def __init__(self, images: dict[str, ImageBuffer]) -> None:
        self._images = images

    def load(self, tile: Tile) -> ImageBuffer:
        image = self._images.get(tile.source_id)
        if image is None:
            raise GridLoadError(f"no in-memory image for {tile.source_id}")
        return image


def load_grayscale(path: Path) -> ImageBuffer:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise GridLoadError(f"cannot decode grayscale image: {path}")
    return ImageBuffer(np.ascontiguousarray(image, dtype=np.float32))
