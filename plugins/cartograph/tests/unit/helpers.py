from __future__ import annotations

import cv2
import numpy as np

from cartograph.domain.coordinates import GridCoordinate, StageCoordinate
from cartograph.domain.tiles import ImageBuffer, Tile, TileGrid


def unique_scene(height: int, width: int, seed: int = 0, sigma: float = 1.5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(128.0, 45.0, (height, width)).astype(np.float32)
    if sigma > 0.0:
        noise = cv2.GaussianBlur(noise, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma)
    return np.clip(noise, 0.0, 255.0).astype(np.float32)


def crop_at(scene: np.ndarray, x: float, y: float, width: int, height: int) -> np.ndarray:
    if float(x).is_integer() and float(y).is_integer():
        left = int(x)
        top = int(y)
        return np.ascontiguousarray(scene[top : top + height, left : left + width], dtype=np.float32)
    matrix = np.array([[1.0, 0.0, x], [0.0, 1.0, y]], dtype=np.float32)
    return cv2.warpAffine(scene, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def striped_scene(height: int, width: int, period: int = 16) -> np.ndarray:
    xs = np.arange(width, dtype=np.float32)
    wave = np.sin(2.0 * np.pi * xs / float(period)) * 80.0 + 128.0
    return np.broadcast_to(wave, (height, width)).copy()


def buffer(pixels: np.ndarray) -> ImageBuffer:
    return ImageBuffer(np.ascontiguousarray(pixels, dtype=np.float32))


def subpixel_shift(pixels: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Apply a small in-plane translation using the same border policy as ``crop_at``."""

    matrix = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    height, width = pixels.shape
    return cv2.warpAffine(
        np.ascontiguousarray(pixels, dtype=np.float32),
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )


def tile_at(row: int, col: int, width: int, height: int, *, stage: StageCoordinate | None = None, source: str | None = None) -> Tile:
    coord = GridCoordinate(row, col)
    return Tile(
        coord=coord,
        source_id=source or f"tile-{row}-{col}",
        width=width,
        height=height,
        stage=stage,
    )


def grid_from_tiles(tiles: list[Tile], overlap_x: float = 0.25, overlap_y: float = 0.25) -> TileGrid:
    return TileGrid(tiles={item.coord: item for item in tiles}, overlap_x=overlap_x, overlap_y=overlap_y, name="synthetic")
