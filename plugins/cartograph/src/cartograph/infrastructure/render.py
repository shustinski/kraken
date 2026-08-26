"""Local mosaic renderer. Originals are resampled at most once onto the viewport canvas."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

from cartograph.domain.coordinates import GridCoordinate, Translation2D
from cartograph.domain.tiles import ImageBuffer, Tile


class BlendMode(StrEnum):
    ALPHA = "alpha"
    HARD_SEAM = "hard_seam"
    DIFFERENCE = "difference"
    FEATHERED = "feathered"


def render_local_mosaic(
    tiles: Mapping[GridCoordinate, Tile],
    images: Mapping[GridCoordinate, ImageBuffer],
    poses: Mapping[GridCoordinate, Translation2D],
    *,
    blend: BlendMode = BlendMode.FEATHERED,
    feather_px: int = 16,
) -> ImageBuffer:
    if not poses:
        raise ValueError("cannot render an empty pose set")
    origins: list[tuple[GridCoordinate, float, float, int, int]] = []
    for coord, pose in poses.items():
        image = images.get(coord)
        tile = tiles.get(coord)
        if image is None or tile is None:
            continue
        origins.append((coord, pose.dx, pose.dy, image.width, image.height))
    if not origins:
        raise ValueError("no tile images available for mosaic rendering")

    min_x = min(item[1] for item in origins)
    min_y = min(item[2] for item in origins)
    max_x = max(item[1] + item[3] for item in origins)
    max_y = max(item[2] + item[4] for item in origins)
    canvas_w = int(np.ceil(max_x - min_x)) + 1
    canvas_h = int(np.ceil(max_y - min_y)) + 1
    accum = np.zeros((canvas_h, canvas_w), dtype=np.float64)
    weight = np.zeros((canvas_h, canvas_w), dtype=np.float64)
    first = np.zeros((canvas_h, canvas_w), dtype=np.float64)
    first_mask = np.zeros((canvas_h, canvas_w), dtype=bool)
    last = np.zeros((canvas_h, canvas_w), dtype=np.float64)

    for coord, origin_x, origin_y, _width, _height in sorted(origins, key=lambda item: (item[0].row, item[0].col)):
        image = images[coord]
        x0 = origin_x - min_x
        y0 = origin_y - min_y
        ix0 = int(np.floor(x0))
        iy0 = int(np.floor(y0))
        # Translation-only: integer paste plus bilinear weights for the fractional part.
        frac_x = float(x0 - ix0)
        frac_y = float(y0 - iy0)
        pasted = _paste_subpixel(image.pixels, canvas_h, canvas_w, ix0, iy0, frac_x, frac_y)
        coverage = _paste_subpixel(np.ones_like(image.pixels), canvas_h, canvas_w, ix0, iy0, frac_x, frac_y)
        tile_weight = coverage
        if blend is BlendMode.FEATHERED:
            tile_weight = _paste_subpixel(
                _feather_mask(image.height, image.width, feather_px),
                canvas_h,
                canvas_w,
                ix0,
                iy0,
                frac_x,
                frac_y,
            )
        if blend is BlendMode.HARD_SEAM:
            write = coverage > 0.5
            accum[write] = pasted[write]
            weight[write] = 1.0
        else:
            accum += pasted * tile_weight
            weight += tile_weight
        write = coverage > 0.5
        unset = write & ~first_mask
        first[unset] = pasted[unset]
        first_mask[unset] = True
        last[write] = pasted[write]

    if blend is BlendMode.DIFFERENCE:
        mosaic = np.abs(last - first)
        mosaic[~first_mask] = 0.0
        return ImageBuffer(np.clip(mosaic, 0.0, 255.0).astype(np.float32))

    safe_weight = np.maximum(weight, 1e-6)
    mosaic = accum / safe_weight
    mosaic[weight <= 1e-6] = 0.0
    if blend is BlendMode.ALPHA:
        # 50/50 in overlaps is the mean of contributing tiles, which is exact for two tiles.
        pass
    return ImageBuffer(np.clip(mosaic, 0.0, 255.0).astype(np.float32))


def _feather_mask(height: int, width: int, feather_px: int) -> NDArray[np.float32]:
    feather = max(1, int(feather_px))
    rows = np.minimum(np.arange(height), np.arange(height)[::-1]).astype(np.float32)
    cols = np.minimum(np.arange(width), np.arange(width)[::-1]).astype(np.float32)
    dist = np.minimum(rows[:, None], cols[None, :])
    return np.clip(dist / float(feather), 0.05, 1.0).astype(np.float32)


def _paste_subpixel(
    image: NDArray[np.float32],
    canvas_h: int,
    canvas_w: int,
    x0: int,
    y0: int,
    frac_x: float,
    frac_y: float,
) -> NDArray[np.float64]:
    canvas = np.zeros((canvas_h, canvas_w), dtype=np.float64)
    weights = (
        (1.0 - frac_x) * (1.0 - frac_y),
        frac_x * (1.0 - frac_y),
        (1.0 - frac_x) * frac_y,
        frac_x * frac_y,
    )
    offsets = ((0, 0), (0, 1), (1, 0), (1, 1))
    pixels = np.asarray(image, dtype=np.float64)
    height, width = pixels.shape
    for (d_row, d_col), factor in zip(offsets, weights, strict=True):
        if factor <= 1e-12:
            continue
        _add_at(canvas, pixels * factor, y0 + d_row, x0 + d_col, height, width)
    return canvas


def _add_at(canvas: NDArray[np.float64], patch: NDArray[np.float64], y0: int, x0: int, height: int, width: int) -> None:
    y1 = y0 + height
    x1 = x0 + width
    cy0 = max(0, y0)
    cx0 = max(0, x0)
    cy1 = min(canvas.shape[0], y1)
    cx1 = min(canvas.shape[1], x1)
    if cy0 >= cy1 or cx0 >= cx1:
        return
    py0 = cy0 - y0
    px0 = cx0 - x0
    canvas[cy0:cy1, cx0:cx1] += patch[py0 : py0 + (cy1 - cy0), px0 : px0 + (cx1 - cx0)]
