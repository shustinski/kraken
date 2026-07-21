from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PatchWindow:
    """A valid source-image region represented by one padded model tile."""

    left: int
    top: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return int(self.left + self.width)

    @property
    def bottom(self) -> int:
        return int(self.top + self.height)

    @property
    def center_x(self) -> float:
        return float(self.left + (self.width / 2.0))

    @property
    def center_y(self) -> float:
        return float(self.top + (self.height / 2.0))


@dataclass(frozen=True)
class TilePlan:
    """Deterministic geometry shared by cutting, context and stitching."""

    base_height: int
    base_width: int
    tile_height: int
    tile_width: int
    overlap: int
    windows: tuple[PatchWindow, ...]

    @property
    def tile_count(self) -> int:
        return len(self.windows)

    @property
    def base_shape_hw(self) -> tuple[int, int]:
        return self.base_height, self.base_width

    @property
    def tile_shape_hw(self) -> tuple[int, int]:
        return self.tile_height, self.tile_width


def _axis_starts(image_size: int, tile_size: int, overlap: int) -> tuple[int, ...]:
    if image_size <= tile_size:
        return (0,)
    stride = tile_size - overlap
    last = image_size - tile_size
    starts = list(range(0, last + 1, stride))
    if not starts or starts[-1] != last:
        starts.append(last)
    return tuple(starts)


def build_tile_plan(
    base_shape_hw: tuple[int, int],
    tile_size_xy: tuple[int, int],
    overlap: int,
) -> TilePlan:
    base_height, base_width = (int(base_shape_hw[0]), int(base_shape_hw[1]))
    tile_width, tile_height = (int(tile_size_xy[0]), int(tile_size_xy[1]))
    overlap = int(overlap)
    if base_height <= 0 or base_width <= 0:
        raise ValueError(f'Image dimensions must be positive, got {base_shape_hw!r}.')
    if tile_height <= 0 or tile_width <= 0:
        raise ValueError(f'Tile dimensions must be positive, got {tile_size_xy!r}.')
    if overlap < 0 or overlap >= min(tile_height, tile_width):
        raise ValueError(
            'overlap must satisfy 0 <= overlap < min(tile_width, tile_height), '
            f'got overlap={overlap}, tile={tile_size_xy!r}.'
        )

    row_starts = _axis_starts(base_height, tile_height, overlap)
    column_starts = _axis_starts(base_width, tile_width, overlap)
    windows = tuple(
        PatchWindow(
            left=left,
            top=top,
            width=min(tile_width, base_width - left),
            height=min(tile_height, base_height - top),
        )
        for top in row_starts
        for left in column_starts
    )
    return TilePlan(
        base_height=base_height,
        base_width=base_width,
        tile_height=tile_height,
        tile_width=tile_width,
        overlap=overlap,
        windows=windows,
    )


def build_legacy_tile_plan(
    base_shape_hw: tuple[int, int],
    tile_size_xy: tuple[int, int],
    overlap: int,
) -> TilePlan:
    """Reproduce the duplicated edge-window geometry used by pipeline v1."""

    base_height, base_width = int(base_shape_hw[0]), int(base_shape_hw[1])
    tile_width, tile_height = int(tile_size_xy[0]), int(tile_size_xy[1])
    overlap = int(overlap)
    if base_height <= 0 or base_width <= 0 or tile_height <= 0 or tile_width <= 0:
        raise ValueError('Image and tile dimensions must be positive.')
    if overlap < 0 or overlap >= min(tile_height, tile_width):
        raise ValueError(
            'overlap must satisfy 0 <= overlap < min(tile_width, tile_height), '
            f'got overlap={overlap}, tile={tile_size_xy!r}.'
        )
    stride_height = tile_height - overlap
    stride_width = tile_width - overlap
    row_steps = int(base_height / stride_height) + 1
    column_steps = int(base_width / stride_width) + 1
    windows: list[PatchWindow] = []
    for row in range(row_steps):
        for column in range(column_steps):
            nominal_top = row * stride_height
            nominal_left = column * stride_width
            top = nominal_top if nominal_top + tile_height <= base_height else max(0, base_height - tile_height)
            left = nominal_left if nominal_left + tile_width <= base_width else max(0, base_width - tile_width)
            windows.append(
                PatchWindow(
                    left=left,
                    top=top,
                    width=min(tile_width, base_width - left),
                    height=min(tile_height, base_height - top),
                )
            )
    return TilePlan(
        base_height=base_height,
        base_width=base_width,
        tile_height=tile_height,
        tile_width=tile_width,
        overlap=overlap,
        windows=tuple(windows),
    )
