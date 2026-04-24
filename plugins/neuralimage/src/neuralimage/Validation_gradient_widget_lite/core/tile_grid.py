"""Tile-grid planning helpers for the validation gradient widget."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TileGridPlan:
    """Describe one tile-grid configuration for a source frame."""

    source_shape: tuple[int, int]
    requested_tile_width: int
    requested_tile_height: int
    requested_overlap: int
    tile_width: int
    tile_height: int
    overlap: int
    stride_x: int
    stride_y: int
    columns: int
    rows: int
    requested_exact: bool
    applied_exact: bool
    x_starts: tuple[int, ...] = ()
    y_starts: tuple[int, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def is_exact(self) -> bool:
        return bool(self.applied_exact)


def _axis_starts(length: int, tile_size: int, stride: int) -> tuple[int, ...]:
    length = max(0, int(length))
    tile_size = max(1, int(tile_size))
    stride = max(1, int(stride))
    if length <= 0:
        return (0,)
    if tile_size >= length:
        return (0,)
    last_start = max(0, length - tile_size)
    starts: list[int] = [0]
    position = 0
    while True:
        next_position = position + stride
        if next_position >= last_start:
            break
        starts.append(int(next_position))
        position = int(next_position)
    if starts[-1] != last_start:
        starts.append(int(last_start))
    return tuple(int(value) for value in starts)


def tile_bounds_for_index(plan: TileGridPlan, row: int, column: int) -> tuple[int, int, int, int]:
    """Return integer bounds for one tile from an existing plan."""

    source_height, source_width = (max(0, int(plan.source_shape[0])), max(0, int(plan.source_shape[1])))
    row_index = int(row)
    column_index = int(column)
    if row_index < 0 or column_index < 0:
        return 0, 0, 0, 0
    if row_index >= len(plan.y_starts) or column_index >= len(plan.x_starts):
        return 0, 0, 0, 0
    left = max(0, int(plan.x_starts[column_index]))
    top = max(0, int(plan.y_starts[row_index]))
    right = min(source_width, left + int(plan.tile_width))
    bottom = min(source_height, top + int(plan.tile_height))
    return left, top, max(0, right - left), max(0, bottom - top)


def plan_tile_grid(
    source_shape: tuple[int, int],
    tile_width: int,
    tile_height: int,
    overlap: int,
    *,
    search_radius: int = 32,
) -> TileGridPlan:
    """Plan a tile grid using explicit tile size and edge-aligned final tiles."""

    _ = search_radius
    source_height, source_width = (max(0, int(source_shape[0])), max(0, int(source_shape[1])))
    requested_tile_width = max(1, int(tile_width))
    requested_tile_height = max(1, int(tile_height))
    requested_overlap = max(0, int(overlap))

    if source_height <= 0 or source_width <= 0:
        return TileGridPlan(
            source_shape=(source_height, source_width),
            requested_tile_width=requested_tile_width,
            requested_tile_height=requested_tile_height,
            requested_overlap=requested_overlap,
            tile_width=requested_tile_width,
            tile_height=requested_tile_height,
            overlap=0,
            stride_x=max(1, requested_tile_width),
            stride_y=max(1, requested_tile_height),
            columns=1,
            rows=1,
            requested_exact=False,
            applied_exact=False,
            x_starts=(0,),
            y_starts=(0,),
            notes=("source_frame_is_empty",),
        )

    clipped_tile_width = min(requested_tile_width, source_width)
    clipped_tile_height = min(requested_tile_height, source_height)
    max_overlap = max(0, min(clipped_tile_width, clipped_tile_height) - 1)
    applied_overlap = max(0, min(requested_overlap, max_overlap))
    stride_x = max(1, clipped_tile_width - applied_overlap)
    stride_y = max(1, clipped_tile_height - applied_overlap)
    x_starts = _axis_starts(source_width, clipped_tile_width, stride_x)
    y_starts = _axis_starts(source_height, clipped_tile_height, stride_y)

    notes: list[str] = []
    if clipped_tile_width != requested_tile_width or clipped_tile_height != requested_tile_height:
        notes.append("tile_size_clipped_to_source")
    if applied_overlap != requested_overlap:
        notes.append("overlap_clipped_to_tile_size")
    last_column_start = max(0, source_width - clipped_tile_width)
    last_row_start = max(0, source_height - clipped_tile_height)
    if len(x_starts) > 1 and x_starts[-1] != max(0, (len(x_starts) - 1) * stride_x):
        notes.append("right_edge_tile_aligned")
    if len(y_starts) > 1 and y_starts[-1] != max(0, (len(y_starts) - 1) * stride_y):
        notes.append("bottom_edge_tile_aligned")
    if x_starts[-1] != last_column_start or y_starts[-1] != last_row_start:
        notes.append("edge_alignment_adjusted")

    requested_exact = (
        requested_tile_width == clipped_tile_width
        and requested_tile_height == clipped_tile_height
        and requested_overlap == applied_overlap
    )
    return TileGridPlan(
        source_shape=(source_height, source_width),
        requested_tile_width=requested_tile_width,
        requested_tile_height=requested_tile_height,
        requested_overlap=requested_overlap,
        tile_width=clipped_tile_width,
        tile_height=clipped_tile_height,
        overlap=applied_overlap,
        stride_x=stride_x,
        stride_y=stride_y,
        columns=len(x_starts),
        rows=len(y_starts),
        requested_exact=requested_exact,
        applied_exact=True,
        x_starts=x_starts,
        y_starts=y_starts,
        notes=tuple(notes),
    )
