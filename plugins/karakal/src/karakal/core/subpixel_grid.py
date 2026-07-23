"""Helpers for parent-pixel and subpixel grid representation."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import numpy as np

from .tile_grid import plan_tile_grid, tile_bounds_for_index


@dataclass(frozen=True, slots=True)
class SubpixelGridSpec:
    """Describe a regular or pixel-tile subpixel grid inside one parent pixel."""

    rows: int = 1
    columns: int = 1
    mode: str = "grid"
    tile_width: int = 0
    tile_height: int = 0
    overlap: int = 0

    def normalized(self) -> "SubpixelGridSpec":
        rows = max(1, int(self.rows))
        columns = max(1, int(self.columns))
        mode = "tile" if str(self.mode or "grid").strip().lower() == "tile" else "grid"
        tile_width = max(1, int(self.tile_width or 1))
        tile_height = max(1, int(self.tile_height or 1))
        max_overlap = max(0, min(tile_width, tile_height) - 1)
        overlap = max(0, min(int(self.overlap or 0), max_overlap))
        return SubpixelGridSpec(
            rows=rows,
            columns=columns,
            mode=mode,
            tile_width=tile_width,
            tile_height=tile_height,
            overlap=overlap,
        )

    @property
    def is_tile_mode(self) -> bool:
        return self.normalized().mode == "tile"

    @classmethod
    def from_tile_plan(cls, tile_width: int, tile_height: int, overlap: int, rows: int, columns: int) -> "SubpixelGridSpec":
        return cls(
            rows=int(rows),
            columns=int(columns),
            mode="tile",
            tile_width=int(tile_width),
            tile_height=int(tile_height),
            overlap=int(overlap),
        ).normalized()


@dataclass(slots=True)
class SubpixelGrid:
    """Store values and optional confidence scores for one parent pixel."""

    spec: SubpixelGridSpec
    values: np.ndarray
    confidences: np.ndarray | None = None
    aggregation: str = "mean"
    value_kind: str = "value"

    def __post_init__(self) -> None:
        normalized_spec = self.spec.normalized()
        values = np.asarray(self.values, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError("SubpixelGrid.values must be a 2D array")
        if values.shape != (normalized_spec.rows, normalized_spec.columns):
            raise ValueError(
                f"SubpixelGrid.values shape {values.shape} does not match spec {(normalized_spec.rows, normalized_spec.columns)}"
            )
        self.spec = normalized_spec
        self.values = values
        if self.confidences is not None:
            confidences = np.asarray(self.confidences, dtype=np.float32)
            if confidences.shape != values.shape:
                raise ValueError(
                    f"SubpixelGrid.confidences shape {confidences.shape} does not match values shape {values.shape}"
                )
            self.confidences = confidences

    def aggregate_value(self, aggregation: str | None = None) -> float:
        return aggregate_subpixel_values(self.values, self.confidences, aggregation or self.aggregation)

    def value_at(self, row: int, column: int) -> float:
        return float(self.values[int(row), int(column)])

    def confidence_at(self, row: int, column: int) -> float | None:
        if self.confidences is None:
            return None
        return float(self.confidences[int(row), int(column)])


def _partition_edges(total: int, parts: int) -> np.ndarray:
    total = max(0, int(total))
    parts = max(1, int(parts))
    edges = np.rint(np.linspace(0.0, float(total), parts + 1)).astype(np.int32)
    edges[0] = 0
    edges[-1] = total
    for index in range(1, edges.size):
        if edges[index] < edges[index - 1]:
            edges[index] = edges[index - 1]
    return edges


def _resolved_spec_for_shape(shape: tuple[int, int], spec: SubpixelGridSpec) -> tuple[SubpixelGridSpec, list[tuple[int, int, int, int]]]:
    normalized_spec = spec.normalized()
    source_height = max(0, int(shape[0]))
    source_width = max(0, int(shape[1]))
    if normalized_spec.is_tile_mode:
        plan = plan_tile_grid(
            (source_height, source_width),
            normalized_spec.tile_width,
            normalized_spec.tile_height,
            normalized_spec.overlap,
        )
        bounds = [
            tile_bounds_for_index(plan, row, column)
            for row in range(int(plan.rows))
            for column in range(int(plan.columns))
        ]
        resolved_spec = replace(normalized_spec, rows=int(plan.rows), columns=int(plan.columns))
        return resolved_spec, bounds

    rows = max(1, int(normalized_spec.rows))
    columns = max(1, int(normalized_spec.columns))
    y_edges = _partition_edges(source_height, rows)
    x_edges = _partition_edges(source_width, columns)
    bounds: list[tuple[int, int, int, int]] = []
    for row in range(rows):
        top = int(y_edges[row])
        bottom = int(y_edges[row + 1])
        for column in range(columns):
            left = int(x_edges[column])
            right = int(x_edges[column + 1])
            bounds.append((left, top, max(0, right - left), max(0, bottom - top)))
    return normalized_spec, bounds


def build_subpixel_grid_from_array(
    array,
    spec: SubpixelGridSpec,
    *,
    score_fn: Callable[[np.ndarray], float],
    aggregation: str = "mean",
    value_kind: str = "value",
) -> SubpixelGrid:
    """Build a subpixel grid by scoring cropped regions from one aligned 2D array."""

    source = np.asarray(array)
    if source.ndim != 2 or source.size == 0:
        raise ValueError("Subpixel source array must be a non-empty 2D array")
    resolved_spec, bounds = _resolved_spec_for_shape(source.shape, spec)
    values = np.zeros((resolved_spec.rows, resolved_spec.columns), dtype=np.float32)
    confidences = np.ones((resolved_spec.rows, resolved_spec.columns), dtype=np.float32)
    for index, (left, top, width, height) in enumerate(bounds):
        row = index // int(resolved_spec.columns)
        column = index % int(resolved_spec.columns)
        tile = source[top:top + height, left:left + width]
        if tile.size == 0:
            continue
        values[row, column] = float(np.clip(score_fn(tile), 0.0, 1.0))
    return SubpixelGrid(
        spec=resolved_spec,
        values=values,
        confidences=confidences,
        aggregation=aggregation,
        value_kind=str(value_kind or "value"),
    )


def build_subpixel_grid_from_pair(
    first,
    second,
    spec: SubpixelGridSpec,
    *,
    score_fn: Callable[[np.ndarray, np.ndarray], float],
    aggregation: str = "mean",
    value_kind: str = "risk",
) -> SubpixelGrid:
    """Build a subpixel grid by scoring cropped regions from two aligned arrays."""

    first_array = np.asarray(first)
    second_array = np.asarray(second)
    if first_array.ndim != 2 or second_array.ndim != 2 or first_array.size == 0 or second_array.size == 0:
        raise ValueError("Subpixel pair source arrays must be non-empty 2D arrays")
    if first_array.shape != second_array.shape:
        raise ValueError("Subpixel pair source arrays must have the same shape")
    resolved_spec, bounds = _resolved_spec_for_shape(first_array.shape, spec)
    values = np.zeros((resolved_spec.rows, resolved_spec.columns), dtype=np.float32)
    confidences = np.ones((resolved_spec.rows, resolved_spec.columns), dtype=np.float32)
    for index, (left, top, width, height) in enumerate(bounds):
        row = index // int(resolved_spec.columns)
        column = index % int(resolved_spec.columns)
        first_tile = first_array[top:top + height, left:left + width]
        second_tile = second_array[top:top + height, left:left + width]
        if first_tile.size == 0 or second_tile.size == 0:
            continue
        values[row, column] = float(np.clip(score_fn(first_tile, second_tile), 0.0, 1.0))
    return SubpixelGrid(
        spec=resolved_spec,
        values=values,
        confidences=confidences,
        aggregation=aggregation,
        value_kind=str(value_kind or "risk"),
    )


def aggregate_subpixel_values(values, confidences=None, aggregation: str = "mean") -> float:
    """Aggregate a subpixel matrix to one parent-pixel scalar."""

    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.size == 0:
        return 0.0
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return 0.0
    mode = str(aggregation or "mean").strip().lower()
    if mode == "weighted_mean" and confidences is not None:
        weights = np.asarray(confidences, dtype=np.float32)
        if weights.shape == array.shape:
            weights = np.clip(weights, 0.0, None)
            weight_sum = float(np.sum(weights, dtype=np.float64))
            if weight_sum > 0.0:
                return float(np.sum(array * weights, dtype=np.float64) / weight_sum)
    if mode == "median":
        return float(np.median(finite))
    return float(np.mean(finite, dtype=np.float64))
