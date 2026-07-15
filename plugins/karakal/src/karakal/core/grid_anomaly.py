"""Detect defective cells in regular square-grid model outputs."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - OpenCV is optional at runtime
    cv2 = None


@dataclass(frozen=True, slots=True)
class GridCellAnomaly:
    """One suspicious grid cell in image coordinates."""

    row: int
    column: int
    center_x: float
    center_y: float
    left: int
    top: int
    width: int
    height: int
    score: float
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GridCellAnomalyResult:
    """Frame-level result for regular grid defect detection."""

    cells: tuple[GridCellAnomaly, ...]
    intensity: np.ndarray
    score: float
    component_count: int
    x_axes: tuple[float, ...] = ()
    y_axes: tuple[float, ...] = ()
    cell_width: int = 0
    cell_height: int = 0


def detect_grid_cell_anomalies(image: np.ndarray) -> GridCellAnomalyResult:
    """Find filled, merged, or strongly distorted square cells in a grid output."""

    gray = _normalize_grayscale(image)
    height, width = gray.shape
    empty = np.zeros((height, width), dtype=np.float32)
    if gray.size <= 1 or cv2 is None:
        return GridCellAnomalyResult(tuple(), empty, 0.0, 0)

    foreground = _foreground_mask(gray)
    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(foreground.astype(np.uint8), 8)
    components = _filtered_components(stats, centroids, (height, width))
    normal = _normal_components(components)
    if len(normal) < 8:
        return GridCellAnomalyResult(tuple(), empty, 0.0, len(components))

    median_width = float(np.median([item["width"] for item in normal]))
    median_height = float(np.median([item["height"] for item in normal]))
    median_area = float(np.median([item["area"] for item in normal]))
    median_fill = float(np.median([item["fill"] for item in normal]))
    x_axes = _cluster_axis([item["center_x"] for item in normal], max(6.0, median_width * 0.6))
    y_axes = _cluster_axis([item["center_y"] for item in normal], max(6.0, median_height * 0.6))
    cell_width = int(round(max(8.0, median_width + 8.0)))
    cell_height = int(round(max(8.0, median_height + 8.0)))

    anomalies: dict[tuple[int, int], GridCellAnomaly] = {}
    for component in components:
        score, reasons = _component_anomaly_score(component, median_area, median_fill, median_width, median_height)
        if score <= 0.0:
            continue
        for center_x in _covered_axis_values(component, x_axes, median_width, "x"):
            for center_y in _covered_axis_values(component, y_axes, median_height, "y"):
                column = _axis_index(x_axes, center_x)
                row = _axis_index(y_axes, center_y)
                left = int(round(float(center_x) - cell_width / 2.0))
                top = int(round(float(center_y) - cell_height / 2.0))
                cell = GridCellAnomaly(
                    row=row,
                    column=column,
                    center_x=float(center_x),
                    center_y=float(center_y),
                    left=max(0, min(width - 1, left)),
                    top=max(0, min(height - 1, top)),
                    width=max(1, min(cell_width, width - max(0, min(width - 1, left)))),
                    height=max(1, min(cell_height, height - max(0, min(height - 1, top)))),
                    score=float(np.clip(score, 0.0, 1.0)),
                    reasons=tuple(reasons),
                )
                key = (int(round(cell.center_x)), int(round(cell.center_y)))
                if key not in anomalies or cell.score > anomalies[key].score:
                    anomalies[key] = cell

    cells = tuple(sorted(anomalies.values(), key=lambda item: (item.row, item.column, item.top, item.left)))
    intensity = np.zeros((height, width), dtype=np.float32)
    for cell in cells:
        y0 = max(0, int(cell.top))
        x0 = max(0, int(cell.left))
        y1 = min(height, y0 + int(cell.height))
        x1 = min(width, x0 + int(cell.width))
        if y1 > y0 and x1 > x0:
            intensity[y0:y1, x0:x1] = np.maximum(intensity[y0:y1, x0:x1], float(cell.score))
    score = float(max((cell.score for cell in cells), default=0.0))
    return GridCellAnomalyResult(
        cells=cells,
        intensity=intensity,
        score=score,
        component_count=len(components),
        x_axes=x_axes,
        y_axes=y_axes,
        cell_width=cell_width,
        cell_height=cell_height,
    )


def _normalize_grayscale(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image)
    if values.ndim == 3:
        values = values[..., :3].mean(axis=2)
    if values.ndim != 2:
        return np.zeros((1, 1), dtype=np.uint8)
    if values.size > 0 and float(np.nanmax(values)) <= 1.0:
        values = values * 255.0
    return np.clip(np.nan_to_num(values, nan=0.0, posinf=255.0, neginf=0.0), 0.0, 255.0).astype(np.uint8)


def _foreground_mask(gray: np.ndarray) -> np.ndarray:
    background = float(np.percentile(gray, 50.0))
    threshold = max(150.0, background + 35.0)
    return np.asarray(gray >= threshold, dtype=bool)


def _filtered_components(stats: np.ndarray, centroids: np.ndarray, shape: tuple[int, int]) -> list[dict[str, float]]:
    height, width = shape
    components: list[dict[str, float]] = []
    for index in range(1, int(stats.shape[0])):
        x, y, component_width, component_height, area = [int(value) for value in stats[index]]
        if component_width < 4 or component_height < 4 or area < 12:
            continue
        if component_width > 100 or component_height > 140:
            continue
        if x <= 1 or y <= 1 or x + component_width >= width - 1 or y + component_height >= height - 1:
            continue
        components.append(
            {
                "x": float(x),
                "y": float(y),
                "width": float(component_width),
                "height": float(component_height),
                "area": float(area),
                "center_x": float(centroids[index][0]),
                "center_y": float(centroids[index][1]),
                "fill": float(area / max(1, component_width * component_height)),
            }
        )
    return components


def _normal_components(components: list[dict[str, float]]) -> list[dict[str, float]]:
    return [
        item
        for item in components
        if 8.0 <= item["width"] <= 45.0
        and 8.0 <= item["height"] <= 45.0
        and 40.0 <= item["area"] <= 400.0
        and item["fill"] < 0.45
    ]


def _cluster_axis(values: list[float], tolerance: float) -> tuple[float, ...]:
    clusters: list[list[float]] = []
    for value in sorted(float(item) for item in values):
        if not clusters or abs(value - clusters[-1][-1]) > float(tolerance):
            clusters.append([value])
        else:
            clusters[-1].append(value)
    return tuple(float(sum(cluster) / len(cluster)) for cluster in clusters)


def _component_anomaly_score(
    component: dict[str, float],
    median_area: float,
    median_fill: float,
    median_width: float,
    median_height: float,
) -> tuple[float, tuple[str, ...]]:
    score = 0.0
    reasons: list[str] = []
    fill_threshold = max(0.46, float(median_fill) + 0.17)
    if component["fill"] > fill_threshold:
        score = max(score, (component["fill"] - float(median_fill)) / 0.45)
        reasons.append("filled_cell")
    area_threshold = max(float(median_area) * 1.75, float(median_area) + 170.0)
    if component["area"] > area_threshold:
        score = max(score, component["area"] / max(1.0, float(median_area) * 2.0))
        reasons.append("large_component")
    if component["height"] > float(median_height) * 1.65 or component["width"] > float(median_width) * 1.65:
        score = max(score, 0.8)
        reasons.append("merged_cells")
    return float(score), tuple(reasons)


def _covered_axis_values(component: dict[str, float], axes: tuple[float, ...], median_size: float, axis: str) -> tuple[float, ...]:
    start_key = "x" if axis == "x" else "y"
    size_key = "width" if axis == "x" else "height"
    center_key = "center_x" if axis == "x" else "center_y"
    start = float(component[start_key]) - float(median_size) * 0.7
    stop = float(component[start_key]) + float(component[size_key]) + float(median_size) * 0.7
    covered = tuple(value for value in axes if start <= value <= stop)
    if covered:
        return covered
    return (float(component[center_key]),)


def _axis_index(axes: tuple[float, ...], value: float) -> int:
    if not axes:
        return -1
    distances = [abs(float(axis_value) - float(value)) for axis_value in axes]
    nearest = int(np.argmin(np.asarray(distances, dtype=np.float32)))
    if distances[nearest] <= 16.0:
        return nearest
    return -1
