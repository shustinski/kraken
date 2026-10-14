"""serializers: cv point points."""
from __future__ import annotations
import json
from pathlib import Path
from contour.domain import PolygonData, compute_polygon_metrics
from contour.value_conversion import to_int

from contour.serializers_parts.cv_object_polygon import (
    _cv_json_object,
    _cv_objects_from_polygons,
    _pixel_box_bounds,
)

def _dumps_cv_payload(payload: dict[str, object]) -> str:
    objects = payload.get("objects", [])
    lines = ["{"]
    prefix = " " * 2
    lines.append(f'{prefix}"format": {json.dumps(payload["format"], ensure_ascii=False)},')
    lines.append(f'{prefix}"version": {json.dumps(payload["version"], ensure_ascii=False)},')
    lines.append(f'{prefix}"image": {json.dumps(payload["image"], ensure_ascii=False)},')
    lines.append(f'{prefix}"objects": [')
    if isinstance(objects, list):
        object_blocks = [_cv_json_object(item, indent=4) for item in objects if isinstance(item, dict)]
        for index, block in enumerate(object_blocks):
            suffix = "," if index < len(object_blocks) - 1 else ""
            lines.append(block + suffix)
    lines.append(f"{prefix}]")
    lines.append("}")
    return "\n".join(lines) + "\n"

def save_polygons_cv(
    path: str | Path,
    image_path: str,
    polygons: list[PolygonData],
    image_size: tuple[int, int] | None = None,
    metadata: dict[str, object] | None = None,
) -> Path:
    output = Path(path)
    payload: dict[str, object] = {
        "format": "contour-vector",
        "version": 2,
        "image": {
            "path": image_path,
            **({"size": [int(image_size[0]), int(image_size[1])]} if image_size is not None else {}),
        },
        "objects": _cv_objects_from_polygons(polygons),
    }
    del metadata
    output.write_text(_dumps_cv_payload(payload), encoding="utf-8")
    return output

def _as_float_pair(raw: object) -> tuple[float, float]:
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        raise ValueError("Point coordinates must contain at least two numbers")
    return float(raw[0]), float(raw[1])

def _cv_box_points(left: float, top: float, right: float, bottom: float) -> list[tuple[float, float]]:
    return [(left, top), (right, top), (right, bottom), (left, bottom)]

def _cv_point_points(item: dict[str, object]) -> tuple[list[tuple[float, float]], str, str]:
    point_shape = str(item.get("shape") or "").lower()
    raw_coordinates = item.get("coordinates", [])
    if point_shape == "rectangle" or (
        isinstance(raw_coordinates, (list, tuple)) and len(raw_coordinates) >= 4 and point_shape != "ellipse"
    ):
        if not isinstance(raw_coordinates, (list, tuple)) or len(raw_coordinates) < 4:
            raise ValueError("Rectangle point requires [left, top, right, bottom] coordinates")
        left = float(raw_coordinates[0])
        top = float(raw_coordinates[1])
        right = float(raw_coordinates[2])
        bottom = float(raw_coordinates[3])
        return _cv_box_points(left, top, right, bottom), "conductor", "box"

    center_x, center_y = _as_float_pair(item.get("center", raw_coordinates))
    raw_diameters = item.get("diagonals") or item.get("diameters")
    if not isinstance(raw_diameters, (list, tuple)) or len(raw_diameters) < 2:
        raise ValueError("Ellipse point requires diagonals [width, height]")
    left, right = _pixel_box_bounds(center_x, float(raw_diameters[0]))
    top, bottom = _pixel_box_bounds(center_y, float(raw_diameters[1]))
    ellipse_points = _cv_box_points(
        left,
        top,
        right,
        bottom,
    )
    return ellipse_points, "via", "box"

def _cv_polygon_from_points(
    *,
    polygon_id: int,
    points: list[tuple[float, float]],
    is_hole: bool,
    parent_id: int | None,
) -> PolygonData | None:
    if len(points) >= 2 and points[0] == points[-1]:
        points = points[:-1]
    if len(points) < 3:
        return None
    area, perimeter, bbox = compute_polygon_metrics(points)
    return PolygonData(
        id=polygon_id,
        points=points,
        is_hole=is_hole,
        parent_id=parent_id,
        category="conductor",
        shape_hint="polygon",
        area=area,
        perimeter=perimeter,
        bbox=bbox,
    )

def _cv_points_from_ring(raw_points: object) -> list[tuple[float, float]]:
    if not isinstance(raw_points, (list, tuple)):
        return []
    return [
        (float(point[0]), float(point[1]))
        for point in raw_points
        if isinstance(point, (list, tuple)) and len(point) >= 2
    ]

def _cv_object_id(item: dict[str, object], fallback_id: int) -> int:
    return to_int(item.get("id", fallback_id))
