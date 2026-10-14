"""serializers: cv object polygon."""
from __future__ import annotations
import json
import os
from collections.abc import Iterable
from pathlib import Path
from time import perf_counter
from contour.domain import PolygonData

from contour.serializers_parts.module_state import (
    _CIF_PARSE_CACHE,
    _CIF_PARSE_CACHE_LOCK,
    _CIF_PARSE_CACHE_MAX_ENTRIES,
)

def _note_cif_phase_timing(phase: str, started_at: float) -> None:
    from ..infrastructure.cif_operation_profiler import note_cif_operation_timing

    note_cif_operation_timing(phase, (perf_counter() - started_at) * 1000.0)

def _note_cif_slit_target(*, neighbor_hole: bool) -> None:
    from ..infrastructure.cif_operation_profiler import note_cif_operation_count

    note_cif_operation_count(
        "cif_hole_neighbor_slit" if neighbor_hole else "cif_hole_outer_slit",
    )

def _path_cache_identity(path: str | Path) -> str:
    return os.path.normcase(os.path.abspath(os.path.normpath(os.fspath(path))))

def clear_cif_parse_cache() -> None:
    with _CIF_PARSE_CACHE_LOCK:
        _CIF_PARSE_CACHE.clear()

def invalidate_cif_parse_cache(paths: Iterable[str | Path]) -> None:
    identities = {_path_cache_identity(path) for path in paths}
    if not identities:
        return
    with _CIF_PARSE_CACHE_LOCK:
        for key in list(_CIF_PARSE_CACHE.keys()):
            if key[0] in identities:
                _CIF_PARSE_CACHE.pop(key, None)

def _cif_parse_cache_key(path: Path) -> tuple[str, int, int] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return (_path_cache_identity(path), stat.st_mtime_ns, stat.st_size)

def _cache_cif_parse_result(
    cache_key: tuple[str, int, int],
    image_name: str | None,
    image_size: tuple[int, int] | None,
    polygons: list[PolygonData],
) -> None:
    with _CIF_PARSE_CACHE_LOCK:
        if len(_CIF_PARSE_CACHE) >= _CIF_PARSE_CACHE_MAX_ENTRIES:
            _CIF_PARSE_CACHE.pop(next(iter(_CIF_PARSE_CACHE)))
        _CIF_PARSE_CACHE[cache_key] = (
            image_name,
            image_size,
            [polygon.clone() for polygon in polygons],
        )

def save_polygons_json(
    path: str | Path,
    image_path: str,
    polygons: list[PolygonData],
    metadata: dict[str, object] | None = None,
) -> Path:
    output = Path(path)
    payload = {
        "image_path": image_path,
        "polygon_count": len(polygons),
        "polygons": [polygon.to_dict() for polygon in polygons],
        "metadata": metadata or {},
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return output

def _polygon_bbox_values(polygon: PolygonData) -> tuple[float, float, float, float]:
    if polygon.points:
        x_values = [float(point[0]) for point in polygon.points]
        y_values = [float(point[1]) for point in polygon.points]
        return min(x_values), min(y_values), max(x_values), max(y_values)
    left, top, width, height = polygon.bbox
    return float(left), float(top), float(left + width), float(top + height)

def _cv_coord(value: float) -> int:
    return round(float(value))

def _cv_ring(points: list[tuple[float, float]]) -> list[list[int]]:
    ring = [[_cv_coord(x_coord), _cv_coord(y_coord)] for x_coord, y_coord in points]
    if ring and ring[0] != ring[-1]:
        ring.append(list(ring[0]))
    return ring

def _pixel_box_center_and_size(
    left: float,
    top: float,
    right: float,
    bottom: float,
) -> tuple[int, int, int, int]:
    """Encode integer pixel bounds without moving odd-sized boxes."""

    left_i = _cv_coord(left)
    top_i = _cv_coord(top)
    right_i = _cv_coord(right)
    bottom_i = _cv_coord(bottom)
    width = max(1, right_i - left_i)
    height = max(1, bottom_i - top_i)
    return left_i + width // 2, top_i + height // 2, width, height

def _pixel_box_bounds(center: float, size: float) -> tuple[float, float]:
    """Decode the pixel-bound convention used by CIF/CV box records."""

    size_i = max(1, _cv_coord(size))
    center_i = _cv_coord(center)
    start = center_i - size_i // 2
    return float(start), float(start + size_i)

def _cv_object_from_polygon(polygon: PolygonData, holes: list[PolygonData]) -> dict[str, object]:
    if polygon.shape_hint == "box" or polygon.category == "via":
        left, top, right, bottom = _polygon_bbox_values(polygon)
        center_x, center_y, width, height = _pixel_box_center_and_size(left, top, right, bottom)
        if polygon.category == "via":
            return {
                "type": "Point",
                "id": int(polygon.id),
                "shape": "ellipse",
                "center": [center_x, center_y],
                "diagonals": [width, height],
            }
        return {
            "type": "Point",
            "id": int(polygon.id),
            "shape": "rectangle",
            "coordinates": [_cv_coord(left), _cv_coord(top), _cv_coord(right), _cv_coord(bottom)],
        }

    coordinates = [_cv_ring(polygon.points)]
    coordinates.extend(_cv_ring(hole.points) for hole in holes if len(hole.points) >= 3)
    return {
        "type": "Polygon",
        "id": int(polygon.id),
        "coordinates": coordinates,
    }

def _cv_objects_from_polygons(polygons: list[PolygonData]) -> list[dict[str, object]]:
    sorted_polygons = sorted(polygons, key=lambda item: item.id)
    holes_by_parent: dict[int, list[PolygonData]] = {}
    orphan_holes: list[PolygonData] = []
    for polygon in sorted_polygons:
        if not polygon.is_hole:
            continue
        if polygon.parent_id is None:
            orphan_holes.append(polygon)
        else:
            holes_by_parent.setdefault(int(polygon.parent_id), []).append(polygon)

    objects: list[dict[str, object]] = []
    for polygon in sorted_polygons:
        if polygon.is_hole:
            continue
        objects.append(_cv_object_from_polygon(polygon, holes_by_parent.get(int(polygon.id), [])))
    for hole in orphan_holes:
        clone = hole.clone()
        clone.is_hole = False
        clone.parent_id = None
        objects.append(_cv_object_from_polygon(clone, []))
    return objects

def _cv_json_array(values: list[object], *, indent: int) -> str:
    if all(isinstance(value, int) for value in values):
        return "[" + ", ".join(str(value) for value in values) + "]"
    if all(isinstance(value, list) and all(isinstance(coord, int) for coord in value) for value in values):
        prefix = " " * indent
        inner_prefix = " " * (indent + 2)
        rows = []
        for start in range(0, len(values), 8):
            chunk = values[start : start + 8]
            rows.append(", ".join(_cv_json_array(value, indent=indent + 2) for value in chunk if isinstance(value, list)))
        if len(rows) == 1:
            return "[" + rows[0] + "]"
        tail_rows = [inner_prefix + row for row in rows[1:]]
        return "[" + rows[0] + ",\n" + ",\n".join(tail_rows) + "\n" + prefix + "]"
    return json.dumps(values, ensure_ascii=False, indent=2)

def _cv_json_object(item: dict[str, object], *, indent: int) -> str:
    prefix = " " * indent
    child_prefix = " " * (indent + 2)
    lines = [prefix + "{"]
    entries = list(item.items())
    for index, (key, value) in enumerate(entries):
        suffix = "," if index < len(entries) - 1 else ""
        key_text = json.dumps(str(key), ensure_ascii=False)
        if key == "coordinates" and isinstance(value, list):
            if item.get("type") == "Polygon":
                ring_blocks = [_cv_json_array(ring, indent=indent + 4) for ring in value if isinstance(ring, list)]
                coordinates = "[\n" + ",\n".join(" " * (indent + 4) + block for block in ring_blocks) + "\n" + child_prefix + "]"
            else:
                coordinates = _cv_json_array(value, indent=indent + 2)
            lines.append(f"{child_prefix}{key_text}: {coordinates}{suffix}")
        elif isinstance(value, list):
            lines.append(f"{child_prefix}{key_text}: {_cv_json_array(value, indent=indent + 2)}{suffix}")
        else:
            lines.append(f"{child_prefix}{key_text}: {json.dumps(value, ensure_ascii=False)}{suffix}")
    lines.append(prefix + "}")
    return "\n".join(lines)
