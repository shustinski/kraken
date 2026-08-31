from __future__ import annotations

import csv
import json
import os
import shutil
import threading
from bisect import bisect_left
from collections.abc import Iterable
from pathlib import Path
from time import perf_counter
from xml.sax.saxutils import escape

import cv2
import numpy as np

from .application.processing import DisplaySettings, SaveOptions
from .domain import PolygonData, compute_polygon_metrics, integer_points
from .domain.polygon_ring import collapse_redundant_polyline_vertices
from .i18n import tr
from .infrastructure.cif_primitives import CifBox, CifPrimitive
from .utils import draw_polygon_overlay, ensure_directory, imwrite_unicode_safe


def _note_cif_phase_timing(phase: str, started_at: float) -> None:
    from .infrastructure.cif_operation_profiler import note_cif_operation_timing

    note_cif_operation_timing(phase, (perf_counter() - started_at) * 1000.0)


def _note_cif_slit_target(*, neighbor_hole: bool) -> None:
    from .infrastructure.cif_operation_profiler import note_cif_operation_count

    note_cif_operation_count(
        "cif_hole_neighbor_slit" if neighbor_hole else "cif_hole_outer_slit",
    )

_CIF_PARSE_CACHE: dict[tuple[str, int, int], tuple[str | None, tuple[int, int] | None, list[PolygonData]]] = {}
_CIF_PARSE_CACHE_LOCK = threading.Lock()
_CIF_PARSE_CACHE_MAX_ENTRIES = 64


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
    return int(round(float(value)))


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
            rows.append(", ".join(_cv_json_array(value, indent=indent + 2) for value in chunk))
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
    return int(item.get("id", fallback_id))


def _polygons_from_cv_object(
    item: dict[str, object],
    fallback_id: int,
    generated_id_start: int,
) -> tuple[list[PolygonData], int]:
    geometry_type = str(item.get("type", "")).lower()
    if geometry_type == "point":
        points, default_category, default_shape = _cv_point_points(item)
        area, perimeter, bbox = compute_polygon_metrics(points)
        return [
            PolygonData(
                id=int(item.get("id", fallback_id)),
                points=points,
                is_hole=False,
                parent_id=None,
                category=default_category,
                shape_hint=default_shape,
                area=area,
                perimeter=perimeter,
                bbox=bbox,
            )
        ], generated_id_start
    elif geometry_type == "polygon":
        raw_rings = item.get("coordinates", [])
        if not isinstance(raw_rings, (list, tuple)) or not raw_rings:
            return [], generated_id_start
    else:
        return [], generated_id_start

    parent_id = _cv_object_id(item, fallback_id)
    polygons: list[PolygonData] = []
    outer = _cv_polygon_from_points(
        polygon_id=parent_id,
        points=_cv_points_from_ring(raw_rings[0]),
        is_hole=False,
        parent_id=None,
    )
    if outer is None:
        return [], generated_id_start
    polygons.append(outer)
    next_id = generated_id_start
    for raw_hole in raw_rings[1:]:
        hole = _cv_polygon_from_points(
            polygon_id=next_id,
            points=_cv_points_from_ring(raw_hole),
            is_hole=True,
            parent_id=parent_id,
        )
        if hole is not None:
            polygons.append(hole)
            next_id += 1
    return polygons, next_id


def load_polygons_cv(path: str | Path) -> tuple[str | None, tuple[int, int] | None, list[PolygonData]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("CV vector file must contain an object")
    image_payload = payload.get("image")
    image_name: str | None = None
    image_size: tuple[int, int] | None = None
    if isinstance(image_payload, dict):
        if image_payload.get("path") is not None:
            image_name = str(image_payload.get("path"))
        raw_size = image_payload.get("size")
        if isinstance(raw_size, (list, tuple)) and len(raw_size) >= 2:
            image_size = (int(raw_size[0]), int(raw_size[1]))
    objects = payload.get("objects", [])
    if not isinstance(objects, list):
        raise ValueError("CV vector file objects must be a list")
    polygons: list[PolygonData] = []
    object_items = [item for item in objects if isinstance(item, dict)]
    reserved_ids = {_cv_object_id(item, index + 1) for index, item in enumerate(object_items)}
    next_generated_id = (max(reserved_ids) + 1) if reserved_ids else 1
    for index, item in enumerate(object_items, start=1):
        if not isinstance(item, dict):
            continue
        loaded, next_generated_id = _polygons_from_cv_object(item, index, next_generated_id)
        polygons.extend(loaded)
    return image_name, image_size, polygons


def load_polygons_vector(path: str | Path) -> tuple[str | None, tuple[int, int] | None, list[PolygonData]]:
    vector_path = Path(path)
    if vector_path.suffix.lower() == ".cv":
        return load_polygons_cv(vector_path)
    return load_polygons_cif(vector_path)


CIF_CUTOUT_DISPLAY_MARKER = "( CONTOUR cutout_display );"


def _cif_cutout_display_requested(cif_path: Path) -> bool:
    for line in _read_cif_text(cif_path).splitlines():
        if line.strip() == CIF_CUTOUT_DISPLAY_MARKER.strip():
            return True
    return False


def save_polygons_vector(
    path: str | Path,
    image_path: str,
    polygons: list[PolygonData],
    image_size: tuple[int, int],
    *,
    cutout_display: bool = False,
) -> Path:
    vector_path = Path(path)
    if vector_path.suffix.lower() == ".cv":
        return save_polygons_cv(vector_path, image_path, polygons, image_size=image_size)
    return save_polygons_cif(
        vector_path,
        image_path,
        polygons,
        image_size=image_size,
        cutout_display=cutout_display,
    )


def _parse_cif_int(value: str) -> int:
    normalized = str(value or "").strip().rstrip(";")
    if not normalized:
        raise ValueError(tr("empty_cif_integer_token"))
    return int(normalized)


def _clone_polygon_with_id(polygon: PolygonData, polygon_id: int) -> PolygonData:
    """Copy polygon fields without re-normalizing already-integer vertex coordinates."""

    return PolygonData(
        id=polygon_id,
        points=list(polygon.points),
        is_hole=polygon.is_hole,
        parent_id=polygon.parent_id,
        category=str(polygon.category),
        shape_hint=str(polygon.shape_hint),
        area=float(polygon.area),
        perimeter=float(polygon.perimeter),
        bbox=(int(polygon.bbox[0]), int(polygon.bbox[1]), int(polygon.bbox[2]), int(polygon.bbox[3])),
        reject_reason=str(polygon.reject_reason),
        cif_paint_ring=list(polygon.cif_paint_ring),
    )


def _extract_parenthesized_tokens(line: str) -> list[str]:
    text = line.strip()
    if "(" not in text or ")" not in text:
        return []
    start = text.index("(") + 1
    end = text.rfind(")")
    if end <= start:
        return []
    return text[start:end].replace(";", " ").split()


def _read_cif_text(path: str | Path) -> str:
    cif_path = Path(path)
    payload = cif_path.read_bytes()
    for encoding in ("utf-8", "utf-8-sig", "cp1251", "cp866"):
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError:
            continue
    return payload.decode("cp1251", errors="replace")


def _cif_box_to_polygon(
    box_width: int,
    box_height: int,
    center_x: int,
    center_y: int,
    image_size: tuple[int, int],
    polygon_id: int,
) -> PolygonData:
    _width, height = image_size
    image_center_y = int(height) - center_y
    left, right = _pixel_box_bounds(center_x, box_width)
    top, bottom = _pixel_box_bounds(image_center_y, box_height)
    image_points = [
        (left, top),
        (right, top),
        (right, bottom),
        (left, bottom),
    ]
    area, perimeter, bbox = compute_polygon_metrics(image_points)
    return PolygonData(
        id=polygon_id,
        points=image_points,
        is_hole=False,
        parent_id=None,
        category="via",
        shape_hint="box",
        area=area,
        perimeter=perimeter,
        bbox=bbox,
        _points_normalized=True,
    )


def _cif_polygon_points_to_polygon(
    raw_points: list[tuple[int, int]],
    image_size: tuple[int, int],
    polygon_id: int,
) -> PolygonData | None:
    if len(raw_points) >= 2 and raw_points[0] == raw_points[-1]:
        raw_points = raw_points[:-1]
    if len(raw_points) < 3:
        return None

    _width, height = image_size
    image_height = int(height)
    image_points = [(x_coord, image_height - y_coord) for x_coord, y_coord in raw_points]
    area, perimeter, bbox = compute_polygon_metrics(image_points)
    return PolygonData(
        id=polygon_id,
        points=image_points,
        is_hole=False,
        parent_id=None,
        category="conductor",
        shape_hint="polygon",
        area=area,
        perimeter=perimeter,
        bbox=bbox,
        _points_normalized=True,
    )


def _cif_box_primitive_to_polygon(
    box: CifBox,
    image_size: tuple[int, int],
    polygon_id: int,
) -> PolygonData:
    from .infrastructure.cif_klayout_reader import rotated_box_points

    if box.rotation_x >= 0 and box.rotation_y == 0:
        return _cif_box_to_polygon(
            box.width,
            box.height,
            box.center_x,
            box.center_y,
            image_size,
            polygon_id,
        )

    _width, height = image_size
    image_height = int(height)
    image_points = [
        (float(x_coord), float(image_height - y_coord))
        for x_coord, y_coord in rotated_box_points(
            box.width,
            box.height,
            box.center_x,
            box.center_y,
            box.rotation_x,
            box.rotation_y,
        )
    ]
    area, perimeter, bbox = compute_polygon_metrics(image_points)
    return PolygonData(
        id=polygon_id,
        points=image_points,
        is_hole=False,
        parent_id=None,
        category="via",
        shape_hint="box",
        area=area,
        perimeter=perimeter,
        bbox=bbox,
        _points_normalized=True,
    )


def _cif_primitives_to_polygons(
    cif_path: Path,
    primitives: Iterable[CifPrimitive],
) -> tuple[str | None, tuple[int, int] | None, list[PolygonData]]:
    from .infrastructure.cif_primitives import CifBox, CifComment, CifPolygon

    image_name: str | None = None
    image_size: tuple[int, int] | None = None
    polygons: list[PolygonData] = []

    for primitive in primitives:
        if isinstance(primitive, CifComment):
            tokens = _extract_parenthesized_tokens(primitive.content.strip())
            if len(tokens) >= 2 and tokens[0] == "R":
                image_name = tokens[1]
            elif len(tokens) >= 3 and tokens[0] == "S":
                image_size = (_parse_cif_int(tokens[1]), _parse_cif_int(tokens[2]))
            continue
        if isinstance(primitive, CifBox):
            if image_size is None:
                raise ValueError(tr("cif_size_header_missing", path=cif_path))
            polygons.append(
                _cif_box_primitive_to_polygon(
                    primitive,
                    image_size,
                    len(polygons) + 1,
                )
            )
            continue
        if isinstance(primitive, CifPolygon):
            if image_size is None:
                raise ValueError(tr("cif_size_header_missing", path=cif_path))
            polygon = _cif_polygon_points_to_polygon(
                list(primitive.points),
                image_size,
                len(polygons) + 1,
            )
            if polygon is not None:
                polygons.append(polygon)

    return image_name, image_size, polygons


def _load_polygons_cif_via_klayout(
    cif_path: Path,
) -> tuple[str | None, tuple[int, int] | None, list[PolygonData]] | None:
    from .infrastructure.cif_klayout_reader import klayout_cif_reader_enabled, load_cif_primitives_klayout

    if not klayout_cif_reader_enabled():
        return None

    parsed = load_cif_primitives_klayout(cif_path)
    return _cif_primitives_to_polygons(cif_path, parsed.primitives)


def _load_polygons_cif_via_opencif(
    cif_path: Path,
) -> tuple[str | None, tuple[int, int] | None, list[PolygonData]] | None:
    from .infrastructure.cif_opencif import (
        load_cif_primitives,
        opencif_loader_available,
        opencif_use_enabled,
    )

    if not opencif_use_enabled() or not opencif_loader_available():
        return None

    parsed = load_cif_primitives(cif_path)
    if parsed is None:
        return None
    if parsed.status == "cant_open":
        return None

    return _cif_primitives_to_polygons(cif_path, parsed.primitives)


def load_polygons_cif(path: str | Path) -> tuple[str | None, tuple[int, int] | None, list[PolygonData]]:
    cif_path = Path(path)
    cache_key = _cif_parse_cache_key(cif_path)
    if cache_key is not None:
        with _CIF_PARSE_CACHE_LOCK:
            cached = _CIF_PARSE_CACHE.get(cache_key)
        if cached is not None:
            image_name, image_size, polygons = cached
            return image_name, image_size, [polygon.clone() for polygon in polygons]

    klayout_payload = _load_polygons_cif_via_klayout(cif_path)
    if klayout_payload is not None:
        image_name, image_size, polygons = klayout_payload
        cutout_display = _cif_cutout_display_requested(cif_path)
        polygons = _recover_cut_hole_topology(polygons, image_size, cutout_display=cutout_display)
        _ensure_cif_paint_rings(polygons)
        if cache_key is not None:
            _cache_cif_parse_result(cache_key, image_name, image_size, polygons)
        return image_name, image_size, polygons

    opencif_payload = _load_polygons_cif_via_opencif(cif_path)
    if opencif_payload is not None:
        image_name, image_size, polygons = opencif_payload
        cutout_display = _cif_cutout_display_requested(cif_path)
        polygons = _recover_cut_hole_topology(polygons, image_size, cutout_display=cutout_display)
        _ensure_cif_paint_rings(polygons)
        if cache_key is not None:
            _cache_cif_parse_result(cache_key, image_name, image_size, polygons)
        return image_name, image_size, polygons

    lines = _read_cif_text(cif_path).splitlines()

    image_name: str | None = None
    image_size: tuple[int, int] | None = None
    polygons: list[PolygonData] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("( R "):
            tokens = _extract_parenthesized_tokens(stripped)
            if len(tokens) >= 2 and tokens[0] == "R":
                image_name = tokens[1]
            continue
        if stripped.startswith("( S "):
            tokens = _extract_parenthesized_tokens(stripped)
            if len(tokens) >= 3 and tokens[0] == "S":
                image_size = (_parse_cif_int(tokens[1]), _parse_cif_int(tokens[2]))
            continue
        if stripped.startswith("B "):
            if image_size is None:
                raise ValueError(tr("cif_size_header_missing", path=cif_path))
            payload = stripped[2:].rstrip(";").split()
            if len(payload) != 4:
                continue
            polygons.append(
                _cif_box_to_polygon(
                    _parse_cif_int(payload[0]),
                    _parse_cif_int(payload[1]),
                    _parse_cif_int(payload[2]),
                    _parse_cif_int(payload[3]),
                    image_size,
                    len(polygons) + 1,
                )
            )
            continue
        if not stripped.startswith("P "):
            continue
        if image_size is None:
            raise ValueError(tr("cif_size_header_missing", path=cif_path))

        payload = stripped[2:].rstrip(";").split()
        if len(payload) < 6 or len(payload) % 2 != 0:
            continue

        raw_points = [
            (_parse_cif_int(payload[index]), _parse_cif_int(payload[index + 1])) for index in range(0, len(payload), 2)
        ]
        polygon = _cif_polygon_points_to_polygon(raw_points, image_size, len(polygons) + 1)
        if polygon is not None:
            polygons.append(polygon)

    cutout_display = _cif_cutout_display_requested(cif_path)
    polygons = _recover_cut_hole_topology(polygons, image_size, cutout_display=cutout_display)
    _ensure_cif_paint_rings(polygons)
    if cache_key is not None:
        _cache_cif_parse_result(cache_key, image_name, image_size, polygons)
    return image_name, image_size, polygons


def _polygon_to_cif_line(polygon: PolygonData, image_width: int, image_height: int) -> str:
    if polygon.shape_hint == "box":
        x_values = [point[0] for point in polygon.points]
        y_values = [point[1] for point in polygon.points]
        if len(x_values) < 4 or len(y_values) < 4:
            return ""
        center_x, center_y, width, height = _pixel_box_center_and_size(
            min(x_values),
            min(y_values),
            max(x_values),
            max(y_values),
        )
        cif_x = max(0, min(image_width, center_x))
        cif_y = max(0, min(image_height, round(image_height - center_y)))
        return f"B {width} {height} {cif_x} {cif_y};"
    points = []
    for x_coord, y_coord in polygon.points:
        cif_x = max(0, min(image_width, round(x_coord)))
        cif_y = max(0, min(image_height, round(image_height - y_coord)))
        points.append((cif_x, cif_y))
    if len(points) < 3:
        return ""
    if points[0] != points[-1]:
        points.append(points[0])
    coordinates = " ".join(f"{x_coord} {y_coord}" for x_coord, y_coord in points)
    return f"P {coordinates};"


def _to_local_int_points(points: list[tuple[float, float]], left: int, top: int) -> np.ndarray:
    return np.array(
        [[round(x_coord - left), round(y_coord - top)] for x_coord, y_coord in points],
        dtype=np.int32,
    )


def _dedupe_closed_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not points:
        return []
    first_x, first_y = points[0]
    if isinstance(first_x, int) and isinstance(first_y, int):
        source: list[tuple[float, float]] = [(int(x_coord), int(y_coord)) for x_coord, y_coord in points]
    else:
        source = [(float(x_coord), float(y_coord)) for x_coord, y_coord in points]
    cleaned: list[tuple[float, float]] = [source[0]]
    for point in source[1:]:
        if point != cleaned[-1]:
            cleaned.append(point)
    while len(cleaned) >= 2 and cleaned[0] == cleaned[-1]:
        cleaned = cleaned[:-1]
    return cleaned


def _ring_has_retracing_entry_edge(ring: list[tuple[float, float]]) -> bool:
    """True when closing the ring immediately retraces its first edge.

    Legacy CIF keyholes sometimes repeat the bridge vertex, so a zero-length
    ``A -> A`` pair is treated as the hole mouth and the real bridge is swallowed
    into the hole outline as a long spike.
    """

    return len(ring) >= 3 and ring[-1] == ring[1]


def _is_simple_extracted_hole_ring(ring: list[tuple[float, float]]) -> bool:
    """Reject nested keyholes that still contain their own bridge vertices."""

    return len(ring) >= 3 and not _ring_has_retracing_entry_edge(ring) and not _has_duplicate_points(ring)


def _has_duplicate_points(points: list[tuple[float, float]]) -> bool:
    if len(points) < 2:
        return False
    seen: set[tuple[float, float]] = set()
    for point in points:
        if point in seen:
            return True
        seen.add(point)
    return False


def _rightmost_point_index(points: list[tuple[float, float]]) -> int:
    return max(range(len(points)), key=lambda index: (points[index][0], -points[index][1]))


_BOUNDARY_SNAP_NEIGHBOR_WINDOW = 8


def _point_distance_squared(
    left: tuple[float, float],
    right: tuple[float, float],
) -> float:
    delta_x = float(left[0]) - float(right[0])
    delta_y = float(left[1]) - float(right[1])
    return delta_x * delta_x + delta_y * delta_y


def _boundary_vertices_by_distance(
    boundary: list[tuple[float, float]],
    hole_start: tuple[float, float],
    *,
    neighbor_window: int = _BOUNDARY_SNAP_NEIGHBOR_WINDOW,
    prefer_same_x_side: bool = True,
) -> list[tuple[tuple[float, float], int]]:
    """Rank boundary vertices by distance; bisect window first, then the rest."""

    vertex_count = len(boundary)
    if vertex_count < 3:
        return []

    hole_x = float(hole_start[0])
    indexed_boundary = sorted(enumerate(boundary), key=lambda item: float(item[1][0]))
    x_coords = [float(point[0]) for _, point in indexed_boundary]
    anchor_index = bisect_left(x_coords, hole_x)
    window = max(1, int(neighbor_window))
    start = max(0, anchor_index - window)
    end = min(len(indexed_boundary), anchor_index + window + 1)
    window_pairs = [(boundary[index], index) for index, _point in indexed_boundary[start:end]]
    window_keys = {index for index, _point in indexed_boundary[start:end]}

    def _maybe_prioritize_same_x_side(
        pairs: list[tuple[tuple[float, float], int]],
    ) -> list[tuple[tuple[float, float], int]]:
        if not prefer_same_x_side:
            return pairs
        same_x_side = [item for item in pairs if float(item[0][0]) + 1e-9 >= hole_x]
        if same_x_side:
            return same_x_side
        return pairs

    window_ranked = sorted(
        _maybe_prioritize_same_x_side(window_pairs),
        key=lambda item: _point_distance_squared(hole_start, item[0]),
    )
    remaining_pairs = [(boundary[index], index) for index in range(vertex_count) if index not in window_keys]
    remaining_ranked = sorted(
        _maybe_prioritize_same_x_side(remaining_pairs),
        key=lambda item: _point_distance_squared(hole_start, item[0]),
    )
    return window_ranked + remaining_ranked


def _snap_hole_to_boundary_vertex(
    boundary: list[tuple[float, float]],
    hole_start: tuple[float, float],
    *,
    neighbor_window: int = _BOUNDARY_SNAP_NEIGHBOR_WINDOW,
) -> tuple[tuple[float, float], int] | None:
    """Return the nearest boundary vertex candidate (without material coverage check)."""

    ranked = _boundary_vertices_by_distance(
        boundary,
        hole_start,
        neighbor_window=neighbor_window,
    )
    if not ranked:
        return None
    outer_anchor, boundary_index = ranked[0]
    return outer_anchor, boundary_index


def _select_vertex_slit_for_hole(
    hole_points: list[tuple[float, float]],
    parent_solid,
    *,
    outer: list[tuple[float, float]],
    linked_hole_rings: list[tuple[int, list[tuple[float, float]]]] | None = None,
) -> tuple[int, tuple[float, float], int, int, list[tuple[float, float]]] | None:
    """Pick the shortest vertex-to-vertex slit that stays inside parent material.

    Targets include the outer ring and any already-linked sibling holes.
    Returns ``(target_ring_key, target_anchor, insert_index, hole_vertex_index, slit_path)``.
    ``target_ring_key`` is ``-1`` for the outer ring, otherwise the encoded-hole index.
    """

    from shapely.geometry import LineString

    targets: list[tuple[int, list[tuple[float, float]]]] = [(-1, outer)]
    if linked_hole_rings:
        targets.extend(linked_hole_rings)

    best: tuple[float, int, tuple[float, float], int, int, list[tuple[float, float]]] | None = None
    anchor_index = _rightmost_point_index(hole_points)
    hole_vertex_indices = [anchor_index] + [
        index for index in range(len(hole_points)) if index != anchor_index
    ]
    for hole_index in hole_vertex_indices:
        hole_anchor = hole_points[hole_index]
        for target_key, target_ring in targets:
            for target_anchor, insert_index in _boundary_vertices_by_distance(
                target_ring,
                hole_anchor,
                prefer_same_x_side=False,
            ):
                distance_squared = _point_distance_squared(hole_anchor, target_anchor)
                if best is not None and distance_squared >= best[0]:
                    continue
                if not parent_solid.covers(LineString([target_anchor, hole_anchor])):
                    continue
                best = (
                    distance_squared,
                    target_key,
                    target_anchor,
                    insert_index,
                    hole_index,
                    [target_anchor, hole_anchor],
                )
    if best is None:
        return None
    _distance_squared, target_key, target_anchor, insert_index, hole_index, slit_path = best
    return target_key, target_anchor, insert_index, hole_index, slit_path


_HoleLinkDetour = tuple[
    float,
    tuple[float, float],
    list[tuple[float, float]],
    int,
    int,
]


def _trace_linked_hole_cycle(
    path: list[tuple[float, float]],
    hole_points: list[tuple[float, float]],
    *,
    hole_index: int,
    start_vertex_index: int,
    detours_by_ring: dict[int, dict[int, list[_HoleLinkDetour]]],
    encoded_holes: list[tuple[PolygonData, list[tuple[float, float]]]],
) -> None:
    """Walk one closed hole ring, expanding nested keyhole detours along the way."""

    vertex_count = len(hole_points)
    for step in range(vertex_count):
        edge_index = (start_vertex_index + step) % vertex_count
        detours = sorted(
            detours_by_ring.get(hole_index, {}).get(edge_index, ()),
            key=lambda item: item[0],
        )
        for _parameter, anchor, slit_path, child_index, child_vertex_index in detours:
            if path[-1] != anchor:
                path.append(anchor)
            child_points = encoded_holes[child_index][1]
            path.extend(slit_path[1:])
            _trace_linked_hole_cycle(
                path,
                child_points,
                hole_index=child_index,
                start_vertex_index=child_vertex_index,
                detours_by_ring=detours_by_ring,
                encoded_holes=encoded_holes,
            )
            path.append(child_points[child_vertex_index])
            path.extend(reversed(slit_path[:-1]))
        if step >= vertex_count - 1:
            continue
        next_vertex = hole_points[(edge_index + 1) % vertex_count]
        if path[-1] != next_vertex:
            path.append(next_vertex)


def _rotate_open_ring(points: list[tuple[float, float]], start_index: int) -> list[tuple[float, float]]:
    return points[start_index:] + points[:start_index]


def _signed_ring_area(points: list[tuple[float, float]]) -> float:
    area_twice = 0.0
    for index, (x_coord, y_coord) in enumerate(points):
        next_x, next_y = points[(index + 1) % len(points)]
        area_twice += float(x_coord) * float(next_y) - float(next_x) * float(y_coord)
    return area_twice * 0.5


def _ring_with_orientation(
    points: list[tuple[float, float]],
    *,
    positive_area: bool,
) -> list[tuple[float, float]]:
    ring = _dedupe_closed_points(points)
    if len(ring) < 3:
        return ring
    if (_signed_ring_area(ring) > 0.0) != positive_area:
        ring.reverse()
    return ring


def _edge_parameter(
    start: tuple[float, float],
    end: tuple[float, float],
    point: tuple[float, float],
) -> float:
    delta_x = float(end[0]) - float(start[0])
    delta_y = float(end[1]) - float(start[1])
    if abs(delta_x) >= abs(delta_y) and abs(delta_x) > 1e-12:
        return (float(point[0]) - float(start[0])) / delta_x
    if abs(delta_y) > 1e-12:
        return (float(point[1]) - float(start[1])) / delta_y
    return 0.0


def _encode_parent_with_holes_link_path(parent: PolygonData, holes: list[PolygonData]) -> list[tuple[float, float]]:
    encode_started_at = perf_counter()
    # Image Y grows downwards and CIF Y grows upwards. A positive image-space
    # winding therefore becomes clockwise in CIF; holes use the opposite winding.
    outer = _ring_with_orientation(parent.points, positive_area=True)
    if len(outer) < 3:
        return parent.points

    encoded_holes = [
        (hole, _ring_with_orientation(hole.points, positive_area=False))
        for hole in holes
    ]
    encoded_holes = [(hole, points) for hole, points in encoded_holes if len(points) >= 3]
    encoded_holes.sort(
        key=lambda item: -float(item[1][_rightmost_point_index(item[1])][0]),
    )

    from shapely import make_valid
    from shapely.geometry import Polygon as ShapelyPolygon

    parent_solid = make_valid(
        ShapelyPolygon(
            outer,
            holes=[points for _hole, points in encoded_holes],
        )
    )
    detours_by_ring: dict[int, dict[int, list[_HoleLinkDetour]]] = {}
    linked_hole_indices: set[int] = set()
    for hole_index, (hole, hole_points) in enumerate(encoded_holes):
        slit_started_at = perf_counter()
        linked_hole_rings = [
            (linked_index, encoded_holes[linked_index][1])
            for linked_index in sorted(linked_hole_indices)
        ]
        selected = _select_vertex_slit_for_hole(
            hole_points,
            parent_solid,
            outer=outer,
            linked_hole_rings=linked_hole_rings,
        )
        _note_cif_phase_timing("cif_hole_vertex_slit_search", slit_started_at)
        if selected is None:
            raise ValueError(f"Cannot encode CIF hole {hole.id}: hole is disconnected from parent {parent.id}")
        target_ring_key, target_anchor, insert_index, anchor_index, slit_path = selected
        _note_cif_slit_target(neighbor_hole=target_ring_key >= 0)
        target_ring = outer if target_ring_key < 0 else encoded_holes[target_ring_key][1]
        edge_index = (insert_index - 1) % len(target_ring)
        edge_start = target_ring[edge_index]
        edge_end = target_ring[(edge_index + 1) % len(target_ring)]
        parameter = _edge_parameter(edge_start, edge_end, target_anchor)
        detours_by_ring.setdefault(target_ring_key, {}).setdefault(edge_index, []).append(
            (parameter, target_anchor, slit_path, hole_index, anchor_index)
        )
        linked_hole_indices.add(hole_index)

    path = [outer[0]]
    canonical_start_index: int | None = None
    for edge_index, _edge_start in enumerate(outer):
        detours = sorted(
            detours_by_ring.get(-1, {}).get(edge_index, ()),
            key=lambda item: item[0],
        )
        for _parameter, outer_anchor, slit_path, child_index, child_vertex_index in detours:
            if path[-1] != outer_anchor:
                path.append(outer_anchor)
            child_points = encoded_holes[child_index][1]
            path.extend(slit_path[1:])
            _trace_linked_hole_cycle(
                path,
                child_points,
                hole_index=child_index,
                start_vertex_index=child_vertex_index,
                detours_by_ring=detours_by_ring,
                encoded_holes=encoded_holes,
            )
            path.append(child_points[child_vertex_index])
            path.extend(reversed(slit_path[:-1]))
            if canonical_start_index is None:
                canonical_start_index = len(path) - 1
        edge_end = outer[(edge_index + 1) % len(outer)]
        if path[-1] != edge_end:
            path.append(edge_end)
    if canonical_start_index is not None:
        open_path = path[:-1] if len(path) >= 2 and path[0] == path[-1] else path
        path = _rotate_open_ring(open_path, canonical_start_index)
    _note_cif_phase_timing("cif_hole_link_encode", encode_started_at)
    return path


def _contour_points_to_polygon(
    contour: np.ndarray,
    *,
    left: int,
    top: int,
    polygon_id: int,
    is_hole: bool,
    parent_id: int | None,
) -> PolygonData:
    points = [(float(point[0][0] + left), float(point[0][1] + top)) for point in contour]
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


def _split_linked_polygon_rings(
    points: list[tuple[float, float]],
) -> tuple[list[tuple[float, float]], list[list[tuple[float, float]]]]:
    open_points = _dedupe_closed_points(points)
    if not _has_duplicate_points(open_points):
        return open_points, []
    point_count = len(open_points)
    bridge_return_indices: dict[tuple[tuple[float, float], tuple[float, float]], list[int]] = {}
    for return_index in range(1, point_count - 1):
        bridge_return_indices.setdefault(
            (open_points[return_index], open_points[return_index + 1]),
            [],
        ).append(return_index)
    best_span: int | None = None
    best_hole: list[tuple[float, float]] | None = None
    best_outer: list[tuple[float, float]] | None = None
    for start_index in range(0, point_count - 3):
        outer_anchor = open_points[start_index]
        hole_anchor = open_points[start_index + 1]
        if outer_anchor == hole_anchor:
            continue
        for return_index in bridge_return_indices.get((hole_anchor, outer_anchor), ()):
            if return_index < start_index + 3:
                continue
            span = return_index - start_index
            if best_span is not None and span >= best_span:
                continue
            # Spike-clean the hole so A->B->A bridge leftovers do not look like
            # nested keyholes. Do not collinear-collapse the outer yet: that can
            # delete still-needed bridge anchors (e.g. right-edge multi-hole CIF).
            hole_ring = _remove_out_and_back_spikes(
                _dedupe_closed_points(open_points[start_index + 1 : return_index + 1])
            )
            outer_ring = _dedupe_closed_points(
                [outer_anchor, *open_points[return_index + 2 :], *open_points[: start_index + 1]]
            )
            if len(hole_ring) < 3 or len(outer_ring) < 3:
                continue
            hole_area, _, _ = compute_polygon_metrics(hole_ring)
            outer_area, _, _ = compute_polygon_metrics(outer_ring)
            if hole_area > outer_area:
                hole_ring, outer_ring = outer_ring, hole_ring
            if not _is_simple_extracted_hole_ring(hole_ring):
                continue
            best_span = span
            best_hole = hole_ring
            best_outer = outer_ring
    if best_hole is None or best_outer is None:
        return open_points, []
    base_outer, base_holes = _split_linked_polygon_rings(best_outer)
    return _collapse_recovered_outline(base_outer), [
        _collapse_recovered_outline(hole) for hole in (best_hole, *base_holes)
    ]


def _remove_out_and_back_spikes(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Drop zero-area spikes ``A -> B -> A`` that leftover CIF bridges often leave behind."""

    if len(points) < 3:
        return list(points)
    open_points = _dedupe_closed_points(points)
    stack: list[tuple[float, float]] = []
    for point in open_points:
        if len(stack) >= 2 and stack[-2] == point:
            stack.pop()
            continue
        stack.append(point)
    # Closing the ring can also leave a wrap-around spike: last -> first -> second.
    while len(stack) >= 3 and stack[-1] == stack[1]:
        stack.pop(0)
        stack.pop()
    while len(stack) >= 3 and stack[0] == stack[-2]:
        stack.pop()
        stack.pop(0)
    return stack if len(stack) >= 3 else list(points)


def _collapse_recovered_outline(ring: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Drop leftover keyhole bridges, out-and-back spikes, and collinear stubs."""

    cleaned = _remove_out_and_back_spikes(ring)
    collapsed = collapse_redundant_polyline_vertices(cleaned, closed=True, min_vertices=3)
    return collapsed if len(collapsed) >= 3 else cleaned if len(cleaned) >= 3 else ring


def _cif_paint_mask_from_ring(
    ring: list[tuple[float, float]],
) -> tuple[np.ndarray, int, int]:
    """Rasterize a CIF ``P`` ring with Qt WindingFill (KLayout viewer semantics)."""

    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QBrush, QColor, QImage, QPainter, QPainterPath

    int_points = integer_points(_dedupe_closed_points(ring))
    if len(int_points) < 3:
        return np.zeros((1, 1), dtype=np.uint8), 0, 0
    xs = [point[0] for point in int_points]
    ys = [point[1] for point in int_points]
    left = int(min(xs))
    top = int(min(ys))
    width = max(1, int(max(xs)) - left + 1)
    height = max(1, int(max(ys)) - top + 1)
    path = QPainterPath()
    path.moveTo(float(int_points[0][0] - left), float(int_points[0][1] - top))
    for x_coord, y_coord in int_points[1:]:
        path.lineTo(float(x_coord - left), float(y_coord - top))
    path.closeSubpath()
    path.setFillRule(Qt.FillRule.WindingFill)
    image = QImage(width, height, QImage.Format.Format_Grayscale8)
    image.fill(0)
    painter = QPainter(image)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.fillPath(path, QBrush(QColor(255, 255, 255)))
    finally:
        painter.end()
    mask = np.frombuffer(
        bytes(image.constBits().asarray(image.sizeInBytes())),
        dtype=np.uint8,
    ).reshape((height, image.bytesPerLine()))[:, :width].copy()
    return mask, left, top


def _stamp_cif_paint_ring_on_mask(
    mask: np.ndarray,
    ring: list[tuple[float, float]],
    origin: tuple[int, int],
    *,
    value: int = 255,
) -> None:
    """Paint a CIF ring into ``mask`` using Qt WindingFill (KLayout viewer semantics)."""

    paint_mask, left, top = _cif_paint_mask_from_ring(ring)
    if paint_mask.size == 0 or not np.any(paint_mask):
        return
    x_offset = left - int(origin[0])
    y_offset = top - int(origin[1])
    height, width = paint_mask.shape
    target_x_start = max(0, x_offset)
    target_y_start = max(0, y_offset)
    target_x_stop = min(mask.shape[1], x_offset + width)
    target_y_stop = min(mask.shape[0], y_offset + height)
    if target_x_start >= target_x_stop or target_y_start >= target_y_stop:
        return

    paint_x_start = target_x_start - x_offset
    paint_y_start = target_y_start - y_offset
    paint_x_stop = paint_x_start + (target_x_stop - target_x_start)
    paint_y_stop = paint_y_start + (target_y_stop - target_y_start)
    target = mask[target_y_start:target_y_stop, target_x_start:target_x_stop]
    painted = paint_mask[paint_y_start:paint_y_stop, paint_x_start:paint_x_stop] > 0
    if int(value) == 0:
        target[painted] = 0
        return
    target[painted] = int(value)


def _polygon_uses_authored_cif_paint_ring(polygon: PolygonData) -> bool:
    """True when display must use the authored CIF ring with WindingFill (keyhole split)."""

    if polygon.is_hole or not polygon.cif_paint_ring:
        return False
    return integer_points(polygon.cif_paint_ring) != integer_points(polygon.points)


def _ensure_cif_paint_rings(polygons: list[PolygonData]) -> None:
    """No-op; authored rings are attached during keyhole recovery only."""

    del polygons


def _render_cif_klayout_layer_mask(
    image_size: tuple[int, int],
    polygons: list[PolygonData],
) -> np.ndarray:
    """Rasterize loaded polygons the way KLayout paints independent CIF ``P`` commands."""

    image_width, image_height = image_size
    mask = np.zeros((int(image_height), int(image_width)), dtype=np.uint8)
    for polygon in polygons:
        if polygon.is_hole:
            continue
        ring = polygon.cif_paint_ring if _polygon_uses_authored_cif_paint_ring(polygon) else polygon.points
        _stamp_cif_paint_ring_on_mask(mask, ring, (0, 0), value=255)
    return mask


def _legacy_cut_hole_recovery_enabled() -> bool:
    return str(os.environ.get("CONTOUR_CIF_RECOVER_LEGACY_CUT_HOLES", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _nested_outer_punch_on_load_enabled() -> bool:
    """Opt-in only: punch nested outers after CIF load.

    Default is off so filled coverage matches KLayout / CIF semantics (each ``P``
    is painted independently; overlaps OR together; no cross-polygon subtraction).
    Enable via ``CONTOUR_CIF_PUNCH_NESTED_OUTERS`` when an empty window under a
    nested island is required for editing.
    """

    return str(os.environ.get("CONTOUR_CIF_PUNCH_NESTED_OUTERS", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _ring_coords_from_shapely(coords) -> list[tuple[float, float]]:
    points = [(float(x_coord), float(y_coord)) for x_coord, y_coord in list(coords)[:-1]]
    points = integer_points(points)
    collapsed = collapse_redundant_polyline_vertices(points, closed=True, min_vertices=3)
    return collapsed if len(collapsed) >= 3 else points


def _normalize_cif_polygon_families(
    points: list[tuple[float, float]],
) -> list[tuple[list[tuple[float, float]], list[list[tuple[float, float]]]]] | None:
    """Normalize a CIF ``P`` ring to outer+holes via make_valid (KLayout-like fill).

    Heuristic keyhole splitting can invent extra holes and change filled area. Shapely
    ``make_valid`` matches the geometric fill of self-touching CIF polygons that layout
    viewers normalize when rendering.
    """

    from shapely import make_valid
    from shapely.geometry import GeometryCollection, MultiPolygon
    from shapely.geometry import Polygon as ShapelyPolygon

    if len(points) < 3:
        return None
    shell = [(float(x_coord), float(y_coord)) for x_coord, y_coord in points]
    geom = make_valid(ShapelyPolygon(shell))
    if geom.is_empty:
        return None

    polygons: list = []

    def _collect(part) -> None:
        if part.is_empty:
            return
        if isinstance(part, ShapelyPolygon):
            polygons.append(part)
        elif isinstance(part, MultiPolygon):
            polygons.extend(child for child in part.geoms if isinstance(child, ShapelyPolygon))
        elif isinstance(part, GeometryCollection):
            for child in part.geoms:
                _collect(child)

    _collect(geom)
    if not polygons:
        return None

    families: list[tuple[list[tuple[float, float]], list[list[tuple[float, float]]]]] = []
    for poly in polygons:
        if float(poly.area) <= 1.0:
            continue
        outer = _ring_coords_from_shapely(poly.exterior.coords)
        if len(outer) < 3:
            continue
        holes: list[list[tuple[float, float]]] = []
        for interior in poly.interiors:
            hole = _ring_coords_from_shapely(interior.coords)
            if len(hole) >= 3:
                holes.append(hole)
        families.append((outer, holes))
    return families or None


def _recover_cut_hole_topology(
    polygons: list[PolygonData],
    image_size: tuple[int, int] | None,
    *,
    cutout_display: bool = False,
) -> list[PolygonData]:
    from .application.fix_internal_contours import should_use_cutout_display_for_keyhole_family

    if image_size is None:
        return polygons
    image_width, image_height = image_size
    del image_height
    recovered: list[PolygonData] = []
    next_id = 1
    recover_legacy_cut_holes = _legacy_cut_hole_recovery_enabled()
    for polygon in polygons:
        if polygon.shape_hint == "box" or polygon.category == "via" or len(polygon.points) < 3:
            recovered.append(_clone_polygon_with_id(polygon, next_id))
            next_id += 1
            continue
        open_points = _dedupe_closed_points(polygon.points)
        if not _has_duplicate_points(open_points):
            recovered.append(_clone_polygon_with_id(polygon, next_id))
            next_id += 1
            continue

        families = _normalize_cif_polygon_families(polygon.points)
        linked_outer, linked_holes = _split_linked_polygon_rings(polygon.points)
        # Prefer make_valid when it yields one outer with holes (KLayout-like fill).
        # Fall back to keyhole split when make_valid shatters a linked bridge into
        # separate solids and loses hole topology (right-edge multi-hole CIF).
        if families and len(families) == 1 and families[0][1]:
            pass
        elif linked_holes:
            families = [(linked_outer, linked_holes)]
        elif not families:
            families = []

        if families:
            authored_paint_ring = integer_points(polygon.points)
            for outer_points, hole_rings in families:
                use_cutout_display = bool(cutout_display) or should_use_cutout_display_for_keyhole_family(
                    authored_paint_ring,
                    outer_points,
                    hole_rings,
                    image_size,
                )
                paint_ring = integer_points(outer_points) if use_cutout_display else authored_paint_ring
                area, perimeter, bbox = compute_polygon_metrics(outer_points)
                parent_id = next_id
                recovered.append(
                    PolygonData(
                        id=parent_id,
                        points=outer_points,
                        is_hole=False,
                        parent_id=None,
                        category=polygon.category,
                        shape_hint=polygon.shape_hint,
                        area=area,
                        perimeter=perimeter,
                        bbox=bbox,
                        cif_paint_ring=[(float(x_coord), float(y_coord)) for x_coord, y_coord in paint_ring],
                    )
                )
                next_id += 1
                for hole_points in hole_rings:
                    area, perimeter, bbox = compute_polygon_metrics(hole_points)
                    recovered.append(
                        PolygonData(
                            id=next_id,
                            points=hole_points,
                            is_hole=True,
                            parent_id=parent_id,
                            category=polygon.category,
                            shape_hint=polygon.shape_hint,
                            area=area,
                            perimeter=perimeter,
                            bbox=bbox,
                        )
                    )
                    next_id += 1
            continue
        if not recover_legacy_cut_holes:
            recovered.append(_clone_polygon_with_id(polygon, next_id))
            next_id += 1
            continue
        left = max(0, int(np.floor(min(point[0] for point in polygon.points))) - 3)
        top = max(0, int(np.floor(min(point[1] for point in polygon.points))) - 3)
        right = min(image_width - 1, int(np.ceil(max(point[0] for point in polygon.points))) + 3)
        bottom = int(np.ceil(max(point[1] for point in polygon.points))) + 3
        local_width = max(1, right - left + 1)
        local_height = max(1, bottom - top + 1)
        raw_mask = np.zeros((local_height, local_width), dtype=np.uint8)
        local_points = _to_local_int_points(polygon.points, left, top)
        cv2.fillPoly(raw_mask, [local_points], 255)
        _contours_raw, hierarchy_raw = cv2.findContours(raw_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        raw_hole_count = 0
        if hierarchy_raw is not None and len(hierarchy_raw) > 0:
            raw_hole_count = int(sum(1 for item in hierarchy_raw[0] if int(item[3]) >= 0))
        best_contours: list[np.ndarray] = []
        best_hierarchy: np.ndarray | None = None
        best_hole_count = raw_hole_count
        for kernel_size in (3, 5):
            closed = cv2.morphologyEx(
                raw_mask,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size)),
                iterations=1,
            )
            contours_closed, hierarchy_closed = cv2.findContours(closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            closed_hole_count = 0
            if hierarchy_closed is not None and len(hierarchy_closed) > 0:
                closed_hole_count = int(sum(1 for item in hierarchy_closed[0] if int(item[3]) >= 0))
            if closed_hole_count > best_hole_count:
                best_hole_count = closed_hole_count
                best_contours = [np.asarray(contour) for contour in contours_closed]
                best_hierarchy = hierarchy_closed
        if best_hole_count <= raw_hole_count or best_hierarchy is None or not best_contours or len(best_hierarchy) == 0:
            recovered.append(_clone_polygon_with_id(polygon, next_id))
            next_id += 1
            continue
        hierarchy = best_hierarchy[0]
        contour_to_parent_id: dict[int, int] = {}
        for index, contour in enumerate(best_contours):
            parent_index = int(hierarchy[index][3])
            if parent_index >= 0:
                continue
            parent_poly = _contour_points_to_polygon(
                contour,
                left=left,
                top=top,
                polygon_id=next_id,
                is_hole=False,
                parent_id=None,
            )
            recovered.append(parent_poly)
            contour_to_parent_id[index] = next_id
            next_id += 1
        for index, contour in enumerate(best_contours):
            parent_index = int(hierarchy[index][3])
            if parent_index < 0:
                continue
            parent_id = contour_to_parent_id.get(parent_index)
            if parent_id is None:
                continue
            hole_poly = _contour_points_to_polygon(
                contour,
                left=left,
                top=top,
                polygon_id=next_id,
                is_hole=True,
                parent_id=parent_id,
            )
            recovered.append(hole_poly)
            next_id += 1
        # fall through: punch handling below uses full recovered list after loop
        continue
    # Assumption: KLayout renders each CIF polygon independently with OR/overlap
    # paint and does not punch nested outers out of covering frames. Keep that
    # fidelity on load; opt in only when empty nested windows are needed.
    if _nested_outer_punch_on_load_enabled():
        return _punch_nested_outer_covers(recovered)
    return recovered


def _family_geometry(outer: PolygonData, holes: list[PolygonData]):
    from shapely import make_valid
    from shapely.geometry import Polygon as ShapelyPolygon

    shell = [(float(x_coord), float(y_coord)) for x_coord, y_coord in outer.points]
    interiors = [
        [(float(x_coord), float(y_coord)) for x_coord, y_coord in hole.points]
        for hole in holes
        if len(hole.points) >= 3
    ]
    if len(shell) < 3:
        return ShapelyPolygon()
    return make_valid(ShapelyPolygon(shell, interiors))


def _nested_outer_punch_footprint(small_solid, large_geom):
    """Footprint to subtract from a covering parent for a nested outer.

    Concave outer notches (e.g. a U-cut at the top of a pad island) sit outside the
    island solid but still belong to the nested window. Punching only the solid leaves
    parent metal in those pockets and hides the inner contour. Expand to the convex
    hull when those pockets lie under the covering parent.
    """

    from shapely import make_valid

    hull = make_valid(small_solid.convex_hull)
    if hull.is_empty or float(hull.area) <= float(small_solid.area) + 1.0:
        return small_solid
    extra = make_valid(hull.difference(small_solid))
    if extra.is_empty or float(extra.area) <= 1.0:
        return small_solid
    covered_extra = large_geom.intersection(extra)
    if covered_extra.is_empty or float(covered_extra.area) < 0.5 * float(extra.area):
        return small_solid
    return hull


def _punch_nested_outer_covers(polygons: list[PolygonData]) -> list[PolygonData]:
    """Subtract smaller outers from larger overlapping fills (opt-in topology rewrite).

    Not applied on normal CIF load: overlapping frame+island metal is valid CIF and
    matches KLayout OR paint. When enabled (see ``_nested_outer_punch_on_load_enabled``),
    each nested outer's window footprint (solid, expanded to convex hull when concave
    pockets sit under the parent) is carved from covering parents so island holes open
    onto empty space instead of parent metal.
    """

    from shapely import make_valid
    from shapely.geometry import Point
    from shapely.geometry import Polygon as ShapelyPolygon

    outers = [polygon for polygon in polygons if not polygon.is_hole]
    if len(outers) < 2:
        return polygons
    holes_by_parent: dict[int, list[PolygonData]] = {}
    for polygon in polygons:
        if polygon.is_hole and polygon.parent_id is not None:
            holes_by_parent.setdefault(int(polygon.parent_id), []).append(polygon)

    geoms = {
        int(outer.id): _family_geometry(outer, holes_by_parent.get(int(outer.id), []))
        for outer in outers
    }
    originals = {int(outer.id): outer for outer in outers}
    changed_ids: set[int] = set()

    for large_id in sorted(originals, key=lambda item_id: -float(originals[item_id].area)):
        large_geom = geoms.get(large_id)
        if large_geom is None or large_geom.is_empty:
            continue
        for small_id, small_outer in originals.items():
            if small_id == large_id:
                continue
            if float(small_outer.area) >= float(originals[large_id].area) * 0.5:
                continue
            if len(small_outer.points) < 3:
                continue
            center = Point(
                float(small_outer.bbox[0]) + float(small_outer.bbox[2]) * 0.5,
                float(small_outer.bbox[1]) + float(small_outer.bbox[3]) * 0.5,
            )
            if not large_geom.contains(center):
                continue
            small_solid = make_valid(
                ShapelyPolygon([(float(x_coord), float(y_coord)) for x_coord, y_coord in small_outer.points])
            )
            if small_solid.is_empty or float(small_solid.area) <= 1.0:
                continue
            overlap = large_geom.intersection(small_solid)
            if overlap.is_empty or float(overlap.area) < 0.5 * float(small_solid.area):
                continue
            punch_footprint = _nested_outer_punch_footprint(small_solid, large_geom)
            punched = make_valid(large_geom.difference(punch_footprint))
            if punched.is_empty:
                continue
            if abs(float(punched.area) - float(large_geom.area)) < 1.0:
                continue
            geoms[large_id] = punched
            large_geom = punched
            changed_ids.add(large_id)

    if not changed_ids:
        return polygons

    rebuilt: list[PolygonData] = []
    next_id = 1
    for outer in sorted(outers, key=lambda item: int(item.id)):
        outer_id = int(outer.id)
        family_holes = holes_by_parent.get(outer_id, [])
        if outer_id not in changed_ids:
            clone = _clone_polygon_with_id(outer, next_id)
            parent_id = next_id
            rebuilt.append(clone)
            next_id += 1
            for hole in family_holes:
                hole_clone = _clone_polygon_with_id(hole, next_id)
                hole_clone.is_hole = True
                hole_clone.parent_id = parent_id
                rebuilt.append(hole_clone)
                next_id += 1
            continue
        pieces = _polygons_from_shapely_geometry(geoms[outer_id], start_id=next_id)
        if not pieces:
            clone = _clone_polygon_with_id(outer, next_id)
            rebuilt.append(clone)
            next_id += 1
            continue
        for piece in pieces:
            piece.category = str(outer.category)
            piece.shape_hint = str(outer.shape_hint)
            rebuilt.append(piece)
            next_id = max(next_id, int(piece.id) + 1)
    return rebuilt


def _polygons_from_shapely_geometry(geometry, *, start_id: int) -> list[PolygonData]:
    from shapely import make_valid
    from shapely.geometry import GeometryCollection, MultiPolygon
    from shapely.geometry import Polygon as ShapelyPolygon

    geom = make_valid(geometry)
    if geom.is_empty:
        return []

    def _collect_polygons(part) -> list:
        if isinstance(part, ShapelyPolygon):
            return [part] if not part.is_empty else []
        if isinstance(part, MultiPolygon):
            return [child for child in part.geoms if isinstance(child, ShapelyPolygon) and not child.is_empty]
        if isinstance(part, GeometryCollection):
            collected: list = []
            for child in part.geoms:
                collected.extend(_collect_polygons(child))
            return collected
        return []

    result: list[PolygonData] = []
    next_id = start_id
    for poly in _collect_polygons(geom):
        if float(poly.area) <= 1.0:
            continue
        exterior = [(float(x_coord), float(y_coord)) for x_coord, y_coord in poly.exterior.coords[:-1]]
        if len(exterior) < 3:
            continue
        area, perimeter, bbox = compute_polygon_metrics(exterior)
        parent_id = next_id
        result.append(
            PolygonData(
                id=parent_id,
                points=exterior,
                is_hole=False,
                parent_id=None,
                area=area,
                perimeter=perimeter,
                bbox=bbox,
            )
        )
        next_id += 1
        for interior in poly.interiors:
            hole_points = [(float(x_coord), float(y_coord)) for x_coord, y_coord in interior.coords[:-1]]
            if len(hole_points) < 3:
                continue
            area, perimeter, bbox = compute_polygon_metrics(hole_points)
            result.append(
                PolygonData(
                    id=next_id,
                    points=hole_points,
                    is_hole=True,
                    parent_id=parent_id,
                    area=area,
                    perimeter=perimeter,
                    bbox=bbox,
                )
            )
            next_id += 1
    return result


def _connected_cif_families(
    outer: PolygonData,
    holes: list[PolygonData],
) -> list[tuple[PolygonData, list[PolygonData]]]:
    """Split disconnected material like KLayout's resolve-holes writer stage."""

    from shapely import make_valid, unary_union
    from shapely.geometry import Polygon as ShapelyPolygon

    outer_geometry = make_valid(ShapelyPolygon(outer.points))
    hole_geometries = [
        make_valid(ShapelyPolygon(hole.points))
        for hole in holes
        if len(hole.points) >= 3
    ]
    material = make_valid(
        outer_geometry.difference(unary_union(hole_geometries))
        if hole_geometries
        else outer_geometry
    )
    pieces = _polygons_from_shapely_geometry(material, start_id=1)
    piece_outers = [piece for piece in pieces if not piece.is_hole]
    if len(piece_outers) <= 1:
        return [(outer, holes)]

    result: list[tuple[PolygonData, list[PolygonData]]] = []
    for piece_outer in piece_outers:
        piece_outer.category = str(outer.category)
        piece_outer.shape_hint = str(outer.shape_hint)
        piece_holes = [piece for piece in pieces if piece.is_hole and piece.parent_id == piece_outer.id]
        for piece_hole in piece_holes:
            piece_hole.category = str(outer.category)
            piece_hole.shape_hint = str(outer.shape_hint)
        result.append((piece_outer, piece_holes))
    return result


def save_polygons_cif(
    path: str | Path,
    image_path: str,
    polygons: list[PolygonData],
    image_size: tuple[int, int],
    layer_name: str = "NM",
    *,
    cutout_display: bool = False,
) -> Path:
    output = Path(path)
    width, height = int(image_size[0]), int(image_size[1])
    lines = [
        "DS 1 1 1;",
        f"L {layer_name};",
        f"( R {Path(image_path).name} );",
        f"( S {width} {height} );",
    ]
    if cutout_display:
        lines.append(CIF_CUTOUT_DISPLAY_MARKER)
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
    for polygon in sorted_polygons:
        if polygon.is_hole:
            continue
        save_families = [(polygon, [])]
        if polygon.category != "via" and polygon.shape_hint != "box":
            holes = holes_by_parent.get(int(polygon.id), [])
            if holes:
                save_families = _connected_cif_families(polygon, holes)
        for family_outer, family_holes in save_families:
            save_polygon = family_outer
            if family_holes:
                linked_points = _encode_parent_with_holes_link_path(family_outer, family_holes)
                area, perimeter, bbox = compute_polygon_metrics(linked_points)
                save_polygon = PolygonData(
                    id=family_outer.id,
                    points=linked_points,
                    is_hole=False,
                    parent_id=None,
                    category=family_outer.category,
                    shape_hint=family_outer.shape_hint,
                    area=area,
                    perimeter=perimeter,
                    bbox=bbox,
                )
            line = _polygon_to_cif_line(save_polygon, image_width=width, image_height=height)
            if line:
                lines.append(line)
    for hole in orphan_holes:
        clone = hole.clone()
        clone.is_hole = False
        clone.parent_id = None
        line = _polygon_to_cif_line(clone, image_width=width, image_height=height)
        if line:
            lines.append(line)
    lines.extend(["DF;", "E"])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    invalidate_cif_parse_cache([output])
    return output


def save_polygons_csv(path: str | Path, image_path: str, polygons: list[PolygonData]) -> Path:
    output = Path(path)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "image_path",
                "polygon_id",
                "vertex_index",
                "x",
                "y",
                "is_hole",
                "parent_id",
                "category",
                "shape_hint",
                "area",
                "perimeter",
                "bbox_x",
                "bbox_y",
                "bbox_width",
                "bbox_height",
            ]
        )
        for polygon in polygons:
            for vertex_index, (x_coord, y_coord) in enumerate(polygon.points):
                writer.writerow(
                    [
                        image_path,
                        polygon.id,
                        vertex_index,
                        str(_cv_coord(x_coord)),
                        str(_cv_coord(y_coord)),
                        int(polygon.is_hole),
                        "" if polygon.parent_id is None else polygon.parent_id,
                        polygon.category,
                        polygon.shape_hint,
                        f"{polygon.area:.6f}",
                        f"{polygon.perimeter:.6f}",
                        polygon.bbox[0],
                        polygon.bbox[1],
                        polygon.bbox[2],
                        polygon.bbox[3],
                    ]
                )
    return output


def save_polygons_txt(path: str | Path, image_path: str, polygons: list[PolygonData]) -> Path:
    output = Path(path)
    lines = [f"image_path: {image_path}", f"polygon_count: {len(polygons)}", ""]
    for polygon in polygons:
        points_repr = ", ".join(f"({_cv_coord(x)}, {_cv_coord(y)})" for x, y in polygon.points)
        lines.extend(
            [
                f"polygon_id: {polygon.id}",
                f"  is_hole: {polygon.is_hole}",
                f"  parent_id: {polygon.parent_id}",
                f"  category: {polygon.category}",
                f"  shape_hint: {polygon.shape_hint}",
                f"  area: {polygon.area:.6f}",
                f"  perimeter: {polygon.perimeter:.6f}",
                f"  bbox: {polygon.bbox}",
                f"  points: [{points_repr}]",
                "",
            ]
        )
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def save_svg_preview(
    path: str | Path,
    image_shape: tuple[int, ...],
    polygons: list[PolygonData],
    display_settings: DisplaySettings,
) -> Path:
    output = Path(path)
    height, width = image_shape[:2]
    alpha = max(0.0, min(1.0, display_settings.fill_opacity))
    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#101010"/>',
    ]
    for polygon in polygons:
        color = display_settings.hole_color if polygon.is_hole else display_settings.external_color
        if _is_ellipse_display_polygon(polygon):
            x_values = [float(point[0]) for point in polygon.points]
            y_values = [float(point[1]) for point in polygon.points]
            if len(x_values) < 3 or len(y_values) < 3:
                continue
            left = min(x_values)
            right = max(x_values)
            top = min(y_values)
            bottom = max(y_values)
            svg_lines.append(
                f'<ellipse cx="{(left + right) / 2.0:.3f}" cy="{(top + bottom) / 2.0:.3f}" '
                f'rx="{max(0.5, (right - left) / 2.0):.3f}" ry="{max(0.5, (bottom - top) / 2.0):.3f}" '
                f'fill="{color}" fill-opacity="{alpha:.3f}" stroke="{color}" '
                f'stroke-width="{display_settings.line_width:.2f}"/>'
            )
        else:
            points_attr = " ".join(f"{_cv_coord(x)},{_cv_coord(y)}" for x, y in polygon.points)
            svg_lines.append(
                f'<polygon points="{escape(points_attr)}" fill="{color}" fill-opacity="{alpha:.3f}" '
                f'stroke="{color}" stroke-width="{display_settings.line_width:.2f}"/>'
            )
    svg_lines.append("</svg>")
    output.write_text("\n".join(svg_lines), encoding="utf-8")
    return output


def save_overlay_preview(
    path: str | Path,
    source_image: np.ndarray,
    polygons: list[PolygonData],
    display_settings: DisplaySettings,
) -> Path:
    output = Path(path)
    preview = draw_polygon_overlay(source_image, polygons, display_settings)
    imwrite_unicode_safe(output, preview)
    return output


def _is_ellipse_display_polygon(polygon: PolygonData) -> bool:
    return polygon.shape_hint == "box" or polygon.category == "via"


def _copy_or_write_dataset_image(source_path: Path, target_path: Path, source_image: np.ndarray | None) -> Path:
    if source_path.exists() and source_path.is_file():
        if source_path.resolve() != target_path.resolve():
            shutil.copy2(source_path, target_path)
        return target_path
    if source_image is None:
        raise FileNotFoundError(tr("unable_to_load_image", path=source_path))
    imwrite_unicode_safe(target_path, source_image)
    return target_path


def export_dataset_frame(
    dataset_directory: str | Path,
    image_path: str,
    polygons: list[PolygonData],
    source_image: np.ndarray | None,
) -> dict[str, str]:
    root = ensure_directory(dataset_directory)
    images_root = ensure_directory(root / "images")
    cif_root = ensure_directory(root / "cif")
    source_path = Path(image_path)
    image_name = source_path.name
    if not source_path.suffix:
        image_name = f"{source_path.stem}.png"
    image_target = images_root / image_name
    cif_target = cif_root / f"{source_path.stem}.cif"

    image_size: tuple[int, int] | None = None
    if source_image is not None:
        image_size = (int(source_image.shape[1]), int(source_image.shape[0]))
    if image_size is None:
        raise ValueError(tr("dataset_source_image_missing", path=image_path))

    saved_image = _copy_or_write_dataset_image(source_path, image_target, source_image)
    saved_cif = save_polygons_cif(cif_target, str(saved_image), polygons, image_size=image_size)
    return {"image": str(saved_image), "cif": str(saved_cif)}


def save_result_bundle(
    output_directory: str | Path,
    image_path: str,
    polygons: list[PolygonData],
    source_image: np.ndarray | None,
    display_settings: DisplaySettings,
    save_options: SaveOptions | None = None,
    metadata: dict[str, object] | None = None,
) -> dict[str, str]:
    options = save_options or SaveOptions()
    root = ensure_directory(output_directory)
    stem = Path(image_path).stem
    saved: dict[str, str] = {}
    image_size: tuple[int, int] | None = None
    if source_image is not None:
        image_size = (int(source_image.shape[1]), int(source_image.shape[0]))

    if options.save_cif and image_size is not None:
        path = root / f"{stem}.cif"
        saved["cif"] = str(save_polygons_cif(path, image_path, polygons, image_size=image_size))
    if options.save_cv:
        path = root / f"{stem}.cv"
        saved["cv"] = str(save_polygons_cv(path, image_path, polygons, image_size=image_size, metadata=metadata))
    if options.save_json:
        path = root / f"{stem}.json"
        saved["json"] = str(save_polygons_json(path, image_path, polygons, metadata))
    if options.save_csv:
        path = root / f"{stem}.csv"
        saved["csv"] = str(save_polygons_csv(path, image_path, polygons))
    if options.save_txt:
        path = root / f"{stem}.txt"
        saved["txt"] = str(save_polygons_txt(path, image_path, polygons))
    if options.save_svg and source_image is not None:
        path = root / f"{stem}.svg"
        saved["svg"] = str(save_svg_preview(path, source_image.shape, polygons, display_settings))
    if options.save_preview and source_image is not None:
        path = root / f"{stem}_preview.png"
        saved["preview"] = str(save_overlay_preview(path, source_image, polygons, display_settings))
    return saved
