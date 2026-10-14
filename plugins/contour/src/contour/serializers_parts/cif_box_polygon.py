"""serializers: cif box polygon."""
from __future__ import annotations
import json
from pathlib import Path
from contour.domain import PolygonData, compute_polygon_metrics
from contour.i18n import tr

from contour.serializers_parts.module_state import CIF_CUTOUT_DISPLAY_MARKER

from contour.serializers_parts.cv_object_polygon import _pixel_box_bounds

from contour.serializers_parts.cv_point_points import (
    _cv_object_id,
    save_polygons_cv,
)

from contour.serializers_parts.load_polygons_cif import load_polygons_cif

from contour.serializers_parts.polygons_cv_object import _polygons_from_cv_object

from contour.serializers_parts.save_polygons_cif import save_polygons_cif

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

def _cif_cutout_display_requested(cif_path: Path) -> bool:
    return any(line.strip() == CIF_CUTOUT_DISPLAY_MARKER.strip() for line in _read_cif_text(cif_path).splitlines())

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
    image_points: list[tuple[float, float]] = [(x_coord, image_height - y_coord) for x_coord, y_coord in raw_points]
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
