"""serializers: cif box primitive polygon."""
from __future__ import annotations
from collections.abc import Iterable
from pathlib import Path
from contour.domain import PolygonData, compute_polygon_metrics
from contour.i18n import tr
from ..infrastructure.cif_primitives import CifBox, CifPrimitive

def _cif_box_primitive_to_polygon(
    box: CifBox,
    image_size: tuple[int, int],
    polygon_id: int,
) -> PolygonData:
    from contour.serializers_parts.cif_box_polygon import _cif_box_to_polygon
    from ..infrastructure.cif_klayout_reader import rotated_box_points

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
    from contour.serializers_parts.cif_box_polygon import _cif_polygon_points_to_polygon, _extract_parenthesized_tokens, _parse_cif_int
    from ..infrastructure.cif_primitives import CifBox, CifComment, CifPolygon

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
