"""Display/load behavior for CIF keyhole cut locations."""

from __future__ import annotations

from unittest.mock import patch

from PyQt6.QtCore import Qt

from contour.domain import PolygonData, compute_polygon_metrics
from contour.graphics_items import _vector_display_path_for_polygon
from contour.serializers import _polygon_uses_authored_cif_paint_ring


def _square_polygon(polygon_id: int, left: float, top: float, size: float) -> PolygonData:
    points = [
        (left, top),
        (left + size, top),
        (left + size, top + size),
        (left, top + size),
    ]
    area, perimeter, bbox = compute_polygon_metrics(points)
    return PolygonData(
        id=polygon_id,
        points=points,
        area=area,
        perimeter=perimeter,
        bbox=bbox,
    )


def test_vector_display_hides_cut_locations_for_non_klayout_keyhole_family() -> None:
    outer = _square_polygon(1, 0.0, 0.0, 100.0)
    hole = _square_polygon(2, 30.0, 30.0, 20.0)
    hole.is_hole = True
    hole.parent_id = outer.id
    authored_ring = [
        (0.0, 0.0),
        (100.0, 0.0),
        (100.0, 100.0),
        (50.0, 100.0),
        (50.0, 50.0),
        (0.0, 50.0),
    ]
    outer.cif_paint_ring = list(authored_ring)
    assert _polygon_uses_authored_cif_paint_ring(outer)

    with patch(
        "contour.application.fix_internal_contours._is_klayout_keyhole_slot",
        return_value=False,
    ):
        display_path = _vector_display_path_for_polygon(outer, cutout_polygons=[hole])

    assert display_path.fillRule() == Qt.FillRule.WindingFill
    assert display_path.elementCount() != len(authored_ring) + 1
