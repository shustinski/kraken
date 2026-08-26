from __future__ import annotations

from contour.domain import PolygonData, offset_conductor_polygons


def test_offset_conductor_polygons_expands_and_shrinks_geometry() -> None:
    polygon = PolygonData(
        id=1,
        points=[(10, 10), (30, 10), (30, 20), (10, 20)],
        category="conductor",
    )
    expanded = offset_conductor_polygons([polygon], 2.0)
    shrunk = offset_conductor_polygons([polygon], -2.0)
    via = PolygonData(
        id=2,
        points=[(0, 0), (8, 0), (8, 8), (0, 8)],
        category="via",
        shape_hint="box",
    )

    assert len(expanded) == 1
    assert len(shrunk) == 1
    assert expanded[0].area > shrunk[0].area
    unchanged = offset_conductor_polygons([via], 5.0)
    assert unchanged[0].points == via.points
