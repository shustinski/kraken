from __future__ import annotations

from pathlib import Path

from contour.adapters.qt.antialias_cif import AntialiasCifWorkItem, _antialias_work_item
from contour.application.polygon_antialiasing import antialias_polygons
from contour.domain import PolygonData, compute_polygon_metrics
from contour.serializers import load_polygons_vector, save_polygons_vector


def _oversampled_rectangle() -> PolygonData:
    points = [
        (0.0, 0.0),
        (2.0, 0.0),
        (4.0, 0.0),
        (8.0, 0.0),
        (8.0, 3.0),
        (8.0, 8.0),
        (4.0, 8.0),
        (0.0, 8.0),
        (0.0, 4.0),
    ]
    area, perimeter, bbox = compute_polygon_metrics(points)
    return PolygonData(id=1, points=points, area=area, perimeter=perimeter, bbox=bbox)


def test_antialias_work_item_writes_simplified_cif(tmp_path: Path) -> None:
    original = _oversampled_rectangle()
    cif_path = tmp_path / "frame.cif"
    save_polygons_vector(cif_path, "frame.png", [original], image_size=(16, 16))
    expected, changed = antialias_polygons([original], 2)
    assert changed

    result = _antialias_work_item(
        AntialiasCifWorkItem(
            stem="frame",
            cif_path=str(cif_path),
            image_path="frame.png",
            polygons=None,
            image_size=None,
        ),
        grade=2,
        run_id=1,
    )

    assert result.changed is True
    assert result.error is None
    _name, _size, loaded = load_polygons_vector(cif_path)
    assert [polygon.points for polygon in loaded] == [polygon.points for polygon in expected]
