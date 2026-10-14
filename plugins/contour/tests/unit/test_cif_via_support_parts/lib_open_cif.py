"""test_cif_via_support: test lib open cif."""
from __future__ import annotations
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from contour.domain import PolygonData, compute_polygon_metrics
from contour.serializers import (
    clear_cif_parse_cache,
    load_polygons_cif,
)

def _longest_axis_aligned_edge(points: list[tuple[float, float]]) -> float:
    if len(points) < 2:
        return 0.0
    longest = 0.0
    count = len(points)
    for index in range(count):
        start_x, start_y = points[index]
        end_x, end_y = points[(index + 1) % count]
        if start_x == end_x or start_y == end_y:
            longest = max(longest, abs(end_x - start_x) + abs(end_y - start_y))
    return longest

class _EnvVar:
    """Temporarily set an environment variable and clear the CIF parse cache."""

    def __init__(self, name: str, value: str | None) -> None:
        self._name = name
        self._value = value
        self._previous: str | None = None

    def __enter__(self) -> None:
        self._previous = os.environ.get(self._name)
        if self._value is None:
            os.environ.pop(self._name, None)
        else:
            os.environ[self._name] = self._value
        clear_cif_parse_cache()

    def __exit__(self, *_exc: object) -> None:
        if self._previous is None:
            os.environ.pop(self._name, None)
        else:
            os.environ[self._name] = self._previous
        clear_cif_parse_cache()

def _rectangle_polygon(left: int, top: int, right: int, bottom: int) -> PolygonData:
    points = [
        (float(left), float(top)),
        (float(right), float(top)),
        (float(right), float(bottom)),
        (float(left), float(bottom)),
    ]
    area, perimeter, bbox = compute_polygon_metrics(points)
    return PolygonData(id=1, points=points, area=area, perimeter=perimeter, bbox=bbox)

class TestKLayoutCifLoader(unittest.TestCase):
    def test_klayout_reader_loads_sample_cif(self) -> None:
        with TemporaryDirectory() as temp_root:
            cif_path = Path(temp_root) / "klayout_sample.cif"
            cif_path.write_text(
                "\n".join(
                    [
                        "DS 1 1 1;",
                        "L NM;",
                        "( R sample.jpg );",
                        "( S 80 80 );",
                        "P 10 10 70 10 70 70 10 70 10 10;",
                        "DF;",
                        "E",
                    ]
                ),
                encoding="utf-8",
            )
            clear_cif_parse_cache()
            image_name, image_size, polygons = load_polygons_cif(cif_path)
        self.assertEqual(image_name, "sample.jpg")
        self.assertEqual(image_size, (80, 80))
        self.assertEqual(len(polygons), 1)
        self.assertEqual(polygons[0].points, [(10, 70), (70, 70), (70, 10), (10, 10)])

class TestLibOpenCifLoader(unittest.TestCase):
    def test_opencif_loader_matches_python_fallback_for_sample_cif(self) -> None:
        from contour.infrastructure.cif_opencif import opencif_loader_available

        if not opencif_loader_available():
            self.skipTest("LibOpenCIF extension is not built")

        with TemporaryDirectory() as temp_root:
            cif_path = Path(temp_root) / "opencif_sample.cif"
            cif_path.write_text(
                "\n".join(
                    [
                        "DS 1 1 1;",
                        "L NM;",
                        "( R sample.jpg );",
                        "( S 80 80 );",
                        "P 10 10 70 10 70 70 10 70 10 10;",
                        "DF;",
                        "E",
                    ]
                ),
                encoding="utf-8",
            )

            clear_cif_parse_cache()
            with _EnvVar("CONTOUR_CIF_USE_OPENCIF", "1"):
                opencif_name, opencif_size, opencif_polygons = load_polygons_cif(cif_path)
            clear_cif_parse_cache()
            with _EnvVar("CONTOUR_CIF_USE_OPENCIF", "0"):
                python_name, python_size, python_polygons = load_polygons_cif(cif_path)

        self.assertEqual(opencif_name, python_name)
        self.assertEqual(opencif_size, python_size)
        self.assertEqual(len(opencif_polygons), len(python_polygons))
        self.assertEqual(opencif_polygons[0].points, python_polygons[0].points)
