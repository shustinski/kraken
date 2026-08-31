from __future__ import annotations

import unittest

from contour.domain import PolygonData, compute_polygon_metrics
from contour.infrastructure.cif_operation_profiler import (
    cif_operation_profiling,
    note_cif_operation_count,
    note_cif_operation_timing,
)
from contour.serializers import save_polygons_cif


class CifOperationProfilerTests(unittest.TestCase):
    def test_cif_operation_timings_accumulate_in_active_scope(self) -> None:
        with cif_operation_profiling() as timings:
            note_cif_operation_timing("cif_hole_vertex_slit_search", 0.5)
            note_cif_operation_timing("cif_hole_vertex_slit_search", 0.25)

        self.assertEqual(timings["cif_hole_vertex_slit_search"], 0.75)

    def test_cif_operation_counts_accumulate_in_active_scope(self) -> None:
        with cif_operation_profiling() as timings:
            note_cif_operation_count("cif_hole_outer_slit")
            note_cif_operation_count("cif_hole_neighbor_slit", 2.0)

        self.assertEqual(timings["cif_hole_outer_slit"], 1.0)
        self.assertEqual(timings["cif_hole_neighbor_slit"], 2.0)

    def test_save_polygons_cif_records_hole_vertex_slit_search_timing(self) -> None:
        outer_points = [(10.0, 10.0), (70.0, 10.0), (70.0, 70.0), (10.0, 70.0)]
        inner_points = [(32.0, 32.0), (48.0, 32.0), (48.0, 48.0), (32.0, 48.0)]
        outer_area, outer_perimeter, outer_bbox = compute_polygon_metrics(outer_points)
        inner_area, inner_perimeter, inner_bbox = compute_polygon_metrics(inner_points)
        outer = PolygonData(
            id=1,
            points=outer_points,
            area=outer_area,
            perimeter=outer_perimeter,
            bbox=outer_bbox,
        )
        hole = PolygonData(
            id=2,
            points=inner_points,
            is_hole=True,
            parent_id=1,
            area=inner_area,
            perimeter=inner_perimeter,
            bbox=inner_bbox,
        )

        with cif_operation_profiling() as timings:
            save_polygons_cif(self._artifact_path("profiled_linked_hole.cif"), "sample.png", [outer, hole], image_size=(80, 80))

        self.assertIn("cif_hole_vertex_slit_search", timings)
        self.assertGreater(timings["cif_hole_vertex_slit_search"], 0.0)
        self.assertIn("cif_hole_link_encode", timings)
        self.assertGreaterEqual(timings["cif_hole_link_encode"], timings["cif_hole_vertex_slit_search"])

    def test_save_polygons_cif_records_outer_and_neighbor_slit_counts(self) -> None:
        outer_points = [(0.0, 0.0), (120.0, 0.0), (120.0, 120.0), (0.0, 120.0)]
        hole_bounds = [
            (50, 50, 70, 70),
            (50, 20, 70, 40),
            (50, 80, 70, 100),
            (20, 50, 40, 70),
            (80, 50, 100, 70),
        ]
        outer_area, outer_perimeter, outer_bbox = compute_polygon_metrics(outer_points)
        outer = PolygonData(
            id=1,
            points=outer_points,
            area=outer_area,
            perimeter=outer_perimeter,
            bbox=outer_bbox,
        )
        polygons = [outer]
        for polygon_id, bounds in enumerate(hole_bounds, start=2):
            left, top, right, bottom = bounds
            points = [
                (float(left), float(top)),
                (float(right), float(top)),
                (float(right), float(bottom)),
                (float(left), float(bottom)),
            ]
            area, perimeter, bbox = compute_polygon_metrics(points)
            polygons.append(
                PolygonData(
                    id=polygon_id,
                    points=points,
                    is_hole=True,
                    parent_id=1,
                    area=area,
                    perimeter=perimeter,
                    bbox=bbox,
                )
            )

        with cif_operation_profiling() as timings:
            save_polygons_cif(
                self._artifact_path("profiled_multi_hole_keyhole.cif"),
                "sample.png",
                polygons,
                image_size=(120, 120),
            )

        self.assertEqual(timings.get("cif_hole_outer_slit", 0.0), 1.0)
        self.assertGreaterEqual(timings.get("cif_hole_neighbor_slit", 0.0), 1.0)
        self.assertEqual(
            timings.get("cif_hole_outer_slit", 0.0) + timings.get("cif_hole_neighbor_slit", 0.0),
            5.0,
        )

    def _artifact_path(self, name: str):
        from pathlib import Path

        directory = Path(__file__).resolve().parents[1] / ".tmp-tests"
        directory.mkdir(parents=True, exist_ok=True)
        return directory / name


if __name__ == "__main__":
    unittest.main()
