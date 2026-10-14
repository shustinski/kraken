"""test_cif_via_support: test cif loader keeps."""
from __future__ import annotations
from contour.domain import PolygonData, compute_polygon_metrics
from contour.serializers import (
    load_polygons_cif,
    save_polygons_cif,
)

from test_cif_via_support_parts.lib_open_cif import _longest_axis_aligned_edge


class CifLoaderKeepsMixin:
    def test_cif_loader_splits_keyhole_without_private_metadata(self) -> None:
        cif_path = self._artifact_path("repeated_keyhole_bridge_recover.cif")
        cif_path.write_text(
            "\n".join(
                [
                    "DS 1 1 1;",
                    "L NM;",
                    "( R sample.png );",
                    "( S 100 100 );",
                    "P 90 50 90 50 40 50 40 40 20 40 20 60 40 60 40 50 90 50 90 10 10 10 10 90 90 90 90 50;",
                    "DF;",
                    "E",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        _image_name, _image_size, loaded = load_polygons_cif(cif_path)
        holes = [polygon for polygon in loaded if polygon.is_hole]
        outers = [polygon for polygon in loaded if not polygon.is_hole]

        self.assertEqual(len(outers), 1)
        self.assertEqual(len(holes), 1)
        hole_xs = [point[0] for point in holes[0].points]
        self.assertLessEqual(max(hole_xs), 40)
        self.assertGreaterEqual(min(hole_xs), 20)
        self.assertLess(_longest_axis_aligned_edge(holes[0].points), 50)
        self.assertFalse(any(polygon.description_is_invalid() for polygon in loaded))

    def test_cif_loader_splits_keyhole_with_out_and_back_spike(self) -> None:
        """A->B->A rim spikes must not block keyhole recovery (0560-style)."""

        cif_path = self._artifact_path("keyhole_out_and_back_spike.cif")
        cif_path.write_text(
            "\n".join(
                [
                    "DS 1 1 1;",
                    "L NM;",
                    "( R sample.png );",
                    "( S 100 100 );",
                    "P 90 50 40 50 40 40 20 40 20 60 30 60 30 70 30 60 40 60 40 50 "
                    "90 50 90 10 10 10 10 90 90 90 90 50;",
                    "DF;",
                    "E",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        _image_name, _image_size, loaded = load_polygons_cif(cif_path)
        holes = [polygon for polygon in loaded if polygon.is_hole]
        outers = [polygon for polygon in loaded if not polygon.is_hole]

        self.assertEqual(len(outers), 1)
        self.assertEqual(len(holes), 1)
        self.assertFalse(any(polygon.description_is_invalid() for polygon in loaded))
        hole_xs = [point[0] for point in holes[0].points]
        hole_ys = [point[1] for point in holes[0].points]
        self.assertLessEqual(max(hole_xs), 40)
        self.assertGreaterEqual(min(hole_xs), 20)
        self.assertLessEqual(max(hole_ys), 60)
        self.assertGreaterEqual(min(hole_ys), 40)
        # Spike tip (CIF 30,70 → image y=30) must be dropped, not kept on the hole.
        self.assertFalse(any(abs(x - 30.0) < 1e-6 and abs(y - 30.0) < 1e-6 for x, y in holes[0].points))

    def test_cif_loader_normalizes_keyhole_fill_like_make_valid(self) -> None:
        """Complex keyholes must keep make_valid fill area (KLayout-like), not heuristic over-fill."""

        from shapely import make_valid
        from shapely.geometry import Polygon as ShapelyPolygon

        from contour.serializers import _normalize_cif_polygon_families, _split_linked_polygon_rings

        # Self-touching ring whose heuristic split over-fills relative to make_valid.
        points = [
            (0.0, 0.0),
            (100.0, 0.0),
            (100.0, 100.0),
            (0.0, 100.0),
            (0.0, 50.0),
            (40.0, 50.0),
            (40.0, 70.0),
            (20.0, 70.0),
            (20.0, 30.0),
            (40.0, 30.0),
            (40.0, 50.0),
            (0.0, 50.0),
        ]
        families = _normalize_cif_polygon_families(points)
        self.assertIsNotNone(families)
        assert families is not None
        self.assertEqual(len(families), 1)
        outer, holes = families[0]
        self.assertGreaterEqual(len(holes), 1)
        normalized = make_valid(ShapelyPolygon(outer, holes=holes))
        expected = make_valid(ShapelyPolygon(points))
        self.assertAlmostEqual(float(normalized.area), float(expected.area), delta=1.0)

        heuristic_outer, heuristic_holes = _split_linked_polygon_rings(points)
        if heuristic_holes:
            heuristic = make_valid(ShapelyPolygon(heuristic_outer, holes=heuristic_holes))
            # Guard: if heuristic drifts, normalization must stay on make_valid.
            if abs(float(heuristic.area) - float(expected.area)) > 1.0:
                self.assertLess(abs(float(normalized.area) - float(expected.area)), 1.0)

    def test_cif_loader_keeps_nested_outer_overlap_like_klayout(self) -> None:
        """0518-style: overlapping frame metal under island holes must stay (CIF OR paint).

        KLayout fills each ``P`` independently; nested outers are not punched on load.
        """

        frame_points = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
        island_points = [(20.0, 30.0), (80.0, 30.0), (80.0, 70.0), (20.0, 70.0)]
        hole_points = [(40.0, 40.0), (60.0, 40.0), (60.0, 60.0), (40.0, 60.0)]
        frame_area, frame_perimeter, frame_bbox = compute_polygon_metrics(frame_points)
        island_area, island_perimeter, island_bbox = compute_polygon_metrics(island_points)
        hole_area, hole_perimeter, hole_bbox = compute_polygon_metrics(hole_points)
        frame = PolygonData(
            id=1,
            points=frame_points,
            area=frame_area,
            perimeter=frame_perimeter,
            bbox=frame_bbox,
            category="conductor",
            shape_hint="polygon",
        )
        island = PolygonData(
            id=2,
            points=island_points,
            area=island_area,
            perimeter=island_perimeter,
            bbox=island_bbox,
            category="conductor",
            shape_hint="polygon",
        )
        hole = PolygonData(
            id=3,
            points=hole_points,
            is_hole=True,
            parent_id=2,
            area=hole_area,
            perimeter=hole_perimeter,
            bbox=hole_bbox,
            category="conductor",
            shape_hint="polygon",
        )
        cif_path = self._artifact_path("nested_outer_cover.cif")
        save_polygons_cif(cif_path, "sample.png", [frame, island, hole], image_size=(100, 100))

        _image_name, _image_size, loaded = load_polygons_cif(cif_path)
        holes = [polygon for polygon in loaded if polygon.is_hole]
        outers = [polygon for polygon in loaded if not polygon.is_hole]
        self.assertEqual(len(outers), 2)
        self.assertEqual(len(holes), 1)

        from shapely import make_valid
        from shapely.geometry import Point, Polygon

        island_keyhole = min(
            outers,
            key=lambda polygon: float(make_valid(Polygon(polygon.points)).area),
        )
        island_holes = [polygon.points for polygon in holes if polygon.parent_id == island_keyhole.id]
        valid_island = make_valid(Polygon(island_keyhole.points, holes=island_holes))
        self.assertFalse(valid_island.is_empty)
        # Interior of the island hole must be empty under make_valid (OddEven keyhole fill).
        probe = Point(50.0, 50.0)
        self.assertFalse(valid_island.contains(probe))
        covering = [
            outer
            for outer in outers
            if outer.id != island_keyhole.id and Polygon(outer.points).contains(probe)
        ]
        self.assertTrue(covering, msg="frame should still cover island hole (KLayout OR paint)")
