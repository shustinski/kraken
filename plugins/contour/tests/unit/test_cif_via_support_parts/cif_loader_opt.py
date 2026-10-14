"""test_cif_via_support: test cif loader opt."""
from __future__ import annotations
from contour.domain import PolygonData, compute_polygon_metrics
from contour.serializers import (
    load_polygons_cif,
    save_polygons_cif,
)

from test_cif_via_support_parts.lib_open_cif import (
    _EnvVar,
    _longest_axis_aligned_edge,
)


class CifLoaderOptMixin:
    def test_cif_loader_keeps_concave_island_notch_under_covering_frame(self) -> None:
        """0518-style: U-notch outside the island solid stays filled by the overlapping frame."""

        frame_points = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
        # Outer with a top U-notch: arms at the sides, pocket around (50, 20).
        island_points = [
            (20.0, 10.0),
            (30.0, 10.0),
            (30.0, 30.0),
            (70.0, 30.0),
            (70.0, 10.0),
            (80.0, 10.0),
            (80.0, 70.0),
            (20.0, 70.0),
        ]
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
        cif_path = self._artifact_path("nested_outer_concave_notch.cif")
        save_polygons_cif(cif_path, "sample.png", [frame, island, hole], image_size=(100, 100))

        _image_name, _image_size, loaded = load_polygons_cif(cif_path)
        holes = [polygon for polygon in loaded if polygon.is_hole]
        outers = [polygon for polygon in loaded if not polygon.is_hole]
        self.assertEqual(len(holes), 1)
        self.assertEqual(len(outers), 2)

        from shapely.geometry import Point, Polygon

        notch = Point(50.0, 20.0)
        covering = [outer for outer in outers if Polygon(outer.points).contains(notch)]
        # Frame rectangle covers the notch; island keyhole does not fill it as solid.
        self.assertTrue(covering, msg="U-notch should remain covered by frame (no silent punch)")

    def test_cif_loader_opt_in_punch_nested_outer_empties_island_window(self) -> None:
        """Opt-in CONTOUR_CIF_PUNCH_NESTED_OUTERS restores empty windows under nested islands."""

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
        cif_path = self._artifact_path("nested_outer_cover_opt_in_punch.cif")
        save_polygons_cif(cif_path, "sample.png", [frame, island, hole], image_size=(100, 100))

        with _EnvVar("CONTOUR_CIF_PUNCH_NESTED_OUTERS", "1"):
            _image_name, _image_size, loaded = load_polygons_cif(cif_path)

        holes = [polygon for polygon in loaded if polygon.is_hole]
        outers = [polygon for polygon in loaded if not polygon.is_hole]

        from shapely.geometry import Point, Polygon

        def family_polygon(outer_id: int) -> Polygon:
            outer = next(polygon for polygon in outers if polygon.id == outer_id)
            family_holes = [polygon for polygon in holes if polygon.parent_id == outer_id]
            return Polygon(outer.points, holes=[hole.points for hole in family_holes])

        island_loaded = max(
            (polygon for polygon in outers if any(item.parent_id == polygon.id for item in holes)),
            key=lambda polygon: polygon.area,
        )
        island_hole = next(item for item in holes if item.parent_id == island_loaded.id)
        center_x = island_hole.bbox[0] + island_hole.bbox[2] * 0.5
        center_y = island_hole.bbox[1] + island_hole.bbox[3] * 0.5
        for outer in outers:
            if outer.id == island_loaded.id:
                continue
            covered = family_polygon(outer.id).contains(Point(center_x, center_y))
            self.assertFalse(covered, msg=f"hole still covered by outer {outer.id}")

    def test_cif_loader_splits_offset_two_hole_keyhole_without_span_edge(self) -> None:
        cif_path = self._artifact_path("offset_two_hole_keyhole.cif")
        cif_path.write_text(
            "\n".join(
                [
                    "DS 1 1 1;",
                    "L NM;",
                    "( R sample.png );",
                    "( S 100 100 );",
                    "P 90 51 70 51 70 30 55 30 55 70 70 70 70 51 90 51 90 50 90 50 45 50 45 30 20 30 20 70 45 70 45 50 90 50 90 10 10 10 10 90 90 90 90 51;",
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
        self.assertEqual(len(holes), 2)
        for hole in holes:
            self.assertLess(_longest_axis_aligned_edge(hole.points), 50)
            xs = [point[0] for point in hole.points]
            self.assertLess(max(xs) - min(xs), 40)
        self.assertFalse(any(polygon.description_is_invalid() for polygon in loaded))
