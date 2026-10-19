"""test_cif_via_support: test cif loader matches."""
from __future__ import annotations

from pathlib import Path

from test_cif_via_support_parts.lib_open_cif import _rectangle_polygon

from contour.domain import PolygonData
from contour.serializers import (
    clear_cif_parse_cache,
    load_polygons_cif,
    save_polygons_cif,
)


class CifLoaderMatchesMixin:
    def test_cif_writer_splits_material_when_no_single_slit_exists(self) -> None:
        from shapely import make_valid, unary_union
        from shapely.geometry import Polygon as ShapelyPolygon

        outer = _rectangle_polygon(0, 0, 100, 100)
        # Four touching cutouts isolate the central material island from the
        # outer material. The central hole therefore cannot share one slit with
        # the outer boundary; KLayout-style resolution emits two P commands.
        hole_bounds = [
            (47, 47, 53, 53),
            (20, 20, 80, 45),
            (20, 55, 80, 80),
            (20, 45, 45, 55),
            (55, 45, 80, 55),
        ]
        holes: list[PolygonData] = []
        for polygon_id, bounds in enumerate(hole_bounds, start=2):
            hole = _rectangle_polygon(*bounds)
            hole.id = polygon_id
            hole.is_hole = True
            hole.parent_id = outer.id
            holes.append(hole)

        cif_path = self._artifact_path("disconnected_material_keyholes.cif")
        save_polygons_cif(cif_path, "sample.png", [outer, *holes], image_size=(100, 100))
        _image_name, _image_size, loaded = load_polygons_cif(cif_path)

        expected = make_valid(ShapelyPolygon(outer.points)).difference(
            unary_union([make_valid(ShapelyPolygon(hole.points)) for hole in holes])
        )
        loaded_solids = [
            make_valid(
                ShapelyPolygon(
                    polygon.points,
                    holes=[hole.points for hole in loaded if hole.is_hole and hole.parent_id == polygon.id],
                )
            )
            for polygon in loaded
            if not polygon.is_hole
        ]
        actual = unary_union(loaded_solids)
        payload = cif_path.read_text(encoding="utf-8")

        self.assertEqual(payload.count("\nP "), 2)
        self.assertNotIn("CONTOUR", payload)
        self.assertAlmostEqual(float(expected.symmetric_difference(actual).area), 0.0)

    def test_cif_loader_preserves_valid_multi_hole_frame_0518(self) -> None:
        cif_path = Path(r"D:\OZI\Нейронка\cif_metal\0518.cif")
        if not cif_path.exists():
            self.skipTest("0518.cif fixture not available")
        from contour.application.vector_geometry_postprocess import repair_invalid_polygon_descriptions

        clear_cif_parse_cache()
        _image_name, _image_size, loaded = load_polygons_cif(cif_path)
        holes_before = sum(1 for polygon in loaded if polygon.is_hole)
        self.assertGreater(holes_before, 0)
        self.assertFalse(any(str(polygon.reject_reason).strip() for polygon in loaded))

        repaired = repair_invalid_polygon_descriptions(loaded, None)
        self.assertEqual(len(repaired), len(loaded))
        self.assertEqual(sum(1 for polygon in repaired if polygon.is_hole), holes_before)

    def test_cif_loader_matches_klayout_fill_for_authored_keyhole(self) -> None:
        """Keyhole fill must match KLayout (Qt WindingFill on the authored ring)."""

        import numpy as np

        from contour.infrastructure.cif_klayout_reader import load_cif_primitives_klayout
        from contour.infrastructure.cif_primitives import CifPolygon
        from contour.serializers import (
            _cif_paint_mask_from_ring,
            _dedupe_closed_points,
            _has_duplicate_points,
            _polygon_uses_authored_cif_paint_ring,
            _render_cif_klayout_layer_mask,
            _stamp_cif_paint_ring_on_mask,
        )

        cif_path = Path(__file__).parents[2] / "fixtures" / "authored_keyhole.cif"

        image_height = 100
        parsed = load_cif_primitives_klayout(cif_path)
        reference_mask = np.zeros((image_height, image_height), dtype=np.uint8)
        keyhole_cif_points: list[tuple[float, float]] | None = None
        for primitive in parsed.primitives:
            if not isinstance(primitive, CifPolygon):
                continue
            cif_points = [(float(x_coord), float(y_coord)) for x_coord, y_coord in primitive.points]
            image_points = [
                (float(x_coord), float(image_height - y_coord)) for x_coord, y_coord in cif_points
            ]
            _stamp_cif_paint_ring_on_mask(reference_mask, image_points, (0, 0), value=255)
            if _has_duplicate_points(_dedupe_closed_points(cif_points)):
                keyhole_cif_points = image_points
        self.assertIsNotNone(keyhole_cif_points)
        assert keyhole_cif_points is not None

        clear_cif_parse_cache()
        _image_name, image_size, loaded = load_polygons_cif(cif_path)
        outer = next(polygon for polygon in loaded if _polygon_uses_authored_cif_paint_ring(polygon))
        self.assertTrue(outer.cif_paint_ring)

        rendered_mask = _render_cif_klayout_layer_mask(image_size, loaded)
        xor_pixels = int(np.sum(np.bitwise_xor(reference_mask, rendered_mask) > 0))
        self.assertEqual(xor_pixels, 0, msg=f"Keyhole fill differs from KLayout by {xor_pixels} pixels")

        hole = next(polygon for polygon in loaded if polygon.parent_id == outer.id)
        hole_mask, left, top = _cif_paint_mask_from_ring(outer.cif_paint_ring)
        hole_left, hole_top, hole_width, hole_height = hole.bbox
        local_x = round(hole_left + hole_width / 2) - left
        local_y = round(hole_top + hole_height / 2) - top
        self.assertGreaterEqual(local_x, 0)
        self.assertGreaterEqual(local_y, 0)
        self.assertLess(local_x, hole_mask.shape[1])
        self.assertLess(local_y, hole_mask.shape[0])
        self.assertGreater(int(hole_mask[local_y, local_x]), 0, msg="keyhole slot interior must be filled")

    def test_cif_vector_display_path_uses_authored_ring_for_keyhole(self) -> None:
        from PyQt6.QtCore import Qt

        from contour.graphics_items import _vector_display_path_for_polygon
        from contour.serializers import clear_cif_parse_cache, load_polygons_cif

        cif_path = self._artifact_path("repeated_keyhole_bridge.cif")
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

        clear_cif_parse_cache()
        _image_name, _image_size, loaded = load_polygons_cif(cif_path)
        keyhole_outer = next(polygon for polygon in loaded if not polygon.is_hole)
        holes = [polygon for polygon in loaded if polygon.parent_id == keyhole_outer.id]
        display_path = _vector_display_path_for_polygon(keyhole_outer, cutout_polygons=holes)
        self.assertEqual(display_path.elementCount(), len(keyhole_outer.cif_paint_ring))
        self.assertEqual(display_path.fillRule(), Qt.FillRule.WindingFill)

    def test_cif_loader_recovers_unmarked_standard_keyhole(self) -> None:
        cif_path = self._artifact_path("repeated_keyhole_bridge.cif")
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
        self.assertEqual(holes[0].parent_id, outers[0].id)
        self.assertFalse(any(polygon.description_is_invalid() for polygon in loaded))
