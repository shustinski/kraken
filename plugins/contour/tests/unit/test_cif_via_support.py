from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from contour.application.processing import ContourExtractionSettings, DisplaySettings, SaveOptions
from contour.contour_extractor import extract_polygons
from contour.domain import PolygonData, compute_polygon_metrics
from contour.serializers import (
    clear_cif_parse_cache,
    export_dataset_frame,
    load_polygons_cif,
    load_polygons_cv,
    load_polygons_vector,
    save_polygons_cif,
    save_polygons_cv,
    save_result_bundle,
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


class CifViaSupportTests(unittest.TestCase):
    def _artifact_path(self, name: str) -> Path:
        root = Path(".tmp-tests")
        root.mkdir(exist_ok=True)
        path = root / name
        if path.exists():
            path.unlink()
        return path

    def test_via_profile_extracts_box_shapes(self) -> None:
        mask = np.zeros((80, 80), dtype=np.uint8)
        cv2.ellipse(mask, (40, 40), (8, 6), 0, 0, 360, 255, thickness=-1)

        polygons = extract_polygons(
            mask,
            ContourExtractionSettings(
                extraction_profile="vias",
                object_type="via",
                output_mode="box",
                min_area=10.0,
            ),
        )

        self.assertEqual(len(polygons), 1)
        self.assertEqual(polygons[0].category, "via")
        self.assertEqual(polygons[0].shape_hint, "box")
        self.assertFalse(polygons[0].is_hole)
        self.assertEqual(len(polygons[0].points), 4)

    def test_via_profile_applies_via_size_limits(self) -> None:
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.ellipse(mask, (30, 50), (5, 5), 0, 0, 360, 255, thickness=-1)
        cv2.ellipse(mask, (70, 50), (12, 12), 0, 0, 360, 255, thickness=-1)

        polygons = extract_polygons(
            mask,
            ContourExtractionSettings(
                extraction_profile="vias",
                object_type="via",
                output_mode="box",
                min_via_width=8,
                max_via_width=20,
                min_via_height=8,
                max_via_height=20,
                min_area=10.0,
            ),
        )

        self.assertEqual(len(polygons), 1)
        self.assertEqual(polygons[0].category, "via")

    def test_via_profile_applies_fixed_single_size(self) -> None:
        mask = np.zeros((120, 120), dtype=np.uint8)
        cv2.rectangle(mask, (10, 10), (19, 17), 255, thickness=-1)
        cv2.rectangle(mask, (50, 10), (61, 21), 255, thickness=-1)

        polygons = extract_polygons(
            mask,
            ContourExtractionSettings(
                extraction_profile="vias",
                object_type="via",
                output_mode="box",
                via_size_mode="fixed",
                fixed_via_widths=[10],
                fixed_via_heights=[8],
                min_area=10.0,
            ),
        )

        self.assertEqual(len(polygons), 1)
        self.assertEqual(polygons[0].bbox[2:], (11, 9))

    def test_via_profile_fixed_size_allows_small_mask_variation(self) -> None:
        mask = np.zeros((120, 120), dtype=np.uint8)
        cv2.rectangle(mask, (20, 20), (27, 28), 255, thickness=-1)

        polygons = extract_polygons(
            mask,
            ContourExtractionSettings(
                extraction_profile="vias",
                object_type="via",
                output_mode="box",
                via_size_mode="fixed",
                fixed_via_widths=[7],
                fixed_via_heights=[7],
                min_area=10.0,
            ),
        )

        self.assertEqual(len(polygons), 1)
        self.assertEqual(polygons[0].bbox[2:], (8, 8))

    def test_via_profile_applies_fixed_size_sets(self) -> None:
        mask = np.zeros((160, 160), dtype=np.uint8)
        cv2.rectangle(mask, (10, 10), (19, 17), 255, thickness=-1)
        cv2.rectangle(mask, (40, 10), (51, 19), 255, thickness=-1)
        cv2.rectangle(mask, (40, 40), (49, 49), 255, thickness=-1)
        cv2.rectangle(mask, (80, 10), (94, 24), 255, thickness=-1)

        polygons = extract_polygons(
            mask,
            ContourExtractionSettings(
                extraction_profile="vias",
                object_type="via",
                output_mode="box",
                via_size_mode="fixed",
                fixed_via_widths=[10, 12],
                fixed_via_heights=[8, 10],
                min_area=10.0,
            ),
        )

        self.assertEqual(len(polygons), 2)
        self.assertEqual(sorted(polygon.bbox[2:] for polygon in polygons), [(11, 9), (13, 11)])

    def test_via_profile_suppresses_intersecting_boxes(self) -> None:
        mask = np.zeros((80, 80), dtype=np.uint8)
        cv2.circle(mask, (20, 20), 5, 255, thickness=-1)
        cv2.circle(mask, (29, 29), 5, 255, thickness=-1)

        vias = extract_polygons(
            mask,
            ContourExtractionSettings(
                extraction_profile="vias",
                object_type="via",
                output_mode="box",
                min_area=10.0,
            ),
        )

        self.assertEqual(len(vias), 1)

    def test_contour_settings_parse_fixed_via_values_from_dict(self) -> None:
        settings = ContourExtractionSettings.from_dict(
            {
                "extraction_profile": "vias",
                "object_type": "via",
                "output_mode": "box",
                "via_size_mode": "fixed",
                "fixed_via_widths": "8, 10; 12",
                "fixed_via_heights": [8, "10", 12.0],
            }
        )

        self.assertEqual(settings.via_size_mode, "fixed")
        self.assertEqual(settings.fixed_via_widths, [8, 10, 12])
        self.assertEqual(settings.fixed_via_heights, [8, 10, 12])

    def test_cif_loader_reads_b_commands_as_vias(self) -> None:
        cif_path = self._artifact_path("sample_via_box.cif")
        cif_path.write_text(
            "\n".join(
                [
                    "DS 1 1 1;",
                    "L NM;",
                    "( R sample.png );",
                    "( S 2000 2000 );",
                    "B 10 8 1000 1500;",
                    "DF;",
                    "E",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        image_name, image_size, polygons = load_polygons_cif(cif_path)

        self.assertEqual(image_name, "sample.png")
        self.assertEqual(image_size, (2000, 2000))
        self.assertEqual(len(polygons), 1)
        self.assertEqual(polygons[0].category, "via")
        self.assertEqual(polygons[0].shape_hint, "box")

    def test_cif_saves_box_shapes_using_b_records(self) -> None:
        mask = np.zeros((80, 80), dtype=np.uint8)
        cv2.ellipse(mask, (30, 35), (5, 4), 0, 0, 360, 255, thickness=-1)
        polygons = extract_polygons(
            mask,
            ContourExtractionSettings(
                extraction_profile="vias",
                object_type="via",
                output_mode="box",
                min_area=10.0,
            ),
        )

        cif_path = self._artifact_path("saved_via_box.cif")
        save_polygons_cif(cif_path, "sample.png", polygons, image_size=(80, 80))
        payload = cif_path.read_text(encoding="utf-8")

        self.assertIn("B ", payload)
        self.assertNotIn("P ", payload)

    def test_cif_via_round_trip_preserves_odd_size_and_position_exactly(self) -> None:
        polygon = _rectangle_polygon(11, 13, 22, 22)
        polygon.category = "via"
        polygon.shape_hint = "box"
        cif_path = self._artifact_path("odd_via_box.cif")

        save_polygons_cif(cif_path, "sample.png", [polygon], image_size=(100, 80))
        _image_name, _image_size, loaded = load_polygons_cif(cif_path)

        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].points, polygon.points)
        self.assertEqual(loaded[0].bbox, polygon.bbox)

    def test_dataset_export_writes_image_and_cif_subdirectories(self) -> None:
        with TemporaryDirectory() as temp_root:
            root = Path(temp_root)
            image = np.zeros((32, 32, 3), dtype=np.uint8)
            image_path = root / "frame_1.png"
            cv2.imwrite(str(image_path), image)
            polygon = _rectangle_polygon(4, 4, 20, 20)

            saved_files = export_dataset_frame(root / "dataset", str(image_path), [polygon], image)

            saved_image = Path(saved_files["image"])
            saved_cif = Path(saved_files["cif"])
            self.assertEqual(saved_image.parent.name, "images")
            self.assertEqual(saved_cif.parent.name, "cif")
            self.assertTrue(saved_image.exists())
            self.assertTrue(saved_cif.exists())
            payload = saved_cif.read_text(encoding="utf-8")
            self.assertIn("( R frame_1.png );", payload)

    def test_cif_round_trip_restores_outer_hole_topology_by_default(self) -> None:
        outer_points = [(10.0, 10.0), (70.0, 10.0), (70.0, 70.0), (10.0, 70.0)]
        inner_points = [(32.0, 32.0), (48.0, 32.0), (48.0, 48.0), (32.0, 48.0)]
        outer_area, outer_perimeter, outer_bbox = compute_polygon_metrics(outer_points)
        inner_area, inner_perimeter, inner_bbox = compute_polygon_metrics(inner_points)
        outer = PolygonData(
            id=1,
            points=outer_points,
            is_hole=False,
            parent_id=None,
            category="conductor",
            shape_hint="polygon",
            area=outer_area,
            perimeter=outer_perimeter,
            bbox=outer_bbox,
        )
        hole = PolygonData(
            id=2,
            points=inner_points,
            is_hole=True,
            parent_id=1,
            category="conductor",
            shape_hint="polygon",
            area=inner_area,
            perimeter=inner_perimeter,
            bbox=inner_bbox,
        )
        cif_path = self._artifact_path("round_trip_hole.cif")
        save_polygons_cif(cif_path, "sample.png", [outer, hole], image_size=(80, 80))
        _image_name, _image_size, loaded = load_polygons_cif(cif_path)
        loaded_outers = [polygon for polygon in loaded if not polygon.is_hole]
        loaded_holes = [polygon for polygon in loaded if polygon.is_hole]

        self.assertEqual(len(loaded_outers), 1)
        self.assertEqual(len(loaded_holes), 1)
        self.assertEqual(loaded_holes[0].parent_id, loaded_outers[0].id)
        self.assertEqual(set(loaded_outers[0].points), set(outer.points))
        self.assertEqual(set(loaded_holes[0].points), set(hole.points))
        self.assertAlmostEqual(
            loaded_outers[0].area - loaded_holes[0].area,
            outer.area - hole.area,
        )
        self.assertFalse(any(polygon.description_is_invalid() for polygon in loaded))

    def test_cif_round_trip_restores_hole_topology_from_standard_cutline(self) -> None:
        outer_points = [(10.0, 10.0), (70.0, 10.0), (70.0, 70.0), (10.0, 70.0)]
        inner_points = [(32.0, 32.0), (48.0, 32.0), (48.0, 48.0), (32.0, 48.0)]
        outer_area, outer_perimeter, outer_bbox = compute_polygon_metrics(outer_points)
        inner_area, inner_perimeter, inner_bbox = compute_polygon_metrics(inner_points)
        outer = PolygonData(
            id=1,
            points=outer_points,
            is_hole=False,
            parent_id=None,
            category="conductor",
            shape_hint="polygon",
            area=outer_area,
            perimeter=outer_perimeter,
            bbox=outer_bbox,
        )
        hole = PolygonData(
            id=2,
            points=inner_points,
            is_hole=True,
            parent_id=1,
            category="conductor",
            shape_hint="polygon",
            area=inner_area,
            perimeter=inner_perimeter,
            bbox=inner_bbox,
        )
        cif_path = self._artifact_path("round_trip_hole_recover.cif")
        save_polygons_cif(cif_path, "sample.png", [outer, hole], image_size=(80, 80))
        _image_name, _image_size, loaded = load_polygons_cif(cif_path)
        self.assertTrue(any(not polygon.is_hole for polygon in loaded))
        self.assertTrue(any(polygon.is_hole for polygon in loaded))
        hole_parent_ids = {polygon.parent_id for polygon in loaded if polygon.is_hole}
        self.assertTrue(any(parent_id is not None for parent_id in hole_parent_ids))

    def test_cif_saves_polygon_holes_as_single_linked_polygon(self) -> None:
        outer_points = [(10.0, 10.0), (70.0, 10.0), (70.0, 70.0), (10.0, 70.0)]
        inner_points = [(32.0, 32.0), (48.0, 32.0), (48.0, 48.0), (32.0, 48.0)]
        outer_area, outer_perimeter, outer_bbox = compute_polygon_metrics(outer_points)
        inner_area, inner_perimeter, inner_bbox = compute_polygon_metrics(inner_points)
        outer = PolygonData(
            id=3,
            points=outer_points,
            category="conductor",
            shape_hint="polygon",
            area=outer_area,
            perimeter=outer_perimeter,
            bbox=outer_bbox,
        )
        hole = PolygonData(
            id=4,
            points=inner_points,
            is_hole=True,
            parent_id=3,
            category="conductor",
            shape_hint="polygon",
            area=inner_area,
            perimeter=inner_perimeter,
            bbox=inner_bbox,
        )
        cif_path = self._artifact_path("linked_polygon_hole.cif")

        save_polygons_cif(cif_path, "sample.png", [outer, hole], image_size=(80, 80))
        payload = cif_path.read_text(encoding="utf-8")
        _image_name, _image_size, loaded = load_polygons_cif(cif_path)

        self.assertNotIn("CONTOUR", payload)
        self.assertEqual(payload.count("\nP "), 1)
        self.assertTrue(payload.strip().endswith("E"))
        self.assertEqual(len([polygon for polygon in loaded if not polygon.is_hole]), 1)
        self.assertEqual(len([polygon for polygon in loaded if polygon.is_hole]), 1)
        self.assertFalse(any(polygon.description_is_invalid() for polygon in loaded))

    def test_cif_keyhole_round_trip_preserves_multiple_blocking_holes(self) -> None:
        from shapely import make_valid
        from shapely.geometry import Polygon as ShapelyPolygon

        outer = _rectangle_polygon(0, 0, 120, 120)
        hole_bounds = [
            (50, 50, 70, 70),
            (50, 20, 70, 40),
            (50, 80, 70, 100),
            (20, 50, 40, 70),
            (80, 50, 100, 70),
        ]
        holes: list[PolygonData] = []
        for polygon_id, bounds in enumerate(hole_bounds, start=2):
            hole = _rectangle_polygon(*bounds)
            hole.id = polygon_id
            hole.is_hole = True
            hole.parent_id = outer.id
            holes.append(hole)

        cif_path = self._artifact_path("multi_hole_keyhole.cif")
        save_polygons_cif(cif_path, "sample.png", [outer, *holes], image_size=(120, 120))
        _image_name, _image_size, loaded = load_polygons_cif(cif_path)

        loaded_outer = next(polygon for polygon in loaded if not polygon.is_hole)
        loaded_holes = [polygon for polygon in loaded if polygon.parent_id == loaded_outer.id]
        expected = make_valid(ShapelyPolygon(outer.points, holes=[hole.points for hole in holes]))
        actual = make_valid(
            ShapelyPolygon(
                loaded_outer.points,
                holes=[hole.points for hole in loaded_holes],
            )
        )
        payload = cif_path.read_text(encoding="utf-8")

        self.assertEqual(payload.count("\nP "), 1)
        self.assertNotIn("CONTOUR", payload)
        self.assertEqual(len(loaded_holes), len(holes))
        self.assertAlmostEqual(float(expected.symmetric_difference(actual).area), 0.0)

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

    def test_cif_loader_matches_klayout_fill_for_keyhole_frame_0525(self) -> None:
        """0525 vector fill must match KLayout viewer (Qt WindingFill on authored ring)."""

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

        cif_path = Path(r"D:\OZI\Нейронка\cif_metal\0525.cif")
        if not cif_path.exists():
            self.skipTest("0525.cif fixture not available")

        image_height = 2000
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
        self.assertEqual(xor_pixels, 0, msg=f"0525 layer fill differs from KLayout by {xor_pixels} pixels")

        hole = next(polygon for polygon in loaded if polygon.parent_id == outer.id)
        hole_mask, left, top = _cif_paint_mask_from_ring(outer.cif_paint_ring)
        hole_left, hole_top, hole_width, hole_height = hole.bbox
        local_x = int(round(hole_left + hole_width / 2)) - left
        local_y = int(round(hole_top + hole_height / 2)) - top
        self.assertGreaterEqual(local_x, 0)
        self.assertGreaterEqual(local_y, 0)
        self.assertLess(local_x, hole_mask.shape[1])
        self.assertLess(local_y, hole_mask.shape[0])
        self.assertGreater(int(hole_mask[local_y, local_x]), 0, msg="keyhole slot interior must be filled")

    def test_cif_vector_display_path_uses_authored_ring_for_keyhole(self) -> None:
        from PyQt6.QtCore import Qt

        from contour.graphics_items import _vector_display_path_for_polygon
        from contour.serializers import _dedupe_closed_points, _has_duplicate_points

        cif_path = Path(r"D:\OZI\Нейронка\cif_metal\0525.cif")
        if not cif_path.exists():
            self.skipTest("0525.cif fixture not available")

        clear_cif_parse_cache()
        _image_name, _image_size, loaded = load_polygons_cif(cif_path)
        keyhole_outer = next(
            polygon
            for polygon in loaded
            if not polygon.is_hole
            and polygon.cif_paint_ring
            and _has_duplicate_points(_dedupe_closed_points(polygon.cif_paint_ring))
        )
        holes = [polygon for polygon in loaded if polygon.parent_id == keyhole_outer.id]
        display_path = _vector_display_path_for_polygon(keyhole_outer, cutout_polygons=holes)
        self.assertEqual(display_path.elementCount(), len(keyhole_outer.cif_paint_ring) + 1)
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

    def test_cif_loader_splits_right_edge_keyholes_instead_of_wrapping_them(self) -> None:
        cif_path = self._artifact_path("right_edge_multi_hole.cif")
        cif_path.write_text(
            "\n".join(
                [
                    "DS 1 1 1;",
                    "L NM;",
                    "( R sample.png );",
                    "( S 120 100 );",
                    "P 20 50 100 50 30 50 30 55 25 55 25 45 30 45 30 50 100 50 "
                    "60 50 60 55 55 55 55 45 60 45 60 50 100 50 20 50 20 60 0 60 0 40 20 40 20 50;",
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
        hole_spans = sorted(max(point[0] for point in hole.points) - min(point[0] for point in hole.points) for hole in holes)
        self.assertEqual(hole_spans, [5.0, 5.0])
        for hole in holes:
            xs = [point[0] for point in hole.points]
            self.assertLessEqual(max(xs), 60)
            self.assertGreaterEqual(min(xs), 25)
            self.assertLess(_longest_axis_aligned_edge(hole.points), 40)
        outer_xs = [point[0] for point in outers[0].points]
        self.assertLessEqual(max(outer_xs), 20)
        self.assertFalse(any(polygon.description_is_invalid() for polygon in loaded))

    def test_cif_loader_keeps_self_crossing_ring_as_invalid_description(self) -> None:
        cif_path = self._artifact_path("self_crossing_bowtie.cif")
        cif_path.write_text(
            "\n".join(
                [
                    "DS 1 1 1;",
                    "L NM;",
                    "( R sample.png );",
                    "( S 40 40 );",
                    "P 0 0 30 30 30 0 0 30;",
                    "DF;",
                    "E",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        _image_name, _image_size, loaded = load_polygons_cif(cif_path)

        self.assertEqual(len(loaded), 1)
        self.assertTrue(loaded[0].description_is_invalid())

    def test_cif_loader_skips_legacy_raster_recovery_for_plain_polygons(self) -> None:
        cif_path = self._artifact_path("plain_polygon_fast_load.cif")
        cif_path.write_text(
            "\n".join(
                [
                    "DS 1 1 1;",
                    "L NM;",
                    "( R sample.png );",
                    "( S 100 100 );",
                    "P 10 90 90 90 90 10 10 10 10 90;",
                    "DF;",
                    "E",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        original_morphology_ex = cv2.morphologyEx

        def fail_morphology_ex(*_args, **_kwargs):
            raise AssertionError("plain CIF polygon load should not run legacy raster recovery")

        try:
            cv2.morphologyEx = fail_morphology_ex
            _image_name, _image_size, loaded = load_polygons_cif(cif_path)
        finally:
            cv2.morphologyEx = original_morphology_ex

        self.assertEqual(len(loaded), 1)
        self.assertFalse(loaded[0].is_hole)

    def test_cv_round_trip_saves_via_as_ellipse_point(self) -> None:
        polygon = _rectangle_polygon(10, 12, 22, 28)
        polygon.category = "via"
        polygon.shape_hint = "box"
        cv_path = self._artifact_path("sample_via.cv")

        save_polygons_cv(cv_path, "sample.png", [polygon], image_size=(64, 48))
        payload = cv_path.read_text(encoding="utf-8")
        image_name, image_size, loaded = load_polygons_cv(cv_path)

        self.assertIn('"type": "Point"', payload)
        self.assertIn('"shape": "ellipse"', payload)
        self.assertIn('"diagonals"', payload)
        self.assertNotIn('"features"', payload)
        self.assertNotIn('"properties"', payload)
        self.assertNotIn('"metadata"', payload)
        self.assertEqual(image_name, "sample.png")
        self.assertEqual(image_size, (64, 48))
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].category, "via")
        self.assertEqual(loaded[0].shape_hint, "box")
        self.assertEqual(loaded[0].bbox, polygon.bbox)

    def test_cv_via_round_trip_preserves_odd_size_and_position_exactly(self) -> None:
        polygon = _rectangle_polygon(11, 13, 22, 22)
        polygon.category = "via"
        polygon.shape_hint = "box"
        cv_path = self._artifact_path("odd_via_box.cv")

        save_polygons_cv(cv_path, "sample.png", [polygon], image_size=(100, 80))
        _image_name, _image_size, loaded = load_polygons_cv(cv_path)

        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].points, polygon.points)
        self.assertEqual(loaded[0].bbox, polygon.bbox)

    def test_cv_loader_reads_rectangle_point(self) -> None:
        cv_path = self._artifact_path("sample_rectangle.cv")
        cv_path.write_text(
            """
{
  "format": "contour-vector",
  "image": {"path": "sample.png", "size": [100, 80]},
  "objects": [
    {
      "type": "Point",
      "id": 7,
      "shape": "rectangle",
      "coordinates": [4, 5, 14, 25]
    }
  ]
}
""".strip(),
            encoding="utf-8",
        )

        image_name, image_size, polygons = load_polygons_vector(cv_path)

        self.assertEqual(image_name, "sample.png")
        self.assertEqual(image_size, (100, 80))
        self.assertEqual(len(polygons), 1)
        self.assertEqual(polygons[0].id, 7)
        self.assertEqual(polygons[0].category, "conductor")
        self.assertEqual(polygons[0].shape_hint, "box")
        self.assertEqual(polygons[0].bbox, (4, 5, 11, 21))

    def test_cv_saves_polygon_holes_as_coordinate_rings(self) -> None:
        outer_points = [(10.0, 10.0), (70.0, 10.0), (70.0, 70.0), (10.0, 70.0)]
        inner_points = [(32.0, 32.0), (48.0, 32.0), (48.0, 48.0), (32.0, 48.0)]
        outer_area, outer_perimeter, outer_bbox = compute_polygon_metrics(outer_points)
        inner_area, inner_perimeter, inner_bbox = compute_polygon_metrics(inner_points)
        outer = PolygonData(
            id=3,
            points=outer_points,
            category="conductor",
            shape_hint="polygon",
            area=outer_area,
            perimeter=outer_perimeter,
            bbox=outer_bbox,
        )
        hole = PolygonData(
            id=4,
            points=inner_points,
            is_hole=True,
            parent_id=3,
            category="conductor",
            shape_hint="polygon",
            area=inner_area,
            perimeter=inner_perimeter,
            bbox=inner_bbox,
        )
        cv_path = self._artifact_path("compact_polygon_hole.cv")

        save_polygons_cv(cv_path, "sample.png", [outer, hole], image_size=(80, 80))
        payload = json.loads(cv_path.read_text(encoding="utf-8"))
        _image_name, _image_size, loaded = load_polygons_cv(cv_path)

        self.assertEqual(payload["objects"][0]["type"], "Polygon")
        self.assertEqual(payload["objects"][0]["id"], 3)
        self.assertEqual(len(payload["objects"][0]["coordinates"]), 2)
        self.assertEqual(payload["objects"][0]["coordinates"][0][0], [10, 10])
        self.assertEqual(payload["objects"][0]["coordinates"][0][-1], [10, 10])
        self.assertEqual(payload["objects"][0]["coordinates"][1][0], [32, 32])
        raw_payload = cv_path.read_text(encoding="utf-8")
        self.assertIn("[[10, 10], [70, 10], [70, 70], [10, 70], [10, 10]]", raw_payload)
        self.assertIn('  "format"', raw_payload)
        self.assertNotIn('    "format"', raw_payload)
        self.assertEqual(len(loaded), 2)
        self.assertTrue(any(not polygon.is_hole for polygon in loaded))
        loaded_holes = [polygon for polygon in loaded if polygon.is_hole]
        self.assertEqual(len(loaded_holes), 1)
        self.assertEqual(loaded_holes[0].parent_id, 3)

    def test_cv_loader_generates_hole_ids_without_colliding_with_object_ids(self) -> None:
        cv_path = self._artifact_path("hole_id_collision.cv")
        cv_path.write_text(
            """
{
  "format": "contour-vector",
  "version": 2,
  "image": {"path": "sample.png", "size": [100, 100]},
  "objects": [
    {
      "type": "Polygon",
      "id": 1,
      "coordinates": [
        [[10, 10], [60, 10], [60, 60], [10, 60], [10, 10]],
        [[25, 25], [40, 25], [40, 40], [25, 40], [25, 25]]
      ]
    },
    {
      "type": "Polygon",
      "id": 2,
      "coordinates": [
        [[70, 10], [90, 10], [90, 30], [70, 30], [70, 10]]
      ]
    }
  ]
}
""".strip(),
            encoding="utf-8",
        )

        _image_name, _image_size, loaded = load_polygons_cv(cv_path)

        ids = [polygon.id for polygon in loaded]
        self.assertEqual(len(ids), len(set(ids)))
        hole = next(polygon for polygon in loaded if polygon.is_hole)
        self.assertEqual(hole.parent_id, 1)
        self.assertNotIn(hole.id, {1, 2})

    def test_cv_writes_up_to_eight_vertices_per_line(self) -> None:
        points = [(float(index), float(index + 10)) for index in range(9)]
        area, perimeter, bbox = compute_polygon_metrics(points)
        polygon = PolygonData(id=9, points=points, area=area, perimeter=perimeter, bbox=bbox)
        cv_path = self._artifact_path("compact_rows.cv")

        save_polygons_cv(cv_path, "sample.png", [polygon], image_size=(100, 100))
        payload = cv_path.read_text(encoding="utf-8")

        self.assertIn(
            "[0, 10], [1, 11], [2, 12], [3, 13], [4, 14], [5, 15], [6, 16], [7, 17]",
            payload,
        )
        self.assertIn("[8, 18], [0, 10]", payload)

    def test_result_bundle_can_save_cv_without_legacy_text_formats(self) -> None:
        with TemporaryDirectory() as temp_root:
            root = Path(temp_root)
            image = np.zeros((32, 32, 3), dtype=np.uint8)
            polygon = _rectangle_polygon(4, 4, 20, 20)

            saved = save_result_bundle(
                root,
                "sample.png",
                [polygon],
                image,
                DisplaySettings(),
                SaveOptions(save_cif=False, save_cv=True, save_preview=True),
            )

            self.assertEqual(set(saved), {"cv", "preview"})
            self.assertTrue(Path(saved["cv"]).exists())
            self.assertNotIn('"metadata"', Path(saved["cv"]).read_text(encoding="utf-8"))
            self.assertTrue(Path(saved["preview"]).exists())
            self.assertFalse((root / "sample.csv").exists())
            self.assertFalse((root / "sample.txt").exists())
            self.assertFalse((root / "sample.svg").exists())


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


class BoundaryVertexSnapTests(unittest.TestCase):
    def test_snap_picks_nearest_boundary_vertex_by_x_window(self) -> None:
        from contour.serializers import _snap_hole_to_boundary_vertex

        outer = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
        snap = _snap_hole_to_boundary_vertex(outer, (48.0, 32.0))
        self.assertIsNotNone(snap)
        outer_anchor, boundary_index = snap
        self.assertEqual(outer_anchor, (100.0, 0.0))
        self.assertEqual(boundary_index, 1)

    def test_snap_searches_x_sorted_neighbors_outside_hull(self) -> None:
        from contour.serializers import _snap_hole_to_boundary_vertex

        outer = [(0.0, 0.0), (40.0, 0.0), (40.0, 100.0), (0.0, 100.0)]
        snap = _snap_hole_to_boundary_vertex(outer, (60.0, 50.0))
        self.assertIsNotNone(snap)
        outer_anchor, _boundary_index = snap
        self.assertEqual(outer_anchor, (40.0, 0.0))

    def test_snap_returns_none_for_degenerate_boundary(self) -> None:
        from contour.serializers import _snap_hole_to_boundary_vertex

        snap = _snap_hole_to_boundary_vertex([(0.0, 0.0), (10.0, 0.0)], (5.0, 5.0))
        self.assertIsNone(snap)

    def test_snap_prefers_closest_right_side_vertex_in_local_x_window(self) -> None:
        from contour.serializers import _snap_hole_to_boundary_vertex

        outer = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
        snap = _snap_hole_to_boundary_vertex(outer, (10.0, 10.0))
        self.assertIsNotNone(snap)
        outer_anchor, _boundary_index = snap
        self.assertEqual(outer_anchor, (100.0, 0.0))

    def test_vertex_slit_can_attach_to_linked_neighbor_hole(self) -> None:
        from shapely import make_valid
        from shapely.geometry import LineString, Polygon as ShapelyPolygon

        from contour.serializers import _ring_with_orientation, _select_vertex_slit_for_hole

        outer = _ring_with_orientation(
            [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)],
            positive_area=True,
        )
        right_hole = _ring_with_orientation(
            [(80.0, 50.0), (100.0, 50.0), (100.0, 70.0), (80.0, 70.0)],
            positive_area=False,
        )
        center_hole = _ring_with_orientation(
            [(50.0, 50.0), (70.0, 50.0), (70.0, 70.0), (50.0, 70.0)],
            positive_area=False,
        )
        parent_solid = make_valid(ShapelyPolygon(outer, holes=[right_hole, center_hole]))

        outer_only = _select_vertex_slit_for_hole(
            center_hole,
            parent_solid,
            outer=outer,
            linked_hole_rings=[],
        )
        self.assertIsNotNone(outer_only)
        self.assertEqual(outer_only[0], -1)

        neighbor = _select_vertex_slit_for_hole(
            center_hole,
            parent_solid,
            outer=outer,
            linked_hole_rings=[(0, right_hole)],
        )
        self.assertIsNotNone(neighbor)
        self.assertEqual(neighbor[0], 0)
        self.assertTrue(
            parent_solid.covers(LineString([neighbor[1], center_hole[neighbor[3]]])),
        )
        self.assertLess(
            (neighbor[1][0] - center_hole[neighbor[3]][0]) ** 2
            + (neighbor[1][1] - center_hole[neighbor[3]][1]) ** 2,
            2500.0,
        )


if __name__ == "__main__":
    unittest.main()
