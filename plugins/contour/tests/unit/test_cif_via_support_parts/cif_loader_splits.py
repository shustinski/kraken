"""test_cif_via_support: test cif loader splits."""
from __future__ import annotations
import cv2
from contour.serializers import (
    load_polygons_cif,
)

from test_cif_via_support_parts.lib_open_cif import _longest_axis_aligned_edge


class CifLoaderSplitsMixin:
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
