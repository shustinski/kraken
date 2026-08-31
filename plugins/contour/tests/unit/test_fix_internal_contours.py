from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from contour.application.fix_internal_contours import (
    analyze_internal_contour_display,
    fix_internal_contour_display,
)
from contour.domain import PolygonData, compute_polygon_metrics
from contour.serializers import (
    CIF_CUTOUT_DISPLAY_MARKER,
    clear_cif_parse_cache,
    load_polygons_cif,
    save_polygons_cif,
    _polygon_uses_authored_cif_paint_ring,
)

_KEYHOLE_CIF = """
DS 1 1 1;
L NM;
( R sample.png );
( S 100 100 );
P 90 50 90 50 40 50 40 40 20 40 20 60 40 60 40 50 90 50 90 10 10 10 10 90 90 90 90 50;
DF;
E
""".strip()


def _rectangle_polygon(left: int, top: int, right: int, bottom: int, polygon_id: int = 1) -> PolygonData:
    points = [
        (float(left), float(top)),
        (float(right), float(top)),
        (float(right), float(bottom)),
        (float(left), float(bottom)),
    ]
    area, perimeter, bbox = compute_polygon_metrics(points)
    return PolygonData(
        id=polygon_id,
        points=points,
        area=area,
        perimeter=perimeter,
        bbox=bbox,
    )


def _load_sample_keyhole() -> tuple[tuple[int, int], list[PolygonData]]:
    with TemporaryDirectory() as temp_dir:
        cif_path = Path(temp_dir) / "keyhole.cif"
        cif_path.write_text(_KEYHOLE_CIF + "\n", encoding="utf-8")
        clear_cif_parse_cache()
        _image_name, image_size, loaded = load_polygons_cif(cif_path)
        return image_size, [polygon.clone() for polygon in loaded]


class FixInternalContoursTests(unittest.TestCase):
    def test_analyze_detects_cutout_paint_mismatch_for_multi_hole_family(self) -> None:
        image_size, polygons = _load_sample_keyhole()
        outer = next(polygon for polygon in polygons if not polygon.is_hole)
        extra_hole = _rectangle_polygon(70, 70, 85, 85, polygon_id=99)
        extra_hole.is_hole = True
        extra_hole.parent_id = outer.id
        polygons.append(extra_hole)

        analysis = analyze_internal_contour_display(polygons, image_size)
        self.assertTrue(analysis.needs_fix)
        self.assertEqual(len(analysis.issues), 1)

    def test_fix_saves_cutout_display_marker(self) -> None:
        image_size, polygons = _load_sample_keyhole()
        outer = next(polygon for polygon in polygons if not polygon.is_hole)
        extra_hole = _rectangle_polygon(70, 70, 85, 85, polygon_id=99)
        extra_hole.is_hole = True
        extra_hole.parent_id = outer.id
        polygons.append(extra_hole)

        fixed, analysis, changed = fix_internal_contour_display(polygons, image_size)
        self.assertTrue(changed)
        self.assertEqual(len(analysis.issues), 1)

        with TemporaryDirectory() as temp_dir:
            cif_path = Path(temp_dir) / "fixed_keyhole.cif"
            save_polygons_cif(
                cif_path,
                "sample.png",
                fixed,
                image_size=image_size,
                cutout_display=True,
            )
            payload = cif_path.read_text(encoding="utf-8")
            self.assertIn(CIF_CUTOUT_DISPLAY_MARKER, payload)
            clear_cif_parse_cache()
            _image_name, _image_size, reloaded = load_polygons_cif(cif_path)
            reloaded_outer = next(polygon for polygon in reloaded if not polygon.is_hole)
            self.assertFalse(_polygon_uses_authored_cif_paint_ring(reloaded_outer))

    def test_klayout_keyhole_slot_is_skipped_on_0525(self) -> None:
        cif_path = Path(r"D:\OZI\Нейронка\cif_metal\0525.cif")
        if not cif_path.exists():
            self.skipTest("0525.cif fixture not available")
        clear_cif_parse_cache()
        _image_name, image_size, loaded = load_polygons_cif(cif_path)
        analysis = analyze_internal_contour_display(loaded, image_size)
        self.assertGreater(analysis.skipped_klayout_keyholes, 0)
        self.assertFalse(analysis.needs_fix)


if __name__ == "__main__":
    unittest.main()
