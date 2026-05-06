from __future__ import annotations

import unittest

from contour.application.vector_geometry_postprocess import (
    VectorGeometrySettings,
    apply_vertex_position_to_clone,
    clip_polygons_to_frame_raster,
    dissolve_small_holes,
    drop_triangle_outer_artifacts,
    merge_overlapping_root_families,
    postprocess_after_editor_mutation,
    postprocess_polygons_for_frame_navigation,
    remove_spikes_from_polygon_ring,
)
from contour.domain import PolygonData, compute_polygon_metrics
from contour.graphics.editor_scene import PolygonEditorScene


def _rect(left: float, top: float, right: float, bottom: float, pid: int) -> PolygonData:
    pts = [
        (left, top),
        (right, top),
        (right, bottom),
        (left, bottom),
    ]
    a, per, bbox = compute_polygon_metrics(pts)
    return PolygonData(id=pid, points=pts, area=a, perimeter=per, bbox=bbox)


class VectorGeometryPostprocessTests(unittest.TestCase):
    def test_clip_intersecting_rectangle_trims_geometry(self) -> None:
        square = _rect(-10.0, -10.0, 50.0, 50.0, 1)
        out = clip_polygons_to_frame_raster([square], 40, 40)
        self.assertTrue(out)
        all_x = [x for p in out for x, _ in p.points]
        all_y = [y for p in out for _, y in p.points]
        self.assertGreaterEqual(min(all_x), 0.0)
        self.assertGreaterEqual(min(all_y), 0.0)
        self.assertLessEqual(max(all_x), 40.0)
        self.assertLessEqual(max(all_y), 40.0)

    def test_remove_polygon_fully_outside_frame(self) -> None:
        far = _rect(200.0, 200.0, 250.0, 250.0, 1)
        inside = _rect(10.0, 10.0, 30.0, 30.0, 2)
        vg = VectorGeometrySettings(min_outer_area_px2=1.0, min_hole_area_to_remove_px2=0.1)
        merged, changed = postprocess_polygons_for_frame_navigation([far, inside], 100, 100, vg)
        self.assertTrue(changed)
        self.assertEqual(len([p for p in merged if not p.is_hole]), 1)

    def test_preserve_inside_polygon_through_clip(self) -> None:
        poly = _rect(5.0, 5.0, 15.0, 15.0, 9)
        vg = VectorGeometrySettings(min_outer_area_px2=4.0)
        out, _changed = postprocess_polygons_for_frame_navigation([poly], 32, 32, vg)
        outer_areas = [abs(float(p.area)) for p in out if not p.is_hole]
        self.assertTrue(max(outer_areas, default=0.0) > 70.0)

    def test_remove_small_outer_polygon(self) -> None:
        big = _rect(0.0, 0.0, 80.0, 80.0, 1)
        tiny = _rect(82.0, 82.0, 83.5, 83.5, 2)
        vg = VectorGeometrySettings(clip_to_frame_on_sync=False, min_outer_area_px2=50.0)
        out, changed = postprocess_polygons_for_frame_navigation([big, tiny], 200, 200, vg)
        self.assertTrue(changed)
        outers = [p for p in out if not p.is_hole and p.category != "via"]
        self.assertGreaterEqual(len(outers), 1)
        self.assertTrue(all(abs(float(p.area)) >= 49.5 for p in outers))

    def test_preserves_large_outer_polygon(self) -> None:
        big = _rect(0.0, 0.0, 70.0, 70.0, 11)
        vg = VectorGeometrySettings(min_outer_area_px2=400.0, clip_to_frame_on_sync=False)
        out, _ = postprocess_polygons_for_frame_navigation([big], 200, 200, vg)
        self.assertTrue(any(abs(float(p.area)) >= 4900 * 0.99 for p in out if not p.is_hole))

    def test_dissolves_small_hole(self) -> None:
        outer = PolygonData(
            id=1,
            points=[(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)],
            is_hole=False,
            area=10000.0,
            perimeter=400.0,
            bbox=(0, 0, 100, 100),
        )
        hole = PolygonData(
            id=2,
            points=[(40.0, 40.0), (43.0, 40.0), (43.0, 43.0), (40.0, 43.0)],
            is_hole=True,
            parent_id=1,
            category="conductor",
            area=9.0,
            perimeter=12.0,
            bbox=(40, 40, 4, 4),
        )
        out = dissolve_small_holes([outer, hole], min_area_px2=20.0)
        self.assertEqual(len(out), 1)
        self.assertFalse(out[0].is_hole)

    def test_keeps_large_hole(self) -> None:
        outer = PolygonData(
            id=1,
            points=[(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)],
            is_hole=False,
            area=10000.0,
            perimeter=400.0,
            bbox=(0, 0, 100, 100),
        )
        hole = PolygonData(
            id=2,
            points=[(40.0, 40.0), (70.0, 40.0), (70.0, 70.0), (40.0, 70.0)],
            is_hole=True,
            parent_id=1,
            category="conductor",
            area=900.0,
            perimeter=120.0,
            bbox=(40, 40, 31, 31),
        )
        out = dissolve_small_holes([outer, hole], min_area_px2=20.0)
        self.assertEqual(len(out), 2)

    def test_remove_spike_from_ring(self) -> None:
        spiked = [(0.0, 0.0), (30.0, 0.0), (30.1, -35.0), (31.0, 0.0), (60.0, 0.0), (60.0, 40.0), (0.0, 40.0)]
        cleaned = remove_spikes_from_polygon_ring(spiked, min_interior_angle_deg=40.0)
        self.assertLess(len(cleaned), len(spiked))
        self.assertGreaterEqual(len(cleaned), 4)

    def test_drop_three_vertex_triangle_artifact_when_enabled(self) -> None:
        tri = PolygonData(
            id=5,
            points=[(100.0, 100.0), (122.0, 100.0), (111.0, 120.0)],
            is_hole=False,
            category="conductor",
            shape_hint="polygon",
            area=420.0,
            bbox=(99, 99, 25, 25),
        )
        out_drop = drop_triangle_outer_artifacts([tri], enabled=True, min_outer_area_px2=500.0)
        self.assertFalse(out_drop)

    def test_preserves_large_triangle_when_above_threshold(self) -> None:
        tri = PolygonData(
            id=5,
            points=[(100.0, 100.0), (122.0, 100.0), (111.0, 120.0)],
            is_hole=False,
            category="conductor",
            shape_hint="polygon",
            area=420.0,
            bbox=(99, 99, 25, 25),
        )
        kept = drop_triangle_outer_artifacts([tri], enabled=True, min_outer_area_px2=100.0)
        self.assertEqual(len(kept), 1)

    def test_merge_overlapping_rectangles_into_one_topology(self) -> None:
        a = _rect(0.0, 0.0, 70.0, 70.0, 101)
        b = _rect(35.0, 35.0, 95.0, 95.0, 102)
        merged = merge_overlapping_root_families([a, b])
        roots = [p for p in merged if not p.is_hole and p.parent_id is None]
        self.assertEqual(len(roots), 1)

    def test_set_polygons_clears_selection(self) -> None:
        scene = PolygonEditorScene()
        sq = _rect(0.0, 0.0, 40.0, 40.0, 77)
        scene.set_polygons([sq, _rect(50.0, 50.0, 55.0, 55.0, 88)])
        self.assertIsNone(scene.selected_polygon_id())
        self.assertEqual(len(scene.get_polygons()), 2)

    def test_geometry_postprocess_with_no_changes_reports_clean(self) -> None:
        poly = _rect(5.0, 5.0, 35.0, 35.0, 1)
        out, changed = postprocess_polygons_for_frame_navigation(
            [poly],
            100,
            100,
            VectorGeometrySettings(
                clip_to_frame_on_sync=False,
                min_outer_area_px2=1.0,
                min_spike_interior_angle_deg=0.0,
            ),
        )
        self.assertFalse(changed)
        self.assertEqual(len(out), 1)

    def test_geometry_postprocess_with_real_changes_reports_dirty(self) -> None:
        tiny = _rect(5.0, 5.0, 6.0, 6.0, 1)
        out, changed = postprocess_polygons_for_frame_navigation(
            [tiny],
            100,
            100,
            VectorGeometrySettings(min_outer_area_px2=10.0, min_spike_interior_angle_deg=0.0),
        )
        self.assertTrue(changed)
        self.assertFalse(out)

    def test_vertex_move_valid_polygon_succeeds(self) -> None:
        poly = _rect(0.0, 0.0, 40.0, 40.0, 1)
        moved = apply_vertex_position_to_clone([poly], 1, 1, (50.0, 0.0))
        self.assertEqual(moved[0].points[1], (50.0, 0.0))

    def test_vertex_move_invalid_polygon_is_rejected(self) -> None:
        poly = _rect(0.0, 0.0, 40.0, 40.0, 1)
        moved = apply_vertex_position_to_clone([poly], 1, 1, (0.0, 40.0))
        self.assertEqual(moved[0].points, poly.points)

    def test_vertex_move_causing_merge_merges_when_enabled(self) -> None:
        left = _rect(0.0, 0.0, 40.0, 40.0, 1)
        right = _rect(50.0, 0.0, 90.0, 40.0, 2)
        moved = apply_vertex_position_to_clone([left, right], 1, 1, (60.0, 0.0))
        processed, changed = postprocess_after_editor_mutation(
            moved,
            VectorGeometrySettings(merge_overlapping_on_edit=True, min_outer_area_px2=1.0, min_spike_interior_angle_deg=0.0),
            include_merge=True,
        )
        roots = [p for p in processed if p.parent_id is None and not p.is_hole]
        self.assertTrue(changed)
        self.assertEqual(len(roots), 1)


if __name__ == "__main__":
    unittest.main()
