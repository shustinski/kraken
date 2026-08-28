from __future__ import annotations

import unittest

from contour.application.vector_geometry_postprocess import (
    VectorGeometrySettings,
    apply_vertex_delete_to_clone,
    apply_vertex_position_to_clone,
    clip_polygons_to_frame_raster,
    dissolve_self_intersecting_polygons,
    dissolve_small_holes,
    drop_triangle_outer_artifacts,
    merge_overlapping_root_families,
    polygon_description_is_invalid,
    polygons_needing_repair,
    postprocess_after_editor_mutation,
    postprocess_after_vertex_move,
    postprocess_changed_polygon_edit,
    postprocess_changed_polygon_only,
    postprocess_polygons_for_frame_navigation,
    remove_spikes_from_polygon_ring,
    repair_invalid_polygon_descriptions,
    summarize_invalid_polygon_description_reasons,
    union_after_removing_polygon_ids,
)
from contour.domain import PolygonData, compute_polygon_metrics
from contour.domain.polygon_ring import is_valid_closed_polygon_ring


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
    def test_default_manual_tool_postprocess_settings(self) -> None:
        settings = VectorGeometrySettings()
        self.assertTrue(settings.clip_to_frame_on_sync)
        self.assertEqual(settings.min_outer_area_px2, 60_000.0)
        self.assertEqual(settings.min_hole_area_to_remove_px2, 100_000.0)
        self.assertTrue(settings.merge_overlapping_on_edit)
        self.assertEqual(settings.min_spike_interior_angle_deg, 30.0)
        self.assertTrue(settings.drop_three_vertex_triangle_artifacts)

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

    def test_merge_overlapping_skips_vias(self) -> None:
        conductor = _rect(0.0, 0.0, 70.0, 70.0, 101)
        via = _rect(20.0, 20.0, 40.0, 40.0, 102)
        via.category = "via"
        via.shape_hint = "box"
        merged = merge_overlapping_root_families([conductor, via])
        self.assertEqual({polygon.id for polygon in merged}, {101, 102})
        self.assertEqual(next(polygon.category for polygon in merged if polygon.id == 102), "via")

    def test_union_after_removing_hole_merges_island(self) -> None:
        outer = _rect(0.0, 0.0, 80.0, 80.0, 1)
        island = _rect(30.0, 30.0, 50.0, 50.0, 3)
        island.parent_id = 2
        remaining = union_after_removing_polygon_ids([outer, island], {2})
        filled = [polygon for polygon in remaining if not polygon.is_hole]
        self.assertEqual(len(filled), 1)

    def test_dissolve_self_crossing_after_vertex_delete(self) -> None:
        slot = PolygonData(
            id=1,
            points=[
                (0.0, 0.0),
                (100.0, 0.0),
                (100.0, 20.0),
                (30.0, 20.0),
                (30.0, 40.0),
                (100.0, 40.0),
                (100.0, 60.0),
                (0.0, 60.0),
            ],
        )
        area, perimeter, bbox = compute_polygon_metrics(slot.points)
        slot.area = area
        slot.perimeter = perimeter
        slot.bbox = bbox
        after_delete = apply_vertex_delete_to_clone([slot], 1, 0)
        self.assertFalse(is_valid_closed_polygon_ring(after_delete[0].points))
        healed = dissolve_self_intersecting_polygons(after_delete)
        self.assertTrue(healed)
        self.assertTrue(all(is_valid_closed_polygon_ring(polygon.points) for polygon in healed))
        filled_area = sum(abs(float(polygon.area)) for polygon in healed if not polygon.is_hole)
        self.assertGreater(filled_area, 1000.0)

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

    def test_vertex_move_updates_both_closed_duplicate_endpoints(self) -> None:
        points = [(0.0, 0.0), (40.0, 0.0), (40.0, 40.0), (0.0, 40.0), (0.0, 0.0)]
        area, perimeter, bbox = compute_polygon_metrics(points)
        poly = PolygonData(id=1, points=points, area=area, perimeter=perimeter, bbox=bbox)

        moved = apply_vertex_position_to_clone([poly], 1, 0, (5.0, 5.0))

        self.assertEqual(moved[0].points[0], (5.0, 5.0))
        self.assertEqual(moved[0].points[-1], (5.0, 5.0))

    def test_vertex_move_invalid_polygon_is_rejected(self) -> None:
        poly = _rect(0.0, 0.0, 40.0, 40.0, 1)
        moved = apply_vertex_position_to_clone([poly], 1, 1, (0.0, 40.0))
        self.assertEqual(moved[0].points, poly.points)

    def test_vertex_move_allows_unrelated_existing_ring_defect(self) -> None:
        points = [(0.0, 0.0), (40.0, 0.0), (40.0, 40.0), (10.0, 10.0), (0.0, 40.0), (10.0, 10.0)]
        area, perimeter, bbox = compute_polygon_metrics(points)
        poly = PolygonData(id=1, points=points, area=area, perimeter=perimeter, bbox=bbox)

        moved = apply_vertex_position_to_clone([poly], 1, 1, (45.0, 0.0))

        self.assertEqual(moved[0].points[1], (45.0, 0.0))

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

    def test_postprocess_after_vertex_move_merges_when_enabled(self) -> None:
        left = _rect(0.0, 0.0, 40.0, 40.0, 1)
        right = _rect(50.0, 0.0, 90.0, 40.0, 2)
        moved = apply_vertex_position_to_clone([left, right], 1, 1, (60.0, 0.0))
        processed, changed = postprocess_after_vertex_move(
            moved,
            VectorGeometrySettings(
                merge_overlapping_on_edit=True,
                min_outer_area_px2=1.0,
                min_spike_interior_angle_deg=0.0,
            ),
            polygon_id=1,
        )
        roots = [p for p in processed if p.parent_id is None and not p.is_hole]
        self.assertTrue(changed)
        self.assertEqual(len(roots), 1)

    def test_postprocess_after_vertex_move_merges_even_when_setting_disabled(self) -> None:
        left = _rect(0.0, 0.0, 40.0, 40.0, 1)
        right = _rect(50.0, 0.0, 90.0, 40.0, 2)
        moved = apply_vertex_position_to_clone([left, right], 1, 1, (60.0, 0.0))
        processed, changed = postprocess_after_vertex_move(
            moved,
            VectorGeometrySettings(
                merge_overlapping_on_edit=False,
                min_outer_area_px2=1.0,
                min_spike_interior_angle_deg=0.0,
            ),
            polygon_id=1,
        )
        roots = [p for p in processed if p.parent_id is None and not p.is_hole]
        self.assertTrue(changed)
        self.assertEqual(len(roots), 1)

    def test_postprocess_changed_polygon_only_touches_target(self) -> None:
        large = _rect(0.0, 0.0, 100.0, 100.0, 1)
        tiny = _rect(120.0, 0.0, 121.0, 1.0, 2)
        processed, changed = postprocess_changed_polygon_only(
            [large, tiny],
            VectorGeometrySettings(min_outer_area_px2=50.0, min_spike_interior_angle_deg=0.0),
            polygon_id=1,
        )
        self.assertFalse(changed)
        self.assertEqual({p.id for p in processed}, {1, 2})

    def test_vertex_move_preserves_small_inner_hole_target(self) -> None:
        outer = _rect(0.0, 0.0, 100.0, 100.0, 1)
        hole = _rect(40.0, 40.0, 42.0, 42.0, 2)
        hole.is_hole = True
        hole.parent_id = outer.id
        moved = apply_vertex_position_to_clone([outer, hole], 2, 1, (43.0, 40.0))

        processed, changed = postprocess_changed_polygon_only(
            moved,
            VectorGeometrySettings(min_hole_area_to_remove_px2=11.0, min_spike_interior_angle_deg=0.0),
            polygon_id=2,
        )

        self.assertFalse(changed)
        moved_hole = next((polygon for polygon in processed if polygon.id == 2), None)
        self.assertIsNotNone(moved_hole)
        self.assertTrue(moved_hole.is_hole)
        self.assertEqual(moved_hole.points[1], (43, 40))

    def test_postprocess_changed_polygon_edit_rejects_too_small_outer(self) -> None:
        tiny = _rect(0.0, 0.0, 2.0, 2.0, 1)
        processed, accepted, changed = postprocess_changed_polygon_edit(
            [tiny],
            VectorGeometrySettings(min_outer_area_px2=50.0, min_spike_interior_angle_deg=0.0),
            polygon_id=1,
        )
        self.assertFalse(accepted)
        self.assertFalse(changed)
        self.assertEqual(len(processed), 1)
        self.assertEqual(processed[0].points, tiny.points)

    def test_postprocess_changed_polygon_only_keeps_polygon_when_filters_fail(self) -> None:
        tiny = _rect(0.0, 0.0, 2.0, 2.0, 1)
        processed, changed = postprocess_changed_polygon_only(
            [tiny],
            VectorGeometrySettings(min_outer_area_px2=50.0, min_spike_interior_angle_deg=0.0),
            polygon_id=1,
        )
        self.assertFalse(changed)
        self.assertEqual(len(processed), 1)
        self.assertEqual(processed[0].points, tiny.points)

    def test_via_and_box_rings_are_not_flagged_as_invalid_descriptions(self) -> None:
        via = _rect(0.0, 0.0, 8.0, 8.0, 1)
        via.category = "via"
        via.shape_hint = "box"
        via.points = [(0.0, 0.0), (8.0, 8.0), (8.0, 0.0), (0.0, 8.0)]
        self.assertFalse(polygon_description_is_invalid(via))

    def test_repair_invalid_polygon_descriptions_splits_keyhole(self) -> None:
        keyhole = PolygonData(
            id=1,
            points=[
                (0.0, 0.0),
                (80.0, 0.0),
                (80.0, 80.0),
                (0.0, 80.0),
                (0.0, 40.0),
                (40.0, 40.0),
                (40.0, 50.0),
                (20.0, 50.0),
                (20.0, 30.0),
                (40.0, 30.0),
                (40.0, 40.0),
                (0.0, 40.0),
            ],
        )
        self.assertFalse(polygon_description_is_invalid(keyhole))
        self.assertEqual(keyhole.description_invalid_reason(), "repeated_vertex")
        repaired = repair_invalid_polygon_descriptions([keyhole])
        self.assertTrue(repaired)
        self.assertFalse(any(polygon_description_is_invalid(polygon) for polygon in repaired))
        self.assertTrue(any(polygon.is_hole for polygon in repaired))
        self.assertTrue(any(not polygon.is_hole for polygon in repaired))

    def test_repair_invalid_polygon_descriptions_splits_bowtie(self) -> None:
        bowtie = PolygonData(
            id=1,
            points=[(0.0, 0.0), (40.0, 40.0), (40.0, 0.0), (0.0, 40.0)],
        )
        self.assertTrue(polygon_description_is_invalid(bowtie))
        repaired = repair_invalid_polygon_descriptions([bowtie])
        self.assertGreaterEqual(len(repaired), 2)
        self.assertFalse(any(polygon_description_is_invalid(polygon) for polygon in repaired))
        self.assertTrue(all(not polygon.is_hole for polygon in repaired))

    def test_repair_invalid_polygon_descriptions_is_noop_for_simple_rings(self) -> None:
        square = _rect(0.0, 0.0, 40.0, 40.0, 1)
        repaired = repair_invalid_polygon_descriptions([square])
        self.assertEqual(len(repaired), 1)
        self.assertEqual(repaired[0].points, square.points)
        self.assertFalse(polygon_description_is_invalid(repaired[0]))

    def test_summarize_invalid_polygon_description_reasons(self) -> None:
        keyhole = PolygonData(
            id=1,
            points=[
                (0.0, 0.0),
                (80.0, 0.0),
                (80.0, 80.0),
                (0.0, 80.0),
                (0.0, 40.0),
                (40.0, 40.0),
                (40.0, 50.0),
                (20.0, 50.0),
                (20.0, 30.0),
                (40.0, 30.0),
                (40.0, 40.0),
                (0.0, 40.0),
            ],
        )
        bowtie = PolygonData(
            id=2,
            points=[(0.0, 0.0), (40.0, 40.0), (40.0, 0.0), (0.0, 40.0)],
        )
        square = _rect(100.0, 100.0, 120.0, 120.0, 3)
        self.assertEqual(
            summarize_invalid_polygon_description_reasons([keyhole, bowtie, square, keyhole]),
            [("repeated_vertex", 2), ("self_intersecting", 1)],
        )

    def test_repair_invalid_polygon_descriptions_keeps_other_polygons(self) -> None:
        keepers = [
            _rect(0.0, 0.0, 40.0, 40.0, 1),
            _rect(100.0, 0.0, 140.0, 40.0, 2),
            _rect(200.0, 0.0, 240.0, 40.0, 3),
        ]
        keyhole = PolygonData(
            id=10,
            points=[
                (0.0, 100.0),
                (80.0, 100.0),
                (80.0, 180.0),
                (0.0, 180.0),
                (0.0, 140.0),
                (40.0, 140.0),
                (40.0, 150.0),
                (20.0, 150.0),
                (20.0, 130.0),
                (40.0, 130.0),
                (40.0, 140.0),
                (0.0, 140.0),
            ],
        )
        self.assertFalse(polygon_description_is_invalid(keyhole))
        self.assertEqual(keyhole.description_invalid_reason(), "repeated_vertex")

        repaired = repair_invalid_polygon_descriptions([*keepers, keyhole])

        self.assertEqual(len({polygon.id for polygon in repaired}), len(repaired))
        for keeper in keepers:
            match = next(polygon for polygon in repaired if polygon.points == keeper.points)
            self.assertEqual(match.id, keeper.id)
            self.assertFalse(match.is_hole)
        self.assertFalse(any(polygon_description_is_invalid(polygon) for polygon in repaired))
        self.assertTrue(any(polygon.is_hole for polygon in repaired))
        self.assertGreaterEqual(len(repaired), len(keepers) + 2)

    def test_repair_keyhole_with_spikes_keeps_authored_geometry(self) -> None:
        """0560-style: repair must split keyholes, not reshape via shapely dissolve."""

        # Incomplete multi-keyhole bar with A->B->A spikes on two hole rims.
        points = [
            (1958.0, 1805.0),
            (1937.0, 1805.0),
            (1931.0, 1799.0),
            (1825.0, 1800.0),
            (1820.0, 1805.0),
            (1823.0, 1811.0),
            (1828.0, 1813.0),
            (1862.0, 1814.0),
            (1915.0, 1812.0),
            (1928.0, 1813.0),
            (1933.0, 1812.0),
            (1937.0, 1807.0),
            (1937.0, 1805.0),
            (1958.0, 1805.0),
            (1799.0, 1805.0),
            (1795.0, 1801.0),
            (1790.0, 1799.0),
            (1689.0, 1800.0),
            (1683.0, 1806.0),
            (1684.0, 1809.0),
            (1689.0, 1813.0),
            (1794.0, 1813.0),
            (1799.0, 1809.0),
            (1799.0, 1805.0),
            (1958.0, 1805.0),
            (1958.0, 1804.0),
            (1525.0, 1804.0),
            (1519.0, 1800.0),
            (1416.0, 1800.0),
            (1413.0, 1801.0),
            (1408.0, 1807.0),
            (1414.0, 1813.0),
            (1511.0, 1813.0),
            (1512.0, 1827.0),
            (1511.0, 1813.0),
            (1521.0, 1813.0),
            (1525.0, 1809.0),
            (1525.0, 1804.0),
            (1958.0, 1804.0),
            (1963.0, 1800.0),
            (1998.0, 1800.0),
            (1998.0, 1775.0),
            (1990.0, 1774.0),
            (1.0, 1776.0),
            (1.0, 1794.0),
            (1011.0, 1793.0),
            (1016.0, 1794.0),
            (1026.0, 1800.0),
            (1052.0, 1801.0),
            (1066.0, 1800.0),
            (1077.0, 1794.0),
            (1201.0, 1793.0),
            (1216.0, 1800.0),
            (1232.0, 1799.0),
            (1385.0, 1801.0),
            (1389.0, 1806.0),
            (1382.0, 1813.0),
            (1.0, 1813.0),
            (1.0, 1828.0),
            (1790.0, 1828.0),
            (1937.0, 1826.0),
            (1998.0, 1827.0),
            (1998.0, 1813.0),
            (1963.0, 1813.0),
            (1959.0, 1809.0),
            (1662.0, 1809.0),
            (1660.0, 1802.0),
            (1655.0, 1799.0),
            (1552.0, 1800.0),
            (1546.0, 1805.0),
            (1546.0, 1807.0),
            (1552.0, 1813.0),
            (1589.0, 1813.0),
            (1589.0, 1827.0),
            (1589.0, 1813.0),
            (1657.0, 1813.0),
            (1662.0, 1809.0),
            (1959.0, 1809.0),
            (1957.0, 1807.0),
            (1958.0, 1805.0),
        ]
        area, perimeter, bbox = compute_polygon_metrics(points)
        keyhole = PolygonData(id=1, points=points, area=area, perimeter=perimeter, bbox=bbox)
        self.assertEqual(keyhole.description_invalid_reason(), "repeated_vertex")

        repaired = repair_invalid_polygon_descriptions([keyhole])
        outers = [polygon for polygon in repaired if not polygon.is_hole]
        holes = [polygon for polygon in repaired if polygon.is_hole]

        self.assertFalse(any(polygon_description_is_invalid(polygon) for polygon in repaired))
        self.assertEqual(len(outers), 1)
        self.assertEqual(len(holes), 4)
        # Shapely dissolve used to spawn a tiny floating outer (~576) and merge holes.
        self.assertFalse(any(polygon.area < 1000.0 for polygon in outers))
        self.assertAlmostEqual(outers[0].area, 79595.0, delta=1.0)
        self.assertEqual(outers[0].bbox[1], bbox[1])
        self.assertEqual(outers[0].bbox[3], bbox[3])

    def test_polygons_needing_repair_skips_keyhole_repeated_vertex(self) -> None:
        keyhole = PolygonData(
            id=1,
            points=[
                (0.0, 0.0),
                (80.0, 0.0),
                (80.0, 80.0),
                (0.0, 80.0),
                (0.0, 40.0),
                (40.0, 40.0),
                (40.0, 50.0),
                (20.0, 50.0),
                (20.0, 30.0),
                (40.0, 30.0),
                (40.0, 40.0),
                (0.0, 40.0),
            ],
        )
        settings = VectorGeometrySettings(min_outer_area_px2=0.0, min_hole_area_to_remove_px2=0.0)
        self.assertEqual(keyhole.description_invalid_reason(), "repeated_vertex")
        self.assertFalse(polygons_needing_repair([keyhole], settings))

    def test_polygons_needing_repair_flags_overlapping_small_object_and_hole(self) -> None:
        big_a = _rect(0.0, 0.0, 80.0, 80.0, 1)
        big_b = _rect(40.0, 40.0, 120.0, 120.0, 2)
        tiny = _rect(200.0, 200.0, 205.0, 205.0, 3)
        outer = _rect(300.0, 300.0, 400.0, 400.0, 4)
        hole = PolygonData(
            id=5,
            points=[(320.0, 320.0), (330.0, 320.0), (330.0, 330.0), (320.0, 330.0)],
            is_hole=True,
            parent_id=4,
            area=100.0,
            perimeter=40.0,
            bbox=(320, 320, 10, 10),
        )
        settings = VectorGeometrySettings(min_outer_area_px2=50.0, min_hole_area_to_remove_px2=150.0)
        reasons = polygons_needing_repair([big_a, big_b, tiny, outer, hole], settings)

        self.assertEqual(reasons[1], ["overlapping"])
        self.assertEqual(reasons[2], ["overlapping"])
        self.assertEqual(reasons[3], ["small_object"])
        self.assertEqual(reasons[5], ["small_hole"])
        self.assertNotIn(4, reasons)

    def test_summarize_with_settings_includes_new_reason_codes(self) -> None:
        a = _rect(0.0, 0.0, 60.0, 60.0, 1)
        b = _rect(30.0, 30.0, 90.0, 90.0, 2)
        tiny = _rect(200.0, 200.0, 204.0, 204.0, 3)
        settings = VectorGeometrySettings(min_outer_area_px2=50.0, min_hole_area_to_remove_px2=0.0)
        summary = summarize_invalid_polygon_description_reasons([a, b, tiny], settings)
        self.assertEqual(summary, [("overlapping", 2), ("small_object", 1)])

    def test_repair_with_settings_merges_overlap_and_drops_small_geometry(self) -> None:
        a = _rect(0.0, 0.0, 70.0, 70.0, 1)
        b = _rect(35.0, 35.0, 95.0, 95.0, 2)
        tiny = _rect(200.0, 200.0, 203.0, 203.0, 3)
        outer = _rect(300.0, 300.0, 420.0, 420.0, 4)
        hole = PolygonData(
            id=5,
            points=[(320.0, 320.0), (328.0, 320.0), (328.0, 328.0), (320.0, 328.0)],
            is_hole=True,
            parent_id=4,
            area=64.0,
            perimeter=32.0,
            bbox=(320, 320, 8, 8),
        )
        settings = VectorGeometrySettings(min_outer_area_px2=50.0, min_hole_area_to_remove_px2=100.0)
        repaired = repair_invalid_polygon_descriptions([a, b, tiny, outer, hole], settings)

        self.assertFalse(polygons_needing_repair(repaired, settings))
        roots = [polygon for polygon in repaired if not polygon.is_hole and polygon.parent_id is None]
        self.assertEqual(len(roots), 2)
        self.assertFalse(any(polygon.is_hole for polygon in repaired))
        self.assertTrue(all(abs(float(polygon.area)) >= 49.5 for polygon in roots))

    def test_repair_without_settings_keeps_topology_only_behavior(self) -> None:
        tiny = _rect(0.0, 0.0, 5.0, 5.0, 1)
        repaired = repair_invalid_polygon_descriptions([tiny])
        self.assertEqual(len(repaired), 1)
        self.assertEqual(repaired[0].points, tiny.points)


if __name__ == "__main__":
    unittest.main()