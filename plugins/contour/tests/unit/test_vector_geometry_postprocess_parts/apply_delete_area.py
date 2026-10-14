"""test_vector_geometry_postprocess: test apply delete area."""
from __future__ import annotations
from contour.application.vector_geometry_postprocess import (
    VectorGeometrySettings,
    apply_delete_area_geometry_repair,
    apply_overlap_repair_patch,
    apply_vertex_delete_to_clone,
    collapse_redundant_vertices_for_polygon_ids,
    merge_overlapping_root_families_near_polygons,
    overlap_check_roots_for_layer_patch,
    polygons_needing_repair,
    postprocess_vertex_move_edit,
    repair_invalid_polygon_descriptions,
    summarize_invalid_polygon_description_reasons,
)
from contour.domain import PolygonData, compute_polygon_metrics
from contour.domain.polygon_ring import is_valid_closed_polygon_ring

from test_vector_geometry_postprocess_parts.rect import _rect


class ApplyDeleteAreaMixin:
    def test_apply_delete_area_geometry_repair_dissolves_self_crossing(self) -> None:
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
        repaired = apply_delete_area_geometry_repair(
            after_delete,
            VectorGeometrySettings(min_outer_area_px2=1.0),
            changed_polygon_ids={1},
            removed_ids=set(),
        )
        self.assertTrue(all(is_valid_closed_polygon_ring(polygon.points) for polygon in repaired))

    def test_apply_delete_area_geometry_repair_merges_after_hole_removal(self) -> None:
        outer = _rect(0.0, 0.0, 80.0, 80.0, 1)
        island = _rect(30.0, 30.0, 50.0, 50.0, 3)
        island.parent_id = 2
        hole = _rect(25.0, 25.0, 55.0, 55.0, 2)
        hole.is_hole = True
        hole.parent_id = 1
        remaining = [outer, island]
        repaired = apply_delete_area_geometry_repair(
            remaining,
            VectorGeometrySettings(merge_overlapping_on_edit=True, min_outer_area_px2=1.0),
            changed_polygon_ids=set(),
            removed_ids={2},
            removed_polygons=[hole],
        )
        filled = [polygon for polygon in repaired if not polygon.is_hole]
        self.assertEqual(len(filled), 1)

    def test_postprocess_vertex_move_edit_uses_structural_sharing(self) -> None:
        left = _rect(0.0, 0.0, 40.0, 40.0, 1)
        right = _rect(300.0, 300.0, 360.0, 360.0, 2)
        layer = [left, right]
        processed, accepted, focus_id = postprocess_vertex_move_edit(
            layer,
            VectorGeometrySettings(min_outer_area_px2=1.0, min_spike_interior_angle_deg=0.0),
            polygon_id=1,
            vertex_index=1,
            new_point=(45.0, 0.0),
        )
        self.assertTrue(accepted)
        self.assertEqual(focus_id, 1)
        self.assertIs(processed[1], right)
        self.assertEqual(processed[0].points[1], (45.0, 0.0))

    def test_collapse_redundant_vertices_for_polygon_ids_only_clones_targets(self) -> None:
        left = _rect(0.0, 0.0, 40.0, 40.0, 1)
        right = _rect(300.0, 300.0, 360.0, 360.0, 2)
        layer = [left, right]
        collapsed, changed_ids = collapse_redundant_vertices_for_polygon_ids(layer, {1})
        self.assertEqual(changed_ids, set())
        self.assertIs(collapsed[1], right)

    def test_merge_near_polygon_skips_distant_roots(self) -> None:
        near_a = _rect(0.0, 0.0, 70.0, 70.0, 1)
        near_b = _rect(35.0, 35.0, 95.0, 95.0, 2)
        far = _rect(300.0, 300.0, 400.0, 400.0, 3)
        merged = merge_overlapping_root_families_near_polygons([near_a, near_b, far], 1)
        roots = [polygon for polygon in merged if not polygon.is_hole and polygon.parent_id is None]
        self.assertEqual(len(roots), 2)
        self.assertIn(3, {polygon.id for polygon in merged})

    def test_overlap_check_roots_for_layer_patch_limits_to_edit_region(self) -> None:
        near = _rect(0.0, 0.0, 80.0, 80.0, 1)
        far = _rect(300.0, 300.0, 400.0, 400.0, 2)
        changed = _rect(10.0, 10.0, 20.0, 20.0, 1)
        roots = overlap_check_roots_for_layer_patch([near, far], [], [changed])
        self.assertIn(1, roots)
        self.assertNotIn(2, roots)

    def test_patch_polygons_needing_repair_updates_only_local_overlap(self) -> None:
        big_a = _rect(0.0, 0.0, 80.0, 80.0, 1)
        big_b = _rect(40.0, 40.0, 120.0, 120.0, 2)
        untouched = _rect(300.0, 300.0, 400.0, 400.0, 3)
        settings = VectorGeometrySettings(min_outer_area_px2=0.0, min_hole_area_to_remove_px2=0.0)
        baseline = polygons_needing_repair([big_a, big_b, untouched], settings)
        self.assertEqual(baseline[1], ["overlapping"])
        self.assertEqual(baseline[2], ["overlapping"])

        moved_b = _rect(200.0, 200.0, 280.0, 280.0, 2)
        patched = apply_overlap_repair_patch(
            baseline,
            [big_a, moved_b, untouched],
            {1, 2},
        )
        self.assertNotIn(1, patched)
        self.assertNotIn(2, patched)
        self.assertNotIn(3, patched)

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
