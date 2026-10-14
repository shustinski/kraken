"""test_vector_geometry_postprocess: test manual family edit."""
from __future__ import annotations
from contour.application.vector_geometry_postprocess import (
    VectorGeometrySettings,
    apply_edge_translation_to_clone,
    apply_polygon_points_to_clone,
    apply_vertex_delete_to_clone,
    apply_vertex_position_to_clone,
    dissolve_self_intersecting_polygons,
    postprocess_after_editor_mutation,
    postprocess_polygons_for_frame_navigation,
    union_after_removing_polygon_ids,
)
from contour.domain import PolygonData, compute_polygon_metrics
from contour.domain.polygon_ring import is_valid_closed_polygon_ring

from test_vector_geometry_postprocess_parts.rect import _rect


class ManualFamilyEditMixin:
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

    def test_dissolve_scoped_to_changed_polygon_ids_skips_unchanged_roots(self) -> None:
        invalid = PolygonData(
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
        valid = _rect(200.0, 200.0, 260.0, 260.0, 2)
        after_delete = apply_vertex_delete_to_clone([invalid, valid], 1, 0)
        healed = dissolve_self_intersecting_polygons(
            after_delete,
            changed_polygon_ids={1},
        )
        self.assertTrue(all(is_valid_closed_polygon_ring(polygon.points) for polygon in healed))
        valid_after = next(polygon for polygon in healed if polygon.id == 2)
        self.assertEqual(valid_after.points, valid.points)

    def test_merge_after_removal_only_checks_promoted_roots(self) -> None:
        outer = _rect(0.0, 0.0, 80.0, 80.0, 1)
        island = _rect(30.0, 30.0, 50.0, 50.0, 3)
        island.parent_id = 2
        far = _rect(300.0, 300.0, 360.0, 360.0, 4)
        remaining = union_after_removing_polygon_ids([outer, island, far], {2})
        filled = [polygon for polygon in remaining if not polygon.is_hole]
        self.assertEqual(len(filled), 2)

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

    def test_edge_move_valid_polygon_succeeds(self) -> None:
        poly = _rect(0.0, 0.0, 100.0, 100.0, 1)
        moved = apply_edge_translation_to_clone([poly], 1, 0, (0.0, 10.0))
        self.assertEqual(moved[0].points[0], (0.0, 10.0))
        self.assertEqual(moved[0].points[1], (100.0, 10.0))

    def test_edge_move_invalid_polygon_is_rejected(self) -> None:
        poly = _rect(0.0, 0.0, 100.0, 100.0, 1)
        # Collapsing the translated edge onto the opposite edge is degenerate.
        moved = apply_edge_translation_to_clone([poly], 1, 0, (0.0, 100.0))
        self.assertEqual(moved[0].points, poly.points)

    def test_manual_family_edit_discards_authored_cif_paint_ring(self) -> None:
        outer = _rect(0.0, 0.0, 100.0, 100.0, 1)
        outer.cif_paint_ring = [
            (0.0, 0.0),
            (100.0, 0.0),
            (100.0, 100.0),
            (0.0, 100.0),
            (0.0, 40.0),
            (40.0, 40.0),
            (40.0, 60.0),
            (0.0, 60.0),
        ]
        hole = _rect(40.0, 40.0, 60.0, 60.0, 2)
        hole.is_hole = True
        hole.parent_id = outer.id

        moved = apply_polygon_points_to_clone(
            [outer, hole],
            hole.id,
            [(42.0, 40.0), (60.0, 40.0), (60.0, 60.0), (42.0, 60.0)],
        )

        self.assertEqual(next(polygon for polygon in moved if polygon.id == outer.id).cif_paint_ring, [])

        moved_outer = apply_vertex_position_to_clone([outer, hole], outer.id, 1, (101.0, 0.0))
        self.assertEqual(next(polygon for polygon in moved_outer if polygon.id == outer.id).cif_paint_ring, [])

    def test_cif_paint_stamp_clips_ring_outside_target_mask(self) -> None:
        import numpy as np

        from contour.serializers import _stamp_cif_paint_ring_on_mask

        mask = np.zeros((10, 10), dtype=np.uint8)
        _stamp_cif_paint_ring_on_mask(
            mask,
            [(20.0, 0.0), (30.0, 0.0), (30.0, 9.0), (20.0, 9.0)],
            (0, 0),
        )
        self.assertFalse(np.any(mask))

        _stamp_cif_paint_ring_on_mask(
            mask,
            [(-5.0, 0.0), (5.0, 0.0), (5.0, 9.0), (-5.0, 9.0)],
            (0, 0),
        )
        self.assertTrue(np.all(mask[:9, :5] == 255))
        self.assertTrue(np.all(mask[9, :] == 0))
        self.assertTrue(np.all(mask[:, 6:] == 0))

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
