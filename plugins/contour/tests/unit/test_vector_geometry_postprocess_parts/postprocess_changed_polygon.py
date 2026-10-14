"""test_vector_geometry_postprocess: test postprocess changed polygon."""
from __future__ import annotations
from contour.application.vector_geometry_postprocess import (
    VectorGeometrySettings,
    apply_heavy_geometry_repair_after_edit,
    apply_topology_repair_after_edit,
    apply_vertex_delete_to_clone,
    polygons_needing_repair,
    postprocess_changed_polygon_edit,
    should_defer_geometry_repair,
    should_defer_overlap_merge_after_edit,
)
from contour.domain import PolygonData, compute_polygon_metrics
from contour.domain.polygon_ring import is_valid_closed_polygon_ring

from test_vector_geometry_postprocess_parts.rect import _rect


class PostprocessChangedPolygonMixin:
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

    def test_should_defer_overlap_merge_only_when_removals_are_bulk(self) -> None:
        settings = VectorGeometrySettings(deferred_geometry_repair_affected_threshold=48)
        changed = set(range(30))
        removed = set(range(30, 60))
        self.assertTrue(
            should_defer_overlap_merge_after_edit(
                settings,
                changed_polygon_ids=changed,
                removed_ids=removed,
            )
        )
        self.assertFalse(
            should_defer_overlap_merge_after_edit(
                settings,
                changed_polygon_ids=changed,
                removed_ids=set(),
            )
        )
        self.assertFalse(
            should_defer_overlap_merge_after_edit(
                settings,
                changed_polygon_ids={1},
                removed_ids=set(),
            )
        )

    def test_should_defer_geometry_repair_for_bulk_layer_edits(self) -> None:
        settings = VectorGeometrySettings(deferred_geometry_repair_affected_threshold=48)
        changed = set(range(30))
        removed = set(range(30, 60))
        self.assertTrue(
            should_defer_geometry_repair(
                settings,
                changed_polygon_ids=changed,
                removed_ids=removed,
            )
        )
        self.assertFalse(
            should_defer_geometry_repair(
                settings,
                changed_polygon_ids={1},
                removed_ids=set(),
            )
        )

    def test_apply_heavy_geometry_repair_after_edit_dissolves_invalid(self) -> None:
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
        repaired = apply_heavy_geometry_repair_after_edit(
            after_delete,
            changed_polygon_ids={1},
            removed_ids=set(),
        )
        self.assertTrue(all(is_valid_closed_polygon_ring(polygon.points) for polygon in repaired))

    def test_apply_topology_repair_after_edit_dissolves_invalid(self) -> None:
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
        repaired = apply_topology_repair_after_edit(after_delete, changed_polygon_ids={1})
        self.assertTrue(all(is_valid_closed_polygon_ring(polygon.points) for polygon in repaired))

    def test_postprocess_changed_polygon_edit_dissolves_invalid_contours(self) -> None:
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
        trial = apply_vertex_delete_to_clone([slot], 1, 0)
        processed, accepted, _changed = postprocess_changed_polygon_edit(
            trial,
            VectorGeometrySettings(),
            polygon_id=1,
        )
        self.assertTrue(accepted)
        self.assertTrue(all(is_valid_closed_polygon_ring(polygon.points) for polygon in processed))
