"""test_vector_geometry_postprocess: test postprocess after layer."""
from __future__ import annotations
from contour.application.vector_geometry_postprocess import (
    VectorGeometrySettings,
    apply_vertex_position_to_clone,
    postprocess_after_layer_patch,
    postprocess_after_vertex_move,
    postprocess_changed_polygon_only,
    remove_spikes_from_polygon_ring,
)

from test_vector_geometry_postprocess_parts.rect import _rect


class PostprocessAfterLayerMixin:
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

    def test_postprocess_after_vertex_move_skips_merge_when_setting_disabled(self) -> None:
        left = _rect(0.0, 0.0, 40.0, 40.0, 1)
        right = _rect(50.0, 0.0, 90.0, 40.0, 2)
        moved = apply_vertex_position_to_clone([left, right], 1, 1, (60.0, 0.0))
        processed, _changed = postprocess_after_vertex_move(
            moved,
            VectorGeometrySettings(
                merge_overlapping_on_edit=False,
                min_outer_area_px2=1.0,
                min_spike_interior_angle_deg=0.0,
            ),
            polygon_id=1,
        )
        roots = [p for p in processed if p.parent_id is None and not p.is_hole]
        self.assertEqual(len(roots), 2)

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

    def test_postprocess_after_layer_patch_only_spike_removes_changed_family(self) -> None:
        from unittest.mock import patch

        unchanged = _rect(0.0, 0.0, 40.0, 40.0, 1)
        changed = _rect(100.0, 0.0, 140.0, 40.0, 2)
        settings = VectorGeometrySettings(
            min_hole_area_to_remove_px2=1.0,
            min_outer_area_px2=1.0,
            min_spike_interior_angle_deg=45.0,
        )
        with patch(
            "contour.application.vector_geometry_postprocess.remove_spikes_from_polygon_ring",
            wraps=remove_spikes_from_polygon_ring,
        ) as spike_removal:
            processed, _did_change = postprocess_after_layer_patch(
                [unchanged, changed],
                settings,
                changed_polygon_ids={2},
            )
        self.assertEqual({polygon.id for polygon in processed}, {1, 2})
        self.assertLessEqual(spike_removal.call_count, 1)
