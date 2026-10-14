from __future__ import annotations
import unittest
from contour.application.vector_geometry_postprocess import (
    VectorGeometrySettings,
    apply_delete_area_geometry_repair,
    apply_edge_translation_to_clone,
    apply_heavy_geometry_repair_after_edit,
    apply_overlap_repair_patch,
    apply_polygon_points_to_clone,
    apply_topology_repair_after_edit,
    apply_vertex_delete_to_clone,
    apply_vertex_position_to_clone,
    clip_polygons_to_frame_raster,
    collapse_redundant_vertices_for_polygon_ids,
    dissolve_self_intersecting_polygons,
    dissolve_small_holes,
    drop_triangle_outer_artifacts,
    merge_overlapping_root_families,
    merge_overlapping_root_families_near_polygons,
    overlap_check_roots_for_layer_patch,
    polygon_description_is_invalid,
    polygons_needing_repair,
    postprocess_after_editor_mutation,
    postprocess_after_layer_patch,
    postprocess_after_vertex_move,
    postprocess_changed_polygon_edit,
    postprocess_changed_polygon_only,
    postprocess_polygons_for_frame_navigation,
    postprocess_vertex_move_edit,
    remove_spikes_from_polygon_ring,
    repair_invalid_polygon_descriptions,
    should_defer_geometry_repair,
    should_defer_overlap_merge_after_edit,
    summarize_invalid_polygon_description_reasons,
    union_after_removing_polygon_ids,
)
from contour.domain import PolygonData, compute_polygon_metrics
from contour.domain.polygon_ring import is_valid_closed_polygon_ring

from test_vector_geometry_postprocess_parts.dissolves_small_hole import DissolvesSmallHoleMixin
from test_vector_geometry_postprocess_parts.manual_family_edit import ManualFamilyEditMixin
from test_vector_geometry_postprocess_parts.postprocess_after_layer import PostprocessAfterLayerMixin
from test_vector_geometry_postprocess_parts.summarize_invalid_polygon import SummarizeInvalidPolygonMixin
from test_vector_geometry_postprocess_parts.repair_keyhole_spikes import RepairKeyholeSpikesMixin
from test_vector_geometry_postprocess_parts.postprocess_changed_polygon import PostprocessChangedPolygonMixin
from test_vector_geometry_postprocess_parts.apply_delete_area import ApplyDeleteAreaMixin
from test_vector_geometry_postprocess_parts.rect import _rect

class VectorGeometryPostprocessTests(
    DissolvesSmallHoleMixin,
    ManualFamilyEditMixin,
    PostprocessAfterLayerMixin,
    SummarizeInvalidPolygonMixin,
    RepairKeyholeSpikesMixin,
    PostprocessChangedPolygonMixin,
    ApplyDeleteAreaMixin,
    unittest.TestCase,
):
    pass

if __name__ == "__main__":
    unittest.main()

# Imported so existing callers and tests can patch or import these names
# from this module after the implementation moved into focused units.
_FACADE_EXPORTS = (
    _rect,
)
