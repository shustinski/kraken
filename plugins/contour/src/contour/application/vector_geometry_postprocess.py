"""Testable geometric post-processing for manual vector overlays (paths / editor).

Uses existing NumPy/OpenCV stack (same as graphics.geometry raster helpers).
See :class:`VectorGeometrySettings` for defaults and knobs.
"""

from __future__ import annotations
from dataclasses import dataclass
from math import atan2, hypot, pi, radians
import cv2
import numpy as np
from ..domain import PolygonData, compute_polygon_metrics, integer_point, integer_points
from ..domain.polygon_ring import (
    TOPOLOGY_CHECK_MAX_VERTICES,
    collapse_redundant_polyline_vertices,
    is_valid_closed_polygon_edge_move,
    is_valid_closed_polygon_ring,
    is_valid_closed_polygon_vertex_move,
)

from .vector_geometry_postprocess_parts.module_state import _MASK_HELPERS
from .vector_geometry_postprocess_parts.collapse_redundant_vertices_polygon import (
    _ring_ok_after_spike_removal,
    _polygon_contains_point,
    _mask_helpers,
    _bboxes_intersect,
    VectorGeometrySettings,
    _point_finite,
    _refresh_metrics,
    _ring_has_usable_geometry,
    collapse_redundant_vertices_in_polygons,
    collapse_redundant_vertices_for_polygon_ids,
    _replace_polygon_in_list,
    drop_polygons_invalid_points,
    filter_simple_valid_polygons,
    _is_small_inner_area,
    dissolve_small_holes,
    dissolve_small_holes_scoped,
    drop_orphan_holes,
    _is_via_like,
    reparent_polygons_orphaned_by_ids,
    _root_id_for_polygon,
)
from .vector_geometry_postprocess_parts.merge_candidate_roots_after import (
    _promoted_root_ids_before_reparent,
    _expand_roots_by_touching_bboxes,
    _merge_candidate_roots_after_removal,
    union_after_removing_polygon_ids,
    apply_orphan_cleanup_after_removal,
    apply_topology_repair_after_edit,
    apply_overlap_merge_after_edit,
    _repair_still_invalid_ring_families,
    apply_delete_area_geometry_repair,
    apply_heavy_geometry_repair_after_edit,
)
from .vector_geometry_postprocess_parts.drop_small_outer_polygons import (
    should_defer_overlap_merge_after_edit,
    should_defer_geometry_repair,
    drop_small_outer_polygons,
    drop_small_outer_polygons_scoped,
)
from .vector_geometry_postprocess_parts.drop_triangle_outer_artifacts import (
    drop_triangle_outer_artifacts,
    drop_triangle_outer_artifacts_scoped,
)
from .vector_geometry_postprocess_parts.apply_spike_removal_polygon import (
    _interior_turn_angle_rad,
    remove_spikes_from_polygon_ring,
    apply_spike_removal_all,
    apply_spike_removal_for_polygon_ids,
    drop_polygons_invalid_points_scoped,
    filter_simple_valid_polygons_scoped,
    _polygon_ids_in_root_families,
    _affected_polygon_ids_for_layer_patch,
    clip_polygons_to_frame_raster,
    _centroid,
)
from .vector_geometry_postprocess_parts.root_id_map import (
    _stamp_visual_metadata,
    _relabel_polygon_ids,
    _next_polygon_id_after,
    _UnionFind,
    _children_by_parent,
    _pad_bbox,
    _root_id_map,
    _root_family_bboxes,
    _collect_family,
    _family_bbox,
)
from .vector_geometry_postprocess_parts.merge_candidate_roots_near import (
    _families_mask_overlap,
    _families_mask_overlap_cached,
    _merge_candidate_roots_near_polygons,
)
from .vector_geometry_postprocess_parts.merge_overlapping_root_families import (
    merge_overlapping_root_families_near_polygons,
    merge_overlapping_root_families,
)
from .vector_geometry_postprocess_parts.rebuild_family_filled_mask import (
    _rebuild_family_from_filled_mask,
    _polygon_ring_needs_dissolve,
    _polygon_ring_has_keyhole_bridge,
    _polygon_ring_needs_explicit_topology_repair,
)
from .vector_geometry_postprocess_parts.dissolve_self_intersecting_polygons import dissolve_self_intersecting_polygons
from .vector_geometry_postprocess_parts.repair_family_by_keyhole import _repair_family_by_keyhole_split
from .vector_geometry_postprocess_parts.overlapping_root_family_ids import (
    _dissolve_family_to_simple_rings,
    _make_valid_union_of_family_rings,
    polygon_description_is_invalid,
    polygon_description_invalid_reason,
    overlapping_root_family_ids,
    overlapping_root_family_ids_near_roots,
    overlap_check_roots_for_layer_patch,
    apply_overlap_repair_patch,
)
from .vector_geometry_postprocess_parts.patch_polygons_needing_repair import patch_polygons_needing_repair
from .vector_geometry_postprocess_parts.polygons_needing_repair import polygons_needing_repair
from .vector_geometry_postprocess_parts.repair_invalid_polygon_descriptions import (
    summarize_invalid_polygon_description_reasons,
    repair_invalid_polygon_descriptions,
    _polygon_topo_points_key,
    _single_polygon_topo_signature,
    _polygons_topo_signature,
)
from .vector_geometry_postprocess_parts.postprocess_after_layer_patch import (
    postprocess_after_editor_mutation,
    postprocess_after_layer_patch,
)
from .vector_geometry_postprocess_parts.postprocess_scoped_single_polygon import _postprocess_scoped_single_polygon
from .vector_geometry_postprocess_parts.postprocess_changed_polygon_edit import (
    postprocess_changed_polygon_edit,
    postprocess_changed_polygon_only,
)
from .vector_geometry_postprocess_parts.postprocess_after_vertex_move import postprocess_after_vertex_move
from .vector_geometry_postprocess_parts.postprocess_vertex_move_edit import postprocess_vertex_move_edit
from .vector_geometry_postprocess_parts.postprocess_polygons_frame_navigation import (
    postprocess_polygons_for_frame_navigation,
    _invalidate_authored_cif_paint_for_edit,
    apply_edge_translation_to_clone,
)
from .vector_geometry_postprocess_parts.apply_vertex_position_scoped import (
    apply_vertex_position_to_clone,
    apply_vertex_position_scoped,
    apply_vertex_delete_to_clone,
)
from .vector_geometry_postprocess_parts.resolve_focus_id_after import (
    apply_polygon_points_to_clone,
    resolve_focus_id_after_geometry_pass,
)

# Imported so existing callers and tests can patch or import these names
# from this module after the implementation moved into focused units.
_FACADE_EXPORTS = (
    _MASK_HELPERS,
    _ring_ok_after_spike_removal,
    _polygon_contains_point,
    _mask_helpers,
    _bboxes_intersect,
    VectorGeometrySettings,
    _point_finite,
    _refresh_metrics,
    _ring_has_usable_geometry,
    collapse_redundant_vertices_in_polygons,
    collapse_redundant_vertices_for_polygon_ids,
    _replace_polygon_in_list,
    drop_polygons_invalid_points,
    filter_simple_valid_polygons,
    _is_small_inner_area,
    dissolve_small_holes,
    dissolve_small_holes_scoped,
    drop_orphan_holes,
    _is_via_like,
    reparent_polygons_orphaned_by_ids,
    _root_id_for_polygon,
    _promoted_root_ids_before_reparent,
    _expand_roots_by_touching_bboxes,
    _merge_candidate_roots_after_removal,
    union_after_removing_polygon_ids,
    apply_orphan_cleanup_after_removal,
    apply_topology_repair_after_edit,
    apply_overlap_merge_after_edit,
    _repair_still_invalid_ring_families,
    apply_delete_area_geometry_repair,
    apply_heavy_geometry_repair_after_edit,
    should_defer_overlap_merge_after_edit,
    should_defer_geometry_repair,
    drop_small_outer_polygons,
    drop_small_outer_polygons_scoped,
    drop_triangle_outer_artifacts,
    drop_triangle_outer_artifacts_scoped,
    _interior_turn_angle_rad,
    remove_spikes_from_polygon_ring,
    apply_spike_removal_all,
    apply_spike_removal_for_polygon_ids,
    drop_polygons_invalid_points_scoped,
    filter_simple_valid_polygons_scoped,
    _polygon_ids_in_root_families,
    _affected_polygon_ids_for_layer_patch,
    clip_polygons_to_frame_raster,
    _centroid,
    _stamp_visual_metadata,
    _relabel_polygon_ids,
    _next_polygon_id_after,
    _UnionFind,
    _children_by_parent,
    _pad_bbox,
    _root_id_map,
    _root_family_bboxes,
    _collect_family,
    _family_bbox,
    _families_mask_overlap,
    _families_mask_overlap_cached,
    _merge_candidate_roots_near_polygons,
    merge_overlapping_root_families_near_polygons,
    merge_overlapping_root_families,
    _rebuild_family_from_filled_mask,
    _polygon_ring_needs_dissolve,
    _polygon_ring_has_keyhole_bridge,
    _polygon_ring_needs_explicit_topology_repair,
    dissolve_self_intersecting_polygons,
    _repair_family_by_keyhole_split,
    _dissolve_family_to_simple_rings,
    _make_valid_union_of_family_rings,
    polygon_description_is_invalid,
    polygon_description_invalid_reason,
    overlapping_root_family_ids,
    overlapping_root_family_ids_near_roots,
    overlap_check_roots_for_layer_patch,
    apply_overlap_repair_patch,
    patch_polygons_needing_repair,
    polygons_needing_repair,
    summarize_invalid_polygon_description_reasons,
    repair_invalid_polygon_descriptions,
    _polygon_topo_points_key,
    _single_polygon_topo_signature,
    _polygons_topo_signature,
    postprocess_after_editor_mutation,
    postprocess_after_layer_patch,
    _postprocess_scoped_single_polygon,
    postprocess_changed_polygon_edit,
    postprocess_changed_polygon_only,
    postprocess_after_vertex_move,
    postprocess_vertex_move_edit,
    postprocess_polygons_for_frame_navigation,
    _invalidate_authored_cif_paint_for_edit,
    apply_edge_translation_to_clone,
    apply_vertex_position_to_clone,
    apply_vertex_position_scoped,
    apply_vertex_delete_to_clone,
    apply_polygon_points_to_clone,
    resolve_focus_id_after_geometry_pass,
)
