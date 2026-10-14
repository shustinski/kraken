from __future__ import annotations
import csv
import json
import os
import shutil
import threading
from bisect import bisect_left
from collections.abc import Iterable
from pathlib import Path
from time import perf_counter
from xml.sax.saxutils import escape
import cv2
import numpy as np
from .application.processing import DisplaySettings, SaveOptions
from .domain import PolygonData, compute_polygon_metrics, integer_points
from .domain.polygon_ring import collapse_redundant_polyline_vertices
from .i18n import tr
from .infrastructure.cif_primitives import CifBox, CifPrimitive
from .utils import draw_polygon_overlay, ensure_directory, imwrite_unicode_safe
from .value_conversion import to_int

from .serializers_parts.module_state import (
    _CIF_PARSE_CACHE,
    _CIF_PARSE_CACHE_LOCK,
    _CIF_PARSE_CACHE_MAX_ENTRIES,
    CIF_CUTOUT_DISPLAY_MARKER,
    _BOUNDARY_SNAP_NEIGHBOR_WINDOW,
    _HoleLinkDetour,
)
from .serializers_parts.cv_object_polygon import (
    _note_cif_phase_timing,
    _note_cif_slit_target,
    _path_cache_identity,
    clear_cif_parse_cache,
    invalidate_cif_parse_cache,
    _cif_parse_cache_key,
    _cache_cif_parse_result,
    save_polygons_json,
    _polygon_bbox_values,
    _cv_coord,
    _cv_ring,
    _pixel_box_center_and_size,
    _pixel_box_bounds,
    _cv_object_from_polygon,
    _cv_objects_from_polygons,
    _cv_json_array,
    _cv_json_object,
)
from .serializers_parts.cv_point_points import (
    _dumps_cv_payload,
    save_polygons_cv,
    _as_float_pair,
    _cv_box_points,
    _cv_point_points,
    _cv_polygon_from_points,
    _cv_points_from_ring,
    _cv_object_id,
)
from .serializers_parts.polygons_cv_object import _polygons_from_cv_object
from .serializers_parts.cif_box_polygon import (
    load_polygons_cv,
    load_polygons_vector,
    _cif_cutout_display_requested,
    save_polygons_vector,
    _parse_cif_int,
    _clone_polygon_with_id,
    _extract_parenthesized_tokens,
    _read_cif_text,
    _cif_box_to_polygon,
    _cif_polygon_points_to_polygon,
)
from .serializers_parts.cif_box_primitive_polygon import (
    _cif_box_primitive_to_polygon,
    _cif_primitives_to_polygons,
)
from .serializers_parts.load_polygons_cif import (
    _load_polygons_cif_via_klayout,
    _load_polygons_cif_via_opencif,
    load_polygons_cif,
)
from .serializers_parts.boundary_vertices_by_distance import (
    _polygon_to_cif_line,
    _to_local_int_points,
    _dedupe_closed_points,
    _ring_has_retracing_entry_edge,
    _is_simple_extracted_hole_ring,
    _has_duplicate_points,
    _rightmost_point_index,
    _point_distance_squared,
    _boundary_vertices_by_distance,
    _snap_hole_to_boundary_vertex,
)
from .serializers_parts.select_vertex_slit_hole import (
    _select_vertex_slit_for_hole,
    _trace_linked_hole_cycle,
    _rotate_open_ring,
    _signed_ring_area,
    _ring_with_orientation,
    _edge_parameter,
)
from .serializers_parts.encode_parent_holes_link import _encode_parent_with_holes_link_path
from .serializers_parts.contour_points_polygon import _contour_points_to_polygon
from .serializers_parts.split_linked_polygon_rings import _split_linked_polygon_rings
from .serializers_parts.cif_paint_mask_ring import (
    _remove_out_and_back_spikes,
    _collapse_recovered_outline,
    _cif_paint_mask_from_ring,
    _stamp_cif_paint_ring_on_mask,
    _polygon_uses_authored_cif_paint_ring,
    _ensure_cif_paint_rings,
    _render_cif_klayout_layer_mask,
    _legacy_cut_hole_recovery_enabled,
    _nested_outer_punch_on_load_enabled,
    _ring_coords_from_shapely,
)
from .serializers_parts.normalize_cif_polygon_families import _normalize_cif_polygon_families
from .serializers_parts.recover_cut_hole_topology import _recover_cut_hole_topology
from .serializers_parts.nested_outer_punch_footprint import (
    _family_geometry,
    _nested_outer_punch_footprint,
)
from .serializers_parts.punch_nested_outer_covers import _punch_nested_outer_covers
from .serializers_parts.polygons_shapely_geometry import _polygons_from_shapely_geometry
from .serializers_parts.connected_cif_families import _connected_cif_families
from .serializers_parts.save_polygons_cif import (
    save_polygons_cif,
    save_polygons_csv,
    save_polygons_txt,
)
from .serializers_parts.save_svg_preview import (
    save_svg_preview,
    save_overlay_preview,
    _is_ellipse_display_polygon,
    _copy_or_write_dataset_image,
    export_dataset_frame,
    save_result_bundle,
)

# Imported so existing callers and tests can patch or import these names
# from this module after the implementation moved into focused units.
_FACADE_EXPORTS = (
    _CIF_PARSE_CACHE,
    _CIF_PARSE_CACHE_LOCK,
    _CIF_PARSE_CACHE_MAX_ENTRIES,
    CIF_CUTOUT_DISPLAY_MARKER,
    _BOUNDARY_SNAP_NEIGHBOR_WINDOW,
    _HoleLinkDetour,
    _note_cif_phase_timing,
    _note_cif_slit_target,
    _path_cache_identity,
    clear_cif_parse_cache,
    invalidate_cif_parse_cache,
    _cif_parse_cache_key,
    _cache_cif_parse_result,
    save_polygons_json,
    _polygon_bbox_values,
    _cv_coord,
    _cv_ring,
    _pixel_box_center_and_size,
    _pixel_box_bounds,
    _cv_object_from_polygon,
    _cv_objects_from_polygons,
    _cv_json_array,
    _cv_json_object,
    _dumps_cv_payload,
    save_polygons_cv,
    _as_float_pair,
    _cv_box_points,
    _cv_point_points,
    _cv_polygon_from_points,
    _cv_points_from_ring,
    _cv_object_id,
    _polygons_from_cv_object,
    load_polygons_cv,
    load_polygons_vector,
    _cif_cutout_display_requested,
    save_polygons_vector,
    _parse_cif_int,
    _clone_polygon_with_id,
    _extract_parenthesized_tokens,
    _read_cif_text,
    _cif_box_to_polygon,
    _cif_polygon_points_to_polygon,
    _cif_box_primitive_to_polygon,
    _cif_primitives_to_polygons,
    _load_polygons_cif_via_klayout,
    _load_polygons_cif_via_opencif,
    load_polygons_cif,
    _polygon_to_cif_line,
    _to_local_int_points,
    _dedupe_closed_points,
    _ring_has_retracing_entry_edge,
    _is_simple_extracted_hole_ring,
    _has_duplicate_points,
    _rightmost_point_index,
    _point_distance_squared,
    _boundary_vertices_by_distance,
    _snap_hole_to_boundary_vertex,
    _select_vertex_slit_for_hole,
    _trace_linked_hole_cycle,
    _rotate_open_ring,
    _signed_ring_area,
    _ring_with_orientation,
    _edge_parameter,
    _encode_parent_with_holes_link_path,
    _contour_points_to_polygon,
    _split_linked_polygon_rings,
    _remove_out_and_back_spikes,
    _collapse_recovered_outline,
    _cif_paint_mask_from_ring,
    _stamp_cif_paint_ring_on_mask,
    _polygon_uses_authored_cif_paint_ring,
    _ensure_cif_paint_rings,
    _render_cif_klayout_layer_mask,
    _legacy_cut_hole_recovery_enabled,
    _nested_outer_punch_on_load_enabled,
    _ring_coords_from_shapely,
    _normalize_cif_polygon_families,
    _recover_cut_hole_topology,
    _family_geometry,
    _nested_outer_punch_footprint,
    _punch_nested_outer_covers,
    _polygons_from_shapely_geometry,
    _connected_cif_families,
    save_polygons_cif,
    save_polygons_csv,
    save_polygons_txt,
    save_svg_preview,
    save_overlay_preview,
    _is_ellipse_display_polygon,
    _copy_or_write_dataset_image,
    export_dataset_frame,
    save_result_bundle,
)
