from __future__ import annotations
import logging
from contextlib import contextmanager
from dataclasses import dataclass, replace
from math import ceil, floor, hypot
from time import perf_counter
import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QImage,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QTransform,
    QUndoStack,
)
from PyQt6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
)
from shapely import make_valid, unary_union
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry import box as shapely_box
from ..adapters.qt.image_conversion import cv_to_qimage
from ..application.polygon_antialiasing import antialias_polygons
from ..application.processing import DisplaySettings, normalize_via_display_mode
from ..application.vector_geometry_postprocess import (
    VectorGeometrySettings,
    _repair_still_invalid_ring_families,
    apply_delete_area_geometry_repair,
    apply_overlap_merge_after_edit,
    apply_overlap_repair_patch,
    apply_topology_repair_after_edit,
    overlap_check_roots_for_layer_patch,
    patch_polygons_needing_repair,
    polygons_needing_repair,
    postprocess_after_editor_mutation,
    postprocess_after_layer_patch,
    postprocess_changed_polygon_edit,
    repair_invalid_polygon_descriptions,
    resolve_focus_id_after_geometry_pass,
    union_after_removing_polygon_ids,
)
from ..commands import (
    AddPolygonCommand,
    AddPolygonsCommand,
    AddVertexCommand,
    MovePolygonCommand,
    ReplacePolygonSetCommand,
    ReplacePolygonsPatchCommand,
)
from ..domain import PolygonData, compute_polygon_metrics, integer_point, integer_points
from ..domain.polygon_ring import collapse_redundant_polyline_vertices, is_valid_closed_polygon_ring
from ..graphics_items import (
    EditablePolygonItem,
    VectorPolygonDisplayItem,
    VertexHandleItem,
    ZoomContactBatchItem,
    _closed_polygon_path,
    _display_path_for_polygon,
    _hole_display_hidden,
)
from ..i18n import active_language, tr
from ..infrastructure.polygon_change_profiler import PolygonChangeProfile
from ..infrastructure.profiling import polygon_change_profiling_enabled, write_profile_report
from .brush_vector import (
    QUAD_SEGS_BRUSH_DEFAULT,
    PreservedPolygonMatchCache,
    apply_boolean,
    bbox_intersects_geom_bounds,
    extract_polygonal_union,
    geometry_area_components,
    polygon_equivalent_preserved,
    polygon_footprint_geom,
    polygon_paint_footprint_geom,
    region_geometry,
    shapely_to_polygon_data_list,
    simplify_polygonal_geometry,
    tool_geometry,
)
from .geometry import (
    _bbox_from_points,
    _bboxes_intersect,
    _centered_rect,
    _contrasting_object_colors,
    _distance_to_segment,
    _measurement_label_position,
    _polygon_data_rect,
    _smallest_containing_polygon,
    is_valid_open_polyline_last_edge,
    resolve_conductor_hover_target_id,
    resolve_hover_polygon_id,
)
from .polygon_creation import (
    POLYGON_COMMIT_INVALID_RING,
    POLYGON_COMMIT_TOO_FEW_VERTICES,
    POLYGON_COMMIT_TOO_SMALL_AREA,
    polygon_commit_acceptability,
)
from .tool_mode_logic import (
    available_editor_tools,
    can_add_polygon,
    can_add_polygon_set,
    can_add_via,
    is_recognized_via,
)
from .tool_mode_logic import (
    is_via_polygon as _is_via_polygon,
)
from .tools import EditorTool

from .editor_scene_parts.module_state import (
    _LOGGER,
    _ZOOM_COLOR_QUANTIZATION_STEP,
    _ZOOM_BATCH_TILE_SIZE,
    _ZOOM_RASTER_MAX_DIMENSION,
    _CONTACT_PASTE_SPATIAL_CELL_SIZE,
    _POLYGON_SPATIAL_CELL_SIZE,
    _OUTER_POLYGON_ITEM_Z,
    _HOLE_POLYGON_ITEM_Z,
    _OUTER_PICK_Z_SPAN,
    _PICK_CYCLE_DISTANCE,
    _DEFERRED_GEOMETRY_REPAIR_DELAY_MS,
    _ZOOM_QUANTIZED_CHANNELS,
)
from .editor_scene_parts.construction import ConstructionMixin
from .editor_scene_parts.image_pixmap import ImagePixmapMixin
from .editor_scene_parts.render_pending_neighbor_frames import RenderPendingNeighborFramesMixin
from .editor_scene_parts.debug_candidates import DebugCandidatesMixin
from .editor_scene_parts.extra_layers import ExtraLayersMixin
from .editor_scene_parts.begin_zoom_vector_render import BeginZoomVectorRenderMixin
from .editor_scene_parts.polygon_change_profile_scope import PolygonChangeProfileScopeMixin
from .editor_scene_parts.push_polygon_layer_replacement import PushPolygonLayerReplacementMixin
from .editor_scene_parts.compute_polygon_layer_patch import ComputePolygonLayerPatchMixin
from .editor_scene_parts.apply_polygon_patch import ApplyPolygonPatchMixin
from .editor_scene_parts.apply_vector_postprocess_polygons import ApplyVectorPostprocessPolygonsMixin
from .editor_scene_parts.bulk_restore_polygons import BulkRestorePolygonsMixin
from .editor_scene_parts.polygons import PolygonsMixin
from .editor_scene_parts.sync_conductor_hover_highlight import SyncConductorHoverHighlightMixin
from .editor_scene_parts.pick_move_target import PickMoveTargetMixin
from .editor_scene_parts.refresh_move_target_highlight import RefreshMoveTargetHighlightMixin
from .editor_scene_parts.move_target_candidate_polygon import MoveTargetCandidatePolygonMixin
from .editor_scene_parts.edge_at import EdgeAtMixin
from .editor_scene_parts.delete_polygon import DeletePolygonMixin
from .editor_scene_parts.add_pasted_contacts import AddPastedContactsMixin
from .editor_scene_parts.clip_pasted_polygons_inset import ClipPastedPolygonsInsetMixin
from .editor_scene_parts.merge_pasted_polygons import MergePastedPolygonsMixin
from .editor_scene_parts.delete_vertex_at import DeleteVertexAtMixin
from .editor_scene_parts.finish_pending_polygon import FinishPendingPolygonMixin
from .editor_scene_parts.add_via_at import AddViaAtMixin
from .editor_scene_parts.add_rectangle_polygon import AddRectanglePolygonMixin
from .editor_scene_parts.clip_authored_polygons_image import ClipAuthoredPolygonsImageMixin
from .editor_scene_parts.add_brush_stroke import AddBrushStrokeMixin
from .editor_scene_parts.add_trace_stroke import AddTraceStrokeMixin
from .editor_scene_parts.subtract_pending_polygon import SubtractPendingPolygonMixin
from .editor_scene_parts.delete_vertices_in_rect import DeleteVerticesInRectMixin
from .editor_scene_parts.add_or_merge_polygon import AddOrMergePolygonMixin
from .editor_scene_parts.subtract_shape_scene import SubtractShapeSceneMixin
from .editor_scene_parts.merge_shape_into_scene import MergeShapeIntoSceneMixin
from .editor_scene_parts.small_inner_candidate import SmallInnerCandidateMixin
from .editor_scene_parts.find_overlapping_polygon_ids import FindOverlappingPolygonIdsMixin
from .editor_scene_parts.update_pending_path import UpdatePendingPathMixin
from .editor_scene_parts.tool_preview_path import ToolPreviewPathMixin
from .editor_scene_parts.refresh_polygon_item import RefreshPolygonItemMixin
from .editor_scene_parts.refresh_polygon_appearance import RefreshPolygonAppearanceMixin
from .editor_scene_parts.touched_polygon_ids import TouchedPolygonIdsMixin
from .editor_scene_parts.add_polygon_internal import AddPolygonInternalMixin
from .editor_scene_parts.add_polygons_internal import AddPolygonsInternalMixin
from .editor_scene_parts.gradient_arrows_path import (
    _DeferredGeometryRepairRequest,
    _gradient_arrows_path,
    _zoom_quantized_rgba,
)
from .editor_scene_parts.contact_spatial_index import _ContactSpatialIndex
from .editor_scene_parts.polygon_spatial_index import _PolygonSpatialIndex

class PolygonEditorScene(
    ConstructionMixin,
    ImagePixmapMixin,
    RenderPendingNeighborFramesMixin,
    DebugCandidatesMixin,
    ExtraLayersMixin,
    BeginZoomVectorRenderMixin,
    PolygonChangeProfileScopeMixin,
    PushPolygonLayerReplacementMixin,
    ComputePolygonLayerPatchMixin,
    ApplyPolygonPatchMixin,
    ApplyVectorPostprocessPolygonsMixin,
    BulkRestorePolygonsMixin,
    PolygonsMixin,
    SyncConductorHoverHighlightMixin,
    PickMoveTargetMixin,
    RefreshMoveTargetHighlightMixin,
    MoveTargetCandidatePolygonMixin,
    EdgeAtMixin,
    DeletePolygonMixin,
    AddPastedContactsMixin,
    ClipPastedPolygonsInsetMixin,
    MergePastedPolygonsMixin,
    DeleteVertexAtMixin,
    FinishPendingPolygonMixin,
    AddViaAtMixin,
    AddRectanglePolygonMixin,
    ClipAuthoredPolygonsImageMixin,
    AddBrushStrokeMixin,
    AddTraceStrokeMixin,
    SubtractPendingPolygonMixin,
    DeleteVerticesInRectMixin,
    AddOrMergePolygonMixin,
    SubtractShapeSceneMixin,
    MergeShapeIntoSceneMixin,
    SmallInnerCandidateMixin,
    FindOverlappingPolygonIdsMixin,
    UpdatePendingPathMixin,
    ToolPreviewPathMixin,
    RefreshPolygonItemMixin,
    RefreshPolygonAppearanceMixin,
    TouchedPolygonIdsMixin,
    AddPolygonInternalMixin,
    AddPolygonsInternalMixin,
    QGraphicsScene,
):
    polygonsChanged = pyqtSignal()
    activePolygonChanged = pyqtSignal(object)
    logRequested = pyqtSignal(str)

# Imported so existing callers and tests can patch or import these names
# from this module after the implementation moved into focused units.
_FACADE_EXPORTS = (
    _LOGGER,
    _ZOOM_COLOR_QUANTIZATION_STEP,
    _ZOOM_BATCH_TILE_SIZE,
    _ZOOM_RASTER_MAX_DIMENSION,
    _CONTACT_PASTE_SPATIAL_CELL_SIZE,
    _POLYGON_SPATIAL_CELL_SIZE,
    _OUTER_POLYGON_ITEM_Z,
    _HOLE_POLYGON_ITEM_Z,
    _OUTER_PICK_Z_SPAN,
    _PICK_CYCLE_DISTANCE,
    _DEFERRED_GEOMETRY_REPAIR_DELAY_MS,
    _ZOOM_QUANTIZED_CHANNELS,
    _DeferredGeometryRepairRequest,
    _gradient_arrows_path,
    _zoom_quantized_rgba,
    _ContactSpatialIndex,
    _PolygonSpatialIndex,
)
