from __future__ import annotations
import os
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch
import cv2
import numpy as np
from PyQt6.QtCore import QEvent, QPoint, QPointF, QRectF, QSignalBlocker, QSize, Qt, QTimer
from PyQt6.QtGui import QColor, QIcon, QImage, QMouseEvent, QPainter, QPixmap, QRegion, QWheelEvent
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsView,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QSlider,
    QSpinBox,
)
import contour.widget as widget_module
import contour.widget_parts.processing_mixin as processing_mixin_module
from contour.application.processing import (
    BatchImageResult,
    ContourDebugCandidate,
    ContourExtractionSettings,
    DisplaySettings,
    ImageProcessingState,
)
from contour.application.services.workspace_session import WorkspaceLoadResult
from contour.application.vector_geometry_postprocess import VectorGeometrySettings
from contour.domain import PolygonData, compute_polygon_metrics
from contour.graphics_items import EditablePolygonItem
from contour.graphics_view import (
    BrushMode,
    DeleteVertexMode,
    EditorTool,
    PolygonCreateMode,
    PolygonEditorScene,
    PolygonEditorView,
)
from contour.utils import draw_polygon_overlay
from contour.widget import PolygonExtractionWidget

from test_widget_interactions_parts.prepared_filter_result import PreparedFilterResultMixin
from test_widget_interactions_parts.auto_tune_preserves import AutoTunePreservesMixin
from test_widget_interactions_parts.recognition_mode_hides import RecognitionModeHidesMixin
from test_widget_interactions_parts.cached_frame_switch import CachedFrameSwitchMixin
from test_widget_interactions_parts.frame_apply_defers import FrameApplyDefersMixin
from test_widget_interactions_parts.x_hold_shows import XHoldShowsMixin
from test_widget_interactions_parts.applying_new_frame import ApplyingNewFrameMixin
from test_widget_interactions_parts.neighbor_sync_drops import NeighborSyncDropsMixin
from test_widget_interactions_parts.ingest_dropped_paths import IngestDroppedPathsMixin
from test_widget_interactions_parts.directory_session_restore import DirectorySessionRestoreMixin
from test_widget_interactions_parts.completed_background_load import CompletedBackgroundLoadMixin
from test_widget_interactions_parts.inner_hole_stays import InnerHoleStaysMixin
from test_widget_interactions_parts.toolbar_icons_use import ToolbarIconsUseMixin
from test_widget_interactions_parts.thumbnail_grid_uses import ThumbnailGridUsesMixin
from test_widget_interactions_parts.reapplying_same_image import ReapplyingSameImageMixin
from test_widget_interactions_parts.radial_thumbnail_fill import RadialThumbnailFillMixin
from test_widget_interactions_parts.frame_matrix_navigation import FrameMatrixNavigationMixin
from test_widget_interactions_parts.thumbnail_lod_change import ThumbnailLodChangeMixin
from test_widget_interactions_parts.asset_tabs_venn import AssetTabsVennMixin
from test_widget_interactions_parts.work_simulation_restores import WorkSimulationRestoresMixin
from test_widget_interactions_parts.extra_layer_row import ExtraLayerRowMixin
from test_widget_interactions_parts.display_settings_are import DisplaySettingsAreMixin
from test_widget_interactions_parts.neighbor_frame_navigation import NeighborFrameNavigationMixin
from test_widget_interactions_parts.contact_drag_commit import ContactDragCommitMixin
from test_widget_interactions_parts.zoom_uses_visible import ZoomUsesVisibleMixin
from test_widget_interactions_parts.middle_button_temporarily import MiddleButtonTemporarilyMixin
from test_widget_interactions_parts.ruler_tool_reports import RulerToolReportsMixin
from test_widget_interactions_parts.rectangle_polygon_created import RectanglePolygonCreatedMixin
from test_widget_interactions_parts.vertex_edit_tools import VertexEditToolsMixin
from test_widget_interactions_parts.brush_vertex_spacing import BrushVertexSpacingMixin
from test_widget_interactions_parts.middle_pan_during import MiddlePanDuringMixin
from test_widget_interactions_parts.small_independent_erase import SmallIndependentEraseMixin
from test_widget_interactions_parts.brush_editing_outer import BrushEditingOuterMixin
from test_widget_interactions_parts.delete_vertices_area import DeleteVerticesAreaMixin
from test_widget_interactions_parts.add_vertex_click import AddVertexClickMixin
from test_widget_interactions_parts.move_edge_translates import MoveEdgeTranslatesMixin
from test_widget_interactions_parts.polygon_tool_right import PolygonToolRightMixin
from test_widget_interactions_parts.area_selection_emits import AreaSelectionEmitsMixin
from test_widget_interactions_parts.move_vertex_keeps import MoveVertexKeepsMixin
from test_widget_interactions_parts.repeated_outer_edits import RepeatedOuterEditsMixin
from test_widget_interactions_parts.cutting_shape_hole import CuttingShapeHoleMixin
from test_widget_interactions_parts.tear_down import TearDownMixin
from test_widget_interactions_parts.switching_frames_autosaves import SwitchingFramesAutosavesMixin
from test_widget_interactions_parts.fast_switch_before import FastSwitchBeforeMixin
from test_widget_interactions_parts.switching_frames_does import SwitchingFramesDoesMixin
from test_widget_interactions_parts.extraction_mode_switch import ExtractionModeSwitchMixin
from test_widget_interactions_parts.dataset_mode_exports import DatasetModeExportsMixin
from test_widget_interactions_parts.persist_resolves_missing import PersistResolvesMissingMixin
from test_widget_interactions_parts.via_placement_skips import ViaPlacementSkipsMixin
from test_widget_interactions_parts.contact_drag_preview import ContactDragPreviewMixin
from test_widget_interactions_parts.paste_skips_contacts import PasteSkipsContactsMixin
from test_widget_interactions_parts.paste_refreshes_emits import PasteRefreshesEmitsMixin
from test_widget_interactions_parts.via_candidate_overlay_tests import ViaCandidateOverlayTests
from test_widget_interactions_parts.oversampled_rectangle_polygon import (
    _app,
    _rectangle_polygon,
    _oversampled_rectangle_polygon,
    _net_outline_area,
    _all_points_within,
)
from test_widget_interactions_parts.polygon_extraction_widget_load import (
    PolygonExtractionWidgetLoadImageTests,
    PolygonExtractionWidgetColorPickTests,
)
from test_widget_interactions_parts.polygon_extraction_widget_brush import PolygonExtractionWidgetBrushModeUiTests

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

class PolygonExtractionWidgetExtractionAutoApplyTests(
    PreparedFilterResultMixin,
    AutoTunePreservesMixin,
    RecognitionModeHidesMixin,
    CachedFrameSwitchMixin,
    FrameApplyDefersMixin,
    XHoldShowsMixin,
    ApplyingNewFrameMixin,
    NeighborSyncDropsMixin,
    IngestDroppedPathsMixin,
    DirectorySessionRestoreMixin,
    CompletedBackgroundLoadMixin,
    InnerHoleStaysMixin,
    ToolbarIconsUseMixin,
    ThumbnailGridUsesMixin,
    ReapplyingSameImageMixin,
    RadialThumbnailFillMixin,
    FrameMatrixNavigationMixin,
    ThumbnailLodChangeMixin,
    AssetTabsVennMixin,
    WorkSimulationRestoresMixin,
    ExtraLayerRowMixin,
    DisplaySettingsAreMixin,
    unittest.TestCase,
):
    pass

class PolygonEditorViewMiddleClickTests(
    NeighborFrameNavigationMixin,
    ContactDragCommitMixin,
    ZoomUsesVisibleMixin,
    MiddleButtonTemporarilyMixin,
    RulerToolReportsMixin,
    RectanglePolygonCreatedMixin,
    VertexEditToolsMixin,
    BrushVertexSpacingMixin,
    MiddlePanDuringMixin,
    SmallIndependentEraseMixin,
    BrushEditingOuterMixin,
    DeleteVerticesAreaMixin,
    AddVertexClickMixin,
    MoveEdgeTranslatesMixin,
    PolygonToolRightMixin,
    AreaSelectionEmitsMixin,
    MoveVertexKeepsMixin,
    RepeatedOuterEditsMixin,
    CuttingShapeHoleMixin,
    unittest.TestCase,
):
    pass

class PolygonExtractionWidgetAutosaveTests(
    TearDownMixin,
    SwitchingFramesAutosavesMixin,
    FastSwitchBeforeMixin,
    SwitchingFramesDoesMixin,
    ExtractionModeSwitchMixin,
    DatasetModeExportsMixin,
    PersistResolvesMissingMixin,
    unittest.TestCase,
):
    pass

class PolygonEditorSceneBrushPreviewTests(
    ViaPlacementSkipsMixin,
    ContactDragPreviewMixin,
    PasteSkipsContactsMixin,
    PasteRefreshesEmitsMixin,
    unittest.TestCase,
):
    pass

if __name__ == "__main__":
    unittest.main()

# Imported so existing callers and tests can patch or import these names
# from this module after the implementation moved into focused units.
_FACADE_EXPORTS = (
    ViaCandidateOverlayTests,
    _app,
    _rectangle_polygon,
    _oversampled_rectangle_polygon,
    _net_outline_area,
    _all_points_within,
    PolygonExtractionWidgetLoadImageTests,
    PolygonExtractionWidgetColorPickTests,
    PolygonExtractionWidgetBrushModeUiTests,
)
