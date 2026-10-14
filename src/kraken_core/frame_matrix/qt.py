"""Virtualized, domain-neutral frame matrix for project workspaces."""

from __future__ import annotations
import math
import logging
from collections.abc import Iterable, Iterator, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any
from PyQt6.QtCore import QPoint, QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QContextMenuEvent, QKeyEvent, QMouseEvent, QPainter, QPen, QPixmap, QWheelEvent
from PyQt6.QtWidgets import (
    QGraphicsItem,
    QGraphicsScene,
    QGraphicsView,
    QStyleOptionGraphicsItem,
    QWidget,
)
from .cache import ThumbnailCacheCoordinator
from .models import (
    LodBand,
    MatrixAggregate,
    MatrixBounds,
    MatrixItem,
    MatrixOrientation,
    MatrixSession,
    MatrixViewportRequest,
    MatrixViewportResult,
)
from .ports import MatrixAssetSource, MatrixDataSource, MatrixLayerRenderer
from .qt_renderers import DEFAULT_RENDERERS
from .storage import StoreNamespace, StorePolicy, ThumbnailKey, ThumbnailStore

from .qt_parts.module_state import (
    GridOrientation,
    _LOGGER,
    DEFAULT_STATUS_COLORS,
    STATUS_PRIORITY,
)
from .qt_parts.construction import ConstructionMixin
from .qt_parts.update_visible_tiles import UpdateVisibleTilesMixin
from .qt_parts.paint_tile_cells import PaintTileCellsMixin
from .qt_parts.cell_visual_color import CellVisualColorMixin
from .qt_parts.context_menu_event import ContextMenuEventMixin
from .qt_parts.construction_2 import Construction2Mixin
from .qt_parts.poll_background_results import PollBackgroundResultsMixin
from .qt_parts.apply_viewport_result import ApplyViewportResultMixin
from .qt_parts.matrix_lod import MatrixLod
from .qt_parts.frame_rect import FrameRect
from .qt_parts.viewport_tile_item import (
    FrameSelection,
    FrameCellData,
    FrameContext,
    _ViewportTileItem,
)
from .qt_parts.frame_matrix_minimap import FrameMatrixMinimap

class FrameMatrixView(
    ConstructionMixin,
    UpdateVisibleTilesMixin,
    PaintTileCellsMixin,
    CellVisualColorMixin,
    ContextMenuEventMixin,
    QGraphicsView,
):
    """Sparse frame matrix with tile virtualization, LOD and rectangle selection.

    The view never creates an object for every project frame. It maintains only
    tiles intersecting the viewport; tile size grows as the user zooms out.
    """
    selectionChanged = pyqtSignal(object)
    frameActivated = pyqtSignal(int, int)
    contextMenuRequested = pyqtSignal(object, object)
    viewportChanged = pyqtSignal(object)
    lodChanged = pyqtSignal(str)
    CELL_SIZE = 48.0
    CELL_GAP = 0.0
    CELL_PITCH = CELL_SIZE + CELL_GAP
    BASE_TILE_CELLS = 16
    TILE_DEVICE_TARGET = 110.0
    MIN_ZOOM = 1.0e-8
    MAX_ZOOM = 8.0
    DETAILS_LOD = 0.72
    CELLS_LOD = 0.18

class FrameMatrixWidget(
    Construction2Mixin,
    PollBackgroundResultsMixin,
    ApplyViewportResultMixin,
    FrameMatrixView,
):
    """Universal matrix view backed by a viewport data source and thumbnail store."""
    loadingChanged = pyqtSignal(bool)
    errorOccurred = pyqtSignal(str)
    sessionChanged = pyqtSignal(object)

# Imported so existing callers and tests can patch or import these names
# from this module after the implementation moved into focused units.
_FACADE_EXPORTS = (
    GridOrientation,
    _LOGGER,
    DEFAULT_STATUS_COLORS,
    STATUS_PRIORITY,
    MatrixLod,
    FrameRect,
    FrameSelection,
    FrameCellData,
    FrameContext,
    _ViewportTileItem,
    FrameMatrixMinimap,
)

__all__ = [
    "FrameCellData",
    "FrameContext",
    "FrameMatrixMinimap",
    "FrameMatrixView",
    "FrameMatrixWidget",
    "FrameRect",
    "FrameSelection",
    "GridOrientation",
    "MatrixLod",
]
