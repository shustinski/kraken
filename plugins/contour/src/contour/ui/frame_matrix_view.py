from __future__ import annotations
import re
from pathlib import Path
from PyQt6.QtCore import QPoint, QRect, QRectF, QSize, Qt, QThreadPool, QTimer, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QGuiApplication, QImage, QPainter, QPen, QPixmap, QWheelEvent
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QListWidget,
    QListWidgetItem,
    QWidget,
)
from ..adapters.qt.pyramid import PyramidThumbnailLoadRunnable
from ..application.frame_lod import PyramidFrameStore
from ..graphics.viewport_navigation import (
    MAX_ZOOM_FACTOR,
    MIN_ZOOM_FACTOR,
    clamp_zoom_factor,
    viewport_scroll_correction_after_scale_reanchor,
    zoom_factor_for_wheel_delta,
)
from .item_status_painting import FRAME_STATUS_ROLE
from .large_dataset import clamp_thumbnail_source_size

from .frame_matrix_view_parts.module_state import (
    _OPENGL_VIEWPORT_ENABLED,
    _OPENGL_DISABLED_PLATFORMS,
    _MATRIX_ZOOM_FRAME_MS,
    _MATRIX_ZOOM_EASING_FRACTION,
    _MATRIX_ZOOM_SETTLE_RATIO,
    _LOD_LABEL_MIN_ZOOM,
    _LOD_STATUS_MIN_ZOOM,
    _THUMBNAIL_LOD_LEVELS,
    _THUMBNAIL_PIXMAP_ROLE,
    _FRAME_ID_ROLE,
    _RENDER_REGION_BUFFER_CELLS,
    _FOCUS_NEIGHBOR_ROW_BUFFER,
    _STATUS_MARKER_COLORS,
)
from .frame_matrix_view_parts.construction import ConstructionMixin
from .frame_matrix_view_parts.attach_graphics_group_item import AttachGraphicsGroupItemMixin
from .frame_matrix_view_parts.mouse_press_event import MousePressEventMixin
from .frame_matrix_view_parts.apply_zoom_at_viewport import ApplyZoomAtViewportMixin
from .frame_matrix_view_parts.layout_item_geometry import LayoutItemGeometryMixin
from .frame_matrix_view_parts.apply_lod_indexes import ApplyLodIndexesMixin
from .frame_matrix_view_parts.pyramid_thumbnail_loaded import PyramidThumbnailLoadedMixin
from .frame_matrix_view_parts.cover_crop_pixmap_cell import CoverCropPixmapCellMixin
from .frame_matrix_view_parts.trailing_frame_number import _trailing_frame_number

try:
    import shiboken6
except ImportError:
    shiboken6 = None

class FrameMatrixGraphicsView(
    ConstructionMixin,
    AttachGraphicsGroupItemMixin,
    MousePressEventMixin,
    ApplyZoomAtViewportMixin,
    LayoutItemGeometryMixin,
    ApplyLodIndexesMixin,
    PyramidThumbnailLoadedMixin,
    CoverCropPixmapCellMixin,
    QGraphicsView,
):
    """QGraphicsView-backed frame matrix with the QListWidget subset Contour uses."""
    itemClicked = pyqtSignal(QListWidgetItem)
    thumbnailLodChanged = pyqtSignal(int, int)
    frameNavigationRequested = pyqtSignal(object)
    frameContextMenuRequested = pyqtSignal(object, QPoint)

# Imported so existing callers and tests can patch or import these names
# from this module after the implementation moved into focused units.
_FACADE_EXPORTS = (
    _OPENGL_VIEWPORT_ENABLED,
    _OPENGL_DISABLED_PLATFORMS,
    _MATRIX_ZOOM_FRAME_MS,
    _MATRIX_ZOOM_EASING_FRACTION,
    _MATRIX_ZOOM_SETTLE_RATIO,
    _LOD_LABEL_MIN_ZOOM,
    _LOD_STATUS_MIN_ZOOM,
    _THUMBNAIL_LOD_LEVELS,
    _THUMBNAIL_PIXMAP_ROLE,
    _FRAME_ID_ROLE,
    _RENDER_REGION_BUFFER_CELLS,
    _FOCUS_NEIGHBOR_ROW_BUFFER,
    _STATUS_MARKER_COLORS,
    _trailing_frame_number,
)
