"""frame_matrix_view: module state."""
from __future__ import annotations
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor

_OPENGL_VIEWPORT_ENABLED = True

_OPENGL_DISABLED_PLATFORMS = {"offscreen", "minimal"}

_MATRIX_ZOOM_FRAME_MS = 16

_MATRIX_ZOOM_EASING_FRACTION = 0.55

_MATRIX_ZOOM_SETTLE_RATIO = 0.001

_LOD_LABEL_MIN_ZOOM = 0.75

_LOD_STATUS_MIN_ZOOM = 0.22

_THUMBNAIL_LOD_LEVELS = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0)

_THUMBNAIL_PIXMAP_ROLE = int(Qt.ItemDataRole.UserRole) + 1001

_FRAME_ID_ROLE = int(Qt.ItemDataRole.UserRole) + 1002

_RENDER_REGION_BUFFER_CELLS = 2

# Keep full rows around the current frame visible (matches thumbnail load window).
_FOCUS_NEIGHBOR_ROW_BUFFER = 0

_STATUS_MARKER_COLORS = {
    "viewed": QColor("#475569"),
    "modified": QColor("#B45309"),
    "saved": QColor("#047857"),
    "no_vector": QColor("#B91C1C"),
}
