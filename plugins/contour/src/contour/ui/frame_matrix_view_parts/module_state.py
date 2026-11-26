"""frame_matrix_view: module state."""
from __future__ import annotations
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor

# Zoom levels at which the thumbnail source size steps up (multiples of the cell size).
_THUMBNAIL_LOD_LEVELS = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0)

_THUMBNAIL_PIXMAP_ROLE = int(Qt.ItemDataRole.UserRole) + 1001

_FRAME_ID_ROLE = int(Qt.ItemDataRole.UserRole) + 1002

_STATUS_MARKER_COLORS = {
    "viewed": QColor("#475569"),
    "modified": QColor("#B45309"),
    "saved": QColor("#047857"),
    "no_vector": QColor("#B91C1C"),
}
