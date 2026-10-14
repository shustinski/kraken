"""qt: module state."""
from __future__ import annotations
import logging
from kraken_core.frame_matrix.models import (
    MatrixOrientation,
)

GridOrientation = MatrixOrientation

_LOGGER = logging.getLogger(__name__)

DEFAULT_STATUS_COLORS: dict[str, str] = {
    "empty": "#1f2937",
    "image_ready": "#2563eb",
    "processing": "#7c3aed",
    "vectorized": "#0891b2",
    "in_review": "#d97706",
    "returned_unchanged": "#65a30d",
    "returned_changed": "#ca8a04",
    "approved": "#15803d",
    "changes_requested": "#ea580c",
    "conflict": "#dc2626",
    "error": "#b91c1c",
}

STATUS_PRIORITY = {
    "empty": 0,
    "image_ready": 1,
    "vectorized": 2,
    "processing": 3,
    "in_review": 4,
    "returned_unchanged": 5,
    "returned_changed": 6,
    "approved": 7,
    "changes_requested": 8,
    "conflict": 9,
    "error": 10,
}
