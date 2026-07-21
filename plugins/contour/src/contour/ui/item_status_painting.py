from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QBrush, QColor
from PyQt6.QtWidgets import QListWidgetItem

from ..application.frame_asset_sync import (
    background_hex_image_paint_status_for_theme,
    background_hex_vector_status_for_theme,
    classify_image_side_paint_status,
    foreground_hex_image_paint_status_for_theme,
    foreground_hex_vector_status_for_theme,
)

FRAME_STATUS_ROLE = int(Qt.ItemDataRole.UserRole) + 1


def paint_image_row_item(
    item: QListWidgetItem,
    image_path: str,
    *,
    image_has_changes: bool,
    has_vector_overlay: bool,
    vector_index_active: bool = False,
    extraction_enabled: bool,
    viewed: bool,
    persisted_highlight: bool,
    theme: str | None = "dark",
    show_text: bool = True,
) -> None:
    normalized = str(Path(image_path))
    painted = classify_image_side_paint_status(
        has_matching_cif=has_vector_overlay,
        vector_index_active=vector_index_active,
        never_opened=True if extraction_enabled else not viewed,
        polygons_dirty=False if extraction_enabled else image_has_changes,
        persist_highlight=False if extraction_enabled else persisted_highlight,
    )
    item.setData(FRAME_STATUS_ROLE, painted.value)
    _set_background(item, background_hex_image_paint_status_for_theme(painted, theme=theme))
    fg = QColor(
        foreground_hex_image_paint_status_for_theme(
            painted,
            has_matching_cif=has_vector_overlay,
            theme=theme,
        )
    )
    item.setForeground(QBrush(fg))
    item.setText(Path(normalized).stem if show_text else "")


def paint_vector_row_item(item: QListWidgetItem, stem: str, status, *, theme: str | None = "dark") -> None:
    item.setData(FRAME_STATUS_ROLE, status.value)
    _set_background(item, background_hex_vector_status_for_theme(status, theme=theme))
    foreground = foreground_hex_vector_status_for_theme(status, theme=theme)
    item.setForeground(QBrush(QColor(foreground)) if foreground else QBrush())
    raw_path = item.data(Qt.ItemDataRole.UserRole)
    if raw_path:
        item.setText(Path(str(raw_path)).stem)


def _set_background(item: QListWidgetItem, hex_background: str | None) -> None:
    if hex_background:
        tint = QColor(hex_background)
        item.setBackground(QBrush(tint))
        # Windows / some styles ignore setBackground for items; BackgroundRole is respected more reliably.
        item.setData(Qt.ItemDataRole.BackgroundRole, tint)
    else:
        item.setBackground(QBrush())
        item.setData(Qt.ItemDataRole.BackgroundRole, None)
