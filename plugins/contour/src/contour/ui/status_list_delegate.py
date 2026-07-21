"""List row painting helpers for Contour sidebars.

Kraken applies application-wide QSS for ``QListWidget::item``, which overrides
per-item ``BackgroundRole`` painting from the default delegate. This delegate
fills the row from ``Qt.ItemDataRole.BackgroundRole`` first, then lets the base
delegate draw focus, selection, and labels.
"""

from __future__ import annotations

from PyQt6.QtCore import QModelIndex, Qt
from PyQt6.QtGui import QColor, QBrush, QPainter
from PyQt6.QtWidgets import QAbstractItemView, QStyledItemDelegate, QStyleOptionViewItem


class StatusBackgroundListDelegate(QStyledItemDelegate):
    """Paint status tint from ``BackgroundRole`` under global application QSS."""

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex) -> None:
        raw = index.data(Qt.ItemDataRole.BackgroundRole)
        if isinstance(raw, QColor) and raw.isValid():
            painter.save()
            painter.fillRect(option.rect, raw)
            painter.restore()
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        opt.backgroundBrush = QBrush()
        super().paint(painter, opt, index)


def attach_status_row_delegate(list_view: QAbstractItemView) -> None:
    list_view.setItemDelegate(StatusBackgroundListDelegate(list_view))
