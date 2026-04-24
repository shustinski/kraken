"""Custom :class:`QListWidget` subclass used by the pipeline tab."""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QAbstractItemView, QListWidget, QMenu

from ..i18n import active_language


class PipelineListWidget(QListWidget):
    """List widget that supports drag-reorder, delete-key removal and a context menu."""

    deletePressed = pyqtSignal()
    orderChanged = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.deletePressed.emit()
            event.accept()
            return
        super().keyPressEvent(event)

    def dropEvent(self, event) -> None:
        super().dropEvent(event)
        self.orderChanged.emit()

    def contextMenuEvent(self, event) -> None:
        if self.currentRow() < 0:
            return
        menu = QMenu(self)
        delete_action = menu.addAction("Удалить" if active_language() == "ru" else "Delete")
        if menu.exec(event.globalPos()) == delete_action:
            self.deletePressed.emit()
