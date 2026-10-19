"""layer_management: layer order list."""
from __future__ import annotations
from typing import Callable
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QListWidget,
    QListWidgetItem,
    QMenu,
    QWidget,
)
from kraken_manager.presentation.qt.models import LayerListItem

class LayerOrderList(QListWidget):
    orderChanged = pyqtSignal(object)
    layerSelected = pyqtSignal(str)
    layerActionRequested = pyqtSignal(str, str)
    layerPropertiesRequested = pyqtSignal(str)

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        action_availability: Callable[[str], tuple[bool, str]] | None = None,
    ) -> None:
        super().__init__(parent)
        self._action_availability = action_availability or (lambda _action: (True, ""))
        self.setObjectName("layerManagerList")
        self.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.model().rowsMoved.connect(lambda *_: self.orderChanged.emit(self.layer_ids()))
        self.currentItemChanged.connect(self._emit_selection)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._context_menu)

    def set_layers(self, layers: list[LayerListItem], selected_id: str = "") -> None:
        self.blockSignals(True)
        self.clear()
        selected_row = 0
        for row, layer in enumerate(layers):
            item = QListWidgetItem(layer.name)
            item.setData(Qt.ItemDataRole.UserRole, layer.layer_id)
            item.setToolTip(layer.layer_type)
            item.setForeground(QColor(layer.color))
            self.addItem(item)
            if layer.layer_id == selected_id:
                selected_row = row
        if self.count():
            self.setCurrentRow(selected_row)
        self.blockSignals(False)

    def layer_ids(self) -> tuple[str, ...]:
        return tuple(str(self.item(row).data(Qt.ItemDataRole.UserRole)) for row in range(self.count()))

    def select_layer(self, layer_id: str) -> None:
        for row in range(self.count()):
            if str(self.item(row).data(Qt.ItemDataRole.UserRole)) == str(layer_id):
                self.setCurrentRow(row)
                return

    def _emit_selection(self, current: QListWidgetItem | None, _previous: QListWidgetItem | None) -> None:
        if current is not None:
            self.layerSelected.emit(str(current.data(Qt.ItemDataRole.UserRole)))

    def _context_menu(self, position) -> None:
        item = self.itemAt(position)
        if item is None:
            return
        layer_id = str(item.data(Qt.ItemDataRole.UserRole))
        menu = QMenu(self)
        add_images = menu.addAction("Добавить слой изображений…")
        karakal = menu.addAction("Отправить слой в Karakal")
        stitching = menu.addAction("Управление сшивкой…")
        menu.addSeparator()
        rename_layer = menu.addAction("Переименовать…")
        archive_layer = menu.addAction("Архивировать")
        delete_layer = menu.addAction("Удалить слой…")
        menu.addSeparator()
        properties = menu.addAction("Свойства")
        by_action = {
            add_images: "add_image_representation",
            karakal: "karakal",
            stitching: "manage_stitching",
            rename_layer: "rename_layer",
            archive_layer: "archive_layer",
            delete_layer: "delete_layer",
        }
        for menu_action, code in by_action.items():
            enabled, reason = self._action_availability(code)
            menu_action.setEnabled(enabled)
            if not enabled and reason.startswith("Недостаточно прав"):
                menu_action.setVisible(False)
            if reason:
                menu_action.setToolTip(reason)
                menu_action.setStatusTip(reason)
        selected = menu.exec(self.viewport().mapToGlobal(position))
        if selected is properties:
            self.layerPropertiesRequested.emit(layer_id)
        elif selected in by_action:
            self.layerActionRequested.emit(layer_id, by_action[selected])
