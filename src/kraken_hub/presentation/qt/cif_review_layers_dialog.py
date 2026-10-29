"""Choose and order the image layers checked together with a CIF block."""

from __future__ import annotations

from collections.abc import Sequence

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
)

from kraken_manager.domain.layer_board import order_review_images


class CifReviewLayersDialog(QDialog):
    def __init__(
        self,
        *,
        source_layer_name: str,
        images: Sequence[tuple[str, str]],
        selected_ids: Sequence[str] = (),
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Назначить слои для проверки")
        self.setMinimumSize(760, 460)
        self._ids = {str(identifier) for identifier, _label in images}
        root = QVBoxLayout(self)
        hint = QLabel(
            "Слева — все слои с изображениями, включая варианты одного слоя. "
            f"Справа закреплён слой «{source_layer_name}»: перетащите сюда изображения и расставьте их в нужном порядке.",
            self,
        )
        hint.setWordWrap(True)
        root.addWidget(hint)
        columns = QHBoxLayout()
        self.available = self._list("cifReviewAvailable")
        self.assigned = self._list("cifReviewAssigned")
        columns.addWidget(self._column("Все изображения", self.available))
        columns.addWidget(self._column(f"Слой «{source_layer_name}»", self.assigned))
        root.addLayout(columns, 1)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Сохранить")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("Отмена")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)
        assigned = [str(value) for value in selected_ids if str(value) in self._ids]
        labels = {str(identifier): label for identifier, label in images}
        for identifier in assigned:
            self._add(self.assigned, identifier, labels[identifier])
        for identifier, label in images:
            if str(identifier) not in assigned:
                self._add(self.available, str(identifier), label)
        self.available.model().rowsInserted.connect(self._sync_order)
        self.assigned.model().rowsInserted.connect(self._sync_order)
        self.assigned.model().rowsMoved.connect(self._sync_order)

    def ordered_ids(self) -> tuple[str, ...]:
        order: tuple[str, ...] = ()
        for row in range(self.assigned.count()):
            identifier = str(self.assigned.item(row).data(Qt.ItemDataRole.UserRole))
            order = order_review_images(order, identifier, len(order))
        return order

    def _sync_order(self, *_args) -> None:
        # Dropping the same image twice is rejected by the order rule.
        self.ordered_ids()

    @staticmethod
    def _column(title: str, widget: QListWidget) -> QVBoxLayout:
        column = QVBoxLayout()
        label = QLabel(title)
        label.setWordWrap(True)
        column.addWidget(label)
        column.addWidget(widget, 1)
        return column

    @staticmethod
    def _list(object_name: str) -> QListWidget:
        widget = QListWidget()
        widget.setObjectName(object_name)
        widget.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        widget.setDragEnabled(True)
        widget.setAcceptDrops(True)
        widget.setDropIndicatorShown(True)
        widget.setDragDropMode(QAbstractItemView.DragDropMode.DragDrop)
        widget.setDefaultDropAction(Qt.DropAction.MoveAction)
        return widget

    @staticmethod
    def _add(widget: QListWidget, identifier: str, label: str) -> None:
        item = QListWidgetItem(label)
        item.setData(Qt.ItemDataRole.UserRole, identifier)
        item.setFlags(
            item.flags()
            | Qt.ItemFlag.ItemIsDragEnabled
            | Qt.ItemFlag.ItemIsDropEnabled
            | Qt.ItemFlag.ItemIsEnabled
            | Qt.ItemFlag.ItemIsSelectable
        )
        widget.addItem(item)
