"""layer_management: pipeline node item."""
from __future__ import annotations
from typing import TYPE_CHECKING
from typing import Callable
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPen
from PyQt6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsRectItem,
    QGraphicsSceneMouseEvent,
    QGraphicsSimpleTextItem,
    QMenu,
)

class PipelineNodeItem(QGraphicsRectItem):
    COLORS = {
        "source": "#4d738a",
        "binary": "#278b78",
        "vector": "#9a6324",
        "dataset": "#73559b",
        "model": "#9b4055",
        "job": "#4c5350",
        "blackbox": "#303430",
        "missing": "#303430",
        "karakal": "#3d718b",
        "confidence": "#3d718b",
        "stitching": "#73559b",
    }

    def __init__(
        self,
        node: PipelineNode,
        *,
        activate: Callable[[PipelineNode], None],
        request_action: Callable[[PipelineNode, str], None],
        request_properties: Callable[[PipelineNode], None],
        expand: Callable[[], None],
        collapse: Callable[[], None] | None,
        action_availability: Callable[[str], tuple[bool, str]],
        position_changed: Callable[[], None],
    ) -> None:
        super().__init__(0, 0, 190, 72)
        self.node = node
        self._activate = activate
        self._request_action = request_action
        self._request_properties = request_properties
        self._expand = expand
        self._collapse = collapse
        self._action_availability = action_availability
        self._position_changed = position_changed
        self.setFlag(self.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(self.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(self.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setBrush(QColor(self.COLORS.get(node.kind, "#4c5350")))
        pen = QPen(QColor("#f59e0b" if node.active else "#101210"), 3 if node.active else 1.5)
        pen.setStyle(Qt.PenStyle.DashLine if node.kind == "missing" else Qt.PenStyle.SolidLine)
        self.setPen(pen)
        title = QGraphicsSimpleTextItem(node.title, self)
        title.setBrush(QColor("#f3f4f6"))
        title.setPos(12, 9)
        if node.subtitle:
            subtitle = QGraphicsSimpleTextItem(node.subtitle[:52], self)
            subtitle.setBrush(QColor("#c3c8c3"))
            subtitle.setPos(12, 38)
        visible = {"время", "автор", "программа", "прогресс", "ошибка", "создан"}
        self.setToolTip("\n".join(f"{key}: {value}" for key, value in node.details.items() if key in visible and value))
        port_color = QColor("#22d3ee" if node.kind in {"source", "binary"} else "#facc15")
        for x in (-5.0, 185.0):
            port = QGraphicsEllipseItem(x, 31.0, 10.0, 10.0, self)
            port.setBrush(port_color)
            port.setPen(QPen(QColor("#111827"), 1.0))

    def itemChange(self, change: QGraphicsItem.GraphicsItemChange, value):  # type: ignore[override]
        result = super().itemChange(change, value)
        if change is QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self._position_changed()
        return result

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:  # type: ignore[override]
        if event.button() is Qt.MouseButton.LeftButton and self.node.kind == "blackbox":
            self._expand()
            event.accept()
            return
        super().mousePressEvent(event)
        if event.button() is Qt.MouseButton.LeftButton:
            self._activate(self.node)

    def mouseDoubleClickEvent(self, event: QGraphicsSceneMouseEvent) -> None:  # type: ignore[override]
        super().mouseDoubleClickEvent(event)

    def contextMenuEvent(self, event) -> None:  # type: ignore[override]
        actions = {
            "source": (
                ("Выдать работу…", "issue_work"),
                ("Подготовить выборку в Contour", "prepare_dataset"),
                ("Распознать готовой моделью", "recognize_external"),
                ("Получить уверенность модели", "confidence"),
                ("Получить CIF в Contour", "vectorize"),
                (
                    "Удалить слой изображений из проекта",
                    "archive_representation",
                ),
            ),
            "dataset": (
                ("Выдать работу…", "issue_work"),
                ("Обучить модель в NeuralImage", "train"),
            ),
            "model": (("Распознать исходники", "recognize"), ("Получить уверенность модели", "confidence")),
            "binary": (("Получить CIF в Contour", "vectorize"),),
            "vector": (
                ("Выдать работу…", "issue_work"),
                ("Добавить CIF из внешнего источника…", "add_external_vector"),
            ),
            "missing": (("Добавить CIF из внешнего источника…", "add_external_vector"),),
            "stitching": (
                ("Выдать работу…", "issue_work"),
                ("Управление сшивкой…", "manage_stitching"),
            ),
        }.get(self.node.kind, ())
        if bool(self.node.details.get("deletable", False)):
            actions = (*actions, ("Удалить шаг из pipeline", "delete_pipeline_step"))
        if self.node.representation_id:
            lifecycle = (
                ("Переименовать…", "rename_representation"),
                ("Изменить заметку…", "edit_representation_note"),
                (
                    "Деактивировать" if self.node.active else "Активировать",
                    "deactivate_representation" if self.node.active else "activate_representation",
                ),
                ("Архивировать", "archive_representation"),
            )
            actions = (*actions, *lifecycle)
        if self._collapse is not None:
            actions = (*actions, ("Свернуть", "collapse_pipeline"))
        menu = QMenu()
        by_action = {}
        menu.setToolTipsVisible(True)
        for label, code in actions:
            menu_action = menu.addAction(label)
            enabled, reason = (
                (True, "")
                if code == "collapse_pipeline"
                else self._action_availability(code)
            )
            menu_action.setEnabled(enabled)
            if not enabled and reason.startswith("Недостаточно прав"):
                menu_action.setVisible(False)
            if reason:
                menu_action.setToolTip(reason)
                menu_action.setStatusTip(reason)
            by_action[menu_action] = code
        if actions:
            menu.addSeparator()
        properties = menu.addAction("Свойства")
        selected = menu.exec(event.screenPos())
        if selected is properties:
            self._request_properties(self.node)
        elif selected in by_action:
            code = by_action[selected]
            if code == "collapse_pipeline" and self._collapse is not None:
                self._collapse()
            else:
                self._request_action(self.node, code)

if TYPE_CHECKING:
    from kraken_manager.presentation.qt.layer_management_parts.format_scalar import PipelineNode
