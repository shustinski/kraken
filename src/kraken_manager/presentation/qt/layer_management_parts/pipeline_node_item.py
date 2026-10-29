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
    QGraphicsTextItem,
    QApplication,
    QMenu,
)
from kraken_manager.domain.layer_board import (
    BoardNode,
    multi_selection_actions,
    parameter_lines,
    selection_kind,
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
        request_selection_action: Callable[[tuple[PipelineNode, ...], str], None] | None = None,
        selected_nodes: Callable[[], tuple[PipelineNode, ...]] | None = None,
    ) -> None:
        self._parameter_lines = _node_parameter_lines(node)
        note = str(node.details.get("note") or "").strip()
        note_item = QGraphicsTextItem() if note else None
        note_height = 0.0
        if note_item is not None:
            note_item.setPlainText(note)
            note_item.setTextWidth(166)
            note_item.document().setDocumentMargin(0)
            note_height = note_item.boundingRect().height() + 8
        height = 72 + note_height + 16 * len(self._parameter_lines)
        super().__init__(0, 0, 190, height)
        self.node = node
        self._press_screen_pos = None
        self._dragged = False
        self._activate = activate
        self._request_action = request_action
        self._request_properties = request_properties
        self._expand = expand
        self._collapse = collapse
        self._action_availability = action_availability
        self._position_changed = position_changed
        self._request_selection_action = request_selection_action
        self._selected_nodes = selected_nodes or (lambda: ())
        self.setFlag(self.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(self.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(self.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        paint_kind = str(node.details.get("collapsed_kind") or node.kind)
        self.setBrush(QColor(self.COLORS.get(paint_kind, self.COLORS.get(node.kind, "#4c5350"))))
        pen = QPen(QColor("#f59e0b" if node.active else "#101210"), 3 if node.active else 1.5)
        pen.setStyle(Qt.PenStyle.DashLine if node.kind == "missing" else Qt.PenStyle.SolidLine)
        self.setPen(pen)
        title = QGraphicsSimpleTextItem(node.title, self)
        title.setBrush(QColor("#f3f4f6"))
        title.setPos(12, 9)
        if note_item is not None:
            note_item.setParentItem(self)
            note_item.setDefaultTextColor(QColor("#f3f4f6"))
            note_item.setPos(12, 30)
        if node.subtitle:
            subtitle = QGraphicsSimpleTextItem(node.subtitle[:52], self)
            subtitle.setBrush(QColor("#c3c8c3"))
            subtitle.setPos(12, 38 + note_height)
        for offset, line in enumerate(self._parameter_lines):
            parameter = QGraphicsSimpleTextItem(line[:42], self)
            parameter.setBrush(QColor("#e5e7eb"))
            parameter.setPos(12, 58 + note_height + offset * 16)
        visible = {"время", "автор", "программа", "прогресс", "ошибка", "создан"}
        self.setToolTip("\n".join(f"{key}: {value}" for key, value in node.details.items() if key in visible and value))
        port_color = QColor("#22d3ee" if node.kind in {"source", "binary"} else "#facc15")
        port_y = height / 2 - 5
        for x in (-5.0, 185.0):
            port = QGraphicsEllipseItem(x, port_y, 10.0, 10.0, self)
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
            self._press_screen_pos = event.screenPos()
            self._dragged = False

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if self._press_screen_pos is not None:
            distance = (event.screenPos() - self._press_screen_pos).manhattanLength()
            self._dragged |= distance >= QApplication.startDragDistance()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        activate = (
            event.button() is Qt.MouseButton.LeftButton
            and self._press_screen_pos is not None
            and not self._dragged
            and not event.modifiers() & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier)
        )
        self._press_screen_pos = None
        super().mouseReleaseEvent(event)
        if activate:
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
                ("Назначить слои для проверки", "assign_review_layers"),
                ("Добавить CIF из внешнего источника…", "add_external_vector"),
            ),
            "missing": (("Добавить CIF из внешнего источника…", "add_external_vector"),),
            "stitching": (
                ("Выдать работу…", "issue_work"),
                ("Управление сшивкой…", "manage_stitching"),
            ),
        }.get(self.node.kind, ())
        if (
            self.node.kind in {"job", "operation"}
            and self.node.state not in {"", "succeeded", "cancelled", "failed"}
            and (self.node.details.get("job_id") or self.node.details.get("workflow_operation"))
        ):
            actions = (*actions, ("Продолжить выполнение", "continue_pipeline_step"))
        if not self.node.representation_id:
            actions = (*actions, ("Изменить заметку…", "edit_node_note"))
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
        selected = self._selected_nodes()
        multi = None if self not in self.scene().selectedItems() else multi_selection_actions(
            tuple(
                BoardNode(node.node_id, selection_kind(node.kind, node.subtitle), node.title)
                for node in selected
            )
        )
        selection_menu = bool(multi)
        if selection_menu:
            actions = tuple((item.label, item.code) for item in multi)
        menu = QMenu()
        by_action = {}
        menu.setToolTipsVisible(True)
        for label, code in actions:
            menu_action = menu.addAction(label)
            enabled, reason = (
                (True, "")
                if code in {"collapse_pipeline", "edit_node_note"}
                else self._action_availability(code)
            )
            menu_action.setEnabled(enabled)
            if not enabled and reason.startswith("Недостаточно прав"):
                menu_action.setVisible(False)
            if reason:
                menu_action.setToolTip(reason)
                menu_action.setStatusTip(reason)
            by_action[menu_action] = code
        if actions and not selection_menu:
            menu.addSeparator()
        properties = None if selection_menu else menu.addAction("Свойства")
        chosen = menu.exec(event.screenPos())
        if properties is not None and chosen is properties:
            self._request_properties(self.node)
        elif chosen in by_action:
            code = by_action[chosen]
            if code == "collapse_pipeline" and self._collapse is not None:
                self._collapse()
            elif selection_menu and self._request_selection_action is not None:
                self._request_selection_action(selected, code)
            else:
                self._request_action(self.node, code)


def _node_parameter_lines(node: PipelineNode) -> tuple[str, ...]:
    stored = node.details.get("parameter_lines")
    if isinstance(stored, (list, tuple)) and stored:
        return tuple(str(line) for line in stored)
    parameters = node.details.get("параметры")
    capability = str(node.details.get("capability") or node.details.get("operation") or "")
    if node.kind not in {"job", "blackbox"}:
        capability = ""
    return parameter_lines(parameters if isinstance(parameters, dict) else None, capability=capability)

if TYPE_CHECKING:
    from kraken_manager.presentation.qt.layer_management_parts.format_scalar import PipelineNode
