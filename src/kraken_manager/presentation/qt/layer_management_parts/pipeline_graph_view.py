"""layer_management: pipeline graph view."""
from __future__ import annotations
from dataclasses import replace
from typing import TYPE_CHECKING
from typing import Callable
from PyQt6.QtCore import QPointF, Qt, pyqtSignal
from kraken_core.settings import QSettings
from PyQt6.QtGui import QColor, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import (
    QGraphicsPathItem,
    QGraphicsView,
    QInputDialog,
    QWidget,
)

from kraken_manager.domain.layer_board import BoardNode, collapsed_block_title, parameter_lines
from kraken_manager.presentation.qt.layer_management_parts.format_scalar import PipelineNode

from kraken_manager.presentation.qt.layer_management_parts.pipeline_node_item import PipelineNodeItem

from kraken_manager.presentation.qt.layer_management_parts.pipeline_scene import PipelineScene

class PipelineGraphView(QGraphicsView):
    nodeActivated = pyqtSignal(object)
    nodeActionRequested = pyqtSignal(object, str)
    selectionActionRequested = pyqtSignal(object, str)
    nodePropertiesRequested = pyqtSignal(object)

    def __init__(
        self,
        settings: QSettings,
        parent: QWidget | None = None,
        *,
        action_availability: Callable[[str], tuple[bool, str]] | None = None,
    ) -> None:
        self.graph_scene = PipelineScene()
        super().__init__(self.graph_scene, parent)
        self.setObjectName("layerPipelineGraph")
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setRubberBandSelectionMode(Qt.ItemSelectionMode.IntersectsItemShape)
        self._panning = False
        self._pan_position = QPointF()
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self._settings = settings
        self._action_availability = action_availability or (lambda _action: (True, ""))
        self._snapshot: LayerPipelineSnapshot | None = None
        self._expanded_lane_id = ""
        self._items: dict[str, PipelineNodeItem] = {}
        self._edges: list[tuple[QGraphicsPathItem, PipelineNodeItem, PipelineNodeItem]] = []

    def apply_kraken_scene_theme(self) -> None:
        self.viewport().update()

    def set_snapshot(self, snapshot: LayerPipelineSnapshot) -> None:
        self.save_layout()
        self._snapshot = snapshot
        self._expanded_lane_id = str(
            self._settings.value(self._key("expanded-lane"), "", type=str)
        )
        if self._expanded_lane_id not in {lane.lane_id for lane in snapshot.lanes}:
            self._expanded_lane_id = ""
        self._rebuild()

    def _key(self, suffix: str) -> str:
        snap = self._snapshot
        return f"layer-manager/{snap.project_id}/{snap.layer_id}/{suffix}" if snap else f"layer-manager/{suffix}"

    def _rebuild(self) -> None:
        self.graph_scene.clear()
        self._items.clear()
        self._edges.clear()
        snapshot = self._snapshot
        if snapshot is None:
            return
        y = 30.0
        for lane in snapshot.lanes:
            visible_nodes = list(lane.nodes)
            visible_edges = list(lane.edges)
            internal_nodes = [
                node
                for node in lane.nodes
                if node.kind not in {"source", "vector", "missing"}
            ]
            is_expanded = lane.lane_id == self._expanded_lane_id
            blackbox = _collapsed_node(lane.lane_id, internal_nodes)
            if internal_nodes and not is_expanded:
                sources = [node for node in lane.nodes if node.kind == "source"]
                vectors = [node for node in lane.nodes if node.kind in {"vector", "missing"}]
                visible_nodes = [*sources, blackbox, *vectors]
                visible_edges = []
                if sources:
                    visible_edges.append((sources[0].node_id, blackbox.node_id))
                for vector in vectors:
                    visible_edges.append((blackbox.node_id, vector.node_id))
            for column, node in enumerate(visible_nodes):
                if not node.representation_id:
                    note = str(self._settings.value(self._key(f"note/{node.node_id}"), "") or "")
                    node = replace(node, details={**node.details, "note": note})
                belongs_to_expanded_box = is_expanded and any(value.node_id == node.node_id for value in internal_nodes)
                item = PipelineNodeItem(
                    node,
                    activate=lambda value: self.nodeActivated.emit(value),
                    request_action=self._request_node_action,
                    request_selection_action=lambda nodes, action: self.selectionActionRequested.emit(nodes, action),
                    selected_nodes=self._selected_nodes,
                    request_properties=lambda value: self.nodePropertiesRequested.emit(value),
                    expand=lambda lane_id=lane.lane_id: self.expand_lane(lane_id),
                    collapse=(
                        (lambda lane_id=lane.lane_id: self.collapse_lane(lane_id))
                        if belongs_to_expanded_box
                        else None
                    ),
                    action_availability=self._action_availability,
                    position_changed=self._update_edges,
                )
                saved = self._settings.value(self._key(f"node/{node.node_id}"))
                if saved is not None and isinstance(saved, QPointF):
                    item.setPos(saved)
                else:
                    item.setPos(35 + column * 245, y)
                self.graph_scene.addItem(item)
                self._items[node.node_id] = item
            for source_id, target_id in visible_edges:
                source, target = self._items.get(source_id), self._items.get(target_id)
                if source is None or target is None:
                    continue
                edge = QGraphicsPathItem()
                edge.setPen(QPen(QColor("#9ca39c"), 2))
                edge.setZValue(-1)
                self.graph_scene.addItem(edge)
                self._edges.append((edge, source, target))
            self._update_edges()
            y += 168
        self.graph_scene.setSceneRect(self.graph_scene.itemsBoundingRect().adjusted(-80, -80, 120, 120))

    def _request_node_action(self, node: PipelineNode, action: str) -> None:
        if action != "edit_node_note":
            self.nodeActionRequested.emit(node, action)
            return
        value, accepted = QInputDialog.getMultiLineText(
            self, "Заметка блока", "Заметка:", text=str(node.details.get("note") or "")
        )
        if accepted:
            self.save_layout()
            self._settings.setValue(self._key(f"note/{node.node_id}"), value.strip())
            self._rebuild()

    def _update_edges(self) -> None:
        for edge, source, target in self._edges:
            source_rect = source.sceneBoundingRect()
            target_rect = target.sceneBoundingRect()
            start = QPointF(source_rect.right(), source_rect.center().y())
            end = QPointF(target_rect.left(), target_rect.center().y())
            bend = max(60.0, abs(end.x() - start.x()) * 0.45)
            path = QPainterPath(start)
            path.cubicTo(start.x() + bend, start.y(), end.x() - bend, end.y(), end.x(), end.y())
            edge.setPath(path)

    def expand_lane(self, lane_id: str) -> None:
        self.save_layout()
        self._expanded_lane_id = str(lane_id)
        self._settings.setValue(self._key("expanded-lane"), self._expanded_lane_id)
        self._rebuild()

    def collapse_lane(self, lane_id: str) -> None:
        if self._expanded_lane_id != str(lane_id):
            return
        self.save_layout()
        self._expanded_lane_id = ""
        self._settings.remove(self._key("expanded-lane"))
        self._rebuild()

    def save_layout(self) -> None:
        for identifier, item in self._items.items():
            self._settings.setValue(self._key(f"node/{identifier}"), item.pos())

    def _selected_nodes(self) -> tuple[PipelineNode, ...]:
        return tuple(
            item.node
            for item in self.graph_scene.selectedItems()
            if isinstance(item, PipelineNodeItem)
        )

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() is Qt.MouseButton.MiddleButton:
            self._panning = True
            self._pan_position = event.position()
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        if self._panning:
            delta = event.position() - self._pan_position
            self._pan_position = event.position()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(delta.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(delta.y()))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        if event.button() is Qt.MouseButton.MiddleButton and self._panning:
            self._panning = False
            self.viewport().unsetCursor()
            event.accept()
            return
        super().mouseReleaseEvent(event)
        self.save_layout()

    def wheelEvent(self, event) -> None:  # type: ignore[override]
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
            self.scale(factor, factor)
            event.accept()
            return
        super().wheelEvent(event)

def _collapsed_node(lane_id: str, internal_nodes: list[PipelineNode]) -> PipelineNode:
    described = tuple(
        BoardNode(
            node.node_id,
            node.kind,
            node.title,
            capability=str(node.details.get("capability") or node.details.get("operation") or ""),
            parameters=node.details.get("параметры") if isinstance(node.details.get("параметры"), dict) else None,
        )
        for node in internal_nodes
    )
    primary = next(
        (
            node
            for node in internal_nodes
            if str(node.details.get("capability") or "") == "frames.dataset.prepare.v1"
            or str(node.details.get("operation") or "") == "frames.dataset.prepare.v1"
        ),
        internal_nodes[-1] if internal_nodes else None,
    )
    details = {} if primary is None else dict(primary.details)
    if primary is not None:
        details["collapsed_kind"] = primary.kind
        capability = str(details.get("capability") or details.get("operation") or "")
        parameters = details.get("параметры") if isinstance(details.get("параметры"), dict) else None
        lines = parameter_lines(parameters, capability=capability)
        if lines:
            details["parameter_lines"] = lines
    return PipelineNode(
        f"{lane_id}:blackbox",
        collapsed_block_title(described),
        "blackbox",
        "Нажмите, чтобы раскрыть",
        details=details,
    )


if TYPE_CHECKING:
    from kraken_manager.presentation.qt.layer_management_parts.format_scalar import LayerPipelineSnapshot
