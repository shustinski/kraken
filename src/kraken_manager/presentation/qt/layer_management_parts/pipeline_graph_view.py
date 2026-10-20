"""layer_management: pipeline graph view."""
from __future__ import annotations
from typing import TYPE_CHECKING
from typing import Callable
from PyQt6.QtCore import QPointF, QSettings, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import (
    QGraphicsPathItem,
    QGraphicsView,
    QWidget,
)

from kraken_manager.presentation.qt.layer_management_parts.format_scalar import PipelineNode

from kraken_manager.presentation.qt.layer_management_parts.pipeline_node_item import PipelineNodeItem

from kraken_manager.presentation.qt.layer_management_parts.pipeline_scene import PipelineScene

class PipelineGraphView(QGraphicsView):
    nodeActivated = pyqtSignal(object)
    nodeActionRequested = pyqtSignal(object, str)
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
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
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
            blackbox = PipelineNode(
                f"{lane.lane_id}:blackbox",
                "Чёрный ящик",
                "blackbox",
                "Нажмите, чтобы раскрыть",
            )
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
                belongs_to_expanded_box = is_expanded and node in internal_nodes
                item = PipelineNodeItem(
                    node,
                    activate=lambda value: self.nodeActivated.emit(value),
                    request_action=lambda value, action: self.nodeActionRequested.emit(value, action),
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
            y += 135
        self.graph_scene.setSceneRect(self.graph_scene.itemsBoundingRect().adjusted(-80, -80, 120, 120))

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

    def wheelEvent(self, event) -> None:  # type: ignore[override]
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
            self.scale(factor, factor)
            event.accept()
            return
        super().wheelEvent(event)

if TYPE_CHECKING:
    from kraken_manager.presentation.qt.layer_management_parts.format_scalar import LayerPipelineSnapshot
