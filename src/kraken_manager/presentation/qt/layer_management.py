"""Modeless layer manager and provenance graph for a project workspace."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from PyQt6.QtCore import QLineF, QPointF, QRectF, QSettings, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import (
    QDialog,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSceneMouseEvent,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .models import LayerListItem


@dataclass(frozen=True, slots=True)
class PipelineNode:
    node_id: str
    title: str
    kind: str
    subtitle: str = ""
    representation_id: str = ""
    active: bool = False
    state: str = ""
    details: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PipelineLane:
    lane_id: str
    title: str
    nodes: tuple[PipelineNode, ...]
    edges: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class LayerPipelineSnapshot:
    project_id: str
    layer_id: str
    lanes: tuple[PipelineLane, ...] = ()


class LayerOrderList(QListWidget):
    orderChanged = pyqtSignal(object)
    layerSelected = pyqtSignal(str)
    layerActionRequested = pyqtSignal(str, str)

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
        menu.addSeparator()
        delete_layer = menu.addAction("Удалить слой…")
        by_action = {
            add_images: "add_image_representation",
            karakal: "karakal",
            delete_layer: "delete_layer",
        }
        for menu_action, code in by_action.items():
            enabled, reason = self._action_availability(code)
            menu_action.setEnabled(enabled)
            if reason:
                menu_action.setToolTip(reason)
                menu_action.setStatusTip(reason)
        selected = menu.exec(self.viewport().mapToGlobal(position))
        if selected in by_action:
            self.layerActionRequested.emit(layer_id, by_action[selected])


class PipelineScene(QGraphicsScene):
    def drawBackground(self, painter: QPainter, rect: QRectF) -> None:  # type: ignore[override]
        painter.fillRect(rect, QColor("#1b1d1b"))
        minor, major = 20, 100
        for spacing, color in ((minor, "#272a27"), (major, "#343834")):
            pen = QPen(QColor(color), 0)
            painter.setPen(pen)
            left = int(rect.left()) - (int(rect.left()) % spacing)
            top = int(rect.top()) - (int(rect.top()) % spacing)
            for x in range(left, int(rect.right()) + spacing, spacing):
                painter.drawLine(QLineF(float(x), rect.top(), float(x), rect.bottom()))
            for y in range(top, int(rect.bottom()) + spacing, spacing):
                painter.drawLine(QLineF(rect.left(), float(y), rect.right(), float(y)))


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
    }

    def __init__(
        self,
        node: PipelineNode,
        *,
        activate: Callable[[PipelineNode], None],
        request_action: Callable[[PipelineNode, str], None],
        expand: Callable[[], None],
        collapse: Callable[[], None] | None,
        action_availability: Callable[[str], tuple[bool, str]],
        position_changed: Callable[[], None],
    ) -> None:
        super().__init__(0, 0, 190, 72)
        self.node = node
        self._activate = activate
        self._request_action = request_action
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
        self.setToolTip("\n".join(f"{key}: {value}" for key, value in node.details.items()))
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
                ("Подготовить выборку в Contour", "prepare_dataset"),
                ("Распознать готовой моделью", "recognize_external"),
                ("Получить CIF в Contour", "vectorize"),
            ),
            "dataset": (("Обучить модель в NeuralImage", "train"),),
            "model": (("Распознать исходники", "recognize"),),
            "binary": (("Получить CIF в Contour", "vectorize"),),
            "vector": (("Добавить CIF из внешнего источника…", "add_external_vector"),),
            "missing": (("Добавить CIF из внешнего источника…", "add_external_vector"),),
        }.get(self.node.kind, ())
        if self.node.kind == "source":
            actions = (*actions, ("Удалить слой изображений из проекта", "archive_representation"))
        if bool(self.node.details.get("deletable", False)):
            actions = (*actions, ("Удалить шаг из pipeline", "delete_pipeline_step"))
        if self._collapse is not None:
            actions = (*actions, ("Свернуть", "collapse_pipeline"))
        if not actions:
            return
        menu = QMenu()
        by_action = {}
        for label, code in actions:
            menu_action = menu.addAction(label)
            enabled, reason = (
                (True, "")
                if code == "collapse_pipeline"
                else self._action_availability(code)
            )
            menu_action.setEnabled(enabled)
            if reason:
                menu_action.setToolTip(reason)
                menu_action.setStatusTip(reason)
            by_action[menu_action] = code
        selected = menu.exec(event.screenPos())
        if selected in by_action:
            code = by_action[selected]
            if code == "collapse_pipeline" and self._collapse is not None:
                self._collapse()
            else:
                self._request_action(self.node, code)


class PipelineGraphView(QGraphicsView):
    nodeActivated = pyqtSignal(object)
    nodeActionRequested = pyqtSignal(object, str)

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


class LayerManagerDialog(QDialog):
    addLayerRequested = pyqtSignal()
    layerSelected = pyqtSignal(str)
    orderChanged = pyqtSignal(object)
    representationActivated = pyqtSignal(str)
    nodeActionRequested = pyqtSignal(str, object, str)
    layerActionRequested = pyqtSignal(str, str)

    def __init__(
        self,
        project_id: str,
        parent: QWidget | None = None,
        *,
        action_availability: Callable[[str], tuple[bool, str]] | None = None,
    ) -> None:
        super().__init__(parent)
        self.project_id = str(project_id)
        self.setObjectName("layerManagerDialog")
        self.setWindowTitle("Управление слоями")
        self.setModal(False)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self.resize(1180, 720)
        self.settings = QSettings("Kraken", "KrakenHub")
        splitter = QSplitter(self)
        splitter.setOrientation(Qt.Orientation.Horizontal)
        layer_panel = QWidget(splitter)
        layer_layout = QVBoxLayout(layer_panel)
        layer_layout.setContentsMargins(0, 0, 0, 0)
        self.layer_list = LayerOrderList(action_availability=action_availability)
        self.add_layer_button = QPushButton("Добавить слой", layer_panel)
        self.add_layer_button.setObjectName("layerManagerAddLayer")
        self.add_layer_button.setToolTip("Создать слой из папки или подключить внешние каталоги")
        layer_layout.addWidget(self.layer_list, 1)
        layer_layout.addWidget(self.add_layer_button)
        self.graph = PipelineGraphView(self.settings, action_availability=action_availability)
        splitter.addWidget(layer_panel)
        splitter.addWidget(self.graph)
        splitter.setSizes([270, 910])
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(splitter)
        self.layer_list.layerSelected.connect(self.layerSelected)
        self.layer_list.orderChanged.connect(self.orderChanged)
        self.layer_list.layerActionRequested.connect(self.layerActionRequested)
        self.add_layer_button.clicked.connect(self.addLayerRequested)
        self.graph.nodeActivated.connect(self._node_activated)
        self.graph.nodeActionRequested.connect(self._node_action)

    def set_layers(self, layers: list[LayerListItem], selected_id: str = "") -> None:
        self.layer_list.set_layers(layers, selected_id)

    def select_layer(self, layer_id: str) -> None:
        self.layer_list.select_layer(layer_id)

    def set_pipeline(self, snapshot: LayerPipelineSnapshot) -> None:
        self.graph.set_snapshot(snapshot)

    def _node_activated(self, node: PipelineNode) -> None:
        if node.representation_id:
            self.representationActivated.emit(node.representation_id)

    def _node_action(self, node: PipelineNode, action: str) -> None:
        snapshot = self.graph._snapshot
        if snapshot is not None:
            self.nodeActionRequested.emit(snapshot.layer_id, node, action)

    def hideEvent(self, event) -> None:  # type: ignore[override]
        self.graph.save_layout()
        super().hideEvent(event)


__all__ = [
    "LayerManagerDialog",
    "LayerPipelineSnapshot",
    "PipelineLane",
    "PipelineNode",
]
