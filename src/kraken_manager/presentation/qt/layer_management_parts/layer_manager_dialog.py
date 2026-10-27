"""layer_management: layer manager dialog."""
from __future__ import annotations
from typing import TYPE_CHECKING
from typing import Callable
from PyQt6.QtCore import QSettings, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from kraken_manager.presentation.qt.models import LayerListItem

from kraken_manager.presentation.qt.layer_management_parts.layer_order_list import LayerOrderList

from kraken_manager.presentation.qt.layer_management_parts.pipeline_graph_view import PipelineGraphView

class LayerManagerDialog(QDialog):
    addLayerRequested = pyqtSignal()
    layerSelected = pyqtSignal(str)
    orderChanged = pyqtSignal(object)
    representationActivated = pyqtSignal(str)
    nodeActionRequested = pyqtSignal(str, object, str)
    layerActionRequested = pyqtSignal(str, str)
    nodePropertiesRequested = pyqtSignal(str, object)
    layerPropertiesRequested = pyqtSignal(str)

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
        self.workflow_button = QPushButton("Технологический процесс… / Workflow…", layer_panel)
        self.workflow_button.setObjectName("layerWorkflow")
        layer_layout.addWidget(self.workflow_button)
        def open_workflow():
            item = self.layer_list.currentItem()
            if item is not None:
                self.layerActionRequested.emit(str(item.data(Qt.ItemDataRole.UserRole)), "edit_workflow")
        self.workflow_button.clicked.connect(open_workflow)
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
        self.layer_list.layerPropertiesRequested.connect(self.layerPropertiesRequested)
        self.add_layer_button.clicked.connect(self.addLayerRequested)
        self.graph.nodeActivated.connect(self._node_activated)
        self.graph.nodeActionRequested.connect(self._node_action)
        self.graph.nodePropertiesRequested.connect(self._node_properties)

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

    def _node_properties(self, node: PipelineNode) -> None:
        snapshot = self.graph._snapshot
        if snapshot is not None:
            self.nodePropertiesRequested.emit(snapshot.layer_id, node)

    def hideEvent(self, event) -> None:  # type: ignore[override]
        self.graph.save_layout()
        super().hideEvent(event)

if TYPE_CHECKING:
    from kraken_manager.presentation.qt.layer_management_parts.format_scalar import LayerPipelineSnapshot, PipelineNode
