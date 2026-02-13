from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter
from PyQt6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QGraphicsScene,
    QGraphicsView,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from logic_analyzer.application.ports import SceneData
from logic_analyzer.application.use_cases import LoadSceneData
from logic_analyzer.presentation.qt.scene_renderer import EdifSceneRenderer


class EdifViewerWindow(QMainWindow):
    def __init__(self, load_scene_data_use_case: LoadSceneData, initial_path: str | Path | None = None):
        super().__init__()
        self._load_scene_data = load_scene_data_use_case
        self._current_path = Path(initial_path) if initial_path else None
        self._scene_data: SceneData | None = None
        self._filtered_net_indexes: list[int] = []
        self._filtered_instance_indexes: list[int] = []
        self._instance_to_connections: dict[str, list[tuple[str, str]]] = {}

        self.setWindowTitle("EDF Netlist Viewer")
        self.resize(1300, 820)
        self._init_ui()
        if self._current_path:
            self.load_edf(self._current_path)

    def _init_ui(self) -> None:
        root = QWidget(self)
        root_layout = QVBoxLayout(root)

        toolbar = QHBoxLayout()
        self.path_label = QLabel("No EDF file loaded")
        self.path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        open_btn = QPushButton("Open EDF")
        open_btn.clicked.connect(self.on_open_file)
        toolbar.addWidget(open_btn)
        toolbar.addWidget(self.path_label, 1)
        root_layout.addLayout(toolbar)

        self.summary_group = QGroupBox("Top-Level Summary")
        summary_layout = QFormLayout(self.summary_group)
        self.design_value = QLabel("-")
        self.library_value = QLabel("-")
        self.cell_value = QLabel("-")
        self.view_value = QLabel("-")
        self.instance_count_value = QLabel("0")
        self.net_count_value = QLabel("0")
        summary_layout.addRow("Design:", self.design_value)
        summary_layout.addRow("Library:", self.library_value)
        summary_layout.addRow("Cell:", self.cell_value)
        summary_layout.addRow("View:", self.view_value)
        summary_layout.addRow("Instances:", self.instance_count_value)
        summary_layout.addRow("Nets:", self.net_count_value)
        root_layout.addWidget(self.summary_group)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_nets_tab(), "Nets")
        self.tabs.addTab(self._build_instances_tab(), "Instances")
        self.tabs.addTab(self._build_scene_tab(), "Scene")
        root_layout.addWidget(self.tabs, 1)

        self.setCentralWidget(root)

    def _build_nets_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.net_search = QLineEdit()
        self.net_search.setPlaceholderText("Filter nets by name...")
        self.net_search.textChanged.connect(self.refresh_nets_table)
        layout.addWidget(self.net_search)

        split = QSplitter(Qt.Orientation.Horizontal)
        self.nets_table = QTableWidget(0, 2)
        self.nets_table.setHorizontalHeaderLabels(["Net", "Connections"])
        self.nets_table.horizontalHeader().setStretchLastSection(True)
        self.nets_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.nets_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.nets_table.itemSelectionChanged.connect(self.on_net_selected)
        split.addWidget(self.nets_table)

        self.net_connections_table = QTableWidget(0, 2)
        self.net_connections_table.setHorizontalHeaderLabels(["Port", "Instance"])
        self.net_connections_table.horizontalHeader().setStretchLastSection(True)
        self.net_connections_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        split.addWidget(self.net_connections_table)

        split.setSizes([700, 500])
        layout.addWidget(split, 1)
        return tab

    def _build_instances_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.instance_search = QLineEdit()
        self.instance_search.setPlaceholderText("Filter instances by name/cell/view/designator...")
        self.instance_search.textChanged.connect(self.refresh_instances_table)
        layout.addWidget(self.instance_search)

        split = QSplitter(Qt.Orientation.Horizontal)
        self.instances_table = QTableWidget(0, 4)
        self.instances_table.setHorizontalHeaderLabels(["Instance", "Cell", "View", "Designator"])
        self.instances_table.horizontalHeader().setStretchLastSection(True)
        self.instances_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.instances_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.instances_table.itemSelectionChanged.connect(self.on_instance_selected)
        split.addWidget(self.instances_table)

        self.instance_connections_table = QTableWidget(0, 2)
        self.instance_connections_table.setHorizontalHeaderLabels(["Net", "Port"])
        self.instance_connections_table.horizontalHeader().setStretchLastSection(True)
        self.instance_connections_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        split.addWidget(self.instance_connections_table)
        split.setSizes([700, 500])
        layout.addWidget(split, 1)
        return tab

    def _build_scene_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.scene_info = QLabel("Load EDF file to render schematic scene.")
        layout.addWidget(self.scene_info)
        self.graphics_scene = QGraphicsScene(self)
        self.graphics_view = QGraphicsView(self.graphics_scene)
        self.graphics_view.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.graphics_view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        layout.addWidget(self.graphics_view, 1)
        self.scene_renderer = EdifSceneRenderer(self.graphics_scene)
        return tab

    def on_open_file(self) -> None:
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Open EDF File",
            str(self._current_path.parent if self._current_path else Path.cwd()),
            "EDIF Files (*.edf *.EDF);;All Files (*.*)",
        )
        if file_name:
            self.load_edf(Path(file_name))

    def load_edf(self, path: Path) -> None:
        try:
            self._scene_data = self._load_scene_data.execute(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Parse Error", f"Failed to parse EDF:\n{exc}")
            return

        self._current_path = path
        self.path_label.setText(str(path.resolve()))
        self._rebuild_instance_connection_index()
        self._update_summary()
        self.refresh_nets_table()
        self.refresh_instances_table()
        self.refresh_graphics_scene()

    def _update_summary(self) -> None:
        if self._scene_data is None:
            return
        netlist = self._scene_data.netlist
        self.design_value.setText(netlist.design)
        self.library_value.setText(netlist.library)
        self.cell_value.setText(netlist.cell)
        self.view_value.setText(netlist.view)
        self.instance_count_value.setText(str(len(netlist.instances)))
        self.net_count_value.setText(str(len(netlist.nets)))

    def _rebuild_instance_connection_index(self) -> None:
        self._instance_to_connections = {}
        if self._scene_data is None:
            return
        for net in self._scene_data.netlist.nets:
            for connection in net.connections:
                if connection.instance is None:
                    continue
                self._instance_to_connections.setdefault(connection.instance, []).append((net.name, connection.port))
        for key in self._instance_to_connections:
            self._instance_to_connections[key].sort(key=lambda row: (row[0], row[1]))

    def refresh_nets_table(self) -> None:
        if self._scene_data is None:
            return
        filter_text = self.net_search.text().strip().lower()
        self._filtered_net_indexes = []
        self.nets_table.setRowCount(0)
        for idx, net in enumerate(self._scene_data.netlist.nets):
            if filter_text and filter_text not in net.name.lower():
                continue
            row = self.nets_table.rowCount()
            self.nets_table.insertRow(row)
            self.nets_table.setItem(row, 0, QTableWidgetItem(net.name))
            self.nets_table.setItem(row, 1, QTableWidgetItem(str(len(net.connections))))
            self._filtered_net_indexes.append(idx)
        self.net_connections_table.setRowCount(0)
        if self.nets_table.rowCount():
            self.nets_table.selectRow(0)

    def on_net_selected(self) -> None:
        if self._scene_data is None:
            return
        selected = self.nets_table.selectedItems()
        if not selected:
            self.net_connections_table.setRowCount(0)
            return
        row = selected[0].row()
        net = self._scene_data.netlist.nets[self._filtered_net_indexes[row]]
        self.net_connections_table.setRowCount(0)
        for connection in net.connections:
            table_row = self.net_connections_table.rowCount()
            self.net_connections_table.insertRow(table_row)
            self.net_connections_table.setItem(table_row, 0, QTableWidgetItem(connection.port))
            self.net_connections_table.setItem(table_row, 1, QTableWidgetItem(connection.instance or "<top-port>"))

    def refresh_instances_table(self) -> None:
        if self._scene_data is None:
            return
        filter_text = self.instance_search.text().strip().lower()
        self._filtered_instance_indexes = []
        self.instances_table.setRowCount(0)
        for idx, instance in enumerate(self._scene_data.netlist.instances):
            haystack = " ".join(
                [instance.name or "", instance.cell or "", instance.view or "", instance.designator or "", instance.library or ""]
            ).lower()
            if filter_text and filter_text not in haystack:
                continue
            row = self.instances_table.rowCount()
            self.instances_table.insertRow(row)
            self.instances_table.setItem(row, 0, QTableWidgetItem(instance.name))
            self.instances_table.setItem(row, 1, QTableWidgetItem(instance.cell or ""))
            self.instances_table.setItem(row, 2, QTableWidgetItem(instance.view or ""))
            self.instances_table.setItem(row, 3, QTableWidgetItem(instance.designator or ""))
            self._filtered_instance_indexes.append(idx)
        self.instance_connections_table.setRowCount(0)
        if self.instances_table.rowCount():
            self.instances_table.selectRow(0)

    def on_instance_selected(self) -> None:
        if self._scene_data is None:
            return
        selected = self.instances_table.selectedItems()
        if not selected:
            self.instance_connections_table.setRowCount(0)
            return
        row = selected[0].row()
        instance = self._scene_data.netlist.instances[self._filtered_instance_indexes[row]]
        connections = self._instance_to_connections.get(instance.name, [])
        self.instance_connections_table.setRowCount(0)
        for net_name, port_name in connections:
            table_row = self.instance_connections_table.rowCount()
            self.instance_connections_table.insertRow(table_row)
            self.instance_connections_table.setItem(table_row, 0, QTableWidgetItem(net_name))
            self.instance_connections_table.setItem(table_row, 1, QTableWidgetItem(port_name))

    def refresh_graphics_scene(self) -> None:
        if self._scene_data is None:
            return
        stats = self.scene_renderer.render(self._scene_data)
        bounds = self.graphics_scene.itemsBoundingRect()
        if not bounds.isNull():
            self.graphics_scene.setSceneRect(bounds.adjusted(-40, -40, 40, 40))
            self.graphics_view.fitInView(self.graphics_scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self.scene_info.setText(
            f"Rendered primitives: {stats.primitive_count}, wire segments: {stats.wire_segment_count}, text labels: {stats.text_count}"
        )
