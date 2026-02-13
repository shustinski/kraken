from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import (
    QApplication,
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

from model.edif import EdifParser, TopLevelNetlist


class EdifViewerWindow(QMainWindow):
    def __init__(self, initial_path: str | Path | None = None):
        super().__init__()
        self.setWindowTitle("EDF Netlist Viewer")
        self.resize(1300, 820)

        self.current_path: Path | None = Path(initial_path) if initial_path else None
        self.netlist: TopLevelNetlist | None = None
        self.filtered_net_indexes: list[int] = []
        self.filtered_instance_indexes: list[int] = []
        self.instance_to_connections: dict[str, list[tuple[str, str]]] = {}

        self._init_ui()
        if self.current_path is not None:
            self.load_edf(self.current_path)

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

        self.scene_info = QLabel("Load EDF file to render instance and wire geometry.")
        layout.addWidget(self.scene_info)

        self.graphics_scene = QGraphicsScene(self)
        self.graphics_view = QGraphicsView(self.graphics_scene)
        self.graphics_view.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.graphics_view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        layout.addWidget(self.graphics_view, 1)
        return tab

    def on_open_file(self) -> None:
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Open EDF File",
            str(self.current_path.parent if self.current_path else Path.cwd()),
            "EDIF Files (*.edf *.EDF);;All Files (*.*)",
        )
        if not file_name:
            return
        self.load_edf(Path(file_name))

    def load_edf(self, path: Path) -> None:
        try:
            parser = EdifParser(path)
            self.netlist = parser.extract_top_level_netlist()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Parse Error", f"Failed to parse EDF:\n{exc}")
            return

        self.current_path = path
        self.path_label.setText(str(path.resolve()))
        self._rebuild_indexes()
        self._update_summary()
        self.refresh_nets_table()
        self.refresh_instances_table()
        self.refresh_graphics_scene()

    def _update_summary(self) -> None:
        if self.netlist is None:
            return
        self.design_value.setText(self.netlist.design)
        self.library_value.setText(self.netlist.library)
        self.cell_value.setText(self.netlist.cell)
        self.view_value.setText(self.netlist.view)
        self.instance_count_value.setText(str(len(self.netlist.instances)))
        self.net_count_value.setText(str(len(self.netlist.nets)))

    def _rebuild_indexes(self) -> None:
        if self.netlist is None:
            self.instance_to_connections = {}
            return
        mapping: dict[str, list[tuple[str, str]]] = {}
        for net in self.netlist.nets:
            for conn in net.connections:
                if conn.instance is None:
                    continue
                mapping.setdefault(conn.instance, []).append((net.name, conn.port))
        for key in mapping:
            mapping[key].sort(key=lambda row: (row[0], row[1]))
        self.instance_to_connections = mapping

    def refresh_nets_table(self) -> None:
        if self.netlist is None:
            return
        filter_text = self.net_search.text().strip().lower()
        self.filtered_net_indexes = []
        self.nets_table.setRowCount(0)

        for idx, net in enumerate(self.netlist.nets):
            if filter_text and filter_text not in net.name.lower():
                continue
            row = self.nets_table.rowCount()
            self.nets_table.insertRow(row)
            self.nets_table.setItem(row, 0, QTableWidgetItem(net.name))
            self.nets_table.setItem(row, 1, QTableWidgetItem(str(len(net.connections))))
            self.filtered_net_indexes.append(idx)

        self.net_connections_table.setRowCount(0)
        if self.nets_table.rowCount() > 0:
            self.nets_table.selectRow(0)

    def on_net_selected(self) -> None:
        if self.netlist is None:
            return
        selected = self.nets_table.selectedItems()
        if not selected:
            self.net_connections_table.setRowCount(0)
            return
        row = selected[0].row()
        net_idx = self.filtered_net_indexes[row]
        net = self.netlist.nets[net_idx]

        self.net_connections_table.setRowCount(0)
        for conn in net.connections:
            table_row = self.net_connections_table.rowCount()
            self.net_connections_table.insertRow(table_row)
            self.net_connections_table.setItem(table_row, 0, QTableWidgetItem(conn.port))
            self.net_connections_table.setItem(
                table_row, 1, QTableWidgetItem(conn.instance if conn.instance else "<top-port>")
            )

    def refresh_instances_table(self) -> None:
        if self.netlist is None:
            return

        filter_text = self.instance_search.text().strip().lower()
        self.filtered_instance_indexes = []
        self.instances_table.setRowCount(0)

        for idx, inst in enumerate(self.netlist.instances):
            haystack = " ".join(
                [
                    inst.name or "",
                    inst.cell or "",
                    inst.view or "",
                    inst.designator or "",
                    inst.library or "",
                ]
            ).lower()
            if filter_text and filter_text not in haystack:
                continue
            row = self.instances_table.rowCount()
            self.instances_table.insertRow(row)
            self.instances_table.setItem(row, 0, QTableWidgetItem(inst.name))
            self.instances_table.setItem(row, 1, QTableWidgetItem(inst.cell or ""))
            self.instances_table.setItem(row, 2, QTableWidgetItem(inst.view or ""))
            self.instances_table.setItem(row, 3, QTableWidgetItem(inst.designator or ""))
            self.filtered_instance_indexes.append(idx)

        self.instance_connections_table.setRowCount(0)
        if self.instances_table.rowCount() > 0:
            self.instances_table.selectRow(0)

    def on_instance_selected(self) -> None:
        if self.netlist is None:
            return
        selected = self.instances_table.selectedItems()
        if not selected:
            self.instance_connections_table.setRowCount(0)
            return
        row = selected[0].row()
        instance_idx = self.filtered_instance_indexes[row]
        instance = self.netlist.instances[instance_idx]
        connections = self.instance_to_connections.get(instance.name, [])

        self.instance_connections_table.setRowCount(0)
        for net_name, port_name in connections:
            table_row = self.instance_connections_table.rowCount()
            self.instance_connections_table.insertRow(table_row)
            self.instance_connections_table.setItem(table_row, 0, QTableWidgetItem(net_name))
            self.instance_connections_table.setItem(table_row, 1, QTableWidgetItem(port_name))

    def refresh_graphics_scene(self) -> None:
        self.graphics_scene.clear()
        if self.netlist is None:
            return

        wire_pen = QPen(QColor("#3b82f6"))
        wire_pen.setWidthF(1.2)
        wire_pen.setCosmetic(True)

        point_pen = QPen(QColor("#1d4ed8"))
        point_pen.setCosmetic(True)

        instance_pen = QPen(QColor("#374151"))
        instance_pen.setWidthF(1.3)
        instance_pen.setCosmetic(True)

        scale = 0.12
        instance_w = 68.0
        instance_h = 28.0
        wire_count = 0
        instance_count = 0

        for net in self.netlist.nets:
            for wire in net.wires:
                if len(wire.points) == 1:
                    p = wire.points[0]
                    x, y = p.x * scale, -p.y * scale
                    self.graphics_scene.addEllipse(x - 1.8, y - 1.8, 3.6, 3.6, point_pen)
                    wire_count += 1
                    continue
                for a, b in zip(wire.points, wire.points[1:]):
                    x1, y1 = a.x * scale, -a.y * scale
                    x2, y2 = b.x * scale, -b.y * scale
                    self.graphics_scene.addLine(x1, y1, x2, y2, wire_pen)
                    wire_count += 1

        for inst in self.netlist.instances:
            if inst.x is None or inst.y is None:
                continue
            center_x = inst.x * scale
            center_y = -inst.y * scale
            rect = self.graphics_scene.addRect(
                center_x - instance_w / 2.0,
                center_y - instance_h / 2.0,
                instance_w,
                instance_h,
                instance_pen,
            )
            rect.setToolTip(f"{inst.name}\n{inst.cell or ''}\n({inst.x}, {inst.y})")
            text = self.graphics_scene.addSimpleText(inst.name)
            text.setPos(center_x - instance_w / 2.0 + 3.0, center_y - instance_h / 2.0 + 4.0)
            instance_count += 1

        bounds = self.graphics_scene.itemsBoundingRect()
        if not bounds.isNull():
            self.graphics_scene.setSceneRect(bounds.adjusted(-40, -40, 40, 40))
            self.graphics_view.fitInView(self.graphics_scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

        self.scene_info.setText(
            f"Rendered instances: {instance_count}, wire segments/points: {wire_count}"
        )


def parse_to_dict(path: str | Path) -> dict:
    parser = EdifParser(path)
    netlist = parser.extract_top_level_netlist()
    return asdict(netlist)


def run_gui(initial_path: str | Path | None = None) -> int:
    app = QApplication(sys.argv)
    window = EdifViewerWindow(initial_path=initial_path)
    window.show()
    return app.exec()
