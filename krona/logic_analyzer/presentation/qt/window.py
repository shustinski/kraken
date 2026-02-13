from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
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
from logic_analyzer.application.logic_functions import ExtractLogicFunctions
from logic_analyzer.application.use_cases import LoadSceneData
from logic_analyzer.presentation.qt.scene_renderer import EdifSceneRenderer
from logic_analyzer.presentation.qt.theme.manager import ThemeManager


class EdifViewerWindow(QMainWindow):
    def __init__(
        self,
        load_scene_data_use_case: LoadSceneData,
        extract_logic_functions_use_case: ExtractLogicFunctions,
        initial_path: str | Path | None = None,
        theme_manager: ThemeManager | None = None,
        initial_theme: str = "Dark",
    ):
        super().__init__()
        self._load_scene_data = load_scene_data_use_case
        self._extract_logic_functions = extract_logic_functions_use_case
        self._current_path = Path(initial_path) if initial_path else None
        self._theme_manager = theme_manager
        self._current_theme = initial_theme
        self._scene_data: SceneData | None = None
        self._logic_data: dict | None = None
        self._filtered_net_indexes: list[int] = []
        self._filtered_instance_indexes: list[int] = []
        self._instance_to_connections: dict[str, list[tuple[str, str]]] = {}

        self.setWindowTitle("EDF Netlist Viewer")
        self.resize(1300, 820)
        self.setAcceptDrops(True)
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
        zoom_in_btn = QPushButton("Zoom In")
        zoom_in_btn.clicked.connect(lambda: self._apply_zoom(1.2))
        zoom_out_btn = QPushButton("Zoom Out")
        zoom_out_btn.clicked.connect(lambda: self._apply_zoom(1 / 1.2))
        zoom_reset_btn = QPushButton("Reset Zoom")
        zoom_reset_btn.clicked.connect(self._fit_scene_in_view)
        self.hitbox_toggle_btn = QPushButton("Show Hitboxes")
        self.hitbox_toggle_btn.setCheckable(True)
        self.hitbox_toggle_btn.setChecked(False)
        self.hitbox_toggle_btn.toggled.connect(self._on_hitbox_toggle_changed)
        theme_label = QLabel("Theme:")
        self.theme_combo = QComboBox()
        theme_items = self._theme_manager.available_themes() if self._theme_manager else ["Dark"]
        self.theme_combo.addItems(theme_items)
        if self._current_theme in theme_items:
            self.theme_combo.setCurrentText(self._current_theme)
        self.theme_combo.currentTextChanged.connect(self._on_theme_changed)
        toolbar.addWidget(open_btn)
        toolbar.addWidget(zoom_in_btn)
        toolbar.addWidget(zoom_out_btn)
        toolbar.addWidget(zoom_reset_btn)
        toolbar.addWidget(self.hitbox_toggle_btn)
        toolbar.addWidget(theme_label)
        toolbar.addWidget(self.theme_combo)
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
        self.tabs.addTab(self._build_logic_tab(), "Logic")
        self.tabs.addTab(self._build_diagnostics_tab(), "Diagnostics")
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
        self.graphics_view.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self._default_graphics_wheel_event = self.graphics_view.wheelEvent
        self._default_graphics_mouse_press_event = self.graphics_view.mousePressEvent
        self.graphics_view.wheelEvent = self._graphics_wheel_event
        self.graphics_view.mousePressEvent = self._graphics_mouse_press_event
        layout.addWidget(self.graphics_view, 1)
        self.scene_renderer = EdifSceneRenderer(self.graphics_scene)
        self.scene_renderer.set_dark_mode(self._current_theme.lower() == "dark")
        return tab

    def _build_diagnostics_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.diagnostics_status_label = QLabel("Load EDF file to view parser diagnostics.")
        self.diagnostics_status_label.setWordWrap(True)
        layout.addWidget(self.diagnostics_status_label)
        self.diagnostics_table = QTableWidget(0, 2)
        self.diagnostics_table.setHorizontalHeaderLabels(["Severity", "Message"])
        self.diagnostics_table.horizontalHeader().setStretchLastSection(True)
        self.diagnostics_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.diagnostics_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        layout.addWidget(self.diagnostics_table, 1)
        return tab

    def _build_logic_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.logic_status_label = QLabel("Load EDF file to extract logic functions.")
        self.logic_status_label.setWordWrap(True)
        layout.addWidget(self.logic_status_label)

        highlight_row = QHBoxLayout()
        highlight_row.addWidget(QLabel("Highlight output:"))
        self.logic_highlight_combo = QComboBox()
        self.logic_highlight_combo.currentTextChanged.connect(self._on_logic_highlight_output_changed)
        highlight_row.addWidget(self.logic_highlight_combo, 1)
        clear_highlight_btn = QPushButton("Clear Highlight")
        clear_highlight_btn.clicked.connect(self._clear_logic_highlight)
        highlight_row.addWidget(clear_highlight_btn)
        layout.addLayout(highlight_row)

        self.logic_simplified_label = QLabel("Simplified: -")
        self.logic_simplified_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.logic_simplified_label.setWordWrap(True)
        layout.addWidget(self.logic_simplified_label)

        self.logic_sop_label = QLabel("SOP: -")
        self.logic_sop_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.logic_sop_label.setWordWrap(True)
        layout.addWidget(self.logic_sop_label)

        self.logic_truth_table = QTableWidget(0, 0)
        self.logic_truth_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.logic_truth_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.logic_truth_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.logic_truth_table, 1)
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

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        mime_data = event.mimeData()
        if not mime_data.hasUrls():
            event.ignore()
            return
        for url in mime_data.urls():
            if not url.isLocalFile():
                continue
            path = Path(url.toLocalFile())
            if path.suffix.lower() == ".edf":
                event.acceptProposedAction()
                return
        event.ignore()

    def dragMoveEvent(self, event) -> None:  # noqa: N802
        self.dragEnterEvent(event)

    def dropEvent(self, event) -> None:  # noqa: N802
        mime_data = event.mimeData()
        if not mime_data.hasUrls():
            event.ignore()
            return
        for url in mime_data.urls():
            if not url.isLocalFile():
                continue
            path = Path(url.toLocalFile())
            if path.suffix.lower() == ".edf":
                self.load_edf(path)
                event.acceptProposedAction()
                return
        event.ignore()

    def load_edf(self, path: Path) -> None:
        try:
            self._scene_data = self._load_scene_data.execute(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Parse Error", f"Failed to parse EDF:\n{exc}")
            return
        try:
            self._logic_data = self._extract_logic_functions.execute(path)
        except Exception as exc:  # noqa: BLE001
            self._logic_data = None
            QMessageBox.warning(self, "Logic Extraction", f"Failed to extract logic functions:\n{exc}")

        self._current_path = path
        self.path_label.setText(str(path.resolve()))
        self._rebuild_instance_connection_index()
        self._update_summary()
        self.refresh_nets_table()
        self.refresh_instances_table()
        self.refresh_logic_tab()
        self.refresh_diagnostics_tab()
        self.refresh_graphics_scene(reset_view=True)

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

    def refresh_graphics_scene(self, reset_view: bool = False) -> None:
        if self._scene_data is None:
            return
        stats = self.scene_renderer.render(self._scene_data)
        bounds = self.graphics_scene.itemsBoundingRect()
        if not bounds.isNull():
            self.graphics_scene.setSceneRect(bounds.adjusted(-40, -40, 40, 40))
            if reset_view:
                self._fit_scene_in_view()
        self.scene_info.setText(
            f"Rendered primitives: {stats.primitive_count}, wire segments: {stats.wire_segment_count}, text labels: {stats.text_count}"
        )

    def refresh_logic_tab(self) -> None:
        self.logic_highlight_combo.blockSignals(True)
        self.logic_highlight_combo.clear()
        self.logic_truth_table.setRowCount(0)
        self.logic_truth_table.setColumnCount(0)
        if not self._logic_data:
            self.logic_status_label.setText("Logic function data unavailable for current file.")
            self.logic_simplified_label.setText("Simplified: -")
            self.logic_sop_label.setText("SOP: -")
            self.logic_highlight_combo.blockSignals(False)
            self._clear_logic_highlight()
            return

        outputs = self._logic_data.get("outputs", {})
        if not outputs:
            self.logic_status_label.setText("No outputs detected for current file.")
            self.logic_simplified_label.setText("Simplified: -")
            self.logic_sop_label.setText("SOP: -")
            self.logic_highlight_combo.blockSignals(False)
            self._clear_logic_highlight()
            return

        input_count = len(self._logic_data.get("inputs", []))
        self.logic_status_label.setText(
            f"Inputs: {input_count}, Outputs: {len(outputs)}, "
            f"Transistors: {self._logic_data.get('meta', {}).get('transistor_count', 0)}"
        )
        input_names = self._logic_data.get("inputs", [])
        output_names = list(outputs.keys())
        for output_name in output_names:
            self.logic_highlight_combo.addItem(output_name)
        self.logic_highlight_combo.blockSignals(False)
        if self.logic_highlight_combo.count():
            self.logic_highlight_combo.setCurrentIndex(0)
        reference_truth_table = outputs[output_names[0]].get("truth_table", [])
        self.logic_truth_table.setColumnCount(len(input_names) + len(output_names))
        self.logic_truth_table.setHorizontalHeaderLabels([*input_names, *output_names])
        self.logic_truth_table.setRowCount(0)

        for row_idx, row_data in enumerate(reference_truth_table):
            row = self.logic_truth_table.rowCount()
            self.logic_truth_table.insertRow(row)
            for col, input_name in enumerate(input_names):
                value = row_data.get("inputs", {}).get(input_name)
                self.logic_truth_table.setItem(row, col, QTableWidgetItem("1" if value == 1 else "0"))
            for out_col, output_name in enumerate(output_names):
                output_rows = outputs.get(output_name, {}).get("truth_table", [])
                out_value = output_rows[row_idx].get("value") if row_idx < len(output_rows) else None
                rendered = "X" if out_value is None else str(out_value)
                self.logic_truth_table.setItem(row, len(input_names) + out_col, QTableWidgetItem(rendered))

        expr_parts = []
        sop_parts = []
        for name in output_names:
            output_data = outputs.get(name, {})
            simplified = output_data.get("simplified_expression")
            sop = output_data.get("sum_of_products")
            if simplified:
                expr_parts.append(f"{name}: {simplified}")
            if sop:
                sop_parts.append(f"{name}: {sop}")
        self.logic_simplified_label.setText("Simplified: " + (" | ".join(expr_parts) if expr_parts else "-"))
        self.logic_sop_label.setText("SOP: " + (" | ".join(sop_parts) if sop_parts else "-"))
        if self.logic_highlight_combo.currentText():
            self._on_logic_highlight_output_changed(self.logic_highlight_combo.currentText())

    def refresh_diagnostics_tab(self) -> None:
        self.diagnostics_table.setRowCount(0)
        if self._scene_data is None:
            self.diagnostics_status_label.setText("Load EDF file to view parser diagnostics.")
            return
        diagnostics = list(self._scene_data.diagnostics)
        if not diagnostics:
            self.diagnostics_status_label.setText("No parser diagnostics.")
            return
        severity_counts = {"error": 0, "warning": 0, "info": 0}
        for item in diagnostics:
            severity_counts[item.severity] = severity_counts.get(item.severity, 0) + 1
        self.diagnostics_status_label.setText(
            f"Parser diagnostics: {len(diagnostics)} total "
            f"(errors: {severity_counts.get('error', 0)}, "
            f"warnings: {severity_counts.get('warning', 0)}, "
            f"info: {severity_counts.get('info', 0)})."
        )
        severity_colors = {
            "error": QColor(180, 40, 40),
            "warning": QColor(160, 110, 0),
            "info": QColor(40, 70, 150),
        }
        for item in diagnostics:
            row = self.diagnostics_table.rowCount()
            self.diagnostics_table.insertRow(row)
            severity_item = QTableWidgetItem(item.severity.upper())
            color = severity_colors.get(item.severity)
            if color is not None:
                severity_item.setForeground(color)
            self.diagnostics_table.setItem(row, 0, severity_item)
            self.diagnostics_table.setItem(row, 1, QTableWidgetItem(item.message))

    def _fit_scene_in_view(self) -> None:
        if self.graphics_scene.sceneRect().isNull():
            return
        self.graphics_view.resetTransform()
        self.graphics_view.fitInView(self.graphics_scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def _apply_zoom(self, factor: float) -> None:
        if factor <= 0:
            return
        self.graphics_view.scale(factor, factor)

    def _graphics_wheel_event(self, event) -> None:
        delta = event.angleDelta().y()
        if delta > 0:
            self._apply_zoom(1.15)
        elif delta < 0:
            self._apply_zoom(1 / 1.15)
        event.accept()

    def _graphics_mouse_press_event(self, event) -> None:
        try:
            view_pos = event.position().toPoint()
        except AttributeError:
            view_pos = event.pos()
        for item in self.graphics_view.items(view_pos):
            instance_name = item.data(0)
            if isinstance(instance_name, str) and instance_name:
                self._highlight_instance_connected_nets(instance_name)
                break
        self._default_graphics_mouse_press_event(event)

    def _on_logic_highlight_output_changed(self, output_name: str) -> None:
        if not output_name or not self._logic_data:
            self._clear_logic_highlight()
            return
        outputs = self._logic_data.get("outputs", {})
        output_data = outputs.get(output_name, {})
        logic_path = output_data.get("logic_path", {})
        nets = set(logic_path.get("nets", []))
        instances = set(logic_path.get("instances", []))
        self.scene_renderer.set_logic_highlight(nets=nets, instances=instances)
        self.refresh_graphics_scene(reset_view=False)

    def _clear_logic_highlight(self) -> None:
        self.scene_renderer.set_logic_highlight(nets=set(), instances=set())
        if self._scene_data is not None:
            self.refresh_graphics_scene(reset_view=False)

    def _highlight_instance_connected_nets(self, instance_name: str) -> None:
        if self._scene_data is None:
            return
        connections = self._instance_to_connections.get(instance_name, [])
        if not connections:
            self.scene_renderer.set_logic_highlight(nets=set(), instances={instance_name})
            self.refresh_graphics_scene(reset_view=False)
            return
        direct_nets = {net_name for net_name, _ in connections}
        net_names = self._expand_connected_nets(direct_nets)
        self.scene_renderer.set_logic_highlight(nets=net_names, instances={instance_name})
        self.refresh_graphics_scene(reset_view=False)

    def _expand_connected_nets(self, seed_nets: set[str]) -> set[str]:
        if self._scene_data is None or not seed_nets:
            return set(seed_nets)

        point_to_nets: dict[tuple[int, int], set[str]] = {}
        net_to_points: dict[str, set[tuple[int, int]]] = {}
        for net in self._scene_data.netlist.nets:
            for wire in net.wires:
                for point in wire.points:
                    key = (point.x, point.y)
                    point_to_nets.setdefault(key, set()).add(net.name)
                    net_to_points.setdefault(net.name, set()).add(key)

        seen = set(seed_nets)
        queue = list(seed_nets)
        while queue:
            current = queue.pop()
            # Walk all points belonging to current net, then include all nets touching those points.
            for point in net_to_points.get(current, set()):
                nets_at_point = point_to_nets.get(point, set())
                for candidate in nets_at_point:
                    if candidate in seen:
                        continue
                    seen.add(candidate)
                    queue.append(candidate)
        return seen

    def _on_theme_changed(self, theme_name: str) -> None:
        if not theme_name or self._theme_manager is None:
            return
        qapp = QApplication.instance()
        if qapp is None:
            return
        if not self._theme_manager.apply_theme(qapp, theme_name):
            return
        self._theme_manager.save_theme(theme_name)
        self._current_theme = theme_name
        self.scene_renderer.set_dark_mode(self._current_theme.lower() == "dark")
        if self._scene_data is not None:
            self.refresh_graphics_scene(reset_view=False)

    def _on_hitbox_toggle_changed(self, enabled: bool) -> None:
        self.scene_renderer.set_show_hitboxes(enabled)
        if self._scene_data is not None:
            self.refresh_graphics_scene(reset_view=False)
