from __future__ import annotations

from pathlib import Path
import sys

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QIcon, QPainter
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
from logic_analyzer.presentation.qt.ui_strings import available_languages, load_ui_strings


def _resolve_window_icon_path() -> Path | None:
    candidates: list[Path] = []
    frozen_base = getattr(sys, "_MEIPASS", None)
    if frozen_base:
        candidates.append(Path(frozen_base) / "icons" / "app_icon.ico")
    candidates.append(Path(__file__).resolve().parents[3] / "icons" / "app_icon.ico")
    candidates.append(Path.cwd() / "icons" / "app_icon.ico")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


class EdifViewerWindow(QMainWindow):
    def __init__(
        self,
        load_scene_data_use_case: LoadSceneData,
        extract_logic_functions_use_case: ExtractLogicFunctions,
        initial_path: str | Path | None = None,
        theme_manager: ThemeManager | None = None,
        initial_theme: str = "Dark",
        initial_language: str = "English",
        ui_strings: dict[str, str] | None = None,
    ):
        super().__init__()
        self._load_scene_data = load_scene_data_use_case
        self._extract_logic_functions = extract_logic_functions_use_case
        self._current_path = Path(initial_path) if initial_path else None
        self._theme_manager = theme_manager
        self._current_theme = initial_theme
        self._current_language = initial_language
        self._scene_data: SceneData | None = None
        self._logic_data: dict | None = None
        self._logic_highlight_targets: dict[str, dict] = {}
        self._filtered_net_indexes: list[int] = []
        self._filtered_instance_indexes: list[int] = []
        self._instance_to_connections: dict[str, list[tuple[str, str]]] = {}
        self._ui = ui_strings if ui_strings is not None else load_ui_strings(self._current_language)

        self.setWindowTitle(self._t("window.title", "EDF Netlist Viewer"))
        icon_path = _resolve_window_icon_path()
        if icon_path is not None:
            self.setWindowIcon(QIcon(str(icon_path)))
        self.resize(1300, 820)
        self.setAcceptDrops(True)
        self._init_ui()
        if self._current_path:
            self.load_edf(self._current_path)

    def _t(self, key: str, default: str, **kwargs) -> str:
        template = self._ui.get(key, default)
        if kwargs:
            try:
                return template.format(**kwargs)
            except (KeyError, ValueError):
                return default.format(**kwargs)
        return template

    def _init_ui(self) -> None:
        root = QWidget(self)
        root_layout = QVBoxLayout(root)

        toolbar = QHBoxLayout()
        self.path_label = QLabel(self._t("path.no_file", "No EDF file loaded"))
        self.path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.open_btn = QPushButton(self._t("button.open_edf", "Open EDF"))
        self.open_btn.clicked.connect(self.on_open_file)
        self.zoom_in_btn = QPushButton(self._t("button.zoom_in", "Zoom In"))
        self.zoom_in_btn.clicked.connect(lambda: self._apply_zoom(1.2))
        self.zoom_out_btn = QPushButton(self._t("button.zoom_out", "Zoom Out"))
        self.zoom_out_btn.clicked.connect(lambda: self._apply_zoom(1 / 1.2))
        self.zoom_reset_btn = QPushButton(self._t("button.reset_zoom", "Reset Zoom"))
        self.zoom_reset_btn.clicked.connect(self._fit_scene_in_view)
        self.hitbox_toggle_btn = QPushButton(self._t("button.show_hitboxes", "Show Hitboxes"))
        self.hitbox_toggle_btn.setCheckable(True)
        self.hitbox_toggle_btn.setChecked(False)
        self.hitbox_toggle_btn.toggled.connect(self._on_hitbox_toggle_changed)
        self.language_label = QLabel(self._t("label.language", "Language:"))
        self.language_combo = QComboBox()
        self.language_combo.addItems(available_languages())
        if self._current_language in available_languages():
            self.language_combo.setCurrentText(self._current_language)
        self.language_combo.currentTextChanged.connect(self._on_language_changed)
        self.theme_label = QLabel(self._t("label.theme", "Theme:"))
        self.theme_combo = QComboBox()
        theme_items = self._theme_manager.available_themes() if self._theme_manager else ["Dark"]
        self.theme_combo.addItems(theme_items)
        if self._current_theme in theme_items:
            self.theme_combo.setCurrentText(self._current_theme)
        self.theme_combo.currentTextChanged.connect(self._on_theme_changed)
        toolbar.addWidget(self.open_btn)
        toolbar.addWidget(self.zoom_in_btn)
        toolbar.addWidget(self.zoom_out_btn)
        toolbar.addWidget(self.zoom_reset_btn)
        toolbar.addWidget(self.hitbox_toggle_btn)
        toolbar.addWidget(self.language_label)
        toolbar.addWidget(self.language_combo)
        toolbar.addWidget(self.theme_label)
        toolbar.addWidget(self.theme_combo)
        toolbar.addWidget(self.path_label, 1)
        root_layout.addLayout(toolbar)

        self.summary_group = QGroupBox(self._t("group.summary", "Top-Level Summary"))
        summary_layout = QFormLayout(self.summary_group)
        self.design_value = QLabel("-")
        self.library_value = QLabel("-")
        self.cell_value = QLabel("-")
        self.view_value = QLabel("-")
        self.instance_count_value = QLabel("0")
        self.net_count_value = QLabel("0")
        summary_layout.addRow(self._t("summary.design", "Design:"), self.design_value)
        summary_layout.addRow(self._t("summary.library", "Library:"), self.library_value)
        summary_layout.addRow(self._t("summary.cell", "Cell:"), self.cell_value)
        summary_layout.addRow(self._t("summary.view", "View:"), self.view_value)
        summary_layout.addRow(self._t("summary.instances", "Instances:"), self.instance_count_value)
        summary_layout.addRow(self._t("summary.nets", "Nets:"), self.net_count_value)
        root_layout.addWidget(self.summary_group)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_nets_tab(), self._t("tab.nets", "Nets"))
        self.tabs.addTab(self._build_instances_tab(), self._t("tab.instances", "Instances"))
        self.tabs.addTab(self._build_logic_tab(), self._t("tab.logic", "Logic"))
        self.tabs.addTab(self._build_diagnostics_tab(), self._t("tab.diagnostics", "Diagnostics"))
        self.tabs.addTab(self._build_scene_tab(), self._t("tab.scene", "Scene"))
        root_layout.addWidget(self.tabs, 1)

        self.setCentralWidget(root)

    def _build_nets_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.net_search = QLineEdit()
        self.net_search.setPlaceholderText(self._t("placeholder.filter_nets", "Filter nets by name..."))
        self.net_search.textChanged.connect(self.refresh_nets_table)
        layout.addWidget(self.net_search)

        split = QSplitter(Qt.Orientation.Horizontal)
        self.nets_table = QTableWidget(0, 2)
        self.nets_table.setHorizontalHeaderLabels(
            [self._t("header.net", "Net"), self._t("header.connections", "Connections")]
        )
        self.nets_table.horizontalHeader().setStretchLastSection(True)
        self.nets_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.nets_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.nets_table.itemSelectionChanged.connect(self.on_net_selected)
        split.addWidget(self.nets_table)

        self.net_connections_table = QTableWidget(0, 2)
        self.net_connections_table.setHorizontalHeaderLabels(
            [self._t("header.port", "Port"), self._t("header.instance", "Instance")]
        )
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
        self.instance_search.setPlaceholderText(
            self._t("placeholder.filter_instances", "Filter instances by name/cell/view/designator...")
        )
        self.instance_search.textChanged.connect(self.refresh_instances_table)
        layout.addWidget(self.instance_search)

        split = QSplitter(Qt.Orientation.Horizontal)
        self.instances_table = QTableWidget(0, 4)
        self.instances_table.setHorizontalHeaderLabels(
            [
                self._t("header.instance", "Instance"),
                self._t("header.cell", "Cell"),
                self._t("header.view", "View"),
                self._t("header.designator", "Designator"),
            ]
        )
        self.instances_table.horizontalHeader().setStretchLastSection(True)
        self.instances_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.instances_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.instances_table.itemSelectionChanged.connect(self.on_instance_selected)
        split.addWidget(self.instances_table)

        self.instance_connections_table = QTableWidget(0, 2)
        self.instance_connections_table.setHorizontalHeaderLabels(
            [self._t("header.net", "Net"), self._t("header.port", "Port")]
        )
        self.instance_connections_table.horizontalHeader().setStretchLastSection(True)
        self.instance_connections_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        split.addWidget(self.instance_connections_table)
        split.setSizes([700, 500])
        layout.addWidget(split, 1)
        return tab

    def _build_scene_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.scene_info = QLabel(self._t("scene.status.initial", "Load EDF file to render schematic scene."))
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
        self.diagnostics_status_label = QLabel(
            self._t("diagnostics.status.initial", "Load EDF file to view parser diagnostics.")
        )
        self.diagnostics_status_label.setWordWrap(True)
        layout.addWidget(self.diagnostics_status_label)
        self.diagnostics_table = QTableWidget(0, 2)
        self.diagnostics_table.setHorizontalHeaderLabels(
            [self._t("diagnostics.header.severity", "Severity"), self._t("diagnostics.header.message", "Message")]
        )
        self.diagnostics_table.horizontalHeader().setStretchLastSection(True)
        self.diagnostics_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.diagnostics_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        layout.addWidget(self.diagnostics_table, 1)
        return tab

    def _build_logic_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.logic_status_label = QLabel(self._t("logic.status.initial", "Load EDF file to extract logic functions."))
        self.logic_status_label.setWordWrap(True)
        layout.addWidget(self.logic_status_label)

        highlight_row = QHBoxLayout()
        self.logic_highlight_label = QLabel(self._t("logic.label.highlight_target", "Highlight target:"))
        highlight_row.addWidget(self.logic_highlight_label)
        self.logic_highlight_combo = QComboBox()
        self.logic_highlight_combo.currentTextChanged.connect(self._on_logic_highlight_output_changed)
        highlight_row.addWidget(self.logic_highlight_combo, 1)
        self.clear_highlight_btn = QPushButton(self._t("button.clear_highlight", "Clear Highlight"))
        self.clear_highlight_btn.clicked.connect(self._clear_logic_highlight)
        highlight_row.addWidget(self.clear_highlight_btn)
        layout.addLayout(highlight_row)

        self.logic_simplified_label = QLabel(
            self._t("logic.label.simplified.prefix", "Simplified: ") + self._t("logic.label.none", "-")
        )
        self.logic_simplified_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.logic_simplified_label.setWordWrap(True)
        layout.addWidget(self.logic_simplified_label)

        self.logic_sop_label = QLabel(self._t("logic.label.sop.prefix", "SOP: ") + self._t("logic.label.none", "-"))
        self.logic_sop_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.logic_sop_label.setWordWrap(True)
        layout.addWidget(self.logic_sop_label)

        self.logic_sequential_label = QLabel(
            self._t("logic.label.sequential.prefix", "Sequential elements: ") + self._t("logic.label.none", "-")
        )
        self.logic_sequential_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.logic_sequential_label.setWordWrap(True)
        layout.addWidget(self.logic_sequential_label)

        self.logic_truth_table = QTableWidget(0, 0)
        self.logic_truth_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.logic_truth_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.logic_truth_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.logic_truth_table, 1)
        return tab

    def on_open_file(self) -> None:
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            self._t("dialog.open_edf.title", "Open EDF File"),
            str(self._current_path.parent if self._current_path else Path.cwd()),
            self._t("dialog.open_edf.filter", "EDIF Files (*.edf *.EDF);;All Files (*.*)"),
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
            QMessageBox.critical(
                self,
                self._t("dialog.parse_error.title", "Parse Error"),
                self._t("dialog.parse_error.body", "Failed to parse EDF:\n{error}", error=str(exc)),
            )
            return
        try:
            self._logic_data = self._extract_logic_functions.execute(path)
        except Exception as exc:  # noqa: BLE001
            self._logic_data = None
            QMessageBox.warning(
                self,
                self._t("dialog.logic_error.title", "Logic Extraction"),
                self._t("dialog.logic_error.body", "Failed to extract logic functions:\n{error}", error=str(exc)),
            )

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
            self.net_connections_table.setItem(
                table_row,
                1,
                QTableWidgetItem(connection.instance or self._t("value.top_port", "<top-port>")),
            )

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
            self._t(
                "scene.status.rendered",
                "Rendered primitives: {primitive_count}, wire segments: {wire_segment_count}, text labels: {text_count}",
                primitive_count=stats.primitive_count,
                wire_segment_count=stats.wire_segment_count,
                text_count=stats.text_count,
            )
        )

    def refresh_logic_tab(self) -> None:
        self._logic_highlight_targets = {}
        self.logic_highlight_combo.blockSignals(True)
        self.logic_highlight_combo.clear()
        self.logic_truth_table.setRowCount(0)
        self.logic_truth_table.setColumnCount(0)
        if not self._logic_data:
            self.logic_status_label.setText(self._t("logic.status.unavailable", "Logic function data unavailable for current file."))
            self.logic_simplified_label.setText(
                self._t("logic.label.simplified.prefix", "Simplified: ") + self._t("logic.label.none", "-")
            )
            self.logic_sop_label.setText(self._t("logic.label.sop.prefix", "SOP: ") + self._t("logic.label.none", "-"))
            self.logic_sequential_label.setText(
                self._t("logic.label.sequential.prefix", "Sequential elements: ") + self._t("logic.label.none", "-")
            )
            self.logic_highlight_combo.blockSignals(False)
            self._clear_logic_highlight()
            return

        outputs = self._logic_data.get("outputs", {})
        sequential_elements = list(self._logic_data.get("sequential_elements", []))
        if not outputs and not sequential_elements:
            self.logic_status_label.setText(
                self._t(
                    "logic.status.no_logic",
                    "No outputs or sequential elements detected for current file.",
                )
            )
            self.logic_simplified_label.setText(
                self._t("logic.label.simplified.prefix", "Simplified: ") + self._t("logic.label.none", "-")
            )
            self.logic_sop_label.setText(self._t("logic.label.sop.prefix", "SOP: ") + self._t("logic.label.none", "-"))
            self.logic_sequential_label.setText(
                self._t("logic.label.sequential.prefix", "Sequential elements: ") + self._t("logic.label.none", "-")
            )
            self.logic_highlight_combo.blockSignals(False)
            self._clear_logic_highlight()
            return

        input_count = len(self._logic_data.get("inputs", []))
        meta = self._logic_data.get("meta", {})
        self.logic_status_label.setText(
            self._t(
                "logic.status.summary.extended",
                "Inputs: {input_count}, Outputs: {output_count}, Sequential: {sequential_count}, Transistors: {transistor_count}",
                input_count=input_count,
                output_count=len(outputs),
                sequential_count=meta.get("sequential_count", len(sequential_elements)),
                transistor_count=meta.get("transistor_count", 0),
            )
        )
        input_names = self._logic_data.get("inputs", [])
        output_names = list(outputs.keys())
        for output_name in output_names:
            label = f"OUT: {output_name}"
            self.logic_highlight_combo.addItem(label)
            self._logic_highlight_targets[label] = outputs.get(output_name, {}).get("logic_path", {})
        for element in sequential_elements:
            instance_name = str(element.get("instance") or "?")
            kind_name = str(element.get("kind") or "sequential")
            subtype = str(element.get("subtype") or "").strip()
            type_text = f"{subtype}-{kind_name}" if subtype and subtype != "unknown" else kind_name
            label = f"SEQ: {instance_name} ({type_text})"
            self.logic_highlight_combo.addItem(label)
            self._logic_highlight_targets[label] = element.get("highlight") or element.get("logic_path") or {}
        self.logic_highlight_combo.blockSignals(False)
        if self.logic_highlight_combo.count():
            self.logic_highlight_combo.setCurrentIndex(0)

        if output_names:
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
                    rendered = self._t("logic.table.unknown_value", "X") if out_value is None else str(out_value)
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
        self.logic_simplified_label.setText(
            self._t("logic.label.simplified.prefix", "Simplified: ")
            + (" | ".join(expr_parts) if expr_parts else self._t("logic.label.none", "-"))
        )
        self.logic_sop_label.setText(
            self._t("logic.label.sop.prefix", "SOP: ")
            + (" | ".join(sop_parts) if sop_parts else self._t("logic.label.none", "-"))
        )
        self.logic_sequential_label.setText(
            self._t("logic.label.sequential.prefix", "Sequential elements: ")
            + self._render_sequential_elements_summary(sequential_elements)
        )
        if self.logic_highlight_combo.currentText():
            self._on_logic_highlight_output_changed(self.logic_highlight_combo.currentText())
        else:
            self._clear_logic_highlight()

    def _format_logic_activation(self, activation: dict | None) -> str:
        if not isinstance(activation, dict):
            return self._t("logic.label.none", "-")
        mode = str(activation.get("mode") or "").lower()
        if mode == "edge":
            edge = str(activation.get("edge") or "unknown").lower()
            if edge == "rising":
                return self._t("logic.activation.rising_edge", "rising edge")
            if edge == "falling":
                return self._t("logic.activation.falling_edge", "falling edge")
            return self._t("logic.activation.edge_unknown", "edge")
        if mode == "level":
            level = str(activation.get("level") or "unknown").lower()
            if level == "high":
                return self._t("logic.activation.high_level", "high level")
            if level == "low":
                return self._t("logic.activation.low_level", "low level")
            return self._t("logic.activation.level_unknown", "level")
        return self._t("logic.label.none", "-")

    def _logic_port_labels_by_net(self) -> dict[str, list[str]]:
        if not isinstance(self._logic_data, dict):
            return {}
        labels_by_net: dict[str, list[str]] = {}

        def add(net_name: str, label: str) -> None:
            if not net_name or not label:
                return
            labels = labels_by_net.setdefault(str(net_name), [])
            if label not in labels:
                labels.append(label)

        meta = self._logic_data.get("meta", {})
        for input_name, net_name in (meta.get("input_nets", {}) or {}).items():
            add(str(net_name), str(input_name))
        outputs = self._logic_data.get("outputs", {})
        for output_name, output_data in outputs.items():
            if not isinstance(output_data, dict):
                continue
            add(str(output_data.get("net") or ""), str(output_name))
        return labels_by_net

    def _logic_output_path_labels_by_net(self) -> dict[str, list[str]]:
        if not isinstance(self._logic_data, dict):
            return {}
        labels_by_net: dict[str, list[str]] = {}
        outputs = self._logic_data.get("outputs", {})
        for output_name, output_data in outputs.items():
            if not isinstance(output_data, dict):
                continue
            logic_path = output_data.get("logic_path", {})
            if not isinstance(logic_path, dict):
                continue
            for net_name in logic_path.get("nets", []) or []:
                if not net_name:
                    continue
                labels = labels_by_net.setdefault(str(net_name), [])
                label = str(output_name)
                if label not in labels:
                    labels.append(label)
        return labels_by_net

    def _logic_display_net_name(self, net_name: str, *, prefer_output_path: bool = False) -> str:
        if not net_name:
            return net_name
        direct_labels = self._logic_port_labels_by_net().get(str(net_name), [])
        if direct_labels:
            return "/".join(direct_labels)
        if prefer_output_path:
            output_path_labels = self._logic_output_path_labels_by_net().get(str(net_name), [])
            if output_path_labels:
                return "/".join(output_path_labels) + "~"
        return net_name

    @staticmethod
    def _logic_function_expression(function_data: dict | None) -> str | None:
        if not isinstance(function_data, dict):
            return None
        mux = function_data.get("mux")
        if isinstance(mux, dict):
            mux_expr = str(mux.get("expression") or "").strip()
            if mux_expr:
                return mux_expr
        for key in ("expression", "simplified_expression", "sum_of_products"):
            value = str(function_data.get(key) or "").strip()
            if value:
                return value
        return None

    def _render_sequential_elements_summary(self, elements: list[dict]) -> str:
        if not elements:
            return self._t("logic.label.none", "-")
        parts: list[str] = []
        for element in elements:
            instance_name = str(element.get("instance") or "?")
            kind_name = str(element.get("kind") or "sequential")
            subtype = str(element.get("subtype") or "").strip()
            if subtype and subtype != "unknown":
                type_label = f"{subtype}-{kind_name}"
            else:
                type_label = kind_name

            trigger_info = element.get("triggering")
            trigger_text = ""
            if isinstance(trigger_info, dict):
                trigger_mode = str(trigger_info.get("mode") or "").lower()
                trigger_net = str(trigger_info.get("net") or "")
                trigger_net_label = self._logic_display_net_name(trigger_net)
                trigger_function_text = self._logic_function_expression(trigger_info.get("function")) if isinstance(trigger_info.get("function"), dict) else None
                if trigger_function_text and (not trigger_net_label or trigger_net_label == trigger_net):
                    trigger_net_label = trigger_function_text
                if trigger_mode == "edge":
                    edge = self._format_logic_activation({"mode": "edge", "edge": trigger_info.get("edge")})
                    if trigger_net_label:
                        trigger_text = f", CLK={trigger_net_label} ({edge})"
                    else:
                        trigger_text = f", {edge}"
                elif trigger_mode == "level":
                    level = self._format_logic_activation({"mode": "level", "level": trigger_info.get("level")})
                    if trigger_net_label:
                        trigger_text = f", EN={trigger_net_label} ({level})"
                    else:
                        trigger_text = f", {level}"

            data_parts: list[str] = []
            for idx, data_input in enumerate(element.get("data_inputs", []), start=1):
                net_name = str(data_input.get("net") or "")
                if not net_name:
                    continue
                net_label = self._logic_display_net_name(net_name)
                function_expr = self._logic_function_expression(data_input.get("function"))
                signal_label = "D" if idx == 1 else f"D{idx}"
                if function_expr:
                    data_parts.append(f"{signal_label}={function_expr}")
                else:
                    data_parts.append(f"{signal_label}={net_label}")
            data_text = f", {'; '.join(data_parts)}" if data_parts else ""

            control_parts: list[str] = []
            for control in element.get("control_signals", []):
                role = str(control.get("role") or "")
                if role in {"clock", "enable"}:
                    continue
                net_name = str(control.get("net") or "")
                if not net_name:
                    continue
                net_label = self._logic_display_net_name(net_name)
                function_expr = self._logic_function_expression(control.get("function"))
                if function_expr and (not control.get("labels")) and net_label == net_name:
                    net_label = function_expr
                activation_text = self._format_logic_activation(control.get("activation"))
                control_parts.append(f"{role}={net_label} ({activation_text})")
            control_text = f", {'; '.join(control_parts)}" if control_parts else ""

            outputs: list[str] = []
            for item in element.get("outputs", []):
                if not item.get("net"):
                    continue
                mapped_name = self._logic_display_net_name(str(item.get("net") or ""), prefer_output_path=True)
                if mapped_name and mapped_name not in outputs:
                    outputs.append(mapped_name)
            output_text = f", Q={','.join(outputs)}" if outputs else ""

            parts.append(f"{instance_name}: {type_label}{trigger_text}{data_text}{control_text}{output_text}")
        return " | ".join(parts)

    def refresh_diagnostics_tab(self) -> None:
        self.diagnostics_table.setRowCount(0)
        if self._scene_data is None:
            self.diagnostics_status_label.setText(
                self._t("diagnostics.status.initial", "Load EDF file to view parser diagnostics.")
            )
            return
        diagnostics = list(self._scene_data.diagnostics)
        if not diagnostics:
            self.diagnostics_status_label.setText(self._t("diagnostics.status.no_diagnostics", "No parser diagnostics."))
            return
        severity_counts = {"error": 0, "warning": 0, "info": 0}
        for item in diagnostics:
            severity_counts[item.severity] = severity_counts.get(item.severity, 0) + 1
        self.diagnostics_status_label.setText(
            self._t(
                "diagnostics.status.summary",
                "Parser diagnostics: {total} total (errors: {errors}, warnings: {warnings}, info: {info}).",
                total=len(diagnostics),
                errors=severity_counts.get("error", 0),
                warnings=severity_counts.get("warning", 0),
                info=severity_counts.get("info", 0),
            )
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

    def _on_logic_highlight_output_changed(self, selection_label: str) -> None:
        if not selection_label or not self._logic_data:
            self._clear_logic_highlight()
            return
        target = self._logic_highlight_targets.get(selection_label)
        if not isinstance(target, dict):
            outputs = self._logic_data.get("outputs", {})
            output_data = outputs.get(selection_label, {})
            target = output_data.get("logic_path", {})
        nets = set(target.get("nets", []))
        instances = set(target.get("instances", []))
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

    def _on_language_changed(self, language_name: str) -> None:
        if not language_name:
            return
        self._current_language = language_name
        self._ui = load_ui_strings(language_name)
        if self._theme_manager is not None:
            self._theme_manager.save_language(language_name)
        self._reload_ui_texts()

    def _reload_ui_texts(self) -> None:
        self.setWindowTitle(self._t("window.title", "EDF Netlist Viewer"))
        if self._current_path is None:
            self.path_label.setText(self._t("path.no_file", "No EDF file loaded"))

        self.open_btn.setText(self._t("button.open_edf", "Open EDF"))
        self.zoom_in_btn.setText(self._t("button.zoom_in", "Zoom In"))
        self.zoom_out_btn.setText(self._t("button.zoom_out", "Zoom Out"))
        self.zoom_reset_btn.setText(self._t("button.reset_zoom", "Reset Zoom"))
        self.hitbox_toggle_btn.setText(self._t("button.show_hitboxes", "Show Hitboxes"))
        self.language_label.setText(self._t("label.language", "Language:"))
        self.theme_label.setText(self._t("label.theme", "Theme:"))

        self.summary_group.setTitle(self._t("group.summary", "Top-Level Summary"))
        form_layout = self.summary_group.layout()
        if isinstance(form_layout, QFormLayout):
            form_layout.labelForField(self.design_value).setText(self._t("summary.design", "Design:"))
            form_layout.labelForField(self.library_value).setText(self._t("summary.library", "Library:"))
            form_layout.labelForField(self.cell_value).setText(self._t("summary.cell", "Cell:"))
            form_layout.labelForField(self.view_value).setText(self._t("summary.view", "View:"))
            form_layout.labelForField(self.instance_count_value).setText(self._t("summary.instances", "Instances:"))
            form_layout.labelForField(self.net_count_value).setText(self._t("summary.nets", "Nets:"))

        self.tabs.setTabText(0, self._t("tab.nets", "Nets"))
        self.tabs.setTabText(1, self._t("tab.instances", "Instances"))
        self.tabs.setTabText(2, self._t("tab.logic", "Logic"))
        self.tabs.setTabText(3, self._t("tab.diagnostics", "Diagnostics"))
        self.tabs.setTabText(4, self._t("tab.scene", "Scene"))

        self.net_search.setPlaceholderText(self._t("placeholder.filter_nets", "Filter nets by name..."))
        self.nets_table.setHorizontalHeaderLabels([self._t("header.net", "Net"), self._t("header.connections", "Connections")])
        self.net_connections_table.setHorizontalHeaderLabels(
            [self._t("header.port", "Port"), self._t("header.instance", "Instance")]
        )

        self.instance_search.setPlaceholderText(
            self._t("placeholder.filter_instances", "Filter instances by name/cell/view/designator...")
        )
        self.instances_table.setHorizontalHeaderLabels(
            [
                self._t("header.instance", "Instance"),
                self._t("header.cell", "Cell"),
                self._t("header.view", "View"),
                self._t("header.designator", "Designator"),
            ]
        )
        self.instance_connections_table.setHorizontalHeaderLabels(
            [self._t("header.net", "Net"), self._t("header.port", "Port")]
        )

        self.logic_highlight_label.setText(self._t("logic.label.highlight_target", "Highlight target:"))
        self.clear_highlight_btn.setText(self._t("button.clear_highlight", "Clear Highlight"))
        self.diagnostics_table.setHorizontalHeaderLabels(
            [self._t("diagnostics.header.severity", "Severity"), self._t("diagnostics.header.message", "Message")]
        )

        self.refresh_logic_tab()
        self.refresh_diagnostics_tab()
        if self._scene_data is None:
            self.scene_info.setText(self._t("scene.status.initial", "Load EDF file to render schematic scene."))
