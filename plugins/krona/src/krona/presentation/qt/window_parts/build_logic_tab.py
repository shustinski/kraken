"""window: build logic tab."""
from __future__ import annotations
from pathlib import Path
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSplitter,
    QTableWidget,
    QVBoxLayout,
    QWidget,

)
from krona.presentation.qt.scene_renderer import EdifSceneRenderer


class BuildLogicTabMixin:
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
