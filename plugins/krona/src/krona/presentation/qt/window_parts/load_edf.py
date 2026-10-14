"""window: load edf."""
from __future__ import annotations
from pathlib import Path
from PyQt6.QtWidgets import (
    QMessageBox,
    QTableWidgetItem,

)


class LoadEdfMixin:
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
