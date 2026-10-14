"""window: refresh diagnostics tab."""
from __future__ import annotations
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QApplication,
    QTableWidgetItem,

)
from krona.presentation.qt.ui_strings import load_ui_strings


class RefreshDiagnosticsTabMixin:
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
        self._sync_theme_controls()
        self.scene_renderer.set_dark_mode(self._current_theme.lower() == "dark")
        if self._scene_data is not None:
            self.refresh_graphics_scene(reset_view=False)

    def _sync_theme_controls(self) -> None:
        if hasattr(self, "theme_combo") and self.theme_combo.currentText() != self._current_theme:
            self.theme_combo.blockSignals(True)
            self.theme_combo.setCurrentText(self._current_theme)
            self.theme_combo.blockSignals(False)
        if hasattr(self, "_kraken_theme_dark_action"):
            self._kraken_theme_dark_action.setChecked(self._current_theme.lower() == "dark")
        if hasattr(self, "_kraken_theme_light_action"):
            self._kraken_theme_light_action.setChecked(self._current_theme.lower() == "light")

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
