"""window: reload ui texts."""
from __future__ import annotations
from PyQt6.QtWidgets import (
    QFormLayout,

)


class ReloadUiTextsMixin:
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
