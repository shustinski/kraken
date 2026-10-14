"""manager_app: open layer manager."""
from __future__ import annotations
from kraken_manager.presentation.qt import (
    LayerManagerDialog,
)
from kraken_manager.presentation.qt.models import LayerListItem


class OpenLayerManagerMixin:
    def _load_layers(self, workspace, project) -> None:
        colors = {
            "metal": "#60A5FA",
            "contact": "#F59E0B",
            "gate": "#A78BFA",
            "diffusion": "#34D399",
        }
        workspace.layer_model.replace_items(
            LayerListItem(str(layer.id), layer.name, layer.type.value, colors[layer.type.value])
            for layer in self.service.list_layers(project.id)
        )
        workspace.sync_layer_tabs()
        if self._layer_dialog is not None:
            self._layer_dialog.set_layers(
                list(workspace.layer_model.items()),
                str(getattr(workspace, "_selected_layer_id", "")),
            )

    def _show_created_layer(self, workspace, project_id, layer) -> None:
        """Reload and select a layer so its registered image source is visible."""

        latest = self.service.get_project(project_id)
        if latest is None:
            return
        self._load_layers(workspace, latest)
        item = workspace.layer_model.layer_by_id(str(layer.id))
        if item is None:
            return
        for index in range(workspace.layer_tabs.count()):
            if str(workspace.layer_tabs.tabData(index)) == str(layer.id):
                if workspace.layer_tabs.currentIndex() != index:
                    workspace.layer_tabs.setCurrentIndex(index)
                break
        self._select_layer(workspace, project_id, item)

    def _select_layer(self, workspace, project_id, item: LayerListItem) -> None:
        workspace._selected_layer_id = item.layer_id
        workspace.add_image_representation_button.setEnabled(True)
        self._load_representations(workspace, project_id, item.layer_id)
        if self._layer_dialog is not None:
            self._layer_dialog.select_layer(item.layer_id)
            self._refresh_layer_pipeline(project_id, item.layer_id)

    def open_layer_manager(self) -> None:
        workspace = self._workspace
        project_id = self._project_id
        if workspace is None or not project_id or self.shell.current_page_key() != "workspace":
            return
        if self._layer_dialog is None:
            dialog = LayerManagerDialog(
                project_id,
                self.shell,
                action_availability=self._action_availability,
            )
            dialog.layerSelected.connect(self._layer_manager_select)
            dialog.orderChanged.connect(self._layer_manager_reorder)
            dialog.representationActivated.connect(self._layer_manager_activate)
            dialog.nodeActionRequested.connect(self._layer_manager_action)
            dialog.layerActionRequested.connect(self._layer_manager_layer_action)
            dialog.nodePropertiesRequested.connect(self._show_node_properties)
            dialog.layerPropertiesRequested.connect(self._show_layer_properties)
            dialog.addLayerRequested.connect(
                lambda: self._add_layer(workspace, project_id)
            )
            self._layer_dialog = dialog
        self._layer_dialog.set_layers(
            list(workspace.layer_model.items()),
            str(getattr(workspace, "_selected_layer_id", "")),
        )
        selected = str(getattr(workspace, "_selected_layer_id", ""))
        if selected and not self._refresh_layer_pipeline(project_id, selected):
            return
        self._layer_dialog.show()
        self._layer_dialog.raise_()
        self._layer_dialog.activateWindow()
