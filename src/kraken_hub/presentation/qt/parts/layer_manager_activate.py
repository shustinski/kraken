"""manager_app: layer manager activate."""
from __future__ import annotations
from uuid import uuid4
from kraken_manager.domain.project import RepresentationKind


class LayerManagerActivateMixin:
    def _layer_manager_select(self, layer_id: str) -> None:
        workspace = self._workspace
        if workspace is None:
            return
        item = workspace.layer_model.layer_by_id(layer_id)
        if item is None:
            return
        for index in range(workspace.layer_tabs.count()):
            if str(workspace.layer_tabs.tabData(index)) == layer_id:
                if workspace.layer_tabs.currentIndex() != index:
                    workspace.layer_tabs.setCurrentIndex(index)
                    return
                break
        self._select_layer(workspace, self._project_id, item)

    def _layer_manager_reorder(self, layer_ids: tuple[str, ...]) -> None:
        workspace = self._workspace
        project = self.service.get_project(self._project_id)
        if workspace is None or project is None:
            return
        layers = self.service.list_layers(project.id)
        try:
            self.service.reorder_layers(
                principal=self.session.principal,
                project=project,
                layers=layers,
                layer_ids=layer_ids,
                idempotency_key=str(uuid4()),
            )
        except Exception as exc:
            self._error(str(exc))
        latest = self.service.get_project(project.id)
        if latest is not None:
            self._load_layers(workspace, latest)

    def _layer_manager_activate(self, representation_id: str) -> None:
        workspace = self._workspace
        if workspace is None:
            return
        layer_id = str(getattr(workspace, "_selected_layer_id", ""))
        representations = self.service.list_representations(self._project_id, layer_id)
        selected = next(
            (item for item in representations if str(item.id) == str(representation_id)),
            None,
        )
        if (
            selected is not None
            and selected.kind is RepresentationKind.VECTOR
            and selected.source_image_representation_id is not None
        ):
            # A vector is rendered together with its linked raster.  Activating
            # both keeps the hidden compatibility selectors and the matrix data
            # source on the same pair selected in the graph.
            self._activate_representation(
                workspace,
                self._project_id,
                str(selected.source_image_representation_id),
            )
        self._activate_representation(workspace, self._project_id, representation_id)
        if self._layer_dialog is not None and layer_id:
            self._refresh_layer_pipeline(self._project_id, layer_id)
