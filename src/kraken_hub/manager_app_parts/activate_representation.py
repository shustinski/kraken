"""manager_app: activate representation."""
from __future__ import annotations
from uuid import uuid4
from kraken_manager.domain.project import RepresentationKind


class ActivateRepresentationMixin:
    def _activate_representation(self, workspace, project_id, representation_id: str) -> None:
        if not representation_id:
            self._refresh_matrix(workspace, project_id)
            return
        layer_id = getattr(workspace, "_selected_layer_id", None)
        project = self.service.get_project(project_id)
        if project is None or layer_id is None:
            return
        layer = next(
            (item for item in self.service.list_layers(project.id) if str(item.id) == str(layer_id)),
            None,
        )
        representation = next(
            (
                item
                for item in self.service.list_representations(project.id, layer_id)
                if str(item.id) == str(representation_id)
            ),
            None,
        )
        if layer is None or representation is None:
            self._refresh_matrix(workspace, project_id)
            return
        if representation.kind is RepresentationKind.VECTOR:
            source_image_id = str(representation.source_image_representation_id or "")
            image_index = workspace.image_representation_combo.findData(source_image_id)
            if image_index >= 0:
                workspace.image_representation_combo.blockSignals(True)
                workspace.image_representation_combo.setCurrentIndex(image_index)
                workspace.image_representation_combo.blockSignals(False)
            self._load_representations(workspace, project_id, layer_id)
            vector_index = workspace.vector_representation_combo.findData(
                str(representation.id)
            )
            if vector_index >= 0:
                workspace.vector_representation_combo.blockSignals(True)
                workspace.vector_representation_combo.setCurrentIndex(vector_index)
                workspace.vector_representation_combo.blockSignals(False)
        if representation.active:
            self._load_representations(workspace, project_id, layer_id)
            return
        try:
            self.service.activate_representation(
                principal=self.session.principal,
                project=project,
                layer=layer,
                representation=representation,
                idempotency_key=str(uuid4()),
            )
        except Exception as exc:
            self._error(str(exc))
            return
        self._load_representations(workspace, project_id, layer_id)
