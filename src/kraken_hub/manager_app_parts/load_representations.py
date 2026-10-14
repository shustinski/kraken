"""manager_app: load representations."""
from __future__ import annotations
from kraken_manager.domain.identity import Permission
from kraken_manager.domain.project import RepresentationKind, RepresentationPurpose


class LoadRepresentationsMixin:
    def _set_matrix_visual_mode(self, _channel: str, _mode: str) -> None:
        workspace = self._workspace
        if workspace is not None:
            workspace.matrix_view.set_visual_modes(*self.shell.visual_modes())

    def _load_representations(self, workspace, project_id, layer_id) -> None:
        representations = self.service.list_representations(project_id, layer_id)
        images = sorted(
            (value for value in representations if value.kind is RepresentationKind.IMAGE),
            key=lambda value: (not value.active, value.name.casefold()),
        )
        current_image_id = str(workspace.image_representation_combo.currentData() or "")
        if not any(str(item.id) == current_image_id for item in images):
            current_image_id = str(images[0].id) if images else ""
        current_vector_id = str(workspace.vector_representation_combo.currentData() or "")
        vectors = sorted(
            (
                value
                for value in representations
                if value.kind is RepresentationKind.VECTOR
                and str(value.source_image_representation_id or "") == current_image_id
            ),
            key=lambda value: (not value.active, value.name.casefold()),
        )
        if not any(str(item.id) == current_vector_id for item in vectors):
            current_vector_id = str(vectors[0].id) if vectors else ""
        workspace.set_representations(
            images=[
                (str(item.id), item.name)
                for item in images
            ],
            vectors=[
                (str(item.id), item.name)
                for item in vectors
            ],
            selected_image_id=current_image_id,
            selected_vector_id=current_vector_id,
        )
        workspace.add_vector_representation_button.setEnabled(bool(images))
        self._refresh_matrix(workspace, project_id)
        selected = next((item for item in images if str(item.id) == current_image_id), None)
        if (
            selected is not None
            and selected.purpose is RepresentationPurpose.SOURCE
            and selected.source
            and selected.source != "managed-import"
            and getattr(self.service, "is_remote_project", lambda _id: False)(project_id)
            and self._has_permission(Permission.IMPORT_ARTIFACT)
        ):
            self._ensure_agent_source_ready(
                layer_id=str(layer_id),
                source_representation_id=current_image_id,
                retry=lambda: self._refresh_matrix(workspace, project_id),
            )
