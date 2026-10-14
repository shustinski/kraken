"""manager_app: archive layer representation."""
from __future__ import annotations
from uuid import uuid4
from kraken_manager.presentation.qt import (
    PipelineNode,
)


class ArchiveLayerRepresentationMixin:
    def _archive_layer_representation(self, layer_id: str, node: PipelineNode) -> None:
        from PyQt6.QtWidgets import QMessageBox

        workspace = self._workspace
        project = self.service.get_project(self._project_id)
        layer = next(
            (
                item
                for item in self.service.list_layers(self._project_id)
                if str(item.id) == str(layer_id)
            ),
            None,
        )
        representation_id = str(node.representation_id or node.node_id)
        representation = next(
            (
                item
                for item in self.service.list_representations(
                    self._project_id, layer_id
                )
                if str(item.id) == representation_id
            ),
            None,
        )
        if workspace is None or project is None or layer is None or representation is None:
            self._error("Слой изображений больше не доступен")
            return
        answer = QMessageBox.question(
            self._layer_dialog or workspace,
            "Удалить слой изображений",
            (
                f"Удалить «{representation.name}» из проекта?\n\n"
                "Связанная история обработки останется в журнале проекта."
            ),
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            self.service.archive_representation(
                principal=self.session.principal,
                project=project,
                layer=layer,
                representation=representation,
                idempotency_key=str(uuid4()),
            )
        except Exception as exc:
            self._error(str(exc))
            return
        self._load_representations(workspace, self._project_id, layer_id)
        if self._layer_dialog is not None:
            self._layer_dialog.set_pipeline(
                self._pipeline_snapshot(self._project_id, layer_id)
            )
