"""Translate a frontend request to the shared layer-rename command."""
from kraken_manager.application.dto import CommandContext, RenameLayerCommand
from kraken_manager.application.errors import ConflictError
from kraken_manager.domain.identity import Principal
from kraken_manager.domain.project import Project, Layer
from kraken_manager.workspace import WorkspaceValidationError


class RenameLayerMixin:
    def rename_layer(self, *, principal: Principal, project: Project, layer: Layer, name: str, idempotency_key: str) -> Layer:
        try:
            return self._commands(str(project.id)).execute(RenameLayerCommand(
                context=CommandContext(actor=principal, idempotency_key=idempotency_key),
                project_id=project.id, layer_id=layer.id, name=name, expected_revision=layer.revision,
            ))
        except ConflictError as exc:
            raise WorkspaceValidationError(str(exc)) from exc
