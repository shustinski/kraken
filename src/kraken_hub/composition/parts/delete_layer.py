"""Local adapter for the shared layer-deletion use case."""
from kraken_manager.application.dto import CommandContext, DeleteLayerCommand
from kraken_manager.application.errors import ConflictError, NotFoundError
from kraken_manager.domain.identity import Principal
from kraken_manager.domain.project import Layer, Project
from kraken_manager.workspace import WorkspaceValidationError


class DeleteLayerMixin:
    def delete_layer(self, *, principal: Principal, project: Project, layer: Layer,
                     confirmation_name: str, idempotency_key: str) -> Layer:
        try:
            return self._commands(str(project.id)).execute(DeleteLayerCommand(
                context=CommandContext(actor=principal, idempotency_key=idempotency_key),
                project_id=project.id, layer_id=layer.id, confirmation_name=confirmation_name,
                expected_revision=layer.revision,
            ))
        except (ConflictError, NotFoundError, ValueError) as exc:
            raise WorkspaceValidationError(str(exc)) from exc
