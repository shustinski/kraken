"""Translate a frontend request to the shared project-rename command."""
from kraken_manager.application.dto import CommandContext, RenameProjectCommand
from kraken_manager.application.errors import ConflictError
from kraken_manager.domain.identity import Principal
from kraken_manager.domain.project import Project
from kraken_manager.workspace import WorkspaceValidationError


class RenameProjectMixin:
    def rename_project(self, *, principal: Principal, project: Project, name: str, idempotency_key: str) -> Project:
        try:
            return self._commands(str(project.id)).execute(RenameProjectCommand(
                context=CommandContext(actor=principal, idempotency_key=idempotency_key),
                project_id=project.id, name=name, expected_revision=project.revision,
            ))
        except ConflictError as exc:
            raise WorkspaceValidationError(str(exc)) from exc
