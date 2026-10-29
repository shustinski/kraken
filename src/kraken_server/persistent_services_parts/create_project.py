"""persistent_services: create project."""
from __future__ import annotations
from kraken_server.project_deletion import guarded_storage
from dataclasses import replace
from collections.abc import Mapping
from typing import Any
from uuid import NAMESPACE_URL, uuid5
from kraken_manager.application.dto import (
    CreateProjectCommand,
    RenameProjectCommand,
)
from kraken_manager.application.dto import (
    CommandContext as ApplicationCommandContext,
)
from kraken_manager.application.errors import (
    AuthorizationError as ApplicationAuthorizationError,
)
from kraken_manager.application.errors import (
    ConcurrencyError as ApplicationConcurrencyError,
)
from kraken_manager.application.errors import (
    ConflictError as ApplicationConflictError,
)
from kraken_manager.application.errors import (
    NotFoundError as ApplicationNotFoundError,
)
from kraken_manager.application.errors import (
    StorageCapabilityError,
)
from kraken_manager.domain.common import (
    PrincipalId,
    ProjectId,
    validate_uuid,
)
from kraken_manager.domain.identity import SystemRole
from kraken_manager.domain.project import (
    GridOrientation,
)
from kraken_manager.infrastructure.filesystem._codec import encode_model
from kraken_server.services import CommandContext, ConflictError, ForbiddenError, NotFoundError, ValidationError

from kraken_server.persistent_services_parts.layer_source_scan import (
    _project_dict,
    _project_id,
)


class CreateProjectMixin:
    def health(self) -> dict[str, Any]:
        try:
            from sqlalchemy import text

            with self.engine.connect() as connection:
                connection.execute(text("SELECT 1")).scalar_one()
        except Exception as exc:
            return {"status": "degraded", "metadata": "unavailable", "detail": str(exc)[:500]}
        status = {"status": "ok", "metadata": "postgresql", "api_version": "v1"}
        gateway = getattr(self, "agent_gateway", None)
        if gateway is not None:
            status["compute_admission"] = gateway.resource_status()
        else:
            status["compute_admission"] = {"state": "unavailable", "admission_ready": False}
        return status

    def list_projects(self, *, include_archived: bool = False) -> list[dict[str, Any]]:
        return [
            _project_dict(project)
            for project in self.projections.list_projects(
                include_archived=include_archived
            )
        ]

    def project_ids_for_principal(self, principal_id: str) -> tuple[str, ...]:
        try:
            identifier = PrincipalId(validate_uuid(principal_id, field="principal_id"))
        except (TypeError, ValueError):
            return ()
        return tuple(sorted(self.identities.project_ids_for(identifier)))

    def principal_system_roles(self, principal_id: str) -> tuple[str, ...]:
        try:
            identifier = PrincipalId(validate_uuid(principal_id, field="principal_id"))
        except (TypeError, ValueError):
            return ()
        principal = self.identities.get(identifier)
        if principal is None:
            return ()
        return tuple(sorted(role.value for role in principal.system_roles))

    def list_principals(self, *, include_inactive: bool = False) -> list[dict[str, Any]]:
        people = {
            str(principal.id): {
                "principal_id": str(principal.id),
                "provider": principal.provider.value,
                "subject": principal.subject,
                "issuer": principal.issuer,
                "display_name": principal.display_name,
                "email": principal.email,
                "active": principal.active,
                "system_roles": sorted(role.value for role in principal.system_roles),
            }
            for principal in self.identities.list(include_inactive=include_inactive)
        }
        accounts = getattr(self, "accounts", None)
        list_accounts = getattr(accounts, "list_accounts", None)
        if callable(list_accounts):
            roles_for = getattr(accounts, "global_roles_for", None)
            for account in list_accounts(include_disabled=include_inactive):
                identifier = str(account.account_id)
                if identifier in people:
                    continue
                system_roles = () if not callable(roles_for) else roles_for(identifier)
                people[identifier] = {
                    "principal_id": identifier,
                    "provider": "local",
                    "subject": account.username,
                    "issuer": None,
                    "display_name": account.display_name,
                    "email": None,
                    "active": bool(account.enabled),
                    "system_roles": sorted(str(role) for role in system_roles),
                }
        return list(people.values())

    def list_performers(self, *, include_archived: bool = False) -> list[dict[str, Any]]:
        if self.performer_store is None:
            raise ConflictError("Server performer store is not configured")
        return [
            encode_model(item)
            for item in self.performer_store.list(include_archived=include_archived)
        ]

    def _actor(self, actor_id: str) -> Any:
        try:
            principal_id = PrincipalId(validate_uuid(actor_id, field="actor_id"))
            actor = self.identities.get(principal_id)
        except (TypeError, ValueError) as exc:
            raise ValidationError("Authenticated principal id is invalid") from exc
        if actor is None or not actor.active:
            raise ValidationError("Authenticated principal is unknown or inactive")
        # Server administration is a local console concern. A network session
        # must not carry that role into project commands.
        if SystemRole.SERVER_ADMIN in actor.system_roles:
            actor = replace(
                actor,
                system_roles=frozenset(role for role in actor.system_roles if role is not SystemRole.SERVER_ADMIN),
            )
        return actor

    @guarded_storage
    def create_project(self, payload: Mapping[str, Any], context: CommandContext) -> dict[str, Any]:
        actor = self._actor(context.actor_id)
        if context.expected_revision not in {None, 0}:
            raise ConflictError("A new project has expected revision 0")
        stable_id = payload.get("project_id") or str(
            uuid5(NAMESPACE_URL, f"kraken:project:{actor.id}:{context.idempotency_key}")
        )
        try:
            command = CreateProjectCommand(
                context=ApplicationCommandContext(
                    actor=actor,
                    idempotency_key=context.idempotency_key,
                    gitlab_identity_verified=True,
                ),
                name=str(payload.get("name", "")),
                width=int(payload.get("width", 0)),
                height=int(payload.get("height", 0)),
                orientation=GridOrientation(str(payload.get("orientation", GridOrientation.Y_DOWN.value))),
                storage_profile_id=str(payload.get("storage_profile_id", self.profiles.profile.id)),
                project_id=ProjectId(str(stable_id)),
            )
            project = self._create_project(command)
            self._ensure_server_workspace(project)
            return _project_dict(project)
        except ApplicationNotFoundError as exc:
            raise NotFoundError(str(exc)) from exc
        except (ApplicationConcurrencyError, ApplicationConflictError) as exc:
            raise ConflictError(str(exc)) from exc
        except ApplicationAuthorizationError as exc:
            raise ForbiddenError(str(exc)) from exc
        except (StorageCapabilityError, ValueError, TypeError) as exc:
            raise ValidationError(str(exc)) from exc

    @guarded_storage
    def _ensure_server_workspace(self, project: Any) -> None:
        files = self.workspace_files
        if files is None:
            return
        if not self.source_root or not self.derived_root:
            raise ValidationError(
                "В server.toml не заданы storage.source_root и storage.derived_root"
            )
        if files.registry.get_project(str(project.id)) is None:
            from kraken_manager.workspace import WorkspaceValidationError

            try:
                files.create_server_project(
                    project_id=str(project.id),
                    project_name=project.name,
                    source_root=self.source_root,
                    derived_root=self.derived_root,
                )
            except (WorkspaceValidationError, FileExistsError, OSError) as exc:
                raise ValidationError(str(exc)) from exc
        binding = files.registry.get_project(str(project.id))
        if binding is not None:
            self.projections.record_project_directories(
                str(project.id),
                source_directory=binding.source_project_dir,
                derived_directory=binding.derived_project_dir,
            )

    def project_workspace(self, project_id: str) -> dict[str, Any] | None:
        files = self.workspace_files
        if files is None:
            return None
        from kraken_manager.workspace import project_workspace_to_dict

        binding = files.registry.get_project(project_id)
        return None if binding is None else project_workspace_to_dict(binding)

    def get_project(self, project_id: str) -> dict[str, Any]:
        try:
            project = self.projections.get_project(_project_id(project_id))
        except (TypeError, ValueError) as exc:
            raise ValidationError("Invalid project id") from exc
        if project is None:
            raise NotFoundError(project_id)
        return _project_dict(project)

    def _application_context(self, context: CommandContext) -> ApplicationCommandContext:
        return ApplicationCommandContext(
            actor=self._actor(context.actor_id),
            idempotency_key=context.idempotency_key,
            gitlab_identity_verified=True,
        )

    @staticmethod
    def _translate_lifecycle_error(exc: Exception) -> None:
        if isinstance(exc, ApplicationNotFoundError):
            raise NotFoundError(str(exc)) from exc
        if isinstance(exc, (ApplicationConcurrencyError, ApplicationConflictError)):
            raise ConflictError(str(exc)) from exc
        if isinstance(exc, ApplicationAuthorizationError):
            raise ForbiddenError(str(exc)) from exc
        if isinstance(exc, (StorageCapabilityError, ValueError, TypeError)):
            raise ValidationError(str(exc)) from exc
        raise exc

    def rename_project(self, project_id: str, name: str, context: CommandContext) -> dict[str, Any]:
        try:
            return _project_dict(
                self._rename_project(
                    RenameProjectCommand(
                        context=self._application_context(context),
                        project_id=_project_id(project_id),
                        name=name,
                        expected_revision=self._require_revision(context),
                    )
                )
            )
        except Exception as exc:
            self._translate_lifecycle_error(exc)
            raise AssertionError("unreachable")
