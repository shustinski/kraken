"""persistent_services: discard workspace layer."""
from __future__ import annotations
from collections.abc import Mapping
from typing import Any
from kraken_manager.application.dto import (
    ArchiveLayerCommand,
    AssignProjectRoleCommand,
    RenameLayerCommand,
    ReorderLayerCommand,
    ReorderLayersCommand,
    RevokeProjectRoleCommand,
)
from kraken_manager.domain.common import (
    PrincipalId,
    validate_uuid,
)
from kraken_manager.domain.identity import parse_project_role
from kraken_server.services import CommandContext, NotFoundError, ValidationError

from kraken_server.persistent_services_parts.layer_source_scan import (
    _layer_dict,
    _layer_id,
    _project_id,
)


class DiscardWorkspaceLayerMixin:
    def _discard_workspace_layer(
        self,
        files: Any,
        workspace: Any,
        binding: Any,
        project_id: str,
        layer_id: str,
        context: CommandContext,
        layer: Any,
        *,
        managed: bool,
        copied_directories: list[Any] | None = None,
    ) -> None:
        if layer is not None:
            try:
                self.archive_layer(
                    project_id,
                    str(layer.id),
                    CommandContext(
                        context.actor_id,
                        f"{context.idempotency_key}:compensate",
                        layer.revision,
                    ),
                )
            except Exception:
                pass
        try:
            if managed:
                files.remove_managed_layer_layout(binding, workspace)
            if copied_directories:
                files.remove_placed_directories(workspace, copied_directories)
            files.registry.remove_layer(project_id, layer_id)
        except Exception:
            pass

    def rename_layer(
        self, project_id: str, layer_id: str, name: str, context: CommandContext
    ) -> dict[str, Any]:
        try:
            return _layer_dict(
                self._rename_layer(
                    RenameLayerCommand(
                        context=self._application_context(context),
                        project_id=_project_id(project_id),
                        layer_id=_layer_id(layer_id),
                        name=name,
                        expected_revision=self._require_revision(context),
                    )
                )
            )
        except Exception as exc:
            self._translate_lifecycle_error(exc)
            raise AssertionError("unreachable")

    def reorder_layer(
        self, project_id: str, layer_id: str, order: int, context: CommandContext
    ) -> dict[str, Any]:
        try:
            return _layer_dict(
                self._reorder_layer(
                    ReorderLayerCommand(
                        context=self._application_context(context),
                        project_id=_project_id(project_id),
                        layer_id=_layer_id(layer_id),
                        order=order,
                        expected_revision=self._require_revision(context),
                    )
                )
            )
        except Exception as exc:
            self._translate_lifecycle_error(exc)
            raise AssertionError("unreachable")

    def reorder_layers(
        self, project_id: str, payload: Mapping[str, Any], context: CommandContext
    ) -> list[dict[str, Any]]:
        raw_revisions = payload.get("expected_revisions", {})
        if not isinstance(raw_revisions, Mapping):
            raise ValidationError("expected_revisions must be an object")
        try:
            layers = self._reorder_layers(
                ReorderLayersCommand(
                    context=self._application_context(context),
                    project_id=_project_id(project_id),
                    layer_ids=tuple(_layer_id(str(value)) for value in payload.get("layer_ids", ())),
                    expected_revisions=tuple(
                        (_layer_id(str(identifier)), int(revision))
                        for identifier, revision in raw_revisions.items()
                    ),
                )
            )
            return [_layer_dict(layer) for layer in layers]
        except Exception as exc:
            self._translate_lifecycle_error(exc)
            raise AssertionError("unreachable")

    def archive_layer(
        self, project_id: str, layer_id: str, context: CommandContext
    ) -> dict[str, Any]:
        try:
            return _layer_dict(
                self._archive_layer(
                    ArchiveLayerCommand(
                        context=self._application_context(context),
                        project_id=_project_id(project_id),
                        layer_id=_layer_id(layer_id),
                        expected_revision=self._require_revision(context),
                    )
                )
            )
        except Exception as exc:
            self._translate_lifecycle_error(exc)
            raise AssertionError("unreachable")

    def project_roles(self, project_id: str, principal_id: str) -> dict[str, Any]:
        project = self.get_project(project_id)
        del project
        try:
            project_identifier = _project_id(project_id)
            principal_identifier = PrincipalId(validate_uuid(principal_id, field="principal_id"))
        except (TypeError, ValueError) as exc:
            raise ValidationError("Invalid project or principal id") from exc
        if self.identities.get(principal_identifier) is None:
            raise NotFoundError(principal_id)
        stream_id = f"acl:{project_identifier}:{principal_identifier}"
        return {
            "project_id": project_id,
            "principal_id": principal_id,
            "roles": sorted(
                role.value for role in self.identities.roles_for(project_identifier, principal_identifier)
            ),
            "revision": self.events.current_revision(stream_id),
        }

    def assign_project_role(
        self,
        project_id: str,
        principal_id: str,
        role: str,
        context: CommandContext,
    ) -> dict[str, Any]:
        try:
            self._assign_project_role(
                AssignProjectRoleCommand(
                    context=self._application_context(context),
                    project_id=_project_id(project_id),
                    principal_id=PrincipalId(validate_uuid(principal_id, field="principal_id")),
                    role=parse_project_role(role),
                    expected_revision=self._require_revision(context),
                )
            )
            return self.project_roles(project_id, principal_id)
        except Exception as exc:
            self._translate_lifecycle_error(exc)
            raise AssertionError("unreachable")

    def revoke_project_role(
        self,
        project_id: str,
        principal_id: str,
        role: str,
        context: CommandContext,
    ) -> dict[str, Any]:
        try:
            self._revoke_project_role(
                RevokeProjectRoleCommand(
                    context=self._application_context(context),
                    project_id=_project_id(project_id),
                    principal_id=PrincipalId(validate_uuid(principal_id, field="principal_id")),
                    role=parse_project_role(role),
                    expected_revision=self._require_revision(context),
                )
            )
            return self.project_roles(project_id, principal_id)
        except Exception as exc:
            self._translate_lifecycle_error(exc)
            raise AssertionError("unreachable")
