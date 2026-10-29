"""persistent_services: attach workspace layer."""
from __future__ import annotations
from kraken_server.project_deletion import guarded_storage
from collections.abc import Mapping
from typing import Any
from uuid import uuid4
from kraken_manager.application.dto import (
    CreateLayerCommand,
    CreateRepresentationCommand,
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
from kraken_manager.domain.project import (
    LayerType,
    RepresentationKind,
    RepresentationPurpose,
)
from kraken_server.services import CommandContext, ConflictError, ForbiddenError, NotFoundError, ValidationError

from kraken_server.persistent_services_parts.layer_source_scan import (
    _layer_dict,
    _layer_id,
    _project_id,
    _representation_dict,
)


class AttachWorkspaceLayerMixin:
    @guarded_storage
    def _attach_workspace_layer(
        self,
        project_id: str,
        payload: Mapping[str, Any],
        context: CommandContext,
        *,
        bind: Any,
        managed: bool,
        copied_directories: list[Any] | None = None,
    ) -> dict[str, Any]:
        from kraken_manager.workspace import WorkspaceValidationError, layer_binding_to_dict

        project = self.projections.get_project(_project_id(project_id))
        if project is None:
            raise NotFoundError(f"Project {project_id} was not found")
        files = self.workspace_files
        if files is None:
            raise ValidationError("На сервере не настроено файловое хранилище")
        workspace = files.registry.get_project(project_id)
        if workspace is None:
            raise ValidationError("У проекта нет двухдискового хранилища")
        prior = self.events.find_by_idempotency_key(_project_id(project_id), context.idempotency_key)
        created = next((event for event in reversed(prior) if event.event_type == "LayerCreated"), None)
        replay = created is not None
        layer_id = str(created.payload["layer_id"]) if created is not None else str(uuid4())
        name = str(payload.get("name", ""))
        try:
            binding = files.registry.get_layer(project_id, layer_id) if replay else bind(files, layer_id, name)
            if binding is None:
                raise WorkspaceValidationError("Existing layer storage binding is missing")
        except (WorkspaceValidationError, FileExistsError, OSError, ValueError) as exc:
            raise ValidationError(str(exc)) from exc
        actor = self._actor(context.actor_id)
        layer = None
        try:
            layer = self._create_layer(
                CreateLayerCommand(
                    context=ApplicationCommandContext(
                        actor=actor,
                        idempotency_key=context.idempotency_key,
                        gitlab_identity_verified=True,
                    ),
                    project_id=_project_id(project_id),
                    name=name,
                    type=LayerType(str(payload.get("type", ""))),
                    order=int(payload.get("order", 0)),
                    expected_project_revision=project.revision,
                    layer_id=_layer_id(layer_id),
                )
            )
            representation = self._create_representation(
                CreateRepresentationCommand(
                    context=ApplicationCommandContext(
                        actor=actor,
                        idempotency_key=f"{context.idempotency_key}:representation",
                        gitlab_identity_verified=True,
                    ),
                    project_id=_project_id(project_id),
                    layer_id=_layer_id(str(layer.id)),
                    name="Исходные изображения",
                    kind=RepresentationKind.IMAGE,
                    purpose=RepresentationPurpose.SOURCE,
                    expected_layer_revision=layer.revision,
                    source=binding.image_directory,
                    active=True,
                )
            )
        except Exception as exc:
            if not replay:
                self._discard_workspace_layer(
                    files,
                    workspace,
                    binding,
                    project_id,
                    layer_id,
                    context,
                    layer,
                    managed=managed,
                    copied_directories=copied_directories or [],
                )
            if isinstance(exc, ApplicationNotFoundError):
                raise NotFoundError(str(exc)) from exc
            if isinstance(exc, (ApplicationConcurrencyError, ApplicationConflictError)):
                raise ConflictError(str(exc)) from exc
            if isinstance(exc, ApplicationAuthorizationError):
                raise ForbiddenError(str(exc)) from exc
            if isinstance(exc, (StorageCapabilityError, ValueError, TypeError)):
                raise ValidationError(str(exc)) from exc
            raise
        return {
            "layer": _layer_dict(layer),
            "representation": _representation_dict(representation),
            "binding": layer_binding_to_dict(binding),
        }
