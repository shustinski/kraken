"""persistent_services: create external layer."""
from __future__ import annotations
from collections.abc import Mapping
from typing import Any
from kraken_manager.application.authorization import AuthorizationPolicy
from kraken_manager.application.dto import (
    ArchiveProjectCommand,
    CreateLayerCommand,
    RestoreProjectCommand,
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
from kraken_manager.domain.identity import Permission
from kraken_manager.domain.project import (
    LayerType,
)
from kraken_server.services import CommandContext, ConflictError, ForbiddenError, NotFoundError, ValidationError

from kraken_server.persistent_services_parts.layer_source_scan import (
    _layer_dict,
    _layer_id,
    _layer_source_scan,
    _project_dict,
    _project_id,
)


class CreateExternalLayerMixin:
    def archive_project(self, project_id: str, context: CommandContext) -> dict[str, Any]:
        try:
            return _project_dict(
                self._archive_project(
                    ArchiveProjectCommand(
                        context=self._application_context(context),
                        project_id=_project_id(project_id),
                        expected_revision=self._require_revision(context),
                    )
                )
            )
        except Exception as exc:
            self._translate_lifecycle_error(exc)
            raise AssertionError("unreachable")

    def restore_project(self, project_id: str, context: CommandContext) -> dict[str, Any]:
        try:
            return _project_dict(
                self._restore_project(
                    RestoreProjectCommand(
                        context=self._application_context(context),
                        project_id=_project_id(project_id),
                        expected_revision=self._require_revision(context),
                    )
                )
            )
        except Exception as exc:
            self._translate_lifecycle_error(exc)
            raise AssertionError("unreachable")

    @staticmethod
    def _require_revision(context: CommandContext) -> int:
        if context.expected_revision is None:
            raise ValidationError("If-Match is required for this command")
        return context.expected_revision

    def list_layers(
        self, project_id: str, *, include_archived: bool = False, as_of=None
    ) -> list[dict[str, Any]]:
        self.get_project(project_id)
        return [
            _layer_dict(layer)
            for layer in self.projections.list_layers(
                _project_id(project_id), include_archived=include_archived, as_of=as_of
            )
        ]

    def create_layer(
        self, project_id: str, payload: Mapping[str, Any], context: CommandContext
    ) -> dict[str, Any]:
        actor = self._actor(context.actor_id)
        try:
            layer = self._create_layer(
                CreateLayerCommand(
                    context=ApplicationCommandContext(
                        actor=actor,
                        idempotency_key=context.idempotency_key,
                        gitlab_identity_verified=True,
                    ),
                    project_id=_project_id(project_id),
                    name=str(payload.get("name", "")),
                    type=LayerType(str(payload.get("type", ""))),
                    order=int(payload.get("order", 0)),
                    expected_project_revision=self._require_revision(context),
                        **({} if payload.get("layer_id") is None else {"layer_id": _layer_id(str(payload["layer_id"]))}),
                )
            )
            return _layer_dict(layer)
        except ApplicationNotFoundError as exc:
            raise NotFoundError(str(exc)) from exc
        except (ApplicationConcurrencyError, ApplicationConflictError) as exc:
            raise ConflictError(str(exc)) from exc
        except ApplicationAuthorizationError as exc:
            raise ForbiddenError(str(exc)) from exc
        except (StorageCapabilityError, ValueError, TypeError) as exc:
            raise ValidationError(str(exc)) from exc

    def layer_file_binding(self, project_id: str, layer_id: str) -> dict[str, Any] | None:
        files = self.workspace_files
        if files is None:
            return None
        from kraken_manager.workspace import layer_binding_to_dict

        binding = files.registry.get_layer(project_id, layer_id)
        return None if binding is None else layer_binding_to_dict(binding)

    def scan_layer_source(
        self, project_id: str, directory: str, context: CommandContext
    ) -> dict[str, Any]:
        from dataclasses import asdict

        project = self.projections.get_project(_project_id(project_id))
        if project is None:
            raise NotFoundError(project_id)
        actor = self._actor(context.actor_id)
        try:
            AuthorizationPolicy().require(
                principal=actor, storage=self.profiles.get(project.storage_profile),
                permission=Permission.MANAGE_STRUCTURE,
                roles=self.identities.roles_for(project.id, actor.id), gitlab_identity_verified=True,
            )
            if self.workspace_files is None:
                raise ValidationError("Server workspace storage is not configured")
            return asdict(self.workspace_files.scan(directory, maximum_frames=project.frame_count))
        except Exception as exc:
            self._translate_lifecycle_error(exc)
            raise AssertionError("unreachable")

    def create_external_layer(
        self, project_id: str, payload: Mapping[str, Any], context: CommandContext
    ) -> dict[str, Any]:
        copied: list[Any] = []

        def bind(files: Any, layer_id: str, name: str) -> Any:
            workspace = files.registry.get_project(project_id)
            if workspace is None:
                raise ValidationError("У проекта нет двухдискового хранилища")
            binding, created = files.place_server_layer(
                project=workspace,
                layer_id=layer_id,
                layer_name=name,
                image_directory=str(payload.get("image_directory", "")),
                ssc_directory=payload.get("ssc_directory") or None,
                prv_directory=payload.get("prv_directory") or None,
                maximum_frames=self._project_frame_count(project_id),
            )
            copied.extend(created)
            return binding

        return self._attach_workspace_layer(
            project_id,
            payload,
            context,
            bind=bind,
            managed=False,
            copied_directories=copied,
        )

    def import_workspace_layer(
        self, project_id: str, payload: Mapping[str, Any], context: CommandContext
    ) -> dict[str, Any]:
        from kraken_manager.workspace import ImageConversionSettings

        def bind(files: Any, layer_id: str, name: str) -> Any:
            workspace = files.registry.get_project(project_id)
            if workspace is None:
                raise ValidationError("У проекта нет двухдискового хранилища")
            scan_payload = payload.get("scan")
            if not isinstance(scan_payload, Mapping):
                raise ValidationError("Результат сканирования слоя не передан")
            conversion_payload = payload.get("conversion")
            conversion = ImageConversionSettings(
                **(dict(conversion_payload) if isinstance(conversion_payload, Mapping) else {})
            )
            return files.import_layer(
                project=workspace,
                layer_id=layer_id,
                layer_name=name,
                scan=_layer_source_scan(scan_payload),
                conversion=conversion,
            )

        return self._attach_workspace_layer(project_id, payload, context, bind=bind, managed=True)

    def _project_frame_count(self, project_id: str) -> int:
        project = self.projections.get_project(_project_id(project_id))
        if project is None:
            raise NotFoundError(f"Project {project_id} was not found")
        return int(project.width) * int(project.height)
