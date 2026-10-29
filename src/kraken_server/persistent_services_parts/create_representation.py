"""persistent_services: create representation."""
from __future__ import annotations
from collections.abc import Mapping
from typing import Any
from kraken_manager.application.dto import (
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
from kraken_manager.domain.common import (
    RepresentationId,
)
from kraken_manager.domain.project import (
    RepresentationKind,
    RepresentationPurpose,
)
from kraken_server.services import CommandContext, ConflictError, ForbiddenError, NotFoundError, ValidationError

from kraken_server.persistent_services_parts.layer_source_scan import (
    _layer_id,
    _project_id,
    _representation_dict,
)


class CreateRepresentationMixin:
    def _authorize_representation_import(self, project_id, context):
        from kraken_manager.application.authorization import AuthorizationPolicy
        from kraken_manager.domain.identity import Permission

        project = self.projections.get_project(_project_id(project_id))
        if project is None:
            raise NotFoundError(project_id)
        actor = self._actor(context.actor_id)
        AuthorizationPolicy().require(
            principal=actor, storage=self.profiles.get(project.storage_profile),
            permission=Permission.MANAGE_STRUCTURE,
            roles=self.identities.roles_for(project.id, actor.id), gitlab_identity_verified=True,
        )
        if project.state.value == "archived":
            raise ConflictError("Archived project is read-only")
        return project, actor

    def plan_import_directory(self, project_id, payload, context):
        from kraken_core.safe_files import ensure_regular_directory, open_regular_read
        from kraken_manager.application.imports import ImportMappingMode, ImportPlanner, ImportSource
        from kraken_manager.infrastructure.filesystem._codec import encode_model
        import hashlib

        try:
            project, _actor = self._authorize_representation_import(project_id, context)
            root = ensure_regular_directory(self.resolve_storage_path(payload["directory"]))
            sources = []
            for path in sorted(root.iterdir(), key=lambda item: item.name.casefold()):
                if not path.is_file():
                    continue
                digest = hashlib.sha256()
                with open_regular_read(path, root=root) as stream:
                    while chunk := stream.read(1024 * 1024):
                        digest.update(chunk)
                sources.append(ImportSource(str(path), path.name, path.stat().st_size, digest.hexdigest()))
            return encode_model(ImportPlanner().plan(
                width=project.width, height=project.height, sources=tuple(sources),
                mode=ImportMappingMode(payload.get("mode", "xy_filename")), regex=payload.get("regex"),
            ))
        except Exception as exc:
            self._translate_lifecycle_error(exc)
            raise AssertionError("unreachable")

    def commit_managed_import(self, project_id, layer_id, representation_id, payload, context):
        from dataclasses import replace
        from kraken_manager.application.imports import ImportPlan
        from kraken_manager.application.managed_import import commit_managed_import
        from kraken_manager.infrastructure.import_content import FilesystemImportContentReader
        from kraken_manager.infrastructure.filesystem._codec import decode_model, encode_model
        from kraken_manager.infrastructure.blob.filesystem import BlobIntegrityError
        from kraken_server.persistent_services_parts.layer_source_scan import SystemClock

        try:
            project, actor = self._authorize_representation_import(project_id, context)
            layer = self.projections.get_layer(_layer_id(layer_id))
            representation = self.projections.get_representation(RepresentationId(str(representation_id)))
            if layer is None or representation is None or layer.project_id != project.id or representation.layer_id != layer.id:
                raise NotFoundError("Representation was not found in the project and layer")
            plan = decode_model(ImportPlan, payload["plan"])
            plan = replace(plan, items=tuple(
                replace(item, source=replace(item.source, source_key=str(self.resolve_storage_path(item.source.source_key))))
                for item in plan.items
            ))
            versions = commit_managed_import(
                principal=actor, project=project, layer=layer, representation=representation,
                plan=plan, idempotency_key=context.idempotency_key, projections=self.projections,
                events=self.events, uow_factory=self.uow_factory, profiles=self.profiles,
                clock=SystemClock(), content_reader=FilesystemImportContentReader(), gitlab_identity_verified=True,
            )
            return {"versions": [encode_model(version) for version in versions]}
        except (KeyError, BlobIntegrityError, FileNotFoundError) as exc:
            raise ValidationError(str(exc)) from exc
        except Exception as exc:
            self._translate_lifecycle_error(exc)
            raise AssertionError("unreachable")

    def list_representations(
        self, project_id: str, layer_id: str, *, include_archived: bool = False, as_of=None
    ) -> list[dict[str, Any]]:
        self.get_project(project_id)
        try:
            layer = self.projections.get_layer(_layer_id(layer_id), as_of=as_of)
        except (TypeError, ValueError) as exc:
            raise ValidationError("Invalid layer id") from exc
        if layer is None or str(layer.project_id) != project_id:
            raise NotFoundError(layer_id)
        return [
            _representation_dict(representation)
            for representation in self.projections.list_representations(
                layer.id, include_archived=include_archived, as_of=as_of
            )
        ]

    def create_representation(
        self,
        project_id: str,
        layer_id: str,
        payload: Mapping[str, Any],
        context: CommandContext,
    ) -> dict[str, Any]:
        actor = self._actor(context.actor_id)
        try:
            representation = self._create_representation(
                CreateRepresentationCommand(
                    context=ApplicationCommandContext(
                        actor=actor,
                        idempotency_key=context.idempotency_key,
                        gitlab_identity_verified=True,
                    ),
                    project_id=_project_id(project_id),
                    layer_id=_layer_id(layer_id),
                    name=str(payload.get("name", "")),
                    kind=RepresentationKind(str(payload.get("kind", ""))),
                    purpose=RepresentationPurpose(
                        str(
                            payload.get(
                                "purpose",
                                "vector" if str(payload.get("kind", "")) == "vector" else "source",
                            )
                        )
                    ),
                    expected_layer_revision=self._require_revision(context),
                    note=str(payload.get("note", "")),
                    source=None if payload.get("source") is None else str(payload["source"]),
                    source_image_representation_id=(
                        None
                        if payload.get("source_image_representation_id") is None
                        else RepresentationId(str(payload["source_image_representation_id"]))
                    ),
                    active=bool(payload.get("active", False)),
                )
            )
            return _representation_dict(representation)
        except ApplicationNotFoundError as exc:
            raise NotFoundError(str(exc)) from exc
        except (ApplicationConcurrencyError, ApplicationConflictError) as exc:
            raise ConflictError(str(exc)) from exc
        except ApplicationAuthorizationError as exc:
            raise ForbiddenError(str(exc)) from exc
        except (StorageCapabilityError, ValueError, TypeError) as exc:
            raise ValidationError(str(exc)) from exc
