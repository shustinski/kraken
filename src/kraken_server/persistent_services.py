"""Production service facade backed by the clean-architecture application API."""

from __future__ import annotations

import base64
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from kraken_manager.application.acl import AssignProjectRoleHandler, RevokeProjectRoleHandler
from kraken_manager.application.dto import (
    ActivateRepresentationCommand,
    ArchiveLayerCommand,
    ArchiveProjectCommand,
    ArchiveRepresentationCommand,
    AssignProjectRoleCommand,
    CreateLayerCommand,
    CreateProjectCommand,
    CreateRepresentationCommand,
    DeactivateRepresentationCommand,
    RenameLayerCommand,
    RenameProjectCommand,
    RenameRepresentationCommand,
    ReorderLayerCommand,
    ReorderLayersCommand,
    RestoreProjectCommand,
    RevokeProjectRoleCommand,
    StorageBackendKind,
    StorageScope,
    UpdateRepresentationNoteCommand,
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
from kraken_manager.application.lifecycle import (
    ArchiveLayerHandler,
    ArchiveProjectHandler,
    RenameLayerHandler,
    RenameProjectHandler,
    ReorderLayerHandler,
    ReorderLayersHandler,
    RestoreProjectHandler,
)
from kraken_manager.application.ports import StorageCapabilities, StorageProfile
from kraken_manager.application.representation_lifecycle import (
    ActivateRepresentationHandler,
    ArchiveRepresentationHandler,
    DeactivateRepresentationHandler,
    RenameRepresentationHandler,
    UpdateRepresentationNoteHandler,
)
from kraken_manager.application.use_cases import CreateLayerHandler, CreateProjectHandler, CreateRepresentationHandler
from kraken_manager.domain.common import LayerId, PrincipalId, ProjectId, RepresentationId, validate_uuid
from kraken_manager.domain.identity import ProjectRole
from kraken_manager.domain.project import (
    GridOrientation,
    Layer,
    LayerType,
    Project,
    Representation,
    RepresentationKind,
    RepresentationPurpose,
)
from kraken_manager.infrastructure.postgres import PostgresEventStore, PostgresIdentityAclStore, PostgresProjectionStore

from .services import CommandContext, ConflictError, ForbiddenError, NotFoundError, ValidationError


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)


class ServerStorageProfiles:
    """Server-owned profile catalog; persistence/configuration may replace it."""

    def __init__(self, *, profile_id: str = "server-postgres", max_frames: int | None = 1_000_000) -> None:
        self.profile = StorageProfile(
            id=profile_id,
            name="Kraken Server PostgreSQL",
            metadata_backend=StorageBackendKind.POSTGRESQL,
            blob_backend="filesystem",
            scope=StorageScope.SHARED,
            capabilities=StorageCapabilities(
                multi_writer=True,
                transactions=True,
                snapshots=True,
                streaming=True,
                external_references=True,
                max_frames=max_frames,
            ),
        )

    def get(self, profile_id: str) -> StorageProfile | None:
        return self.profile if profile_id == self.profile.id else None

    def list(self) -> tuple[StorageProfile, ...]:
        return (self.profile,)


def _project_dict(project: Project) -> dict[str, Any]:
    return {
        "project_id": str(project.id),
        "name": project.name,
        "width": project.width,
        "height": project.height,
        "orientation": project.orientation.value,
        "storage_profile": project.storage_profile,
        "state": project.state.value,
        "revision": project.revision,
        "created_at": project.created_at.isoformat(),
    }


def _layer_dict(layer: Layer) -> dict[str, Any]:
    return {
        "layer_id": str(layer.id),
        "project_id": str(layer.project_id),
        "name": layer.name,
        "type": layer.type.value,
        "order": layer.order,
        "state": layer.state.value,
        "revision": layer.revision,
        "created_at": layer.created_at.isoformat(),
    }


def _representation_dict(representation: Representation) -> dict[str, Any]:
    return {
        "representation_id": str(representation.id),
        "project_id": str(representation.project_id),
        "layer_id": str(representation.layer_id),
        "name": representation.name,
        "kind": representation.kind.value,
        "purpose": representation.purpose.value,
        "source_image_representation_id": (
            None
            if representation.source_image_representation_id is None
            else str(representation.source_image_representation_id)
        ),
        "note": representation.note,
        "source": representation.source,
        "active": representation.active,
        "state": representation.state.value,
        "revision": representation.revision,
        "created_at": representation.created_at.isoformat(),
    }


def _project_id(value: str) -> ProjectId:
    return ProjectId(validate_uuid(value, field="project_id"))


def _layer_id(value: str) -> LayerId:
    return LayerId(validate_uuid(value, field="layer_id"))


class PostgresServerServices:
    """Thin transport facade; mutations execute application command handlers."""

    def __init__(self, engine: Any, uow_factory: Any, *, profiles: ServerStorageProfiles | None = None) -> None:
        self.engine = engine
        self.uow_factory = uow_factory
        self.profiles = profiles or ServerStorageProfiles()
        self.projections = PostgresProjectionStore(engine)
        self.events = PostgresEventStore(engine)
        self.identities = PostgresIdentityAclStore(engine)
        self._create_project = CreateProjectHandler(uow_factory, self.profiles, SystemClock())
        self._create_layer = CreateLayerHandler(uow_factory, self.profiles, SystemClock())
        self._create_representation = CreateRepresentationHandler(uow_factory, self.profiles, SystemClock())
        self._rename_project = RenameProjectHandler(uow_factory, self.profiles, SystemClock())
        self._archive_project = ArchiveProjectHandler(uow_factory, self.profiles, SystemClock())
        self._restore_project = RestoreProjectHandler(uow_factory, self.profiles, SystemClock())
        self._rename_layer = RenameLayerHandler(uow_factory, self.profiles, SystemClock())
        self._reorder_layer = ReorderLayerHandler(uow_factory, self.profiles, SystemClock())
        self._reorder_layers = ReorderLayersHandler(uow_factory, self.profiles, SystemClock())
        self._archive_layer = ArchiveLayerHandler(uow_factory, self.profiles, SystemClock())
        self._assign_project_role = AssignProjectRoleHandler(uow_factory, self.profiles, SystemClock())
        self._revoke_project_role = RevokeProjectRoleHandler(uow_factory, self.profiles, SystemClock())
        self._activate_representation = ActivateRepresentationHandler(uow_factory, self.profiles, SystemClock())
        self._deactivate_representation = DeactivateRepresentationHandler(
            uow_factory,
            self.profiles,
            SystemClock(),
        )
        self._archive_representation = ArchiveRepresentationHandler(uow_factory, self.profiles, SystemClock())
        self._rename_representation = RenameRepresentationHandler(uow_factory, self.profiles, SystemClock())
        self._update_representation_note = UpdateRepresentationNoteHandler(uow_factory, self.profiles, SystemClock())

    def health(self) -> dict[str, Any]:
        try:
            from sqlalchemy import text

            with self.engine.connect() as connection:
                connection.execute(text("SELECT 1")).scalar_one()
        except Exception as exc:
            return {"status": "degraded", "metadata": "unavailable", "detail": str(exc)[:500]}
        return {"status": "ok", "metadata": "postgresql", "api_version": "v1"}

    def list_projects(self) -> list[dict[str, Any]]:
        return [_project_dict(project) for project in self.projections.list_projects()]

    def list_principals(self, *, include_inactive: bool = False) -> list[dict[str, Any]]:
        return [
            {
                "principal_id": str(principal.id),
                "provider": principal.provider.value,
                "subject": principal.subject,
                "issuer": principal.issuer,
                "display_name": principal.display_name,
                "email": principal.email,
                "active": principal.active,
                "system_roles": sorted(
                    role.value for role in principal.system_roles
                ),
            }
            for principal in self.identities.list(
                include_inactive=include_inactive
            )
        ]

    def _actor(self, actor_id: str) -> Any:
        try:
            principal_id = PrincipalId(validate_uuid(actor_id, field="actor_id"))
            actor = self.identities.get(principal_id)
        except (TypeError, ValueError) as exc:
            raise ValidationError("Authenticated principal id is invalid") from exc
        if actor is None or not actor.active:
            raise ValidationError("Authenticated principal is unknown or inactive")
        return actor

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
            return _project_dict(self._create_project(command))
        except ApplicationNotFoundError as exc:
            raise NotFoundError(str(exc)) from exc
        except (ApplicationConcurrencyError, ApplicationConflictError) as exc:
            raise ConflictError(str(exc)) from exc
        except ApplicationAuthorizationError as exc:
            raise ForbiddenError(str(exc)) from exc
        except (StorageCapabilityError, ValueError, TypeError) as exc:
            raise ValidationError(str(exc)) from exc

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

    def list_layers(self, project_id: str) -> list[dict[str, Any]]:
        self.get_project(project_id)
        return [_layer_dict(layer) for layer in self.projections.list_layers(_project_id(project_id))]

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
                    role=ProjectRole(role),
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
                    role=ProjectRole(role),
                    expected_revision=self._require_revision(context),
                )
            )
            return self.project_roles(project_id, principal_id)
        except Exception as exc:
            self._translate_lifecycle_error(exc)
            raise AssertionError("unreachable")

    def list_representations(self, project_id: str, layer_id: str) -> list[dict[str, Any]]:
        self.get_project(project_id)
        try:
            layer = self.projections.get_layer(_layer_id(layer_id))
        except (TypeError, ValueError) as exc:
            raise ValidationError("Invalid layer id") from exc
        if layer is None or str(layer.project_id) != project_id:
            raise NotFoundError(layer_id)
        return [
            _representation_dict(representation)
            for representation in self.projections.list_representations(layer.id)
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

    def update_representation(
        self,
        project_id: str,
        layer_id: str,
        representation_id: str,
        payload: Mapping[str, Any],
        context: CommandContext,
    ) -> dict[str, Any]:
        operations = [key for key in ("name", "note", "active", "archive") if key in payload]
        if len(operations) != 1:
            raise ValidationError("Exactly one representation operation is required")
        try:
            common = {
                "context": self._application_context(context),
                "project_id": _project_id(project_id),
                "layer_id": _layer_id(layer_id),
                "representation_id": RepresentationId(
                    validate_uuid(representation_id, field="representation_id")
                ),
                "expected_layer_revision": self._require_revision(context),
                "expected_representation_revision": int(
                    payload.get("expected_representation_revision", -1)
                ),
            }
            operation = operations[0]
            if operation == "name":
                value = self._rename_representation(
                    RenameRepresentationCommand(name=str(payload["name"]), **common)
                )
            elif operation == "note":
                value = self._update_representation_note(
                    UpdateRepresentationNoteCommand(note=str(payload["note"]), **common)
                )
            elif operation == "active":
                value = (
                    self._activate_representation(
                        ActivateRepresentationCommand(**common)
                    )
                    if bool(payload["active"])
                    else self._deactivate_representation(
                        DeactivateRepresentationCommand(**common)
                    )
                )
            else:
                if not bool(payload["archive"]):
                    raise ValueError("archive must be true")
                value = self._archive_representation(ArchiveRepresentationCommand(**common))
            return _representation_dict(value)
        except Exception as exc:
            self._translate_lifecycle_error(exc)
            raise AssertionError("unreachable")

    def matrix_viewport(
        self,
        project_id: str,
        *,
        layer_id: str,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        lod: int,
    ) -> dict[str, Any]:
        project = self.get_project(project_id)
        if not (1 <= x1 <= x2 <= project["width"] and 1 <= y1 <= y2 <= project["height"]):
            raise ValidationError("Viewport is outside the project grid")
        if lod < 0 or lod > 24:
            raise ValidationError("LOD must be between 0 and 24")
        # Empty frames are never materialized.  Frame status projections can be
        # added to this sparse response without changing its public envelope.
        return {
            "project_id": project_id,
            "layer_id": layer_id,
            "bounds": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "lod": lod,
            "revision": str(project["revision"]),
            "cells": [],
            "aggregates": [],
        }

    @staticmethod
    def _decode_cursor(cursor: str | None) -> int:
        if not cursor:
            return 0
        try:
            raw = base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4))
            value = int(json.loads(raw)["position"])
        except Exception as exc:
            raise ValidationError("Invalid history cursor") from exc
        if value < 0:
            raise ValidationError("Invalid history cursor")
        return value

    @staticmethod
    def _encode_cursor(position: int) -> str:
        raw = json.dumps({"position": position}, separators=(",", ":")).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    def history(self, project_id: str, *, cursor: str | None, limit: int) -> dict[str, Any]:
        self.get_project(project_id)
        if limit < 1 or limit > 500:
            raise ValidationError("History limit must be between 1 and 500")
        after = self._decode_cursor(cursor)
        rows = self.events.list_project_events(_project_id(project_id), after_position=after, limit=limit + 1)
        has_more = len(rows) > limit
        page = rows[:limit]
        items = [
            {
                "position": position,
                "event_id": event.event_id,
                "stream_id": event.stream_id,
                "revision": event.revision,
                "event_type": event.event_type,
                "schema_version": event.schema_version,
                "recorded_at": event.recorded_at.isoformat(),
                "effective_at": None if event.effective_at is None else event.effective_at.isoformat(),
                "actor": {
                    "principal_id": str(event.actor.principal_id),
                    "provider": event.actor.provider.value,
                    "subject": event.actor.subject,
                    "display_name": event.actor.display_name,
                },
                "performer_id": None if event.performer_id is None else str(event.performer_id),
                "payload": dict(event.payload),
            }
            for position, event in page
        ]
        next_cursor = self._encode_cursor(page[-1][0]) if has_more and page else None
        return {"items": items, "next_cursor": next_cursor}


__all__ = ["PostgresServerServices", "ServerStorageProfiles", "SystemClock"]
