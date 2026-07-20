"""Production service facade backed by the clean-architecture application API."""

from __future__ import annotations

import base64
import json
from datetime import UTC, datetime
from typing import Any, Mapping
from uuid import NAMESPACE_URL, uuid5

from kraken_manager.application.dto import (
    CommandContext as ApplicationCommandContext,
    CreateLayerCommand,
    CreateProjectCommand,
    CreateRepresentationCommand,
    StorageBackendKind,
    StorageScope,
)
from kraken_manager.application.errors import (
    AuthorizationError as ApplicationAuthorizationError,
    ConcurrencyError as ApplicationConcurrencyError,
    ConflictError as ApplicationConflictError,
    NotFoundError as ApplicationNotFoundError,
    StorageCapabilityError,
)
from kraken_manager.application.ports import StorageCapabilities, StorageProfile
from kraken_manager.application.use_cases import CreateLayerHandler, CreateProjectHandler, CreateRepresentationHandler
from kraken_manager.domain.common import LayerId, PrincipalId, ProjectId, validate_uuid
from kraken_manager.domain.project import GridOrientation, Layer, LayerType, Project, Representation, RepresentationKind
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
                    expected_layer_revision=self._require_revision(context),
                    note=str(payload.get("note", "")),
                    source=None if payload.get("source") is None else str(payload["source"]),
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
