"""Transport-facing server service contract.

FastAPI depends on this narrow facade.  Production composition maps it to
application use cases; the in-memory implementation is intentionally only a
development/test profile.
"""

from __future__ import annotations

import base64
import json
import tempfile
import threading
from pathlib import Path
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol
from uuid import uuid4

from kraken_manager.domain.common import LayerId, PrincipalId, ProjectId, RepresentationId
from kraken_manager.domain.selection import FrameRowRange, FrameSelectionV1
from kraken_manager.domain.workflows import PluginJob, PluginJobState
from kraken_manager.infrastructure.filesystem._codec import encode_model


class ConflictError(RuntimeError):
    pass


class NotFoundError(KeyError):
    pass


class ValidationError(ValueError):
    pass


class ForbiddenError(PermissionError):
    pass


@dataclass(frozen=True, slots=True)
class CommandContext:
    actor_id: str
    idempotency_key: str
    expected_revision: int | None


class ServerServices(Protocol):
    def request_project_deletion(self, project_id: str, actor_id: str) -> dict[str, Any]: ...

    def project_deletion_requests(self, project_id: str) -> list[dict[str, Any]]: ...

    def health(self) -> dict[str, Any]: ...

    def list_projects(self, *, include_archived: bool = False) -> list[dict[str, Any]]: ...

    def list_principals(self, *, include_inactive: bool = False) -> list[dict[str, Any]]: ...

    def list_performers(self, *, include_archived: bool = False) -> list[dict[str, Any]]: ...

    def create_project(self, payload: Mapping[str, Any], context: CommandContext) -> dict[str, Any]: ...

    def get_project(self, project_id: str) -> dict[str, Any]: ...

    def rename_project(
        self, project_id: str, name: str, context: CommandContext
    ) -> dict[str, Any]: ...

    def archive_project(self, project_id: str, context: CommandContext) -> dict[str, Any]: ...

    def restore_project(self, project_id: str, context: CommandContext) -> dict[str, Any]: ...

    def list_layers(
        self, project_id: str, *, include_archived: bool = False
    ) -> list[dict[str, Any]]: ...

    def create_layer(
        self, project_id: str, payload: Mapping[str, Any], context: CommandContext
    ) -> dict[str, Any]: ...

    def rename_layer(
        self, project_id: str, layer_id: str, name: str, context: CommandContext
    ) -> dict[str, Any]: ...

    def reorder_layer(
        self, project_id: str, layer_id: str, order: int, context: CommandContext
    ) -> dict[str, Any]: ...

    def reorder_layers(
        self, project_id: str, payload: Mapping[str, Any], context: CommandContext
    ) -> list[dict[str, Any]]: ...

    def archive_layer(
        self, project_id: str, layer_id: str, context: CommandContext
    ) -> dict[str, Any]: ...

    def project_roles(self, project_id: str, principal_id: str) -> dict[str, Any]: ...

    def assign_project_role(
        self,
        project_id: str,
        principal_id: str,
        role: str,
        context: CommandContext,
    ) -> dict[str, Any]: ...

    def revoke_project_role(
        self,
        project_id: str,
        principal_id: str,
        role: str,
        context: CommandContext,
    ) -> dict[str, Any]: ...

    def list_representations(
        self, project_id: str, layer_id: str, *, include_archived: bool = False
    ) -> list[dict[str, Any]]: ...

    def create_representation(
        self,
        project_id: str,
        layer_id: str,
        payload: Mapping[str, Any],
        context: CommandContext,
    ) -> dict[str, Any]: ...

    def update_representation(
        self, project_id: str, layer_id: str, representation_id: str,
        payload: Mapping[str, Any], context: CommandContext,
    ) -> dict[str, Any]: ...

    def matrix_viewport(
        self, project_id: str, *, layer_id: str, representation_ids: Iterable[str],
        x1: int, y1: int, x2: int, y2: int, lod: int, include_missing: bool = True
    ) -> dict[str, Any]: ...

    def history(self, project_id: str, *, cursor: str | None, limit: int) -> dict[str, Any]: ...

    def statistics(
        self, project_id: str, *, start: datetime, end: datetime, timezone: Any
    ) -> dict[str, Any]: ...

    def publish_karakal_analysis(
        self, project_id: str, payload: Mapping[str, Any], context: CommandContext
    ) -> dict[str, Any]: ...

    def append_pipeline_event(
        self, project_id: str, payload: Mapping[str, Any], context: CommandContext
    ) -> dict[str, Any]: ...

    def list_artifact_series(
        self, project_id: str, *, layer_id: str | None = None,
        representation_id: str | None = None, frame_id: str | None = None,
        include_archived: bool = False,
    ) -> list[dict[str, Any]]: ...

    def list_artifact_versions(self, project_id: str, series_id: str) -> list[dict[str, Any]]: ...

    def artifact_stream_revision(self, project_id: str, series_id: str) -> int: ...

    def get_active_artifact_version(self, project_id: str, series_id: str) -> dict[str, Any] | None: ...

    def get_artifact_version(self, project_id: str, version_id: str) -> dict[str, Any]: ...

    def create_artifact_series(
        self, project_id: str, payload: Mapping[str, Any], context: CommandContext
    ) -> dict[str, Any]: ...

    def add_managed_artifact_version(
        self, project_id: str, series_id: str, payload: Mapping[str, Any],
        chunks: Iterable[bytes], context: CommandContext,
    ) -> dict[str, Any]: ...

    def prepare_managed_artifact_upload(
        self, project_id: str, series_id: str, payload: Mapping[str, Any], context: CommandContext
    ) -> dict[str, Any] | None: ...

    def register_managed_artifact_upload(
        self, project_id: str, series_id: str, payload: Mapping[str, Any], context: CommandContext
    ) -> dict[str, Any]: ...

    def add_external_artifact_version(
        self, project_id: str, series_id: str, payload: Mapping[str, Any],
        context: CommandContext,
    ) -> dict[str, Any]: ...

    def mutate_artifact_series(
        self, project_id: str, series_id: str, payload: Mapping[str, Any],
        context: CommandContext,
    ) -> dict[str, Any]: ...

    def list_notes(
        self, project_id: str, *, layer_id: str | None = None, frame_id: str | None = None
    ) -> list[dict[str, Any]]: ...

    def create_note(
        self, project_id: str, payload: Mapping[str, Any], context: CommandContext
    ) -> dict[str, Any]: ...

    def revise_note(
        self, project_id: str, note_id: str, payload: Mapping[str, Any], context: CommandContext
    ) -> dict[str, Any]: ...

    def iter_artifact_bytes(
        self, project_id: str, version_id: str, *, chunk_size: int = 1024 * 1024
    ) -> Iterator[bytes]: ...

    def list_review_batches(
        self, project_id: str, *, active_only: bool = False
    ) -> list[dict[str, Any]]: ...

    def create_review_batch(
        self, project_id: str, payload: Mapping[str, Any], context: CommandContext
    ) -> dict[str, Any]: ...

    def mutate_review_batch(
        self, project_id: str, batch_id: str, payload: Mapping[str, Any],
        context: CommandContext,
    ) -> dict[str, Any]: ...

    def export_review_package(
        self, project_id: str, batch_id: str, destination: str,
        context: CommandContext,
    ) -> dict[str, Any]: ...

    def inspect_review_return(
        self, project_id: str, batch_id: str, source: str,
        context: CommandContext,
    ) -> dict[str, Any]: ...

    def commit_review_return(
        self, project_id: str, batch_id: str, source: str,
        context: CommandContext,
    ) -> dict[str, Any]: ...

    def list_plugin_jobs(self, project_id: str | None = None) -> list[dict[str, Any]]: ...

    def submit_plugin_job(
        self, project_id: str, payload: Mapping[str, Any], context: CommandContext
    ) -> dict[str, Any]: ...

    def cancel_plugin_job(
        self, project_id: str, job_id: str, context: CommandContext
    ) -> dict[str, Any]: ...

    def fail_agent_job(self, job_id: str, error: str) -> dict[str, Any]: ...

    def import_agent_result(self, job_id: str, agent: Any) -> dict[str, Any]: ...


class InMemoryServerServices:
    def request_project_deletion(self, project_id: str, actor_id: str) -> dict[str, Any]:
        raise ConflictError("Deletion requests require a persistent PostgreSQL server")

    def project_deletion_requests(self, project_id: str) -> list[dict[str, Any]]:
        self.get_project(project_id)
        return []

    """Sparse, concurrency-safe development backend used by API tests."""

    def __init__(self) -> None:
        self._projects: dict[str, dict[str, Any]] = {}
        self._events: dict[str, list[dict[str, Any]]] = {}
        self._idempotency: dict[tuple[str, str], dict[str, Any]] = {}
        self._layers: dict[str, list[dict[str, Any]]] = {}
        self._representations: dict[str, list[dict[str, Any]]] = {}
        self._acl: dict[tuple[str, str], set[str]] = {}
        self._acl_revisions: dict[tuple[str, str], int] = {}
        self._granted_by: dict[tuple[str, str, str], str] = {}
        self._plugin_jobs: dict[str, dict[str, Any]] = {}
        self._workspaces: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

    def health(self) -> dict[str, Any]:
        return {"status": "ok", "metadata": "memory", "api_version": "v1"}

    def list_projects(self, *, include_archived: bool = False) -> list[dict[str, Any]]:
        with self._lock:
            return [
                dict(item)
                for item in sorted(
                    self._projects.values(), key=lambda value: value["name"].casefold()
                )
                if include_archived or item["state"] != "archived"
            ]

    def list_principals(self, *, include_inactive: bool = False) -> list[dict[str, Any]]:
        del include_inactive
        with self._lock:
            identifiers = {
                principal_id for _project_id, principal_id in self._acl
            }
            identifiers.update(
                str(event.get("actor_id", ""))
                for events in self._events.values()
                for event in events
                if event.get("actor_id")
            )
        return [
            {
                "principal_id": identifier,
                "provider": "local",
                "subject": identifier,
                "issuer": None,
                "display_name": identifier,
                "email": None,
                "active": True,
                "system_roles": [],
            }
            for identifier in sorted(identifiers)
        ]

    def list_performers(self, *, include_archived: bool = False) -> list[dict[str, Any]]:
        del include_archived
        return []

    def create_project(self, payload: Mapping[str, Any], context: CommandContext) -> dict[str, Any]:
        key = (context.actor_id, context.idempotency_key)
        with self._lock:
            if key in self._idempotency:
                return dict(self._idempotency[key])
            name = str(payload.get("name", "")).strip()
            width = int(payload.get("width", 0))
            height = int(payload.get("height", 0))
            orientation = str(payload.get("orientation", "y_down"))
            if not name:
                raise ValidationError("Project name is required")
            if width < 1 or height < 1:
                raise ValidationError("Project dimensions must be positive")
            if orientation not in {"y_down", "y_up"}:
                raise ValidationError("Unsupported matrix orientation")
            project_id = str(uuid4())
            now = datetime.now(UTC).isoformat()
            project = {
                "project_id": project_id,
                "name": name,
                "width": width,
                "height": height,
                "orientation": orientation,
                "state": "active",
                "revision": 0,
                "created_at": now,
            }
            event = {
                "event_id": str(uuid4()),
                "event_type": "project.created",
                "revision": 0,
                "recorded_at": now,
                "actor_id": context.actor_id,
                "payload": dict(project),
            }
            self._projects[project_id] = project
            self._events[project_id] = [event]
            self._layers[project_id] = []
            self._acl[(project_id, context.actor_id)] = {"maintainer"}
            self._granted_by[(project_id, context.actor_id, "maintainer")] = context.actor_id
            self._acl_revisions[(project_id, context.actor_id)] = 0
            self._idempotency[key] = project
            return dict(project)

    def project_workspace(self, project_id: str) -> dict[str, Any] | None:
        with self._lock:
            project = self._projects.get(project_id)
            if project is None:
                return None
            binding = self._workspaces.get(project_id)
            if binding is None:
                root = Path(tempfile.mkdtemp(prefix=f"kraken-dev-{project_id}-"))
                source = root / "source"
                derived = root / "derived"
                source.mkdir()
                derived.mkdir()
                binding = {
                    "project_id": project_id,
                    "project_name": str(project.get("name", "")),
                    "source_root": str(root),
                    "derived_root": str(root),
                    "source_project_dir": str(source),
                    "derived_project_dir": str(derived),
                    "schema_version": 1,
                }
                self._workspaces[project_id] = binding
            return dict(binding)

    def get_project(self, project_id: str) -> dict[str, Any]:
        with self._lock:
            try:
                return dict(self._projects[project_id])
            except KeyError as exc:
                raise NotFoundError(project_id) from exc

    def _project_lifecycle(
        self,
        project_id: str,
        context: CommandContext,
        *,
        operation: str,
        name: str | None = None,
    ) -> dict[str, Any]:
        key = (f"project:{operation}:{project_id}:{context.actor_id}", context.idempotency_key)
        with self._lock:
            if key in self._idempotency:
                return dict(self._idempotency[key])
            project = self._projects.get(project_id)
            if project is None:
                raise NotFoundError(project_id)
            if context.expected_revision != project["revision"]:
                raise ConflictError("Project revision changed")
            if operation == "rename":
                value = str(name or "").strip()
                if not value:
                    raise ValidationError("Project name is required")
                if project["state"] == "archived":
                    raise ConflictError("Archived project is read-only")
                project["name"] = value
            elif operation == "archive":
                if project["state"] == "archived":
                    raise ConflictError("Project is already archived")
                project["state"] = "archived"
            elif operation == "restore":
                if project["state"] == "active":
                    raise ConflictError("Project is already active")
                project["state"] = "active"
            project["revision"] += 1
            snapshot = dict(project)
            self._events[project_id].append(
                {
                    "event_id": str(uuid4()),
                    "event_type": f"Project{operation.title()}",
                    "revision": project["revision"],
                    "recorded_at": datetime.now(UTC).isoformat(),
                    "actor_id": context.actor_id,
                    "payload": {"project": snapshot},
                }
            )
            self._idempotency[key] = snapshot
            return snapshot

    def rename_project(self, project_id: str, name: str, context: CommandContext) -> dict[str, Any]:
        return self._project_lifecycle(project_id, context, operation="rename", name=name)

    def archive_project(self, project_id: str, context: CommandContext) -> dict[str, Any]:
        return self._project_lifecycle(project_id, context, operation="archive")

    def restore_project(self, project_id: str, context: CommandContext) -> dict[str, Any]:
        return self._project_lifecycle(project_id, context, operation="restore")

    def list_layers(
        self, project_id: str, *, include_archived: bool = False
    ) -> list[dict[str, Any]]:
        self.get_project(project_id)
        with self._lock:
            return [
                dict(item)
                for item in self._layers[project_id]
                if include_archived or item["state"] != "archived"
            ]

    def create_layer(
        self, project_id: str, payload: Mapping[str, Any], context: CommandContext
    ) -> dict[str, Any]:
        project = self.get_project(project_id)
        key = (f"layer:{project_id}:{context.actor_id}", context.idempotency_key)
        with self._lock:
            if key in self._idempotency:
                return dict(self._idempotency[key])
        if context.expected_revision != project["revision"]:
            raise ConflictError("Project revision changed")
        name = str(payload.get("name", "")).strip()
        layer_type = str(payload.get("type", ""))
        if not name or layer_type not in {"metal", "contact", "gate", "diffusion"}:
            raise ValidationError("Layer name and a supported type are required")
        with self._lock:
            if any(layer["name"].casefold() == name.casefold() for layer in self._layers[project_id]):
                raise ConflictError("Layer name already exists")
            layer = {
                "layer_id": str(uuid4()),
                "project_id": project_id,
                "name": name,
                "type": layer_type,
                "order": int(payload.get("order", len(self._layers[project_id]) + 1)),
                "state": "active",
                "revision": 0,
            }
            self._layers[project_id].append(layer)
            self._representations[layer["layer_id"]] = []
            self._projects[project_id]["revision"] += 1
            self._idempotency[key] = layer
            return dict(layer)

    def _layer_lifecycle(
        self,
        project_id: str,
        layer_id: str,
        context: CommandContext,
        *,
        operation: str,
        value: object | None = None,
    ) -> dict[str, Any]:
        self.get_project(project_id)
        key = (f"layer:{operation}:{layer_id}:{context.actor_id}", context.idempotency_key)
        with self._lock:
            if key in self._idempotency:
                return dict(self._idempotency[key])
            layer = next((item for item in self._layers[project_id] if item["layer_id"] == layer_id), None)
            if layer is None:
                raise NotFoundError(layer_id)
            if context.expected_revision != layer["revision"]:
                raise ConflictError("Layer revision changed")
            if operation != "archive" and layer["state"] == "archived":
                raise ConflictError("Archived layer is read-only")
            if operation == "rename":
                name = str(value or "").strip()
                if not name:
                    raise ValidationError("Layer name is required")
                layer["name"] = name
            elif operation == "reorder":
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValidationError("Layer order must be a non-negative integer")
                layer["order"] = value
            elif operation == "archive":
                if layer["state"] == "archived":
                    raise ConflictError("Layer is already archived")
                layer["state"] = "archived"
            layer["revision"] += 1
            snapshot = dict(layer)
            self._events[project_id].append(
                {
                    "event_id": str(uuid4()),
                    "event_type": f"Layer{operation.title()}",
                    "revision": layer["revision"],
                    "recorded_at": datetime.now(UTC).isoformat(),
                    "actor_id": context.actor_id,
                    "payload": {"layer": snapshot},
                }
            )
            self._idempotency[key] = snapshot
            return snapshot

    def rename_layer(
        self, project_id: str, layer_id: str, name: str, context: CommandContext
    ) -> dict[str, Any]:
        return self._layer_lifecycle(project_id, layer_id, context, operation="rename", value=name)

    def reorder_layer(
        self, project_id: str, layer_id: str, order: int, context: CommandContext
    ) -> dict[str, Any]:
        return self._layer_lifecycle(project_id, layer_id, context, operation="reorder", value=order)

    def reorder_layers(
        self, project_id: str, payload: Mapping[str, Any], context: CommandContext
    ) -> list[dict[str, Any]]:
        self.get_project(project_id)
        layer_ids = tuple(str(value) for value in payload.get("layer_ids", ()))
        raw_revisions = payload.get("expected_revisions", {})
        if not isinstance(raw_revisions, Mapping):
            raise ValidationError("expected_revisions must be an object")
        expected = {str(identifier): int(revision) for identifier, revision in raw_revisions.items()}
        key = (f"layers:reorder:{project_id}:{context.actor_id}", context.idempotency_key)
        with self._lock:
            cached = self._idempotency.get(key)
            if cached is not None:
                return [dict(item) for item in cached["items"]]
            layers = self._layers.get(project_id)
            if layers is None:
                raise NotFoundError(project_id)
            current_ids = {str(item["layer_id"]) for item in layers}
            if not layer_ids or len(layer_ids) != len(set(layer_ids)) or set(layer_ids) != current_ids:
                raise ConflictError("Layer order must contain every project layer exactly once")
            if set(expected) != current_ids:
                raise ValidationError("expected_revisions must cover every project layer")
            for layer in layers:
                if int(layer["revision"]) != expected[str(layer["layer_id"])]:
                    raise ConflictError("Layer revision changed")
            by_id = {str(item["layer_id"]): item for item in layers}
            reordered: list[dict[str, Any]] = []
            now = datetime.now(UTC).isoformat()
            for order, identifier in enumerate(layer_ids):
                layer = by_id[identifier]
                layer["order"] = order
                layer["revision"] += 1
                reordered.append(layer)
                self._events[project_id].append(
                    {
                        "event_id": str(uuid4()),
                        "event_type": "LayersReordered",
                        "revision": layer["revision"],
                        "recorded_at": now,
                        "actor_id": context.actor_id,
                        "payload": {"layer": dict(layer), "layer_ids": list(layer_ids)},
                    }
                )
            self._layers[project_id] = reordered
            result = {"items": [dict(item) for item in reordered]}
            self._idempotency[key] = result
            return [dict(item) for item in reordered]

    def archive_layer(
        self, project_id: str, layer_id: str, context: CommandContext
    ) -> dict[str, Any]:
        return self._layer_lifecycle(project_id, layer_id, context, operation="archive")

    def project_roles(self, project_id: str, principal_id: str) -> dict[str, Any]:
        self.get_project(project_id)
        key = (project_id, principal_id)
        with self._lock:
            return {
                "project_id": project_id,
                "principal_id": principal_id,
                "roles": sorted(self._acl.get(key, set())),
                "revision": self._acl_revisions.get(key, 0),
            }

    def _change_project_role(
        self,
        project_id: str,
        principal_id: str,
        role: str,
        context: CommandContext,
        *,
        revoke: bool,
    ) -> dict[str, Any]:
        self.get_project(project_id)
        from kraken_manager.domain.roles import CATALOG, cascade_lost_assignments, parse_role_id

        role = parse_role_id(role)
        if role not in CATALOG.roles:
            raise ValidationError("Unsupported project role")
        actor_roles = self._acl.get((project_id, context.actor_id), set())
        if not CATALOG.can_grant(actor_roles, role):
            raise ForbiddenError(
                f"Недостаточно прав: нельзя изменить роль «{CATALOG.roles[role].title}»."
            )
        if revoke and role == "maintainer":
            maintainers = {
                principal
                for (candidate_project, principal), roles in self._acl.items()
                if candidate_project == project_id and "maintainer" in roles
            }
            if principal_id in maintainers and len(maintainers) <= 1:
                raise ForbiddenError("Нельзя отозвать роль последнего сопровождающего проекта.")
        operation = "revoke" if revoke else "assign"
        idempotency = (f"acl:{operation}:{project_id}:{principal_id}", context.idempotency_key)
        key = (project_id, principal_id)
        with self._lock:
            if idempotency in self._idempotency:
                return dict(self._idempotency[idempotency])
            revision = self._acl_revisions.get(key, 0)
            if context.expected_revision != revision:
                raise ConflictError("ACL revision changed")
            roles = self._acl.setdefault(key, set())
            if revoke and role not in roles:
                raise ConflictError("The principal does not have this role")
            if not revoke and role in roles:
                raise ConflictError("The principal already has this role")
            if revoke:
                roles.remove(role)
                self._granted_by.pop((project_id, principal_id, role), None)
                assignments = tuple(
                    (holder, granted_role, grantor)
                    for (candidate_project, holder, granted_role), grantor in self._granted_by.items()
                    if candidate_project == project_id and granted_role in self._acl.get((candidate_project, holder), set())
                )
                lost = cascade_lost_assignments(
                    assignments,
                    principal_id=principal_id,
                    revoked_role=role,
                    remaining_roles=roles,
                    grantor_is_admin=False,
                    catalog=CATALOG,
                )
                for holder, lost_role in lost:
                    holder_roles = self._acl.get((project_id, holder), set())
                    holder_roles.discard(lost_role)
                    self._granted_by.pop((project_id, holder, lost_role), None)
                    self._acl_revisions[(project_id, holder)] = self._acl_revisions.get((project_id, holder), 0) + 1
            else:
                roles.add(role)
                self._granted_by[(project_id, principal_id, role)] = context.actor_id
            revision += 1
            self._acl_revisions[key] = revision
            result = {
                "project_id": project_id,
                "principal_id": principal_id,
                "roles": sorted(roles),
                "revision": revision,
            }
            self._events[project_id].append(
                {
                    "event_id": str(uuid4()),
                    "event_type": "ProjectRoleRevoked" if revoke else "ProjectRoleAssigned",
                    "revision": revision,
                    "recorded_at": datetime.now(UTC).isoformat(),
                    "actor_id": context.actor_id,
                    "payload": {"principal_id": principal_id, "role": role},
                }
            )
            self._idempotency[idempotency] = result
            return dict(result)

    def assign_project_role(
        self, project_id: str, principal_id: str, role: str, context: CommandContext
    ) -> dict[str, Any]:
        return self._change_project_role(project_id, principal_id, role, context, revoke=False)

    def revoke_project_role(
        self, project_id: str, principal_id: str, role: str, context: CommandContext
    ) -> dict[str, Any]:
        return self._change_project_role(project_id, principal_id, role, context, revoke=True)

    def list_representations(
        self, project_id: str, layer_id: str, *, include_archived: bool = False
    ) -> list[dict[str, Any]]:
        self.get_project(project_id)
        if layer_id not in self._representations:
            raise NotFoundError(layer_id)
        with self._lock:
            return [
                dict(item)
                for item in self._representations[layer_id]
                if include_archived or item["state"] != "archived"
            ]

    def create_representation(
        self,
        project_id: str,
        layer_id: str,
        payload: Mapping[str, Any],
        context: CommandContext,
    ) -> dict[str, Any]:
        self.get_project(project_id)
        layers = {item["layer_id"]: item for item in self._layers[project_id]}
        layer = layers.get(layer_id)
        if layer is None:
            raise NotFoundError(layer_id)
        key = (f"representation:{layer_id}:{context.actor_id}", context.idempotency_key)
        with self._lock:
            if key in self._idempotency:
                return dict(self._idempotency[key])
        if context.expected_revision != layer["revision"]:
            raise ConflictError("Layer revision changed")
        name = str(payload.get("name", "")).strip()
        kind = str(payload.get("kind", ""))
        if not name or kind not in {"image", "vector"}:
            raise ValidationError("Representation name and kind are required")
        purpose = str(payload.get("purpose", "vector" if kind == "vector" else "source"))
        if purpose not in {"source", "binary", "vector"} or (kind == "vector") != (purpose == "vector"):
            raise ValidationError("Representation purpose is incompatible with its kind")
        with self._lock:
            representations = self._representations[layer_id]
            if any(item["name"].casefold() == name.casefold() for item in representations):
                raise ConflictError("Representation name already exists")
            active = bool(payload.get("active", False))
            if active:
                for previous in representations:
                    if previous["kind"] == kind:
                        previous["active"] = False
            representation = {
                "representation_id": str(uuid4()),
                "project_id": project_id,
                "layer_id": layer_id,
                "name": name,
                "kind": kind,
                "purpose": purpose,
                "source_image_representation_id": payload.get("source_image_representation_id"),
                "note": str(payload.get("note", "")),
                "source": payload.get("source"),
                "active": active,
                "state": "active",
                "revision": 0,
            }
            representations.append(representation)
            layer["revision"] += 1
            self._idempotency[key] = representation
            return dict(representation)

    def update_representation(
        self, project_id: str, layer_id: str, representation_id: str,
        payload: Mapping[str, Any], context: CommandContext,
    ) -> dict[str, Any]:
        self.get_project(project_id)
        layer = next((item for item in self._layers[project_id] if item["layer_id"] == layer_id), None)
        if layer is None:
            raise NotFoundError(layer_id)
        values = self._representations.get(layer_id, [])
        representation = next((item for item in values if item["representation_id"] == representation_id), None)
        if representation is None:
            raise NotFoundError(representation_id)
        operations = [key for key in ("name", "note", "active", "archive") if key in payload]
        if len(operations) != 1:
            raise ValidationError("Exactly one representation operation is required")
        operation = operations[0]
        key = (f"representation:{operation}:{representation_id}:{context.actor_id}", context.idempotency_key)
        with self._lock:
            if key in self._idempotency:
                return dict(self._idempotency[key])
            if context.expected_revision != layer["revision"]:
                raise ConflictError("Layer revision changed")
            expected_representation = int(payload.get("expected_representation_revision", -1))
            if expected_representation != representation["revision"]:
                raise ConflictError("Representation revision changed")
            if operation == "name":
                name = str(payload["name"]).strip()
                if not name:
                    raise ValidationError("Representation name is required")
                representation["name"] = name
            elif operation == "note":
                representation["note"] = str(payload["note"])
            elif operation == "active":
                activate = bool(payload["active"])
                if activate:
                    if representation["active"]:
                        raise ConflictError("Representation is already active")
                    for previous in values:
                        if previous is not representation and previous["kind"] == representation["kind"] and previous["active"]:
                            previous["active"] = False
                            previous["revision"] += 1
                representation["active"] = activate
            else:
                representation["state"] = "archived"
                representation["active"] = False
            representation["revision"] += 1
            layer["revision"] += 1
            result = dict(representation)
            self._idempotency[key] = result
            return result

    def matrix_viewport(
        self, project_id: str, *, layer_id: str, representation_ids: Iterable[str] = (),
        x1: int, y1: int, x2: int, y2: int, lod: int, include_missing: bool = True
    ) -> dict[str, Any]:
        del representation_ids, include_missing
        project = self.get_project(project_id)
        if not (1 <= x1 <= x2 <= project["width"] and 1 <= y1 <= y2 <= project["height"]):
            raise ValidationError("Viewport is outside the project grid")
        if lod < 0 or lod > 24:
            raise ValidationError("LOD must be between 0 and 24")
        # Empty frames are deliberately not materialized.
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
    def _encode_cursor(offset: int) -> str:
        return base64.urlsafe_b64encode(json.dumps({"offset": offset}).encode()).decode().rstrip("=")

    @staticmethod
    def _decode_cursor(cursor: str | None) -> int:
        if not cursor:
            return 0
        try:
            payload = json.loads(base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4)))
            offset = int(payload["offset"])
        except Exception as exc:
            raise ValidationError("Invalid history cursor") from exc
        if offset < 0:
            raise ValidationError("Invalid history cursor")
        return offset

    def history(self, project_id: str, *, cursor: str | None, limit: int) -> dict[str, Any]:
        self.get_project(project_id)
        if limit < 1 or limit > 500:
            raise ValidationError("History limit must be between 1 and 500")
        offset = self._decode_cursor(cursor)
        with self._lock:
            events = self._events[project_id]
            page = [dict(item) for item in events[offset : offset + limit]]
        next_offset = offset + len(page)
        return {
            "items": page,
            "next_cursor": self._encode_cursor(next_offset) if next_offset < len(events) else None,
        }

    def _append_auxiliary_event(
        self,
        project_id: str,
        *,
        stream_id: str,
        event_type: str,
        payload: Mapping[str, object],
        context: CommandContext,
    ) -> dict[str, Any]:
        self.get_project(project_id)
        key = (f"event:{project_id}:{context.actor_id}", context.idempotency_key)
        with self._lock:
            if key in self._idempotency:
                return dict(self._idempotency[key])
            revision = max(
                (
                    int(event.get("revision", 0))
                    for event in self._events[project_id]
                    if event.get("stream_id") == stream_id
                ),
                default=0,
            )
            if context.expected_revision != revision:
                raise ConflictError(
                    f"Expected stream revision {context.expected_revision}, found {revision}"
                )
            event = {
                "event_id": str(uuid4()),
                "stream_id": stream_id,
                "event_type": event_type,
                "revision": revision + 1,
                "recorded_at": datetime.now(UTC).isoformat(),
                "actor": {
                    "principal_id": context.actor_id,
                    "display_name": context.actor_id,
                },
                "payload": dict(payload),
            }
            self._events[project_id].append(event)
            self._idempotency[key] = event
            return dict(event)

    def publish_karakal_analysis(
        self, project_id: str, payload: Mapping[str, Any], context: CommandContext
    ) -> dict[str, Any]:
        layer_id = str(payload.get("layer_id", ""))
        if not any(item["layer_id"] == layer_id for item in self._layers.get(project_id, ())):
            raise NotFoundError("Layer was not found")
        confidence = {
            str(frame_id): float(value)
            for frame_id, value in dict(payload.get("frame_confidence", {})).items()
        }
        if any(value < 0.0 or value > 1.0 for value in confidence.values()):
            raise ValidationError("Karakal confidence must be in range 0..1")
        revision = max(
            (
                int(event.get("revision", 0))
                for event in self._events.get(project_id, ())
                if event.get("stream_id") == f"karakal:{layer_id}"
            ),
            default=0,
        )
        return self._append_auxiliary_event(
            project_id,
            stream_id=f"karakal:{layer_id}",
            event_type="KarakalAnalysisPublished",
            payload={
                **dict(payload),
                "frame_confidence": confidence,
                "publication_sequence": revision + 1,
            },
            context=context,
        )

    def append_pipeline_event(
        self, project_id: str, payload: Mapping[str, Any], context: CommandContext
    ) -> dict[str, Any]:
        event_type = str(payload.get("event_type", ""))
        if event_type not in {"LayerPipelineActionRequested", "LayerPipelineActionRemoved"}:
            raise ValidationError("Unsupported pipeline event type")
        layer_id = str(payload.get("layer_id", ""))
        if not any(item["layer_id"] == layer_id for item in self._layers.get(project_id, ())):
            raise NotFoundError("Layer was not found")
        return self._append_auxiliary_event(
            project_id,
            stream_id=f"layer-pipeline:{layer_id}",
            event_type=event_type,
            payload={key: value for key, value in payload.items() if key != "event_type"},
            context=context,
        )

    def list_plugin_jobs(self, project_id: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            jobs = [
                dict(job)
                for job in self._plugin_jobs.values()
                if project_id is None or str(job.get("project_id")) == project_id
            ]
        jobs.sort(key=lambda item: (str(item.get("updated_at", "")), str(item.get("id", ""))), reverse=True)
        return jobs

    def submit_plugin_job(
        self, project_id: str, payload: Mapping[str, Any], context: CommandContext
    ) -> dict[str, Any]:
        key = (f"plugin-job:{project_id}:{context.actor_id}", context.idempotency_key)
        with self._lock:
            if key in self._idempotency:
                return dict(self._idempotency[key])
            if project_id not in self._projects:
                raise NotFoundError(project_id)
            layer_id = str(payload.get("layer_id", "")).strip()
            target_id = str(payload.get("target_representation_id", "")).strip()
            capability = str(payload.get("capability", "")).strip()
            if not layer_id or not target_id or not capability:
                raise ValidationError(
                    "layer_id, target_representation_id and capability are required"
                )
            layers = self._layers.get(project_id, [])
            if layers and not any(str(item.get("layer_id")) == layer_id for item in layers):
                raise NotFoundError(layer_id)
            raw_coordinates = payload.get("coordinates", ())
            if not isinstance(raw_coordinates, list):
                raise ValidationError("coordinates must be an array")
            coordinates = sorted(
                {(int(item[0]), int(item[1])) for item in raw_coordinates},
                key=lambda item: (item[1], item[0]),
            )
            selection = FrameSelectionV1(
                row_ranges=tuple(
                    FrameRowRange(y=y, x_start=x, x_end=x) for x, y in coordinates
                )
            )
            job = PluginJob.create(
                project_id=ProjectId(project_id),
                layer_id=LayerId(layer_id),
                selection=selection,
                actor_principal_id=PrincipalId(context.actor_id),
                target_representation_id=RepresentationId(target_id),
                capability=capability,
            )
            encoded = encode_model(job)
            self._plugin_jobs[str(job.id)] = encoded
            self._idempotency[key] = encoded
            return dict(encoded)

    def cancel_plugin_job(
        self, project_id: str, job_id: str, context: CommandContext
    ) -> dict[str, Any]:
        key = (f"plugin-job-cancel:{job_id}:{context.actor_id}", context.idempotency_key)
        with self._lock:
            if key in self._idempotency:
                return dict(self._idempotency[key])
            job = self._plugin_jobs.get(job_id)
            if job is None or str(job.get("project_id")) != project_id:
                raise NotFoundError(job_id)
            if context.expected_revision is not None and int(job.get("revision", -1)) != context.expected_revision:
                raise ConflictError("Plugin job revision changed")
            if str(job.get("state")) in {
                PluginJobState.SUCCEEDED.value,
                PluginJobState.FAILED.value,
                PluginJobState.CANCELLED.value,
            }:
                raise ConflictError("Plugin job is already terminal")
            now = datetime.now(UTC).isoformat()
            updated = dict(job)
            updated["state"] = PluginJobState.CANCELLED.value
            updated["revision"] = int(job.get("revision", 0)) + 1
            updated["updated_at"] = now
            updated["finished_at"] = now
            self._plugin_jobs[job_id] = updated
            self._idempotency[key] = updated
            return dict(updated)

    def fail_agent_job(self, job_id: str, error: str) -> dict[str, Any]:
        with self._lock:
            job = self._plugin_jobs.get(job_id)
            if job is None:
                raise NotFoundError(job_id)
            now = datetime.now(UTC).isoformat()
            updated = dict(job)
            updated["state"] = PluginJobState.FAILED.value
            updated["error"] = str(error)
            updated["revision"] = int(job.get("revision", 0)) + 1
            updated["updated_at"] = now
            updated["finished_at"] = now
            self._plugin_jobs[job_id] = updated
            return dict(updated)

    def import_agent_result(self, job_id: str, agent: Any) -> dict[str, Any]:
        del agent
        with self._lock:
            job = self._plugin_jobs.get(job_id)
            if job is None:
                raise NotFoundError(job_id)
            now = datetime.now(UTC).isoformat()
            updated = dict(job)
            updated["state"] = PluginJobState.SUCCEEDED.value
            updated["progress"] = 1.0
            updated["revision"] = int(job.get("revision", 0)) + 1
            updated["updated_at"] = now
            updated["finished_at"] = now
            self._plugin_jobs[job_id] = updated
            return dict(updated)


__all__ = [
    "CommandContext",
    "ConflictError",
    "ForbiddenError",
    "InMemoryServerServices",
    "NotFoundError",
    "ServerServices",
    "ValidationError",
]
