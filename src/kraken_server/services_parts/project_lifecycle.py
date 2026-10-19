"""services: project lifecycle."""
from __future__ import annotations
from typing import TYPE_CHECKING
import tempfile
import threading
from pathlib import Path
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from kraken_server.services_parts.command_context import (
    ConflictError,
    NotFoundError,
    ValidationError,
)


class ProjectLifecycleMixin:
    def request_project_deletion(self, project_id: str, actor_id: str, reason: str = "") -> dict[str, Any]:
        del reason
        raise ConflictError("Deletion requests require a persistent PostgreSQL server")

    def project_deletion_requests(self, project_id: str) -> list[dict[str, Any]]:
        self.get_project(project_id)
        return []

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

if TYPE_CHECKING:
    from kraken_server.services_parts.command_context import CommandContext
