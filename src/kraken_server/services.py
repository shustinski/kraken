"""Transport-facing server service contract.

FastAPI depends on this narrow facade.  Production composition maps it to
application use cases; the in-memory implementation is intentionally only a
development/test profile.
"""

from __future__ import annotations

import base64
import json
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Mapping, Protocol
from uuid import uuid4


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
    def health(self) -> dict[str, Any]: ...

    def list_projects(self) -> list[dict[str, Any]]: ...

    def create_project(self, payload: Mapping[str, Any], context: CommandContext) -> dict[str, Any]: ...

    def get_project(self, project_id: str) -> dict[str, Any]: ...

    def list_layers(self, project_id: str) -> list[dict[str, Any]]: ...

    def create_layer(
        self, project_id: str, payload: Mapping[str, Any], context: CommandContext
    ) -> dict[str, Any]: ...

    def list_representations(self, project_id: str, layer_id: str) -> list[dict[str, Any]]: ...

    def create_representation(
        self,
        project_id: str,
        layer_id: str,
        payload: Mapping[str, Any],
        context: CommandContext,
    ) -> dict[str, Any]: ...

    def matrix_viewport(
        self, project_id: str, *, layer_id: str, x1: int, y1: int, x2: int, y2: int, lod: int
    ) -> dict[str, Any]: ...

    def history(self, project_id: str, *, cursor: str | None, limit: int) -> dict[str, Any]: ...


class InMemoryServerServices:
    """Sparse, concurrency-safe development backend used by API tests."""

    def __init__(self) -> None:
        self._projects: dict[str, dict[str, Any]] = {}
        self._events: dict[str, list[dict[str, Any]]] = {}
        self._idempotency: dict[tuple[str, str], dict[str, Any]] = {}
        self._layers: dict[str, list[dict[str, Any]]] = {}
        self._representations: dict[str, list[dict[str, Any]]] = {}
        self._lock = threading.RLock()

    def health(self) -> dict[str, Any]:
        return {"status": "ok", "metadata": "memory", "api_version": "v1"}

    def list_projects(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(item) for item in sorted(self._projects.values(), key=lambda value: value["name"].casefold())]

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
            self._idempotency[key] = project
            return dict(project)

    def get_project(self, project_id: str) -> dict[str, Any]:
        with self._lock:
            try:
                return dict(self._projects[project_id])
            except KeyError as exc:
                raise NotFoundError(project_id) from exc

    def list_layers(self, project_id: str) -> list[dict[str, Any]]:
        self.get_project(project_id)
        with self._lock:
            return [dict(item) for item in self._layers[project_id]]

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

    def list_representations(self, project_id: str, layer_id: str) -> list[dict[str, Any]]:
        self.get_project(project_id)
        if layer_id not in self._representations:
            raise NotFoundError(layer_id)
        with self._lock:
            return [dict(item) for item in self._representations[layer_id]]

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

    def matrix_viewport(
        self, project_id: str, *, layer_id: str, x1: int, y1: int, x2: int, y2: int, lod: int
    ) -> dict[str, Any]:
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


__all__ = [
    "CommandContext",
    "ConflictError",
    "ForbiddenError",
    "InMemoryServerServices",
    "NotFoundError",
    "ServerServices",
    "ValidationError",
]
