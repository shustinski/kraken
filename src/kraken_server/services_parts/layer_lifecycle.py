"""services: layer lifecycle."""
from __future__ import annotations
from typing import TYPE_CHECKING
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from kraken_server.services_parts.command_context import (
    ConflictError,
    NotFoundError,
    ValidationError,
)


class LayerLifecycleMixin:
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

if TYPE_CHECKING:
    from kraken_server.services_parts.command_context import CommandContext
