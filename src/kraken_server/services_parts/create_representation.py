"""services: create representation."""
from __future__ import annotations
from collections.abc import Mapping
from typing import Any
from uuid import uuid4

from kraken_server.services_parts.command_context import (
    ConflictError,
    NotFoundError,
    ValidationError,
)


class CreateRepresentationMixin:
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
