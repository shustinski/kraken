"""services: update representation."""
from __future__ import annotations
import base64
import json
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from kraken_server.services_parts.command_context import (
    ConflictError,
    NotFoundError,
    ValidationError,
)


class UpdateRepresentationMixin:
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
