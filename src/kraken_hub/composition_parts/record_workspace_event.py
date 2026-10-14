"""composition: record workspace event."""
from __future__ import annotations
from typing import Any
from kraken_manager.domain.common import (
    ProjectId,
)
from kraken_manager.domain.events import ActorSnapshot, EventEnvelope, ProgramSnapshot
from kraken_manager.domain.identity import (
    Principal,
)
from kraken_manager.infrastructure.filesystem import (
    FilesystemEventStore,
)


class RecordWorkspaceEventMixin:
    def _record_workspace_event(
        self,
        *,
        principal: Principal,
        project_id: ProjectId | str,
        stream_id: str,
        event_type: str,
        payload: dict[str, Any],
        idempotency_key: str,
    ) -> EventEnvelope:
        store = FilesystemEventStore(self.catalog_root, str(project_id))
        revision = store.current_revision(stream_id)
        event = EventEnvelope.create(
            stream_id=stream_id,
            project_id=ProjectId(str(project_id)),
            revision=revision + 1,
            event_type=event_type,
            payload=payload,
            actor=ActorSnapshot.from_principal(principal),
            program=ProgramSnapshot("Kraken workspace", "1"),
            idempotency_key=idempotency_key,
        )
        store.append(stream_id, expected_revision=revision, events=(event,))
        return event
