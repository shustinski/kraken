"""Versioned event envelope used as the canonical project history."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime

from .common import (
    DomainValidationError,
    FrameId,
    FrozenJson,
    LayerId,
    PerformerId,
    PrincipalId,
    ProjectId,
    as_utc,
    freeze_mapping,
    new_uuid,
    require_non_empty,
    utc_now,
    validate_uuid,
)
from .identity import Principal, PrincipalProvider


@dataclass(frozen=True, slots=True)
class ActorSnapshot:
    """Historical identity label, preserved after account changes/deletion."""

    principal_id: PrincipalId
    provider: PrincipalProvider
    subject: str
    display_name: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "principal_id", PrincipalId(validate_uuid(str(self.principal_id), field="actor.principal_id"))
        )
        if not isinstance(self.provider, PrincipalProvider):
            object.__setattr__(self, "provider", PrincipalProvider(self.provider))
        object.__setattr__(self, "subject", require_non_empty(self.subject, field="actor.subject", maximum=512))
        object.__setattr__(
            self, "display_name", require_non_empty(self.display_name, field="actor.display_name", maximum=255)
        )

    @classmethod
    def from_principal(cls, principal: Principal) -> ActorSnapshot:
        return cls(
            principal_id=principal.id,
            provider=principal.provider,
            subject=principal.subject,
            display_name=principal.display_name,
        )


@dataclass(frozen=True, slots=True)
class ProgramSnapshot:
    name: str
    version: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", require_non_empty(self.name, field="program.name"))
        if self.version is not None:
            object.__setattr__(self, "version", require_non_empty(self.version, field="program.version"))


@dataclass(frozen=True, slots=True)
class EventEnvelope:
    """Immutable event metadata and JSON-compatible payload.

    ``recorded_at`` is authoritative for temporal projections. ``effective_at``
    is only a claimed business time and never rewrites already-recorded history.
    """

    event_id: str
    stream_id: str
    project_id: ProjectId
    revision: int
    event_type: str
    payload: Mapping[str, object]
    schema_version: int
    recorded_at: datetime
    actor: ActorSnapshot
    effective_at: datetime | None = None
    performer_id: PerformerId | None = None
    program: ProgramSnapshot | None = None
    correlation_id: str | None = None
    causation_id: str | None = None
    idempotency_key: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_id", validate_uuid(self.event_id, field="event.event_id"))
        object.__setattr__(self, "stream_id", require_non_empty(self.stream_id, field="event.stream_id", maximum=512))
        object.__setattr__(
            self, "project_id", ProjectId(validate_uuid(str(self.project_id), field="event.project_id"))
        )
        if isinstance(self.revision, bool) or not isinstance(self.revision, int) or self.revision < 1:
            raise DomainValidationError("event.revision must start at 1")
        if isinstance(self.schema_version, bool) or not isinstance(self.schema_version, int) or self.schema_version < 1:
            raise DomainValidationError("event.schema_version must start at 1")
        object.__setattr__(self, "event_type", require_non_empty(self.event_type, field="event.event_type"))
        object.__setattr__(self, "payload", freeze_mapping(self.payload, field="event.payload"))
        object.__setattr__(self, "recorded_at", as_utc(self.recorded_at, field="event.recorded_at"))
        if self.effective_at is not None:
            object.__setattr__(self, "effective_at", as_utc(self.effective_at, field="event.effective_at"))
        if self.performer_id is not None:
            object.__setattr__(
                self,
                "performer_id",
                PerformerId(validate_uuid(str(self.performer_id), field="event.performer_id")),
            )
        for field_name in ("correlation_id", "causation_id"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, validate_uuid(value, field=f"event.{field_name}"))
        if self.idempotency_key is not None:
            object.__setattr__(
                self,
                "idempotency_key",
                require_non_empty(self.idempotency_key, field="event.idempotency_key", maximum=255),
            )

    @classmethod
    def create(
        cls,
        *,
        stream_id: str,
        project_id: ProjectId,
        revision: int,
        event_type: str,
        payload: Mapping[str, object],
        actor: ActorSnapshot,
        schema_version: int = 1,
        recorded_at: datetime | None = None,
        effective_at: datetime | None = None,
        performer_id: PerformerId | None = None,
        program: ProgramSnapshot | None = None,
        correlation_id: str | None = None,
        causation_id: str | None = None,
        idempotency_key: str | None = None,
        event_id: str | None = None,
    ) -> EventEnvelope:
        return cls(
            event_id=event_id or new_uuid(),
            stream_id=stream_id,
            project_id=project_id,
            revision=revision,
            event_type=event_type,
            payload=payload,
            schema_version=schema_version,
            recorded_at=recorded_at or utc_now(),
            effective_at=effective_at,
            actor=actor,
            performer_id=performer_id,
            program=program,
            correlation_id=correlation_id,
            causation_id=causation_id,
            idempotency_key=idempotency_key,
        )


@dataclass(frozen=True, slots=True)
class ActivityEvent:
    """Read-model row used by statistics without replaying aggregates."""

    event_id: str
    project_id: ProjectId
    event_type: str
    recorded_at: datetime
    actor_principal_id: PrincipalId
    performer_id: PerformerId | None = None
    layer_id: LayerId | None = None
    frame_id: FrameId | None = None
    attributes: Mapping[str, FrozenJson] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_id", validate_uuid(self.event_id, field="activity.event_id"))
        object.__setattr__(
            self, "project_id", ProjectId(validate_uuid(str(self.project_id), field="activity.project_id"))
        )
        object.__setattr__(self, "event_type", require_non_empty(self.event_type, field="activity.event_type"))
        object.__setattr__(self, "recorded_at", as_utc(self.recorded_at, field="activity.recorded_at"))
        object.__setattr__(
            self,
            "actor_principal_id",
            PrincipalId(validate_uuid(str(self.actor_principal_id), field="activity.actor_principal_id")),
        )
        if self.performer_id is not None:
            object.__setattr__(
                self,
                "performer_id",
                PerformerId(validate_uuid(str(self.performer_id), field="activity.performer_id")),
            )
        if self.layer_id is not None:
            object.__setattr__(
                self, "layer_id", LayerId(validate_uuid(str(self.layer_id), field="activity.layer_id"))
            )
        if self.frame_id is not None:
            object.__setattr__(
                self, "frame_id", FrameId(validate_uuid(str(self.frame_id), field="activity.frame_id"))
            )
        object.__setattr__(self, "attributes", freeze_mapping(self.attributes, field="activity.attributes"))


__all__ = ["ActivityEvent", "ActorSnapshot", "EventEnvelope", "ProgramSnapshot"]
