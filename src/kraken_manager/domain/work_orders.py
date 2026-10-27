"""Issued performer work: vector correction, vectorization, training, or stitching."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from enum import StrEnum

from kraken_manager.domain.common import (
    DomainValidationError,
    InvalidStateTransition,
    LayerId,
    PerformerId,
    PrincipalId,
    ProjectId,
    as_utc,
    require_non_empty,
    utc_now,
    validate_uuid,
)
from kraken_manager.domain.selection import FrameSelectionV1


class WorkKind(StrEnum):
    VECTOR_REVIEW = "vector_review"
    VECTORIZE = "vectorize"
    TRAIN = "train"
    STITCH = "stitch"

    @property
    def title(self) -> str:
        return {
            WorkKind.VECTOR_REVIEW: "Исправление векторов",
            WorkKind.VECTORIZE: "Векторизация",
            WorkKind.TRAIN: "Обучение модели",
            WorkKind.STITCH: "Сшивка",
        }[self]


class WorkPlacement(StrEnum):
    EXTERNAL = "external"
    SERVER = "server"

    @property
    def title(self) -> str:
        return "Внешний накопитель" if self is WorkPlacement.EXTERNAL else "На сервере"


class WorkOrderState(StrEnum):
    ISSUED = "issued"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class WorkOrder:
    id: str
    project_id: ProjectId
    layer_id: LayerId
    kind: WorkKind
    placement: WorkPlacement
    assignee_id: PerformerId
    created_by: PrincipalId
    instructions: str
    state: WorkOrderState
    revision: int
    created_at: datetime
    updated_at: datetime
    destination: str
    frame_count: int
    selection: FrameSelectionV1
    due_at: datetime | None = None
    related_batch_id: str = ""
    related_job_id: str = ""
    transport: str = "files"
    image_representation_id: str = ""
    vector_representation_id: str = ""
    dataset_version_id: str = ""
    capability: str = ""
    launch_mode: str = ""
    idempotency_key: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", validate_uuid(str(self.id), field="work_order.id"))
        object.__setattr__(
            self, "project_id", ProjectId(validate_uuid(str(self.project_id), field="work_order.project_id"))
        )
        object.__setattr__(
            self, "layer_id", LayerId(validate_uuid(str(self.layer_id), field="work_order.layer_id"))
        )
        if not isinstance(self.kind, WorkKind):
            object.__setattr__(self, "kind", WorkKind(self.kind))
        if not isinstance(self.placement, WorkPlacement):
            object.__setattr__(self, "placement", WorkPlacement(self.placement))
        object.__setattr__(
            self, "assignee_id", PerformerId(validate_uuid(str(self.assignee_id), field="work_order.assignee_id"))
        )
        object.__setattr__(
            self, "created_by", PrincipalId(validate_uuid(str(self.created_by), field="work_order.created_by"))
        )
        object.__setattr__(self, "instructions", self.instructions.strip())
        if not isinstance(self.state, WorkOrderState):
            object.__setattr__(self, "state", WorkOrderState(self.state))
        if isinstance(self.revision, bool) or not isinstance(self.revision, int) or self.revision < 0:
            raise DomainValidationError("work_order.revision must not be negative")
        object.__setattr__(self, "created_at", as_utc(self.created_at, field="work_order.created_at"))
        object.__setattr__(self, "updated_at", as_utc(self.updated_at, field="work_order.updated_at"))
        if self.updated_at < self.created_at:
            raise DomainValidationError("work_order.updated_at cannot precede creation")
        # A destination is a path/locator, not a 255-character display name.
        # The filesystem adapter applies its actual platform/path permissions.
        destination = require_non_empty(self.destination, field="work_order.destination", maximum=32767)
        if '\x00' in destination:
            raise DomainValidationError('work_order.destination must not contain NUL')
        object.__setattr__(self, "destination", destination)
        if isinstance(self.frame_count, bool) or not isinstance(self.frame_count, int) or self.frame_count < 0:
            raise DomainValidationError("work_order.frame_count must not be negative")
        if self.kind is not WorkKind.TRAIN and self.frame_count < 1:
            raise DomainValidationError("work order requires at least one frame")
        if self.due_at is not None:
            due_at = as_utc(self.due_at, field="work_order.due_at")
            if due_at < self.created_at:
                raise DomainValidationError("work order due date cannot precede creation")
            object.__setattr__(self, "due_at", due_at)
        transport = require_non_empty(self.transport, field="work_order.transport", maximum=16)
        if transport not in {"files", "http"}:
            raise DomainValidationError("work order transport must be files or http")
        object.__setattr__(self, "transport", transport)
        object.__setattr__(self, "idempotency_key", self.idempotency_key.strip())

    @classmethod
    def create(
        cls,
        *,
        project_id: ProjectId,
        layer_id: LayerId,
        kind: WorkKind,
        placement: WorkPlacement,
        assignee_id: PerformerId,
        created_by: PrincipalId,
        destination: str,
        frame_count: int,
        selection: FrameSelectionV1,
        instructions: str = "",
        due_at: datetime | None = None,
        transport: str = "files",
        image_representation_id: str = "",
        vector_representation_id: str = "",
        dataset_version_id: str = "",
        capability: str = "",
        launch_mode: str = "",
        related_batch_id: str = "",
        idempotency_key: str = "",
        order_id: str | None = None,
        created_at: datetime | None = None,
    ) -> WorkOrder:
        now = created_at or utc_now()
        from kraken_manager.domain.common import new_uuid

        return cls(
            id=str(order_id) if order_id is not None else new_uuid(),
            project_id=project_id,
            layer_id=layer_id,
            kind=kind,
            placement=placement,
            assignee_id=assignee_id,
            created_by=created_by,
            instructions=instructions,
            state=WorkOrderState.ISSUED,
            revision=0,
            created_at=now,
            updated_at=now,
            destination=destination,
            frame_count=frame_count,
            selection=selection,
            due_at=due_at,
            transport=transport,
            image_representation_id=image_representation_id,
            vector_representation_id=vector_representation_id,
            dataset_version_id=dataset_version_id,
            capability=capability,
            launch_mode=launch_mode,
            related_batch_id=related_batch_id,
            idempotency_key=idempotency_key,
        )

    def _move(self, state: WorkOrderState, *, at: datetime | None, **changes: object) -> WorkOrder:
        state = WorkOrderState(state)
        if self.state in {WorkOrderState.COMPLETED, WorkOrderState.CANCELLED}:
            raise InvalidStateTransition(f"work order cannot transition from {self.state} to {state}")
        if state is self.state and not changes:
            return self
        if state is WorkOrderState.RUNNING and self.state is not WorkOrderState.ISSUED:
            raise InvalidStateTransition("only an issued work order can be started")
        if state is WorkOrderState.ISSUED:
            raise InvalidStateTransition("work order cannot return to issued")
        changed_at = as_utc(at or utc_now(), field="work_order.transition_at")
        if changed_at < self.updated_at:
            raise DomainValidationError("work order transition time cannot move backwards")
        return replace(self, state=state, revision=self.revision + 1, updated_at=changed_at, **changes)

    def mark_running(self, *, job_id: str = "", at: datetime | None = None) -> WorkOrder:
        return self._move(WorkOrderState.RUNNING, at=at, related_job_id=job_id or self.related_job_id)

    def complete(self, *, at: datetime | None = None) -> WorkOrder:
        return self._move(WorkOrderState.COMPLETED, at=at)

    def cancel(self, *, at: datetime | None = None) -> WorkOrder:
        return self._move(WorkOrderState.CANCELLED, at=at)
