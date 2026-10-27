"""Persist performer work with the same transaction and event log as artifacts."""

from __future__ import annotations

from collections.abc import Callable

from kraken_manager.application.authorization import AuthorizationPolicy
from kraken_manager.application.errors import ConcurrencyError, NotFoundError
from kraken_manager.domain.events import ActorSnapshot, EventEnvelope, ProgramSnapshot
from kraken_manager.domain.identity import Permission
from kraken_manager.domain.work_orders import WorkOrder


def record_work_order(
    *, uow_factory, profiles, principal, project_id, idempotency_key: str,
    operation: str, command: object, change: Callable[[object], WorkOrder], encode: Callable, decode: Callable,
    gitlab_identity_verified: bool = False,
) -> WorkOrder:
    """Authorize, deduplicate, append, and project one work-order change.

    ``change`` reads and checks the current aggregate inside the transaction.
    Event-stream compare-and-swap protects PostgreSQL callers; the filesystem
    unit of work additionally holds the project writer lock.
    """
    with uow_factory() as unit:
        project = unit.projections.get_project(project_id)
        if project is None:
            raise NotFoundError("Project was not found")
        AuthorizationPolicy().require(
            principal=principal, storage=profiles.get(project.storage_profile),
            permission=Permission.ASSIGN_WORK,
            roles=unit.acl.roles_for(project.id, principal.id),
            gitlab_identity_verified=gitlab_identity_verified,
        )
        prior = unit.event_store.find_by_idempotency_key(project.id, idempotency_key)
        if prior:
            event = prior[-1]
            if event.actor.principal_id != principal.id:
                raise ConcurrencyError("Idempotency key belongs to another actor")
            if event.event_type != "WorkOrderRecordedV1" or event.payload.get("operation") != operation:
                raise ConcurrencyError("Idempotency key was already used for another operation")
            # Frozen event JSON uses tuples; compare through the same codec.
            if encode(event.payload.get("command")) != encode(command):
                raise ConcurrencyError("Idempotency key was already used for another command")
            return decode(WorkOrder, event.payload["work_order"])
        order = change(unit.projections)
        if order.project_id != project.id:
            raise ValueError("Work order belongs to another project")
        stream = f"work-order:{order.id}"
        revision = unit.event_store.current_revision(stream)
        current = unit.projections.get_work_order(order.id)
        if current is not None and order.revision != current.revision + 1:
            raise ConcurrencyError("Work order revision changed")
        if current is None and order.revision != 0:
            raise ConcurrencyError("A new work order must start at revision zero")
        event = EventEnvelope.create(
            stream_id=stream, project_id=project.id, revision=revision + 1,
            event_type="WorkOrderRecordedV1",
            payload={"operation": operation, "command": encode(command), "work_order": encode(order)},
            actor=ActorSnapshot.from_principal(principal), performer_id=order.assignee_id,
            program=ProgramSnapshot("Kraken"), idempotency_key=idempotency_key,
        )
        unit.event_store.append(stream, expected_revision=revision, events=(event,))
        unit.projections.save_work_order(order)
        unit.commit()
        return order
