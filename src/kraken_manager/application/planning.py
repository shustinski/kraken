"""Plan commands using Kraken authorization and its existing optimistic event store.

Plans are durable event streams carrying versioned snapshots. No parallel job
queue, work-order repository or database schema is introduced.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from datetime import datetime
from enum import Enum

from kraken_manager.application.authorization import AuthorizationPolicy
from kraken_manager.application.errors import ConcurrencyError, ConflictError, NotFoundError
from kraken_manager.application.ports import EventStore, StorageProfile
from kraken_manager.domain.events import ActorSnapshot, EventEnvelope
from kraken_manager.domain.identity import Permission, Principal, ProjectRole
from kraken_manager.domain.planning import PlanRevisionConflict, WorkPlan
from kraken_manager.domain.work_orders import WorkOrder
from kraken_manager.domain.workflows import PluginJob
from kraken_manager.domain.common import DomainValidationError


class WorkPlanningService:
    def __init__(
        self, events: EventStore, authorization: AuthorizationPolicy | None = None, *, projections=None
    ) -> None:
        self.events = events
        self.authorization = authorization or AuthorizationPolicy()
        self.projections = projections

    @staticmethod
    def stream_id(plan_id: str) -> str:
        from kraken_manager.domain.common import validate_uuid

        return f"work-plan:{validate_uuid(plan_id)}"

    def load(self, plan_id: str) -> WorkPlan:
        events = self.events.load_stream(self.stream_id(plan_id))
        if not events:
            raise NotFoundError("Work plan not found")
        plan = WorkPlan.from_dict(events[-1].payload)
        if plan.id != plan_id or events[-1].revision != plan.revision + 1:
            raise DomainValidationError("work plan stream identity/revision mismatch")
        return plan

    def create(
        self,
        plan: WorkPlan,
        *,
        principal: Principal,
        storage: StorageProfile,
        roles: tuple[ProjectRole, ...] = (),
        gitlab_identity_verified: bool = False,
        idempotency_key: str | None = None,
    ) -> WorkPlan:
        self.authorization.require(
            principal=principal,
            storage=storage,
            permission=Permission.ASSIGN_WORK,
            roles=roles,
            gitlab_identity_verified=gitlab_identity_verified,
        )
        from kraken_manager.domain.planning import WorkPlanState

        if (
            plan.revision != 0
            or plan.items
            or plan.baselines
            or plan.template_applications
            or plan.dependencies
            or plan.milestones
            or plan.state is not WorkPlanState.DRAFT
        ):
            raise DomainValidationError("create requires a newly created empty plan")
        signature = self._command("create", {"plan_id": plan.id, "project_id": plan.project_id, "title": plan.title})
        replay = self._replay(plan, idempotency_key, signature)
        if replay is not None:
            return replay
        event = self._event(plan, principal, "work_plan.created", idempotency_key, signature)
        self.events.append(event.stream_id, expected_revision=0, events=(event,))
        return plan

    @staticmethod
    def _command(operation: str, arguments: Mapping[str, object]) -> dict[str, object]:
        def encode(value: object) -> object:
            if isinstance(value, datetime):
                return value.isoformat()
            if isinstance(value, Enum):
                return value.value
            if is_dataclass(value):
                return {f.name: encode(getattr(value, f.name)) for f in fields(value)}
            if isinstance(value, Mapping):
                return {k: encode(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return tuple(encode(v) for v in value)
            if isinstance(value, (set, frozenset)):
                return tuple(sorted((encode(v) for v in value), key=repr))
            return value

        return {"operation": operation, "arguments": encode(arguments)}

    def _replay(self, plan: WorkPlan, key: str | None, command: dict[str, object]) -> WorkPlan | None:
        if key is None:
            return None
        from kraken_manager.domain.common import require_non_empty

        require_non_empty(key, field="idempotency_key")
        events = self.events.find_by_idempotency_key(plan.project_id, key)
        if not events:
            return None
        event = events[0]
        if event.stream_id != self.stream_id(plan.id) or event.payload.get("command") != command:
            raise ConflictError("Idempotency key belongs to a different plan command")
        return WorkPlan.from_dict(event.payload)

    def _event(
        self,
        plan: WorkPlan,
        principal: Principal,
        event_type: str,
        idempotency_key: str | None = None,
        command: dict[str, object] | None = None,
    ) -> EventEnvelope:
        payload = plan.to_dict()
        payload["command"] = command
        return EventEnvelope.create(
            stream_id=self.stream_id(plan.id),
            project_id=plan.project_id,
            revision=plan.revision + 1,
            event_type=event_type,
            payload=payload,
            actor=ActorSnapshot.from_principal(principal),
            idempotency_key=idempotency_key,
            recorded_at=plan.updated_at,
        )

    def execute(
        self,
        plan_id: str,
        *,
        expected_revision: int,
        principal: Principal,
        storage: StorageProfile,
        operation: str,
        arguments: dict[str, object] | None = None,
        roles: tuple[ProjectRole, ...] = (),
        gitlab_identity_verified: bool = False,
        idempotency_key: str | None = None,
    ) -> WorkPlan:
        # Explicit command allowlist also fixes the permission independently of
        # caller-supplied metadata. Acceptance cannot be disguised as assignment.
        permissions = {
            "add_item": Permission.ASSIGN_WORK,
            "apply_template": Permission.ASSIGN_WORK,
            "revise_item": Permission.ASSIGN_WORK,
            "assign": Permission.ASSIGN_WORK,
            "add_dependency": Permission.ASSIGN_WORK,
            "add_milestone": Permission.ASSIGN_WORK,
            "capture_baseline": Permission.ASSIGN_WORK,
            "transition": Permission.ASSIGN_WORK,
            "link_execution": Permission.ASSIGN_WORK,
            "record_progress": Permission.ASSIGN_WORK,
            "move_item": Permission.ASSIGN_WORK,
            "satisfy_criterion": Permission.ACCEPT_REVIEW,
        }
        if operation not in permissions:
            raise DomainValidationError("unknown plan command")
        args = dict(arguments or {})
        permission = permissions[operation]
        if operation == "move_item":
            from kraken_manager.domain.planning import WorkItemState

            if args.get("state") == WorkItemState.ACCEPTED:
                permission = Permission.ACCEPT_REVIEW
        self.authorization.require(
            principal=principal,
            storage=storage,
            permission=permission,
            roles=roles,
            gitlab_identity_verified=gitlab_identity_verified,
        )
        plan = self.load(plan_id)
        signature = self._command(operation, args)
        replay = self._replay(plan, idempotency_key, signature)
        if replay is not None:
            return replay
        if plan.revision != expected_revision:
            raise ConcurrencyError(f"Expected revision {expected_revision}, found {plan.revision}")
        if operation == "satisfy_criterion" or (operation == "move_item" and args.get("state") == "accepted"):
            from kraken_manager.application.quality import QualityService

            item = plan.item(args["item_id"])
            evidence = (
                tuple(args.get("evidence_version_ids", ()))
                if operation == "satisfy_criterion"
                else tuple(version_id for criterion in item.criteria for version_id in criterion.evidence_version_ids)
            )
            QualityService.require_clear_inputs(
                self.events, plan.project_id, evidence, scope=item.scope, projections=self.projections
            )
        if operation == "satisfy_criterion":
            args["reviewer_id"] = principal.id
        if operation == "link_execution":
            order = args.pop("order", None)
            job = args.pop("job", None)
            if "work_order_id" in args or "plugin_job_id" in args:
                raise DomainValidationError("resolve execution references before linking")
            self.validate_execution_reference(plan, args.get("item_id"), order=order, job=job)
            args["work_order_id"] = order.id if order else None
            args["plugin_job_id"] = job.id if job else None
        try:
            changed = getattr(plan, operation)(expected_revision=expected_revision, **args)
        except PlanRevisionConflict as exc:
            raise ConcurrencyError(str(exc)) from exc
        if changed == plan:
            return plan
        event = self._event(changed, principal, f"work_plan.{operation}", idempotency_key, signature)
        # EventStore.append supplies the atomic compare-and-swap for concurrent
        # writers; the in-memory revision check alone is deliberately insufficient.
        self.events.append(event.stream_id, expected_revision=plan.revision + 1, events=(event,))
        return changed

    @staticmethod
    def validate_execution_reference(
        plan: WorkPlan, item_id: str, *, order: WorkOrder | None = None, job: PluginJob | None = None
    ) -> None:
        """Adapters must resolve references before linking existing executions."""
        item = plan.item(item_id)
        if order is None and job is None:
            raise DomainValidationError("resolved execution reference required")
        if order is not None and not isinstance(order, WorkOrder):
            raise DomainValidationError("order must be a resolved WorkOrder")
        if job is not None and not isinstance(job, PluginJob):
            raise DomainValidationError("job must be a resolved PluginJob")
        for execution in (order, job):
            if execution is None:
                continue
            if execution.project_id != plan.project_id:
                raise DomainValidationError("execution belongs to a different project")
            if item.scope.layer_ids and execution.layer_id not in item.scope.layer_ids:
                raise DomainValidationError("execution layer is outside the work scope")
        if order is not None and item.assignment and order.assignee_id != item.assignment.performer_id:
            raise DomainValidationError("work order has a different assignee")
