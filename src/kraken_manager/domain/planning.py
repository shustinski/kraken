"""Engineering plans, separate from issued WorkOrders and computational PluginJobs.

The aggregate owns dependency and acceptance invariants. Existing job and order
identities are references: their lifecycle remains in workflows/work_orders.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
from datetime import datetime
from enum import StrEnum
from math import isfinite

from .common import (
    DomainValidationError,
    InvalidStateTransition,
    as_utc,
    new_uuid,
    require_non_empty,
    utc_now,
    validate_uuid,
)


class PlanRevisionConflict(DomainValidationError):
    """The caller edited a stale aggregate revision."""


class WorkItemState(StrEnum):
    PLANNED = "planned"
    RUNNING = "running"
    BLOCKED = "blocked"
    IN_REVIEW = "in_review"
    ACCEPTED = "accepted"
    CANCELLED = "cancelled"


class WorkPlanState(StrEnum):
    DRAFT = "draft"
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


WORK_ITEM_TRANSITIONS = {
    WorkItemState.PLANNED: frozenset({WorkItemState.RUNNING, WorkItemState.BLOCKED, WorkItemState.CANCELLED}),
    WorkItemState.RUNNING: frozenset({WorkItemState.BLOCKED, WorkItemState.IN_REVIEW, WorkItemState.CANCELLED}),
    WorkItemState.BLOCKED: frozenset({WorkItemState.PLANNED, WorkItemState.RUNNING, WorkItemState.CANCELLED}),
    WorkItemState.IN_REVIEW: frozenset({WorkItemState.ACCEPTED, WorkItemState.RUNNING, WorkItemState.CANCELLED}),
    WorkItemState.ACCEPTED: frozenset({WorkItemState.RUNNING}),
    WorkItemState.CANCELLED: frozenset(),
}
WORK_PLAN_TRANSITIONS = {
    WorkPlanState.DRAFT: frozenset({WorkPlanState.ACTIVE, WorkPlanState.CANCELLED}),
    WorkPlanState.ACTIVE: frozenset({WorkPlanState.COMPLETED, WorkPlanState.CANCELLED}),
    WorkPlanState.COMPLETED: frozenset(),
    WorkPlanState.CANCELLED: frozenset(),
}


def _uuid(value: str) -> str:
    return validate_uuid(value)


def _integer(value: int, name: str, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise DomainValidationError(f"{name} must be an integer >= {minimum}")


def _number(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value < 0:
        raise DomainValidationError(f"{name} must be a finite nonnegative number")


@dataclass(frozen=True, slots=True)
class WorkScope:
    layer_ids: tuple[str, ...] = ()
    frame_ids: tuple[str, ...] = ()
    # Rectangle in the layer's coordinate system; no implicit registration.
    region: tuple[float, float, float, float] | None = None
    coordinate_system: str | None = None

    def __post_init__(self) -> None:
        for name in ("layer_ids", "frame_ids"):
            ids = tuple(_uuid(value) for value in getattr(self, name))
            if len(ids) != len(set(ids)):
                raise DomainValidationError(f"scope.{name} must be unique")
            object.__setattr__(self, name, ids)
        if self.frame_ids and not self.layer_ids:
            raise DomainValidationError("frame scope requires layers")
        if self.region is not None:
            region = tuple(self.region)
            if len(region) != 4 or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not isfinite(v) for v in region
            ):
                raise DomainValidationError("region requires four finite coordinates")
            if region[2] <= region[0] or region[3] <= region[1] or len(self.layer_ids) != 1:
                raise DomainValidationError("region requires positive extent in exactly one layer")
            if not self.coordinate_system:
                raise DomainValidationError("region requires an explicit coordinate system")
            object.__setattr__(self, "region", region)
        if self.coordinate_system is not None:
            object.__setattr__(
                self, "coordinate_system", require_non_empty(self.coordinate_system, field="coordinate_system")
            )


@dataclass(frozen=True, slots=True)
class WorkAssignment:
    performer_id: str
    principal_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "performer_id", _uuid(self.performer_id))
        if self.principal_id is not None:
            object.__setattr__(self, "principal_id", _uuid(self.principal_id))


@dataclass(frozen=True, slots=True)
class WorkEstimate:
    effort_hours: float = 0
    planned_start: datetime | None = None
    planned_finish: datetime | None = None

    def __post_init__(self) -> None:
        _number(self.effort_hours, "effort_hours")
        for name in ("planned_start", "planned_finish"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, as_utc(value, field=name))
        if self.planned_start and self.planned_finish and self.planned_finish < self.planned_start:
            raise DomainValidationError("planned finish cannot precede start")


@dataclass(frozen=True, slots=True)
class WorkProgress:
    fraction: float = 0
    actual_effort_hours: float = 0
    started_at: datetime | None = None
    finished_at: datetime | None = None

    def __post_init__(self) -> None:
        _number(self.fraction, "fraction")
        _number(self.actual_effort_hours, "actual_effort_hours")
        if self.fraction > 1:
            raise DomainValidationError("progress fraction must be <= 1")
        for name in ("started_at", "finished_at"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, as_utc(value, field=name))
        if self.finished_at and (not self.started_at or self.finished_at < self.started_at):
            raise DomainValidationError("actual finish requires an earlier start")


@dataclass(frozen=True, slots=True)
class AcceptanceCriterion:
    id: str
    description: str
    required: bool = True
    satisfied: bool = False
    evidence_version_ids: tuple[str, ...] = ()
    reviewed_by: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _uuid(self.id))
        object.__setattr__(
            self, "description", require_non_empty(self.description, field="criterion.description", maximum=4000)
        )
        if not isinstance(self.required, bool) or not isinstance(self.satisfied, bool):
            raise DomainValidationError("criterion flags must be boolean")
        versions = tuple(_uuid(v) for v in self.evidence_version_ids)
        if len(versions) != len(set(versions)):
            raise DomainValidationError("criterion evidence versions must be unique")
        object.__setattr__(self, "evidence_version_ids", versions)
        if self.reviewed_by is not None:
            object.__setattr__(self, "reviewed_by", _uuid(self.reviewed_by))
        if self.satisfied and (not self.evidence_version_ids or self.reviewed_by is None):
            raise DomainValidationError("satisfied criterion requires versioned evidence and a reviewer")
        if not self.satisfied and (self.evidence_version_ids or self.reviewed_by):
            raise DomainValidationError("unsatisfied criterion cannot carry acceptance evidence")


@dataclass(frozen=True, slots=True)
class WorkDependency:
    predecessor_id: str
    successor_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "predecessor_id", _uuid(self.predecessor_id))
        object.__setattr__(self, "successor_id", _uuid(self.successor_id))
        if self.predecessor_id == self.successor_id:
            raise DomainValidationError("work cannot depend on itself")


@dataclass(frozen=True, slots=True)
class WorkItem:
    id: str
    title: str
    scope: WorkScope = WorkScope()
    parent_id: str | None = None
    assignment: WorkAssignment | None = None
    priority: int = 0
    estimate: WorkEstimate = WorkEstimate()
    progress: WorkProgress = WorkProgress()
    state: WorkItemState = WorkItemState.PLANNED
    criteria: tuple[AcceptanceCriterion, ...] = ()
    blocking_reason: str | None = None
    attempt: int = 0
    work_order_ids: tuple[str, ...] = ()
    plugin_job_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _uuid(self.id))
        object.__setattr__(self, "title", require_non_empty(self.title, field="item.title"))
        object.__setattr__(self, "state", WorkItemState(self.state))
        if self.parent_id is not None:
            object.__setattr__(self, "parent_id", _uuid(self.parent_id))
        _integer(self.priority, "priority")
        _integer(self.attempt, "attempt")
        for name, cls in (("scope", WorkScope), ("estimate", WorkEstimate), ("progress", WorkProgress)):
            if not isinstance(getattr(self, name), cls):
                raise DomainValidationError(f"item.{name} has invalid type")
        if self.assignment is not None and not isinstance(self.assignment, WorkAssignment):
            raise DomainValidationError("item.assignment has invalid type")
        criteria = tuple(self.criteria)
        if any(not isinstance(c, AcceptanceCriterion) for c in criteria) or len({c.id for c in criteria}) != len(
            criteria
        ):
            raise DomainValidationError("item criteria must have unique identities")
        object.__setattr__(self, "criteria", criteria)
        if self.state not in {WorkItemState.IN_REVIEW, WorkItemState.ACCEPTED, WorkItemState.CANCELLED} and any(
            c.satisfied for c in criteria
        ):
            raise DomainValidationError("acceptance evidence requires review or accepted work")
        if self.state is WorkItemState.PLANNED and (
            self.progress.started_at is not None or self.progress.fraction != 0 or self.attempt != 0
        ):
            raise DomainValidationError("planned work cannot carry execution progress")
        for name in ("work_order_ids", "plugin_job_ids"):
            values = tuple(_uuid(v) for v in getattr(self, name))
            if len(values) != len(set(values)):
                raise DomainValidationError(f"item.{name} must be unique")
            object.__setattr__(self, name, values)
        if self.state is WorkItemState.BLOCKED:
            object.__setattr__(
                self,
                "blocking_reason",
                require_non_empty(self.blocking_reason or "", field="blocking_reason", maximum=4000),
            )
        elif self.blocking_reason is not None:
            raise DomainValidationError("only blocked work can have a blocking reason")
        executed = self.state in {WorkItemState.RUNNING, WorkItemState.IN_REVIEW, WorkItemState.ACCEPTED}
        if executed and (self.assignment is None or self.progress.started_at is None or self.attempt < 1):
            raise DomainValidationError("executed work requires assignment, start time and attempt")
        if self.state in {WorkItemState.IN_REVIEW, WorkItemState.ACCEPTED} and self.progress.fraction != 1:
            raise DomainValidationError("review requires complete execution progress")
        if self.state is WorkItemState.ACCEPTED:
            if self.progress.finished_at is None or any(c.required and not c.satisfied for c in criteria):
                raise DomainValidationError("accepted work requires finish time and satisfied required criteria")
        elif self.progress.finished_at is not None:
            raise DomainValidationError("only accepted work carries an acceptance finish time")

    @classmethod
    def create(cls, title: str, **kwargs: object) -> WorkItem:
        return cls(id=new_uuid(), title=title, **kwargs)


@dataclass(frozen=True, slots=True)
class ProjectMilestone:
    id: str
    title: str
    item_ids: tuple[str, ...]
    due_at: datetime | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _uuid(self.id))
        object.__setattr__(self, "title", require_non_empty(self.title, field="milestone.title"))
        ids = tuple(_uuid(v) for v in self.item_ids)
        if not ids or len(ids) != len(set(ids)):
            raise DomainValidationError("milestone requires unique work items")
        object.__setattr__(self, "item_ids", ids)
        if self.due_at is not None:
            object.__setattr__(self, "due_at", as_utc(self.due_at))


@dataclass(frozen=True, slots=True)
class PlanBaseline:
    id: str
    source_revision: int
    captured_at: datetime
    items: tuple[WorkItem, ...]
    dependencies: tuple[WorkDependency, ...]
    milestones: tuple[ProjectMilestone, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _uuid(self.id))
        _integer(self.source_revision, "source_revision")
        object.__setattr__(self, "captured_at", as_utc(self.captured_at))
        for name in ("items", "dependencies", "milestones"):
            object.__setattr__(self, name, tuple(getattr(self, name)))


@dataclass(frozen=True, slots=True)
class AppliedPlanTemplate:
    application_id: str
    template_id: str
    template_version: int
    template_digest: str
    item_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("application_id", "template_id"):
            object.__setattr__(self, name, _uuid(getattr(self, name)))
        _integer(self.template_version, "template_version", minimum=1)
        if len(self.template_digest) != 64 or any(c not in "0123456789abcdef" for c in self.template_digest):
            raise DomainValidationError("applied template requires a SHA256 digest")
        ids = tuple(_uuid(value) for value in self.item_ids)
        if not ids or len(set(ids)) != len(ids):
            raise DomainValidationError("applied template requires unique work items")
        object.__setattr__(self, "item_ids", ids)


@dataclass(frozen=True, slots=True)
class WorkPlan:
    id: str
    project_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    revision: int = 0
    state: WorkPlanState = WorkPlanState.DRAFT
    items: tuple[WorkItem, ...] = ()
    dependencies: tuple[WorkDependency, ...] = ()
    milestones: tuple[ProjectMilestone, ...] = ()
    baselines: tuple[PlanBaseline, ...] = ()
    template_applications: tuple[AppliedPlanTemplate, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _uuid(self.id))
        object.__setattr__(self, "project_id", _uuid(self.project_id))
        object.__setattr__(self, "title", require_non_empty(self.title, field="plan.title"))
        object.__setattr__(self, "state", WorkPlanState(self.state))
        _integer(self.revision, "plan.revision")
        for name in ("created_at", "updated_at"):
            object.__setattr__(self, name, as_utc(getattr(self, name), field=name))
        if self.updated_at < self.created_at:
            raise DomainValidationError("plan update cannot precede creation")
        for name in ("items", "dependencies", "milestones", "baselines", "template_applications"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        items = {i.id: i for i in self.items}
        if len(items) != len(self.items):
            raise DomainValidationError("work item identities must be unique")
        application_ids = [a.application_id for a in self.template_applications]
        if len(application_ids) != len(set(application_ids)):
            raise DomainValidationError("template application identities must be unique")
        for application in self.template_applications:
            if not set(application.item_ids) <= items.keys():
                raise DomainValidationError("template application references unknown work")
        parents = []
        for item in self.items:
            if item.parent_id is not None:
                if item.parent_id not in items:
                    raise DomainValidationError("work parent is outside the plan")
                parents.append(WorkDependency(item.parent_id, item.id))
        self._validate_dag(parents, items)
        self._validate_dag(self.dependencies, items)
        # A parent cannot be accepted before its children. A dependency from
        # that parent to its child would therefore deadlock even if each of
        # the two graphs were acyclic in isolation.
        acceptance_edges = [WorkDependency(e.successor_id, e.predecessor_id) for e in parents]
        self._validate_dag([*self.dependencies, *acceptance_edges], items)
        if len(set(self.dependencies)) != len(self.dependencies):
            raise DomainValidationError("dependencies must be unique")
        if len({m.id for m in self.milestones}) != len(self.milestones):
            raise DomainValidationError("milestone identities must be unique")
        for milestone in self.milestones:
            if not set(milestone.item_ids) <= items.keys():
                raise DomainValidationError("milestone references unknown work")
        if len({b.id for b in self.baselines}) != len(self.baselines):
            raise DomainValidationError("baseline identities must be unique")
        for baseline in self.baselines:
            if (
                baseline.source_revision >= self.revision
                or not self.created_at <= baseline.captured_at <= self.updated_at
            ):
                raise DomainValidationError("baseline must reference an earlier plan revision and time")
        if self.state is WorkPlanState.COMPLETED and (
            not items or any(i.state is not WorkItemState.ACCEPTED for i in self.items)
        ):
            raise DomainValidationError("completed plan requires all work accepted")
        for dependency in self.dependencies:
            successor = items[dependency.successor_id]
            if successor.state in {WorkItemState.RUNNING, WorkItemState.IN_REVIEW, WorkItemState.ACCEPTED}:
                if items[dependency.predecessor_id].state is not WorkItemState.ACCEPTED:
                    raise DomainValidationError("started work requires accepted dependencies")
        for item in self.items:
            if item.state is WorkItemState.ACCEPTED and any(
                c.parent_id == item.id and c.state is not WorkItemState.ACCEPTED for c in self.items
            ):
                raise DomainValidationError("accepted parent requires accepted children")

    @staticmethod
    def _validate_dag(edges: tuple[WorkDependency, ...] | list[WorkDependency], items: dict[str, WorkItem]) -> None:
        successors = {key: [] for key in items}
        indegree = dict.fromkeys(items, 0)
        for edge in edges:
            if edge.predecessor_id not in items or edge.successor_id not in items:
                raise DomainValidationError("dependency references unknown work")
            successors[edge.predecessor_id].append(edge.successor_id)
            indegree[edge.successor_id] += 1
        ready = [key for key, value in indegree.items() if value == 0]
        count = 0
        while ready:
            key = ready.pop()
            count += 1
            for successor in successors[key]:
                indegree[successor] -= 1
                if indegree[successor] == 0:
                    ready.append(successor)
        if count != len(items):
            raise DomainValidationError("work graph contains a cycle")

    @classmethod
    def create(cls, project_id: str, title: str, *, at: datetime | None = None, plan_id: str | None = None) -> WorkPlan:
        now = at or utc_now()
        return cls(id=plan_id or new_uuid(), project_id=project_id, title=title, created_at=now, updated_at=now)

    def item(self, item_id: str) -> WorkItem:
        for item in self.items:
            if item.id == item_id:
                return item
        raise DomainValidationError("work item does not belong to this plan")

    def _change(self, expected_revision: int, at: datetime | None, **changes: object) -> WorkPlan:
        _integer(expected_revision, "expected_revision")
        if expected_revision != self.revision:
            raise PlanRevisionConflict(f"expected revision {expected_revision}, found {self.revision}")
        if self.state in {WorkPlanState.COMPLETED, WorkPlanState.CANCELLED}:
            raise InvalidStateTransition("terminal plans cannot be changed")
        now = as_utc(at or utc_now())
        if now < self.updated_at:
            raise DomainValidationError("plan time cannot move backwards")
        return replace(self, revision=self.revision + 1, updated_at=now, **changes)

    def add_item(self, item: WorkItem, *, expected_revision: int, at: datetime | None = None) -> WorkPlan:
        if item.state is not WorkItemState.PLANNED:
            raise DomainValidationError("new work must be planned")
        return self._change(expected_revision, at, items=(*self.items, item))

    def apply_template(self, materialization, *, expected_revision: int, at: datetime | None = None) -> WorkPlan:
        from .planning_templates import TemplateMaterialization

        if not isinstance(materialization, TemplateMaterialization) or materialization.project_id != self.project_id:
            raise DomainValidationError("template materialization belongs to a different project")
        if any(i.state is not WorkItemState.PLANNED for i in materialization.items):
            raise DomainValidationError("templates can only create planned work")
        application = AppliedPlanTemplate(
            materialization.application_id,
            materialization.template_id,
            materialization.template_version,
            materialization.template_digest,
            tuple(i.id for i in materialization.items),
        )
        # Aggregate validation checks the complete expanded hierarchy and DAG;
        # one revision/event makes partial materialization impossible.
        return self._change(
            expected_revision,
            at,
            items=(*self.items, *materialization.items),
            dependencies=(*self.dependencies, *materialization.dependencies),
            template_applications=(*self.template_applications, application),
        )

    def add_dependency(
        self, dependency: WorkDependency, *, expected_revision: int, at: datetime | None = None
    ) -> WorkPlan:
        return self._change(expected_revision, at, dependencies=(*self.dependencies, dependency))

    def add_milestone(
        self, milestone: ProjectMilestone, *, expected_revision: int, at: datetime | None = None
    ) -> WorkPlan:
        return self._change(expected_revision, at, milestones=(*self.milestones, milestone))

    def capture_baseline(self, *, expected_revision: int, at: datetime | None = None) -> WorkPlan:
        now = at or utc_now()
        baseline = PlanBaseline(new_uuid(), self.revision, now, self.items, self.dependencies, self.milestones)
        return self._change(expected_revision, now, baselines=(*self.baselines, baseline))

    def transition(self, state: WorkPlanState, *, expected_revision: int, at: datetime | None = None) -> WorkPlan:
        state = WorkPlanState(state)
        if state not in WORK_PLAN_TRANSITIONS[self.state]:
            raise InvalidStateTransition(f"plan cannot transition from {self.state} to {state}")
        if state is WorkPlanState.ACTIVE and not self.items:
            raise DomainValidationError("cannot activate an empty plan")
        changes: dict[str, object] = {"state": state}
        if state is WorkPlanState.CANCELLED:
            changes["items"] = tuple(
                replace(i, state=WorkItemState.CANCELLED, blocking_reason=None)
                if i.state is not WorkItemState.ACCEPTED
                else i
                for i in self.items
            )
        return self._change(expected_revision, at, **changes)

    def _with_item(self, item: WorkItem, expected_revision: int, at: datetime | None) -> WorkPlan:
        return self._change(
            expected_revision, at, items=tuple(item if old.id == item.id else old for old in self.items)
        )

    def assign(
        self, item_id: str, assignment: WorkAssignment, *, expected_revision: int, at: datetime | None = None
    ) -> WorkPlan:
        item = self.item(item_id)
        if item.state not in {WorkItemState.PLANNED, WorkItemState.BLOCKED}:
            raise InvalidStateTransition("only planned or blocked work may be reassigned")
        return self._with_item(replace(item, assignment=assignment), expected_revision, at)

    def revise_item(
        self,
        item_id: str,
        *,
        expected_revision: int,
        title: str | None = None,
        scope: WorkScope | None = None,
        estimate: WorkEstimate | None = None,
        priority: int | None = None,
        criteria: tuple[AcceptanceCriterion, ...] | None = None,
        at: datetime | None = None,
    ) -> WorkPlan:
        """Revise planned work without rewriting completed execution history."""
        item = self.item(item_id)
        if item.state not in {WorkItemState.PLANNED, WorkItemState.BLOCKED} or item.progress.started_at is not None:
            raise InvalidStateTransition("only unstarted work can be replanned")
        changes = {
            name: value
            for name, value in (
                ("title", title),
                ("scope", scope),
                ("estimate", estimate),
                ("priority", priority),
                ("criteria", criteria),
            )
            if value is not None
        }
        if not changes:
            raise DomainValidationError("work revision requires at least one change")
        if criteria is not None and any(c.satisfied for c in criteria):
            raise DomainValidationError("new criteria cannot carry acceptance evidence")
        return self._with_item(replace(item, **changes), expected_revision, at)

    def move_item(
        self,
        item_id: str,
        state: WorkItemState,
        *,
        expected_revision: int,
        at: datetime | None = None,
        reason: str | None = None,
    ) -> WorkPlan:
        if self.state is not WorkPlanState.ACTIVE:
            raise InvalidStateTransition("work execution requires an active plan")
        item = self.item(item_id)
        state = WorkItemState(state)
        if state not in WORK_ITEM_TRANSITIONS[item.state]:
            raise InvalidStateTransition(f"work cannot transition from {item.state} to {state}")
        now = as_utc(at or utc_now())
        progress = item.progress
        criteria = item.criteria
        attempt = item.attempt
        if state is WorkItemState.RUNNING:
            if item.state in {WorkItemState.PLANNED, WorkItemState.ACCEPTED, WorkItemState.IN_REVIEW}:
                attempt += 1
                progress = WorkProgress(actual_effort_hours=progress.actual_effort_hours, started_at=now)
                criteria = tuple(
                    replace(c, satisfied=False, evidence_version_ids=(), reviewed_by=None) for c in criteria
                )
            elif progress.started_at is None:
                attempt += 1
                progress = replace(progress, started_at=now)
        if state is WorkItemState.PLANNED and progress.started_at is not None:
            raise InvalidStateTransition("started blocked work must resume running")
        if state is WorkItemState.ACCEPTED:
            progress = replace(progress, finished_at=now)
        updated = replace(
            item,
            state=state,
            progress=progress,
            criteria=criteria,
            attempt=attempt,
            blocking_reason=reason if state is WorkItemState.BLOCKED else None,
        )
        return self._with_item(updated, expected_revision, now)

    def record_progress(
        self,
        item_id: str,
        fraction: float,
        actual_effort_hours: float,
        *,
        expected_revision: int,
        at: datetime | None = None,
    ) -> WorkPlan:
        item = self.item(item_id)
        if self.state is not WorkPlanState.ACTIVE or item.state is not WorkItemState.RUNNING:
            raise InvalidStateTransition("progress may only be reported for running work")
        if fraction < item.progress.fraction or actual_effort_hours < item.progress.actual_effort_hours:
            raise DomainValidationError("progress and cumulative effort cannot decrease")
        progress = replace(item.progress, fraction=fraction, actual_effort_hours=actual_effort_hours)
        return self._with_item(replace(item, progress=progress), expected_revision, at)

    def satisfy_criterion(
        self,
        item_id: str,
        criterion_id: str,
        *,
        reviewer_id: str,
        evidence_version_ids: tuple[str, ...],
        expected_revision: int,
        at: datetime | None = None,
    ) -> WorkPlan:
        item = self.item(item_id)
        if self.state is not WorkPlanState.ACTIVE or item.state is not WorkItemState.IN_REVIEW:
            raise InvalidStateTransition("acceptance evidence requires work in review")
        if not any(c.id == criterion_id for c in item.criteria):
            raise DomainValidationError("unknown acceptance criterion")
        criteria = tuple(
            replace(c, satisfied=True, reviewed_by=reviewer_id, evidence_version_ids=evidence_version_ids)
            if c.id == criterion_id
            else c
            for c in item.criteria
        )
        return self._with_item(replace(item, criteria=criteria), expected_revision, at)

    def link_execution(
        self,
        item_id: str,
        *,
        work_order_id: str | None = None,
        plugin_job_id: str | None = None,
        expected_revision: int,
        at: datetime | None = None,
    ) -> WorkPlan:
        item = self.item(item_id)
        if item.state in {WorkItemState.ACCEPTED, WorkItemState.CANCELLED}:
            raise InvalidStateTransition("terminal work cannot acquire execution references")
        if work_order_id is None and plugin_job_id is None:
            raise DomainValidationError("execution reference required")
        item = replace(
            item,
            work_order_ids=(*item.work_order_ids, work_order_id) if work_order_id else item.work_order_ids,
            plugin_job_ids=(*item.plugin_job_ids, plugin_job_id) if plugin_job_id else item.plugin_job_ids,
        )
        return self._with_item(item, expected_revision, at)

    @property
    def ready_items(self) -> tuple[WorkItem, ...]:
        if self.state is not WorkPlanState.ACTIVE:
            return ()
        accepted = {i.id for i in self.items if i.state is WorkItemState.ACCEPTED}
        return tuple(
            i
            for i in self.items
            if i.state is WorkItemState.PLANNED
            and i.assignment is not None
            and all(d.predecessor_id in accepted for d in self.dependencies if d.successor_id == i.id)
        )

    def milestone_reached(self, milestone_id: str) -> bool:
        milestone = next((m for m in self.milestones if m.id == milestone_id), None)
        if milestone is None:
            raise DomainValidationError("unknown milestone")
        return all(self.item(key).state is WorkItemState.ACCEPTED for key in milestone.item_ids)

    def to_dict(self) -> dict[str, object]:
        """Versioned JSON payload for the existing event store (no new DB schema)."""

        def encode(value: object) -> object:
            if isinstance(value, datetime):
                return value.isoformat()
            if isinstance(value, StrEnum):
                return value.value
            if is_dataclass(value):
                return {f.name: encode(getattr(value, f.name)) for f in fields(value)}
            if isinstance(value, tuple):
                return [encode(v) for v in value]
            return value

        return {"schema_version": 1, "plan": encode(self)}

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> WorkPlan:
        if payload.get("schema_version") != 1:
            raise DomainValidationError("unsupported work plan schema version")

        def dates(data: dict, names: tuple[str, ...]) -> dict:
            values = dict(data)
            for name in names:
                if values.get(name) is not None:
                    values[name] = datetime.fromisoformat(values[name])
            return values

        def item(data: dict) -> WorkItem:
            values = dict(data)
            values["scope"] = WorkScope(**values["scope"])
            values["assignment"] = WorkAssignment(**values["assignment"]) if values["assignment"] else None
            values["estimate"] = WorkEstimate(**dates(values["estimate"], ("planned_start", "planned_finish")))
            values["progress"] = WorkProgress(**dates(values["progress"], ("started_at", "finished_at")))
            values["criteria"] = tuple(AcceptanceCriterion(**v) for v in values["criteria"])
            return WorkItem(**values)

        def collection(values: dict) -> dict:
            result = dict(values)
            result["items"] = tuple(item(v) for v in result["items"])
            result["dependencies"] = tuple(WorkDependency(**v) for v in result["dependencies"])
            result["milestones"] = tuple(ProjectMilestone(**dates(v, ("due_at",))) for v in result["milestones"])
            return result

        values = collection(dates(payload["plan"], ("created_at", "updated_at")))
        values["template_applications"] = tuple(
            AppliedPlanTemplate(**v) for v in values.get("template_applications", ())
        )
        values["baselines"] = tuple(PlanBaseline(**collection(dates(v, ("captured_at",)))) for v in values["baselines"])
        return cls(**values)
