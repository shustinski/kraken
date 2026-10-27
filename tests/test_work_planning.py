from dataclasses import replace
from datetime import UTC, datetime, timedelta
from uuid import uuid4
import json

import pytest

from kraken_manager.domain.common import DomainValidationError, InvalidStateTransition
from kraken_manager.domain.planning import (
    AcceptanceCriterion,
    PlanRevisionConflict,
    ProjectMilestone,
    WorkAssignment,
    WorkDependency,
    WorkEstimate,
    WorkItem,
    WorkItemState,
    WorkPlan,
    WorkPlanState,
    WorkProgress,
    WorkScope,
)

NOW = datetime(2026, 10, 23, tzinfo=UTC)


def uid():
    return str(uuid4())


def change(plan, operation, *args, **kwargs):
    return getattr(plan, operation)(
        *args, expected_revision=plan.revision, at=plan.updated_at + timedelta(seconds=1), **kwargs
    )


def plan_with_items(count=1, criterion=False):
    plan = WorkPlan.create(uid(), "Recovery", at=NOW)
    for index in range(count):
        criteria = (AcceptanceCriterion(uid(), "Engineer verified mask"),) if criterion else ()
        item = WorkItem.create(f"Task {index}", assignment=WorkAssignment(uid()), criteria=criteria)
        plan = change(plan, "add_item", item)
    return plan


def accept(plan, item_id):
    plan = change(plan, "move_item", item_id, WorkItemState.RUNNING)
    plan = change(plan, "record_progress", item_id, 1, 3)
    plan = change(plan, "move_item", item_id, WorkItemState.IN_REVIEW)
    return change(plan, "move_item", item_id, WorkItemState.ACCEPTED)


def test_dependency_readiness_and_accepted_gate():
    plan = plan_with_items(3)
    a, b, independent = plan.items
    plan = change(plan, "add_dependency", WorkDependency(a.id, b.id))
    plan = change(plan, "transition", WorkPlanState.ACTIVE)
    assert {i.id for i in plan.ready_items} == {a.id, independent.id}
    with pytest.raises(DomainValidationError, match="dependencies"):
        change(plan, "move_item", b.id, WorkItemState.RUNNING)
    plan = accept(plan, a.id)
    assert b.id in {i.id for i in plan.ready_items}
    plan = change(plan, "move_item", b.id, WorkItemState.RUNNING)
    with pytest.raises(DomainValidationError, match="dependencies"):
        change(plan, "move_item", a.id, WorkItemState.RUNNING)


def test_cycles_unknown_references_duplicate_and_self_dependencies():
    plan = plan_with_items(3)
    a, b, c = plan.items
    plan = change(plan, "add_dependency", WorkDependency(a.id, b.id))
    plan = change(plan, "add_dependency", WorkDependency(b.id, c.id))
    with pytest.raises(DomainValidationError, match="cycle"):
        change(plan, "add_dependency", WorkDependency(c.id, a.id))
    with pytest.raises(DomainValidationError, match="unknown"):
        change(plan, "add_dependency", WorkDependency(uid(), a.id))
    with pytest.raises(DomainValidationError, match="unique"):
        change(plan, "add_dependency", WorkDependency(a.id, b.id))
    with pytest.raises(DomainValidationError, match="itself"):
        WorkDependency(a.id, a.id)


def test_explicit_acceptance_evidence_and_retry_clears_evidence():
    plan = change(plan_with_items(criterion=True), "transition", WorkPlanState.ACTIVE)
    item = plan.items[0]
    plan = change(plan, "move_item", item.id, WorkItemState.RUNNING)
    with pytest.raises(DomainValidationError, match="complete execution"):
        change(plan, "move_item", item.id, WorkItemState.IN_REVIEW)
    plan = change(plan, "record_progress", item.id, 1, 2)
    plan = change(plan, "move_item", item.id, WorkItemState.IN_REVIEW)
    with pytest.raises(DomainValidationError, match="criteria"):
        change(plan, "move_item", item.id, WorkItemState.ACCEPTED)
    with pytest.raises(DomainValidationError, match="evidence"):
        change(plan, "satisfy_criterion", item.id, item.criteria[0].id, reviewer_id=uid(), evidence_version_ids=())
    plan = change(
        plan, "satisfy_criterion", item.id, item.criteria[0].id, reviewer_id=uid(), evidence_version_ids=(uid(),)
    )
    plan = change(plan, "move_item", item.id, WorkItemState.ACCEPTED)
    assert plan.items[0].progress.finished_at == plan.updated_at
    plan = change(plan, "move_item", item.id, WorkItemState.RUNNING)
    assert plan.items[0].attempt == 2
    assert plan.items[0].progress.fraction == 0
    assert plan.items[0].progress.actual_effort_hours == 2
    assert not plan.items[0].criteria[0].satisfied
    assert not plan.items[0].criteria[0].evidence_version_ids


def test_block_and_resume_does_not_create_new_attempt():
    plan = change(plan_with_items(), "transition", WorkPlanState.ACTIVE)
    item = plan.items[0]
    with pytest.raises(DomainValidationError, match="blocking_reason"):
        change(plan, "move_item", item.id, WorkItemState.BLOCKED)
    plan = change(plan, "move_item", item.id, WorkItemState.RUNNING)
    plan = change(plan, "record_progress", item.id, 0.4, 2)
    plan = change(plan, "move_item", item.id, WorkItemState.BLOCKED, reason="GPU unavailable")
    with pytest.raises(InvalidStateTransition):
        change(plan, "move_item", item.id, WorkItemState.PLANNED)
    plan = change(plan, "move_item", item.id, WorkItemState.RUNNING)
    assert plan.items[0].attempt == 1
    assert plan.items[0].progress.fraction == 0.4
    assert plan.items[0].blocking_reason is None


def test_parent_completion_requires_children_and_hierarchy_cycle_is_rejected():
    plan = plan_with_items()
    parent = plan.items[0]
    child = WorkItem.create("Child", parent_id=parent.id, assignment=WorkAssignment(uid()))
    plan = change(plan, "add_item", child)
    with pytest.raises(DomainValidationError, match="cycle"):
        replace(plan, items=(replace(parent, parent_id=child.id), child))
    plan = change(plan, "transition", WorkPlanState.ACTIVE)
    plan = change(plan, "move_item", parent.id, WorkItemState.RUNNING)
    plan = change(plan, "record_progress", parent.id, 1, 0)
    plan = change(plan, "move_item", parent.id, WorkItemState.IN_REVIEW)
    with pytest.raises(DomainValidationError, match="children"):
        change(plan, "move_item", parent.id, WorkItemState.ACCEPTED)
    plan = accept(plan, child.id)
    plan = change(plan, "move_item", parent.id, WorkItemState.ACCEPTED)
    assert all(i.state is WorkItemState.ACCEPTED for i in plan.items)


def test_baseline_is_immutable_and_json_round_trip_includes_full_history():
    plan = plan_with_items(2)
    plan = change(plan, "add_milestone", ProjectMilestone(uid(), "Masks accepted", (plan.items[0].id,), NOW))
    plan = change(plan, "add_dependency", WorkDependency(plan.items[0].id, plan.items[1].id))
    plan = change(plan, "capture_baseline")
    baseline = plan.baselines[0]
    assert baseline.source_revision == plan.revision - 1
    plan = change(plan, "assign", plan.items[0].id, WorkAssignment(uid()))
    assert baseline.items[0].assignment != plan.items[0].assignment
    payload = json.loads(json.dumps(plan.to_dict()))
    assert WorkPlan.from_dict(payload) == plan
    payload["schema_version"] = 2
    with pytest.raises(DomainValidationError, match="schema"):
        WorkPlan.from_dict(payload)


def test_revision_and_time_reject_stale_or_backward_mutation():
    plan = plan_with_items()
    with pytest.raises(PlanRevisionConflict):
        plan.capture_baseline(expected_revision=0)
    with pytest.raises(DomainValidationError, match="backwards"):
        plan.capture_baseline(expected_revision=plan.revision, at=NOW)
    assert len(plan.baselines) == 0


def test_terminal_plans_and_completion_conditions():
    plan = plan_with_items()
    with pytest.raises(InvalidStateTransition):
        change(plan, "move_item", plan.items[0].id, WorkItemState.RUNNING)
    plan = change(plan, "transition", WorkPlanState.ACTIVE)
    with pytest.raises(DomainValidationError, match="accepted"):
        change(plan, "transition", WorkPlanState.COMPLETED)
    plan = accept(plan, plan.items[0].id)
    plan = change(plan, "transition", WorkPlanState.COMPLETED)
    with pytest.raises(InvalidStateTransition):
        change(plan, "capture_baseline")
    cancelled = change(plan_with_items(), "transition", WorkPlanState.CANCELLED)
    assert cancelled.items[0].state is WorkItemState.CANCELLED


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), True])
def test_nonfinite_or_negative_estimates_and_progress_rejected(value):
    with pytest.raises(DomainValidationError):
        WorkEstimate(effort_hours=value)
    with pytest.raises(DomainValidationError):
        WorkProgress(actual_effort_hours=value)


def test_scope_requires_explicit_coordinate_frame_and_valid_extent():
    layer = uid()
    with pytest.raises(DomainValidationError, match="coordinate"):
        WorkScope(layer_ids=(layer,), region=(0, 0, 10, 10))
    with pytest.raises(DomainValidationError, match="extent"):
        WorkScope(layer_ids=(layer,), region=(1, 0, 0, 1), coordinate_system="layer pixels")
    with pytest.raises(DomainValidationError):
        WorkScope(frame_ids=(uid(),))
    scope = WorkScope(layer_ids=(layer,), frame_ids=(uid(),), region=(-1, 0, 10, 10), coordinate_system="layer pixels")
    assert scope.region[0] == -1


def test_execution_reference_history_is_distinct_from_task_lifecycle():
    plan = plan_with_items()
    order, job = uid(), uid()
    plan = change(plan, "link_execution", plan.items[0].id, work_order_id=order, plugin_job_id=job)
    assert plan.items[0].work_order_ids == (order,)
    assert plan.items[0].plugin_job_ids == (job,)
    assert plan.items[0].state is WorkItemState.PLANNED
    with pytest.raises(DomainValidationError):
        change(plan, "link_execution", plan.items[0].id, plugin_job_id=job)


def test_progress_cannot_move_backwards_and_milestones_derive_from_acceptance():
    plan = plan_with_items()
    item = plan.items[0]
    milestone = ProjectMilestone(uid(), "Ready", (item.id,))
    plan = change(plan, "add_milestone", milestone)
    plan = change(plan, "transition", WorkPlanState.ACTIVE)
    plan = change(plan, "move_item", item.id, WorkItemState.RUNNING)
    plan = change(plan, "record_progress", item.id, 0.5, 3)
    with pytest.raises(DomainValidationError, match="decrease"):
        change(plan, "record_progress", item.id, 0.4, 3)
    with pytest.raises(DomainValidationError, match="decrease"):
        change(plan, "record_progress", item.id, 0.8, 2)
    assert not plan.milestone_reached(milestone.id)
    plan = change(plan, "record_progress", item.id, 1, 3)
    plan = change(plan, "move_item", item.id, WorkItemState.IN_REVIEW)
    plan = change(plan, "move_item", item.id, WorkItemState.ACCEPTED)
    assert plan.milestone_reached(milestone.id)


def app_context(tmp_path):
    from kraken_manager.application.authorization import AuthorizationPolicy
    from kraken_manager.application.dto import StorageBackendKind, StorageScope
    from kraken_manager.application.planning import WorkPlanningService
    from kraken_manager.application.ports import StorageCapabilities, StorageProfile
    from kraken_manager.domain.identity import Principal, ProjectRole
    from kraken_manager.infrastructure.filesystem.event_store import FilesystemEventStore

    plan = WorkPlan.create(uid(), "Persisted recovery", at=NOW)
    events = FilesystemEventStore(tmp_path, plan.project_id)
    service = WorkPlanningService(events, AuthorizationPolicy("acl"))
    principal = Principal.local(subject="engineer", display_name="Engineer")
    storage = StorageProfile(
        "test",
        "Test",
        StorageBackendKind.FILESYSTEM,
        "filesystem",
        StorageScope.SHARED,
        StorageCapabilities(False, True, True, True, True, None),
    )
    context = dict(principal=principal, storage=storage, roles=(ProjectRole.ADMIN,))
    return plan, events, service, context


def test_real_filesystem_event_store_reopen_audit_cas_and_idempotent_replay(tmp_path):
    from kraken_manager.application.errors import ConcurrencyError, ConflictError
    from kraken_manager.application.planning import WorkPlanningService
    from kraken_manager.infrastructure.filesystem.event_store import FilesystemEventStore

    plan, events, service, context = app_context(tmp_path)
    assert service.create(plan, idempotency_key="create", **context) == plan
    assert (
        service.create(replace(plan, updated_at=NOW + timedelta(seconds=1)), idempotency_key="create", **context)
        == plan
    )
    item = WorkItem.create("Segment M1", assignment=WorkAssignment(uid()))
    command = dict(
        operation="add_item", arguments={"item": item}, expected_revision=0, idempotency_key="add", **context
    )
    changed = service.execute(plan.id, **command)
    assert service.execute(plan.id, **command) == changed
    assert len(events.load_stream(service.stream_id(plan.id))) == 2
    with pytest.raises(ConflictError):
        service.execute(plan.id, **(command | {"arguments": {"item": WorkItem.create("Different")}}))
    with pytest.raises(ConcurrencyError):
        service.execute(plan.id, **(command | {"idempotency_key": "different-key"}))
    reopened = WorkPlanningService(FilesystemEventStore(tmp_path, plan.project_id))
    assert reopened.load(plan.id) == changed
    event = events.load_stream(service.stream_id(plan.id))[-1]
    assert event.actor.principal_id == context["principal"].id
    assert event.event_type == "work_plan.add_item"
    assert event.payload["command"]["operation"] == "add_item"
    assert event.revision == changed.revision + 1


def test_application_acl_runs_before_idempotent_replay_and_acceptance_is_distinct(tmp_path):
    from kraken_manager.application.errors import AuthorizationError
    from kraken_manager.domain.identity import Permission, ProjectRole
    from kraken_manager.application.authorization import AuthorizationPolicy

    plan, events, service, context = app_context(tmp_path)
    service.create(plan, idempotency_key="create", **context)
    with pytest.raises(AuthorizationError):
        service.create(plan, idempotency_key="create", **(context | {"roles": (ProjectRole.VIEWER,)}))

    # Test the command's permission independently of the role catalogue: a caller
    # authorized to assign must still be checked for acceptance on acceptance.
    class RecordingPolicy(AuthorizationPolicy):
        requested = []

        def require(self, **kwargs):
            self.requested.append(kwargs["permission"])
            if kwargs["permission"] is Permission.ACCEPT_REVIEW:
                raise AuthorizationError("review permission missing")

    policy = RecordingPolicy()
    service.authorization = policy
    with pytest.raises(AuthorizationError):
        service.execute(
            plan.id,
            expected_revision=0,
            operation="move_item",
            arguments={"item_id": uid(), "state": "accepted"},
            **context,
        )
    assert policy.requested == [Permission.ACCEPT_REVIEW]
    assert len(events.load_stream(service.stream_id(plan.id))) == 1


def test_reviewer_identity_cannot_be_spoofed_by_command(tmp_path):
    plan, events, service, context = app_context(tmp_path)
    service.create(plan, **context)
    item = WorkItem.create("Review", assignment=WorkAssignment(uid()), criteria=(AcceptanceCriterion(uid(), "Mask"),))
    for operation, arguments in [
        ("add_item", {"item": item}),
        ("transition", {"state": "active"}),
        ("move_item", {"item_id": item.id, "state": "running"}),
        ("record_progress", {"item_id": item.id, "fraction": 1, "actual_effort_hours": 1}),
        ("move_item", {"item_id": item.id, "state": "in_review"}),
        (
            "satisfy_criterion",
            {
                "item_id": item.id,
                "criterion_id": item.criteria[0].id,
                "reviewer_id": uid(),
                "evidence_version_ids": (uid(),),
            },
        ),
    ]:
        plan = service.execute(
            plan.id, expected_revision=plan.revision, operation=operation, arguments=arguments, **context
        )
    assert plan.items[0].criteria[0].reviewed_by == context["principal"].id


def test_combined_hierarchy_and_dependency_deadlock_rejected():
    plan = plan_with_items()
    parent = plan.items[0]
    child = WorkItem.create("Child", parent_id=parent.id)
    plan = change(plan, "add_item", child)
    with pytest.raises(DomainValidationError, match="cycle"):
        change(plan, "add_dependency", WorkDependency(parent.id, child.id))


def test_application_links_existing_jobs_and_orders_and_rejects_cross_project(tmp_path):
    from kraken_manager.domain.selection import FrameSelectionV1
    from kraken_manager.domain.work_orders import WorkKind, WorkOrder, WorkPlacement
    from kraken_manager.domain.workflows import PluginJob

    plan, events, service, context = app_context(tmp_path)
    service.create(plan, **context)
    layer, performer = uid(), uid()
    item = WorkItem.create("Vectorize", scope=WorkScope(layer_ids=(layer,)), assignment=WorkAssignment(performer))
    plan = service.execute(plan.id, expected_revision=0, operation="add_item", arguments={"item": item}, **context)
    selection = FrameSelectionV1.rectangle(1, 1, 1, 1)
    order = WorkOrder.create(
        project_id=plan.project_id,
        layer_id=layer,
        kind=WorkKind.VECTORIZE,
        placement=WorkPlacement.SERVER,
        assignee_id=performer,
        created_by=context["principal"].id,
        destination="server",
        frame_count=1,
        selection=selection,
    )
    job = PluginJob.create(
        project_id=plan.project_id,
        layer_id=layer,
        selection=selection,
        actor_principal_id=context["principal"].id,
        target_representation_id=uid(),
        capability="vectorize",
    )
    command = dict(
        operation="link_execution",
        expected_revision=plan.revision,
        arguments={"item_id": item.id, "order": order, "job": job},
        idempotency_key="link",
        **context,
    )
    linked = service.execute(plan.id, **command)
    assert linked.items[0].work_order_ids == (order.id,)
    assert linked.items[0].plugin_job_ids == (job.id,)
    assert service.execute(plan.id, **command) == linked
    for bad_order in (
        replace(order, project_id=uid()),
        replace(order, assignee_id=uid()),
        replace(order, layer_id=uid()),
    ):
        with pytest.raises(DomainValidationError):
            service.execute(
                plan.id,
                expected_revision=linked.revision,
                operation="link_execution",
                arguments={"item_id": item.id, "order": bad_order},
                **context,
            )
    with pytest.raises(DomainValidationError, match="resolve"):
        service.execute(
            plan.id,
            expected_revision=linked.revision,
            operation="link_execution",
            arguments={"item_id": item.id, "plugin_job_id": job.id},
            **context,
        )


def test_replanning_preserves_baseline_and_forbids_rewriting_execution_history():
    plan = plan_with_items()
    plan = change(plan, "capture_baseline")
    item = plan.items[0]
    estimate = WorkEstimate(effort_hours=8, planned_start=NOW, planned_finish=NOW + timedelta(days=1))
    plan = change(plan, "revise_item", item.id, estimate=estimate, priority=4, title="Replanned work")
    assert plan.items[0].estimate == estimate
    assert plan.items[0].priority == 4
    assert plan.baselines[0].items[0].estimate.effort_hours == 0
    plan = change(plan, "transition", WorkPlanState.ACTIVE)
    plan = change(plan, "move_item", item.id, WorkItemState.RUNNING)
    with pytest.raises(InvalidStateTransition, match="unstarted"):
        change(plan, "revise_item", item.id, priority=9)


def test_unstarted_work_cannot_carry_forged_progress_or_acceptance():
    satisfied = AcceptanceCriterion(uid(), "Forged", satisfied=True, evidence_version_ids=(uid(),), reviewed_by=uid())
    with pytest.raises(DomainValidationError, match="evidence"):
        WorkItem.create("Forged", criteria=(satisfied,))
    with pytest.raises(DomainValidationError, match="execution progress"):
        WorkItem.create("Forged", progress=WorkProgress(fraction=0.5))
