"""Work-plan analytics calculated from versioned snapshots and recorded events.

Reported effort is never inferred from elapsed time. Execution forecasts use
only declared estimates, progress and caller-supplied calendar-day capacities;
formal acceptance and measured CPU/GPU utilization remain unknown without data.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from datetime import datetime, timedelta
from math import isfinite

from kraken_manager.domain.common import DomainValidationError, as_utc, validate_uuid
from kraken_manager.domain.planning import WorkItemState, WorkPlan


def field(event, name, default=None):
    return event.get(name, default) if isinstance(event, Mapping) else getattr(event, name, default)


def timestamp(value):
    return as_utc(datetime.fromisoformat(value) if isinstance(value, str) else value)


def iso(value):
    return value.isoformat() if value is not None else None


def mean(values):
    return sum(values) / len(values) if values else None


def _forecast(rows, plans, capacities, as_of):
    incomplete = [r for r in rows if r["state"] not in {"accepted", "cancelled"}]
    if not incomplete:
        return {
            "execution_finish_at": None,
            "remaining_calendar_days": 0,
            "acceptance_finish_at": None,
            "reason": "no_remaining_execution",
            "method": "explicit_estimates_and_calendar_capacity",
        }
    for row in incomplete:
        if row["state"] == "blocked":
            reason = "blocked_work"
        elif row["planned_effort_hours"] <= 0:
            reason = "missing_estimate"
        elif not row["performer_id"]:
            reason = "unassigned_work"
        elif row["performer_id"] not in capacities:
            reason = "missing_capacity"
        else:
            continue
        return {
            "execution_finish_at": None,
            "remaining_calendar_days": None,
            "acceptance_finish_at": None,
            "reason": reason,
            "method": "explicit_estimates_and_calendar_capacity",
        }
    by_id = {(r["plan_id"], r["item_id"]): r for r in rows}
    all_items = {(p.id, i.id): i for p in plans for i in p.items}
    dependency_map = {(r["plan_id"], r["item_id"]): [] for r in incomplete}
    for plan in plans:
        for dependency in plan.dependencies:
            key = (plan.id, dependency.successor_id)
            if key not in dependency_map:
                continue
            predecessor = (plan.id, dependency.predecessor_id)
            item = all_items[predecessor]
            if item.state is WorkItemState.ACCEPTED:
                continue
            if predecessor not in by_id or item.state is WorkItemState.CANCELLED:
                return {
                    "execution_finish_at": None,
                    "remaining_calendar_days": None,
                    "acceptance_finish_at": None,
                    "reason": "dependency_outside_forecast",
                    "method": "explicit_estimates_and_calendar_capacity",
                }
            dependency_map[key].append(predecessor)
    pending = dict(dependency_map)
    finish = {}
    available = defaultdict(float)
    while pending:
        ready = [key for key, dependencies in pending.items() if all(d in finish for d in dependencies)]
        if not ready:
            return {
                "execution_finish_at": None,
                "remaining_calendar_days": None,
                "acceptance_finish_at": None,
                "reason": "unresolved_dependencies",
                "method": "explicit_estimates_and_calendar_capacity",
            }
        key = min(ready, key=lambda k: (-by_id[k]["priority"], k))
        row = by_id[key]
        performer = row["performer_id"]
        planned_start = timestamp(row["planned_start"]) if row["planned_start"] else as_of
        earliest = max(0, (planned_start - as_of).total_seconds() / 86400)
        start = max(earliest, available[performer], max((finish[d] for d in pending[key]), default=0))
        finish[key] = start + row["remaining_estimated_hours"] / capacities[performer]
        available[performer] = finish[key]
        del pending[key]
    days = max(finish.values(), default=0)
    try:
        predicted = as_of + timedelta(days=days)
    except OverflowError, ValueError:
        return {
            "execution_finish_at": None,
            "remaining_calendar_days": None,
            "acceptance_finish_at": None,
            "reason": "forecast_outside_calendar_range",
            "method": "explicit_estimates_and_calendar_capacity",
        }
    return {
        "execution_finish_at": iso(predicted),
        "remaining_calendar_days": days,
        "acceptance_finish_at": None,
        "reason": "acceptance_timing_not_estimated",
        "method": "explicit_estimates_and_calendar_capacity",
    }


class WorkPlanAnalytics:
    @classmethod
    def report(
        cls,
        events,
        *,
        project_id,
        as_of,
        start=None,
        end=None,
        plan_ids=(),
        layer_ids=(),
        performer_ids=(),
        states=(),
        capacity_by_performer=None,
        baseline_ids=None,
        assume_sorted=False,
    ):
        project_id = validate_uuid(str(project_id))
        as_of = timestamp(as_of)
        start = timestamp(start) if start is not None else as_of - timedelta(days=30)
        end = timestamp(end) if end is not None else as_of
        if end < start or end > as_of:
            raise DomainValidationError("analytics period must satisfy start <= end <= as_of")
        for values in (plan_ids, layer_ids, performer_ids):
            for value in values:
                validate_uuid(str(value))
        for state in states:
            WorkItemState(state)
        capacities = dict(capacity_by_performer or {})
        for identifier, value in capacities.items():
            validate_uuid(str(identifier))
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value <= 0:
                raise DomainValidationError(
                    "performer capacity must be finite positive reported hours per calendar day"
                )
        source = (
            events
            if assume_sorted
            else sorted(events, key=lambda e: (str(field(e, "stream_id", "")), int(field(e, "revision", 0))))
        )
        latest, previous_by_plan, last_revision = {}, {}, {}
        pipelines = {}
        timelines, completeness, source_event_ids = {}, {}, []
        for event in source:
            if str(field(event, "project_id")) != project_id:
                continue
            stream_id = str(field(event, "stream_id", ""))
            if field(event, "event_type") == "LayerPipelineWorkflowV1":
                at = timestamp(field(event, "recorded_at"))
                state = field(event, "payload", {}).get("workflow", {})
                layer_id = str(state.get("layer_id", ""))
                if at <= as_of and (not layer_ids or layer_id in layer_ids):
                    if str(state.get("project_id")) != project_id or stream_id != f"layer-pipeline:{layer_id}":
                        raise DomainValidationError("analytics pipeline source identity mismatch")
                    pipelines[layer_id] = (state, int(field(event, "revision")), str(field(event, "event_id")))
                continue
            if not stream_id.startswith("work-plan:"):
                continue
            stream_plan_id = str(field(event, "stream_id")).partition(":")[2]
            if plan_ids and stream_plan_id not in plan_ids:
                continue
            at = timestamp(field(event, "recorded_at"))
            if at > as_of:
                continue
            payload = field(event, "payload", {})
            if "plan" in payload:
                plan = WorkPlan.from_dict(payload)
                if plan.project_id != project_id or str(field(event, "stream_id")) != f"work-plan:{plan.id}":
                    raise DomainValidationError("analytics source plan identity mismatch")
                revision = int(field(event, "revision"))
                previous_revision = last_revision.get(plan.id, 0)
                if revision <= previous_revision or revision != plan.revision + 1:
                    raise DomainValidationError("planning analytics requires ordered, consistent stream revisions")
                completeness[plan.id] = completeness.get(plan.id, True) and revision == previous_revision + 1
                last_revision[plan.id] = revision
                latest[plan.id] = plan
                identifier = plan.id
                previous = previous_by_plan.get(identifier, {})
                event_id = str(field(event, "event_id"))
                actor = field(event, "actor", {})
                command = payload.get("command") or {}
                source_event_ids.append(event_id)
                for item in plan.items:
                    key = (identifier, item.id)
                    facts = timelines.setdefault(
                        key,
                        {
                            "created": at,
                            "started": None,
                            "accepted": [],
                            "changes": [],
                            "event_ids": [],
                        },
                    )
                    old = previous.get(item.id)
                    if old != item:
                        facts["changes"].append(
                            {
                                "at": iso(at),
                                "state": str(item.state),
                                "attempt": item.attempt,
                                "event_id": event_id,
                                "plan_revision": plan.revision,
                                "actor_id": field(actor, "principal_id"),
                                "actor_name": field(actor, "display_name"),
                                "operation": command.get("operation") or field(event, "event_type"),
                            }
                        )
                        facts["event_ids"].append(event_id)
                    if facts["started"] is None and item.progress.started_at is not None:
                        facts["started"] = at
                    if item.state is WorkItemState.ACCEPTED and (
                        old is None or old.state is not WorkItemState.ACCEPTED
                    ):
                        facts["accepted"].append(at)
                previous_by_plan[identifier] = {item.id: item for item in plan.items}
        plans = list(latest.values())
        for (identifier, item_id), facts in timelines.items():
            if not completeness[identifier]:
                facts["created"] = facts["started"] = None
                facts["accepted"] = []
        baseline_ids = dict(baseline_ids or {})
        if set(baseline_ids) - {p.id for p in plans}:
            raise DomainValidationError("baseline selector references an unavailable plan")
        rows, variances = [], []
        for plan in plans:
            parents = {i.parent_id for i in plan.items if i.parent_id}
            by_id = {i.id: i for i in plan.items}
            for item in plan.items:
                if layer_ids and not set(item.scope.layer_ids).intersection(layer_ids):
                    continue
                performer_id = item.assignment.performer_id if item.assignment else None
                if performer_ids and performer_id not in performer_ids:
                    continue
                if states and str(item.state) not in states:
                    continue
                facts = timelines[(plan.id, item.id)]
                accepted_at = (
                    facts["accepted"][-1] if item.state is WorkItemState.ACCEPTED and facts["accepted"] else None
                )
                lead = (
                    (accepted_at - facts["created"]).total_seconds() / 3600
                    if accepted_at and facts["created"]
                    else None
                )
                cycle = (
                    (accepted_at - facts["started"]).total_seconds() / 3600
                    if accepted_at and facts["started"]
                    else None
                )
                remaining = (
                    0
                    if item.state in {WorkItemState.ACCEPTED, WorkItemState.CANCELLED}
                    else (
                        item.estimate.effort_hours * (1 - item.progress.fraction)
                        if item.estimate.effort_hours > 0
                        else None
                    )
                )
                breadcrumb = []
                ancestor = item.parent_id
                while ancestor:
                    breadcrumb.append(by_id[ancestor].title)
                    ancestor = by_id[ancestor].parent_id
                rows.append(
                    {
                        "plan_id": plan.id,
                        "plan_title": plan.title,
                        "plan_revision": plan.revision,
                        "item_id": item.id,
                        "title": item.title,
                        "parent_id": item.parent_id,
                        "hierarchy": list(reversed(breadcrumb)),
                        "is_leaf": item.id not in parents,
                        "state": str(item.state),
                        "priority": item.priority,
                        "performer_id": performer_id,
                        "layer_ids": list(item.scope.layer_ids),
                        "frame_ids": list(item.scope.frame_ids),
                        "region": item.scope.region,
                        "coordinate_system": item.scope.coordinate_system,
                        "progress_fraction": item.progress.fraction,
                        "reported_effort_hours": item.progress.actual_effort_hours,
                        "planned_effort_hours": item.estimate.effort_hours,
                        "remaining_estimated_hours": remaining,
                        "planned_start": iso(item.estimate.planned_start),
                        "planned_finish": iso(item.estimate.planned_finish),
                        "first_observed_at": iso(facts["created"]),
                        "first_started_at": iso(facts["started"]),
                        "accepted_at": iso(accepted_at),
                        "lead_hours": lead,
                        "cycle_hours": cycle,
                        "attempts": item.attempt,
                        "rework_attempts": max(0, item.attempt - 1),
                        "blocking_reason": item.blocking_reason,
                        "overdue": bool(
                            item.estimate.planned_finish
                            and item.estimate.planned_finish < as_of
                            and item.state not in {WorkItemState.ACCEPTED, WorkItemState.CANCELLED}
                        ),
                        "criteria": [
                            {
                                "id": c.id,
                                "description": c.description,
                                "required": c.required,
                                "satisfied": c.satisfied,
                                "evidence_version_ids": list(c.evidence_version_ids),
                                "reviewed_by": c.reviewed_by,
                            }
                            for c in item.criteria
                        ],
                        "work_order_ids": list(item.work_order_ids),
                        "plugin_job_ids": list(item.plugin_job_ids),
                        "event_ids": facts["event_ids"],
                        "changes": facts["changes"],
                        "acceptances_in_period": sum(start <= at <= end for at in facts["accepted"])
                        if completeness[plan.id]
                        else None,
                    }
                )
            baseline = (
                next((b for b in plan.baselines if b.id == baseline_ids.get(plan.id)), None)
                if plan.id in baseline_ids
                else (plan.baselines[-1] if plan.baselines else None)
            )
            if plan.id in baseline_ids and baseline is None:
                raise DomainValidationError("baseline not found in the selected plan revision")
            if baseline:
                old = {i.id: i for i in baseline.items}
                current = {i.id: i for i in plan.items}
                old_parents = {i.parent_id for i in baseline.items if i.parent_id}
                old_leaves = [
                    i for i in baseline.items if i.id not in old_parents and i.state is not WorkItemState.CANCELLED
                ]
                current_leaves = [
                    i for i in plan.items if i.id not in parents and i.state is not WorkItemState.CANCELLED
                ]
                original_effort = (
                    sum(i.estimate.effort_hours for i in old_leaves)
                    if all(i.estimate.effort_hours > 0 for i in old_leaves)
                    else None
                )
                current_effort = (
                    sum(i.estimate.effort_hours for i in current_leaves)
                    if all(i.estimate.effort_hours > 0 for i in current_leaves)
                    else None
                )
                for identifier in sorted(old.keys() | current.keys()):
                    before, after = old.get(identifier), current.get(identifier)
                    facts = timelines.get((plan.id, identifier), {})
                    accepted_at = (
                        facts.get("accepted", [])[-1]
                        if after and after.state is WorkItemState.ACCEPTED and facts.get("accepted")
                        else None
                    )
                    variances.append(
                        {
                            "plan_id": plan.id,
                            "baseline_id": baseline.id,
                            "baseline_source_revision": baseline.source_revision,
                            "item_id": identifier,
                            "title": (after or before).title,
                            "scope_change": "added" if before is None else "removed" if after is None else "retained",
                            "planned_effort_delta_hours": (after.estimate.effort_hours - before.estimate.effort_hours)
                            if before and after and before.estimate.effort_hours > 0 and after.estimate.effort_hours > 0
                            else None,
                            "planned_finish_delta_hours": (
                                after.estimate.planned_finish - before.estimate.planned_finish
                            ).total_seconds()
                            / 3600
                            if before and after and before.estimate.planned_finish and after.estimate.planned_finish
                            else None,
                            "acceptance_finish_variance_hours": (
                                accepted_at - before.estimate.planned_finish
                            ).total_seconds()
                            / 3600
                            if accepted_at and before and before.estimate.planned_finish
                            else None,
                            "baseline_plan_effort_hours": original_effort,
                            "current_plan_effort_hours": current_effort,
                        }
                    )
        leaves = [r for r in rows if r["is_leaf"]]
        active = [r for r in leaves if r["state"] != "cancelled"]
        accepted = [r for r in active if r["state"] == "accepted"]
        started = [r for r in active if r["attempts"] > 0]
        workload = []
        for performer_id in sorted({r["performer_id"] for r in rows}, key=lambda v: v or ""):
            assigned = [r for r in rows if r["performer_id"] == performer_id]
            remaining_rows = [r for r in assigned if r["state"] not in {"accepted", "cancelled"}]
            remaining = (
                None
                if any(r["remaining_estimated_hours"] is None for r in remaining_rows)
                else sum(r["remaining_estimated_hours"] for r in remaining_rows)
            )
            workload.append(
                {
                    "performer_id": performer_id,
                    "open_items": len(remaining_rows),
                    "reported_effort_hours": sum(r["reported_effort_hours"] for r in assigned),
                    "remaining_estimated_hours": remaining,
                    "capacity_hours_per_calendar_day": capacities.get(performer_id),
                    "item_ids": [r["item_id"] for r in assigned],
                }
            )
        layers = []
        for layer_id in sorted({value for r in active for value in r["layer_ids"]}):
            matching = [r for r in active if layer_id in r["layer_ids"]]
            ready = [r for r in matching if r["state"] == "accepted"]
            layers.append(
                {
                    "layer_id": layer_id,
                    "items": len(matching),
                    "accepted_items": len(ready),
                    "readiness_fraction": len(ready) / len(matching) if matching else None,
                    "item_ids": [r["item_id"] for r in matching],
                }
            )
        period_days = (end - start).total_seconds() / 86400
        acceptance_events = (
            sum(r["acceptances_in_period"] for r in leaves)
            if all(r["acceptances_in_period"] is not None for r in leaves)
            else None
        )
        summary = {
            "total_items": len(rows),
            "plan_count": len(plans),
            "completed_plans": sum(str(p.state) == "completed" for p in plans),
            "leaf_items": len(leaves),
            "open_items": sum(r["state"] not in {"accepted", "cancelled"} for r in leaves),
            "accepted_items": len(accepted),
            "readiness_fraction": len(accepted) / len(active) if active else None,
            "reported_effort_hours": sum(r["reported_effort_hours"] for r in rows),
            "leaf_reported_effort_hours": sum(r["reported_effort_hours"] for r in leaves),
            "mean_lead_hours": mean([r["lead_hours"] for r in accepted if r["lead_hours"] is not None]),
            "mean_cycle_hours": mean([r["cycle_hours"] for r in accepted if r["cycle_hours"] is not None]),
            "rework_fraction": sum(r["rework_attempts"] > 0 for r in started) / len(started) if started else None,
            "rework_attempts": sum(r["rework_attempts"] for r in leaves),
            "acceptance_events_in_period": acceptance_events,
            "acceptances_per_calendar_day": acceptance_events / period_days
            if period_days > 0 and acceptance_events is not None
            else None,
            "overdue_items": sum(r["overdue"] for r in leaves),
            "accepted_frame_scope_count": len(
                {(layer, frame) for r in accepted for layer in r["layer_ids"] for frame in r["frame_ids"]}
            ),
            "measured_compute_utilization": None,
        }
        forecast = _forecast(leaves, plans, capacities, as_of)
        compute = []
        if pipelines:
            from kraken_manager.application.pipeline_execution import ACTIVE
            from kraken_manager.domain.pipeline import LayerPipelineDefinition

            for layer_id, (state, revision, event_id) in sorted(pipelines.items()):
                definition = LayerPipelineDefinition.from_dict(state["definition"])
                if definition.project_id != project_id or definition.layer_id != layer_id:
                    raise DomainValidationError("analytics pipeline definition identity mismatch")
                active_nodes = [
                    op for op in definition.operations if state.get("nodes", {}).get(op.id, {}).get("status") in ACTIVE
                ]
                reserved = {
                    name: sum(getattr(op.resources, name) for op in active_nodes)
                    for name in ("cpu", "gpu", "memory_mb")
                }
                capacity = definition.policy.capacity.to_dict()
                compute.append(
                    {
                        "layer_id": layer_id,
                        "revision": revision,
                        "status": state["status"],
                        "active_operations": len(active_nodes),
                        "reserved": reserved,
                        "declared_capacity": capacity,
                        "reservation_fraction": {
                            key: reserved[key] / capacity[key] if capacity[key] > 0 else None for key in reserved
                        },
                        "measured_utilization": None,
                        "source_event_id": event_id,
                    }
                )
                source_event_ids.append(event_id)
        forecast["scope"] = "filtered_leaf_execution"
        forecast["assumptions"] = [
            "Declared capacity is constant per calendar day; no implicit working-day calendar.",
            "Only declared leaf estimates and reported progress are scheduled; stage effort is excluded.",
            "Unestimated future review/acceptance waits are excluded; formal acceptance has no forecast.",
            "Ready work is scheduled by descending priority then stable item identity, not optimal scheduling.",
        ]
        return {
            "schema_version": 1,
            "project_id": project_id,
            "as_of": iso(as_of),
            "start": iso(start),
            "end": iso(end),
            "source_plan_revisions": {p.id: p.revision for p in plans},
            "complete_event_history": completeness,
            "filters": {
                "plan_ids": list(plan_ids),
                "layer_ids": list(layer_ids),
                "performer_ids": list(performer_ids),
                "states": list(states),
            },
            "summary": summary,
            "state_counts": dict(Counter(r["state"] for r in leaves)),
            "blocking_reasons": dict(Counter(r["blocking_reason"] for r in leaves if r["blocking_reason"])),
            "layers": layers,
            "workload": workload,
            "baseline_variance": variances,
            "compute_workload": compute,
            "items": rows,
            "forecast": forecast,
            "source_event_ids": source_event_ids,
        }
