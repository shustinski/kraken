"""Strict transport decoding for engineering planning commands."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from uuid import UUID, uuid5

from kraken_manager.domain.common import DomainValidationError, require_non_empty, validate_uuid
from kraken_manager.domain.planning import (
    AcceptanceCriterion,
    ProjectMilestone,
    WorkAssignment,
    WorkDependency,
    WorkEstimate,
    WorkItem,
    WorkItemState,
    WorkPlan,
    WorkPlanState,
    WorkScope,
)
from kraken_manager.domain.project import StructureState


def command_id(project_id, key, suffix):
    require_non_empty(key, field="idempotency_key")
    return str(uuid5(UUID(str(project_id)), f"work-planning:{key}:{suffix}"))


def _mapping(value, allowed, name, *, required=()):
    if not isinstance(value, Mapping) or set(value) - set(allowed):
        raise DomainValidationError(f"Invalid {name} fields")
    missing = set(required) - set(value)
    if missing:
        raise DomainValidationError(f"Missing {name} fields: {', '.join(sorted(missing))}")
    return dict(value)


def decode_arguments(operation, args, projections, project, performers, idempotency_key, *, plan=None, events=None):
    """Resolve project-owned references; never trust client runtime snapshots."""
    allowed = {
        "add_item": {"item"},
        "apply_template": {"template_id", "version", "scope", "assignments"},
        "revise_item": {"item_id", "title", "scope", "estimate", "priority", "criteria"},
        "assign": {"item_id", "assignment"},
        "add_dependency": {"dependency"},
        "add_milestone": {"milestone"},
        "capture_baseline": set(),
        "transition": {"state"},
        "move_item": {"item_id", "state", "reason"},
        "record_progress": {"item_id", "fraction", "actual_effort_hours"},
        "satisfy_criterion": {"item_id", "criterion_id", "evidence_version_ids"},
        "link_execution": {"item_id", "work_order_id", "plugin_job_id"},
    }
    if operation not in allowed:
        raise DomainValidationError("Unknown planning command")
    required = {
        "add_item": {"item"}, "apply_template": {"template_id", "version"},
        "revise_item": {"item_id"}, "assign": {"item_id", "assignment"},
        "add_dependency": {"dependency"}, "add_milestone": {"milestone"},
        "capture_baseline": set(), "transition": {"state"}, "move_item": {"item_id", "state"},
        "record_progress": {"item_id", "fraction", "actual_effort_hours"},
        "satisfy_criterion": {"item_id", "criterion_id", "evidence_version_ids"}, "link_execution": {"item_id"},
    }
    result = _mapping(args, allowed[operation], "planning command", required=required[operation])

    def scope(value):
        data = _mapping(value, {"layer_ids", "frame_ids", "region", "coordinate_system", "frame_coordinates"}, "scope")
        coordinates = data.pop("frame_coordinates", ())
        if coordinates:
            if data.get("frame_ids"):
                raise DomainValidationError("Use frame coordinates or frame IDs")
            ids = []
            for xy in coordinates:
                if len(xy) != 2:
                    raise DomainValidationError("Frame coordinates require X and Y")
                ids.append(project.frame_id_at(xy[0], xy[1]))
            data["frame_ids"] = ids
        elif data.get("frame_ids"):
            known = {str(s.frame_id) for s in projections.list_artifact_series(project.id) if s.frame_id}
            if not set(data["frame_ids"]) <= known:
                raise DomainValidationError("Unknown frame IDs; use project frame_coordinates for unimported frames")
        decoded = WorkScope(**data)
        for layer_id in decoded.layer_ids:
            layer = projections.get_layer(layer_id)
            if layer is None or layer.project_id != project.id or layer.state is StructureState.ARCHIVED:
                raise DomainValidationError("Scope layer is outside the project or archived")
        return decoded

    def assignment(value):
        data = _mapping(value, {"performer_id"}, "assignment", required={"performer_id"})
        performer = performers.get(validate_uuid(data["performer_id"])) if performers is not None else None
        if performer is None or not performer.active:
            raise DomainValidationError("Performer is unknown or inactive")
        return WorkAssignment(performer.id, performer.principal_id)

    def estimate(value):
        data = _mapping(value, {"effort_hours", "planned_start", "planned_finish"}, "estimate")
        for name in ("planned_start", "planned_finish"):
            if data.get(name) is not None:
                data[name] = datetime.fromisoformat(data[name])
        return WorkEstimate(**data)

    def criteria(values):
        result = []
        for index, value in enumerate(values):
            data = _mapping(value, {"description", "required"}, "criterion", required={"description"})
            result.append(AcceptanceCriterion(command_id(project.id, idempotency_key, f"criterion:{index}"), **data))
        return tuple(result)

    if operation == "apply_template":
        from kraken_manager.application.planning_templates import WorkTemplateService

        if events is None or plan is None:
            raise DomainValidationError("template application requires the project event store and target plan")
        template = WorkTemplateService(events).load(project.id, result["template_id"], version=result["version"])
        if not isinstance(result.get("assignments", {}), Mapping):
            raise DomainValidationError("template assignments must map task keys to performers")
        assignments = {key: assignment(value) for key, value in result.get("assignments", {}).items()}
        batch = template.materialize(
            scope(result.get("scope", {})),
            command_id(project.id, idempotency_key, f"template-application:{plan.id}"),
            assignments=assignments,
        )
        return {"materialization": batch}

    if "item_id" in result:
        result["item_id"] = validate_uuid(result["item_id"])
    if operation in {"add_item", "revise_item"}:
        data = (
            _mapping(
                result["item"],
                {"title", "parent_id", "scope", "assignment", "priority", "estimate", "criteria"},
                "work item",
                required={"title"},
            )
            if operation == "add_item"
            else dict(result)
        )
        for name, decoder in (
            ("scope", scope),
            ("assignment", assignment),
            ("estimate", estimate),
            ("criteria", criteria),
        ):
            if name in data:
                data[name] = decoder(data[name])
        if operation == "add_item":
            result = {"item": WorkItem(id=command_id(project.id, idempotency_key, "item"), **data)}
        else:
            result = data
    elif operation == "assign":
        result["assignment"] = assignment(result["assignment"])
    elif operation == "add_dependency":
        result["dependency"] = WorkDependency(
            **_mapping(result["dependency"], {"predecessor_id", "successor_id"}, "dependency", required={"predecessor_id", "successor_id"})
        )
    elif operation == "add_milestone":
        data = _mapping(result["milestone"], {"title", "item_ids", "due_at"}, "milestone", required={"title", "item_ids"})
        if data.get("due_at") is not None:
            data["due_at"] = datetime.fromisoformat(data["due_at"])
        result["milestone"] = ProjectMilestone(id=command_id(project.id, idempotency_key, "milestone"), **data)
    elif operation in {"transition", "move_item"}:
        result["state"] = (WorkPlanState if operation == "transition" else WorkItemState)(result["state"])
    elif operation == "satisfy_criterion":
        result["criterion_id"] = validate_uuid(result["criterion_id"])
        versions = tuple(validate_uuid(v) for v in result["evidence_version_ids"])
        item = plan.item(result["item_id"]) if plan is not None else None
        for version_id in versions:
            version = projections.get_artifact_version(version_id)
            series = None if version is None else projections.get_artifact_series(version.series_id)
            if series is None or series.project_id != project.id or series.archived:
                raise DomainValidationError("Acceptance evidence is outside the project or archived")
            if item is not None:
                if item.scope.layer_ids and series.layer_id not in item.scope.layer_ids:
                    raise DomainValidationError("Acceptance evidence layer is outside work scope")
                if item.scope.frame_ids and series.frame_id not in item.scope.frame_ids:
                    raise DomainValidationError("Acceptance evidence frame is outside work scope")
        result["evidence_version_ids"] = versions
    elif operation == "link_execution":
        order_id = result.pop("work_order_id", None)
        job_id = result.pop("plugin_job_id", None)
        if order_id is not None:
            result["order"] = projections.get_work_order(validate_uuid(order_id))
            if result["order"] is None:
                raise DomainValidationError("Work order not found")
        if job_id is not None:
            result["job"] = projections.get_plugin_job(validate_uuid(job_id))
            if result["job"] is None:
                raise DomainValidationError("Plugin job not found")
    return result


def list_plans(events, project_id):
    """Read plan snapshots from either supported event-store backend."""
    latest = {}
    if hasattr(events, "list_project_events"):
        position = 0
        while True:
            batch = events.list_project_events(project_id, after_position=position, limit=1000)
            if not batch:
                break
            for position, event in batch:
                if event.stream_id.startswith("work-plan:"):
                    latest[event.stream_id] = event.payload
    else:
        for event in events.iter_project():
            if event.stream_id.startswith("work-plan:"):
                latest[event.stream_id] = event.data["payload"]
    plans = (WorkPlan.from_dict(payload) for payload in latest.values())
    return tuple(
        sorted((p for p in plans if p.project_id == str(project_id)), key=lambda p: (p.updated_at, p.id), reverse=True)
    )
