"""Authorized collection of versioned planning facts for report generation."""

from __future__ import annotations

from collections.abc import Mapping

from kraken_manager.application.authorization import AuthorizationPolicy
from kraken_manager.application.work_plan_analytics import WorkPlanAnalytics
from kraken_manager.domain.common import DomainValidationError, utc_now
from kraken_manager.domain.identity import Permission

FIELDS = {
    "as_of",
    "start",
    "end",
    "plan_ids",
    "layer_ids",
    "performer_ids",
    "states",
    "capacity_by_performer",
    "baseline_ids",
}


def project_events(events, project_id):
    if hasattr(events, "list_project_events"):
        position = 0
        while True:
            batch = events.list_project_events(project_id, after_position=position, limit=1000)
            if not batch:
                break
            for position, event in batch:
                yield event
    else:
        backing = getattr(events, "backing", events)
        for event in backing.iter_project():
            yield event.data
        yield from getattr(events, "_pending", ())


def planning_report(unit, project, *, principal, profile, request, performers=None):
    if not isinstance(request, Mapping) or set(request) - FIELDS:
        raise DomainValidationError("unknown work-plan analytics fields")
    AuthorizationPolicy().require(
        principal=principal,
        storage=profile,
        roles=unit.acl.roles_for(project.id, principal.id),
        permission=Permission.EXPORT_STATISTICS,
    )
    options = dict(request)
    options.setdefault("as_of", utc_now())
    report = WorkPlanAnalytics.report(
        project_events(unit.event_store, project.id), project_id=project.id, assume_sorted=True, **options
    )
    performer_ids = {item["performer_id"] for item in report["items"] if item["performer_id"]}
    layer_ids = {identifier for item in report["items"] for identifier in item["layer_ids"]}
    layer_ids.update(item["layer_id"] for item in report["compute_workload"])
    report["performer_names"] = (
        {str(p.id): p.name for p in performers.list(include_archived=True) if str(p.id) in performer_ids}
        if performers is not None
        else {}
    )
    report["layer_names"] = {
        str(layer.id): layer.name
        for layer in unit.projections.list_layers(project.id, include_archived=True)
        if str(layer.id) in layer_ids
    }
    return report
