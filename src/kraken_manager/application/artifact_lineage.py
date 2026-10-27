"""Traverse declared immutable provenance, without creating a second graph."""

from __future__ import annotations

from kraken_manager.application.errors import ConflictError
from kraken_manager.domain.common import validate_uuid


def artifact_ancestors(projections, project_id, version_ids, *, stop_at=()):
    """Include exact inputs and prior versions; fail closed on broken lineage.

    Iterative traversal supports long restoration histories and shared ancestors.
    The authoritative edges are ArtifactVersion.input_version_ids/parent_version_id.
    ``stop_at`` preserves the named certified versions but does not traverse
    beyond them. Quality gates first validate the entire graph, then apply that
    barrier independently for each accepted correction.
    """
    certified = frozenset(str(identifier) for identifier in stop_at)
    status = {}
    stack = [(validate_uuid(str(identifier)), False) for identifier in version_ids]
    while stack:
        identifier, leaving = stack.pop()
        if leaving:
            status[identifier] = 2
            continue
        if status.get(identifier) == 2:
            continue
        if status.get(identifier) == 1:
            raise ConflictError("Artifact provenance contains a cycle")
        version = projections.get_artifact_version(identifier)
        series = None if version is None else projections.get_artifact_series(version.series_id)
        if series is None or str(series.project_id) != str(project_id):
            raise ConflictError("Artifact provenance is missing or crosses project ownership")
        status[identifier] = 1
        stack.append((identifier, True))
        parents = () if identifier in certified else tuple(version.input_version_ids)
        if identifier not in certified and version.parent_version_id is not None:
            parents += (version.parent_version_id,)
        for parent in set(parents):
            stack.append((str(parent), False))
    return frozenset(status)
