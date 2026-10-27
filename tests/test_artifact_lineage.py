from types import SimpleNamespace
from uuid import uuid4

import pytest

from kraken_manager.application.artifact_lineage import artifact_ancestors
from kraken_manager.application.errors import ConflictError
from kraken_manager.application.quality import QualityService

pytest_plugins = ["tests.test_quality_workflow"]


def graph(edges):
    project_id, series_id = str(uuid4()), str(uuid4())
    versions = {identifier: SimpleNamespace(series_id=series_id, input_version_ids=parents, parent_version_id=None)
                for identifier, parents in edges.items()}
    projections = SimpleNamespace(get_artifact_version=versions.get,
        get_artifact_series=lambda key: SimpleNamespace(project_id=project_id) if key == series_id else None)
    return project_id, projections


def test_shared_immutable_ancestors_are_traversed_once_and_cycles_rejected():
    a, b, c, d = (str(uuid4()) for _ in range(4))
    project_id, projections = graph({a: (b, c), b: (d,), c: (d,), d: ()})
    assert artifact_ancestors(projections, project_id, (a, b)) == frozenset((a, b, c, d))
    project_id, projections = graph({a: (b,), b: (a,)})
    with pytest.raises(ConflictError, match="cycle"):
        artifact_ancestors(projections, project_id, (a,))


def test_long_histories_do_not_depend_on_python_recursion_limit():
    ids = [str(uuid4()) for _ in range(6000)]
    project_id, projections = graph({identifier: (ids[i + 1],) if i + 1 < len(ids) else ()
                                     for i, identifier in enumerate(ids)})
    assert len(artifact_ancestors(projections, project_id, (ids[0],))) == len(ids)


def test_missing_or_cross_project_provenance_cannot_silently_pass_a_gate():
    identifier, missing = str(uuid4()), str(uuid4())
    project_id, projections = graph({identifier: (missing,)})
    with pytest.raises(ConflictError, match="missing"):
        artifact_ancestors(projections, project_id, (identifier,))
    project_id, projections = graph({identifier: ()})
    with pytest.raises(ConflictError, match="ownership"):
        artifact_ancestors(projections, str(uuid4()), (identifier,))


def test_real_quality_issue_in_an_ancestor_blocks_derived_evidence_until_resolved(quality):
    q = quality
    issue = q.create()
    with q.service._uow(str(q.project.id))() as unit:
        ancestors = artifact_ancestors(unit.projections, q.project.id, (q.corrected.id,))
        assert q.source.id in ancestors
        with pytest.raises(ConflictError, match=issue.id):
            QualityService.require_clear_inputs(unit.event_store, q.project.id, (q.corrected.id,),
                                                projections=unit.projections)
    q.command(issue, "exception", {"note": "Explicit independent engineering waiver"}, principal=q.reviewer)
    with q.service._uow(str(q.project.id))() as unit:
        QualityService.require_clear_inputs(unit.event_store, q.project.id, (q.corrected.id,),
                                            projections=unit.projections)
