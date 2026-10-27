"""Acceptance certifies a correction branch, never unrelated use of defective bytes."""

import pytest

from kraken_manager.application.dto import AddArtifactVersionCommand, CommandContext
from kraken_manager.application.errors import ConflictError
from kraken_manager.application.quality import QualityService
from kraken_manager.application.use_cases import AddArtifactVersionHandler
from kraken_manager.domain.artifacts import ArtifactScope
from tests.test_quality_workflow import quality as quality, corrected_issue


def test_actual_accepted_repair_descendant_clears_but_bad_and_mixed_inputs_stay_blocked(quality):
    q = quality
    issue = corrected_issue(q)
    checked = q.command(issue, "recheck", {"passed": True, "evidence_version_ids": [str(q.corrected.id)],
        "note": "Independent review of the actual correction"}, principal=q.reviewer)
    accepted = q.command(checked, "accept", {"note": "Correction accepted"}, principal=q.reviewer)

    def derived(name, inputs):
        series = q.service.create_artifact_series(principal=q.owner, project_id=q.project.id,
            scope=ArtifactScope.PROJECT_ATTACHMENT, name=name, idempotency_key=f"{name}-series")
        return AddArtifactVersionHandler(q.service._uow(str(q.project.id)), q.service.profiles, q.service.clock)(
            AddArtifactVersionCommand(context=CommandContext(actor=q.owner, idempotency_key=f"{name}-version"),
                project_id=q.project.id, series_id=series.id, filename=f"{name}.bin", media_type="application/octet-stream",
                expected_series_revision=q.service.artifact_stream_revision(q.project.id, series.id),
                input_version_ids=tuple(inputs), tool_name="Quality lineage regression"), (name.encode(),))

    repaired_descendant = derived("repaired-descendant", (q.corrected.id,))
    unrelated_recompute = derived("unrelated-recompute", (q.source.id,))
    mixed = derived("mixed-accepted-and-defective", (repaired_descendant.id, q.source.id))
    with q.service._uow(str(q.project.id))() as unit:
        for version in (q.corrected, repaired_descendant):
            QualityService.require_clear_inputs(unit.event_store, q.project.id, (version.id,), projections=unit.projections)
        for version in (q.source, unrelated_recompute, mixed):
            with pytest.raises(ConflictError, match=accepted.id):
                QualityService.require_clear_inputs(unit.event_store, q.project.id, (version.id,), projections=unit.projections)
