"""Attachment metadata replay; no model execution, training or quality claim."""

from dataclasses import replace
from uuid import uuid4

import pytest

from kraken_manager.application import CommandContext, ImportPluginResultCommand, SubmitPluginJobCommand, SubmitPluginJobHandler
from kraken_manager.domain.common import DomainValidationError
from kraken_manager.domain.events import ActorSnapshot, EventEnvelope
from kraken_manager.infrastructure.projections.rebuild import ProjectionRebuilder
from tests.test_plugin_job_application import Clock, Fixture, NOW, Projections


def test_aggregate_import_records_null_ids_and_replays_without_cached_projection():
    fixture = Fixture()
    fixture.gateway.is_available = lambda *_args: True
    job = SubmitPluginJobHandler(fixture.factory, fixture.profiles, Clock(), fixture.gateway)(
        SubmitPluginJobCommand(
            context=CommandContext(actor=fixture.actor, idempotency_key="metadata-import"),
            project_id=fixture.project.id, layer_id=fixture.layer.id, target_representation_id=fixture.target.id,
            selection=fixture.selection, inputs=fixture.inputs, capability="dataset.model.train.v1", parameters={},
        )
    )
    result = fixture.result(job)
    result = replace(result, results=(replace(result.results[0], role="model"),))
    imported = fixture.importer()(ImportPluginResultCommand(
        context=CommandContext(actor=fixture.actor, idempotency_key="metadata-result"), manifest=result))
    version, = imported.versions
    event, = [event for events in fixture.factory.uow.event_store.streams.values() for event in events
              if event.event_type == "ArtifactSeriesCreated"]
    assert event.payload["representation_id"] is None and event.payload["frame_id"] is None
    fresh = Projections()
    assert ProjectionRebuilder(fresh).apply(event)
    assert fresh.get_artifact_series(version.series_id) == fixture.factory.uow.projections.get_artifact_series(version.series_id)


@pytest.mark.parametrize("scope", ["layer_attachment", "project_attachment"])
def test_legacy_string_none_attachment_ids_replay_without_rewriting_events(scope):
    fixture = Fixture()
    payload = dict(artifact_series_id=str(uuid4()), name="Legacy imported model", scope=scope,
                   layer_id=str(fixture.layer.id) if scope == "layer_attachment" else "None",
                   representation_id="None", frame_id="None", archived=False)
    event = EventEnvelope.create(
        stream_id="artifact-series:" + payload["artifact_series_id"], project_id=fixture.project.id,
        revision=1, event_type="ArtifactSeriesCreated", payload=payload,
        actor=ActorSnapshot.from_principal(fixture.actor), recorded_at=NOW,
    )
    fresh = Projections()
    assert ProjectionRebuilder(fresh).apply(event)
    series = fresh.get_artifact_series(payload["artifact_series_id"])
    assert series.representation_id is None and series.frame_id is None
    assert event.payload["representation_id"] == "None"  # Immutable original retained.
    malformed = replace(event, payload=dict(payload, representation_id="invalid-uuid"))
    with pytest.raises(DomainValidationError):
        ProjectionRebuilder(Projections()).apply(malformed)
    frame_scope = replace(event, payload=dict(payload, scope="frame_representation"))
    with pytest.raises(DomainValidationError):
        ProjectionRebuilder(Projections()).apply(frame_scope)
