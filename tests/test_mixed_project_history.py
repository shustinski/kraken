"""Analysis and domain events coexist without breaking Hub's read surfaces."""
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from kraken_manager.infrastructure.filesystem import FilesystemEventStore
from kraken_manager.infrastructure.filesystem.event_store import CorruptEventLogError
from kraken_manager.domain.events import ActorSnapshot
from tests.test_work_orders import _service, _project_with_frames


def test_generic_analysis_history_retains_identity_time_and_unknown_actor(tmp_path):
    local = _service(tmp_path)
    _, _, project, layer, images, _ = _project_with_frames(local, tmp_path / "source", with_vectors=False)
    store = FilesystemEventStore(local.catalog_root, project.id)
    before = tuple(s.data for s in store.iter_project())
    timestamp = "2020-01-02T03:04:05+00:00"
    store.append("analysis-run:"+str(uuid4()), expected_revision=0, events=({"event_type": "analysis_run.created",
        "payload": {"run_id": str(uuid4()), "state": "created"}, "recorded_at": timestamp},))
    stored = tuple(store.iter_project())[-1]
    history = local.history(project.id)
    event = history[-1]
    assert event.event_id == stored.event_id and event.payload == stored.data["payload"]
    assert event.recorded_at == datetime.fromisoformat(timestamp) and event.actor is None
    assert event not in local.history(project.id, as_of=datetime(2020, 1, 1, tzinfo=UTC))
    assert local.plugin_jobs() == ()
    assert local.latest_karakal_analysis(project.id, layer.id) is None
    assert local.frame_cells(project.id, layer.id, images.id)
    activity = next(r for r in local.activity_records() if r.event_id == event.event_id)
    assert activity.event_type == event.event_type and activity.actor_id == "" and activity.recorded_at == event.recorded_at
    assert tuple(s.data for s in store.iter_project())[:-1] == before
    assert local.history(project.id) == history


def test_malformed_generic_history_is_reported_instead_of_hidden(tmp_path):
    local = _service(tmp_path)
    _, _, project, *_ = _project_with_frames(local, tmp_path / "source", with_vectors=False)
    store = FilesystemEventStore(local.catalog_root, project.id)
    store.append("generic:"+str(uuid4()), expected_revision=0, events=({"event_type": "broken", "payload": {},
        "recorded_at": "2020-01-02T03:04:05"},))
    with pytest.raises(CorruptEventLogError, match="timezone"):
        local.history(project.id)


def test_generic_history_preserves_a_recorded_principal_snapshot(tmp_path):
    from dataclasses import asdict
    local = _service(tmp_path)
    owner, _, project, *_ = _project_with_frames(local, tmp_path / "source", with_vectors=False)
    store = FilesystemEventStore(local.catalog_root, project.id)
    actor = ActorSnapshot.from_principal(owner.principal)
    store.append("generic:"+str(uuid4()), expected_revision=0, events=({"event_type": "generic.audit", "payload": {},
        "recorded_at": "2020-01-02T03:04:05+00:00", "actor": asdict(actor)},))
    event = local.history(project.id)[-1]
    assert event.actor == actor
    assert next(r for r in local.activity_records() if r.event_id == event.event_id).actor_id == str(actor.principal_id)
