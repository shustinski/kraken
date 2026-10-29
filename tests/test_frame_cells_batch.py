"""Large current reads retain version ancestry, review state and temporal reads."""

from datetime import UTC, datetime
import sqlite3
from uuid import uuid4

from kraken_manager.infrastructure.filesystem import SQLiteProjectionStore
from tests.test_work_orders import _project_with_frames, _service


def test_current_cells_batch_versions_and_keep_parent_coordinates(tmp_path, monkeypatch):
    service = _service(tmp_path)
    owner, performer, project, layer, images, vectors = _project_with_frames(service, tmp_path / "source")
    original = service.frame_cells(project.id, layer.id, vectors.id)[0]
    before = datetime.now(UTC)
    series = next(s for s in service.list_artifact_series(project.id, representation_id=vectors.id)
                  if str(s.frame_id) == original.frame_id)
    source = tmp_path / "changed.cif"
    source.write_text("DS 1 1 1; DF; E;", encoding="utf-8")
    changed = service.add_managed_artifact_version(
        principal=owner.principal, project_id=project.id, series_id=series.id,
        source=source, parent_version_id=original.artifact_version_id, idempotency_key="changed",
    )
    assert "x" not in changed.parameters
    service.publish_karakal_analysis(
        principal=owner.principal, project_id=project.id, layer_id=layer.id,
        frame_confidence={original.frame_id: 0.75}, report={}, parameters={},
        plugin_version="test", idempotency_key="analysis",
    )
    batch = service.create_review_batch(
        principal=owner.principal, project_id=project.id, layer_id=layer.id,
        image_representation_id=images.id, vector_representation_id=vectors.id,
        coordinates=((1, 1),), assignee_id=performer.id, instructions="Check",
        due_at=None, idempotency_key="review",
    )
    service.export_review_batch(
        principal=owner.principal, batch=batch, destination=tmp_path / "review-package",
        idempotency_key="issue-review",
    )
    single = SQLiteProjectionStore.get_active_artifact_version

    def temporal_only(self, series_id, *, as_of=None):
        assert as_of is not None, "Current matrix must not issue one active-version query per frame"
        return single(self, series_id, as_of=as_of)

    monkeypatch.setattr(SQLiteProjectionStore, "get_active_artifact_version", temporal_only)
    current = service.frame_cells(project.id, layer.id, vectors.id)
    updated = next(c for c in current if c.frame_id == original.frame_id)
    assert (updated.x, updated.y) == (original.x, original.y)
    assert updated.artifact_version_id == str(changed.id)
    assert updated.sha256 == changed.sha256
    assert updated.status == "in_review" and updated.review_status == "in_review"
    assert updated.quality == 0.75
    assert service.frame_cells(project.id, layer.id, vectors.id, bounds=(1, 1, 1, 1)) == (updated,)
    historical = service.frame_cells(project.id, layer.id, vectors.id, as_of=before)
    assert next(c for c in historical if c.frame_id == original.frame_id) == original
    assert service.frame_cells(project.id, layer.id, vectors.id, as_of=before, bounds=(1, 1, 1, 1)) == (original,)


def test_large_batch_is_bounded_and_reads_new_heads(tmp_path, monkeypatch):
    service = _service(tmp_path)
    owner, _, project, _, images, _ = _project_with_frames(service, tmp_path / "source", with_vectors=False)
    series = service.list_artifact_series(project.id, representation_id=images.id)
    projection = service._projection(project.id)
    expected = {str(s.id): projection.get_active_artifact_version(s.id) for s in series}
    connect = SQLiteProjectionStore._connect

    def limited(self):
        connection = connect(self)
        connection.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 500)
        return connection

    monkeypatch.setattr(SQLiteProjectionStore, "_connect", limited)
    identifiers = [str(s.id) for s in series] + [str(uuid4()) for _ in range(1100)]
    assert projection.list_active_artifact_versions(iter(identifiers + identifiers)) == expected
    old = expected[str(series[0].id)]
    source = tmp_path / "replacement.png"
    source.write_bytes(b"changed image")
    new = service.add_managed_artifact_version(
        principal=owner.principal, project_id=project.id, series_id=series[0].id,
        source=source, parent_version_id=old.id, idempotency_key="replacement",
    )
    expected[str(series[0].id)] = new
    assert projection.list_active_artifact_versions(identifiers) == expected
    assert projection.list_active_artifact_versions(()) == {}
