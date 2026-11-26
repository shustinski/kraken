"""Server actions read the one record they need instead of the whole project log."""

from urllib.parse import parse_qs, urlsplit

import pytest

from kraken_hub.infrastructure.remote.parts.remove_layer_pipeline_action import (
    RemoveLayerPipelineActionMixin,
)
from kraken_hub.infrastructure.remote.parts.submit_plugin_job import SubmitPluginJobMixin
from kraken_manager.domain.common import LayerId, PrincipalId, ProjectId, RepresentationId
from kraken_manager.domain.selection import FrameRowRange, FrameSelectionV1
from kraken_manager.domain.workflows import PluginJob
from kraken_manager.infrastructure.filesystem._codec import encode_model
from kraken_server.services import InMemoryServerServices, NotFoundError

_JOB_ID = "00000000-0000-0000-0000-0000000000aa"
_PROJECT_ID = "00000000-0000-0000-0000-0000000000bb"


class _Remote(RemoveLayerPipelineActionMixin, SubmitPluginJobMixin):
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def history(self, *_args, **_kwargs):
        raise AssertionError("full project history")

    def _call(self, method, path, **_kwargs):
        self.calls.append((method, path.split("?", 1)[0]))
        if "/events/" in path:
            return {
                "event_id": "event",
                "event_type": "LayerPipelineActionRequested",
                "stream_id": "layer-pipeline:layer",
                "revision": 3,
                "recorded_at": "2026-01-01T00:00:00+00:00",
                "payload": {"layer_id": "layer", "action": "train", "plugin_job_id": "job"},
            }
        if path.endswith("/stream-revision") or "stream-revision?" in path:
            return {"revision": 7}
        if path.endswith(f"/jobs/{_JOB_ID}"):
            return encode_model(
                PluginJob.create(
                    project_id=ProjectId(_PROJECT_ID),
                    layer_id=LayerId("00000000-0000-0000-0000-0000000000cc"),
                    selection=FrameSelectionV1(row_ranges=(FrameRowRange(y=1, x_start=1, x_end=1),)),
                    actor_principal_id=PrincipalId("00000000-0000-0000-0000-0000000000dd"),
                    target_representation_id=RepresentationId("00000000-0000-0000-0000-0000000000ee"),
                    capability="frames.vectorize.v1",
                    job_id=_JOB_ID,
                )
            )
        return {"event_id": "removed", "recorded_at": "2026-01-01T00:00:00+00:00"}


def test_history_asks_the_server_for_the_requested_event_types():
    class Client(SubmitPluginJobMixin):
        def __init__(self) -> None:
            self.paths: list[str] = []

        def _call(self, method, path):
            self.paths.append(path)
            return {"items": [], "next_cursor": None}

    client = Client()
    assert client.history("project", event_types=("Wanted", "Other")) == ()
    query = parse_qs(urlsplit(client.paths[0]).query)
    assert query["event_type"] == ["Wanted", "Other"]
    assert query["limit"] == ["500"]


def test_removing_a_pipeline_step_does_not_download_project_history():
    client = _Remote()
    client.remove_layer_pipeline_action(
        principal=None,
        project_id="project",
        layer_id="layer",
        action_event_id="event",
    )
    assert client.calls == [
        ("GET", "/api/v1/projects/project/events/event"),
        ("GET", "/api/v1/projects/project/stream-revision"),
        ("POST", "/api/v1/projects/project/pipeline-actions"),
    ]


def test_plugin_job_reads_one_job():
    client = _Remote()
    job = client.plugin_job(_PROJECT_ID, _JOB_ID)
    assert str(job.id) == _JOB_ID
    assert client.calls == [("GET", f"/api/v1/projects/{_PROJECT_ID}/jobs/{_JOB_ID}")]


def test_history_requests_only_the_named_scope():
    class Client(SubmitPluginJobMixin):
        def __init__(self) -> None:
            self.paths: list[str] = []

        def _call(self, method, path):
            self.paths.append(path)
            return {"items": [], "next_cursor": None}

    client = Client()
    assert client.history("project", scope_id="layer-1") == ()
    assert parse_qs(urlsplit(client.paths[0]).query)["scope_id"] == ["layer-1"]


def test_pipeline_snapshot_requests_the_layer_history():
    from types import SimpleNamespace

    from kraken_hub.presentation.qt.parts.pipeline_snapshot import (
        PipelineSnapshotMixin,
        _PIPELINE_HISTORY_TYPES,
    )

    calls = []

    def history(project_id, **kwargs):
        calls.append((project_id, kwargs))
        return ()

    snapshot = PipelineSnapshotMixin()
    snapshot.service = SimpleNamespace(
        list_representations=lambda *_args: (),
        history=history,
        list_derived_runs=lambda *_args: (),
        list_artifact_series=lambda *_args, **_kwargs: (),
    )
    snapshot._pipeline_snapshot("project", "layer-1")
    assert calls == [("project", {"scope_id": "layer-1", "event_types": _PIPELINE_HISTORY_TYPES})]
    assert "ArtifactVersionCreated" not in _PIPELINE_HISTORY_TYPES
    assert "ArtifactSeriesCreated" not in _PIPELINE_HISTORY_TYPES


def test_pipeline_links_a_dataset_from_the_job_result_without_file_history():
    from datetime import UTC, datetime
    from types import SimpleNamespace

    from kraken_hub.presentation.qt.parts.pipeline_snapshot import PipelineSnapshotMixin
    from kraken_manager.domain.project import RepresentationKind, RepresentationPurpose

    now = datetime.now(UTC)
    source = SimpleNamespace(
        id="source", name="Images", note="", kind=RepresentationKind.IMAGE,
        purpose=RepresentationPurpose.SOURCE, active=True, source="managed-import", created_at=now,
    )

    def created(job_id, capability, **parameters):
        return SimpleNamespace(
            event_type="PluginJobCreated", event_id=f"created:{job_id}", recorded_at=now,
            actor=SimpleNamespace(display_name="Owner"),
            payload={
                "plugin_job_id": job_id,
                "job": {"layer_id": "layer", "state": "succeeded"},
                "manifest": {
                    "layer_id": "layer",
                    "capability": capability,
                    "target_representation_id": "source",
                    "parameters": {"source_representation_id": "source", **parameters},
                },
            },
        )

    events = (
        created("dataset-job", "frames.dataset.prepare.v1"),
        SimpleNamespace(
            event_type="PluginResultImported", event_id="imported:dataset-job", recorded_at=now,
            actor=SimpleNamespace(display_name="Owner"),
            payload={
                "plugin_job_id": "dataset-job",
                "job": {"layer_id": "layer", "state": "succeeded"},
                "artifact_version_ids": ["dataset-version"],
            },
        ),
        created("train-job", "dataset.model.train.v1", dataset_version_id="dataset-version"),
    )
    snapshot = PipelineSnapshotMixin()
    snapshot.service = SimpleNamespace(
        list_representations=lambda *_args: (source,),
        history=lambda *_args, **_kwargs: events,
        list_derived_runs=lambda *_args: (),
    )
    edges = snapshot._pipeline_snapshot("project", "layer").lanes[0].edges
    assert ("dataset-job:dataset", "train-job") in edges


def test_memory_history_returns_only_the_requested_scope():
    service = InMemoryServerServices()
    service._projects["project"] = {"name": "Project", "state": "active"}
    service._events["project"] = [
        {"event_id": "1", "event_type": "LayerPipelineActionRequested", "stream_id": "layer-pipeline:layer-1", "payload": {"layer_id": "layer-1"}},
        {"event_id": "2", "event_type": "PluginJobCreated", "stream_id": "job:other", "payload": {"manifest": {"layer_id": "layer-2"}}},
    ]
    page = service.history("project", cursor=None, limit=10, scope_id="layer-1")
    assert [item["event_id"] for item in page["items"]] == ["1"]


def test_memory_history_returns_only_requested_event_types():
    service = InMemoryServerServices()
    service._projects["project"] = {"name": "Project", "state": "active"}
    service._events["project"] = [
        {"event_id": "1", "event_type": "Wanted", "stream_id": "layer-pipeline:layer", "revision": 1},
        {"event_id": "2", "event_type": "Other", "stream_id": "other", "revision": 9},
        {"event_id": "3", "event_type": "Wanted", "stream_id": "layer-pipeline:layer", "revision": 2},
    ]
    page = service.history("project", cursor=None, limit=10, event_types=("Wanted",))
    assert [item["event_id"] for item in page["items"]] == ["1", "3"]
    assert service.stream_revision("project", "layer-pipeline:layer") == 2
    assert service.stream_revision("project", "missing") == 0
    assert service.project_event("project", "2")["event_type"] == "Other"
    with pytest.raises(NotFoundError):
        service.project_event("project", "missing")
