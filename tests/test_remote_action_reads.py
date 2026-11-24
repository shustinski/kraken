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
