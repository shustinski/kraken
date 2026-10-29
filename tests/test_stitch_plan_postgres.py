"""Actual PostgreSQL physical stream and authenticated plan commands."""
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

from fastapi.testclient import TestClient
import sqlalchemy as sa

from kraken_admin.infrastructure.composition import open_project_services
from kraken_manager.domain.artifacts import deterministic_frame_series_id
from kraken_manager.domain.multilayer import MultilayerModel
from kraken_manager.infrastructure.postgres.event_store import PostgresEventStore
from kraken_server.app import SessionPrincipal, create_app
from kraken_server.services import CommandContext, ConflictError
from tests.test_admin_project_workflows import project

pytest_plugins = ["tests.test_admin_project_workflows"]


def test_stitch_plan_http_pg_reopen_and_concurrent_cas(database, monkeypatch):
    service, _, owner, config = database
    identifier = project(service, owner)["project_id"]
    layer = service.create_layer(identifier, {"name": "Metal", "type": "metal", "order": 0},
        CommandContext(owner.id, "layer", service.get_project(identifier)["revision"]))
    layer_id = layer["layer_id"]
    representation = service.create_representation(identifier, layer_id, {"name": "Inputs", "kind": "image"},
        CommandContext(owner.id, "representation", layer["revision"]))["representation_id"]
    frames = []
    for i in range(2):
        frame = service.projections.get_project(identifier).frame_id_at(i+1, 1)
        series = service.create_artifact_series(identifier, {"scope": "frame_representation", "layer_id": layer_id,
            "representation_id": representation, "frame_id": frame, "name": f"tile-{i}",
            "series_id": deterministic_frame_series_id(representation, frame)}, CommandContext(owner.id, f"series-{i}", None))
        version = service.add_managed_artifact_version(identifier, series["id"],
            {"filename": f"tile-{i}.pgm", "media_type": "image/x-portable-graymap"},
            (b"P5\n4 4\n255\n" + bytes([i+1]) * 16,),
            CommandContext(owner.id, f"version-{i}", service.artifact_stream_revision(identifier, series["id"])))
        frames.append({"frame_id": frame, "artifact_version_id": version["id"], "width": 4, "height": 4,
                       "pixel_size_m": [1e-6, 2e-6], "y_axis": "down"})
    frames[1].update(declared_affine=[1, 0, 2e-6, 0, 1, 0], note="controlled functional stage transform")
    series = service.create_artifact_series(identifier, {"scope": "layer_attachment", "layer_id": layer_id,
        "name": "Observation evidence"}, CommandContext(owner.id, "plan-series", None))
    plan = service.add_managed_artifact_version(identifier, series["id"], {"filename": "plan.json", "media_type": "application/json"},
        (b'{"purpose":"functional fixture"}',), CommandContext(owner.id, "plan-blob", service.artifact_stream_revision(identifier, series["id"])))
    stack = {"layers": [{"layer_id": layer_id, "layer_type": "metal", "z_m": 0, "thickness_m": 0}]}
    service.transport_mutate_multilayer_model(identifier, {"operation": "set_stack", "arguments": stack}, CommandContext(owner.id, "stack", 0))
    payload = {"operation": "plan_layer_stitch", "arguments": {"layer_id": layer_id, "plan_version_id": plan["id"],
        "reference_frame_id": frames[0]["frame_id"], "tiles": frames, "max_error_m": 1e-12}}
    app = create_app(services=service, session_resolver=lambda token: SessionPrincipal(str(owner.id), "local", token))
    with TestClient(app) as client:
        base = f"/api/v1/projects/{identifier}/multilayer"
        assert client.post(base + "/commands", json=payload).status_code == 401
        headers = {"Authorization": "Bearer owner", "Idempotency-Key": "stitch-plan"}
        assert client.post(base + "/commands", json=payload, headers=headers).status_code == 422
        response = client.post(base + "/commands", json=payload, headers=headers | {"If-Match": "1"})
        assert response.status_code == 200, response.text
        receipt = response.json()
        assert MultilayerModel.from_dict(receipt).stitches[0].ready
        assert client.post(base + "/commands", json=payload, headers=headers | {"If-Match": "0"}).json() == receipt
        assert client.post(base + "/commands", json=payload,
            headers=headers | {"Idempotency-Key": "stale", "If-Match": "1"}).status_code == 409
    reopened = open_project_services(service.engine.url.render_as_string(hide_password=False), config.blob_root, engine=service.engine)
    assert reopened.get_multilayer_model(identifier) == receipt
    barrier = Barrier(2)
    original = PostgresEventStore.append

    def synchronize(self, stream_id, *, expected_revision, events):
        if stream_id == f"multilayer:{identifier}" and expected_revision == 2:
            barrier.wait(timeout=15)
        return original(self, stream_id, expected_revision=expected_revision, events=events)

    monkeypatch.setattr(PostgresEventStore, "append", synchronize)

    def write(i):
        try:
            return service.transport_mutate_multilayer_model(identifier, payload, CommandContext(owner.id, f"race-{i}", 2))
        except ConflictError:
            return None
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(write, range(2)))
    assert sum(result is not None for result in results) == 1
    current = reopened.get_multilayer_model(identifier)
    assert current["model"]["revision"] == 3
    assert current["assessment"]["stitches"][0]["current"] is False
    assert current["assessment"]["stitches"][1]["current"] is True
    with service.engine.connect() as connection:
        count = connection.execute(sa.text("SELECT count(*) FROM transactional_outbox WHERE event_type='MultilayerModelChangedV1'" )).scalar_one()
    assert count == 3
    # Real mosaic computation, then atomically attach its artifact and physical
    # receipt. Failure after artifact events must roll back the whole database.
    from kraken_agent.jobs import StagingWorkspace
    from kraken_core.plugin_protocol import PluginJobManifestV2
    from kraken_manager.infrastructure.stitch_worker import execute
    from kraken_manager.infrastructure.mosaic import MosaicReader
    from uuid import UUID, uuid5
    import pytest
    stitch = MultilayerModel.from_dict(current).stitches[-1]
    requested = service.transport_mutate_multilayer_model(identifier,
        {"operation": "request_layer_stitch", "arguments": {"stitch_id": stitch.id, "stitch_revision": stitch.revision,
            "block_size": 2, "frame_coordinates": {str(f["frame_id"]): [i+1, 1] for i, f in enumerate(frames)}}},
        CommandContext(owner.id, "render-request", 3))
    request = MultilayerModel.from_dict(requested).stitch_renders[-1]
    manifest = PluginJobManifestV2.from_json(request.manifest_json)
    workspace = StagingWorkspace(config.blob_root / "staging", request.job_id)
    workspace.create()
    for asset in manifest.inputs:
        workspace.stage_file(service.uow_factory.blobs._path(asset.sha256), asset.relative_path, expected_sha256=asset.sha256)
    publication = execute(manifest, workspace.path)
    source_path = workspace.resolve(publication.outputs[0].relative_path)
    base_append = original

    def reject_import(self, stream_id, *, expected_revision, events):
        result = base_append(self, stream_id, expected_revision=expected_revision, events=events)
        if stream_id == f"multilayer:{identifier}" and events[-1].payload.get("operation") == "import_layer_stitch":
            raise OSError("injected mosaic transaction failure")
        return result

    monkeypatch.setattr(PostgresEventStore, "append", reject_import)
    with pytest.raises(OSError, match="transaction failure"):
        service.commit_stitch_publication(identifier, publication, source_path, CommandContext(owner.id, "result", None))
    assert reopened.get_multilayer_model(identifier) == requested
    assert service.projections.get_artifact_version(str(uuid5(UUID(request.job_id), "mosaic-version"))) is None
    assert service.projections.get_artifact_series(str(uuid5(UUID(request.job_id), "mosaic-series"))) is None
    monkeypatch.setattr(PostgresEventStore, "append", base_append)
    imported = service.commit_stitch_publication(identifier, publication, source_path, CommandContext(owner.id, "result", None))
    result = MultilayerModel.from_dict(imported).stitch_renders[-1]
    assert result.state == "imported"
    version = service.projections.get_artifact_version(result.output.version_id)
    assert len(version.input_version_ids) == 3
    with MosaicReader(service.uow_factory.blobs._path(version.sha256)) as reader:
        reader.validate(expected_plan=stitch, expected_job_id=request.job_id)
    assert reopened.get_multilayer_model(identifier) == imported
    assert service.commit_stitch_publication(identifier, publication, None, CommandContext(owner.id, "result", None)) == imported
