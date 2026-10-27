"""Real PostgreSQL pipeline CAS, pinned queue handoff and HTTP contracts."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from uuid import uuid4


from kraken_manager.application.plugin_jobs import SubmitPluginJobHandler, CancelPluginJobHandler
from kraken_manager.domain.artifacts import deterministic_frame_series_id
from kraken_manager.domain.pipeline import LayerPipelineDefinition, PipelineOperation
from kraken_manager.infrastructure.postgres.event_store import PostgresEventStore, PostgresRevisionConflict
from kraken_server.persistent_services_parts.layer_source_scan import SystemClock
from kraken_server.services import CommandContext
from tests.test_admin_project_workflows import project
from tests.test_pipeline_runtime_integration import Gateway

pytest_plugins = ["tests.test_admin_project_workflows"]


def setup(database):
    services, _, actor, _ = database
    record = project(services, actor)
    identifier = record["project_id"]
    layer = services.create_layer(
        identifier, {"name": "M1", "type": "metal", "order": 0}, CommandContext(actor.id, "layer", record["revision"])
    )
    layer_id = layer["layer_id"]
    image = services.create_representation(
        identifier,
        layer_id,
        {"name": "Images", "kind": "image"},
        CommandContext(actor.id, "images", services.projections.get_layer(layer_id).revision),
    )
    image_id = image["representation_id"]
    vector = services.create_representation(
        identifier,
        layer_id,
        {"name": "Vectors", "kind": "vector", "source_image_representation_id": image_id},
        CommandContext(actor.id, "vectors", services.projections.get_layer(layer_id).revision),
    )
    frame_id = services.projections.get_project(identifier).frame_id_at(1, 1)
    series = services.create_artifact_series(
        identifier,
        dict(
            name="Input frame",
            scope="frame_representation",
            layer_id=layer_id,
            representation_id=image_id,
            frame_id=frame_id,
            series_id=deterministic_frame_series_id(image_id, frame_id),
        ),
        CommandContext(actor.id, "series", None),
    )
    version = services.add_managed_artifact_version(
        identifier,
        series["id"],
        {"filename": "frame.png", "media_type": "image/png"},
        (b"queue transport fixture",),
        CommandContext(actor.id, "version", services.artifact_stream_revision(identifier, series["id"])),
    )
    op = PipelineOperation(
        str(uuid4()),
        "Contour",
        "frames.vectorize.v1",
        "1.0",
        image_id,
        vector["representation_id"],
        "image",
        "vector",
        ((1, 1),),
    )
    definition = LayerPipelineDefinition(identifier, layer_id, (op,))
    gateway = Gateway()
    services._submit_plugin_job = SubmitPluginJobHandler(
        services.uow_factory, services.profiles, SystemClock(), gateway
    )
    services._cancel_plugin_job = CancelPluginJobHandler(
        services.uow_factory, services.profiles, SystemClock(), gateway
    )
    state = services.transport_layer_pipeline(
        identifier, layer_id, definition.to_dict(), CommandContext(actor.id, "define", 0), define=True
    )
    return services, actor, identifier, layer_id, definition, state, gateway, version


def test_real_postgres_pipeline_concurrent_reservations_and_exact_inputs(database, monkeypatch):
    services, actor, identifier, layer_id, definition, state, gateway, version = setup(database)
    state = services.transport_layer_pipeline(
        identifier, layer_id, {"operation": "start"}, CommandContext(actor.id, "start", state["revision"])
    )
    barrier = Barrier(2)
    reservation_revision = state["revision"]
    original = PostgresEventStore.append

    def synchronized(self, stream_id, *, expected_revision, events):
        if stream_id == f"layer-pipeline:{layer_id}" and expected_revision == reservation_revision:
            barrier.wait(timeout=10)
        return original(self, stream_id, expected_revision=expected_revision, events=events)

    monkeypatch.setattr(PostgresEventStore, "append", synchronized)

    def tick(_):
        try:
            return services._layer_runtime().tick(identifier, layer_id, principal=actor)
        except PostgresRevisionConflict:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(tick, (1, 2)))
    assert sum(result is not None for result in results) == 1
    state = services.get_layer_pipeline(identifier, layer_id)
    attempt = state["nodes"][definition.operations[0].id]["attempts"][-1]
    assert len(gateway.manifests) == 1
    assert str(next(iter(gateway.manifests.values())).inputs[0].artifact_version_id) == version["id"]
    assert len(services.events.load_stream(f"plugin-job:{attempt['job_id']}")) == 1
    services._layer_runtime().tick(identifier, layer_id, principal=actor)
    assert len(gateway.deliveries) == 1


def test_real_http_pipeline_validation_replay_and_lifespan_scheduler(database):
    from fastapi.testclient import TestClient
    from kraken_server.app import SessionPrincipal, create_app

    services, actor, identifier, layer_id, definition, state, gateway, version = setup(database)
    session = SessionPrincipal(str(actor.id), "local", "owner")
    app = create_app(
        services=services, development=False, session_resolver=lambda token: session if token == "owner" else None
    )
    base = f"/api/v1/projects/{identifier}/layers/{layer_id}/pipeline/workflow"
    with TestClient(app) as client:
        headers = {"Authorization": "Bearer owner", "Idempotency-Key": "start", "If-Match": str(state["revision"])}
        response = client.post(base + "/commands", json={"operation": "start"}, headers=headers)
        assert response.status_code == 200, response.text
        assert client.post(base + "/commands", json={"operation": "start"}, headers=headers).status_code == 200
        assert client.post(base + "/commands", json={"operation": "cancel"}, headers=headers).status_code == 409
        assert client.get(base).status_code == 401
        assert client.put(base, json=definition.to_dict() | {"nodes": {}}, headers=headers).status_code == 422
        # Exercise the same cycle called by lifespan, without a timing-dependent
        # sleep or pretending a transport fixture is an image computation.
        services.run_pipeline_cycle()
        assert gateway.manifests
