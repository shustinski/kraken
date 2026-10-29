"""PostgreSQL graph -> actual remote Contour -> V2 vector mosaic import."""
import json
from pathlib import Path
import socket
import sys
import threading
import time
from uuid import uuid4

from kraken_core.plugins import load_plugin_metadata_from_directory
from kraken_agent.runner import PluginProcessSpec, PluginRegistry
from kraken_agent.server_worker import ServerAgentWorker
from kraken_manager.application.plugin_jobs import SubmitPluginJobHandler, CancelPluginJobHandler
from kraken_manager.domain.multilayer import MultilayerModel
from kraken_manager.domain.pipeline import LayerPipelineDefinition, PipelineOperation
from kraken_manager.domain.pipeline_layer import PipelineLayerContent, PipelineLayerOperation
from kraken_manager.infrastructure.mosaic import MosaicReader, STITCH_OPERATION
from kraken_server.agent_auth import PostgresAgentTokenStore
from kraken_server.persistent_services_parts.layer_source_scan import SystemClock
from kraken_server.services import CommandContext
from tests.test_server_stitch_workflow import setup_stitch
import uvicorn

pytest_plugins = ["tests.test_admin_project_workflows"]


def test_actual_remote_contour_vector_stitch_and_bounded_overlay_api(database, tmp_path, qtbot, monkeypatch):
    service, actor, pid, queue, app, _, _, _ = setup_stitch(database, tile_size=64)
    plan = MultilayerModel.from_dict(service.get_multilayer_model(pid)).stitches[-1]
    lid = plan.layer_id
    first = service.projections.get_artifact_version(plan.tiles[0].calibration.artifact.version_id)
    rid = str(service.projections.get_artifact_series(first.series_id).representation_id)
    vector = service.create_representation(pid, lid, {"name": "Actual CIF", "kind": "vector", "source_image_representation_id": rid},
        CommandContext(actor.id, "vectors", service.projections.get_layer(lid).revision))
    root = Path(__file__).resolve().parents[1]
    metadata = load_plugin_metadata_from_directory(root / "plugins/contour")
    contour = PipelineOperation(str(uuid4()), "Actual server Contour", "frames.vectorize.v1", metadata.version, rid,
        vector["representation_id"], "image", "vector", ((1, 1), (2, 1)), parameters={"execution_mode": "headless"})
    stitch = PipelineLayerOperation(str(uuid4()), "Vector mosaic", STITCH_OPERATION, "1.0.0", plan.id, plan.revision,
        dependencies=(contour.id,), content=PipelineLayerContent("vector", producer_id=contour.id), parameters={"block_size": 32})
    definition = LayerPipelineDefinition(pid, lid, (contour, stitch))
    service._submit_plugin_job = SubmitPluginJobHandler(service.uow_factory, service.profiles, SystemClock(), queue)
    service._cancel_plugin_job = CancelPluginJobHandler(service.uow_factory, service.profiles, SystemClock(), queue)
    runtime = service._layer_runtime()
    state = runtime.define(pid, lid, definition.to_dict(), principal=actor, expected_revision=0, idempotency_key="define")
    runtime.command(pid, lid, "start", principal=actor, expected_revision=state["revision"], idempotency_key="start")
    _, token = PostgresAgentTokenStore(service.engine).create("actual-vector-worker", frozenset({STITCH_OPERATION, contour.capability}))
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="off"))
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    url = f"http://127.0.0.1:{sock.getsockname()[1]}"
    registry = PluginRegistry({STITCH_OPERATION: PluginProcessSpec(STITCH_OPERATION,
        (sys.executable, "-m", "kraken_manager.infrastructure.stitch_worker")), contour.capability:
        PluginProcessSpec(contour.capability, (sys.executable, str(root / "plugins/contour/__main__.py")))})
    worker = ServerAgentWorker(server_url=url, token=token, data_dir=tmp_path / "agent", registry=registry)
    thread = threading.Thread(target=lambda: server.run(sockets=[sock]), daemon=True)
    thread.start()
    started = time.monotonic()
    try:
        while not server.started and time.monotonic()-started < 5:
            time.sleep(.01)
        assert server.started
        while time.monotonic()-started < 80:
            state = runtime.tick(pid, lid, principal=actor)
            if state["status"] in {"completed", "failed"}:
                break
            worker.run_once()
        assert state["status"] == "completed", json.dumps(state)
        output_id, = state["nodes"][stitch.id]["attempts"][-1]["output_version_ids"]
        output = service.projections.get_artifact_version(output_id)
        assert set(state["nodes"][contour.id]["attempts"][-1]["output_version_ids"]) <= set(output.input_version_ids)
        with MosaicReader(service.uow_factory.blobs.path_for_read(output.blob)) as reader:
            reader.validate(expected_plan=plan)
            assert reader.metadata["vector_stitch"]["source_features"] > 0
            assert reader.metadata["vector_stitch"]["conflict_blocks"] > 0
        import httpx
        job = state["nodes"][stitch.id]["attempts"][-1]["job_id"]
        response = httpx.get(url + f"/api/v1/projects/{pid}/stitch-renders/{job}/blocks/0/0", headers={"Authorization": "Bearer owner"})
        assert response.status_code == 200, response.text
        assert response.json()["metadata"]["vector_stitch"]["electrical_connectivity"] == "unverified"
        from kraken_hub.infrastructure.remote.client import RemoteAuth, RemoteServerProjectService
        from kraken_hub.presentation.qt.computed_stitch_dialog import MosaicBlockDialog
        from kraken_manager.infrastructure.vector_exchange import VectorExchangeReader
        from PyQt6.QtWidgets import QFileDialog
        remote = RemoteServerProjectService(url, auth=RemoteAuth(access_token="owner", principal=actor), data_dir=tmp_path / "remote")
        monkeypatch.setattr(remote._http, "download", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("no full mosaic download")))
        endpoint = url + f"/api/v1/projects/{pid}/stitch-renders/{job}"
        assert httpx.get(endpoint + "/vectors").status_code == 401
        assert httpx.get(endpoint + "/vector-blocks/-1/0", headers={"Authorization": "Bearer owner"}).status_code == 422
        exported = tmp_path / "remote-vectors.jsonl"
        dialog = MosaicBlockDialog(None, block_loader=lambda c, r, coverage: remote.remote_stitch_block(pid, job, c, r, coverage=coverage),
            vector_exporter=lambda destination, **kwargs: remote.export_stitch_vectors(project_id=pid, job_id=job, destination=destination, **kwargs))
        qtbot.addWidget(dialog)
        dialog.show()
        qtbot.waitUntil(lambda: dialog._worker is None, timeout=10000)
        assert dialog.export_button.isEnabled()
        monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *_args, **_kwargs: (str(exported), ""))
        dialog.export_vectors()
        qtbot.waitUntil(lambda: dialog._worker is None, timeout=10000)
        assert "сохранены" in dialog.status.text(), dialog.status.text()
        with VectorExchangeReader(exported) as exchange:
            blocks = list(exchange.blocks())
            assert exchange.verified and exchange.header["output_version_id"] == output_id
            assert any(b["conflicts"] for b in blocks)
        first_bytes = exported.read_bytes()
        original_block = remote.stitch_vector_block
        calls = []
        interrupted = tmp_path / "interrupted-vectors.jsonl"
        def lose_second_block(*args, **kwargs):
            payload = original_block(*args, **kwargs)
            calls.append(args[-2:])
            if len(calls) == 2:
                raise TimeoutError("controlled lost vector block response")
            return payload
        monkeypatch.setattr(remote, "stitch_vector_block", lose_second_block)
        monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *_args, **_kwargs: (str(interrupted), ""))
        dialog.export_vectors()
        qtbot.waitUntil(lambda: dialog._worker is None, timeout=10000)
        assert len(calls) == 2 and not interrupted.exists() and not list(tmp_path.glob(".*.part-*"))
        assert "controlled lost" in dialog.status.text()
        assert dialog._export_cancel is None and dialog.export_button.text() == "Экспортировать векторы…"
        monkeypatch.setattr(remote, "stitch_vector_block", original_block)
        monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *_args, **_kwargs: (str(exported), ""))
        dialog.export_vectors()
        qtbot.waitUntil(lambda: dialog._worker is None, timeout=10000)
        assert exported.read_bytes() == first_bytes
        retry = tmp_path / "retried-vectors.jsonl"
        monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *_args, **_kwargs: (str(retry), ""))
        dialog.export_vectors()
        qtbot.waitUntil(lambda: dialog._worker is None, timeout=10000)
        assert retry.read_bytes() == first_bytes
        dialog.close()
        (tmp_path / "receipt.json").write_text(json.dumps({"status": "passed", "training_started": False,
            "project_id": pid, "mosaic_sha256": output.sha256, "seconds": time.monotonic()-started}))
    finally:
        worker.stop()
        server.should_exit = True
        thread.join(timeout=10)
        sock.close()
