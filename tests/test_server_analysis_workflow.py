"""Short functional Server/Agent/Karakal workflow on controlled binary inputs."""

import socket
from dataclasses import replace
import sys
import threading
import time
from uuid import uuid4

from fastapi.testclient import TestClient
import pytest
import uvicorn

from kraken_agent.runner import PluginRegistry, PluginProcessSpec
from kraken_agent.server_worker import ServerAgentWorker
from kraken_core.analysis_run_protocol import AnalysisExpression, AnalysisRecipe
from kraken_manager.domain.artifacts import deterministic_frame_series_id
from kraken_manager.domain.project import FrameCoordinate
from kraken_server.agent_gateway import PostgresAgentGateway
from kraken_server.agent_auth import PostgresAgentTokenStore
from kraken_server.app import create_app, SessionPrincipal
from kraken_server.services import CommandContext
from tests.test_admin_project_workflows import project
from tests.test_electrical_graph import png

pytest_plugins = ["tests.test_admin_project_workflows"]


def setup_analysis(database, *, corrupt=False):
    services, _accounts, owner, _config = database
    pid = project(services, owner, "Short functional analysis")["project_id"]
    layer = services.create_layer(
        pid,
        {"name": "Controlled masks", "type": "metal", "order": 0},
        CommandContext(str(owner.id), str(uuid4()), services.get_project(pid)["revision"]),
    )["layer_id"]
    bindings = {}
    frame = FrameCoordinate(1, 1).frame_id(pid)
    for key in ("A", "B"):
        rep = services.create_representation(
            pid,
            layer,
            {"name": key, "kind": "image", "purpose": "binary"},
            CommandContext(str(owner.id), str(uuid4()), services.projections.get_layer(layer).revision),
        )
        rid = rep.get("representation_id", rep.get("id"))
        bindings[key] = rid
        series = services.create_artifact_series(
            pid,
            {
                "name": key,
                "scope": "frame_representation",
                "layer_id": layer,
                "representation_id": rid,
                "frame_id": frame,
                "series_id": deterministic_frame_series_id(rid, frame),
            },
            CommandContext(str(owner.id), str(uuid4()), None),
        )
        services.add_managed_artifact_version(
            pid,
            series["id"],
            {"filename": key + ".png", "media_type": "image/png"},
            (b"invalid PNG fixture" if corrupt and key == "B" else png([[0, 255, 0, 0]] * 4),),
            CommandContext(str(owner.id), str(uuid4()), services.artifact_stream_revision(pid, series["id"])),
        )
    queue = PostgresAgentGateway(services.engine, None, services.uow_factory.blobs)
    services.agent_gateway = queue
    tokens = PostgresAgentTokenStore(services.engine)
    _, token = tokens.create("functional-analysis-worker", frozenset({"karakal.analyze.v1"}))
    app = create_app(
        services=services,
        session_resolver=lambda value: SessionPrincipal(str(owner.id), "local", value) if value == "owner" else None,
        agent_token_store=tokens,
        agent_gateway=queue,
    )
    payload = {
        "layer_id": layer,
        "coordinates": [[1, 1]],
        "bindings": bindings,
        "recipe": AnalysisRecipe(
            AnalysisExpression.binary("compare", AnalysisExpression.source("A"), AnalysisExpression.source("B")),
            metric_keys=("iou", "dice"),
        ).to_payload(),
    }
    return services, owner, pid, queue, app, token, payload


@pytest.mark.parametrize("corrupt", [False, True], ids=["success", "failed-input-and-retry"])
def test_remote_analysis_executes_actual_short_karakal_and_reopens(database, tmp_path, corrupt, monkeypatch, qtbot):
    services, owner, pid, queue, app, token, payload = setup_analysis(database, corrupt=corrupt)
    client = TestClient(app)
    headers = {"Authorization": "Bearer owner", "Idempotency-Key": "functional-analysis"}
    created = client.post(f"/api/v1/projects/{pid}/analysis-runs", headers=headers, json=payload)
    assert created.status_code == 201, created.text
    first = created.json()
    from kraken_hub.infrastructure.remote.parts.analysis_runs import ServerAnalysisMixin

    class Remote(ServerAnalysisMixin):
        def _call(self, method, path, *, payload=None, idempotency_key=None, if_match=None):
            request_headers = {"Authorization": "Bearer owner"}
            if idempotency_key:
                request_headers["Idempotency-Key"] = idempotency_key
            if if_match is not None:
                request_headers["If-Match"] = str(if_match)
            response = client.request(method, path, headers=request_headers, json=payload)
            response.raise_for_status()
            return response.json()

    remote = Remote()
    assert remote.server_analysis_run(pid, first["run_id"]) == first
    assert remote.server_analysis_runs(pid)[0]["run_id"] == first["run_id"]
    repeated = client.post(f"/api/v1/projects/{pid}/analysis-runs", headers=headers, json=payload)
    assert repeated.status_code == 201 and repeated.json()["run_id"] == first["run_id"], repeated.text
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="off"))
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    worker = ServerAgentWorker(
        server_url=f"http://127.0.0.1:{sock.getsockname()[1]}",
        token=token,
        data_dir=tmp_path / "worker",
        registry=PluginRegistry(
            {"karakal.analyze.v1": PluginProcessSpec("karakal.analyze.v1", (sys.executable, "-m", "karakal.worker"))}
        ),
    )
    thread = threading.Thread(target=lambda: server.run(sockets=[sock]), daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 5
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.01)
        assert server.started
        assert worker.run_once()
        result = services.get_analysis_run(pid, first["run_id"])
        assert result["state"] == ("failed" if corrupt else "completed"), result
        assert result["completed_frames"] == int(not corrupt) and result["failed_frames"] == int(corrupt)
        from karakal.runtime_identity import worker_runtime_identity
        assert result["worker_runtime"] == worker_runtime_identity().to_payload()
        measurements = services.analysis_frame_results(pid, first["run_id"], "iou")
        assert remote.server_analysis_frames(pid, first["run_id"], "iou") == tuple(measurements)
        assert len(measurements) == 1
        assert measurements[0]["raw_value"] is None if corrupt else measurements[0]["raw_value"] == pytest.approx(1)
        from kraken_admin.infrastructure.composition import open_project_services

        reopened = open_project_services(
            str(services.engine.url), services.uow_factory.blobs.root, engine=services.engine
        )
        assert reopened.get_analysis_run(pid, first["run_id"]) == result
        assert reopened.analysis_frame_results(pid, first["run_id"], "iou") == measurements
        if not corrupt:
            from io import BytesIO
            from PIL import Image

            frame_id = first["manifest"]["frame_ids"][0]
            pinned = first["partitions"][0]["manifest"]["frames"][0]["artifacts"][0]
            services.add_managed_artifact_version(
                pid, pinned["artifact_id"], {"filename": "new-head.png", "media_type": "image/png"},
                (b"Changed active head; must never be used by the existing analysis map",),
                CommandContext(str(owner.id), "change-head-before-map", services.artifact_stream_revision(pid, pinned["artifact_id"])),
            )
            path = f"/api/v1/projects/{pid}/analysis-runs/{first['run_id']}/frames/{frame_id}/map"
            map_headers = dict(headers, **{"Idempotency-Key": "render-map", "If-Match": str(result["revision"])})
            submit = queue.submit_analysis

            def unavailable(*_args, **_kwargs):
                raise OSError("Injected outage after durable map intent commit")

            with monkeypatch.context() as scoped:
                scoped.setattr(queue, "submit_analysis", unavailable)
                with pytest.raises(OSError, match="durable map intent"):
                    services.transport_request_analysis_map(
                        pid, first["run_id"], frame_id, CommandContext(str(owner.id), "render-map", result["revision"]))
            queued = services.get_analysis_run(pid, first["run_id"])
            assert queued["maps"][frame_id]["state"] == "queued"
            assert queue.submit_analysis == submit
            services.run_analysis_cycle()  # Completed analysis still repairs its pending map dispatch.
            repaired = services.get_analysis_run(pid, first["run_id"])
            assert repaired["maps"][frame_id]["state"] == "running"
            assert repaired["maps"][frame_id]["manifest"] == queued["maps"][frame_id]["manifest"]
            requested = client.post(path, headers=map_headers)
            assert requested.status_code == 202, requested.text
            again = client.post(path, headers=map_headers)
            assert again.status_code == 202 and again.json() == requested.json()
            assert worker.run_once()
            mapped = services.get_analysis_run(pid, first["run_id"])
            assert mapped["maps"][frame_id]["state"] == "ready", mapped
            assert mapped["maps"][frame_id]["manifest"]["frames"][0]["artifacts"][0] == pinned
            assert mapped["state"] == "completed" and mapped["completed_frames"] == 1
            assert services.analysis_frame_results(pid, first["run_id"], "iou") == measurements
            response = client.get(path, headers=headers)
            assert response.status_code == 200, response.text
            assert response.headers["content-type"] == "image/png"
            with Image.open(BytesIO(response.content)) as image:
                assert image.size == (4, 4)
                assert image.convert("RGB").getpixel((1, 0)) == (255, 255, 255)
                assert image.convert("RGB").getpixel((0, 0)) == (0, 0, 0)
            assert client.get(path).status_code == 401
            assert reopened.get_analysis_run(pid, first["run_id"]) == mapped
            assert reopened.analysis_map_bytes(pid, first["run_id"], frame_id) == response.content
            from types import SimpleNamespace
            from PyQt6.QtWidgets import QWidget, QVBoxLayout, QDialog, QLabel, QPlainTextEdit
            from kraken_hub.presentation.qt.remote_analysis import RemoteAnalysisController
            from kraken_manager.presentation.qt.analysis_ui import AnalysisRunsPanel

            class HttpBridge:
                def iter_bytes(self, request_path, *, token):
                    downloaded = client.get(request_path, headers={"Authorization": "Bearer " + token})
                    downloaded.raise_for_status()
                    yield downloaded.content

            remote._http = HttpBridge()
            remote.auth = SimpleNamespace(access_token="owner")
            widget = QWidget()
            widget.analysis_runs_panel = AnalysisRunsPanel(widget)
            QVBoxLayout(widget).addWidget(widget.analysis_runs_panel)
            qtbot.addWidget(widget)
            widget.show()
            controller = RemoteAnalysisController(widget, remote, pid, language="English")
            panel = widget.analysis_runs_panel
            qtbot.waitUntil(lambda: panel.result_table.rowCount() == 1, timeout=5000)
            qtbot.waitUntil(lambda: controller._worker is None)
            panel.result_table.selectRow(0)
            panel.map_button.click()
            qtbot.waitUntil(lambda: widget.findChild(QDialog, "remoteAnalysisMapDialog") is not None, timeout=5000)
            dialog = widget.findChild(QDialog, "remoteAnalysisMapDialog")
            pixmap = dialog.findChild(QLabel, "analysisMapImage").pixmap()
            assert pixmap.width() == 4 and pixmap.toImage().pixelColor(1, 0).red() == 255
            assert dialog.isVisible() and not dialog.isModal()
            qtbot.waitUntil(lambda: controller._worker is None)
            panel.runtime_button.click()
            qtbot.waitUntil(lambda: widget.findChild(QDialog, "analysisRuntimeDialog") is not None, timeout=5000)
            runtime_dialog = widget.findChild(QDialog, "analysisRuntimeDialog")
            assert result["worker_runtime"]["engine_build"] in runtime_dialog.findChild(QPlainTextEdit, "analysisRuntimeReport").toPlainText()
            qtbot.waitUntil(lambda: controller._worker is None)
            controller.timer.stop()
            runtime_dialog.close()
            dialog.close()
            widget.close()
            from kraken_manager.domain.artifacts import BlobRef
            map_file = services.uow_factory.blobs.path_for_read(BlobRef(**mapped["maps"][frame_id]["blob"]))
            original_png = map_file.read_bytes()
            try:
                map_file.write_bytes(b"corrupt generated test map")
                damaged = client.get(path, headers=headers)
                assert damaged.status_code == 409, damaged.text
            finally:
                map_file.write_bytes(original_png)
            # Explicit cancellation must revoke queued map work without replacing metrics.
            pending = client.post(path, headers=dict(map_headers, **{
                "Idempotency-Key": "render-map-to-cancel", "If-Match": str(mapped["revision"])})).json()
            cancelled = services.transport_analysis_command(
                pid, first["run_id"], {"operation": "cancel"},
                CommandContext(str(owner.id), "cancel-pending-map", pending["revision"]))
            assert cancelled["maps"][frame_id]["state"] == "cancelled"
            assert not worker.run_once()
            assert services.analysis_frame_results(pid, first["run_id"], "iou") == measurements
            mapped = cancelled
            result = mapped
        if corrupt:
            retried = services.transport_analysis_command(
                pid,
                first["run_id"],
                {"operation": "retry"},
                CommandContext(str(owner.id), "retry-invalid-frame", result["revision"]),
            )
            assert retried["state"] == "running"
            assert retried["partitions"][0]["manifest"]["expected_runtime"] == result["worker_runtime"]
            assert worker.run_once()
            retried = services.get_analysis_run(pid, first["run_id"])
            assert retried["state"] == "failed" and retried["failed_frames"] == 1
            result = retried
            measurements = services.analysis_frame_results(pid, first["run_id"], "iou")
        assert reopened.rebuild_analysis_run(pid, first["run_id"]) == result
        assert reopened.analysis_frame_results(pid, first["run_id"], "iou") == measurements
    finally:
        worker.stop()
        server.should_exit = True
        thread.join(5)
        sock.close()


def test_server_analysis_cancel_retry_repeat_and_queue_recovery_are_idempotent(database):
    from kraken_core.analysis_run_protocol import AnalysisPartitionJobManifest

    services, owner, pid, queue, app, token, payload = setup_analysis(database)
    context = CommandContext(str(owner.id), "create-run", None)
    first = services.transport_create_analysis_run(pid, payload, context)
    cancelled = services.transport_analysis_command(
        pid, first["run_id"], {"operation": "cancel"}, CommandContext(str(owner.id), "cancel-run", first["revision"])
    )
    assert cancelled["state"] == "cancelled"
    repeated = services.transport_analysis_command(
        pid, first["run_id"], {"operation": "cancel"}, CommandContext(str(owner.id), "cancel-run", first["revision"])
    )
    assert repeated == cancelled
    old_job = first["partitions"][0]["manifest"]["job_id"]
    retried = services.transport_analysis_command(
        pid, first["run_id"], {"operation": "retry"}, CommandContext(str(owner.id), "retry-run", cancelled["revision"])
    )
    assert retried["partitions"][0]["manifest"]["job_id"] != old_job
    services.dispatch_analysis_run(pid, first["run_id"])
    assert services.get_analysis_run(pid, first["run_id"]) == retried
    repeat = services.transport_analysis_command(
        pid, first["run_id"], {"operation": "repeat"}, CommandContext(str(owner.id), "repeat-run", retried["revision"])
    )
    assert repeat["run_id"] != retried["run_id"]
    manifest = AnalysisPartitionJobManifest.from_payload(repeat["partitions"][0]["manifest"])
    with pytest.raises(ValueError, match="different immutable"):
        queue.submit_analysis(
            replace(manifest, run_fingerprint="f" * 64), actor_id=str(owner.id), resources=repeat["resources"]
        )


def test_server_preserves_actual_runtime_refusal_without_importing_measurements(database, tmp_path):
    from karakal.runtime_identity import worker_runtime_identity
    from kraken_admin.infrastructure.composition import open_project_services

    services, _owner, pid, _queue, app, token, payload = setup_analysis(database)
    measured = worker_runtime_identity()
    payload["runtime"] = replace(measured, numpy_version="incompatible-fixture").to_payload()
    client = TestClient(app)
    created = client.post(f"/api/v1/projects/{pid}/analysis-runs", json=payload,
        headers={"Authorization": "Bearer owner", "Idempotency-Key": "refuse-incompatible-worker"})
    assert created.status_code == 201, created.text
    run_id = created.json()["run_id"]
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="off"))
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    worker = ServerAgentWorker(server_url=f"http://127.0.0.1:{sock.getsockname()[1]}", token=token,
        data_dir=tmp_path / "worker", registry=PluginRegistry({"karakal.analyze.v1": PluginProcessSpec(
            "karakal.analyze.v1", (sys.executable, "-m", "karakal.worker"))}))
    thread = threading.Thread(target=lambda: server.run(sockets=[sock]), daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 5
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.01)
        assert server.started and worker.run_once()
        refused = services.get_analysis_run(pid, run_id)
        assert refused["state"] == "failed" and refused["completed_frames"] == 0
        assert refused["worker_runtime"] is None  # No accepted measurements or pinned runtime.
        assert refused["partitions"][0]["result"]["worker_runtime"] == measured.to_payload()
        assert "pinned requested runtime" in refused["partitions"][0]["error"]
        assert services.analysis_frame_results(pid, run_id, "iou") == []
        reopened = open_project_services(str(services.engine.url), services.uow_factory.blobs.root, engine=services.engine)
        assert reopened.rebuild_analysis_run(pid, run_id) == refused
        repeated = services.transport_analysis_command(pid, run_id, {"operation": "repeat"},
            CommandContext(str(_owner.id), "repeat-same-runtime-constraint", refused["revision"]))
        assert repeated["partitions"][0]["manifest"]["expected_runtime"] == payload["runtime"]
    finally:
        worker.stop()
        server.should_exit = True
        thread.join(5)
        sock.close()
