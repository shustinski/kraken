from dataclasses import asdict
from io import BytesIO
import json
import socket
import sys
import threading
from types import SimpleNamespace
from uuid import uuid4

from fastapi.testclient import TestClient
from PIL import Image
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFileDialog, QMainWindow
import pytest
import uvicorn

from kraken_agent.runner import PluginProcessSpec, PluginRegistry
from kraken_agent.server_worker import ServerAgentWorker
from kraken_hub.computed_stitch_dialog import ComputedStitchDialog, _STITCH_CALLS
from kraken_hub.remote_client import RemoteAuth, RemoteHttpClient, RemoteServerError, RemoteServerProjectService
from kraken_manager.domain.identity import Principal, ProjectRole, ProjectRoleAssignment
from kraken_manager.domain.multilayer import MultilayerModel
from kraken_manager.infrastructure.mosaic import STITCH_OPERATION
from kraken_server.agent_auth import PostgresAgentTokenStore
from kraken_server.app import SessionPrincipal, create_app
from tests.test_server_stitch_workflow import setup_stitch

pytest_plugins = ["tests.test_admin_project_workflows"]


def observations(stitch):
    tiles = []
    for tile in stitch.tiles:
        calibration = tile.calibration
        value = {"frame_id": tile.frame_id, "artifact_version_id": calibration.artifact.version_id,
            "width": tile.width, "height": tile.height, "pixel_size_m": calibration.pixel_size_m,
            "origin_m": calibration.origin_m, "y_axis": calibration.y_axis, "catalog_xy": tile.catalog_xy}
        if tile.provenance == "declared_transform":
            value.update(declared_affine=tile.affine, note=tile.note)
        elif tile.provenance == "control_points":
            value["control_points"] = [asdict(p) for p in tile.control_points]
        tiles.append(value)
    return {"layer_id": stitch.layer_id, "reference_frame_id": stitch.reference_frame_id, "tiles": tiles, "max_error_m": 1e-12}


def test_actual_remote_qt_import_run_native_agent_and_bounded_block_view(database, tmp_path, qtbot, monkeypatch):
    service, owner, pid, queue, app, _, token, _ = setup_stitch(database)
    stitch = MultilayerModel.from_dict(service.get_multilayer_model(pid)).stitches[-1]
    document = tmp_path / "observations.json"
    document.write_text(json.dumps(observations(stitch)), encoding="utf-8")
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="off"))
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    url = f"http://127.0.0.1:{sock.getsockname()[1]}"
    thread = threading.Thread(target=lambda: server.run(sockets=[sock]), daemon=True)
    thread.start()
    calls = []

    class RecordingHttp(RemoteHttpClient):
        def request(self, method, path, **kwargs):
            calls.append((threading.get_ident(), method, path))
            return super().request(method, path, **kwargs)

    remote = RemoteServerProjectService(url, auth=RemoteAuth(access_token="owner", principal=owner),
                                        data_dir=tmp_path / "remote-client", http=RecordingHttp(url))
    mutate = remote.mutate_multilayer_model
    mutation_keys = {}

    def lose_committed_reply(**kwargs):
        result = mutate(**kwargs)
        operation = kwargs["operation"]
        keys = mutation_keys.setdefault(operation, [])
        keys.append(kwargs["idempotency_key"])
        if len(keys) == 1:
            raise RemoteServerError("controlled lost reply after server commit", status=503)
        return result

    monkeypatch.setattr(remote, "mutate_multilayer_model", lose_committed_reply)
    monkeypatch.setattr(remote, "managed_artifact_path", lambda *_: pytest.fail("viewer must not download the entire mosaic"))
    monkeypatch.setattr(remote._http, "download", lambda *_args, **_kwargs: pytest.fail("no full artifact download"))
    workers = set()
    shell = QMainWindow()
    qtbot.addWidget(shell)
    local_starts = []
    controller = SimpleNamespace(shell=shell, service=remote, session=SimpleNamespace(principal=owner), _project_id=pid,
        agent_runtime=SimpleNamespace(ensure_started=lambda: local_starts.append(True)), _background_workers=workers)
    worker = ServerAgentWorker(server_url=url, token=token, data_dir=tmp_path / "server-worker",
        registry=PluginRegistry({STITCH_OPERATION: PluginProcessSpec(STITCH_OPERATION,
            (sys.executable, "-m", "kraken_manager.infrastructure.stitch_worker"))}))
    dialog = None
    try:
        qtbot.waitUntil(lambda: server.started, timeout=5000)
        dialog = ComputedStitchDialog(controller, stitch.layer_id)
        qtbot.addWidget(dialog)
        dialog.timer.setInterval(75)
        dialog.show()
        qtbot.waitUntil(lambda: dialog._worker is None and dialog.snapshot is not None, timeout=10000)
        monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *_: (str(document), "JSON (*.json)"))
        qtbot.mouseClick(dialog.import_button, Qt.MouseButton.LeftButton)
        qtbot.waitUntil(lambda: dialog._worker is None and dialog._pending_import is None
            and len(dialog.snapshot["model"]["stitches"]) == 2, timeout=15000)
        assert dialog.plans.currentData() == (stitch.id, 1)
        imported_plan = MultilayerModel.from_dict(dialog.snapshot).stitches[-1]
        assert service.uow_factory.blobs.read(imported_plan.plan.sha256) == document.read_bytes()
        dialog.mode.setCurrentIndex(dialog.mode.findData("gray"))
        qtbot.mouseClick(dialog.run_button, Qt.MouseButton.LeftButton)
        qtbot.waitUntil(lambda: dialog._worker is None and bool(dialog.snapshot["model"]["stitch_renders"]), timeout=10000)
        job_id = dialog.snapshot["model"]["stitch_renders"][-1]["job_id"]
        assert worker.run_once()
        qtbot.waitUntil(lambda: dialog._worker is None
            and dialog.snapshot["model"]["stitch_renders"][-1]["state"] == "imported", timeout=10000)
        dialog.timer.stop()
        qtbot.mouseClick(dialog.view_button, Qt.MouseButton.LeftButton)
        qtbot.waitUntil(lambda: hasattr(dialog, "_viewer") and dialog._viewer._worker is None
                        and dialog._viewer._metadata is not None, timeout=10000)
        viewer = dialog._viewer
        assert viewer.image.pixmap() is not None and not viewer.image.pixmap().isNull()
        assert viewer._metadata["job_id"] == job_id and viewer._metadata["stitch_revision"] == 1
        assert "plan" not in viewer._metadata
        viewer.coverage.setChecked(True)
        viewer.refresh()
        qtbot.waitUntil(lambda: viewer._worker is None, timeout=10000)
        assert not local_starts
        for operation in ("plan_layer_stitch", "request_layer_stitch"):
            assert len(mutation_keys[operation]) == 2 and len(set(mutation_keys[operation])) == 1
        assert len(service.get_multilayer_model(pid)["model"]["stitch_renders"]) == 1
        assert calls and all(identifier != threading.get_ident() for identifier, _, _ in calls)
        assert any("/blocks/" in path for _, _, path in calls)
        with TestClient(app) as client:
            base = f"/api/v1/projects/{pid}/stitch-renders/{job_id}/blocks"
            assert client.get(base + "/0/0").status_code == 401
            response = client.get(base + "/0/0?coverage=true", headers={"Authorization": "Bearer owner"})
            assert response.status_code == 200, response.text
            import base64
            with Image.open(BytesIO(base64.b64decode(response.json()["image_base64"]))) as image:
                assert image.getpixel((2, 0)) == (0, 220, 255)
            assert len(response.content) < 16 * 1024
            assert client.get(base + "/-1/0", headers={"Authorization": "Bearer owner"}).status_code == 422
            assert client.get(base + "/999/0", headers={"Authorization": "Bearer owner"}).status_code == 422
            assert client.get(base.replace(job_id, str(uuid4())) + "/0/0", headers={"Authorization": "Bearer owner"}).status_code == 404
            # Change only this fixture's managed output to exercise a damaged
            # stored block; the API must refuse it rather than return bad pixels.
            import sqlite3
            result = MultilayerModel.from_dict(service.get_multilayer_model(pid)).stitch_renders[-1]
            version = service.projections.get_artifact_version(result.output.version_id)
            with sqlite3.connect(service.uow_factory.blobs.path_for_read(version.blob)) as db:
                db.execute("UPDATE blocks SET coverage=? WHERE x=0 AND y=0", (b"invalid zlib fixture",))
            assert client.get(base + "/0/0?coverage=true", headers={"Authorization": "Bearer owner"}).status_code == 409
        viewer.close()
        dialog.close()
        qtbot.waitUntil(lambda: not workers and not _STITCH_CALLS, timeout=10000)
    finally:
        if dialog is not None:
            dialog.timer.stop()
        qtbot.waitUntil(lambda: not workers, timeout=10000)
        worker.stop()
        server.should_exit = True
        thread.join(timeout=10)
        sock.close()


def test_authenticated_reader_can_view_but_cannot_request_compute(database):
    service, owner, pid, queue, _, _, _, payload = setup_stitch(database)
    reader = Principal.local(subject="reader", display_name="Read-only account")
    service.identities.save(reader)
    app = create_app(services=service, agent_gateway=queue, agent_token_store=PostgresAgentTokenStore(service.engine),
        session_resolver=lambda token: SessionPrincipal(str(reader.id), "local", token))
    with TestClient(app) as client:
        headers = {"Authorization": "Bearer reader", "Idempotency-Key": "forbidden", "If-Match": "2"}
        assert client.get(f"/api/v1/projects/{pid}/multilayer", headers=headers).status_code == 403
        service.identities.assign(ProjectRoleAssignment.create(project_id=pid, principal_id=reader.id,
            role=ProjectRole.VIEWER, assigned_by=owner.id))
        assert client.get(f"/api/v1/projects/{pid}/multilayer", headers=headers).status_code == 200
        assert client.post(f"/api/v1/projects/{pid}/multilayer/commands", headers=headers, json=payload).status_code == 403
    assert service.get_multilayer_model(pid)["model"]["stitch_renders"] == []
