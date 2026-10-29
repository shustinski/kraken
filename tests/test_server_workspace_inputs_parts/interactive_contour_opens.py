"""test_server_workspace_inputs: test interactive contour opens."""
from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
import pytest
from kraken_hub.manager_app import DesktopController
from kraken_hub.infrastructure.remote.client import RemoteAuth, RemoteServerProjectService
from kraken_server.app import SessionPrincipal, create_app
from kraken_server.services import CommandContext

from test_server_workspace_inputs_parts.desktop_job import _desktop_job

@pytest.mark.parametrize("transfer_mode", ["proxy", "gateway"])
@pytest.mark.parametrize("capability", ["frames.vectorize.v1", "frames.dataset.prepare.v1"])
def test_interactive_contour_opens_with_images_and_returns_results(tmp_path, monkeypatch, capability, transfer_mode):
    import os
    import hashlib
    import socket
    import sys
    import threading
    import time
    import uvicorn
    from kraken_agent.runner import PluginProcessSpec, PluginRegistry
    from kraken_agent.server_worker import ServerAgentWorker
    from kraken_manager.infrastructure.filesystem._codec import decode_model
    from kraken_manager.domain.workflows import PluginJob

    root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv(
        "PYTHONPATH",
        os.pathsep.join((str(root / "src"), str(root / "plugins/contour/src"), str(root / "plugins/neuralimage/src"))),
    )

    server, local, session, project, layer, target, gateway, payload = _desktop_job(tmp_path, capability)
    downloads = []
    input_tickets = []
    original_request = ServerAgentWorker._request
    contour_job_id = str(payload["id"])
    attempt_fingerprints = set()

    def request_with_ticket_check(worker, method, path, payload=None, **options):
        response = original_request(worker, method, path, payload, **options)
        if path.startswith(f"/api/v1/agent/jobs/{contour_job_id}/"):
            attempt_fingerprints.add(hashlib.sha256(worker.headers["X-Kraken-Lease-Claim"].encode()).hexdigest())
        if path.endswith("/download-ticket"):
            version_id = path.split("/")[-2]
            metadata = server.get_artifact_version(str(project.id), version_id)
            assert response["media_type"] == metadata["media_type"]
            input_tickets.append(response)
        return response

    monkeypatch.setattr(ServerAgentWorker, "_request", request_with_ticket_check)

    def transfer_ticket(operation, _digest, _size, *, context):
        if operation != "download":
            return {"mode": "proxy"}
        return {"mode": "gateway", "url": f"{base_url}/test-blobs/{context['version']}", "token": "test-transfer"}

    blob_transfer = SimpleNamespace(ticket=transfer_ticket) if transfer_mode == "gateway" else None
    app = create_app(
        services=server,
        session_resolver=lambda token: (
            SessionPrincipal(str(session.principal.id), "local", token) if token == "owner" else None
        ),
        agent_gateway=gateway,
        blob_gateway=blob_transfer,
    )
    from fastapi.responses import StreamingResponse

    @app.get("/test-blobs/{version_id}")
    def raw_blob(version_id: str):
        downloads.append(version_id)
        return StreamingResponse(
            server.iter_artifact_bytes(str(project.id), version_id), media_type="application/octet-stream"
        )

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    base_url = f"http://127.0.0.1:{sock.getsockname()[1]}"
    http_server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="off"))
    thread = threading.Thread(target=lambda: http_server.run(sockets=[sock]), daemon=True)
    thread.start()
    deadline = time.monotonic() + 5
    while not http_server.started and time.monotonic() < deadline:
        time.sleep(0.01)
    assert http_server.started
    script = tmp_path / "contour_interactive.py"
    script.write_text(
        """
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import faulthandler
faulthandler.enable()
if os.name == "nt":
    import ctypes
    ctypes.windll.kernel32.SetErrorMode(3)
from pathlib import Path
from zipfile import ZipFile
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QMessageBox
from contour.kraken_bridge import prepare_contour_launch
from contour.application import bootstrap
from contour.application.cli import main
session, arguments = prepare_contour_launch([])
assert len(session.input_paths) == 2
original_build = bootstrap.build_application
def build(argv):
    global app, window
    app, window = original_build([*argv, "--no-qss", "--log-file", str(session.staging_root / "contour.log")])
    QTimer.singleShot(1000, publish)
    return app, window
bootstrap.build_application = build
def publish():
    try:
        assert window.isVisible()
        assert window._widget._image_list_model.rowCount() == 2
        window.grab().save(str(session.staging_root / "contour-window.png"))
        if session.manifest.operation == "frames.vectorize.v1":
            for image in session.input_paths:
                (session.output_directory / (image.stem + ".cif")).write_text("L M1; P 0 0 10 0 10 10 0 10; E;\\n")
        else:
            import shutil
            (session.output_directory / "images").mkdir(exist_ok=True)
            (session.output_directory / "cif").mkdir(exist_ok=True)
            for image in session.input_paths:
                shutil.copyfile(image, session.output_directory / "images" / image.name)
                (session.output_directory / "cif" / (image.stem + ".cif")).write_text("L M1; E;")
        QMessageBox.information = lambda *args: QMessageBox.StandardButton.Ok
        window._kraken_return_action.trigger()
    finally:
        app.quit()
main([])
""",
        encoding="utf-8",
    )
    registry = PluginRegistry(
        {capability: PluginProcessSpec(capability, (sys.executable, str(script)), interactive=True)}
    )
    remote = RemoteServerProjectService(
        base_url, auth=RemoteAuth("owner", session.principal), data_dir=tmp_path / "desktop"
    )
    try:
        result = remote.run_interactive_job(project.id, payload["id"], registry=registry)
        assert result["imported"]["job"]["state"] == "succeeded"
        assert len(downloads) == (2 if transfer_mode == "gateway" else 0)
        assert len(input_tickets) == 2
        assert result["imported"]["requires_partial_confirmation"] is False
        job = decode_model(PluginJob, payload)
        cells = local.frame_cells(project.id, layer.id, target.id)
        assert len(cells) == 2
        versions = result["imported"]["versions"]
        assert len(versions) == (2 if capability == "frames.vectorize.v1" else 1)
        assert all(local.read_project_blob(project.id, version["sha256"]) for version in versions)
        if capability == "frames.dataset.prepare.v1":
            assert all(
                local.get_artifact_version(project.id, cell.artifact_version_id).media_type == "image/jpeg"
                for cell in cells
            )
            series = local._projection(project.id).get_artifact_series(versions[0]["series_id"])
            assert series.scope.value == "layer_attachment"
            assert len(versions[0]["input_version_ids"]) == 2
            training = server.submit_plugin_job(
                str(project.id),
                {
                    "layer_id": str(layer.id),
                    "source_representation_id": str(target.id),
                    "target_representation_id": str(target.id),
                    "coordinates": [[1, 1], [1, 2]],
                    "capability": "dataset.model.train.v1",
                    "parameters": {"execution_host": "desktop", "dataset_version_id": versions[0]["id"]},
                },
                CommandContext(str(session.principal.id), "train", None),
            )
            train_script = tmp_path / "training.py"
            train_script.write_text(
                """
import os
from neuralimage.kraken_bridge import load_session_from_values
session = load_session_from_values(job_manifest=None, result_manifest=None, staging_root=None)
images, labels = session.prepare_dataset()
assert len(list(images.glob('*.jpg'))) == 2
assert len(list(labels.glob('*.cif'))) == 2
(session.output_directory / 'weights.pth').write_bytes(b'test-model')
session.publish_model()
""",
                encoding="utf-8",
            )
            train_registry = PluginRegistry(
                {
                    "dataset.model.train.v1": PluginProcessSpec(
                        "dataset.model.train.v1", (sys.executable, str(train_script)), interactive=True
                    )
                }
            )
            trained = remote.run_interactive_job(project.id, training["id"], registry=train_registry)
            assert trained["imported"]["job"]["state"] == "succeeded"
            assert len(trained["imported"]["versions"]) == 1
            controller = object.__new__(DesktopController)
            controller.service = local
            snapshot = controller._pipeline_snapshot(project.id, layer.id)
            output_nodes = [node for lane in snapshot.lanes for node in lane.nodes if node.kind in {"dataset", "model"}]
            assert len(output_nodes) == 2
            assert all(node.details["artifact_version_ids"] for node in output_nodes)

            for model in trained["imported"]["versions"]:
                model_series = local._projection(project.id).get_artifact_series(model["series_id"])
                assert model_series.scope.value == "layer_attachment"
            assert all(
                local.get_artifact_version(project.id, cell.artifact_version_id).media_type == "image/jpeg"
                for cell in local.frame_cells(project.id, layer.id, target.id)
            )

        assert len(attempt_fingerprints) == 1
        assert (
            remote.data_dir / "desktop-jobs" / str(job.id) / "staging" / next(iter(attempt_fingerprints))
            / str(job.id) / "contour-window.png"
        ).is_file()
    finally:
        http_server.should_exit = True
        thread.join(timeout=5)
        sock.close()
