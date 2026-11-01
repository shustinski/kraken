"""Opt-in acceptance on a real configured Server; writes only new quality projects.

KRAKEN_LIVE_WORKER_CONFIG points to JSON containing server_url, session_secret,
machine_secret, protected_project_id and results_directory. Secrets themselves
must be DPAPI files, never JSON values. No local agent substitutes are used.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from urllib.error import HTTPError
from uuid import uuid4

from PIL import Image, ImageDraw
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
import pytest

from kraken_agent.runner import PluginProcessSpec, PluginRegistry
from kraken_agent.server_worker import ServerAgentWorker
from kraken_core.plugins import PluginInventoryItem, load_plugin_metadata_from_directory
from kraken_hub.composition import EmbeddedProjectService
from kraken_hub.composition.catalog import DualCatalogService
from kraken_hub.infrastructure.remote.client import RemoteAuth, RemoteHttpClient, RemoteServerProjectService
from kraken_hub.infrastructure.remote.parts.principal_server_payload import _principal_from_server_payload
from kraken_hub.presentation.qt.manager_app import DesktopController
from kraken_manager.domain.artifacts import deterministic_frame_series_id
from kraken_manager.domain.project import FrameCoordinate, GridOrientation, LayerType, RepresentationKind
from kraken_manager.presentation.qt import ProjectManagerShell
from kraken_server.configuration import unprotect_secret


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _connect(config, directory):
    token = unprotect_secret(Path(config["session_secret"]))
    http = RemoteHttpClient(config["server_url"])
    principal = _principal_from_server_payload(http.request("GET", "/api/v1/session", token=token))
    return RemoteServerProjectService(config["server_url"],
        auth=RemoteAuth(access_token=token, principal=principal), data_dir=directory, http=http)


def _protected_snapshot(remote, pid):
    return {"revision": remote.get_project(pid).revision,
            "history": _digest([str(e.event_id) for e in remote.history(pid)]),
            "jobs": _digest(remote._call("GET", f"/api/v1/projects/{pid}/jobs"))}


def _upload_frame(remote, project_id, layer_id, representation_id, source, x, y):
    frame_id = FrameCoordinate(x, y).frame_id(project_id)
    series = remote._call("POST", f"/api/v1/projects/{project_id}/artifacts",
        payload={"name": source.stem, "scope": "frame_representation", "layer_id": str(layer_id),
                 "representation_id": str(representation_id), "frame_id": str(frame_id),
                 "series_id": str(deterministic_frame_series_id(representation_id, frame_id))},
        idempotency_key=str(uuid4()))
    metadata = {"filename": source.name, "media_type": "image/png",
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "size_bytes": source.stat().st_size,
                "parameters": {"x": x, "y": y, "mapping": "preflight"}}
    key = str(uuid4())
    revision = remote.artifact_stream_revision(project_id, series["id"])
    base = f"/api/v1/projects/{project_id}/artifacts/{series['id']}/uploads"
    transfer = remote._call("POST", base, payload=metadata, idempotency_key=key, if_match=revision)
    assert transfer["mode"] == "gateway"
    remote._http.upload_url(transfer["url"], token=transfer["token"], source=source)
    saved = remote._call("POST", base + "/complete", payload={**metadata, "upload_token": transfer["token"]},
        idempotency_key=key, if_match=revision)
    return remote.artifact_version(project_id, saved["id"])


class ObservedWorker(ServerAgentWorker):
    """Observe actual network transitions; never replace responses or execution."""
    def _request(self, method, path, payload=None, **kwargs):
        result = super()._request(method, path, payload, **kwargs)
        if path.endswith("/publications") and method == "POST":
            self.last_publication = payload
        entry = {"method": method, "path": path, "status": 200}
        if path.endswith("/lease") and result.get("job"):
            entry["job_id"] = result["job"]["job_id"]
            assert entry["job_id"] in self.allowed_jobs, "Unexpected lease; stop acceptance"
            self.last_claim = result["job"].get("lease_claim")
        self.transitions.append(entry)
        return result


@pytest.fixture
def live_worker(tmp_path, request):
    filename = os.environ.get("KRAKEN_LIVE_WORKER_CONFIG")
    if not filename:
        pytest.skip("Set KRAKEN_LIVE_WORKER_CONFIG for real Server acceptance")
    config = json.loads(Path(filename).read_text(encoding="utf-8-sig"))
    from kraken_core.settings import configure_settings_directory
    configure_settings_directory(tmp_path / "settings")
    output = Path(config["results_directory"])
    if request.node.name != "test_contour_machine_worker_publishes_versions_and_hub_reopens":
        output = output / request.node.name
    output.mkdir(parents=True, exist_ok=True)
    remote = _connect(config, tmp_path / "client")
    protected = _protected_snapshot(remote, config["protected_project_id"])
    # The global lease API has capability scope, not project scope. Existing
    # user work must not be eligible before this controlled one-shot worker.
    assert not [j for j in remote.plugin_jobs() if j.state.value in {"queued", "staging", "running", "recovery_required"}], \
        "Do not poll a shared queue containing other active work"
    worker = ObservedWorker(server_url=config["server_url"],
        token=unprotect_secret(Path(config["machine_secret"])), data_dir=output / "worker",
        registry=PluginRegistry({"frames.vectorize.v1": PluginProcessSpec("frames.vectorize.v1",
            (sys.executable, "-m", "contour.worker"))}))
    worker.transitions = []
    worker.allowed_jobs = set()
    receipt = {"status": "failed", "requirement": "Server→machine worker→Contour→version→Hub→reopen",
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "server_url": config["server_url"], "protected_before": protected, "checks": []}
    try:
        yield SimpleNamespace(config=config, remote=remote, worker=worker, receipt=receipt, output=output)
    finally:
        worker.stop()
        receipt["transitions"] = worker.transitions
        receipt["protected_after"] = _protected_snapshot(remote, config["protected_project_id"])
        receipt["protected_unchanged"] = receipt["protected_after"] == protected
        if not receipt["protected_unchanged"]:
            receipt["status"] = "failed"
        (output / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8", newline="\n")
        assert receipt["protected_unchanged"], "Protected user project or jobs changed"


def _open_project(qtbot, remote, project, output, attempt):
    if Path("C:/Windows/Fonts/arial.ttf").is_file():
        from PyQt6.QtGui import QFont, QFontDatabase
        from PyQt6.QtWidgets import QApplication
        QFontDatabase.addApplicationFont("C:/Windows/Fonts/arial.ttf")
        QApplication.instance().setFont(QFont("Arial", 10))
    local = EmbeddedProjectService(output / f"hub-cache-{attempt}")
    local.identities.save(remote.auth.principal)
    dual = DualCatalogService(local, remote)
    shell = ProjectManagerShell()
    qtbot.addWidget(shell)
    root = Path(__file__).resolve().parents[1]
    plugins = [PluginInventoryItem(load_plugin_metadata_from_directory(root / "plugins" / name))
               for name in ("contour", "neuralimage", "karakal")]
    controller = DesktopController(shell, dual, SimpleNamespace(principal=remote.auth.principal),
        thumbnail_store_uri="memory://", plugin_items=plugins)
    controller._agent_timer.stop()
    shell.resize(1440, 1000)
    shell.show()
    page = controller.catalog_page
    model, view = page.project_model, page.project_list
    index = next(model.index(i, 0) for i in range(model.rowCount())
                 if model.index(i, 0).data() == project.name)
    view.setCurrentIndex(index)
    view.scrollTo(index)
    QTest.qWait(20)
    point = view.visualRect(index).center()
    QTest.mouseClick(view.viewport(), Qt.MouseButton.LeftButton, pos=point)
    QTest.mouseDClick(view.viewport(), Qt.MouseButton.LeftButton, pos=point)
    qtbot.waitUntil(lambda: controller._workspace is not None, timeout=15000)
    assert str(controller._project_id) == str(project.id)
    return shell, controller, dual


def test_contour_machine_worker_publishes_versions_and_hub_reopens(live_worker, qtbot, tmp_path):
    env = live_worker
    remote, worker = env.remote, env.worker
    principal = remote.auth.principal
    with pytest.raises(HTTPError) as denied:
        worker._request("POST", "/api/v1/agent/lease", {"capabilities": ["karakal.analyze.v1"]})
    assert denied.value.code == 403
    env.receipt["checks"].append("machine capability boundary:403")
    project = remote.create_project(principal=principal, name=f"quality-worker-{uuid4().hex[:12]}",
        width=2, height=1, orientation=GridOrientation.Y_DOWN, idempotency_key=str(uuid4()))
    env.receipt["project_id"] = str(project.id)
    layer = remote.create_layer(principal=principal, project=project, name="Controlled metal",
        layer_type=LayerType.METAL, order=0, idempotency_key=str(uuid4()))
    image_rep = remote.create_representation(principal=principal, project=project, layer=layer,
        name="Source", kind=RepresentationKind.IMAGE, active=True, idempotency_key=str(uuid4()))
    layer = remote.list_layers(project.id)[0]
    vector_rep = remote.create_representation(principal=principal, project=project, layer=layer,
        name="Contour result", kind=RepresentationKind.VECTOR, source_image_representation_id=image_rep.id,
        active=True, idempotency_key=str(uuid4()))
    input_ids = set()
    input_by_coordinate = {}
    for x in (1, 2):
        source = tmp_path / f"{x}_1.png"
        image = Image.new("RGB", (64, 64), "black")
        ImageDraw.Draw(image).rectangle((8, 8, 44 + x, 44), fill="white")
        image.save(source)
        version = _upload_frame(remote, project.id, layer.id, image_rep.id, source, x, 1)
        input_ids.add(str(version.id))
        input_by_coordinate[(x, 1)] = str(version.id)
    cells = remote.frame_cells(project.id, layer.id, image_rep.id)
    assert {(c.x, c.y) for c in cells if c.sha256} == {(1, 1), (2, 1)}
    job = remote.submit_plugin_job(principal=principal, gateway=None, project_id=project.id,
        layer_id=layer.id, source_representation_id=image_rep.id, target_representation_id=vector_rep.id,
        coordinates=[(1, 1), (2, 1)], capability="frames.vectorize.v1",
        parameters={"input_transport": "server", "execution_mode": "headless", "execution_host": "server"},
        idempotency_key=str(uuid4()))
    worker.allowed_jobs.add(str(job.id))
    env.receipt.update(job_id=str(job.id), input_version_ids=sorted(input_ids), layer_id=str(layer.id),
        image_representation_id=str(image_rep.id), vector_representation_id=str(vector_rep.id))
    assert job.state.value == "queued"
    assert worker.run_once(job_id=str(job.id)), "Actual machine worker did not receive its job"
    done = next(j for j in remote.plugin_jobs() if j.id == job.id)
    assert done.state.value == "succeeded", (done.state, done.error)
    assert any(t["path"].endswith("/heartbeat") for t in worker.transitions)
    assert any(t["path"].endswith("/complete") for t in worker.transitions)
    from contour.serializers import load_polygons_cif
    results = []
    for cell in remote.frame_cells(project.id, layer.id, vector_rep.id):
        assert cell.sha256
        version = remote.artifact_version(project.id, cell.artifact_version_id)
        path = remote.managed_artifact_path(project.id, version.id)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == version.sha256 == cell.sha256
        assert path.stat().st_size == version.size_bytes > 0
        assert Path(version.filename).suffix.lower() == ".cif"
        assert version.media_type == "application/x-cif"
        assert list(map(str, version.input_version_ids)) == [input_by_coordinate[(cell.x, cell.y)]]
        _, dimensions, polygons = load_polygons_cif(path)
        assert tuple(dimensions) == (64, 64) and polygons
        results.append({"id": str(version.id), "sha256": version.sha256, "bytes": version.size_bytes,
            "media_type": version.media_type, "inputs": list(map(str, version.input_version_ids))})
    assert len(results) == 2
    env.receipt["outputs"] = results
    history = [str(e.event_id) for e in remote.history(project.id)]
    assert len(history) > 5
    for attempt in (1, 2):
        reopened = _connect(env.config, tmp_path / f"new-client-{attempt}")
        assert [str(e.event_id) for e in reopened.history(project.id)] == history
        assert next(j for j in reopened.plugin_jobs() if j.id == job.id).state.value == "succeeded"
        shell, controller, dual = _open_project(qtbot, reopened, project, env.output, attempt)
        try:
            qtbot.waitUntil(lambda: not controller._background_workers, timeout=15000)
            workspace = controller._workspace
            assert workspace.layer_tabs.count() == 1
            assert workspace.vector_representation_combo.findData(str(vector_rep.id)) >= 0
            for output in results:
                path = reopened.managed_artifact_path(project.id, output["id"])
                assert hashlib.sha256(path.read_bytes()).hexdigest() == output["sha256"]
            assert shell.grab().save(str(env.output / f"hub-{attempt}.png"))
        finally:
            controller._agent_timer.stop()
            workspace._analysis_timer.stop()
            shell.close_guard = None
            shell.close()
            dual.stop_sync()
    env.receipt["checks"].extend(["real Contour subprocess and heartbeat/complete", "two published CIF versions with input lineage",
        "two fresh HTTP sessions and real Hub catalog double-click opens", "reread bytes and history preserved"])
    env.receipt["status"] = "passed"
