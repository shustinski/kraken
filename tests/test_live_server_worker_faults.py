"""Fault acceptance only on a quality project created by the Contour block."""
from dataclasses import replace
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from uuid import uuid4

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QMessageBox, QPushButton, QTableWidget
import pytest

from kraken_hub.application.remote_error import RemoteServerError
from kraken_hub.presentation.qt.parts.my_work_panel import _my_work_panel
from kraken_manager.domain.project import RepresentationKind
from tests.test_live_server_worker_acceptance import ObservedWorker, _open_project, _upload_frame

pytest_plugins = ["tests.test_live_server_worker_acceptance"]


def _area(env, *, source_id=None, parameters=None):
    state = json.loads((Path(env.config["results_directory"]) / "receipt.json").read_text(encoding="utf-8"))
    assert state["status"] == "passed"
    remote = env.remote
    project = remote.get_project(state["project_id"])
    assert project.name.startswith("quality-worker-") and str(project.id) != env.config["protected_project_id"]
    layer = next(item for item in remote.list_layers(project.id) if str(item.id) == state["layer_id"])
    source_id = source_id or state["image_representation_id"]
    target = remote.create_representation(principal=remote.auth.principal, project=project, layer=layer,
        name=f"Fault output {uuid4().hex[:8]}", kind=RepresentationKind.VECTOR,
        source_image_representation_id=source_id, idempotency_key=str(uuid4()))
    key = str(uuid4())
    arguments = dict(principal=remote.auth.principal, gateway=None, project_id=project.id, layer_id=layer.id,
        source_representation_id=source_id, target_representation_id=target.id,
        coordinates=[(1, 1)], capability="frames.vectorize.v1",
        parameters={"input_transport": "server", "execution_mode": "headless", "execution_host": "server",
                    **(parameters or {})}, idempotency_key=key)
    job = remote.submit_plugin_job(**arguments)
    assert remote.submit_plugin_job(**arguments).id == job.id
    env.worker.allowed_jobs.add(str(job.id))
    env.receipt.update(project_id=str(project.id), job_id=str(job.id), target_id=str(target.id))
    return project, layer, target, job


def _job(env, identifier):
    return next(j for j in env.remote.plugin_jobs() if j.id == identifier)


def _work_ui(env, qtbot, project):
    shell, controller, dual = _open_project(qtbot, env.remote, project, env.output, "fault")
    session = type("Session", (), {"principal": env.remote.auth.principal})()
    panel = _my_work_panel(dual, session, controller)
    shell.page("my_work").set_content(panel)
    item = shell._navigation_items["my_work"]
    QTest.mouseClick(shell.navigation_list.viewport(), Qt.MouseButton.LeftButton,
        pos=shell.navigation_list.visualItemRect(item).center())
    qtbot.waitUntil(lambda: panel.isVisible())
    table = panel.findChild(QTableWidget)
    return shell, controller, dual, table


def _close(shell, controller, dual, qtbot):
    qtbot.waitUntil(lambda: not controller._background_workers, timeout=15000)
    controller._agent_timer.stop()
    controller._workspace._analysis_timer.stop()
    shell.close_guard = None
    shell.close()
    dual.stop_sync()


def test_worker_unavailable_job_is_durable_and_cancelled_by_hub(live_worker, qtbot):
    env = live_worker
    project, layer, target, job = _area(env)
    assert _job(env, job.id).state.value == "queued"
    assert not env.remote.frame_cells(project.id, layer.id, target.id)
    # A network failure before lease leaves the durable server intent queued.
    offline = ObservedWorker(server_url="http://127.0.0.1:1", token=env.worker.token,
        data_dir=env.output / "offline", registry=env.worker.registry)
    with pytest.raises(URLError):
        offline.run_once(job_id=str(job.id))
    offline.stop()
    assert _job(env, job.id).state.value == "queued"
    unauthorized = ObservedWorker(server_url=env.config["server_url"], token="invalid-acceptance-token",
        data_dir=env.output / "unauthorized", registry=env.worker.registry)
    with pytest.raises(HTTPError) as rejected:
        unauthorized.run_once(job_id=str(job.id))
    assert rejected.value.code == 401
    unauthorized.stop()
    assert _job(env, job.id).state.value == "queued"
    with pytest.raises(RemoteServerError) as stale:
        env.remote.cancel_plugin_job(principal=env.remote.auth.principal, gateway=None,
            job=replace(job, revision=job.revision + 99), idempotency_key=str(uuid4()))
    assert stale.value.status == 409
    shell, controller, dual, table = _work_ui(env, qtbot, project)
    timer = QTimer()
    def confirm():
        dialog = QApplication.activeModalWidget()
        if isinstance(dialog, QMessageBox):
            button = dialog.button(QMessageBox.StandardButton.Yes)
            if button is not None:
                QTest.mouseClick(button, Qt.MouseButton.LeftButton)
    timer.timeout.connect(confirm)
    timer.start(20)
    try:
        row = next(i for i in range(table.rowCount()) if table.item(i, 1).text() == project.name
                   and table.item(i, 4).text() == "В очереди")
        button = next(b for b in table.cellWidget(row, 7).findChildren(QPushButton) if b.text() == "Отменить")
        QTest.mouseClick(button, Qt.MouseButton.LeftButton)
        qtbot.waitUntil(lambda: _job(env, job.id).state.value == "cancelled", timeout=15000)
        assert env.worker.run_once(job_id=str(job.id)) is False
        assert not env.remote.frame_cells(project.id, layer.id, target.id)
    finally:
        timer.stop()
        _close(shell, controller, dual, qtbot)
    env.receipt["checks"] = ["worker unavailable:job queued and no output", "connection refusal preserves job",
        "CAS409 without mutation", "real Hub cancel button", "cancelled target cannot be leased"]
    env.receipt["status"] = "passed"


@pytest.mark.parametrize("kind", ["bad-parameters", "bad-image"])
def test_actual_plugin_failures_are_saved_and_visible_in_hub(live_worker, qtbot, tmp_path, kind):
    env = live_worker
    source_id = None
    if kind == "bad-image":
        state = json.loads((Path(env.config["results_directory"]) / "receipt.json").read_text(encoding="utf-8"))
        project = env.remote.get_project(state["project_id"])
        assert project.name.startswith("quality-worker-")
        layer = env.remote.list_layers(project.id)[0]
        source = env.remote.create_representation(principal=env.remote.auth.principal, project=project, layer=layer,
            name=f"Invalid input {uuid4().hex[:8]}", kind=RepresentationKind.IMAGE, idempotency_key=str(uuid4()))
        path = tmp_path / "invalid.png"
        path.write_bytes(b"intentionally invalid PNG in dedicated quality project")
        _upload_frame(env.remote, project.id, layer.id, source.id, path, 1, 1)
        source_id = source.id
    project, layer, target, job = _area(env, source_id=source_id,
        parameters={"pipeline_config": []} if kind == "bad-parameters" else {})
    assert env.worker.run_once(job_id=str(job.id))
    failed = _job(env, job.id)
    assert failed.state.value == "failed" and failed.error
    assert not env.remote.frame_cells(project.id, layer.id, target.id)
    with pytest.raises(RemoteServerError) as invalid_retry:
        env.remote.retry_plugin_job(principal=env.remote.auth.principal, gateway=None,
            job=failed, idempotency_key=str(uuid4()))
    assert invalid_retry.value.status == 409
    assert _job(env, job.id) == failed
    shell, controller, dual, table = _work_ui(env, qtbot, project)
    try:
        errors = [table.cellWidget(i, 7).toolTip() for i in range(table.rowCount())
                  if table.item(i, 1).text() == project.name and table.item(i, 4).text() == "Ошибка"]
        assert any(failed.error in value for value in errors)
        assert shell.grab().save(str(env.output / "hub-failure.png"))
    finally:
        _close(shell, controller, dual, qtbot)
    env.receipt.update(error=failed.error, checks=["actual Contour subprocess failure", "durable failed state and diagnostic",
        "no output version", "real Hub My work status and error tooltip"], status="passed")


def test_lease_expiry_reconnect_and_duplicate_completion_do_not_duplicate_versions(live_worker, qtbot):
    env = live_worker
    project, layer, target, job = _area(env)
    lease = env.worker._request("POST", f"/api/v1/agent/jobs/{job.id}/lease",
        {"capabilities": ["frames.vectorize.v1"], "lease_seconds": 15})["job"]
    old_claim = lease["lease_claim"]
    qtbot.wait(16000)
    env.worker.headers["X-Kraken-Lease-Claim"] = old_claim
    with pytest.raises(HTTPError) as expired:
        env.worker._request("POST", f"/api/v1/agent/jobs/{job.id}/heartbeat", {})
    assert expired.value.code == 403
    fresh = ObservedWorker(server_url=env.config["server_url"], token=env.worker.token,
        data_dir=env.output / "reconnected", registry=env.worker.registry)
    fresh.transitions, fresh.allowed_jobs = env.worker.transitions, env.worker.allowed_jobs
    try:
        assert fresh.run_once(job_id=str(job.id))
        assert fresh.last_claim != old_claim
        assert _job(env, job.id).state.value == "succeeded"
        cells = env.remote.frame_cells(project.id, layer.id, target.id)
        assert len(cells) == 1 and cells[0].sha256
        history = [str(e.event_id) for e in env.remote.history(project.id)]
        fresh.headers["X-Kraken-Lease-Claim"] = fresh.last_claim
        for suffix, body in [("/complete", {}), ("/publications", fresh.last_publication)]:
            with pytest.raises(HTTPError) as duplicate:
                fresh._request("POST", f"/api/v1/agent/jobs/{job.id}" + suffix, body)
            assert duplicate.value.code == 403
        assert [str(e.event_id) for e in env.remote.history(project.id)] == history
        assert env.remote.frame_cells(project.id, layer.id, target.id) == cells
        assert fresh.run_once(job_id=str(job.id)) is False
    finally:
        fresh.stop()
    env.receipt.update(checks=["15s lease actually expires", "expired owner fenced403", "fresh worker reclaims and processes",
        "duplicate completion/publication fenced403", "no duplicate version/history", "completed job not reissued"], status="passed")


def test_lost_completion_response_preserves_committed_result_on_reconnect(live_worker):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import socket
    import threading
    from urllib.request import Request, urlopen

    env = live_worker
    project, layer, target, job = _area(env)
    completed_path = f"/api/v1/agent/jobs/{job.id}/complete"
    dropped = []

    class DisconnectAfterCommit(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def forward(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0))) if self.command == "POST" else None
            request = Request(env.config["server_url"] + self.path, data=body, method=self.command,
                headers={key: value for key, value in self.headers.items() if key.lower() not in {"host", "connection"}})
            try:
                with urlopen(request, timeout=20) as response:
                    status, content, content_type = response.status, response.read(), response.headers.get("Content-Type")
            except HTTPError as error:
                status, content, content_type = error.code, error.read(), error.headers.get("Content-Type")
            if self.path == completed_path and status == 200:
                dropped.append(True)
                self.connection.shutdown(socket.SHUT_RDWR)
                self.connection.close()
                return
            self.send_response(status)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Content-Type", content_type or "application/json")
            self.end_headers()
            self.wfile.write(content)

        do_GET = forward
        do_POST = forward

    proxy = ThreadingHTTPServer(("127.0.0.1", 0), DisconnectAfterCommit)
    thread = threading.Thread(target=proxy.serve_forever, daemon=True)
    thread.start()
    worker = ObservedWorker(server_url=f"http://127.0.0.1:{proxy.server_port}", token=env.worker.token,
        data_dir=env.output / "through-disconnect", registry=env.worker.registry)
    worker.transitions, worker.allowed_jobs = env.worker.transitions, env.worker.allowed_jobs
    try:
        assert worker.run_once(job_id=str(job.id))
        assert dropped == [True]
        assert _job(env, job.id).state.value == "succeeded"
        cells = env.remote.frame_cells(project.id, layer.id, target.id)
        assert len(cells) == 1 and cells[0].sha256
        history = [str(e.event_id) for e in env.remote.history(project.id)]
        assert env.worker.run_once(job_id=str(job.id)) is False
        assert env.remote.frame_cells(project.id, layer.id, target.id) == cells
        assert [str(e.event_id) for e in env.remote.history(project.id)] == history
    finally:
        worker.stop()
        proxy.shutdown()
        proxy.server_close()
        thread.join(5)
    env.receipt.update(checks=["actual TCP disconnect after server complete200", "committed result stays succeeded",
        "fresh worker does not recompute or duplicate version/history"], status="passed")
