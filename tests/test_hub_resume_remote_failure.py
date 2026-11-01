"""A failed jobs HTTP request must not escape Hub's startup Qt callback."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading

from PyQt6.QtCore import QTimer
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QMainWindow

from kraken_hub.infrastructure.remote.client import RemoteAuth, RemoteServerProjectService
from kraken_hub.presentation.qt.manager_app import DesktopController
from kraken_manager.domain.identity import Principal


def test_resume_survives_jobs_http_failure_and_can_retry(qtbot, tmp_path):
    project_id = "11111111-1111-4111-8111-111111111111"

    class Endpoint(BaseHTTPRequestHandler):
        unavailable = True
        successful_jobs_reads = 0

        def log_message(self, *_args):
            pass

        def do_GET(self):
            if self.path.startswith("/api/v1/projects?"):
                status, body = 200, {"items": [{"project_id": project_id, "name": "Remote", "width": 1, "height": 1}]}
            elif self.path == f"/api/v1/projects/{project_id}/jobs":
                status, body = (500, {"detail": "Agent queue unavailable"}) if self.unavailable else (200, {"items": []})
                if status == 200:
                    type(self).successful_jobs_reads += 1
            else:
                status, body = 404, {"detail": "Unknown test endpoint"}
            payload = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Endpoint)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    controller = object.__new__(DesktopController)
    controller.shell = QMainWindow()
    qtbot.addWidget(controller.shell)
    controller.service = RemoteServerProjectService(
        f"http://127.0.0.1:{server.server_port}",
        auth=RemoteAuth(access_token="test-token", principal=Principal.local(subject="owner", display_name="Owner")),
        data_dir=tmp_path,
    )
    try:
        controller._resume_agent_jobs()
        assert "Agent queue unavailable" in controller.shell.statusBar().currentMessage()
        QTimer.singleShot(0, controller._resume_agent_jobs)
        QTest.qWait(30)
        Endpoint.unavailable = False
        controller.shell.statusBar().clearMessage()
        controller._resume_agent_jobs()
        assert Endpoint.successful_jobs_reads == 1
        assert not controller.shell.statusBar().currentMessage()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)
