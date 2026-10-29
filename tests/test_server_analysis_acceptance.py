"""Remote analysis authorization and Qt controls against a real PostgreSQL backend."""

from fastapi.testclient import TestClient
from PyQt6.QtWidgets import QVBoxLayout, QWidget

from kraken_admin.application.roles import set_roles, snapshot
from kraken_hub.remote_analysis import RemoteAnalysisController
from kraken_hub.remote_client_parts.analysis_runs import ServerAnalysisMixin
from kraken_manager.domain.common import ProjectId
from kraken_manager.domain.identity import Principal, ProjectRole
from kraken_manager.presentation.qt.analysis_ui import AnalysisRunsPanel
from kraken_server.app import SessionPrincipal, create_app
from tests.test_admin_project_workflows import project
from tests.test_server_analysis_workflow import setup_analysis

pytest_plugins = ["tests.test_admin_project_workflows"]


def test_analysis_routes_enforce_membership_write_role_scope_and_revision(database):
    services, owner, pid, _queue, _app, _token, payload = setup_analysis(database)
    accounts = database[1]
    outsider = Principal.local(subject="outsider", display_name="Outsider")
    viewer = Principal.local(subject="viewer", display_name="Viewer")
    services.identities.save(outsider)
    services.identities.save(viewer)
    set_roles(
        services, accounts, project_id=pid, principal=viewer,
        desired=frozenset({ProjectRole.VIEWER}), expected_revision=0,
        expected_snapshot=snapshot(services.identities.assignments_for(ProjectId(pid))),
    )
    identities = {"owner": owner, "outsider": outsider, "viewer": viewer}
    app = create_app(
        services=services,
        session_resolver=lambda token: (
            SessionPrincipal(str(identities[token].id), "local", token) if token in identities else None
        ),
    )
    client = TestClient(app)
    base = f"/api/v1/projects/{pid}/analysis-runs"
    owner_headers = {"Authorization": "Bearer owner", "Idempotency-Key": "create-owned"}
    created = client.post(base, headers=owner_headers, json=payload)
    assert created.status_code == 201, created.text
    run = created.json()
    run_path = f"{base}/{run['run_id']}"
    for suffix in ("", f"/{run['run_id']}", f"/{run['run_id']}/frames?metric_key=iou",
                   f"/{run['run_id']}/frames/{run['manifest']['frame_ids'][0]}/map",
                   f"/{run['run_id']}/compare/{run['run_id']}?metric_key=iou"):
        assert client.get(base + suffix).status_code == 401
        assert client.get(base + suffix, headers={"Authorization": "Bearer outsider"}).status_code == 403
    assert client.get(run_path, headers={"Authorization": "Bearer viewer"}).status_code == 200
    for token in ("outsider", "viewer"):
        headers = {"Authorization": f"Bearer {token}", "Idempotency-Key": f"deny-{token}"}
        response = client.post(base, headers=headers, json=payload)
        assert response.status_code == 403, response.text
        response = client.post(f"{run_path}/frames/{run['manifest']['frame_ids'][0]}/map",
                               headers=dict(headers, **{"If-Match": str(run["revision"])}))
        assert response.status_code == 403, response.text
        response = client.post(run_path + "/commands", headers=dict(headers, **{"If-Match": str(run["revision"])}),
                               json={"operation": "cancel"})
        assert response.status_code == 403, response.text
    response = client.post(run_path + "/commands", headers=dict(owner_headers, **{"If-Match": "0"}),
                           json={"operation": "cancel"})
    assert response.status_code == 409, response.text
    other = project(services, owner, "Other analysis scope")["project_id"]
    response = client.get(f"/api/v1/projects/{other}/analysis-runs/{run['run_id']}", headers=owner_headers)
    assert response.status_code == 404, response.text
    assert services.get_analysis_run(pid, run["run_id"]) == run
    assert len(services.list_analysis_runs(pid)) == 1


def test_remote_qt_controls_create_and_cancel_through_actual_routes(database, qtbot):
    services, _owner, pid, _queue, app, _token, payload = setup_analysis(database)
    client = TestClient(app)

    class Remote(ServerAnalysisMixin):
        def _call(self, method, path, *, payload=None, idempotency_key=None, if_match=None):
            headers = {"Authorization": "Bearer owner"}
            if idempotency_key:
                headers["Idempotency-Key"] = idempotency_key
            if if_match is not None:
                headers["If-Match"] = str(if_match)
            result = client.request(method, path, headers=headers, json=payload)
            result.raise_for_status()
            return result.json()

    widget = QWidget()
    panel = widget.analysis_runs_panel = AnalysisRunsPanel(widget)
    QVBoxLayout(widget).addWidget(panel)
    qtbot.addWidget(widget)
    widget.show()
    controller = RemoteAnalysisController(widget, Remote(), pid, language="English")
    qtbot.waitUntil(lambda: controller._worker is None)
    controller.start(payload, payload["layer_id"], payload["coordinates"])
    qtbot.waitUntil(lambda: panel.run_table.rowCount() == 1, timeout=10000)
    qtbot.waitUntil(lambda: controller._worker is None)
    run_id = panel.selected_run_id()
    assert services.get_analysis_run(pid, run_id)["state"] == "running"
    controller.command(run_id, "cancel")
    qtbot.waitUntil(lambda: panel.run_table.item(0, 0).text() == "cancelled", timeout=10000)
    qtbot.waitUntil(lambda: controller._worker is None)
    assert services.get_analysis_run(pid, run_id)["state"] == "cancelled"
    assert len(services.list_analysis_runs(pid)) == 1
    assert panel.result_table.rowCount() == 0
    controller.render_map(run_id, "")
    assert controller.status.text() == "Select a run and frame"
