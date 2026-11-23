"""The Hub polls the agent only while work progresses, and My Work reads only when shown."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QTimer  # noqa: E402
from PyQt6.QtWidgets import QWidget  # noqa: E402

from kraken_hub.presentation.qt.parts.my_work_panel import _my_work_panel  # noqa: E402
from kraken_hub.presentation.qt.parts.poll_agent_jobs import PollAgentJobsMixin  # noqa: E402


def _job(state: str, job_id: str = "job") -> SimpleNamespace:
    return SimpleNamespace(id=job_id, state=SimpleNamespace(value=state), project_id="project")


class _Service:
    def __init__(self, jobs=()):
        self.jobs = tuple(jobs)
        self.sync_calls = 0

    def list_projects(self, *, include_archived):
        return ()

    def synchronize_plugin_jobs(self, *, principal, gateway, projects=None):
        self.sync_calls += 1
        return self.jobs

    def work_orders(self, *, projects=None):
        return ()


class _Controller(PollAgentJobsMixin):
    def __init__(self, service, *, gateway=object()):
        self.service = service
        self.session = SimpleNamespace(principal=None)
        self._agent_gateway = gateway
        self._agent_poll_error = ""
        self._agent_timer = QTimer()
        self._agent_timer.setInterval(2000)
        self.my_work_refresh = None
        self.refreshes = 0

        def refresh():
            self.refreshes += 1

        self.my_work_refresh = refresh


@pytest.mark.parametrize("state", ["succeeded", "failed", "cancelled", "waiting_for_user", "partial"])
def test_poll_stops_when_no_job_progresses_on_its_own(qtbot, state):
    controller = _Controller(_Service([_job(state)]))
    controller._agent_timer.start()

    controller._poll_agent_jobs()

    assert not controller._agent_timer.isActive()
    assert controller.refreshes == 1


@pytest.mark.parametrize("state", ["queued", "staging", "running", "importing"])
def test_poll_keeps_running_while_a_job_progresses(qtbot, state):
    controller = _Controller(_Service([_job(state)]))
    controller._import_agent_job_in_background = lambda job_id, gateway: None
    controller._agent_timer.start()

    controller._poll_agent_jobs()

    assert controller._agent_timer.isActive()


def test_poll_without_a_local_gateway_stops(qtbot):
    controller = _Controller(_Service(), gateway=None)
    controller._agent_timer.start()

    controller._poll_agent_jobs()

    assert not controller._agent_timer.isActive()
    assert controller.refreshes == 1


def test_active_layer_pipeline_keeps_and_restarts_the_followup(qtbot):
    controller = _Controller(_Service())

    controller._on_pipeline_cycle_finished(1)
    assert controller._agent_timer.isActive()

    # A running pipeline keeps the poll alive even with no job in flight yet.
    controller._poll_agent_jobs()
    assert controller._agent_timer.isActive()

    controller._on_pipeline_cycle_finished(0)
    controller._poll_agent_jobs()
    assert not controller._agent_timer.isActive()


def test_request_agent_followup_starts_an_idle_timer(qtbot):
    controller = _Controller(_Service())
    assert not controller._agent_timer.isActive()

    controller._request_agent_followup()

    assert controller._agent_timer.isActive()


def test_live_wake_refreshes_my_work(qtbot):
    controller = _Controller(_Service())
    controller._sync_timer = QTimer()
    controller._pending_wakes = {}
    controller._pending_catalog_refresh = False
    controller._project_workspaces = {}
    controller._project_id = None
    controller.refresh_projects = lambda: None
    wake = SimpleNamespace(
        project_id="project",
        entity_kind="plugin_job",
        entity_id="job",
        stream_id="stream",
        event_type="PluginJobStateChanged",
    )
    controller._pending_wakes[("project", "plugin_job", "job")] = wake

    controller._drain_live_wakes()

    assert controller.refreshes == 1


def test_analysis_wakes_coalesce_and_do_not_refresh_my_work(qtbot):
    from kraken_hub.presentation.qt.remote_change_bus import RemoteChangeBus

    controller = _Controller(_Service())
    controller._pending_wakes = {}
    controller._pending_catalog_refresh = False
    controller._project_workspaces = {}
    controller._project_id = None
    controller._change_bus = RemoteChangeBus()
    received = []
    controller._change_bus.changed.connect(lambda *args: received.append(args))
    controller.refresh_projects = lambda: None
    for index in range(50):
        wake = SimpleNamespace(project_id="project", entity_kind="analysis", entity_id=str(index),
                               stream_id="", event_type="analysis_partition.imported")
        controller._pending_wakes[("project", "analysis", str(index))] = wake

    controller._drain_live_wakes()

    assert controller.refreshes == 0
    assert received == [("project", frozenset({"analysis"}), frozenset(map(str, range(50))))]


def test_wakes_keep_distinct_topics_for_same_entity(qtbot):
    from kraken_hub.application.notifications import ProjectEventWake
    from kraken_hub.presentation.qt.remote_change_bus import RemoteChangeBus

    controller = _Controller(_Service())
    controller._sync_timer = QTimer()
    controller._sync_timer.setSingleShot(True)
    controller._pending_wakes = {}
    controller._pending_catalog_refresh = False
    controller._project_workspaces = {}
    controller._project_id = None
    controller._change_bus = RemoteChangeBus()
    controller.refresh_projects = lambda: None
    received = []
    controller._change_bus.changed.connect(lambda _pid, topics, _ids: received.append(topics))
    controller._on_remote_wake(ProjectEventWake("project", "analysis_partition.imported", "e1",
                                                 entity_kind="job", entity_id="same"))
    controller._on_remote_wake(ProjectEventWake("project", "job.state_changed", "e2",
                                                 entity_kind="job", entity_id="same"))
    controller._sync_timer.stop()
    controller.my_work_refresh = lambda _project_id=None: None
    controller._drain_live_wakes()
    assert received == [frozenset({"analysis", "jobs"})]


def test_reconnect_wakes_every_open_remote_project(qtbot):
    from kraken_hub.presentation.qt.remote_change_bus import RemoteChangeBus

    controller = _Controller(_Service())
    controller.service.is_remote_project = lambda project_id: project_id != "local"
    controller._sync_timer = QTimer()
    controller._sync_timer.setSingleShot(True)
    controller._pending_wakes = {}
    controller._pending_catalog_refresh = False
    controller._project_workspaces = {"remote-a": object(), "local": object(), "remote-b": object()}
    controller._project_id = None
    controller._change_bus = RemoteChangeBus()
    controller._remote_status = "reconnecting"
    controller.my_work_refresh = lambda _project_id=None: None
    controller.refresh_projects = lambda: None
    received = []
    controller._change_bus.changed.connect(lambda project_id, topics, _ids: received.append((project_id, topics)))

    controller._on_remote_status("synchronized")
    controller._sync_timer.stop()
    controller._drain_live_wakes()

    assert set(received) == {("remote-a", frozenset({"project"})),
                             ("remote-b", frozenset({"project"}))}


def test_agent_timer_synchronizes_only_cached_local_projects(qtbot):
    local = SimpleNamespace(id="local")
    remote = SimpleNamespace(id="remote")
    service = _Service()
    service.list_projects = lambda **_kwargs: pytest.fail("timer must not fetch server catalog")
    service.is_remote_project = lambda project_id: project_id == "remote"
    captured = []

    def synchronize(*, principal, gateway, projects=None):
        captured.append(projects)
        return ()

    service.synchronize_plugin_jobs = synchronize
    controller = _Controller(service)
    controller._known_projects = (local, remote)
    controller._poll_agent_jobs()
    assert captured == [(local,)]


def test_hidden_my_work_defers_reads_until_shown(qtbot):
    class Service:
        calls = 0

        def list_projects(self, *, include_archived):
            Service.calls += 1
            return ()

        def list_performers(self, *, include_archived):
            return ()

        def review_batches(self):
            return ()

        def plugin_jobs(self, *, projects=None):
            return ()

    stack = QWidget()
    qtbot.addWidget(stack)
    controller = SimpleNamespace()
    panel = _my_work_panel(Service(), SimpleNamespace(principal=None), controller)
    panel.setParent(stack)
    stack.show()
    panel.show()
    assert Service.calls == 1

    panel.hide()
    controller.my_work_refresh()
    controller.my_work_refresh()
    assert Service.calls == 1

    panel.show()
    assert Service.calls == 2

    # Showing again with nothing new does not read again.
    panel.hide()
    panel.show()
    assert Service.calls == 2


def test_my_work_signal_reads_only_changed_project(qtbot):
    calls = []

    class Service:
        def list_projects(self, *, include_archived):
            calls.append(("catalog", None))
            return (SimpleNamespace(id="a", name="A"), SimpleNamespace(id="b", name="B"))

        def project_permissions(self, project_id, principal):
            calls.append(("permissions", project_id))
            return frozenset()

        def list_layers(self, project_id, *, include_archived):
            calls.append(("layers", project_id))
            return ()

        def list_performers(self, *, include_archived):
            calls.append(("performers", None))
            return ()

        def review_batches(self, *, projects=None):
            calls.append(("reviews", projects[0].id))
            return ()

        def plugin_jobs(self, *, projects=None):
            calls.append(("jobs", projects[0].id))
            return ()

        def work_orders(self, *, projects=None):
            calls.append(("orders", projects[0].id))
            return ()

    host = QWidget()
    qtbot.addWidget(host)
    controller = SimpleNamespace()
    panel = _my_work_panel(Service(), SimpleNamespace(principal=None), controller)
    panel.setParent(host)
    host.show()
    panel.show()
    calls.clear()

    controller.my_work_refresh("a")

    assert calls
    assert all(project_id == "a" for _kind, project_id in calls)
    assert ("jobs", "a") in calls and ("orders", "a") in calls
