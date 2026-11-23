"""manager_app: poll agent jobs."""
from __future__ import annotations
import logging
from PyQt6 import sip
from kraken_core.frame_matrix import (
    MatrixSession,
)
from kraken_manager.domain.identity import Permission
from kraken_manager.presentation.qt.models import ProjectListItem
from kraken_hub.application.workspace import ProjectEventWake
from kraken_hub.presentation.qt.parts.gateway_job import _file_work_job_ids
from kraken_hub.presentation.qt.parts.my_work_panel import _call_with_projects, _catalog_projects
from kraken_hub.presentation.qt.parts.layer_create_thread import _BackgroundCallThread


class PollAgentJobsMixin:
    def _emit_remote_signal(self, signal: str, value: object) -> None:
        bridge = self._remote_bridge
        if not sip.isdeleted(bridge):
            getattr(bridge, signal).emit(value)

    def _wire_live_sync(self) -> None:
        add_wake = getattr(self.service, "add_wake_handler", None)
        add_catalog = getattr(self.service, "add_catalog_handler", None)
        add_status = getattr(self.service, "add_status_handler", None)
        start_sync = getattr(self.service, "start_sync", None)
        if callable(add_wake):
            add_wake(lambda wake: self._emit_remote_signal("wakeReceived", wake))
        if callable(add_catalog):
            add_catalog(lambda: self._emit_remote_signal("wakeReceived", None))
        if callable(add_status):
            add_status(lambda status: self._emit_remote_signal("statusChanged", status))
        if callable(start_sync):
            try:
                start_sync()
            except Exception:
                logging.getLogger(__name__).exception("Failed to start remote live sync")

    def _on_remote_wake(self, wake: ProjectEventWake | None) -> None:
        if sip.isdeleted(self._sync_timer):
            return
        if wake is None:
            self._pending_catalog_refresh = True
        else:
            key = (
                wake.project_id,
                wake.entity_kind or wake.stream_id,
                wake.entity_id or wake.stream_id,
            )
            self._pending_wakes[key] = wake
        if not self._sync_timer.isActive():
            self._sync_timer.start(50)

    def _apply_wake_to_active(self, project_id: str) -> None:
        workspace = self._project_workspaces.get(str(project_id))
        project = self.service.get_project(project_id)
        if workspace is None or project is None:
            return
        workspace.set_project_title(project.name)
        matrix = getattr(workspace, "matrix_view", None)
        if matrix is not None:
            session = getattr(matrix, "session", None)
            if session is not None:
                if callable(session):
                    session = session()
                if session is not None:
                    matrix.set_session(
                        MatrixSession(
                            namespace=str(project.id),
                            width=project.width,
                            height=project.height,
                            source_revision=str(project.revision),
                            orientation=project.orientation.value,
                            generation=getattr(session, "generation", 0) + 1,
                        ),
                        data_source=getattr(workspace, "_matrix_data_source", None),
                        asset_source=getattr(workspace, "_matrix_asset_source", None),
                    )
        self._load_layers(workspace, project)

    def _apply_retained_wakes(self, project_id: str) -> None:
        project_id = str(project_id)
        retained = [wake for key, wake in self._pending_wakes.items() if key[0] == project_id]
        for key in [key for key in self._pending_wakes if key[0] == project_id]:
            del self._pending_wakes[key]
        if retained:
            self._apply_wake_to_active(project_id)

    def _drain_live_wakes(self) -> None:
        wakes = list(self._pending_wakes.values())
        self._pending_wakes.clear()
        catalog_changed = self._pending_catalog_refresh or any(
            wake.event_type.startswith("Project") or wake.event_type.startswith("project.")
            for wake in wakes
        )
        self._pending_catalog_refresh = False
        if catalog_changed:
            self.refresh_projects()
        active = self._project_id
        if active is not None and active in self._project_workspaces:
            has_wakes = any(wake.project_id == active for wake in wakes)
            if catalog_changed or has_wakes:
                self._apply_wake_to_active(active)
        for wake in wakes:
            if wake.project_id in self._project_workspaces and wake.project_id != active:
                self._pending_wakes[
                    (
                        wake.project_id,
                        wake.entity_kind or wake.stream_id,
                        wake.entity_id or wake.stream_id,
                    )
                ] = wake

    def _sync_project_permissions(self, item: ProjectListItem | None) -> None:
        permissions = (
            frozenset()
            if item is None
            else frozenset(
                permission.value
                for permission in self.service.project_permissions(
                    item.project_id,
                    self.session.principal,
                )
            )
        )
        self.catalog_page.set_project_permissions(permissions)
        remote = item is not None and getattr(self.service, "is_remote_project", lambda _: False)(item.project_id)
        self.catalog_page.delete_button.setText("Запросить удаление" if remote else "Удалить")
        self.catalog_page.delete_button.setToolTip(
            "Полное удаление после подтверждения в Kraken Admin" if remote
            else "Удалить из Kraken, сохранив папки исходных и производных данных"
        )

    def _has_permission(self, permission: Permission) -> bool:
        if self._project_id is None:
            return False
        return permission in self.service.project_permissions(
            self._project_id,
            self.session.principal,
        )

    def _resume_agent_jobs(self) -> None:
        try:
            projects = _catalog_projects(self.service)
            local_jobs = tuple(
                job for job in _call_with_projects(self.service.plugin_jobs, projects)
                if job.state.value not in {"succeeded", "failed", "cancelled"}
                and not getattr(self.service, "is_remote_project", lambda _: False)(job.project_id)
            )
            if not local_jobs:
                return
            work_orders = getattr(self.service, "work_orders", None)
            file_jobs = _file_work_job_ids(
                () if work_orders is None else _call_with_projects(work_orders, projects)
            )
            active = next(
                (
                    job
                    for job in local_jobs
                    if str(job.id) not in file_jobs
                ),
                None,
            )
            if active is None:
                return
            self._gateway_for_job(active)
            self._poll_agent_jobs(projects)
        except Exception as exc:
            self.shell.statusBar().showMessage(
                f"Kraken Agent: {exc}",
                10000,
            )

    def _poll_agent_jobs(self, projects=None) -> None:
        schedule_pipeline = getattr(self, "_schedule_pipeline_cycle", None)
        if callable(schedule_pipeline):
            schedule_pipeline()
        gateway = self._agent_gateway
        if projects is None:
            projects = _catalog_projects(self.service)
        if gateway is None:
            if self.my_work_refresh is not None:
                self.my_work_refresh()
            return
        if getattr(self, "_agent_import_worker", None) is not None:
            return
        try:
            def synchronize(projects=None):
                arguments = {"principal": self.session.principal, "gateway": gateway}
                if projects is not None:
                    arguments["projects"] = projects
                return self.service.synchronize_plugin_jobs(**arguments)

            jobs = _call_with_projects(synchronize, projects)
            work_orders = getattr(self.service, "work_orders", None)
            file_jobs = _file_work_job_ids(
                () if work_orders is None else _call_with_projects(work_orders, projects)
            )
            started_import = False
            for job in jobs:
                if job.state.value != "importing" or str(job.id) in file_jobs:
                    continue
                remote_project = getattr(self.service, "is_remote_project", lambda _project_id: False)
                if remote_project(job.project_id):
                    continue
                self._import_agent_job_in_background(job.id, gateway)
                started_import = True
                break
            if self.my_work_refresh is not None:
                self.my_work_refresh()
            if not started_import:
                self._agent_poll_error = ""
        except Exception as exc:
            message = str(exc)
            if message != self._agent_poll_error:
                self.shell.statusBar().showMessage(
                    f"Kraken Agent: {message}",
                    10000,
                )
                self._agent_poll_error = message

    def _import_agent_job_in_background(self, job_id, gateway) -> None:
        service = self.service
        principal = self.session.principal
        staging_root = service.data_dir / "agent" / "staging"

        def import_result():
            publications = gateway.get_publications(job_id)
            if publications:
                imported = service.import_agent_publications(
                    principal=principal,
                    publications=publications,
                    staging_root=staging_root,
                )
            else:
                imported = service.import_agent_result(
                    principal=principal,
                    result_payload=gateway.get_result(job_id),
                    staging_root=staging_root,
                )
            if not imported.requires_partial_confirmation:
                gateway.complete_import(job_id)
            return not imported.requires_partial_confirmation

        worker = _BackgroundCallThread(import_result, parent=self.shell)
        self._agent_import_worker = worker
        self._background_workers.add(worker)

        def failed(message):
            if message != self._agent_poll_error:
                self.shell.statusBar().showMessage(f"Kraken Agent: {message}")
                self._agent_poll_error = message

        def succeeded(completed):
            self._agent_poll_error = ""
            if completed:
                self._refresh_current_matrix()
            if self.my_work_refresh is not None:
                self.my_work_refresh()

        def finished():
            self._background_workers.discard(worker)
            self._agent_import_worker = None

        worker.failed.connect(failed)
        worker.succeeded.connect(succeeded)
        worker.finished.connect(finished)
        worker.finished.connect(worker.deleteLater)
        worker.start()
