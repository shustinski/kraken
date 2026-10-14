"""manager_app: poll agent jobs."""
from __future__ import annotations
import logging
from kraken_core.frame_matrix import (
    MatrixSession,
)
from kraken_manager.domain.identity import Permission
from kraken_manager.presentation.qt.models import ProjectListItem
from kraken_hub.workspace_service import ProjectEventWake


class PollAgentJobsMixin:
    def _wire_live_sync(self) -> None:
        add_wake = getattr(self.service, "add_wake_handler", None)
        add_catalog = getattr(self.service, "add_catalog_handler", None)
        add_status = getattr(self.service, "add_status_handler", None)
        start_sync = getattr(self.service, "start_sync", None)
        if callable(add_wake):
            add_wake(lambda wake: self._remote_bridge.wakeReceived.emit(wake))
        if callable(add_catalog):
            add_catalog(lambda: self._remote_bridge.wakeReceived.emit(None))
        if callable(add_status):
            add_status(lambda status: self._remote_bridge.statusChanged.emit(status))
        if callable(start_sync):
            try:
                start_sync()
            except Exception:
                logging.getLogger(__name__).exception("Failed to start remote live sync")

    def _on_remote_wake(self, wake: ProjectEventWake | None) -> None:
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
        if self._project_id is None or self._workspace is None:
            return
        relevant = [
            wake
            for wake in wakes
            if wake is not None and wake.project_id == self._project_id
        ]
        if catalog_changed or relevant:
            project = self.service.get_project(self._project_id)
            if project is None:
                return
            self._workspace.set_project_title(project.name)
            matrix = getattr(self._workspace, "matrix_view", None)
            if matrix is not None:
                session = getattr(matrix, "session", None)
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
                        data_source=getattr(self._workspace, "_matrix_data_source", None),
                        asset_source=getattr(self._workspace, "_matrix_asset_source", None),
                    )
            self._load_layers(self._workspace, project)

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
        active = next(
            (
                job
                for job in self.service.plugin_jobs()
                if job.state.value not in {"succeeded", "failed", "cancelled"}
            ),
            None,
        )
        if active is None:
            return
        try:
            self._gateway_for_job(active)
            self._poll_agent_jobs()
        except Exception as exc:
            self.shell.statusBar().showMessage(
                f"Kraken Agent: {exc}",
                10000,
            )

    def _poll_agent_jobs(self) -> None:
        gateway = self._agent_gateway
        if gateway is None:
            return
        try:
            jobs = self.service.synchronize_plugin_jobs(
                principal=self.session.principal,
                gateway=gateway,
            )
            for job in jobs:
                if job.state.value != "importing":
                    continue
                remote_project = getattr(self.service, "is_remote_project", lambda _project_id: False)
                if remote_project(job.project_id):
                    continue
                publications = gateway.get_publications(job.id)
                if publications:
                    imported = self.service.import_agent_publications(
                        principal=self.session.principal,
                        publications=publications,
                        staging_root=self.service.data_dir / "agent" / "staging",
                    )
                else:
                    payload = gateway.get_result(job.id)
                    imported = self.service.import_agent_result(
                        principal=self.session.principal,
                        result_payload=payload,
                        staging_root=self.service.data_dir / "agent" / "staging",
                    )
                if not imported.requires_partial_confirmation:
                    gateway.complete_import(job.id)
                    self._refresh_current_matrix()
            if self.my_work_refresh is not None:
                self.my_work_refresh()
            self._agent_poll_error = ""
        except Exception as exc:
            message = str(exc)
            if message != self._agent_poll_error:
                self.shell.statusBar().showMessage(
                    f"Kraken Agent: {message}",
                    10000,
                )
                self._agent_poll_error = message
