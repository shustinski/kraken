"""manager_app: configure analysis."""
from __future__ import annotations
from uuid import uuid4
from kraken_manager.application.analysis_runs import AnalysisRunCoordinator
from kraken_manager.domain.project import Project
from kraken_manager.infrastructure.plugin import AgentAnalysisGateway
from kraken_manager.workspace import ProjectWorkspaceBinding


class ConfigureAnalysisMixin:
    def _setup_project_workspace(self, project: Project) -> ProjectWorkspaceBinding | None:
        from kraken_hub.manager_app import _configure_workspace_roots
        from PyQt6.QtWidgets import QMessageBox

        answer = QMessageBox.question(
            self.shell,
            "Настроить хранилище проекта",
            f"Для проекта «{project.name}» нужно выбрать каталоги исходных и "
            "производных данных. Kraken создаст в них новые папки проекта.\n\n"
            "Существующие слои, представления и история сохранятся. "
            "Существующие файлы не будут перенесены или удалены.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return None
        roots = _configure_workspace_roots(self.shell, self.service, force=True)
        if roots is None:
            return None
        try:
            return self.service.bind_project_workspace(
                principal=self.session.principal,
                project=project,
                source_root=roots[0],
                derived_root=roots[1],
                idempotency_key=str(uuid4()),
            )
        except Exception as exc:
            self._error(str(exc))
            return None

    def _configure_analysis(self, workspace, project_id) -> None:
        from kraken_hub.manager_app import FilesystemAnalysisStore
        from PyQt6.QtCore import QTimer
        workspace._analysis_language = getattr(self, "language", "Русский")

        if getattr(self.service, "is_remote_project", lambda _id: False)(project_id):
            from kraken_hub.remote_analysis import RemoteAnalysisController
            workspace._analysis_store = None
            workspace._analysis_coordinator = RemoteAnalysisController(
                workspace, self.service, project_id, language=getattr(self, "language", "Русский"),
                background_workers=getattr(self, "_background_workers", None))
            workspace._analysis_timer = workspace._analysis_coordinator.timer
            workspace.analysisRequested.connect(
                lambda configuration: self._start_analysis(workspace, project_id, configuration)
            )
            return
        store = FilesystemAnalysisStore(self.service.catalog_root, str(project_id))
        workspace._analysis_store = store
        coordinator = None
        try:
            self.agent_runtime.ensure_started()
        except (OSError, RuntimeError, ValueError) as exc:
            self.shell.statusBar().showMessage(f"Kraken Agent: {exc}", 10000)
        else:
            gateway = AgentAnalysisGateway(
                base_url=self.agent_runtime.base_url,
                token=self.agent_runtime.token,
                staging_root=self.service.data_dir / "agent" / "staging",
            )
            coordinator = AnalysisRunCoordinator(
                store,
                gateway,
                lambda version_id: self.service.artifact_source_path(project_id, version_id),
            )
        workspace._analysis_coordinator = coordinator
        workspace.analysisRequested.connect(
            lambda configuration: self._start_analysis(workspace, project_id, configuration)
        )
        panel = workspace.analysis_runs_panel
        panel.metricChanged.connect(lambda _key: self._refresh_analysis_panel(workspace))
        panel.run_table.itemSelectionChanged.connect(lambda: self._refresh_analysis_results(workspace))
        panel.retryRequested.connect(lambda run_id: self._analysis_action(workspace, "retry", run_id))
        panel.cancelRequested.connect(lambda run_id: self._analysis_action(workspace, "cancel", run_id))
        panel.repeatRequested.connect(lambda run_id: self._analysis_action(workspace, "repeat", run_id))
        panel.exportRequested.connect(lambda run_id: self._export_analysis(workspace, run_id))
        panel.renderMapRequested.connect(
            lambda run_id, frame_id: self._show_discrepancy_map(workspace, run_id, frame_id)
        )
        if coordinator is not None:
            from kraken_hub.local_analysis_maps import LocalAnalysisMapController
            controller = LocalAnalysisMapController(
                workspace, coordinator, language=getattr(self, "language", "Русский"),
                background_workers=getattr(self, "_background_workers", None))
            controller.updated.connect(lambda: self._refresh_analysis_panel(workspace))
            workspace._analysis_map_controller = controller
            workspace._analysis_timer = controller.timer
            self._refresh_analysis_panel(workspace)
            return
        timer = QTimer(workspace)
        timer.setInterval(750)
        timer.timeout.connect(lambda: self._poll_analysis(workspace))
        timer.start()
        workspace._analysis_timer = timer
        self._refresh_analysis_panel(workspace)
