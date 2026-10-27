"""manager_app: construction."""
from __future__ import annotations
from collections.abc import Callable
from PyQt6.QtCore import QThread, QTimer
from kraken_core.plugins import PluginInventoryItem
from kraken_manager.infrastructure.plugin import AgentPluginGateway
from kraken_manager.presentation.qt import (
    LayerManagerDialog,
    ProjectManagerShell,
)
from kraken_hub.composition import DesktopSession, EmbeddedProjectService
from kraken_hub.dual_catalog import DualCatalogService
from kraken_hub.workspace_service import ProjectEventWake

from kraken_hub.manager_app_parts.event_matches_node import _RemoteSignalBridge


class ConstructionMixin:
    def open_electrical(self) -> None:
        from kraken_hub.electrical_dialog import ElectricalDialog

        if self._project_id is None:
            self.shell.statusBar().showMessage("Откройте проект для проверки электрической связности", 6000)
            return
        current = getattr(self, "_electrical_dialog", None)
        if current is not None:
            if current._worker is not None:
                current.show()
                current.raise_()
                return
            current.close()
        self._electrical_dialog = ElectricalDialog(
            self.service, self.session.principal, str(self._project_id), self.shell,
            background_workers=self._background_workers,
        )
        self._electrical_dialog.show()

    def open_multilayer(self) -> None:
        from kraken_hub.multilayer_dialog import MultilayerDialog

        if self._project_id is None:
            self.shell.statusBar().showMessage("Откройте проект для настройки физических слоёв", 6000)
            return
        current = getattr(self, "_multilayer_dialog", None)
        if current is not None:
            if current._worker is not None:
                current.show()
                current.raise_()
                return
            current.close()
        self._multilayer_dialog = MultilayerDialog(
            self.service, self.session.principal, str(self._project_id), self.shell,
            background_workers=self._background_workers,
        )
        self._multilayer_dialog.show()

    def open_quality(self) -> None:
        from kraken_hub.quality_dialog import QualityDialog
        if self._project_id is None:
            self.shell.statusBar().showMessage("Откройте проект для контроля качества", 6000)
            return
        current = getattr(self, "_quality_dialog", None)
        if current is not None:
            if current._worker is not None:
                current.show()
                current.raise_()
                return
            current.close()
        self._quality_dialog = QualityDialog(
            self.service, self.session.principal, str(self._project_id), self.shell,
            background_workers=self._background_workers,
        )
        self._quality_dialog.show()

    def open_work_planning(self) -> None:
        from kraken_hub.work_planning_dialog import WorkPlanningDialog

        project_id = self._project_id
        if project_id is None:
            self.shell.statusBar().showMessage("Откройте проект для планирования работ", 6000)
            return
        dialog = getattr(self, "_work_planning_dialog", None)
        if dialog is not None:
            if dialog._worker is not None:
                dialog.show()
                dialog.raise_()
                return
            dialog.close()
        self._work_planning_dialog = WorkPlanningDialog(
            self.service, self.session.principal, str(project_id), self.shell,
            background_workers=self._background_workers,
        )
        self._work_planning_dialog.show()

    def __init__(
        self,
        shell: ProjectManagerShell,
        service: EmbeddedProjectService | DualCatalogService,
        session: DesktopSession,
        *,
        thumbnail_store_uri: str = "",
        plugin_items: list[PluginInventoryItem] | None = None,
    ) -> None:
        from kraken_hub.manager_app import LocalAgentRuntime
        self.shell = shell
        self.service = service
        self.session = session
        self.thumbnail_store_uri = str(thumbnail_store_uri)
        self.plugin_items = {item.metadata.id: item for item in (plugin_items or [])}
        self.agent_runtime = LocalAgentRuntime(
            self.service.data_dir / "agent",
            tuple(plugin_items or ()),
            builtin_stitching=True,
        )
        self._agent_gateway: AgentPluginGateway | None = None
        self._agent_poll_error = ""
        self.my_work_refresh: Callable[[], None] | None = None
        self._agent_timer = QTimer(self.shell)
        self._agent_timer.setInterval(2000)
        self._agent_timer.timeout.connect(self._poll_agent_jobs)
        self._agent_timer.start()
        QTimer.singleShot(0, self._resume_agent_jobs)
        self.shell.close_guard = self._allow_close
        self._workspace = None
        self._project_id = None
        self._layer_dialog: LayerManagerDialog | None = None
        self._background_workers: set[QThread] = set()
        self._source_preparation_keys: set[tuple[str, str]] = set()
        self._pending_wakes: dict[tuple[str, str, str], ProjectEventWake] = {}
        self._pending_catalog_refresh = False
        self._remote_bridge = _RemoteSignalBridge(self.shell)
        self._remote_bridge.wakeReceived.connect(self._on_remote_wake)
        self._remote_bridge.statusChanged.connect(self.shell.set_sync_status)
        self._sync_timer = QTimer(self.shell)
        self._sync_timer.setSingleShot(True)
        self._sync_timer.timeout.connect(self._drain_live_wakes)
        self.catalog_page = shell.page("projects")
        assert self.catalog_page is not None
        self.catalog_page.createRequested.connect(self.create_project)
        self.catalog_page.refreshRequested.connect(self.refresh_projects)
        self.catalog_page.projectActivated.connect(self.open_project)
        self.catalog_page.renameRequested.connect(self.rename_project)
        self.catalog_page.archiveRequested.connect(self.archive_project)
        self.catalog_page.restoreRequested.connect(self.restore_project)
        self.catalog_page.deleteRequested.connect(self.delete_project)
        self.catalog_page.participantsRequested.connect(self.manage_project_participants)
        self.catalog_page.propertiesRequested.connect(self.show_project_properties)
        self.catalog_page.selectionChanged.connect(self._sync_project_permissions)
        self.shell.layersRequested.connect(self.open_layer_manager)
        self.shell.workPlanningRequested.connect(self.open_work_planning)
        self.shell.multilayerRequested.connect(self.open_multilayer)
        self.shell.electricalRequested.connect(self.open_electrical)
        self.shell.qualityRequested.connect(self.open_quality)

        self.shell.preferencesRequested.connect(self.open_preferences)
        self.shell.cellVisualModeChanged.connect(self._set_matrix_visual_mode)
        self.shell.reviewReturnRequested.connect(self.load_review_return)
        self.shell.framePropertiesRequested.connect(self.show_selected_frame)
        self._wire_live_sync()
        self.refresh_projects()
