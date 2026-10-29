"""manager_app: launch managed plugin."""
from __future__ import annotations
from pathlib import Path
from kraken_core.plugins import is_plugin_launchable
from kraken_manager.application.work_launch import plugin_id_for_capability
from kraken_manager.domain.identity import Permission


class LaunchManagedPluginMixin:
    def _launch_managed_plugin(
        self,
        plugin_id: str,
        *,
        arguments: tuple[str, ...] = (),
    ):
        inventory = self.plugin_items.get(plugin_id)
        if inventory is None or not is_plugin_launchable(inventory) or not inventory.metadata.enabled:
            self._error(f"Плагин {plugin_id} не установлен или отключён")
            return None
        from kraken_hub.app import launch_plugin
        return launch_plugin(inventory.metadata, arguments=arguments)

    def launch_selected_work(self, kind: str, work) -> None:
        """Open the plugin that already handles this work item."""

        project_id = getattr(work, "project_id", None)
        if project_id is not None:
            permissions = self.service.project_permissions(project_id, self.session.principal)
            if Permission.RUN_PLUGIN not in permissions:
                self._error("Недостаточно прав для запуска плагинов")
                return
        if kind == "order":
            self.open_issued_work(work)
            return
        if kind == "job":
            plugin_id = plugin_id_for_capability(getattr(work, "capability", ""))
            if plugin_id is None:
                self._error("Для этого задания нет соответствующего плагина")
                return
            self._launch_managed_plugin(plugin_id, arguments=_agent_job_arguments(self, work))
            return
        if kind == "review":
            self._launch_managed_plugin("contour")
            return
        self._error("Для этого задания нет соответствующего плагина")


def _agent_job_arguments(controller, job) -> tuple[str, ...]:
    data_dir = getattr(controller.service, "data_dir", None)
    if data_dir is None:
        return ()
    workspace = Path(data_dir) / "agent" / "staging" / str(job.id)
    manifest = workspace / "job.json"
    if not manifest.is_file():
        return ()
    return (
        "--kraken-job-manifest",
        str(manifest),
        "--kraken-result-manifest",
        str(workspace / "result.json"),
        "--kraken-staging-root",
        str(workspace),
    )
