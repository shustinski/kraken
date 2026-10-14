"""manager_app: monitor workspace result."""
from __future__ import annotations
import logging
from pathlib import Path
from kraken_core.plugin_protocol import (
    WorkspacePluginResultV1,
)


class MonitorWorkspaceResultMixin:
    def _monitor_workspace_result(
        self,
        process,
        *,
        result_manifest: str,
        workspace_run_id: str,
        layer_id: str,
    ) -> None:
        from PyQt6.QtCore import QTimer
        from PyQt6.QtWidgets import QMessageBox

        timer = QTimer(self.shell)
        timer.setInterval(750)

        def poll() -> None:
            if process.poll() is None:
                return
            timer.stop()
            timer.deleteLater()
            manifest_path = Path(result_manifest)
            if not manifest_path.is_file():
                return_code = process.poll()
                message = (
                    "Плагин закрыт без публикации результата"
                    if return_code == 0
                    else f"Плагин завершился с кодом {return_code} и не опубликовал результат"
                )
                try:
                    self.service.fail_derived_run(
                        principal=self.session.principal,
                        project_id=self._project_id,
                        run_id=workspace_run_id,
                        error=message,
                    )
                except Exception:
                    logging.getLogger(__name__).exception(
                        "Could not mark workspace plugin run as failed"
                    )
                QMessageBox.warning(
                    self.shell,
                    "Плагин не вернул результат",
                    message,
                )
                return
            try:
                result = WorkspacePluginResultV1.read(manifest_path)
                project = self.service.get_project(self._project_id)
                if project is None:
                    raise ValueError("проект больше не доступен")
                layer = next(
                    (
                        value
                        for value in self.service.list_layers(project.id)
                        if str(value.id) == str(layer_id)
                    ),
                    None,
                )
                if layer is None:
                    raise ValueError("слой больше не доступен")
                published, _representation = (
                    self.service.publish_workspace_plugin_result(
                        principal=self.session.principal,
                        project=project,
                        layer=layer,
                        result=result,
                    )
                )
            except Exception as exc:
                try:
                    self.service.fail_derived_run(
                        principal=self.session.principal,
                        project_id=self._project_id,
                        run_id=workspace_run_id,
                        error=str(exc),
                    )
                except Exception:
                    pass
                QMessageBox.warning(
                    self.shell,
                    "Не удалось вернуть результат плагина",
                    str(exc),
                )
                return
            workspace = self._workspace
            if workspace is not None:
                latest = self.service.get_project(self._project_id)
                if latest is not None:
                    self._load_layers(workspace, latest)
                item = workspace.layer_model.layer_by_id(layer_id)
                if item is not None:
                    self._select_layer(workspace, self._project_id, item)
            QMessageBox.information(
                self.shell,
                "Результат опубликован",
                f"Запуск {published.run_id[:8]} сохранён во втором хранилище.",
            )

        timer.timeout.connect(poll)
        timer.start()
