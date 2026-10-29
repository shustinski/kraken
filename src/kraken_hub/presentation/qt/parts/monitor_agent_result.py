"""manager_app: monitor agent result."""
from __future__ import annotations
import shutil
from pathlib import Path
from kraken_core.plugin_protocol import (
    PluginResultManifest,
    WorkspacePluginResultV1,
)


class MonitorAgentResultMixin:
    def _monitor_agent_result(
        self,
        process,
        *,
        result_manifest: str,
        staging_root: str,
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
            root = Path(staging_root)
            try:
                result = PluginResultManifest.from_json(
                    Path(result_manifest).read_text(encoding="utf-8")
                )
                if result.outcome != "succeeded":
                    raise ValueError(
                        "; ".join(result.errors)
                        or f"NeuralImage завершился со статусом {result.outcome}"
                    )
                project = self.service.get_project(self._project_id)
                layer = next(
                    (
                        value
                        for value in self.service.list_layers(self._project_id)
                        if str(value.id) == str(layer_id)
                    ),
                    None,
                )
                if project is None or layer is None:
                    raise ValueError("проект или слой больше не доступен")
                workspace_result = WorkspacePluginResultV1(
                    run_id=workspace_run_id,
                    plugin_id="neuralimage",
                    operation=self._action_requirement("recognize")[1],
                    outcome="succeeded",
                    output_directory=str(root / "outputs"),
                    provenance={
                        "plugin_version": result.plugin_version,
                        "output_count": len(result.outputs),
                    },
                )
                published, _representation = (
                    self.service.publish_workspace_plugin_result(
                        principal=self.session.principal,
                        project=project,
                        layer=layer,
                        result=workspace_result,
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
                    "Распознавание NeuralImage не опубликовано",
                    str(exc),
                )
                return
            finally:
                shutil.rmtree(root, ignore_errors=True)
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
                "Распознавание опубликовано",
                f"Запуск {published.run_id[:8]} сохранён как бинарное представление.",
            )

        timer.timeout.connect(poll)
        timer.start()
