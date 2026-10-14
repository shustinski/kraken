"""manager_app: start analysis."""
from __future__ import annotations
from pathlib import Path
from kraken_core.analysis_protocol import (
    AnalysisParameter,
)
from kraken_core.analysis_run_protocol import (
    AnalysisRecipe,
    canonical_json,
)

from kraken_hub.manager_app_parts.karakal_runtime_identity import _karakal_runtime_identity


class StartAnalysisMixin:
    def _start_analysis(self, workspace, project_id, configuration) -> None:
        from PyQt6.QtWidgets import QMessageBox

        coordinator = getattr(workspace, "_analysis_coordinator", None)
        if coordinator is None:
            if getattr(self.service, "is_remote_project", lambda _id: False)(project_id):
                QMessageBox.warning(workspace, "Karakal", "Серверный анализ Karakal пока не поддерживается.")
                return
            QMessageBox.warning(
                workspace,
                "Kraken Agent не настроен",
                "Задайте KRAKEN_AGENT_TOKEN, KRAKEN_AGENT_URL и KRAKEN_AGENT_STAGING_ROOT, затем перезапустите Kraken.",
            )
            return
        try:
            frames, source_bindings = self._analysis_frames(workspace, project_id, configuration)
            raw_recipe = configuration.get("recipe")
            raw_parameters = configuration.get("parameters", {})
            if not isinstance(raw_recipe, dict) or not isinstance(raw_parameters, dict):
                raise ValueError("Некорректная конфигурация анализа")
            recipe = AnalysisRecipe.from_payload(raw_recipe)
            parameters = tuple(
                AnalysisParameter(str(key), value)
                for key, value in sorted(raw_parameters.items())
                if isinstance(value, (str, int, float, bool))
            )
            coordinator.start(
                project_id=str(project_id),
                frames=frames,
                source_bindings=source_bindings,
                recipe=recipe,
                runtime=_karakal_runtime_identity(),
                parameters=parameters,
            )
        except Exception as exc:
            QMessageBox.warning(workspace, "Не удалось запустить анализ", str(exc))
            self._refresh_analysis_panel(workspace)
            return
        workspace.analysis_runs_panel.setVisible(True)
        self._refresh_analysis_panel(workspace)

    def _poll_analysis(self, workspace) -> None:
        coordinator = getattr(workspace, "_analysis_coordinator", None)
        store = getattr(workspace, "_analysis_store", None)
        if coordinator is None or store is None:
            return
        changed = False
        for run in store.list_runs():
            if run.state in {"queued", "running"}:
                coordinator.refresh(run.run_id)
                changed = True
        if changed:
            self._refresh_analysis_panel(workspace)

    def _refresh_analysis_panel(self, workspace) -> None:
        store = getattr(workspace, "_analysis_store", None)
        if store is None:
            return
        panel = workspace.analysis_runs_panel
        selected = panel.selected_run_id()
        runs = store.list_runs()
        panel.set_runs(
            {
                "run_id": run.run_id,
                "state": run.state,
                "progress": f"{run.completed_frames + run.failed_frames}/{run.total_frames}",
                "models": ", ".join(
                    f"{item.binding_key}={item.display_name or item.source_id}"
                    for item in run.manifest.source_bindings
                ),
                "recipe": canonical_json(run.manifest.recipe.expression.to_payload()),
                "created_at": "",
            }
            for run in runs
        )
        if runs:
            selected_row = next((index for index, run in enumerate(runs) if run.run_id == selected), 0)
            panel.run_table.selectRow(selected_row)
            active = runs[selected_row]
            progress = round(100 * (active.completed_frames + active.failed_frames) / active.total_frames)
            panel.progress.setValue(progress)
        else:
            panel.progress.setValue(0)
        self._refresh_analysis_results(workspace)

    def _refresh_analysis_results(self, workspace) -> None:
        store = getattr(workspace, "_analysis_store", None)
        panel = workspace.analysis_runs_panel
        run_id = panel.selected_run_id()
        if store is None or not run_id:
            panel.set_results(())
            return
        metric_key = str(panel.metric_combo.currentData() or "xor")
        panel.set_results(dict(row) for row in store.frame_results(run_id, metric_key))

    def _analysis_action(self, workspace, action: str, run_id: str) -> None:
        from PyQt6.QtWidgets import QMessageBox

        coordinator = getattr(workspace, "_analysis_coordinator", None)
        if coordinator is None or not run_id:
            return
        try:
            if action == "retry":
                coordinator.retry_failed(run_id)
            elif action == "cancel":
                coordinator.cancel(run_id)
            elif action == "repeat":
                coordinator.repeat(run_id)
        except Exception as exc:
            QMessageBox.warning(workspace, "Операция анализа не выполнена", str(exc))
        self._refresh_analysis_panel(workspace)

    def _export_analysis(self, workspace, run_id: str) -> None:
        import csv
        from PyQt6.QtWidgets import QFileDialog, QMessageBox

        store = getattr(workspace, "_analysis_store", None)
        if store is None or not run_id:
            return
        destination, _selected_filter = QFileDialog.getSaveFileName(
            workspace,
            "Экспорт результатов анализа",
            f"karakal-{run_id}.csv",
            "CSV (*.csv)",
        )
        if not destination:
            return
        metric_key = str(workspace.analysis_runs_panel.metric_combo.currentData() or "xor")
        rows = store.frame_results(run_id, metric_key)
        try:
            with Path(destination).open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(("frame_id", "x", "y", "status", "message", "metric", "raw_value", "goodness", "percentile"))
                for row in rows:
                    writer.writerow(
                        (
                            row["frame_id"], row["x"], row["y"], row["status"], row["message"],
                            metric_key, row["raw_value"], row["goodness"], row["percentile"],
                        )
                    )
        except OSError as exc:
            QMessageBox.warning(workspace, "Не удалось экспортировать анализ", str(exc))

    @staticmethod
    def _show_discrepancy_map(workspace, run_id: str, frame_id: str) -> None:
        from PyQt6.QtWidgets import QMessageBox

        store = getattr(workspace, "_analysis_store", None)
        if store is None or not run_id or not frame_id:
            QMessageBox.information(workspace, "Карта расхождений", "Выберите прогон и кадр анализа.")
            return
        rows = store.metrics_for_frame(run_id, frame_id)
        if not rows:
            QMessageBox.information(
                workspace,
                "Карта расхождений",
                "Для выбранного кадра ещё нет сохранённых метрик анализа.",
            )
            return
        lines = [
            f"{row['metric_key']}: goodness={row['goodness']}, raw={row['raw_value']}, percentile={row['percentile']}"
            for row in rows
        ]
        QMessageBox.information(
            workspace,
            "Карта расхождений",
            f"Кадр {frame_id}\n" + "\n".join(lines),
        )
