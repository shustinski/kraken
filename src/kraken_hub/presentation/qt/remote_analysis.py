"""Nonblocking server AnalysisRun controls using the existing workspace panel."""

from __future__ import annotations

import csv
import hashlib
from functools import partial
from io import StringIO
import json
from uuid import uuid4
import weakref

from PyQt6 import sip
from PyQt6.QtCore import QObject, QTimer
from PyQt6.QtWidgets import QFileDialog, QLabel, QPushButton

from kraken_hub.presentation.qt.parts.layer_create_thread import _BackgroundCallThread
from kraken_manager.infrastructure.filesystem._atomic import atomic_write_bytes

_ACTIVE_CALLS = set()


class RemoteAnalysisController(QObject):
    def __init__(self, workspace, service, project_id, *, language="Русский", background_workers=None):
        super().__init__(workspace)
        self.workspace = weakref.ref(workspace)
        self.service, self.project_id = service, str(project_id)
        self.language = language
        self._background_workers = background_workers
        self._worker = None
        self._pending = None
        self._runs = {}
        self._preferred_run = None
        self._refresh_after = False
        self._retry = None
        self._map_waiting = None
        panel = workspace.analysis_runs_panel
        self.status = QLabel()
        self.status.setObjectName("remoteAnalysisStatus")
        panel.layout().addWidget(self.status)
        refresh = QPushButton(self._t("Обновить", "Refresh"))
        refresh.clicked.connect(self.refresh)
        panel.layout().addWidget(refresh)
        self.refresh_button = refresh
        self.retry_request = QPushButton(self._t("Повторить запрос", "Retry request"))
        self.retry_request.setObjectName("remoteAnalysisRetryRequest")
        self.retry_request.setEnabled(False)
        self.retry_request.clicked.connect(self._retry_request)
        panel.layout().addWidget(self.retry_request)
        panel.metricChanged.connect(lambda _: self.refresh())
        panel.run_table.itemSelectionChanged.connect(self.refresh)
        panel.retryRequested.connect(lambda identifier: self.command(identifier, "retry"))
        panel.cancelRequested.connect(lambda identifier: self.command(identifier, "cancel"))
        panel.repeatRequested.connect(lambda identifier: self.command(identifier, "repeat"))
        panel.exportRequested.connect(self.export)
        panel.renderMapRequested.connect(self.render_map)
        panel.runtime_button.setText(self._t("Среда worker", "Worker runtime"))
        panel.runtimeRequested.connect(self.runtime_details)
        panel.map_button.setToolTip(self._t("Построить карту по сохранённым версиям входов", "Render using the saved input versions"))
        self.timer = QTimer(self)
        self.timer.setInterval(2000)
        self.timer.timeout.connect(self.refresh)
        self.timer.start()
        QTimer.singleShot(0, self.refresh)

    def _t(self, ru, en):
        return ru if self.language.lower().startswith(("ru", "рус")) else en

    def _panel(self):
        owner = self.workspace()
        return None if owner is None or sip.isdeleted(owner) else owner.analysis_runs_panel

    def _run(self, call, loaded, *, mutation=False):
        if self._worker is not None:
            if mutation:
                if self._pending is not None:
                    self.status.setText(self._t("Запрос уже ожидает выполнения", "A request is already pending"))
                    return False
                self._pending = (call, loaded)
                return True
            return False
        worker = _BackgroundCallThread(call)
        self._worker = worker
        _ACTIVE_CALLS.add(worker)
        if self._background_workers is not None:
            self._background_workers.add(worker)
        reference = weakref.ref(self)

        def received(result):
            owner = reference()
            if owner is not None and not sip.isdeleted(owner) and owner._panel() is not None:
                try:
                    if mutation:
                        owner._retry = None
                        owner.retry_request.setEnabled(False)
                    loaded(result)
                except Exception as exc:
                    owner.status.setText(str(exc))

        def failed(message):
            owner = reference()
            if owner is not None and not sip.isdeleted(owner) and owner._panel() is not None:
                owner.status.setText(message)
                if mutation:
                    owner._retry = (call, loaded)
                    owner.retry_request.setEnabled(True)

        background_workers = self._background_workers

        def finished():
            _ACTIVE_CALLS.discard(worker)
            if background_workers is not None:
                background_workers.discard(worker)
            owner = reference()
            if owner is not None and not sip.isdeleted(owner):
                owner._worker = None
                if owner._pending is not None:
                    call, callback = owner._pending
                    owner._pending = None
                    owner._run(call, callback, mutation=True)
                elif owner._refresh_after:
                    owner._refresh_after = False
                    QTimer.singleShot(0, owner.refresh)

        worker.succeeded.connect(received)
        worker.failed.connect(failed)
        worker.finished.connect(finished)
        worker.finished.connect(worker.deleteLater)
        worker.start()
        return True

    def _retry_request(self):
        if self._retry is not None:
            self._run(*self._retry, mutation=True)

    def refresh(self):
        panel = self._panel()
        if panel is None or self._worker is not None:
            return
        service, project = self.service, self.project_id
        selected = self._preferred_run or panel.selected_run_id()
        metric = str(panel.metric_combo.currentData() or "xor")

        def load():
            runs = service.server_analysis_runs(project)
            identifier = (
                selected if any(run["run_id"] == selected for run in runs) else (runs[0]["run_id"] if runs else None)
            )
            rows = () if identifier is None else service.server_analysis_frames(project, identifier, metric)
            return runs, identifier, rows, metric

        self._run(load, self._loaded)

    def _loaded(self, result):
        runs, identifier, rows, metric = result
        panel = self._panel()
        if panel is None:
            return
        self._runs = {run["run_id"]: run for run in runs}
        current = panel.selected_run_id()
        if current and current != identifier and self._preferred_run != identifier:
            self._refresh_after = True
            return
        panel.run_table.blockSignals(True)
        try:
            panel.set_runs(
                dict(
                    run_id=run["run_id"],
                    state=run["state"],
                    progress=f"{run['completed_frames'] + run['failed_frames']}/{run['total_frames']}",
                    models=", ".join(
                        f"{item['binding_key']}={item.get('display_name') or item['source_id']}"
                        for item in run["manifest"]["source_bindings"]
                    ),
                    recipe=json.dumps(run["manifest"]["recipe"]["expression"], ensure_ascii=False),
                    created_at="",
                    runtime_summary=(json.dumps(run["worker_runtime"], ensure_ascii=False) if run.get("worker_runtime")
                                     else self._t("Среда worker не подтверждена", "Worker runtime unverified")),
                )
                for run in runs
            )
            for index, run in enumerate(runs):
                if run["run_id"] == identifier:
                    panel.run_table.selectRow(index)
        finally:
            panel.run_table.blockSignals(False)
        self._preferred_run = None
        if metric == str(panel.metric_combo.currentData() or "xor"):
            panel.set_results(rows)
        run = self._runs.get(identifier)
        panel.progress.setValue(
            0 if run is None else round(100 * (run["completed_frames"] + run["failed_frames"]) / run["total_frames"])
        )
        if self._map_waiting is not None:
            map_run, frame_id = self._map_waiting
            entry = self._runs.get(map_run, {}).get("maps", {}).get(frame_id)
            if entry and entry["state"] == "ready":
                if self._load_map(map_run, frame_id):
                    self._map_waiting = None
            elif entry and entry["state"] in {"failed", "cancelled"}:
                self._map_waiting = None
                self.status.setText(entry.get("error") or self._t("Не удалось построить карту", "Map rendering failed"))

    def render_map(self, run_id, frame_id):
        run = self._runs.get(run_id)
        if run is None or not frame_id:
            self.status.setText(self._t("Выберите прогон и кадр", "Select a run and frame"))
            return
        entry = run.get("maps", {}).get(frame_id)
        if entry and entry["state"] == "ready":
            if not self._load_map(run_id, frame_id):
                self._map_waiting = (run_id, frame_id)
            return
        if entry and entry["state"] in {"queued", "running"}:
            self._map_waiting = (run_id, frame_id)
            self.refresh()
            return
        if self._run(partial(self.service.request_server_analysis_map, self.project_id, run_id, frame_id,
                             expected_revision=run["revision"], idempotency_key=str(uuid4())),
                     self._changed, mutation=True):
            self._map_waiting = (run_id, frame_id)

    def _load_map(self, run_id, frame_id):
        expected = dict(self._runs[run_id]["maps"][frame_id]["blob"])
        service, project_id = self.service, self.project_id

        def load():
            data = service.server_analysis_map_bytes(project_id, run_id, frame_id)
            if len(data) != expected["size_bytes"] or hashlib.sha256(data).hexdigest() != expected["sha256"]:
                raise ValueError("Rendered map does not match the pinned result checksum")
            return data

        return self._run(load, partial(self._show_map, run_id, frame_id), mutation=True)

    def _show_map(self, run_id, frame_id, data):
        from kraken_hub.presentation.qt.analysis_map_view import show_analysis_map

        panel = self._panel()
        if panel is None:
            return
        run = self._runs.get(run_id, {})
        entry = run.get('maps', {}).get(frame_id, {})
        frames = entry.get('manifest', {}).get('frames', ())
        sources = frames[0].get('artifacts', ()) if frames else ()
        show_analysis_map(panel, run_id, frame_id, data,
            expression=run.get('manifest', {}).get('recipe', {}).get('expression', {}),
            sources=sources, language=self.language, runtime=entry.get("result", {}).get("worker_runtime"),
            object_name='remoteAnalysisMapDialog')

    def runtime_details(self, run_id):
        if run_id:
            self._run(partial(self.service.server_analysis_run, self.project_id, run_id), self._show_runtime, mutation=True)

    def _show_runtime(self, run):
        from kraken_hub.presentation.qt.analysis_map_view import show_runtime_report
        panel = self._panel()
        if panel is None:
            return
        report = {"run_id": run["run_id"], "state": run["state"], "declared_runtime": run["manifest"].get("runtime"),
                  "worker_runtime": run.get("worker_runtime"), "partitions": [
                      {"job_id": part["manifest"]["job_id"], "expected_runtime": part["manifest"].get("expected_runtime"),
                       "reported_runtime": part.get("result", {}).get("worker_runtime"), "error": part.get("error", "")}
                      for part in run.get("partitions", ())]}
        show_runtime_report(panel, report, language=self.language)

    def _changed(self, run):
        self._preferred_run = run["run_id"]
        self.status.setText(self._t("Сервер принял запрос", "Server accepted the request"))
        self._refresh_after = True

    def start(self, configuration, layer_id, coordinates):
        if not layer_id or not coordinates:
            raise ValueError(self._t("Выберите слой и кадры", "Select a layer and frames"))
        payload = dict(
            layer_id=str(layer_id),
            coordinates=[list(xy) for xy in coordinates],
            bindings=configuration["bindings"],
            recipe=configuration["recipe"],
            parameters=configuration.get("parameters", {}),
        )
        self._run(
            partial(self.service.create_server_analysis, self.project_id, payload, idempotency_key=str(uuid4())),
            self._changed,
            mutation=True,
        )

    def command(self, run_id, operation):
        run = self._runs.get(run_id)
        if run is None:
            return
        self._run(
            partial(
                self.service.command_server_analysis,
                self.project_id,
                run_id,
                operation,
                expected_revision=run["revision"],
                idempotency_key=str(uuid4()),
            ),
            self._changed,
            mutation=True,
        )

    def export(self, run_id):
        panel = self._panel()
        if not run_id or panel is None:
            return
        destination, _ = QFileDialog.getSaveFileName(
            panel, self._t("Экспорт анализа", "Export analysis"), f"analysis-{run_id}.csv", "CSV (*.csv)"
        )
        if not destination:
            return
        service, project = self.service, self.project_id
        metric = str(panel.metric_combo.currentData() or "xor")

        def write():
            rows = service.server_analysis_frames(project, run_id, metric)
            text = StringIO(newline="")
            fields = ("frame_id", "x", "y", "status", "message", "raw_value", "goodness", "percentile")
            writer = csv.DictWriter(text, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            from pathlib import Path

            atomic_write_bytes(Path(destination), text.getvalue().encode("utf-8-sig"), overwrite=True)
            return destination

        self._run(write, lambda path: self.status.setText(str(path)), mutation=True)
