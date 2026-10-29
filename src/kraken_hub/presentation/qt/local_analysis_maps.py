"""Nonblocking local map execution and recovery on the existing AnalysisRun panel."""
from uuid import uuid4
import weakref

from PyQt6 import sip
from PyQt6.QtCore import QObject, QTimer, pyqtSignal
from PyQt6.QtWidgets import QLabel

from kraken_hub.presentation.qt.analysis_map_view import show_analysis_map, show_runtime_report
from kraken_hub.presentation.qt.parts.layer_create_thread import _BackgroundCallThread

_LOCAL_MAP_CALLS = set()


class LocalAnalysisMapController(QObject):
    updated = pyqtSignal()

    def __init__(self, workspace, coordinator, *, language="Русский", background_workers=None):
        super().__init__(workspace)
        self.workspace = weakref.ref(workspace)
        self.coordinator, self.store = coordinator, coordinator.store
        self.language, self.background_workers = language, background_workers
        self._worker, self._pending, self._waiting = None, None, None
        self.status = QLabel()
        self.status.setObjectName("localAnalysisMapStatus")
        workspace.analysis_runs_panel.layout().addWidget(self.status)
        workspace.analysis_runs_panel.runtime_button.setText(self._text("Среда worker", "Worker runtime"))
        workspace.analysis_runs_panel.runtimeRequested.connect(self.runtime_details)
        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.refresh)
        self.timer.start()
        QTimer.singleShot(0, self.refresh)

    def _text(self, ru, en):
        return ru if self.language.lower().startswith(("ru", "рус")) else en

    def _alive(self):
        parent = self.workspace()
        return parent is not None and not sip.isdeleted(parent)

    def _start(self, call, *, queue=False):
        if self._worker is not None:
            if queue and self._pending is None:
                self._pending = call
                return True
            return False
        worker = _BackgroundCallThread(call)
        self._worker = worker
        _LOCAL_MAP_CALLS.add(worker)
        if self.background_workers is not None:
            self.background_workers.add(worker)
        reference = weakref.ref(self)
        background_workers = self.background_workers

        def loaded(value):
            owner = reference()
            if owner is not None and not sip.isdeleted(owner) and owner._alive():
                try:
                    owner._loaded(value)
                except Exception as exc:
                    owner.status.setText(str(exc))

        def failed(message):
            owner = reference()
            if owner is not None and not sip.isdeleted(owner) and owner._alive():
                owner.status.setText(message)

        def finished():
            _LOCAL_MAP_CALLS.discard(worker)
            if background_workers is not None:
                background_workers.discard(worker)
            owner = reference()
            if owner is not None and not sip.isdeleted(owner) and owner._alive():
                owner._worker = None
                if owner._pending is not None:
                    call, owner._pending = owner._pending, None
                    owner._start(call)

        worker.succeeded.connect(loaded)
        worker.failed.connect(failed)
        worker.finished.connect(finished)
        worker.finished.connect(worker.deleteLater)
        worker.start()
        return True

    def _cycle(self, waiting, *, request=None, command=None):
        # Capture plain application objects; a running call never accesses Qt.
        coordinator, store = self.coordinator, self.store

        def work():
            if request is not None:
                run_id, frame_id, key = request
                coordinator.request_map(run_id, frame_id, idempotency_key=key)
            if command is not None:
                run_id, operation = command
                method = {"cancel": "cancel", "retry": "retry_failed", "repeat": "repeat"}[operation]
                getattr(coordinator, method)(run_id)
            for run in store.list_runs():
                if run.state in {"queued", "running"}:
                    coordinator.refresh(run.run_id)
                else:
                    coordinator.repair_cancellations(run.run_id)
                coordinator.refresh_maps(run.run_id)
            if waiting is None:
                return None
            run_id, frame_id = waiting
            task = store.map_tasks(run_id).get(frame_id)
            data = store.map_bytes(run_id, frame_id) if task and task["state"] == "ready" else None
            run = store.get_run(run_id)
            return waiting, task, data, run.manifest.recipe.expression.to_payload() if run else {}

        return work

    def render(self, run_id, frame_id):
        if not run_id or not frame_id:
            self.status.setText(self._text("Выберите прогон и кадр", "Select a run and frame"))
            return
        waiting = (run_id, frame_id)
        if self._start(self._cycle(waiting, request=(run_id, frame_id, str(uuid4()))), queue=True):
            self._waiting = waiting
        else:
            self.status.setText(self._text("Запрос уже ожидает выполнения", "A request is already pending"))

    def command(self, run_id, operation):
        if run_id:
            self._start(self._cycle(self._waiting, command=(run_id, operation)), queue=True)

    def refresh(self):
        if self._worker is None:
            self._start(self._cycle(self._waiting))

    def runtime_details(self, run_id):
        if run_id:
            store = self.store
            self._start(lambda: {"runtime_report": store.runtime_details(run_id)}, queue=True)

    def _loaded(self, value):
        self.updated.emit()
        if isinstance(value, dict) and "runtime_report" in value:
            show_runtime_report(self.workspace().analysis_runs_panel, value["runtime_report"], language=self.language)
            return
        if value is None or value[0] != self._waiting:
            return
        waiting, task, data, expression = value
        if task and task["state"] == "ready":
            self._waiting = None
            sources = task["manifest"]["frames"][0]["artifacts"]
            show_analysis_map(self.workspace().analysis_runs_panel, *waiting, data, expression=expression,
                              language=self.language, sources=sources, runtime=task.get("result", {}).get("worker_runtime"),
                              object_name="localAnalysisMapDialog")
            self.status.setText(self._text("Карта готова", "Map ready"))
        elif task and task["state"] in {"failed", "cancelled"}:
            self._waiting = None
            self.status.setText(task.get("error") or self._text("Построение карты отменено", "Map rendering cancelled"))
        else:
            self.status.setText(self._text("Карта выполняется через Agent", "Rendering map through Agent"))
