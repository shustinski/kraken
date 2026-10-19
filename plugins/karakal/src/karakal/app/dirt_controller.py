"""Dirty regions in the cell check: which layers, looking for dirt in a thread, the results by
layer and frame, the matrix marks and the frame windows.

Dirt is read from each layer's masks and confidence maps alone, so it runs on its own: after the
cell check when «Find dirt after the cell check» is on, or on «Find dirt» at any time.
"""

from __future__ import annotations

import logging

from PyQt6.QtCore import QObject, QThread, pyqtSignal
from PyQt6.QtGui import QColor

from ..core.dirt_batch import DirtCancelled, DirtFrame, analyze_dirt_frames
from ..core.dirt_regions import LEVEL_HEAVY, LEVEL_LIGHT, LEVEL_MEDIUM, DirtConfig, FrameDirtResult

_LOGGER = logging.getLogger(__name__)

# Corner marks on the matrix, by the frame's dirtiest region.
LEVEL_COLORS = {
    LEVEL_LIGHT: QColor(255, 214, 90),
    LEVEL_MEDIUM: QColor(255, 140, 0),
    LEVEL_HEAVY: QColor(235, 40, 40),
}


class _DirtThread(QThread):
    progressed = pyqtSignal(int, int)
    found = pyqtSignal(object)
    failed = pyqtSignal(str)
    stopped = pyqtSignal()

    def __init__(self, frames_by_model: dict[str, list[DirtFrame]], config: DirtConfig, use_processes: bool, parent=None) -> None:
        super().__init__(parent)
        self._frames_by_model = frames_by_model
        self._config = config
        self._use_processes = use_processes
        self.cancel = False

    def run(self) -> None:
        total = sum(len(frames) for frames in self._frames_by_model.values())
        done_before = 0
        results: dict[str, dict[str, FrameDirtResult]] = {}
        try:
            for model_id, frames in self._frames_by_model.items():
                offset = done_before
                results[model_id] = analyze_dirt_frames(
                    frames,
                    self._config,
                    progress=lambda done, _count, offset=offset: self.progressed.emit(offset + int(done), total),
                    cancelled=lambda: self.cancel,
                    use_processes=self._use_processes,
                    # Inside the app a process pool that does not start soon is given up for threads.
                    startup_timeout=45.0,
                )
                done_before += len(frames)
            self.found.emit(results)
        except DirtCancelled:
            self.stopped.emit()
        except Exception as error:  # shown to the operator
            _LOGGER.exception("Looking for dirt failed")
            self.failed.emit(str(error))


class DirtController(QObject):
    resultsChanged = pyqtSignal()

    def __init__(self, presenter, controls) -> None:
        super().__init__(presenter)
        self._presenter = presenter
        self._controls = controls
        self._thread: _DirtThread | None = None
        self._results: dict[str, dict[str, FrameDirtResult]] = {}
        self._result_signature: tuple = ()
        self.use_processes = True
        controls.runRequested.connect(self.run)
        controls.cancelRequested.connect(self.cancel)
        controls.settingsChanged.connect(self._settings_changed)

    # Settings ---------------------------------------------------------------------------------

    def config(self) -> DirtConfig:
        return self._controls.config()

    def enabled(self) -> bool:
        return self._controls.enabled()

    def settings_payload(self) -> dict:
        return self._controls.settings_payload()

    def apply_settings(self, payload: dict | None) -> None:
        self._controls.apply_settings(payload)

    def _settings_changed(self) -> None:
        if self._results and self.config().signature() != self._result_signature:
            self._controls.set_status("dirt.stale")
        self._presenter._persist_state()

    # Running ----------------------------------------------------------------------------------

    def is_running(self) -> bool:
        return self._thread is not None

    def _model_ids(self, state, records) -> tuple[str, ...]:
        ids = list(self._presenter._grid_inspection_multi_model_ids(state, records))
        shown = self._presenter._grid_inspection_model_id_for_state(state)
        if shown and shown not in ids:
            ids.append(str(shown))
        return tuple(ids)

    def frames(self, state) -> dict[str, list[DirtFrame]]:
        """Masks (and confidence maps) of every analysed layer, frames of the run not dropped."""

        excluded = self._presenter._excluded_record_keys_for_state(state)
        records = tuple(
            record
            for record in tuple(state.build_result.records or ())
            if str(getattr(record, "key", "") or "") not in excluded
        )
        frames_by_model: dict[str, list[DirtFrame]] = {}
        for model_id in self._model_ids(state, records):
            frames = []
            for record in records:
                mask = str((record.model_mask_paths or {}).get(model_id) or "")
                if not mask:
                    continue
                confidence = str((record.model_prob_paths or {}).get(model_id) or "")
                frames.append(DirtFrame(str(record.key), mask, confidence))
            if frames:
                frames_by_model[model_id] = frames
        return frames_by_model

    def analysis_finished(self) -> None:
        """The cell check is done: look for dirt too when the operator asked for it."""

        if self.enabled():
            self.run()

    def run(self) -> None:
        if self._thread is not None:
            return
        state = self._presenter._current_tab_state()
        if state is None:
            self._controls.set_status("dirt.no_masks")
            return
        frames_by_model = self.frames(state)
        if not frames_by_model:
            self._controls.set_status("dirt.no_masks")
            return
        config = self.config()
        total = sum(len(frames) for frames in frames_by_model.values())
        self._controls.set_running(True)
        self._controls.set_status("dirt.running", done=0, total=total)
        thread = _DirtThread(frames_by_model, config, self.use_processes, self)
        thread.progressed.connect(lambda done, count: self._controls.set_status("dirt.running", done=done, total=count))
        thread.found.connect(lambda results, signature=config.signature(): self._found(results, signature))
        thread.failed.connect(lambda message: self._controls.set_status("dirt.failed", error=message))
        thread.stopped.connect(lambda: self._controls.set_status("dirt.cancelled"))
        thread.finished.connect(self._thread_done)
        self._thread = thread
        thread.start()

    def cancel(self) -> None:
        if self._thread is not None:
            self._thread.cancel = True

    def _thread_done(self) -> None:
        thread = self._thread
        self._thread = None
        self._controls.set_running(False)
        if thread is not None:
            thread.deleteLater()

    def _found(self, results: dict, signature: tuple) -> None:
        self._results = {str(model_id): dict(frames) for model_id, frames in dict(results).items()}
        self._result_signature = signature
        self._show_summary()
        self.refresh_views()
        self.resultsChanged.emit()

    def shutdown(self) -> None:
        thread = self._thread
        if thread is None:
            return
        thread.cancel = True
        for signal in (thread.found, thread.failed, thread.stopped, thread.progressed):
            try:
                signal.disconnect()
            except TypeError:
                pass
        thread.wait(5000)

    # Results ----------------------------------------------------------------------------------

    def has_results(self) -> bool:
        return bool(self._results)

    def results_for(self, model_id: str | None) -> dict[str, FrameDirtResult]:
        return dict(self._results.get(str(model_id or ""), {}))

    def result_for(self, model_id: str | None, key: str | None) -> FrameDirtResult | None:
        return self._results.get(str(model_id or ""), {}).get(str(key or ""))

    def shown_model_id(self) -> str:
        state = self._presenter._current_tab_state()
        if state is None:
            return ""
        shown = self._presenter._matrix_layer_model_id(state)
        if shown and shown in self._results:
            return shown
        fallback = self._presenter._grid_inspection_model_id_for_state(state)
        if fallback and str(fallback) in self._results:
            return str(fallback)
        return next(iter(self._results), "")

    def _show_summary(self) -> None:
        results = self.results_for(self.shown_model_id())
        total = len(results)
        dirty = [result for result in results.values() if result.regions]
        if not dirty:
            self._controls.set_status("dirt.summary_none", total=total)
            return
        levels = {LEVEL_LIGHT: 0, LEVEL_MEDIUM: 0, LEVEL_HEAVY: 0}
        regions = 0
        for result in dirty:
            for region in result.regions:
                regions += 1
                levels[region.level] = levels.get(region.level, 0) + 1
        self._controls.set_status(
            "dirt.summary",
            frames=len(dirty),
            total=total,
            regions=regions,
            heavy=levels[LEVEL_HEAVY],
            medium=levels[LEVEL_MEDIUM],
            light=levels[LEVEL_LIGHT],
        )

    def frame_marks(self, model_id: str | None) -> dict[str, QColor]:
        marks: dict[str, QColor] = {}
        for key, result in self.results_for(model_id).items():
            if not result.regions:
                continue
            worst = max(result.regions, key=lambda region: region.dirtiness)
            marks[key] = QColor(LEVEL_COLORS.get(worst.level, LEVEL_COLORS[LEVEL_LIGHT]))
        return marks

    def refresh_views(self) -> None:
        """Matrix marks of the shown layer and the frame windows that are open."""

        if self._results:
            self._show_summary()
        model_id = self.shown_model_id()
        marks = self.frame_marks(model_id)
        for view in self._presenter._grid_inspection_views().values():
            setter = getattr(view, "set_dirt_marks", None)
            if callable(setter):
                setter(marks)
        for dialog in list(getattr(self._presenter, "_details_dialogs", ()) or ()):
            self.update_dialog(dialog)

    def update_dialog(self, dialog) -> None:
        setter = getattr(dialog, "set_dirt_result", None)
        if not callable(setter):
            return
        record = getattr(dialog, "_record", None)
        key = str(getattr(record, "key", "") or "")
        if not self._results:
            setter(None, known=False)
            return
        model_id = self.shown_model_id()
        setter(self.result_for(model_id, key), known=key in self._results.get(model_id, {}), config=self.config())
