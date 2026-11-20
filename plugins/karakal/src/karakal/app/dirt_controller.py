"""Dirty regions in the cell check: which layers, looking for dirt in a thread, the results by
layer and frame, the «Dirt» tab, the matrix marks and the frame windows.

Dirt is read from each layer's masks and confidence maps alone, so it runs on its own: together
with the cell check when «Find dirt with the cell check» is on, or on «Find dirt» at any time.
Frames show up in the tab and on the matrix as they are found, not only at the end.
"""

from __future__ import annotations

import logging
import os

import numpy as np
from PyQt6.QtCore import QObject, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPixmap

from ..core.dirt_batch import DirtCancelled, DirtFrame, analyze_dirt_frames
from ..core.dirt_regions import (
    LEVEL_HEAVY,
    LEVEL_LIGHT,
    LEVEL_MEDIUM,
    DirtConfig,
    FrameDirtResult,
    dirt_layers,
    level_at_least,
    read_gray,
    render_dirt_view,
    worst_level,
)

_LOGGER = logging.getLogger(__name__)

# Corner marks on the matrix, by the frame's dirtiest region.
LEVEL_COLORS = {
    LEVEL_LIGHT: QColor(255, 214, 90),
    LEVEL_MEDIUM: QColor(255, 140, 0),
    LEVEL_HEAVY: QColor(235, 40, 40),
}


# Frames found while the search runs are shown at most this often (ms).
PARTIAL_REFRESH_MS = 1200
PREVIEW_SIDE_PX = 900


class _DirtThread(QThread):
    progressed = pyqtSignal(int, int)
    partial = pyqtSignal(str, object)
    found = pyqtSignal(object)
    failed = pyqtSignal(str)
    stopped = pyqtSignal()

    def __init__(
        self,
        frames_by_model: dict[str, list[DirtFrame]],
        config: DirtConfig,
        use_processes: bool,
        workers: int | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._frames_by_model = frames_by_model
        self._config = config
        self._use_processes = use_processes
        self._workers = workers
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
                    workers=self._workers,
                    partial=lambda found, model_id=model_id: self.partial.emit(model_id, found),
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
        self._progress = (0, 0)
        # The search started with the cell check of this run (not again when the check finishes).
        self._started_with_run = False
        # KARAKAL_DIRT_PROCESSES=0 keeps the work in threads (tests: no process pools after every run).
        self.use_processes = os.environ.get("KARAKAL_DIRT_PROCESSES", "1") != "0"
        self._partial_timer = QTimer(self)
        self._partial_timer.setSingleShot(True)
        self._partial_timer.setInterval(PARTIAL_REFRESH_MS)
        self._partial_timer.timeout.connect(self._refresh_partial)
        controls.runRequested.connect(lambda: self.run())
        controls.cancelRequested.connect(self.cancel)
        controls.settingsChanged.connect(self._settings_changed)
        controls.filterChanged.connect(self._filter_changed)
        controls.frameChosen.connect(self.choose_frame)
        controls.frameOpened.connect(self.open_frame)
        show = getattr(controls, "showOnMatrixRequested", None)
        if show is not None:
            show.connect(self.show_on_matrix)
        self._update_title()

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

    def _filter_changed(self) -> None:
        self.refresh_views()
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

    def analysis_started(self) -> None:
        """The cell check starts: dirt is looked for at the same time (it does not need the cells),
        with half the cores, so both finish sooner than one after the other."""

        self._started_with_run = False
        if self.enabled() and self._thread is None:
            self._started_with_run = self.run(workers=max(1, (os.cpu_count() or 2) // 2))

    def analysis_finished(self) -> None:
        """The cell check is done: dirt too, unless it was already looked for along with it."""

        started, self._started_with_run = self._started_with_run, False
        if self.enabled() and not started and self._thread is None:
            self.run()

    def run(self, workers: int | None = None) -> bool:
        if self._thread is not None:
            return False
        state = self._presenter._current_tab_state()
        if state is None:
            self._controls.set_status("dirt.no_masks")
            return False
        frames_by_model = self.frames(state)
        if not frames_by_model:
            self._controls.set_status("dirt.no_masks")
            return False
        config = self.config()
        total = sum(len(frames) for frames in frames_by_model.values())
        if config.signature() != self._result_signature:
            # Other settings: the regions found before are not these; frames come in as found.
            self._results = {}
        self._result_signature = config.signature()
        self._progress = (0, total)
        self._controls.set_running(True)
        self._controls.set_status("dirt.running", done=0, total=total)
        thread = _DirtThread(frames_by_model, config, self.use_processes, workers, self)
        thread.progressed.connect(self._progressed)
        thread.partial.connect(self._partial)
        thread.found.connect(lambda results, signature=config.signature(): self._found(results, signature))
        thread.failed.connect(lambda message: self._controls.set_status("dirt.failed", error=message))
        thread.stopped.connect(lambda: self._controls.set_status("dirt.cancelled"))
        thread.finished.connect(self._thread_done)
        self._thread = thread
        self._update_title()
        thread.start()
        return True

    def _progressed(self, done: int, total: int) -> None:
        self._progress = (int(done), int(total))
        self._controls.set_status("dirt.running", done=done, total=total)
        self._update_title()

    def _partial(self, model_id: str, found: dict) -> None:
        self._results.setdefault(str(model_id), {}).update(found)
        if not self._partial_timer.isActive():
            self._partial_timer.start()

    def cancel(self) -> None:
        if self._thread is not None:
            self._thread.cancel = True

    def _thread_done(self) -> None:
        thread = self._thread
        self._thread = None
        self._controls.set_running(False)
        self._partial_timer.stop()
        self._partial_timer.setInterval(PARTIAL_REFRESH_MS)
        self._update_title()
        if thread is not None:
            thread.deleteLater()

    def _found(self, results: dict, signature: tuple) -> None:
        self._results = {str(model_id): dict(frames) for model_id, frames in dict(results).items()}
        self._result_signature = signature
        self._partial_timer.stop()
        self._show_summary()
        self.refresh_views()
        self.resultsChanged.emit()
        project = getattr(self._presenter, "_project_controller", None)
        if project is not None:
            project.results_changed()

    def export_state(self) -> dict:
        """What a project keeps: the regions by layer and frame, and the settings they were found with."""

        return {"results": {model_id: dict(frames) for model_id, frames in self._results.items()}, "signature": self._result_signature}

    def import_state(self, payload: dict) -> None:
        """Regions of an opened project, shown as if just found."""

        self._results = {str(model_id): dict(frames) for model_id, frames in dict(payload.get("results") or {}).items()}
        self._result_signature = tuple(payload.get("signature") or ())
        if self._results:
            self._show_summary()
        else:
            self._controls.set_status("dirt.summary_empty")
        self.refresh_views()
        self.resultsChanged.emit()

    def shutdown(self) -> None:
        thread = self._thread
        if thread is None:
            return
        thread.cancel = True
        for signal in (thread.found, thread.failed, thread.stopped, thread.progressed, thread.partial):
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

    def frame_levels(self, model_id: str | None = None) -> dict[str, str]:
        """Worst dirt level of every dirty frame of the layer (the shown one by default)."""

        chosen = self.shown_model_id() if model_id is None else model_id
        levels = {}
        for key, result in self.results_for(chosen).items():
            level = worst_level(result)
            if level:
                levels[key] = level
        return levels

    def listed_frames(self) -> list[tuple[str, FrameDirtResult]]:
        """Dirty frames of the shown layer at the chosen level or dirtier, dirtiest first."""

        minimum = self._controls.filter_level()
        rows = [
            (key, result)
            for key, result in self.results_for(self.shown_model_id()).items()
            if level_at_least(worst_level(result), minimum)
        ]
        rows.sort(key=lambda row: (-row[1].max_dirtiness, -len(row[1].regions), row[0]))
        return rows

    def _update_title(self) -> None:
        """Name of the «Dirt» tab: the search's progress, else how many frames are listed."""

        t = self._presenter._t
        if self._thread is not None:
            done, total = self._progress
            title = t("dirt.tab_running", percent=int(100 * done / total) if total else 0)
        elif self._results:
            title = t("dirt.tab_count", count=len(self.listed_frames()))
        else:
            title = t("dirt.tab")
        setter = getattr(self._controls, "set_title", None)
        if callable(setter):
            setter(title)

    def _record(self, key: str):
        state = self._presenter._current_tab_state()
        if state is None:
            return None, None
        record = next((entry for entry in state.build_result.records if str(entry.key) == str(key)), None)
        return state, record

    def choose_frame(self, key: str) -> None:
        """The frame chosen in the tab: its picture in the preview, and selected on the matrix."""

        _state, record = self._record(key)
        view = getattr(self._presenter, "grid_inspection_matrix_view", None)
        if record is not None and view is not None:
            view.select_record_by_key(record.key, ensure_visible=True)
        self._show_preview(key)

    def _show_preview(self, key: str) -> None:
        setter = getattr(self._controls, "set_preview", None)
        if not callable(setter):
            return
        result = self.result_for(self.shown_model_id(), key)
        if result is None or not result.mask_path:
            setter(None, "")
            return
        mask = read_gray(result.mask_path)
        if mask is None:
            setter(None, self._presenter._t("dirt.preview_missing"))
            return
        confidence = read_gray(result.confidence_path) if result.used_confidence else None
        if confidence is not None and confidence.shape != mask.shape:
            confidence = None
        view = render_dirt_view(mask, result, dirt_layers(mask, confidence, self.config()))
        scale = min(1.0, PREVIEW_SIDE_PX / float(max(view.shape[:2])))
        if scale < 1.0:
            import cv2

            view = cv2.resize(view, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        rgb = np.ascontiguousarray(view[:, :, ::-1])
        image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], int(rgb.strides[0]), QImage.Format.Format_RGB888).copy()
        level = worst_level(result)
        caption = self._presenter._t(
            "dirt.preview_caption",
            frame=key,
            level=self._presenter._t(f"dirt.level.{level}") if level else "",
            dirtiness=int(round(result.max_dirtiness * 100)),
            count=len(result.regions),
        )
        setter(QPixmap.fromImage(image), caption)

    def show_on_matrix(self, key: str) -> None:
        """Back to the matrix with only the dirty frames bright, the chosen one selected."""

        check = getattr(self._controls, "filter_check", None)
        if check is not None and not check.isChecked():
            check.setChecked(True)
        tabs = getattr(self._presenter._view, "grid_inspection_content_tabs", None)
        if tabs is not None:
            tabs.setCurrentIndex(0)
        if key:
            _state, record = self._record(key)
            view = getattr(self._presenter, "grid_inspection_matrix_view", None)
            if record is not None and view is not None:
                view.select_record_by_key(record.key, ensure_visible=True)

    def open_frame(self, key: str) -> None:
        state, record = self._record(key)
        if record is None:
            return
        self.choose_frame(key)
        self._presenter._open_record_details(record, state)

    def frame_marks(self, model_id: str | None) -> dict[str, QColor]:
        marks: dict[str, QColor] = {}
        for key, result in self.results_for(model_id).items():
            if not result.regions:
                continue
            worst = max(result.regions, key=lambda region: region.dirtiness)
            marks[key] = QColor(LEVEL_COLORS.get(worst.level, LEVEL_COLORS[LEVEL_LIGHT]))
        return marks

    def _refresh_partial(self) -> None:
        """Views refreshed while the search runs. Each refresh lists every dirty frame found so far;
        the next one waits ten times as long as it took (a big run stays responsive, the end of the
        run refreshes everything anyway)."""

        from time import perf_counter

        started = perf_counter()
        self.refresh_views()
        spent_ms = 1000.0 * (perf_counter() - started)
        self._partial_timer.setInterval(max(PARTIAL_REFRESH_MS, int(10.0 * spent_ms)))

    def refresh_views(self) -> None:
        """Matrix marks of the shown layer and the frame windows that are open."""

        if self._results:
            self._show_summary()
        model_id = self.shown_model_id()
        marks = self.frame_marks(model_id)
        listed = self.listed_frames()
        self._controls.set_frames(
            [
                {"key": key, "level": worst_level(result), "dirtiness": result.max_dirtiness, "regions": len(result.regions)}
                for key, result in listed
            ]
        )
        self._update_title()
        shown = {key for key, _result in listed} if self._controls.filter_on() and self._results else None
        for view in self._presenter._grid_inspection_views().values():
            setter = getattr(view, "set_dirt_marks", None)
            if callable(setter):
                setter(marks)
            filter_setter = getattr(view, "set_dirt_filter_keys", None)
            if callable(filter_setter):
                filter_setter(shown)
        frame_sets = getattr(self._presenter, "_frame_sets_controller", None)
        if frame_sets is not None:
            frame_sets.schedule_refresh()
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
