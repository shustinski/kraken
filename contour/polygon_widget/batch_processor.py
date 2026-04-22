from __future__ import annotations

import multiprocessing as mp
from dataclasses import replace
from queue import Empty
from typing import Iterable

import cv2
from PyQt6.QtCore import QObject, QRunnable, QThreadPool, pyqtSignal

from .application.processing import BatchImageResult, ContourExtractionSettings, DisplaySettings, SaveOptions
from .application.use_cases.processing import process_image_path as run_image_processing
from .i18n import active_language, tr
from .pipeline import PreprocessingPipeline


def process_image_path(
    image_path: str,
    pipeline_config: dict,
    contour_settings: ContourExtractionSettings,
    output_directory: str | None = None,
    save_options: SaveOptions | None = None,
    display_settings: DisplaySettings | None = None,
    pipeline: PreprocessingPipeline | None = None,
) -> BatchImageResult:
    return run_image_processing(
        image_path=image_path,
        pipeline_config=pipeline_config,
        contour_settings=contour_settings,
        output_directory=output_directory,
        save_options=save_options,
        display_settings=display_settings,
        pipeline=pipeline,
        include_images_in_result=False,
    )


class BatchQueueSignals(QObject):
    result = pyqtSignal(object)
    error = pyqtSignal(str, str)
    progress = pyqtSignal(int, int)
    finished = pyqtSignal()
    log = pyqtSignal(str)


def _batch_process_worker(
    file_queue,
    result_queue,
    cancel_event,
    pipeline_config: dict,
    contour_settings: ContourExtractionSettings,
    output_directory: str | None,
    save_options: SaveOptions,
    display_settings: DisplaySettings,
    ui_language: str,
) -> None:
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass

    pipeline = PreprocessingPipeline.from_dict(pipeline_config)
    while not cancel_event.is_set():
        try:
            image_path = file_queue.get_nowait()
        except Empty:
            break

        result_queue.put(("log", tr("processing_log", language=ui_language, image_path=image_path)))
        try:
            result = process_image_path(
                image_path=image_path,
                pipeline_config=pipeline_config,
                contour_settings=contour_settings,
                output_directory=output_directory,
                save_options=save_options,
                display_settings=display_settings,
                pipeline=pipeline,
            )
            result_queue.put(("result", result))
        except Exception as exc:
            result_queue.put(("error", image_path, str(exc)))
        finally:
            result_queue.put(("progress", image_path))

    result_queue.put(("worker_done", mp.current_process().pid))


class BatchQueueRunnable(QRunnable):
    def __init__(
        self,
        image_paths: list[str],
        pipeline_config: dict,
        contour_settings: ContourExtractionSettings,
        output_directory: str | None,
        save_options: SaveOptions,
        display_settings: DisplaySettings,
        max_workers: int,
        ui_language: str,
    ) -> None:
        super().__init__()
        self.image_paths = list(image_paths)
        self.pipeline_config = dict(pipeline_config)
        self.contour_settings = replace(contour_settings)
        self.output_directory = output_directory
        self.save_options = replace(save_options)
        self.display_settings = replace(display_settings)
        self.max_workers = max(1, int(max_workers))
        self.ui_language = active_language(ui_language)
        self.signals = BatchQueueSignals()
        self._cancel_event = None
        self._stop_requested = False

    def stop(self) -> None:
        self._stop_requested = True
        if self._cancel_event is not None:
            self._cancel_event.set()

    def run(self) -> None:
        context = mp.get_context("spawn")
        file_queue = context.Queue()
        result_queue = context.Queue()
        self._cancel_event = context.Event()
        if self._stop_requested:
            self._cancel_event.set()

        processes: list[mp.Process] = []
        completed = 0
        total = len(self.image_paths)
        finished_processes: set[int] = set()

        try:
            for image_path in self.image_paths:
                file_queue.put(image_path)

            worker_count = min(self.max_workers, max(1, total))
            for _index in range(worker_count):
                process = context.Process(
                    target=_batch_process_worker,
                    args=(
                        file_queue,
                        result_queue,
                        self._cancel_event,
                        self.pipeline_config,
                        self.contour_settings,
                        self.output_directory,
                        self.save_options,
                        self.display_settings,
                        self.ui_language,
                    ),
                )
                process.start()
                processes.append(process)

            while len(finished_processes) < len(processes):
                try:
                    message = result_queue.get(timeout=0.1)
                except Empty:
                    message = None

                if message is not None:
                    kind = message[0]
                    if kind == "log":
                        self.signals.log.emit(message[1])
                    elif kind == "result":
                        self.signals.result.emit(message[1])
                    elif kind == "error":
                        self.signals.error.emit(message[1], message[2])
                    elif kind == "progress":
                        completed += 1
                        self.signals.progress.emit(completed, total)
                    elif kind == "worker_done":
                        pid = message[1]
                        if pid is not None:
                            finished_processes.add(int(pid))

                for process in processes:
                    pid = process.pid
                    if pid is None or int(pid) in finished_processes or process.is_alive():
                        continue
                    if process.exitcode not in (0, None):
                        self.signals.error.emit("", f"Worker process {pid} exited with code {process.exitcode}")
                    finished_processes.add(int(pid))

            while True:
                try:
                    message = result_queue.get_nowait()
                except Empty:
                    break
                kind = message[0]
                if kind == "log":
                    self.signals.log.emit(message[1])
                elif kind == "result":
                    self.signals.result.emit(message[1])
                elif kind == "error":
                    self.signals.error.emit(message[1], message[2])
                elif kind == "progress":
                    completed += 1
                    self.signals.progress.emit(completed, total)
        except Exception as exc:
            self.signals.error.emit("", str(exc))
            self.stop()
        finally:
            for process in processes:
                process.join()
            try:
                file_queue.close()
                result_queue.close()
            except Exception:
                pass
            self.signals.finished.emit()


class BatchProcessor(QObject):
    resultReady = pyqtSignal(object)
    progressChanged = pyqtSignal(int, int)
    finished = pyqtSignal()
    logMessage = pyqtSignal(str)
    errorOccurred = pyqtSignal(str, str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._thread_pool = QThreadPool(self)
        self._thread_pool.setExpiryTimeout(-1)
        self._results: dict[str, BatchImageResult] = {}
        self._completed = 0
        self._total = 0
        self._cancel_requested = False
        self._max_workers = 1
        self._pipeline_config: dict = {}
        self._contour_settings = ContourExtractionSettings()
        self._output_directory: str | None = None
        self._save_options = SaveOptions()
        self._display_settings = DisplaySettings()
        self._ui_language = active_language()
        self._running = False
        self._runner: BatchQueueRunnable | None = None

    @property
    def is_running(self) -> bool:
        return self._running

    def set_ui_language(self, language: str | None) -> None:
        self._ui_language = active_language(language)

    def start(
        self,
        image_paths: Iterable[str],
        pipeline_config: dict,
        contour_settings: ContourExtractionSettings,
        output_directory: str | None,
        save_options: SaveOptions,
        display_settings: DisplaySettings,
        max_workers: int = 4,
    ) -> None:
        if self._running:
            raise RuntimeError(tr("batch_already_running_log", language=self._ui_language))

        image_list = [path for path in image_paths]
        self._results = {}
        self._completed = 0
        self._total = len(image_list)
        self._cancel_requested = False
        self._max_workers = max(1, int(max_workers))
        self._pipeline_config = dict(pipeline_config)
        self._contour_settings = replace(contour_settings)
        self._output_directory = output_directory
        self._save_options = replace(save_options)
        self._display_settings = replace(display_settings)
        self._running = True
        self._thread_pool.setMaxThreadCount(1)

        if self._total == 0:
            self.logMessage.emit(tr("batch_skipped_no_images_log", language=self._ui_language))
            self._running = False
            self.finished.emit()
            return

        self.logMessage.emit(tr("batch_started_log", language=self._ui_language, count=self._total))
        runner = BatchQueueRunnable(
            image_paths=image_list,
            pipeline_config=self._pipeline_config,
            contour_settings=self._contour_settings,
            output_directory=self._output_directory,
            save_options=self._save_options,
            display_settings=self._display_settings,
            max_workers=self._max_workers,
            ui_language=self._ui_language,
        )
        runner.signals.result.connect(self._on_worker_result)
        runner.signals.error.connect(self._on_worker_error)
        runner.signals.progress.connect(self._on_worker_progress)
        runner.signals.log.connect(self.logMessage.emit)
        runner.signals.finished.connect(self._finish)
        self._runner = runner
        self._thread_pool.start(runner)

    def stop(self) -> None:
        if not self._running:
            return
        self._cancel_requested = True
        if self._runner is not None:
            self._runner.stop()
        self.logMessage.emit(tr("batch_stop_requested_log", language=self._ui_language))

    def results(self) -> dict[str, BatchImageResult]:
        return dict(self._results)

    def _on_worker_result(self, result: BatchImageResult) -> None:
        self._results[result.image_path] = result
        self.resultReady.emit(result)

    def _on_worker_error(self, image_path: str, message: str) -> None:
        self.errorOccurred.emit(image_path, message)

    def _on_worker_progress(self, completed: int, total: int) -> None:
        self._completed = completed
        self._total = total
        self.progressChanged.emit(self._completed, self._total)

    def _finish(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._cancel_requested:
            self.logMessage.emit(tr("batch_stopped_log", language=self._ui_language))
        else:
            self.logMessage.emit(tr("batch_finished_log", language=self._ui_language))
        self._runner = None
        self.finished.emit()
