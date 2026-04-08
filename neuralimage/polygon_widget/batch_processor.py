from __future__ import annotations

from collections import deque
from dataclasses import replace
from typing import Iterable

from PyQt6.QtCore import QObject, QRunnable, QThreadPool, pyqtSignal

from .contour_extractor import extract_polygons
from .i18n import active_language, tr
from .models import BatchImageResult, ContourExtractionSettings, DisplaySettings, SaveOptions
from .pipeline import PreprocessingPipeline
from .serializers import save_result_bundle
from .signals import BatchWorkerSignals
from .utils import ensure_binary_mask, load_image_grayscale


def process_image_path(
    image_path: str,
    pipeline_config: dict,
    contour_settings: ContourExtractionSettings,
    output_directory: str | None = None,
    save_options: SaveOptions | None = None,
    display_settings: DisplaySettings | None = None,
) -> BatchImageResult:
    pipeline = PreprocessingPipeline.from_dict(pipeline_config)
    source_image = load_image_grayscale(image_path)
    preprocessed = pipeline.apply(source_image)
    mask = ensure_binary_mask(preprocessed)
    polygons = extract_polygons(mask, contour_settings)
    saved_files: dict[str, str] = {}
    if output_directory:
        saved_files = save_result_bundle(
            output_directory=output_directory,
            image_path=image_path,
            polygons=polygons,
            source_image=source_image,
            display_settings=display_settings or DisplaySettings(),
            save_options=save_options or SaveOptions(),
            metadata={
                "contour_settings": contour_settings.to_dict(),
                "pipeline": pipeline_config,
            },
        )
    return BatchImageResult(
        image_path=image_path,
        source_image=source_image,
        preprocessed_image=preprocessed,
        mask_image=mask,
        polygons=polygons,
        saved_files=saved_files,
    )


class ImageProcessingRunnable(QRunnable):
    def __init__(
        self,
        image_path: str,
        pipeline_config: dict,
        contour_settings: ContourExtractionSettings,
        output_directory: str | None,
        save_options: SaveOptions,
        display_settings: DisplaySettings,
        ui_language: str,
    ) -> None:
        super().__init__()
        self.image_path = image_path
        self.pipeline_config = dict(pipeline_config)
        self.contour_settings = replace(contour_settings)
        self.output_directory = output_directory
        self.save_options = replace(save_options)
        self.display_settings = replace(display_settings)
        self.ui_language = active_language(ui_language)
        self.signals = BatchWorkerSignals()

    def run(self) -> None:
        self.signals.log.emit(tr("processing_log", language=self.ui_language, image_path=self.image_path))
        try:
            result = process_image_path(
                image_path=self.image_path,
                pipeline_config=self.pipeline_config,
                contour_settings=self.contour_settings,
                output_directory=self.output_directory,
                save_options=self.save_options,
                display_settings=self.display_settings,
            )
            self.signals.result.emit(result)
        except Exception as exc:
            self.signals.error.emit(self.image_path, str(exc))
        finally:
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
        self._pending_paths: deque[str] = deque()
        self._results: dict[str, BatchImageResult] = {}
        self._completed = 0
        self._active = 0
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
        self._pending_paths = deque(image_list)
        self._results = {}
        self._completed = 0
        self._active = 0
        self._total = len(image_list)
        self._cancel_requested = False
        self._max_workers = max(1, int(max_workers))
        self._pipeline_config = dict(pipeline_config)
        self._contour_settings = replace(contour_settings)
        self._output_directory = output_directory
        self._save_options = replace(save_options)
        self._display_settings = replace(display_settings)
        self._running = True
        self._thread_pool.setMaxThreadCount(self._max_workers)

        if self._total == 0:
            self.logMessage.emit(tr("batch_skipped_no_images_log", language=self._ui_language))
            self._running = False
            self.finished.emit()
            return

        self.logMessage.emit(tr("batch_started_log", language=self._ui_language, count=self._total))
        self._dispatch_next_workers()

    def stop(self) -> None:
        if not self._running:
            return
        self._cancel_requested = True
        self._pending_paths.clear()
        self.logMessage.emit(tr("batch_stop_requested_log", language=self._ui_language))
        if self._active == 0:
            self._finish()

    def results(self) -> dict[str, BatchImageResult]:
        return dict(self._results)

    def _dispatch_next_workers(self) -> None:
        while not self._cancel_requested and self._pending_paths and self._active < self._max_workers:
            image_path = self._pending_paths.popleft()
            worker = ImageProcessingRunnable(
                image_path=image_path,
                pipeline_config=self._pipeline_config,
                contour_settings=self._contour_settings,
                output_directory=self._output_directory,
                save_options=self._save_options,
                display_settings=self._display_settings,
                ui_language=self._ui_language,
            )
            worker.signals.result.connect(self._on_worker_result)
            worker.signals.error.connect(self._on_worker_error)
            worker.signals.log.connect(self.logMessage.emit)
            worker.signals.finished.connect(self._on_worker_finished)
            self._active += 1
            self._thread_pool.start(worker)

        if self._cancel_requested and self._active == 0:
            self._finish()

    def _on_worker_result(self, result: BatchImageResult) -> None:
        self._results[result.image_path] = result
        self.resultReady.emit(result)

    def _on_worker_error(self, image_path: str, message: str) -> None:
        self.errorOccurred.emit(image_path, message)

    def _on_worker_finished(self) -> None:
        self._active = max(0, self._active - 1)
        self._completed += 1
        self.progressChanged.emit(self._completed, self._total)
        if self._completed >= self._total or (self._cancel_requested and self._active == 0 and not self._pending_paths):
            self._finish()
            return
        self._dispatch_next_workers()

    def _finish(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._cancel_requested:
            self.logMessage.emit(tr("batch_stopped_log", language=self._ui_language))
        else:
            self.logMessage.emit(tr("batch_finished_log", language=self._ui_language))
        self.finished.emit()
