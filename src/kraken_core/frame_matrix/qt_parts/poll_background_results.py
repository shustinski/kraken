"""qt: poll background results."""

from __future__ import annotations
import logging
from queue import Empty
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QPixmap

from kraken_core.frame_matrix.prepared_tiles import MatrixTileNotReady

from kraken_core.frame_matrix import profiling
from kraken_core.frame_matrix.qt_parts.viewport_tile_item import FrameCellData


class PollBackgroundResultsMixin:
    _MAX_THUMBNAILS_PER_POLL = 32
    _matrix_log = logging.getLogger("kraken.matrix")

    def _poll_background_results(self) -> None:
        if profiling.enabled("matrix_ui"):
            with profiling.region("matrix_ui"):
                self._apply_background_results()
            return
        self._apply_background_results()

    def _apply_background_results(self) -> None:
        if self._background_resources.closed:
            return
        while True:
            try:
                error = self._background_resources.errors.get_nowait()
            except Empty:
                break
            self._report_matrix_error(error)
        future = self._viewport_future
        if future is not None and future.done():
            self._viewport_future = None
            completed_signature = self._viewport_future_signature
            self._viewport_future_signature = None
            try:
                result = future.result()
            except Exception as exc:
                self._report_matrix_error(exc)
            else:
                current_signature = self._request_signature(self._current_viewport_request())
                if result.request.generation == self._viewport_generation and completed_signature == current_signature:
                    self._apply_viewport_result(result)
                    self._last_applied_signature = completed_signature
            if self._viewport_request_deferred:
                self._viewport_request_deferred = False
                self._request_visible_viewport()
            else:
                self._set_loading(False)
        completed_cells = []
        processed = 0
        for coordinate, (generation, asset_future) in tuple(self._asset_futures.items()):
            if processed >= self._MAX_THUMBNAILS_PER_POLL:
                break
            if not asset_future.done():
                continue
            processed += 1
            self._asset_futures.pop(coordinate, None)
            if generation != self._viewport_generation:
                continue
            try:
                record = asset_future.result()
                pixmap = self._cached_thumbnail(record.key)
                if pixmap is None:
                    pixmap = QPixmap()
                    if not pixmap.loadFromData(record.payload):
                        if self._thumbnail_batch_profile is not None:
                            self._thumbnail_batch_failed += 1
                            self._thumbnail_batch_error_types["invalid_image_payload"] = (
                                self._thumbnail_batch_error_types.get("invalid_image_payload", 0) + 1
                            )
                        continue
                    self._remember_thumbnail(record.key, pixmap)
                current = self.cell_data(*coordinate)
                completed_cells.append(
                    FrameCellData(
                        current.x,
                        current.y,
                        status=current.status,
                        performer_color=current.performer_color,
                        performer_initials=current.performer_initials,
                        label=current.label,
                        tooltip=current.tooltip,
                        thumbnail=pixmap,
                        payload=current.payload,
                    )
                )
            except Exception as exc:
                if self._thumbnail_batch_profile is not None:
                    self._thumbnail_batch_failed += 1
                    kind = type(exc).__name__
                    self._thumbnail_batch_error_types[kind] = self._thumbnail_batch_error_types.get(kind, 0) + 1
                self._report_matrix_error(exc)
        if completed_cells:
            self.update_cells(completed_cells)
            if self._thumbnail_batch_profile is not None:
                self._thumbnail_batch_completed += len(completed_cells)
        self._apply_mosaic_results()
        batch = self._thumbnail_batch_profile
        if (
            batch is not None
            and self._viewport_future is None
            and not self._asset_futures
            and not self._mosaic_futures
            and not batch.has_pending
            and not self._viewport_request_timer.isActive()
            and not self._viewport_request_deferred
            and not self._update_scheduled
        ):
            if self._thumbnail_batch_total:
                batch.finish(
                    requested=self._thumbnail_batch_total,
                    completed=self._thumbnail_batch_completed,
                    failed=self._thumbnail_batch_failed,
                    superseded=self._thumbnail_batch_superseded,
                    error_types=self._thumbnail_batch_error_types,
                )
            else:
                batch.cancel()
            self._thumbnail_batch_profile = None
            self._thumbnail_batch_total = 0
            self._thumbnail_batch_completed = 0
            self._thumbnail_batch_failed = 0
            self._thumbnail_batch_superseded = 0
            self._thumbnail_batch_error_types = {}
        if (
            self._viewport_future is None
            and not self._asset_futures
            and not self._mosaic_futures
            and self._thumbnail_batch_profile is None
        ):
            self._result_poll_timer.stop()

    def _report_matrix_error(self, error: BaseException | str) -> None:
        message = str(error)
        if isinstance(error, BaseException):
            self._matrix_log.error("Matrix error: %s", message, exc_info=(type(error), error, error.__traceback__))
        else:
            self._matrix_log.error("Matrix error: %s", message)
        self.errorOccurred.emit(message)

    def _apply_mosaic_results(self) -> None:
        changed = False
        for key, (generation, asset_future) in tuple(self._mosaic_futures.items()):
            if not asset_future.done():
                continue
            self._mosaic_futures.pop(key, None)
            if generation != self._viewport_generation:
                continue
            try:
                record = asset_future.result()
                pixmap = QPixmap()
                if not pixmap.loadFromData(record.payload):
                    if self._thumbnail_batch_profile is not None:
                        self._thumbnail_batch_failed += 1
                        self._thumbnail_batch_error_types["invalid_image_payload"] = (
                            self._thumbnail_batch_error_types.get("invalid_image_payload", 0) + 1
                        )
                    continue
                self._mosaic_images[key] = pixmap
                changed = True
                if self._thumbnail_batch_profile is not None:
                    self._thumbnail_batch_completed += 1
            except MatrixTileNotReady:
                self._schedule_mosaic_retry()
            except Exception as exc:
                if "not prepared" in str(exc):
                    self._schedule_mosaic_retry()
                    continue
                if self._thumbnail_batch_profile is not None:
                    self._thumbnail_batch_failed += 1
                    kind = type(exc).__name__
                    self._thumbnail_batch_error_types[kind] = self._thumbnail_batch_error_types.get(kind, 0) + 1
                self._report_matrix_error(exc)
        if changed:
            self._invalidate_tiles()

    def _schedule_mosaic_retry(self) -> None:
        if getattr(self, "_mosaic_retry_scheduled", False):
            return
        self._mosaic_retry_scheduled = True
        QTimer.singleShot(750, self._retry_mosaics)

    def _retry_mosaics(self) -> None:
        self._mosaic_retry_scheduled = False
        items = getattr(self, "_mosaic_source_items", ())
        if not items or self._mosaic_loader() is None:
            return

        class _Pending:
            def __init__(self, pending, revision: str) -> None:
                self.items = pending
                self.source_revision = revision

        self._request_mosaics(_Pending(items, str(getattr(self, "_mosaic_revision", "") or "")))
        self._result_poll_timer.start()
