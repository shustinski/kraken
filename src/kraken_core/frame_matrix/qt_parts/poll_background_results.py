"""qt: poll background results."""
from __future__ import annotations
from queue import Empty
from PyQt6.QtGui import QPixmap

from kraken_core.frame_matrix.qt_parts.viewport_tile_item import FrameCellData


class PollBackgroundResultsMixin:
    def _poll_background_results(self) -> None:
        if self._background_resources.closed:
            return
        while True:
            try:
                error = self._background_resources.errors.get_nowait()
            except Empty:
                break
            self.errorOccurred.emit(str(error))
        future = self._viewport_future
        if future is not None and future.done():
            self._viewport_future = None
            completed_signature = self._viewport_future_signature
            self._viewport_future_signature = None
            try:
                result = future.result()
            except Exception as exc:
                self.errorOccurred.emit(str(exc))
            else:
                current_signature = self._request_signature(self._current_viewport_request())
                if (
                    result.request.generation == self._viewport_generation
                    and completed_signature == current_signature
                ):
                    self._apply_viewport_result(result)
                    self._last_applied_signature = completed_signature
            if self._viewport_request_deferred:
                self._viewport_request_deferred = False
                self._request_visible_viewport()
            else:
                self._set_loading(False)
        for coordinate, (generation, asset_future) in tuple(self._asset_futures.items()):
            if not asset_future.done():
                continue
            self._asset_futures.pop(coordinate, None)
            if generation != self._viewport_generation:
                continue
            try:
                record = asset_future.result()
                pixmap = QPixmap()
                if not pixmap.loadFromData(record.payload):
                    continue
                current = self.cell_data(*coordinate)
                self.update_cells(
                    (
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
                        ),
                    )
                )
            except Exception as exc:
                self.errorOccurred.emit(str(exc))
        if self._viewport_future is None and not self._asset_futures:
            self._result_poll_timer.stop()
