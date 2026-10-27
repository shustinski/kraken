"""qt: apply viewport result."""
from __future__ import annotations
from kraken_core.frame_matrix.models import (
    MatrixViewportResult,
)
from kraken_core.frame_matrix.storage import ThumbnailKey

from kraken_core.frame_matrix.qt_parts.viewport_tile_item import FrameCellData


class ApplyViewportResultMixin:
    def _apply_viewport_result(self, result: MatrixViewportResult) -> None:
        # Completed thumbnail tasks from the previous viewport may still populate
        # the shared cache, but must not materialize off-screen cells again.
        self._asset_futures.clear()
        cells = [
            FrameCellData(
                item.x,
                item.y,
                status=item.status,
                performer_color=str(item.metadata.get("performer_color") or "") or None,
                performer_initials=str(item.metadata.get("performer_initials") or ""),
                label=item.label,
                tooltip=item.tooltip,
                payload={
                    "key": item.key,
                    "request_lod": result.request.lod,
                    **dict(item.metadata),
                },
            )
            for item in result.items
        ]
        self.set_cells(cells)
        self.set_aggregates(result.aggregates)
        if self._thumbnail_cache is None or self._asset_source is None:
            return
        asset_source = self._asset_source
        for item in result.items:
            if item.asset is None:
                continue
            key = ThumbnailKey(
                source_key=item.asset.source_key,
                source_revision=item.asset.source_revision,
                lod=result.request.lod,
                width=result.request.asset_width,
                height=result.request.asset_height,
                codec="png",
            )
            future = self._thumbnail_cache.request(
                key,
                lambda reference=item.asset, request=result.request: asset_source.load_asset(
                    reference,
                    width=request.asset_width,
                    height=request.asset_height,
                    cancellation=None,
                ),
                priority=0 if self.selection().contains(item.x, item.y) else 10,
                generation=self._thumbnail_cache.generation,
            )
            self._asset_futures[(item.x, item.y)] = (self._viewport_generation, future)

    def _set_loading(self, loading: bool) -> None:
        normalized = bool(loading)
        if normalized == self._loading:
            return
        self._loading = normalized
        self.loadingChanged.emit(normalized)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._viewport_generation += 1
        self._viewport_request_timer.stop()
        self._result_poll_timer.stop()
        self._background_resources.close()
        self._thumbnail_cache = None
        self._asset_futures.clear()
        self._viewport_future = None
        super().closeEvent(event)
