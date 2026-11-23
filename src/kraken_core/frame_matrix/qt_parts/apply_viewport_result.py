"""qt: apply viewport result."""

from __future__ import annotations
from kraken_core.frame_matrix.models import (
    MatrixViewportResult,
)
from kraken_core.frame_matrix.storage import ThumbnailKey

from kraken_core.frame_matrix.qt_parts.viewport_tile_item import FrameCellData


class ApplyViewportResultMixin:
    def _apply_viewport_result(self, result: MatrixViewportResult) -> None:
        previous_cells = self._cells
        previous_keys = self._cell_thumbnail_keys
        previous_futures = self._asset_futures
        keys: dict[tuple[int, int], ThumbnailKey] = {}
        cells: list[FrameCellData] = []
        for item in result.items:
            coordinate = (item.x, item.y)
            key = None
            thumbnail = None
            if item.asset is not None:
                key = ThumbnailKey(
                    source_key=item.asset.source_key,
                    source_revision=item.asset.source_revision,
                    lod=result.request.lod,
                    width=result.request.asset_width,
                    height=result.request.asset_height,
                    codec="png",
                )
                keys[coordinate] = key
                previous = previous_cells.get(coordinate)
                if previous is not None and previous_keys.get(coordinate) == key:
                    thumbnail = previous.thumbnail
                if thumbnail is None:
                    thumbnail = self._cached_thumbnail(key)
            cells.append(
                FrameCellData(
                    item.x,
                    item.y,
                    status=item.status,
                    performer_color=str(item.metadata.get("performer_color") or "") or None,
                    performer_initials=str(item.metadata.get("performer_initials") or ""),
                    label=item.label,
                    tooltip=item.tooltip,
                    thumbnail=thumbnail,
                    payload={
                        "key": item.key,
                        "request_lod": result.request.lod,
                        **dict(item.metadata),
                    },
                )
            )
        self._asset_futures = {}
        self._cell_thumbnail_keys = keys
        self.set_cells(cells)
        self.set_aggregates(result.aggregates)
        if self._mosaic_loader() is not None:
            self._request_mosaics(result)
            return
        if self._thumbnail_cache is None or self._asset_source is None:
            return
        asset_source = self._asset_source
        batch = self._thumbnail_batch_profile
        retained: set[tuple[int, int]] = set()
        for item in result.items:
            coordinate = (item.x, item.y)
            key = keys.get(coordinate)
            if key is None or self._cells[coordinate].thumbnail is not None:
                continue
            prior = previous_futures.get(coordinate)
            previous = previous_cells.get(coordinate)
            if (
                prior is not None
                and prior[0] == self._viewport_generation
                and previous is not None
                and previous_keys.get(coordinate) == key
            ):
                future = prior[1]
                retained.add(coordinate)
            else:

                def produce(reference=item.asset, request=result.request):
                    def load():
                        return asset_source.load_asset(
                            reference, width=request.asset_width, height=request.asset_height, cancellation=None
                        )

                    return batch.profile_call(load) if batch is not None else load()

                future = self._thumbnail_cache.request(
                    key,
                    produce,
                    priority=0 if self.selection().contains(item.x, item.y) else 10,
                    generation=self._thumbnail_cache.generation,
                )
                if batch is not None:
                    self._thumbnail_batch_total += 1
            self._asset_futures[coordinate] = (self._viewport_generation, future)
            if batch is not None:
                batch.track(future)
        if batch is not None:
            self._thumbnail_batch_superseded += len(previous_futures) - len(retained)

    def _set_loading(self, loading: bool) -> None:
        normalized = bool(loading)
        if normalized == self._loading:
            return
        self._loading = normalized
        self.loadingChanged.emit(normalized)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self._thumbnail_batch_profile is not None:
            self._thumbnail_batch_profile.cancel()
            self._thumbnail_batch_profile = None
        self._viewport_generation += 1
        self._viewport_request_timer.stop()
        self._result_poll_timer.stop()
        self._background_resources.close()
        self._thumbnail_cache = None
        self._asset_futures.clear()
        self._viewport_future = None
        self._mosaic_futures.clear()
        self._mosaic_images.clear()
        super().closeEvent(event)

    def _mosaic_loader(self):
        for candidate in (self._data_source, self._asset_source):
            method = getattr(candidate, "load_mosaic", None)
            if callable(method):
                return method
        return None

    def _request_mosaics(self, result: MatrixViewportResult) -> bool:
        load_mosaic = self._mosaic_loader()
        if load_mosaic is None or self._thumbnail_cache is None:
            return False
        cells = self._effective_tile_cells()
        self._mosaic_source_items = tuple(item for item in result.items if item.asset is not None)
        wanted: set[tuple[int, int]] = set()
        for item in result.items:
            if item.asset is None:
                continue
            wanted.add(((item.x - 1) // cells, self._row_for_y(item.y) // cells))
        if not wanted:
            return False
        revision = result.source_revision or "0"
        orientation = self.orientation().value
        self._mosaic_revision = revision
        previous = self._mosaic_futures
        self._mosaic_futures = {}
        kept = 0
        batch = self._thumbnail_batch_profile
        for tile_x, tile_y in wanted:
            key = (tile_x, tile_y, cells, revision, orientation)
            x1 = tile_x * cells + 1
            x2 = min(self.matrix_width, x1 + cells - 1)
            row1 = tile_y * cells
            row2 = min(self.matrix_height - 1, row1 + cells - 1)
            y_top = self._y_for_row(row1)
            y_bottom = self._y_for_row(row2)
            y1, y2 = min(y_top, y_bottom), max(y_top, y_bottom)
            prior = previous.get(key)
            if prior is not None and prior[0] == self._viewport_generation:
                self._mosaic_futures[key] = prior
                kept += 1
                continue
            cache_key = ThumbnailKey(
                source_key=f"mosaic:{x1}:{y1}:{x2}:{y2}:{orientation}",
                source_revision=revision,
                lod=cells,
                width=256,
                height=256,
                codec="png",
            )

            def produce(x1=x1, y1=y1, x2=x2, y2=y2, orientation=orientation, load_mosaic=load_mosaic):
                def load():
                    return load_mosaic(x1=x1, y1=y1, x2=x2, y2=y2, orientation=orientation)

                return batch.profile_call(load) if batch is not None else load()

            future = self._thumbnail_cache.request(
                cache_key,
                produce,
                priority=10,
                generation=self._thumbnail_cache.generation,
            )
            if batch is not None:
                self._thumbnail_batch_total += 1
                batch.track(future)
            self._mosaic_futures[key] = (self._viewport_generation, future)
        overview = self._overview_tile_cells()
        if (0, 0, overview, revision, orientation) not in self._mosaic_futures:
            kept += self._enqueue_mosaic(
                0,
                0,
                overview,
                revision,
                orientation,
                load_mosaic,
                previous,
                batch,
                priority=1,
            )
        if batch is not None:
            self._thumbnail_batch_superseded += len(previous) - kept
        return True

    def _overview_tile_cells(self) -> int:
        span = max(self.matrix_width, self.matrix_height, 1)
        cells = self.BASE_TILE_CELLS
        while cells < span:
            cells *= 2
        return cells

    def _enqueue_mosaic(
        self,
        tile_x: int,
        tile_y: int,
        cells: int,
        revision: str,
        orientation: str,
        load_mosaic,
        previous: dict,
        batch,
        priority: int,
    ) -> int:
        key = (tile_x, tile_y, cells, revision, orientation)
        x1 = tile_x * cells + 1
        x2 = min(self.matrix_width, x1 + cells - 1)
        row1 = tile_y * cells
        row2 = min(self.matrix_height - 1, row1 + cells - 1)
        y_top = self._y_for_row(row1)
        y_bottom = self._y_for_row(row2)
        y1, y2 = min(y_top, y_bottom), max(y_top, y_bottom)
        prior = previous.get(key)
        if prior is not None and prior[0] == self._viewport_generation:
            self._mosaic_futures[key] = prior
            return 1
        cache_key = ThumbnailKey(
            source_key=f"mosaic:{x1}:{y1}:{x2}:{y2}:{orientation}",
            source_revision=revision,
            lod=cells,
            width=256,
            height=256,
            codec="png",
        )

        def produce(x1=x1, y1=y1, x2=x2, y2=y2, orientation=orientation, load_mosaic=load_mosaic):
            def load():
                return load_mosaic(x1=x1, y1=y1, x2=x2, y2=y2, orientation=orientation)

            return batch.profile_call(load) if batch is not None else load()

        future = self._thumbnail_cache.request(
            cache_key,
            produce,
            priority=priority,
            generation=self._thumbnail_cache.generation,
        )
        if batch is not None:
            self._thumbnail_batch_total += 1
            batch.track(future)
        self._mosaic_futures[key] = (self._viewport_generation, future)
        return 0
