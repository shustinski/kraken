"""qt: construction 2."""
from __future__ import annotations
import math
import weakref
from concurrent.futures import Future, ThreadPoolExecutor
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import (
    QWidget,
)
from kraken_core.frame_matrix.cache import ThumbnailCacheCoordinator
from kraken_core.frame_matrix.models import (
    LodBand,
    MatrixBounds,
    MatrixSession,
    MatrixViewportRequest,
    MatrixViewportResult,
)
from kraken_core.frame_matrix.ports import MatrixAssetSource, MatrixDataSource, MatrixLayerRenderer
from kraken_core.frame_matrix.qt_renderers import DEFAULT_RENDERERS
from kraken_core.frame_matrix.storage import StoreNamespace, StorePolicy, ThumbnailStore

from kraken_core.frame_matrix.qt_parts.module_state import GridOrientation
from kraken_core.frame_matrix.qt_parts.background_resources import MatrixBackgroundResources


class Construction2Mixin:
    def __init__(
        self,
        width: int = 1,
        height: int = 1,
        orientation: GridOrientation | str = GridOrientation.Y_DOWN,
        parent: QWidget | None = None,
        *,
        data_source: MatrixDataSource | None = None,
        asset_source: MatrixAssetSource | None = None,
        thumbnail_store: ThumbnailStore | None = None,
    ) -> None:
        super().__init__(width, height, orientation, parent)
        self._matrix_session: MatrixSession | None = None
        self._data_source = data_source
        self._asset_source = asset_source
        self._thumbnail_store = thumbnail_store
        self._thumbnail_cache: ThumbnailCacheCoordinator | None = None
        self._renderers: list[MatrixLayerRenderer] = [renderer() for renderer in DEFAULT_RENDERERS]
        self._viewport_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="kraken-matrix-viewport")
        self._background_resources = MatrixBackgroundResources(self._viewport_executor)
        self.destroyed.connect(self._background_resources.close)
        weakref.finalize(self, self._background_resources.close)
        self._viewport_future: Future[MatrixViewportResult] | None = None
        self._viewport_future_signature: tuple[object, ...] | None = None
        self._last_applied_signature: tuple[object, ...] | None = None
        self._viewport_request_deferred = False
        self._viewport_generation = 0
        self._loading = False
        self._asset_futures: dict[tuple[int, int], tuple[int, Future]] = {}
        self._viewport_request_timer = QTimer(self)
        self._viewport_request_timer.setSingleShot(True)
        self._viewport_request_timer.setInterval(60)
        self._viewport_request_timer.timeout.connect(self._request_visible_viewport)
        self._result_poll_timer = QTimer(self)
        self._result_poll_timer.setInterval(20)
        self._result_poll_timer.timeout.connect(self._poll_background_results)
        self.viewportChanged.connect(self._schedule_viewport_request)
        self.lodChanged.connect(self._schedule_viewport_request)

    def set_data_source(
        self,
        data_source: MatrixDataSource | None,
        *,
        asset_source: MatrixAssetSource | None = None,
    ) -> None:
        self._viewport_generation += 1
        self._last_applied_signature = None
        self._data_source = data_source
        if asset_source is not None:
            self._asset_source = asset_source
        self._schedule_viewport_request(immediate=True)

    def set_session(
        self,
        session: MatrixSession,
        *,
        data_source: MatrixDataSource | None = None,
        asset_source: MatrixAssetSource | None = None,
    ) -> None:
        self._viewport_generation += 1
        self._last_applied_signature = None
        self._viewport_request_deferred = False
        self._matrix_session = session
        if data_source is not None:
            self._data_source = data_source
        if asset_source is not None:
            self._asset_source = asset_source
        self.set_matrix_size(session.width, session.height, session.orientation)
        self.clear_cells()
        if self._thumbnail_cache is not None:
            self._thumbnail_cache.close()
            self._thumbnail_cache = None
            self._background_resources.cache = None
        if self._thumbnail_store is not None:
            self._thumbnail_store.open(
                StoreNamespace(
                    plugin="matrix",
                    project=session.namespace,
                    generation="v1",
                ),
                StorePolicy(),
            )
            self._thumbnail_cache = ThumbnailCacheCoordinator(
                self._thumbnail_store,
                error_handler=self._background_resources.errors.put,
            )
            self._background_resources.cache = self._thumbnail_cache
        self.sessionChanged.emit(session)
        self._schedule_viewport_request(immediate=True)

    def session(self) -> MatrixSession | None:
        return self._matrix_session

    def register_layer_renderer(self, renderer: MatrixLayerRenderer) -> None:
        if renderer not in self._renderers:
            self._renderers.append(renderer)
            self.viewport().update()

    def remove_layer_renderer(self, renderer: MatrixLayerRenderer) -> None:
        if renderer in self._renderers:
            self._renderers.remove(renderer)
            self.viewport().update()

    def clear_thumbnail_cache(self) -> None:
        if self._thumbnail_cache is not None:
            self._thumbnail_cache.clear_ram()
        if self._thumbnail_store is None:
            return
        try:
            self._thumbnail_store.clear_namespace()
        except Exception as exc:
            self.errorOccurred.emit(str(exc))

    def semantic_lod(self) -> LodBand:
        session = self._matrix_session
        del session
        zoom = self.zoom_factor()
        if zoom < self.CELLS_LOD:
            return LodBand.OVERVIEW
        if zoom < self.DETAILS_LOD:
            return LodBand.CELLS
        if zoom < 1.5:
            return LodBand.PREVIEWS
        if zoom < 2.5:
            return LodBand.DETAILS
        return LodBand.SUBCELLS

    def _schedule_viewport_request(self, *_args: object, immediate: bool = False) -> None:
        if self._background_resources.closed or self._data_source is None or self._matrix_session is None:
            return
        self._viewport_request_timer.start(0 if immediate else 60)

    def _visible_matrix_bounds(self) -> MatrixBounds:
        visible = self.visible_scene_rect()
        first_column = max(0, int(math.floor(visible.left() / self.CELL_PITCH)) - 1)
        last_column = min(self.matrix_width - 1, int(math.floor(visible.right() / self.CELL_PITCH)) + 1)
        first_row = max(0, int(math.floor(visible.top() / self.CELL_PITCH)) - 1)
        last_row = min(self.matrix_height - 1, int(math.floor(visible.bottom() / self.CELL_PITCH)) + 1)
        first_y = self._y_for_row(first_row)
        last_y = self._y_for_row(last_row)
        return MatrixBounds(
            first_column + 1,
            min(first_y, last_y),
            last_column + 1,
            max(first_y, last_y),
        )

    def _request_lod(self) -> int:
        zoom = max(self.zoom_factor(), self.MIN_ZOOM)
        if zoom >= self.CELLS_LOD:
            return 0
        return min(24, max(1, int(math.ceil(math.log2(self.CELLS_LOD / zoom)))))

    def _current_viewport_request(self) -> MatrixViewportRequest:
        return MatrixViewportRequest(
            self._visible_matrix_bounds(),
            lod=self._request_lod(),
            generation=self._viewport_generation,
            asset_width=max(1, int(self.CELL_SIZE)),
            asset_height=max(1, int(self.CELL_SIZE)),
        )

    @staticmethod
    def _request_signature(request: MatrixViewportRequest) -> tuple[object, ...]:
        bounds = request.bounds
        return (
            request.generation,
            request.lod,
            bounds.x1,
            bounds.y1,
            bounds.x2,
            bounds.y2,
            request.asset_width,
            request.asset_height,
        )

    def _request_visible_viewport(self) -> None:
        if self._background_resources.closed or self._data_source is None or self._matrix_session is None:
            return
        request = self._current_viewport_request()
        signature = self._request_signature(request)
        if signature == self._viewport_future_signature:
            self._viewport_request_deferred = False
            return
        if self._viewport_future is not None:
            self._viewport_request_deferred = signature != self._last_applied_signature
            return
        if signature == self._last_applied_signature:
            return
        self._viewport_request_deferred = False
        self._viewport_future_signature = signature
        self._viewport_future = self._viewport_executor.submit(self._data_source.load_viewport, request, None)
        self._set_loading(True)
        if not self._result_poll_timer.isActive():
            self._result_poll_timer.start()
