from __future__ import annotations

from concurrent.futures import Future

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QBuffer, QEvent, QIODevice, QPoint, QPointF, Qt
from PyQt6.QtGui import QContextMenuEvent, QImage, QKeyEvent, QPixmap, QWheelEvent
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from kraken_core.frame_matrix import (
    MatrixAssetRef,
    MatrixBounds,
    MatrixItem,
    MatrixSession,
    MatrixViewportRequest,
    MatrixViewportResult,
)
from kraken_core.frame_matrix.adapters.memory import MemoryThumbnailStore
from kraken_core.frame_matrix import profiling
from kraken_hub.presentation.qt.matrix_source import KrakenMatrixDataSource
from kraken_manager.presentation.qt import (
    FrameCellData,
    FrameMatrixMinimap,
    FrameMatrixView,
    FrameMatrixWidget,
    FrameRect,
    FrameSelection,
    GridOrientation,
    MatrixLod,
)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_large_sparse_matrix_virtualizes_visible_tiles(qapp):
    view = FrameMatrixView(10_000_000, 1)
    view.resize(900, 480)
    view.show()
    qapp.processEvents()
    view.zoom_to_fit()
    qapp.processEvents()

    assert view.materialized_cell_count() == 0
    assert view.visible_item_count() < 20
    assert view.lod_level() is MatrixLod.OVERVIEW


def test_matrix_coordinates_follow_project_y_orientation(qapp):
    view = FrameMatrixView(4, 3, GridOrientation.Y_DOWN)
    top_left = view.scene_rect_for_frame(1, 1).center()
    assert view.frame_at_scene_pos(top_left) == (1, 1)

    view.set_orientation(GridOrientation.Y_UP)

    top_left = view.scene_rect_for_frame(1, 3).center()
    assert view.frame_at_scene_pos(top_left) == (1, 3)
    assert view.scene_rect_for_frame(1, 1).top() > top_left.y()


def test_matrix_cells_are_adjacent_without_gaps(qapp):
    view = FrameMatrixView(2, 2, GridOrientation.Y_DOWN)

    first = view.scene_rect_for_frame(1, 1)
    right = view.scene_rect_for_frame(2, 1)
    below = view.scene_rect_for_frame(1, 2)

    assert view.CELL_PITCH == view.CELL_SIZE
    assert first.right() == right.left()
    assert first.bottom() == below.top()
    assert view.frame_at_scene_pos(right.center()) == (2, 1)
    assert view.frame_at_scene_pos(below.center()) == (1, 2)


def test_matrix_uses_sparse_presentation_data_and_compact_selection(qapp):
    view = FrameMatrixView(1_000, 1_000)
    cell = FrameCellData(
        20,
        30,
        status="in_review",
        performer_color="#ff00ff",
        performer_initials="AK",
        payload={"opaque": True},
    )
    view.set_cells([cell])
    selection = FrameSelection((FrameRect(10, 20, 30, 40),))
    view.set_selection(selection)

    assert view.materialized_cell_count() == 1
    assert view.cell_data(20, 30) is cell
    assert view.cell_data(21, 30).status == "empty"
    assert view.selection() == selection
    assert selection.contains(20, 30)


def test_context_menu_emits_frame_and_current_selection(qapp):
    view = FrameMatrixView(3, 3)
    view.resize(360, 280)
    view.show()
    qapp.processEvents()
    frame_center = view.mapFromScene(view.scene_rect_for_frame(2, 2).center())
    emitted = []
    view.contextMenuRequested.connect(lambda context, position: emitted.append((context, position)))
    event = QContextMenuEvent(
        QContextMenuEvent.Reason.Mouse,
        frame_center,
        view.viewport().mapToGlobal(frame_center),
    )

    QApplication.sendEvent(view.viewport(), event)

    assert emitted
    context, _position = emitted[-1]
    assert (context.x, context.y) == (2, 2)
    assert context.selection == FrameSelection.single(2, 2)


def test_frame_selection_expansion_can_be_bounded():
    selection = FrameSelection((FrameRect(1, 1, 100, 100),))

    with pytest.raises(ValueError, match="more than 10"):
        tuple(selection.coordinates(maximum=10))


def test_shared_widget_loads_only_the_visible_viewport(qapp):
    requests = []

    class Source:
        def load_viewport(self, request, cancellation=None):
            requests.append(request)
            return MatrixViewportResult(
                request,
                items=(MatrixItem("frame-1", request.bounds.x1, request.bounds.y1, status="image_ready"),),
                source_revision="1",
            )

    view = FrameMatrixWidget(10_000_000, 1, data_source=Source())
    view.resize(900, 480)
    view.show()
    view.set_session(MatrixSession("large", 10_000_000, 1))
    QTest.qWait(180)
    qapp.processEvents()

    assert requests
    assert requests[-1].bounds.width < 10_000_000
    assert view.materialized_cell_count() == 1
    assert (
        view.cell_data(
            requests[-1].bounds.x1,
            requests[-1].bounds.y1,
        ).payload["request_lod"]
        == requests[-1].lod
    )


def _send_wheel(view, delta, modifiers=Qt.KeyboardModifier.NoModifier):
    position = QPointF(view.viewport().rect().center())
    event = QWheelEvent(
        position,
        QPointF(view.viewport().mapToGlobal(position.toPoint())),
        QPoint(),
        QPoint(0, delta),
        Qt.MouseButton.NoButton,
        modifiers,
        Qt.ScrollPhase.ScrollUpdate,
        False,
    )
    QApplication.sendEvent(view.viewport(), event)


def test_wheel_scrolls_and_modifier_changes_axis_or_zoom(qapp):
    view = FrameMatrixView(100, 100)
    view.resize(480, 320)
    view.show()
    view.center_on_frame(50, 50)
    qapp.processEvents()

    vertical_before = view.verticalScrollBar().value()
    _send_wheel(view, -120)
    assert view.verticalScrollBar().value() > vertical_before

    horizontal_before = view.horizontalScrollBar().value()
    _send_wheel(view, -120, Qt.KeyboardModifier.ShiftModifier)
    assert view.horizontalScrollBar().value() > horizontal_before

    zoom_before = view.zoom_factor()
    _send_wheel(view, 120, Qt.KeyboardModifier.ControlModifier)
    assert view.zoom_factor() > zoom_before


def test_shared_widget_deduplicates_viewport_and_loads_thumbnail(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(profiling, "_config", None)
    (tmp_path / "kraken.ini").write_text(
        "[profiling]\nenabled = false\nmatrix_thumbnails_total = true\n", encoding="utf-8"
    )
    profiling.configure(tmp_path)
    requests = []
    image = QImage(8, 8, QImage.Format.Format_RGB32)
    image.fill(Qt.GlobalColor.red)
    output = QBuffer()
    output.open(QIODevice.OpenModeFlag.WriteOnly)
    assert image.save(output, "PNG")
    thumbnail = bytes(output.data())

    class Source:
        def load_viewport(self, request, cancellation=None):
            import time

            time.sleep(0.06)
            requests.append(request)
            return MatrixViewportResult(
                request,
                items=(
                    MatrixItem(
                        "frame-1",
                        1,
                        1,
                        status="image_ready",
                        asset=MatrixAssetRef("asset-1", "revision-1"),
                    ),
                ),
            )

    class Assets:
        def load_asset(self, reference, *, width, height, cancellation=None):
            return thumbnail

    loading = []
    view = FrameMatrixWidget(
        10,
        10,
        data_source=Source(),
        asset_source=Assets(),
        thumbnail_store=MemoryThumbnailStore(),
    )
    view.loadingChanged.connect(loading.append)
    view.resize(480, 320)
    view.show()
    view.set_session(MatrixSession("thumbnail", 10, 10))
    QTest.qWait(300)
    qapp.processEvents()

    assert len(requests) == 1
    assert view.cell_data(1, 1).thumbnail is not None
    assert not view.cell_data(1, 1).thumbnail.isNull()
    assert loading == [True, False]
    report = (tmp_path / "profiling.log").read_text(encoding="utf-8")
    assert "[matrix_thumbnails_total]" in report
    assert "requested=1 completed=1 failed=0" in report
    assert "load_viewport" in report
    assert "load_asset" in report
    assert float(report.split("[matrix_thumbnails_total] elapsed_ms=")[1].split()[0]) >= 60


def test_overlapping_viewport_reuses_decoded_thumbnail_and_pending_work(qapp):
    image = QImage(8, 8, QImage.Format.Format_RGB32)
    image.fill(Qt.GlobalColor.red)
    output = QBuffer()
    output.open(QIODevice.OpenModeFlag.WriteOnly)
    assert image.save(output, "PNG")
    loaded: list[str] = []

    class Assets:
        def load_asset(self, reference, *, width, height, cancellation=None):
            loaded.append(reference.source_key)
            return bytes(output.data())

    view = FrameMatrixWidget(4, 1, asset_source=Assets(), thumbnail_store=MemoryThumbnailStore())
    view.set_session(MatrixSession("overlap", 4, 1))
    request = view._current_viewport_request()

    def item(x: int) -> MatrixItem:
        return MatrixItem(f"frame-{x}", x, 1, asset=MatrixAssetRef(f"asset-{x}", "revision-1"))

    view._apply_viewport_result(MatrixViewportResult(request, items=(item(1), item(2))))
    for _, future in view._asset_futures.values():
        future.result(timeout=3)
    view._poll_background_results()
    first = view.cell_data(2, 1).thumbnail
    assert first is not None

    view._apply_viewport_result(MatrixViewportResult(request, items=(item(2), item(3))))
    assert view.cell_data(2, 1).thumbnail is first
    for _, future in view._asset_futures.values():
        future.result(timeout=3)
    view._poll_background_results()
    assert loaded.count("asset-2") == 1
    view._apply_viewport_result(MatrixViewportResult(request, items=(item(1),)))
    assert view.cell_data(1, 1).thumbnail is not None
    assert not view._asset_futures
    assert loaded.count("asset-1") == 1
    view.close()


def test_overlapping_viewport_keeps_inflight_thumbnail(qapp):
    from threading import Event

    started = Event()
    release = Event()
    loaded: list[str] = []

    class Assets:
        def load_asset(self, reference, *, width, height, cancellation=None):
            loaded.append(reference.source_key)
            started.set()
            assert release.wait(3)
            return b"encoded"

    view = FrameMatrixWidget(2, 1, asset_source=Assets(), thumbnail_store=MemoryThumbnailStore())
    view.set_session(MatrixSession("pending-overlap", 2, 1))
    request = view._current_viewport_request()
    result = MatrixViewportResult(request, items=(MatrixItem("frame", 1, 1, asset=MatrixAssetRef("asset", "1")),))
    try:
        view._apply_viewport_result(result)
        assert started.wait(3)
        original_future = view._asset_futures[(1, 1)][1]
        view._apply_viewport_result(result)
        assert view._asset_futures[(1, 1)][1] is original_future
        release.set()
        original_future.result(timeout=3)
        assert loaded == ["asset"]
    finally:
        release.set()
        view.close()


def test_session_refresh_keeps_thumbnail_cache_for_same_project(qapp):
    class Store(MemoryThumbnailStore):
        def __init__(self):
            super().__init__()
            self.opens = 0
            self.closes = 0

        def open(self, namespace, policy=None):
            self.opens += 1
            super().open(namespace, policy)

        def close(self):
            self.closes += 1
            super().close()

    store = Store()
    view = FrameMatrixWidget(2, 2, thumbnail_store=store)
    view.set_session(MatrixSession("same-project", 2, 2, source_revision="rev", generation=1))
    thumbnail = QPixmap(8, 8)
    thumbnail.fill(Qt.GlobalColor.red)
    view.set_cells((FrameCellData(1, 1, thumbnail=thumbnail),))
    coordinator = view._thumbnail_cache
    view.set_session(MatrixSession("same-project", 2, 2, source_revision="rev", generation=2))
    assert view._thumbnail_cache is coordinator
    assert view.cell_data(1, 1).thumbnail is thumbnail
    assert store.opens == 1
    assert store.closes == 0
    view.set_session(MatrixSession("same-project", 2, 2, source_revision="changed", generation=3))
    assert view.cell_data(1, 1).thumbnail is None
    assert view._thumbnail_cache is coordinator
    view.close()


def test_updating_one_cell_keeps_tile_index_and_viewport(qapp):
    view = FrameMatrixView(30, 30)
    view.resize(800, 600)
    view.show()
    view.set_cells(FrameCellData(x, y) for y in range(1, 31) for x in range(1, 31))
    qapp.processEvents()
    tile_cells = view._indexed_tile_cells
    changes = []
    view.viewportChanged.connect(changes.append)

    view.update_cells((FrameCellData(1, 1, status="image_ready"),))

    assert view._indexed_tile_cells == tile_cells
    assert not changes
    view.close()


def test_updating_one_cell_paints_only_its_exposed_area(qapp, monkeypatch):
    view = FrameMatrixView(30, 30)
    view.resize(800, 600)
    view.show()
    view.set_cells(FrameCellData(x, y) for y in range(1, 31) for x in range(1, 31))
    qapp.processEvents()
    painted = 0
    original = view._cell_visual_color

    def count_painted(cell, mode):
        nonlocal painted
        painted += 1
        return original(cell, mode)

    monkeypatch.setattr(view, "_cell_visual_color", count_painted)
    view.update_cells((FrameCellData(1, 1, status="image_ready"),))
    qapp.processEvents()

    assert 0 < painted < 100
    view.close()


def test_thumbnail_batch_reports_failed_image(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(profiling, "_config", None)
    (tmp_path / "kraken.ini").write_text(
        "[profiling]\nenabled = false\nmatrix_thumbnails_total = true\n", encoding="utf-8"
    )
    profiling.configure(tmp_path)

    class Source:
        def load_viewport(self, request, cancellation=None):
            return MatrixViewportResult(
                request,
                items=(MatrixItem("frame-1", 1, 1, asset=MatrixAssetRef("asset", "1")),),
            )

    class Assets:
        def load_asset(self, reference, *, width, height, cancellation=None):
            return b"invalid image"

    view = FrameMatrixWidget(1, 1, data_source=Source(), asset_source=Assets(), thumbnail_store=MemoryThumbnailStore())
    view.resize(240, 240)
    view.show()
    view.set_session(MatrixSession("failed-thumbnail", 1, 1))
    QTest.qWait(300)
    qapp.processEvents()

    report = (tmp_path / "profiling.log").read_text(encoding="utf-8")
    assert "requested=1 completed=0 failed=1" in report
    assert "status=failed" in report
    assert "error_types=invalid_image_payload:1" in report


def test_thumbnail_batch_waits_for_superseded_worker(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(profiling, "_config", None)
    (tmp_path / "kraken.ini").write_text("[profiling]\nmatrix_thumbnails_total = true\n", encoding="utf-8")
    profiling.configure(tmp_path)
    view = FrameMatrixWidget(1, 1)
    batch = profiling.begin_thumbnail_batch()
    assert batch is not None
    view._thumbnail_batch_profile = batch
    view._thumbnail_batch_total = 1
    view._thumbnail_batch_superseded = 1
    old_future = Future()
    batch.track(old_future)

    view._poll_background_results()
    assert not (tmp_path / "profiling.log").exists()

    old_future.set_result(None)
    view._update_scheduled = True
    view._poll_background_results()
    assert not (tmp_path / "profiling.log").exists()
    view._update_scheduled = False
    view._poll_background_results()
    report = (tmp_path / "profiling.log").read_text(encoding="utf-8")
    assert "superseded=1" in report
    assert "status=complete" in report
    view.close()


def test_cell_lod_paints_loaded_thumbnail_instead_of_status_fill(qapp):
    view = FrameMatrixView(1, 1)
    thumbnail = QPixmap(16, 16)
    thumbnail.fill(Qt.GlobalColor.red)
    view.set_cells(
        (
            FrameCellData(
                1,
                1,
                status="image_ready",
                thumbnail=thumbnail,
            ),
        )
    )
    view.resize(240, 240)
    view.show()
    view.set_zoom_factor(0.5)
    view.centerOn(view.CELL_SIZE / 2, view.CELL_SIZE / 2)
    qapp.processEvents()

    center = view.mapFromScene(view.CELL_SIZE / 2, view.CELL_SIZE / 2)
    pixel = view.grab().toImage().pixelColor(center)

    assert pixel.red() > 200
    assert pixel.red() > pixel.blue() * 2


def test_fit_shortcut_accepts_english_and_russian_layout(qapp):
    view = FrameMatrixView(100, 100)
    view.resize(320, 240)
    view.show()

    view.set_zoom_factor(1.0)
    QTest.keyClick(view, Qt.Key.Key_F)
    english_zoom = view.zoom_factor()

    view.set_zoom_factor(1.0)
    russian_event = QKeyEvent(
        QEvent.Type.KeyPress,
        0,
        Qt.KeyboardModifier.NoModifier,
        "а",
    )
    QApplication.sendEvent(view, russian_event)

    assert english_zoom < 1.0
    assert view.zoom_factor() == pytest.approx(english_zoom)


def test_minimap_overlays_viewport_and_navigates_on_click(qapp):
    view = FrameMatrixView(100, 100)
    view.resize(480, 320)
    view.show()
    minimap = FrameMatrixMinimap(view)
    qapp.processEvents()

    assert minimap.parentWidget() is view.viewport()
    assert minimap.x() + minimap.width() <= view.viewport().width()
    assert minimap.y() + minimap.height() <= view.viewport().height()

    before = (
        view.horizontalScrollBar().value(),
        view.verticalScrollBar().value(),
    )
    QTest.mouseClick(
        minimap,
        Qt.MouseButton.LeftButton,
        pos=QPoint(minimap.width() - 10, minimap.height() - 10),
    )
    qapp.processEvents()

    assert view.horizontalScrollBar().value() > before[0]
    assert view.verticalScrollBar().value() > before[1]


def test_minimap_shows_loaded_thumbnail(qapp):
    view = FrameMatrixView(2, 2)
    thumbnail = QPixmap(16, 16)
    thumbnail.fill(Qt.GlobalColor.red)
    view.set_cells((FrameCellData(1, 1, thumbnail=thumbnail),))
    view.resize(320, 240)
    view.show()
    minimap = FrameMatrixMinimap(view)
    qapp.processEvents()

    matrix = minimap._matrix_rect()
    pixel = (
        minimap.grab()
        .toImage()
        .pixelColor(
            int(matrix.left() + matrix.width() / 4),
            int(matrix.top() + matrix.height() / 4),
        )
    )

    assert pixel.red() > pixel.green() * 2


def test_minimap_shows_the_whole_layer_mosaic(qapp):
    view = FrameMatrixWidget(2, 2)
    overview = QPixmap(32, 32)
    overview.fill(Qt.GlobalColor.green)
    view._mosaic_revision = "rev"
    view._mosaic_images[(0, 0, 16, "rev", view.orientation().value)] = overview
    view.resize(320, 240)
    view.show()
    minimap = FrameMatrixMinimap(view)
    qapp.processEvents()

    matrix = minimap._matrix_rect()
    pixel = minimap.grab().toImage().pixelColor(int(matrix.center().x()), int(matrix.center().y()))

    assert pixel.green() > pixel.red() * 2
    view.close()


def test_minimap_refreshes_after_thumbnail_batch(qapp):
    view = FrameMatrixView(2, 2)
    view.resize(320, 240)
    view.show()
    minimap = FrameMatrixMinimap(view)
    thumbnail = QPixmap(16, 16)
    thumbnail.fill(Qt.GlobalColor.red)
    view.update_cells((FrameCellData(1, 1, thumbnail=thumbnail),))
    assert minimap._content_update_timer.isActive()

    QTest.qWait(130)
    matrix = minimap._matrix_rect()
    pixel = (
        minimap.grab()
        .toImage()
        .pixelColor(int(matrix.left() + matrix.width() / 4), int(matrix.top() + matrix.height() / 4))
    )
    assert pixel.red() > pixel.green() * 2
    view.close()


def test_thumbnail_results_are_applied_in_bounded_batches(qapp):
    view = FrameMatrixWidget(100, 1)
    view.set_cells(FrameCellData(x, 1) for x in range(1, 101))
    thumbnail = QImage(1, 1, QImage.Format.Format_RGB32)
    thumbnail.fill(Qt.GlobalColor.red)
    output = QBuffer()
    output.open(QIODevice.OpenModeFlag.WriteOnly)
    assert thumbnail.save(output, "PNG")
    from kraken_core.frame_matrix.storage import ThumbnailKey, ThumbnailRecord

    for x in range(1, 101):
        future = Future()
        key = ThumbnailKey(str(x), "1", 0, 1, 1, "png")
        future.set_result(ThumbnailRecord(key, bytes(output.data()), "png"))
        view._asset_futures[(x, 1)] = (view._viewport_generation, future)

    view._poll_background_results()

    assert 0 < view.materialized_cell_count() - len(view._asset_futures) < 100
    assert view._asset_futures
    assert view.cell_data(1, 1).thumbnail is not None


def test_kraken_data_source_prefers_image_asset_over_vector_artifact():
    class Service:
        def matrix_viewport(self, *_args, **_kwargs):
            return {
                "revision": "viewport-1",
                "cells": (
                    {
                        "x": 1,
                        "y": 1,
                        "frame_id": "frame-1",
                        "status": "vectorized",
                        "sha256": "vector-artifact",
                        "artifact_version_id": "vector-version",
                        "asset_sha256": "image-artifact",
                        "asset_revision": "image-version",
                    },
                ),
                "aggregates": (),
            }

    source = KrakenMatrixDataSource(
        Service(),
        project_id="project",
        layer_id="layer",
        representation_ids=("image", "vector"),
    )
    request = MatrixViewportRequest(MatrixBounds(1, 1, 1, 1))
    result = source.load_viewport(request)

    assert result.items[0].asset is not None
    assert result.items[0].asset.source_key == "image-artifact"
    assert result.items[0].asset.source_revision == "image-version"


def test_kraken_data_source_does_not_decode_vector_artifact_as_image():
    class Service:
        def matrix_viewport(self, *_args, **_kwargs):
            return {
                "revision": "viewport-1",
                "cells": (
                    {
                        "x": 1,
                        "y": 1,
                        "frame_id": "frame-1",
                        "status": "vectorized",
                        "sha256": "cif-artifact",
                        "artifact_version_id": "vector-version",
                    },
                ),
                "aggregates": (),
            }

    source = KrakenMatrixDataSource(
        Service(),
        project_id="project",
        layer_id="layer",
        representation_ids=("vector",),
    )
    request = MatrixViewportRequest(MatrixBounds(1, 1, 1, 1))

    result = source.load_viewport(request)

    assert result.items[0].asset is None
    assert "SHA-256: cif-artifact" in result.items[0].tooltip
