from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QBuffer, QIODevice, QPoint, QPointF, Qt
from PyQt6.QtGui import QContextMenuEvent, QImage, QWheelEvent
from PyQt6.QtWidgets import QApplication
from PyQt6.QtTest import QTest

from kraken_core.frame_matrix import (
    MatrixAssetRef,
    MatrixBounds,
    MatrixItem,
    MatrixSession,
    MatrixViewportRequest,
    MatrixViewportResult,
)
from kraken_core.frame_matrix.adapters.memory import MemoryThumbnailStore
from kraken_hub.matrix_source import KrakenMatrixDataSource

from kraken_manager.presentation.qt import (
    FrameCellData,
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


def test_shared_widget_deduplicates_viewport_and_loads_thumbnail(qapp):
    requests = []
    image = QImage(8, 8, QImage.Format.Format_RGB32)
    image.fill(Qt.GlobalColor.red)
    output = QBuffer()
    output.open(QIODevice.OpenModeFlag.WriteOnly)
    assert image.save(output, "PNG")
    thumbnail = bytes(output.data())

    class Source:
        def load_viewport(self, request, cancellation=None):
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
