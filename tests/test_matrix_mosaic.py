from __future__ import annotations

import io

import pytest
from PIL import Image

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication

from kraken_core.frame_matrix import (
    MatrixAssetRef,
    MatrixItem,
    MatrixSession,
    MatrixViewportResult,
)
from kraken_core.frame_matrix.adapters.memory import MemoryThumbnailStore
from kraken_core.frame_matrix.mosaic import compose_mosaic, mosaic_slot
from kraken_core.frame_matrix.qt import FrameMatrixWidget
from kraken_hub.composition.parts.matrix_viewport import MatrixViewportMixin


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _png(color: tuple[int, int, int]) -> bytes:
    image = Image.new("RGB", (12, 8), color)
    output = io.BytesIO()
    image.save(output, "PNG")
    return output.getvalue()


def test_compose_mosaic_places_each_source_and_leaves_gaps_clear():
    payload = compose_mosaic(
        width_cells=2,
        height_cells=1,
        images={(0, 0): _png((255, 0, 0)), (1, 0): _png((0, 0, 255))},
    )
    image = Image.open(io.BytesIO(payload))

    assert image.mode == "RGBA"
    assert image.size[0] > image.size[1]
    pixels = image.load()
    assert pixels[image.size[0] // 4, image.size[1] // 2][:3] == (255, 0, 0)
    assert pixels[(image.size[0] * 3) // 4, image.size[1] // 2][:3] == (0, 0, 255)


def test_mosaic_from_cells_reads_each_source_once_and_packs_them():
    from kraken_core.frame_matrix.mosaic import mosaic_from_cells

    seen: list[str] = []

    def read_image(cell: dict) -> bytes | None:
        seen.append(cell["asset"])
        return _png((255, 0, 0) if cell["x"] == 1 else (0, 255, 0))

    payload = mosaic_from_cells(
        (
            {"x": 1, "y": 1, "asset": "a"},
            {"x": 2, "y": 1, "asset": "b"},
            {"x": 9, "y": 9, "asset": "outside"},
        ),
        x1=1,
        y1=1,
        x2=2,
        y2=1,
        orientation="y_down",
        read_image=read_image,
    )
    image = Image.open(io.BytesIO(payload))
    pixels = image.load()

    assert seen == ["a", "b"]
    assert pixels[image.size[0] // 4, image.size[1] // 2][:3] == (255, 0, 0)
    assert pixels[(image.size[0] * 3) // 4, image.size[1] // 2][:3] == (0, 255, 0)


def test_local_matrix_mosaic_reads_a_prepared_tile(tmp_path):
    from kraken_core.frame_matrix.prepared_tiles import MatrixTileNotReady

    class Project:
        def get_project(self, _project_id):
            return {"width": 2, "height": 1, "revision": "rev-1"}

        def matrix_viewport(self, *_args, **_kwargs):
            return {
                "revision": "rev-1",
                "cells": (
                    {"x": 1, "y": 1, "asset_sha256": "red"},
                    {"x": 2, "y": 1, "asset_sha256": "green"},
                ),
            }

        def read_project_blob(self, _project_id, digest):
            color = (255, 0, 0) if digest == "red" else (0, 255, 0)
            return _png(color)

    class Service(Project, MatrixViewportMixin):
        pass

    service = Service()
    service.matrix_tile_root = tmp_path
    with pytest.raises(MatrixTileNotReady):
        service.matrix_mosaic("project", layer_id="layer", x1=1, y1=1, x2=2, y2=1)
    service.prepare_matrix_tiles("project", layer_id="layer")
    payload = service.matrix_mosaic("project", layer_id="layer", x1=1, y1=1, x2=2, y2=1)
    image = Image.open(io.BytesIO(payload))
    pixels = image.load()

    assert pixels[image.size[0] // 4, image.size[1] // 2][:3] == (255, 0, 0)
    assert pixels[(image.size[0] * 3) // 4, image.size[1] // 2][:3] == (0, 255, 0)
    assert service.matrix_mosaic("project", layer_id="layer", x1=1, y1=1, x2=2, y2=1) == payload


def test_widget_requests_one_mosaic_for_the_visible_tile(qapp):
    calls: list[tuple[int, int, int, int]] = []

    class Assets:
        def load_mosaic(self, *, x1, y1, x2, y2, orientation):
            calls.append((x1, y1, x2, y2))
            assert orientation == "y_down"
            return _png((10, 20, 30))

    view = FrameMatrixWidget(4, 1, asset_source=Assets(), thumbnail_store=MemoryThumbnailStore())
    view.set_session(MatrixSession("mosaic", 4, 1))
    request = view._current_viewport_request()
    items = tuple(
        MatrixItem(f"frame-{x}", x, 1, asset=MatrixAssetRef(f"asset-{x}", "1")) for x in range(1, 5)
    )
    view._apply_viewport_result(MatrixViewportResult(request, items=items, source_revision="rev"))
    for _key, (_generation, future) in view._mosaic_futures.items():
        future.result(timeout=3)
    view._poll_background_results()

    assert calls == [(1, 1, 4, 1)]
    assert len(view._mosaic_images) == 1
    assert not view._asset_futures
    view.close()


def test_project_application_delegates_matrix_mosaic():
    from kraken_manager.application.project_application import ProjectApplication

    class Operations:
        def matrix_mosaic(self, project_id, **kwargs):
            return project_id, kwargs["orientation"], kwargs["x2"]

    application = ProjectApplication(lambda _project_id: Operations())

    assert application.matrix_mosaic("project", layer_id="layer", x1=1, y1=1, x2=4, y2=1, orientation="y_up") == (
        "project",
        "y_up",
        4,
    )


def test_pyramid_covers_zoomed_out_tiles_from_small_previews(tmp_path):
    from kraken_core.frame_matrix.prepared_tiles import PreparedTilePyramid, tile_rect

    def load_index():
        return "rev", 32, 32, ({"x": 1, "y": 1, "asset_sha256": "aa"}, {"x": 32, "y": 32, "asset_sha256": "bb"})

    def read_image(cell):
        return _png((255, 0, 0) if cell["x"] == 1 else (0, 0, 255))

    pyramid = PreparedTilePyramid(tmp_path / "pyramid", load_index, read_image)
    pyramid.build_now()
    fine = tile_rect(32, 32, 16, 0, 0, "y_down")
    coarse = tile_rect(32, 32, 32, 0, 0, "y_down")

    assert pyramid.read(x1=fine[0], y1=fine[1], x2=fine[2], y2=fine[3], orientation="y_down")
    assert pyramid.read(x1=coarse[0], y1=coarse[1], x2=coarse[2], y2=coarse[3], orientation="y_down")
    assert (tmp_path / "pyramid" / "previews-256").is_dir()


def test_zoom_keeps_the_loaded_mosaic_until_the_new_level_arrives(qapp):
    from PyQt6.QtGui import QColor, QPixmap

    from kraken_core.frame_matrix.qt_parts.viewport_tile_item import _ViewportTileItem

    view = FrameMatrixWidget(32, 32)
    pixmap = QPixmap(256, 256)
    pixmap.fill(QColor(20, 30, 40))
    view._mosaic_revision = "rev"
    view._mosaic_images[(0, 0, 32, "rev", view.orientation().value)] = pixmap
    tile = _ViewportTileItem(view, 1, 0, 16)
    pieces = view._mosaic_pieces(tile)

    assert len(pieces) == 1
    image, source, _dest = pieces[0]
    assert image is pixmap
    assert source.left() == 128
    assert source.top() == 0
    assert source.width() == 128
    view.close()


def test_far_zoom_uses_the_layer_overview_when_it_is_larger_than_the_matrix(qapp):
    from PyQt6.QtGui import QColor, QPixmap

    from kraken_core.frame_matrix.qt_parts.viewport_tile_item import _ViewportTileItem

    view = FrameMatrixWidget(40, 40)
    overview = QPixmap(64, 64)
    overview.fill(QColor(12, 34, 56))
    view._mosaic_revision = "rev"
    view._mosaic_images[(0, 0, 64, "rev", view.orientation().value)] = overview
    tile = _ViewportTileItem(view, 1, 0, 16)
    pieces = view._mosaic_pieces(tile)

    assert len(pieces) == 1
    image, source, _dest = pieces[0]
    assert image is overview
    assert source.left() > 0
    view.close()


def test_y_up_slot_puts_the_larger_coordinate_at_the_top():
    assert mosaic_slot(2, 5, x1=2, y1=4, x2=3, y2=5, orientation="y_up") == (0, 0)
    assert mosaic_slot(2, 4, x1=2, y1=4, x2=3, y2=5, orientation="y_down") == (0, 0)


def test_close_zoom_tile_gives_one_frame_the_whole_mosaic_edge(tmp_path):
    import io

    from PIL import Image

    from kraken_core.frame_matrix.prepared_tiles import MatrixTileNotReady, PreparedTilePyramid

    def load_index():
        return "rev", 32, 32, ({"x": 3, "y": 5, "asset_sha256": "cc"},)

    def read_image(cell):
        del cell
        image = Image.new("RGB", (1024, 1024))
        for x in range(0, 1024, 64):
            image.paste((255, 255, 255), (x, 0, x + 32, 1024))
        output = io.BytesIO()
        image.save(output, "PNG")
        return output.getvalue()

    pyramid = PreparedTilePyramid(tmp_path / "pyramid", load_index, read_image)
    pyramid.build_now()
    with pytest.raises(MatrixTileNotReady):
        pyramid.read(x1=3, y1=5, x2=3, y2=5, orientation="y_down")
    pyramid.build_now()
    payload = pyramid.read(x1=3, y1=5, x2=3, y2=5, orientation="y_down")

    with Image.open(io.BytesIO(payload)) as tile:
        assert tile.size == (256, 256)
        row = [tile.getpixel((x, 128))[0] for x in range(256)]
    # 16 white stripes survive: the frame was not squeezed into a 16 px slot.
    edges = sum(1 for left, right in zip(row, row[1:]) if (left > 128) != (right > 128))
    assert edges >= 28


def test_mosaic_level_follows_zoom_below_the_graphics_tile(qapp):
    view = FrameMatrixWidget(64, 64)
    view.set_zoom_factor(view.MAX_ZOOM)
    assert view._effective_tile_cells() == view.BASE_TILE_CELLS
    assert view._mosaic_tile_cells() == 1
    view.set_zoom_factor(0.05)
    assert view._mosaic_tile_cells() == view._effective_tile_cells()
    view.close()


def test_close_zoom_paints_one_cell_mosaics_over_the_graphics_tile(qapp):
    from PyQt6.QtGui import QColor, QPixmap

    from kraken_core.frame_matrix.qt_parts.viewport_tile_item import _ViewportTileItem

    view = FrameMatrixWidget(32, 32)
    view.set_zoom_factor(view.MAX_ZOOM)
    orientation = view.orientation().value
    coarse = QPixmap(256, 256)
    coarse.fill(QColor(20, 30, 40))
    sharp = QPixmap(256, 256)
    sharp.fill(QColor(200, 30, 40))
    view._mosaic_revision = "rev"
    view._mosaic_images[(0, 0, 16, "rev", orientation)] = coarse
    view._mosaic_images[(2, 3, 1, "rev", orientation)] = sharp
    view._mosaic_requested = {(2, 3, 1, "rev", orientation), (4, 4, 1, "rev", orientation)}
    view._mosaic_requested_cells = 1
    tile = _ViewportTileItem(view, 0, 0, 16)

    pieces = view._mosaic_pieces(tile)
    # One child is still loading, so the coarse tile stays underneath.
    assert [piece[0] for piece in pieces] == [coarse, sharp]
    _image, source, dest = pieces[1]
    assert source.width() == 256
    assert dest.left() == 2 * view.CELL_PITCH
    assert dest.top() == 3 * view.CELL_PITCH
    assert dest.width() == view.CELL_PITCH

    view._mosaic_requested = {(2, 3, 1, "rev", orientation)}
    assert [piece[0] for piece in view._mosaic_pieces(tile)] == [sharp]
    view.close()


def test_small_source_fills_the_whole_slot():
    from PIL import Image

    source = Image.new("RGB", (128, 64), (250, 250, 250))
    output = io.BytesIO()
    source.save(output, "PNG")

    payload = compose_mosaic(width_cells=1, height_cells=1, images={(0, 0): output.getvalue()})

    with Image.open(io.BytesIO(payload)) as tile:
        assert tile.size == (256, 256)
        # Wide frame: full width, half height, centred vertically.
        assert tile.getbbox() == (0, 64, 256, 192)


def test_urgent_tiles_are_written_smallest_first(tmp_path):
    from kraken_core.frame_matrix.prepared_tiles import PreparedTilePyramid

    pyramid = PreparedTilePyramid(tmp_path / "pyramid", lambda: ("rev", 70, 70, ()), lambda _cell: None)
    written = []
    pyramid._write_tile = lambda orientation, x1, y1, x2, y2, cells=None: written.append((x1, y1, x2, y2))
    pyramid._urgent.put(("y_down", 1, 1, 70, 70))
    pyramid._urgent.put(("y_down", 17, 1, 20, 4))
    pyramid._urgent.put(("y_down", 1, 1, 16, 16))
    pyramid._urgent.put(("y_down", 5, 5, 5, 5))

    pyramid._drain_urgent()

    # The whole-matrix overview no longer blocks the tiles on screen.
    assert written == [(5, 5, 5, 5), (17, 1, 20, 4), (1, 1, 16, 16), (1, 1, 70, 70)]


def test_connection_errors_are_retried_not_reported():
    from kraken_core.frame_matrix.qt_parts.poll_background_results import _transient_mosaic_error

    class RemoteServerError(RuntimeError):
        pass

    try:
        try:
            raise TimeoutError("timed out")
        except TimeoutError as exc:
            raise RemoteServerError("Cannot reach Kraken Server at http://127.0.0.1:8080") from exc
    except RemoteServerError as wrapped:
        assert _transient_mosaic_error(wrapped)
    assert _transient_mosaic_error(ConnectionResetError())
    assert not _transient_mosaic_error(ValueError("broken png"))
