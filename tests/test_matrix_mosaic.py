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
    assert (tmp_path / "pyramid" / "previews").is_dir()


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


def test_y_up_slot_puts_the_larger_coordinate_at_the_top():
    assert mosaic_slot(2, 5, x1=2, y1=4, x2=3, y2=5, orientation="y_up") == (0, 0)
    assert mosaic_slot(2, 4, x1=2, y1=4, x2=3, y2=5, orientation="y_down") == (0, 0)
