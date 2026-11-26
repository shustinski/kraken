"""Frame-matrix data and asset sources over Contour's list of frames.

Both run on the shared Kraken matrix engine: the data source answers viewport
requests from an immutable snapshot of the frame list, the asset source turns a
frame into PNG bytes of the requested size, from the pyramid store when one is
set and otherwise from the thumbnail Contour already loaded.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable
from dataclasses import dataclass

from PyQt6.QtCore import QBuffer, QByteArray, QIODevice, QRect, QSize, Qt
from PyQt6.QtGui import QImage

from kraken_core.frame_matrix.models import (
    MatrixAssetRef,
    MatrixItem,
    MatrixViewportRequest,
    MatrixViewportResult,
)
from kraken_core.frame_matrix.ports import CancellationToken

from ..application.frame_lod import PyramidFrameStore


@dataclass(frozen=True, slots=True)
class ContourFrameEntry:
    """One visible frame of the matrix, captured on the UI thread."""

    index: int
    x: int
    y: int
    path: str = ""
    status: str = ""
    label: str = ""
    tooltip: str = ""
    image: QImage | None = None
    revision: str = ""


class ContourMatrixDataSource:
    """Viewport answers from the latest snapshot of Contour's frame list."""

    def __init__(self, entries: Iterable[ContourFrameEntry] = ()) -> None:
        self._entries: tuple[ContourFrameEntry, ...] = tuple(entries)
        self._by_index: dict[int, ContourFrameEntry] = {entry.index: entry for entry in self._entries}

    def replace(self, entries: Iterable[ContourFrameEntry]) -> None:
        snapshot = tuple(entries)
        by_index = {entry.index: entry for entry in snapshot}
        # Readers take one reference each; swapping both keeps them consistent.
        self._entries, self._by_index = snapshot, by_index

    def entries(self) -> tuple[ContourFrameEntry, ...]:
        return self._entries

    def entry(self, index: int) -> ContourFrameEntry | None:
        return self._by_index.get(int(index))

    def load_viewport(
        self,
        request: MatrixViewportRequest,
        cancellation: CancellationToken | None = None,
    ) -> MatrixViewportResult:
        del cancellation
        bounds = request.bounds
        items: list[MatrixItem] = []
        for entry in self._entries:
            if not bounds.contains(entry.x, entry.y):
                continue
            metadata = {"frame_id": entry.index, "path": entry.path}
            asset = (
                MatrixAssetRef(
                    source_key=entry.path or f"frame:{entry.index}",
                    source_revision=entry.revision,
                    media_type="image/png",
                    metadata=metadata,
                )
                if entry.revision
                else None
            )
            items.append(
                MatrixItem(
                    key=str(entry.index),
                    x=entry.x,
                    y=entry.y,
                    status=entry.status or "empty",
                    label=entry.label,
                    tooltip=entry.tooltip,
                    asset=asset,
                    metadata=metadata,
                )
            )
        return MatrixViewportResult(request=request, items=tuple(items))


class ContourFrameAssetSource:
    """PNG thumbnails for matrix cells."""

    def __init__(self, data_source: ContourMatrixDataSource) -> None:
        self._data_source = data_source
        self._pyramid_store: PyramidFrameStore | None = None
        self._lock = threading.Lock()

    def set_pyramid_store(self, store: PyramidFrameStore | None) -> None:
        with self._lock:
            self._pyramid_store = store

    def pyramid_store(self) -> PyramidFrameStore | None:
        return self._pyramid_store

    def load_asset(
        self,
        reference: MatrixAssetRef,
        *,
        width: int,
        height: int,
        cancellation: CancellationToken | None = None,
    ) -> bytes:
        del cancellation
        frame_id = int(reference.metadata.get("frame_id", -1))
        width = max(1, int(width))
        height = max(1, int(height))
        image = self._pyramid_image(frame_id, width, height)
        if image is None:
            entry = self._data_source.entry(frame_id)
            image = entry.image if entry is not None else None
        if image is None or image.isNull():
            raise LookupError(f"no thumbnail for frame {frame_id}")
        return encode_png(cover_scaled(image, width, height))

    def _pyramid_image(self, frame_id: int, width: int, height: int) -> QImage | None:
        with self._lock:
            store = self._pyramid_store
        if store is None or frame_id < 0:
            return None
        from ..adapters.qt.pyramid import qimage_from_array

        lod = choose_pyramid_lod(store, frame_id, width, height)
        array = store.get_thumbnail(frame_id, lod, max_size=max(width, height))
        return qimage_from_array(array)


def choose_pyramid_lod(store: PyramidFrameStore, frame_id: int, width: int, height: int) -> int:
    """Coarsest pyramid level that still covers ``width`` x ``height``."""

    try:
        lods = sorted(int(lod) for lod in store.available_lods())
    except Exception:
        return 0
    if not lods:
        return 0
    for lod in reversed(lods):
        try:
            frame_w, frame_h = store.get_frame_size(frame_id, lod)
        except Exception:
            continue
        if frame_w >= width and frame_h >= height:
            return lod
    return lods[0]


def cover_scaled(image: QImage, width: int, height: int) -> QImage:
    """Center-crop ``image`` to the cell aspect and scale it to the cell size."""

    source_w = max(1, image.width())
    source_h = max(1, image.height())
    target_aspect = width / float(height)
    if source_w / float(source_h) > target_aspect:
        crop_w = max(1, round(source_h * target_aspect))
        cropped = image.copy(QRect((source_w - crop_w) // 2, 0, crop_w, source_h))
    else:
        crop_h = max(1, round(source_w / target_aspect))
        cropped = image.copy(QRect(0, (source_h - crop_h) // 2, source_w, crop_h))
    if cropped.size() == QSize(width, height):
        return cropped
    return cropped.scaled(
        QSize(width, height),
        Qt.AspectRatioMode.IgnoreAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )


def encode_png(image: QImage) -> bytes:
    payload = QByteArray()
    buffer = QBuffer(payload)
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    if not image.save(buffer, "PNG"):
        raise ValueError("thumbnail could not be encoded")
    buffer.close()
    return bytes(payload)


__all__ = [
    "ContourFrameAssetSource",
    "ContourFrameEntry",
    "ContourMatrixDataSource",
    "choose_pyramid_lod",
    "cover_scaled",
    "encode_png",
]
