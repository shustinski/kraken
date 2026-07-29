from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from math import ceil
from pathlib import Path
from threading import RLock

import cv2
import numpy as np
from PyQt6.QtCore import QRectF

from ..utils import load_image_color


class FixedGridFrameLayout:
    """Deterministic frame-id <-> scene geometry mapping for pyramid/contact-sheet views."""

    def __init__(
        self,
        *,
        frame_count: int,
        columns: int,
        frame_store: PyramidFrameStore,
        gap: int = 16,
    ) -> None:
        self.frame_count = max(0, int(frame_count))
        self.columns = max(1, int(columns))
        self.frame_store = frame_store
        self.gap = max(0, int(gap))

    def frame_id_to_row_col(self, frame_id: int) -> tuple[int, int]:
        frame_id = int(frame_id)
        return frame_id // self.columns, frame_id % self.columns

    def row_col_to_frame_id(self, row: int, col: int) -> int | None:
        if row < 0 or col < 0 or col >= self.columns:
            return None
        frame_id = int(row) * self.columns + int(col)
        if frame_id < 0 or frame_id >= self.frame_count:
            return None
        return frame_id

    def frame_size(self, frame_id: int, lod: int) -> tuple[int, int]:
        width, height = self.frame_store.get_frame_size(frame_id, lod)
        return max(1, int(width)), max(1, int(height))

    def step_size(self, lod: int) -> tuple[int, int]:
        width, height = self.frame_size(0, lod)
        return width + self.gap, height + self.gap

    def frame_id_to_scene_rect(self, frame_id: int, lod: int) -> QRectF:
        row, col = self.frame_id_to_row_col(frame_id)
        width, height = self.frame_size(frame_id, lod)
        step_x, step_y = self.step_size(lod)
        return QRectF(float(col * step_x), float(row * step_y), float(width), float(height))

    def scene_pos_to_frame_id(self, x: float, y: float, lod: int) -> int | None:
        step_x, step_y = self.step_size(lod)
        if step_x <= 0 or step_y <= 0 or x < 0.0 or y < 0.0:
            return None
        col = int(float(x) // float(step_x))
        row = int(float(y) // float(step_y))
        frame_id = self.row_col_to_frame_id(row, col)
        if frame_id is None:
            return None
        return frame_id if self.frame_id_to_scene_rect(frame_id, lod).contains(float(x), float(y)) else None

    def scene_rect(self, lod: int) -> QRectF:
        if self.frame_count <= 0:
            return QRectF(0, 0, 1, 1)
        rows = ceil(self.frame_count / float(self.columns))
        step_x, step_y = self.step_size(lod)
        return QRectF(0, 0, max(1, self.columns * step_x - self.gap), max(1, rows * step_y - self.gap))

    def frame_ids_intersecting(self, rect: QRectF, lod: int, *, buffer_cells: int = 1) -> list[int]:
        if self.frame_count <= 0 or rect.isEmpty():
            return []
        step_x, step_y = self.step_size(lod)
        first_col = max(0, int(rect.left() // step_x) - buffer_cells)
        last_col = min(self.columns - 1, int(rect.right() // step_x) + buffer_cells)
        first_row = max(0, int(rect.top() // step_y) - buffer_cells)
        last_row = max(first_row, int(rect.bottom() // step_y) + buffer_cells)
        frame_ids: list[int] = []
        for row in range(first_row, last_row + 1):
            for col in range(first_col, last_col + 1):
                frame_id = self.row_col_to_frame_id(row, col)
                if frame_id is not None:
                    frame_ids.append(frame_id)
        return frame_ids


class PyramidFrameStore:
    """Image-backed frame pyramid built with OpenCV resize on demand."""

    def __init__(
        self,
        image_paths: Sequence[str | Path] = (),
        *,
        max_cache_bytes: int = 256 * 1024 * 1024,
    ) -> None:
        self.image_paths = [str(Path(path)) for path in image_paths]
        self._source_size: tuple[int, int] | None = None
        self._frame_cache: OrderedDict[tuple[int, int], np.ndarray] = OrderedDict()
        self._frame_cache_bytes = 0
        self._max_cache_bytes = max(1, int(max_cache_bytes))
        self._cache_lock = RLock()

    @classmethod
    def from_image_paths(cls, image_paths: Sequence[str | Path]) -> PyramidFrameStore:
        return cls(image_paths)

    def has_lod(self) -> bool:
        if not self.image_paths:
            return False
        return Path(self.image_paths[0]).is_file()

    def available_lods(self) -> tuple[int, ...]:
        if not self.has_lod():
            return ()
        width, height = self.get_frame_size(0, 0)
        return tuple(range(len(_pyramid_lod_shapes(width, height))))

    def available_lods_hint(self) -> tuple[int, ...]:
        """Return LOD candidates without decoding image pixels on the UI thread."""

        if not self.has_lod():
            return ()
        if self._source_size is None:
            return tuple(range(8))
        width, height = self._source_size
        return tuple(range(len(_pyramid_lod_shapes(width, height))))

    def max_lod(self) -> int:
        return max(self.available_lods(), default=0)

    def frame_count(self) -> int:
        return len(self.image_paths)

    def get_frame(self, frame_id: int, lod: int = 0) -> np.ndarray:
        normalized_frame_id = int(frame_id)
        lod = max(0, int(lod))
        cache_key = (normalized_frame_id, lod)
        with self._cache_lock:
            cached = self._frame_cache.get(cache_key)
            if cached is not None:
                self._frame_cache.move_to_end(cache_key)
                return cached
        source = (
            _load_source_image(Path(self.image_paths[normalized_frame_id]))
            if lod <= 0
            else self.get_frame(normalized_frame_id, 0)
        )
        if lod <= 0:
            frame = source
        else:
            source_width = int(source.shape[1])
            source_height = int(source.shape[0])
            shapes = _pyramid_lod_shapes(source_width, source_height)
            width, height = shapes[min(lod, len(shapes) - 1)]
            frame = (
                source
                if width == source_width and height == source_height
                else cv2.resize(source, (width, height), interpolation=cv2.INTER_AREA)
            )
        self._cache_frame(cache_key, frame)
        return frame

    def _cache_frame(self, cache_key: tuple[int, int], frame: np.ndarray) -> None:
        with self._cache_lock:
            previous = self._frame_cache.pop(cache_key, None)
            if previous is not None:
                self._frame_cache_bytes -= int(previous.nbytes)
            self._frame_cache[cache_key] = frame
            self._frame_cache_bytes += int(frame.nbytes)
            while self._frame_cache_bytes > self._max_cache_bytes and len(self._frame_cache) > 1:
                _old_key, old_frame = self._frame_cache.popitem(last=False)
                self._frame_cache_bytes -= int(old_frame.nbytes)

    def get_frame_size(self, frame_id: int, lod: int = 0) -> tuple[int, int]:
        if int(frame_id) == 0 and int(lod) == 0 and self._source_size is not None:
            return self._source_size
        frame = self.get_frame(int(frame_id), 0)
        source_size = (int(frame.shape[1]), int(frame.shape[0]))
        if int(frame_id) == 0:
            self._source_size = source_size
        lod = max(0, int(lod))
        if lod <= 0:
            return source_size
        shapes = _pyramid_lod_shapes(*source_size)
        if lod >= len(shapes):
            return shapes[-1]
        return shapes[lod]

    def get_thumbnail(self, frame_id: int, lod: int = 0, max_size: int = 256) -> np.ndarray:
        frame = self.get_frame(frame_id, lod)
        height, width = frame.shape[:2]
        longest = max(width, height)
        max_size = max(1, int(max_size))
        if longest <= max_size:
            return frame
        scale = max_size / float(longest)
        target_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        return cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)


def _normalize_image_array(array: np.ndarray) -> np.ndarray:
    image = np.asarray(array)
    if image.ndim == 2:
        pass
    elif image.ndim == 3 and image.shape[-1] in (1, 3, 4):
        if image.shape[-1] == 1:
            image = image[..., 0]
    elif image.ndim == 3 and image.shape[0] in (1, 3, 4):
        image = np.moveaxis(image, 0, -1)
        if image.shape[-1] == 1:
            image = image[..., 0]
    else:
        image = np.squeeze(image)
        if image.ndim > 3:
            image = image.reshape(image.shape[-3:])
    if image.dtype == np.uint8:
        return np.ascontiguousarray(image)
    if np.issubdtype(image.dtype, np.floating):
        finite = image[np.isfinite(image)]
        if finite.size and float(finite.max()) <= 1.0 and float(finite.min()) >= 0.0:
            image = image * 255.0
        image = np.nan_to_num(image, nan=0.0, posinf=255.0, neginf=0.0)
    image = np.clip(image, 0, 255).astype(np.uint8, copy=False)
    return np.ascontiguousarray(image)


def _load_source_image(path: Path) -> np.ndarray:
    image = load_image_color(path)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    elif image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = _normalize_image_array(image)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return np.ascontiguousarray(image[..., :3], dtype=np.uint8)


def _pyramid_lod_shapes(width: int, height: int, *, max_lods: int = 8, min_dimension: int = 64) -> list[tuple[int, int]]:
    shapes: list[tuple[int, int]] = []
    lod_width = max(1, int(width))
    lod_height = max(1, int(height))
    while True:
        shapes.append((lod_width, lod_height))
        if len(shapes) >= max_lods:
            break
        if len(shapes) >= 2 and lod_width <= min_dimension and lod_height <= min_dimension:
            break
        lod_width = max(1, lod_width // 2)
        lod_height = max(1, lod_height // 2)
    return shapes
