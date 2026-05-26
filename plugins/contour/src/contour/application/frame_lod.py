from __future__ import annotations

from collections.abc import Sequence
from math import ceil
from pathlib import Path
from typing import Any

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
        frame_store: "PyramidFrameStore",
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
        rows = int(ceil(self.frame_count / float(self.columns)))
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
    def __init__(self, image_paths: Sequence[str | Path] = ()) -> None:
        self.image_paths = [str(Path(path)) for path in image_paths]

    def has_zarr(self) -> bool:
        return False

    def available_lods(self) -> tuple[int, ...]:
        return ()

    def max_lod(self) -> int:
        return max(self.available_lods(), default=0)

    def frame_count(self) -> int:
        return 0

    def get_frame(self, frame_id: int, lod: int = 0) -> np.ndarray:
        raise RuntimeError("Pyramid frames are available only from Zarr storage.")

    def get_frame_size(self, frame_id: int, lod: int = 0) -> tuple[int, int]:
        frame = self.get_frame(frame_id, lod)
        return int(frame.shape[1]), int(frame.shape[0])

    def get_thumbnail(self, frame_id: int, lod: int = 0, max_size: int = 256) -> np.ndarray:
        return self.get_frame(frame_id, lod)


class ZarrFrameStore(PyramidFrameStore):
    """Zarr-backed frame pyramid store."""

    def __init__(self, image_paths: Sequence[str | Path] = (), zarr_path: str | Path | None = None) -> None:
        super().__init__(image_paths)
        self.zarr_path = Path(zarr_path) if zarr_path else None
        self._root: Any | None = None
        self._source_size: tuple[int, int] | None = None
        if self.zarr_path is not None:
            self._open_zarr(self.zarr_path)
        self._load_source_size_metadata()
        if self._source_size is None:
            self._probe_source_size()

    @classmethod
    def from_image_paths(cls, image_paths: Sequence[str | Path]) -> "ZarrFrameStore":
        paths = [Path(path) for path in image_paths]
        candidates: list[Path] = []
        if paths:
            parent = paths[0].parent
            candidates.extend(
                [
                    parent / "frames.zarr",
                    parent / "pyramid.zarr",
                    parent / "images.zarr",
                    parent / "zarr",
                ]
            )
            candidates.extend(sorted(parent.glob("*.zarr")))
        for candidate in candidates:
            if candidate.exists():
                return cls(paths, candidate)
        return cls(paths, paths[0].parent / "frames.zarr" if paths else None)

    def needs_lod_build(self) -> bool:
        return bool(self.zarr_path is not None and self._source_size is not None and self.max_lod() <= 0)

    def refresh(self) -> None:
        if self.zarr_path is not None:
            self._open_zarr(self.zarr_path)
        self._load_source_size_metadata()

    @classmethod
    def _build_zarr_pyramid(cls, image_paths: Sequence[Path], zarr_path: Path) -> Path | None:
        try:
            import zarr  # type: ignore
        except Exception:
            return None
        if not image_paths:
            return None
        try:
            first = _load_zarr_source_image(image_paths[0])
        except Exception:
            return None
        height, width = first.shape[:2]
        if height <= 0 or width <= 0:
            return None
        try:
            root = zarr.open_group(str(zarr_path), mode="w")
            root.attrs["source_count"] = int(len(image_paths))
            root.attrs["source_paths"] = [str(path) for path in image_paths]
            root.attrs["lod0_source"] = True
            root.attrs["source_width"] = int(width)
            root.attrs["source_height"] = int(height)
            lod_shapes = _pyramid_lod_shapes(width, height)
            for lod, (lod_width, lod_height) in enumerate(lod_shapes[1:], start=1):
                array = root.create_array(
                    f"lod_{lod}",
                    shape=(len(image_paths), lod_height, lod_width, 3),
                    dtype=np.uint8,
                    chunks=(1, lod_height, lod_width, 3),
                    overwrite=True,
                )
                for frame_id, path in enumerate(image_paths):
                    image = first if frame_id == 0 else _load_zarr_source_image(path)
                    if image.shape[1] != width or image.shape[0] != height:
                        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
                    if lod_width != width or lod_height != height:
                        frame = cv2.resize(image, (lod_width, lod_height), interpolation=cv2.INTER_AREA)
                    else:
                        frame = image
                    array[frame_id] = np.ascontiguousarray(frame, dtype=np.uint8)
            return zarr_path
        except Exception:
            return None

    def _open_zarr(self, path: Path) -> None:
        try:
            import zarr  # type: ignore
        except Exception:
            self._root = None
            return
        try:
            self._root = zarr.open(str(path), mode="r")
        except Exception:
            self._root = None

    def has_zarr(self) -> bool:
        return self._source_size is not None and bool(self.image_paths)

    def _load_source_size_metadata(self) -> None:
        if self._root is None:
            return
        try:
            width = int(self._root.attrs.get("source_width", 0))
            height = int(self._root.attrs.get("source_height", 0))
        except Exception:
            return
        if width > 0 and height > 0:
            self._source_size = (width, height)

    def _probe_source_size(self) -> None:
        if not self.image_paths:
            return
        try:
            frame = _load_zarr_source_image(Path(self.image_paths[0]))
        except Exception:
            return
        self._source_size = (int(frame.shape[1]), int(frame.shape[0]))

    def available_lods(self) -> tuple[int, ...]:
        if self._source_size is None:
            return ()
        lods: set[int] = {0} if self.image_paths else set()
        if self._root is not None:
            try:
                keys = list(self._root.keys())
            except Exception:
                keys = []
            for key in keys:
                text = str(key).lower()
                if text.startswith("lod_"):
                    try:
                        lods.add(int(text.split("_", 1)[1]))
                    except ValueError:
                        pass
                elif text.isdigit():
                    lods.add(int(text))
            if not lods and _zarr_node_ndim(self._root) >= 3:
                lods.add(0)
        return tuple(sorted(lods))

    def frame_count(self) -> int:
        if self._source_size is None:
            return 0
        if self.image_paths:
            return len(self.image_paths)
        try:
            for lod in self.available_lods():
                node = self._lod_node(lod)
                if node is not None and _zarr_node_ndim(node) >= 3:
                    return int(node.shape[0])
        except Exception:
            pass
        return 0

    def _lod_node(self, lod: int):
        if int(lod) == 0 and self.image_paths:
            return None
        if self._root is None:
            return None
        for key in (f"lod_{int(lod)}", str(int(lod)), f"level_{int(lod)}"):
            try:
                return self._root[key]
            except Exception:
                continue
        if int(lod) == 0 and _zarr_node_ndim(self._root) >= 3:
            return self._root
        return None

    def get_frame(self, frame_id: int, lod: int = 0) -> np.ndarray:
        if self._source_size is None:
            raise RuntimeError(f"Zarr LOD {lod} is not available.")
        if int(lod) == 0:
            try:
                return _load_zarr_source_image(Path(self.image_paths[int(frame_id)]))
            except (IndexError, ValueError) as exc:
                raise RuntimeError(f"Source frame {frame_id} is not available for LOD 0.") from exc
        node = self._lod_node(lod)
        if node is None:
            raise RuntimeError(f"Zarr LOD {lod} is not available.")
        frame = np.asarray(node[int(frame_id)])
        return _normalize_image_array(frame)

    def get_frame_size(self, frame_id: int, lod: int = 0) -> tuple[int, int]:
        if int(lod) == 0:
            if self._source_size is not None:
                return self._source_size
            if self._root is not None:
                try:
                    width = int(self._root.attrs.get("source_width", 0))
                    height = int(self._root.attrs.get("source_height", 0))
                    if width > 0 and height > 0:
                        self._source_size = (width, height)
                        return self._source_size
                except Exception:
                    pass
            frame = self.get_frame(frame_id, lod)
            self._source_size = (int(frame.shape[1]), int(frame.shape[0]))
            return self._source_size
        node = self._lod_node(lod)
        try:
            if node is not None and _zarr_node_ndim(node) >= 3:
                shape = tuple(int(value) for value in node.shape)
                if len(shape) == 3:
                    return (shape[2], shape[1])
                return (shape[2], shape[1])
        except Exception:
            pass
        return super().get_frame_size(frame_id, lod)

    def get_thumbnail(self, frame_id: int, lod: int = 0, max_size: int = 256) -> np.ndarray:
        if int(lod) == 0:
            return self.get_frame(frame_id, lod)
        node = self._lod_node(lod)
        if node is None:
            raise RuntimeError(f"Zarr LOD {lod} is not available.")
        return _normalize_image_array(np.asarray(node[int(frame_id)]))


def _zarr_node_ndim(node: object) -> int:
    try:
        return int(getattr(node, "ndim"))
    except Exception:
        shape = getattr(node, "shape", ())
        try:
            return len(shape)
        except Exception:
            return 0


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


def _load_zarr_source_image(path: Path) -> np.ndarray:
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
