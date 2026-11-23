"""Tile pyramid prepared ahead of a viewport request.

The finest level covers 16 matrix cells. Each coarser level doubles that span,
the same steps the view uses when it zooms out. A request only reads a PNG
that was written earlier. Missing tiles are queued and the caller is told to
try again.
"""

from __future__ import annotations

import hashlib
import logging
import os
import queue
import tempfile
import threading
from collections.abc import Callable, Iterable, Mapping
from contextlib import contextmanager
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from kraken_core.frame_matrix.mosaic import compose_mosaic, mosaic_slot

_LOGGER = logging.getLogger(__name__)
FINEST_CELLS = 16
_PREVIEW_EDGE = 128
_BUILDING = threading.local()


class MatrixTileNotReady(LookupError):
    """The tile file does not exist yet."""


def building_tiles() -> bool:
    return bool(getattr(_BUILDING, "active", False))


@contextmanager
def tile_build_scope():
    previous = building_tiles()
    _BUILDING.active = True
    try:
        yield
    finally:
        _BUILDING.active = previous


def tile_root_for(owner: object) -> Path:
    configured = getattr(owner, "matrix_tile_root", None)
    if configured:
        return Path(configured)
    factory = getattr(owner, "uow_factory", None)
    blobs = getattr(factory, "blobs", None)
    root = getattr(blobs, "root", None)
    if root is not None:
        return Path(root).parent / "matrix-tiles"
    return Path(tempfile.gettempdir()) / "kraken-matrix-tiles"


def grid_size(project: object) -> tuple[int, int]:
    if isinstance(project, Mapping):
        return int(project["width"]), int(project["height"])
    return int(getattr(project, "width")), int(getattr(project, "height"))


def content_stamp(project: object, fallback: str) -> str:
    if isinstance(project, Mapping):
        revision = project.get("revision", fallback)
    else:
        revision = getattr(project, "revision", fallback)
    return str(revision if revision not in (None, "") else fallback)


def pyramid_cell_sizes(width: int, height: int) -> tuple[int, ...]:
    """Powers of two from 16 up to a tile that covers the whole matrix."""

    span = max(int(width), int(height), 1)
    size = FINEST_CELLS
    sizes = []
    while True:
        sizes.append(size)
        if size >= span:
            return tuple(sizes)
        size *= 2


def tile_rect(
    width: int,
    height: int,
    cells: int,
    tile_x: int,
    tile_y: int,
    orientation: str,
) -> tuple[int, int, int, int]:
    """Bounds of one graphics tile. Y-up rows start at the top of the matrix."""

    x1 = int(tile_x) * int(cells) + 1
    x2 = min(int(width), x1 + int(cells) - 1)
    row1 = int(tile_y) * int(cells)
    row2 = min(int(height) - 1, row1 + int(cells) - 1)
    if orientation == "y_up":
        y_top = int(height) - row1
        y_bottom = int(height) - row2
    else:
        y_top = row1 + 1
        y_bottom = row2 + 1
    return x1, min(y_top, y_bottom), x2, max(y_top, y_bottom)


def iter_tile_rects(width: int, height: int, cells: int, orientation: str):
    columns = max(1, -(-int(width) // int(cells)))
    rows = max(1, -(-int(height) // int(cells)))
    for tile_y in range(rows):
        for tile_x in range(columns):
            yield tile_rect(width, height, cells, tile_x, tile_y, orientation)


def preview_png(payload: bytes, edge: int = _PREVIEW_EDGE) -> bytes | None:
    """Shrink one source before it is pasted into a tile."""

    import io

    try:
        with Image.open(io.BytesIO(payload)) as source:
            if source.format == "JPEG":
                source.draft("RGB", (int(edge), int(edge)))
            frame = source.convert("RGBA")
            frame.thumbnail((int(edge), int(edge)), Image.Resampling.BILINEAR)
    except (UnidentifiedImageError, OSError, ValueError):
        return None
    output = io.BytesIO()
    frame.save(output, "PNG")
    return output.getvalue()


def _directory_name(key: tuple[object, ...]) -> str:
    raw = "\0".join(str(part) for part in key)
    return hashlib.sha256(raw.encode()).hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, suffix=".part")
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _newer_stamp(stored: str, observed: str) -> bool:
    if not stored or not observed or stored == observed:
        return False
    if "T" in stored or "T" in observed:
        return observed > stored
    return True


class PreparedTilePyramid:
    """One layer revision. Files stay on disk; the request path never composes."""

    def __init__(
        self,
        directory: Path,
        load_index: Callable[[], tuple[str, int, int, tuple[Mapping[str, object], ...]]],
        read_image: Callable[[Mapping[str, object]], bytes | None],
    ) -> None:
        self.directory = Path(directory)
        self._load_index = load_index
        self._read_image = read_image
        self._lock = threading.Lock()
        self._run_lock = threading.Lock()
        self._urgent: queue.SimpleQueue[tuple[str, int, int, int, int]] = queue.SimpleQueue()
        self._thread: threading.Thread | None = None
        self._generation = 0
        self._stamp = ""
        self._width = 0
        self._height = 0
        self._cells: tuple[Mapping[str, object], ...] = ()
        self._indexed = False

    def note_content_stamp(self, stamp: str) -> None:
        """Start or restart preparation when the layer content changes."""

        observed = str(stamp or "")
        restart = False
        with self._lock:
            if _newer_stamp(self._stamp, observed):
                self._generation += 1
                self._indexed = False
                self._marker().unlink(missing_ok=True)
                restart = True
        self.ensure_started(force=restart)

    def read(self, *, x1: int, y1: int, x2: int, y2: int, orientation: str) -> bytes:
        token = self._published_token()
        path = None if token is None else self._tile_path(token, orientation, x1, y1, x2, y2)
        if path is not None and path.is_file():
            return path.read_bytes()
        self._urgent.put((str(orientation), int(x1), int(y1), int(x2), int(y2)))
        self.ensure_started(force=True)
        raise MatrixTileNotReady("matrix tile is not prepared")

    def build_now(self) -> None:
        with self._run_lock:
            while True:
                generation = self._generation
                self._build(generation)
                if generation == self._generation:
                    return

    def ensure_started(self, *, force: bool = False) -> None:
        with self._lock:
            if not force and self._marker().is_file() and self._urgent.empty():
                return
            if self._thread is not None and self._thread.is_alive():
                return
            self._thread = threading.Thread(target=self._execute, name="matrix-tiles", daemon=True)
            self._thread.start()

    def _execute(self) -> None:
        if not self._run_lock.acquire(blocking=False):
            return
        try:
            while True:
                generation = self._generation
                self._build(generation)
                if generation == self._generation:
                    return
        except Exception:
            _LOGGER.exception("Matrix tile preparation failed in %s", self.directory)
        finally:
            self._run_lock.release()

    def _build(self, generation: int) -> None:
        if not self._indexed:
            stamp, width, height, cells = self._load_index()
            if generation != self._generation:
                return
            self._stamp = str(stamp)
            self._width = int(width)
            self._height = int(height)
            self._cells = tuple(cells)
            self._indexed = True
            self._publish(self._token())
        if generation != self._generation:
            return
        width, height, cells = self._width, self._height, self._cells
        self._drain_urgent()
        if self._marker().is_file() and self._urgent.empty():
            return
        for tile_cells in pyramid_cell_sizes(width, height):
            for orientation in ("y_down", "y_up"):
                for rect in iter_tile_rects(width, height, tile_cells, orientation):
                    if generation != self._generation:
                        return
                    self._drain_urgent()
                    self._write_tile(orientation, *rect, cells=cells)
        self._drain_urgent()
        if generation != self._generation:
            return
        _atomic_write(self._marker(), b"ok")
        _LOGGER.info("Matrix tiles ready at %s (%sx%s, %s cells)", self.directory, width, height, len(cells))

    def _drain_urgent(self) -> None:
        while True:
            try:
                orientation, x1, y1, x2, y2 = self._urgent.get_nowait()
            except queue.Empty:
                return
            self._write_tile(orientation, x1, y1, x2, y2, cells=self._cells)

    def _write_tile(
        self,
        orientation: str,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        *,
        cells: tuple[Mapping[str, object], ...] | None = None,
    ) -> None:
        token = self._published_token() or self._token()
        path = self._tile_path(token, orientation, x1, y1, x2, y2)
        if path.is_file():
            return
        images: dict[tuple[int, int], bytes] = {}
        for cell in self._cells if cells is None else cells:
            x = int(cell["x"])
            y = int(cell["y"])
            if not (int(x1) <= x <= int(x2) and int(y1) <= y <= int(y2)):
                continue
            preview = self._preview(cell)
            if preview:
                images[mosaic_slot(x, y, x1=x1, y1=y1, x2=x2, y2=y2, orientation=orientation)] = preview
        _atomic_write(
            path,
            compose_mosaic(
                width_cells=int(x2) - int(x1) + 1,
                height_cells=int(y2) - int(y1) + 1,
                images=images,
            ),
        )

    def _preview(self, cell: Mapping[str, object]) -> bytes | None:
        digest = str(cell.get("asset_sha256") or cell.get("sha256") or "")
        path = self._preview_path(digest) if digest else None
        if path is not None and path.is_file():
            return path.read_bytes()
        raw = self._read_image(cell)
        if not raw:
            return None
        small = preview_png(raw)
        if small and path is not None:
            _atomic_write(path, small)
        return small

    def _preview_path(self, digest: str) -> Path:
        safe = "".join(character for character in digest.lower() if character in "0123456789abcdef")
        if len(safe) < 4:
            safe = hashlib.sha256(digest.encode()).hexdigest()
        return self.directory / "previews" / safe[:2] / safe

    def _token(self) -> str:
        return hashlib.sha256(self._stamp.encode()).hexdigest()

    def _current_path(self) -> Path:
        return self.directory / "current"

    def _marker(self) -> Path:
        return self.directory / "complete"

    def _published_token(self) -> str | None:
        path = self._current_path()
        if not path.is_file():
            return None
        token = path.read_text(encoding="utf-8").strip()
        return token or None

    def _publish(self, token: str) -> None:
        _atomic_write(self._current_path(), token.encode("utf-8"))

    def _tile_path(self, token: str, orientation: str, x1: int, y1: int, x2: int, y2: int) -> Path:
        name = f"{int(x1)}_{int(y1)}_{int(x2)}_{int(y2)}.png"
        return self.directory / token / str(orientation) / name


def pyramid_for(
    owner: object,
    key: tuple[object, ...],
    load_index: Callable[[], tuple[str, int, int, tuple[Mapping[str, object], ...]]],
    read_image: Callable[[Mapping[str, object]], bytes | None],
) -> PreparedTilePyramid:
    store = getattr(owner, "_prepared_pyramids", None)
    if not isinstance(store, dict):
        store = {}
        setattr(owner, "_prepared_pyramids", store)
    found = store.get(key)
    if isinstance(found, PreparedTilePyramid):
        return found
    directory = tile_root_for(owner) / _directory_name(key)
    found = PreparedTilePyramid(directory, load_index, read_image)
    store[key] = found
    return found


def schedule_key(project_id: object, layer_id: object, representation_ids: Iterable[object]) -> tuple[object, ...]:
    return (str(project_id), str(layer_id), tuple(sorted(str(value) for value in representation_ids)))
