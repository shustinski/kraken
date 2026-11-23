"""Packed thumbnail tiles. One image covers many matrix cells."""

from __future__ import annotations

import io
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping

from PIL import Image, UnidentifiedImageError

MOSAIC_EDGE = 256
_CACHE_LIMIT = 128


def mosaic_slot(
    x: int,
    y: int,
    *,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    orientation: str,
) -> tuple[int, int]:
    """Return the cell's position inside a tile image.

    The first row of the image is the top of the graphics tile. With Y increasing
    upward, that top row is the larger Y.
    """

    slot_x = int(x) - int(x1)
    if orientation == "y_up":
        slot_y = int(y2) - int(y)
    else:
        slot_y = int(y) - int(y1)
    return slot_x, slot_y


def compose_mosaic(
    *,
    width_cells: int,
    height_cells: int,
    images: Mapping[tuple[int, int], bytes],
    edge: int = MOSAIC_EDGE,
) -> bytes:
    """Paste source images into one transparent PNG.

    Keys are zero-based slots. Cells without an image stay transparent so the
    status color painted underneath remains visible. A broken source is skipped.
    """

    width = max(1, int(width_cells))
    height = max(1, int(height_cells))
    slot = max(1, int(edge) // max(width, height))
    canvas = Image.new("RGBA", (width * slot, height * slot), (0, 0, 0, 0))
    for (slot_x, slot_y), payload in images.items():
        if not (0 <= int(slot_x) < width and 0 <= int(slot_y) < height) or not payload:
            continue
        try:
            with Image.open(io.BytesIO(payload)) as source:
                frame = _fit_to_slot(source.convert("RGBA"), slot)
        except (UnidentifiedImageError, OSError, ValueError):
            continue
        left = int(slot_x) * slot + (slot - frame.width) // 2
        top = int(slot_y) * slot + (slot - frame.height) // 2
        canvas.paste(frame, (left, top), frame)
    output = io.BytesIO()
    canvas.save(output, "PNG")
    return output.getvalue()


def _fit_to_slot(frame: Image.Image, slot: int) -> Image.Image:
    """Scale up or down so the longer side fills the slot, keeping the aspect ratio.

    ``Image.thumbnail`` never enlarges, so a preview smaller than the slot would
    sit as a small square in the middle of the cell.
    """

    width, height = frame.size
    if width <= 0 or height <= 0:
        return frame
    scale = slot / max(width, height)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    if size == frame.size:
        return frame
    return frame.resize(size, Image.Resampling.LANCZOS)


def cached_mosaic(owner: object, key: tuple[object, ...], build: Callable[[], bytes]) -> bytes:
    cache = getattr(owner, "_mosaic_cache", None)
    if not isinstance(cache, MosaicCache):
        cache = MosaicCache()
        setattr(owner, "_mosaic_cache", cache)
    found = cache.get(key)
    if found is not None:
        return found
    payload = build()
    cache.put(key, payload)
    return payload


def mosaic_from_cells(
    cells: Iterable[Mapping[str, object]],
    *,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    orientation: str,
    read_image: Callable[[Mapping[str, object]], bytes | None],
) -> bytes:
    """Pack the images inside one tile. Cells outside the tile are ignored."""

    images: dict[tuple[int, int], bytes] = {}
    for cell in cells:
        x = int(cell["x"])
        y = int(cell["y"])
        if not (int(x1) <= x <= int(x2) and int(y1) <= y <= int(y2)):
            continue
        payload = read_image(cell)
        if payload:
            images[mosaic_slot(x, y, x1=x1, y1=y1, x2=x2, y2=y2, orientation=orientation)] = payload
    return compose_mosaic(width_cells=int(x2) - int(x1) + 1, height_cells=int(y2) - int(y1) + 1, images=images)


class MosaicCache:
    """Remember composed tiles until their source revision changes."""

    def __init__(self, limit: int = _CACHE_LIMIT) -> None:
        self._items: OrderedDict[tuple[object, ...], bytes] = OrderedDict()
        self._limit = max(1, int(limit))

    def get(self, key: tuple[object, ...]) -> bytes | None:
        payload = self._items.get(key)
        if payload is None:
            return None
        self._items.move_to_end(key)
        return payload

    def put(self, key: tuple[object, ...], payload: bytes) -> None:
        self._items[key] = payload
        self._items.move_to_end(key)
        while len(self._items) > self._limit:
            self._items.popitem(last=False)
