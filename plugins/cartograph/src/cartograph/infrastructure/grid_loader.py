"""Filesystem and JSON-manifest tile grid loading."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Mapping

import cv2

from cartograph.domain.coordinates import GridCoordinate, StageCoordinate
from cartograph.domain.errors import GridLoadError
from cartograph.domain.tiles import Tile, TileGrid

_LOGGER = logging.getLogger(__name__)

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
# Matches Kraken ImportPlanner XY_FILENAME: x is column, y is row.
_XY_FILENAME = re.compile(r"(?<!\d)(?P<x>\d+)_(?P<y>\d+)(?!\d)")
_GRID_MANIFEST_SCHEMA = "cartograph.grid.v1"


def load_tile_grid(path: Path, *, overlap_x: float = 0.1, overlap_y: float = 0.1) -> TileGrid:
    source = Path(path)
    if source.is_file():
        return _load_from_manifest(source)
    if source.is_dir():
        manifest = source / "grid.json"
        if manifest.is_file():
            return _load_from_manifest(manifest, overlap_fallback=(overlap_x, overlap_y))
        return _load_from_directory(source, overlap_x=overlap_x, overlap_y=overlap_y)
    raise GridLoadError(f"grid path does not exist: {source}")


def _load_from_manifest(path: Path, overlap_fallback: tuple[float, float] | None = None) -> TileGrid:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GridLoadError(f"cannot read grid manifest {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise GridLoadError("grid manifest must be a JSON object")
    schema = str(payload.get("schema", _GRID_MANIFEST_SCHEMA))
    if schema != _GRID_MANIFEST_SCHEMA:
        raise GridLoadError(f"unsupported grid schema: {schema}")
    overlap_x = float(payload.get("overlap_x", overlap_fallback[0] if overlap_fallback else 0.1))
    overlap_y = float(payload.get("overlap_y", overlap_fallback[1] if overlap_fallback else 0.1))
    root = path.parent
    raw_tiles = payload.get("tiles")
    if raw_tiles is None:
        return _load_from_directory(root, overlap_x=overlap_x, overlap_y=overlap_y)
    if not isinstance(raw_tiles, list) or not raw_tiles:
        raise GridLoadError("grid manifest tiles must be a non-empty array")
    tiles: dict[GridCoordinate, Tile] = {}
    for item in raw_tiles:
        if not isinstance(item, Mapping):
            raise GridLoadError("each tile entry must be an object")
        tile = _tile_from_manifest_item(item, root)
        if tile.coord in tiles:
            raise GridLoadError(f"duplicate tile at {tile.coord}")
        tiles[tile.coord] = tile
    _LOGGER.info("loaded %s tiles from manifest %s", len(tiles), path)
    return TileGrid(tiles=tiles, overlap_x=overlap_x, overlap_y=overlap_y, name=str(payload.get("name", path.parent.name)))


def _tile_from_manifest_item(item: Mapping[str, Any], root: Path) -> Tile:
    relative = str(item.get("path", "")).strip()
    if not relative:
        raise GridLoadError("tile path is required")
    path = (root / relative).resolve() if not Path(relative).is_absolute() else Path(relative)
    if not path.is_file():
        raise GridLoadError(f"tile file is missing: {path}")
    width, height = _probe_size(path)
    row = item.get("row")
    col = item.get("col")
    if row is None or col is None:
        parsed = _parse_xy_stem(path.stem)
        if parsed is None:
            raise GridLoadError(f"tile {path.name} has no row/col")
        col, row = parsed
        coord = _normalize_filename_coord(col, row, one_based=True)
    else:
        coord = GridCoordinate(int(row), int(col))
    return Tile(
        coord=coord,
        source_id=str(path),
        width=width,
        height=height,
        path=path,
        stage=_optional_stage(item.get("stage")),
        metadata={str(key): str(value) for key, value in item.items() if key not in {"path", "row", "col", "stage"}},
    )


def _load_from_directory(folder: Path, *, overlap_x: float, overlap_y: float) -> TileGrid:
    files = sorted(
        path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
    )
    parsed: list[tuple[int, int, Path]] = []
    for path in files:
        xy = _parse_xy_stem(path.stem)
        if xy is None:
            _LOGGER.debug("skipping non-grid filename %s", path.name)
            continue
        parsed.append((xy[0], xy[1], path))
    if not parsed:
        raise GridLoadError(f"no x_y image files found in {folder}")
    min_x = min(item[0] for item in parsed)
    min_y = min(item[1] for item in parsed)
    one_based = min_x >= 1 and min_y >= 1
    tiles: dict[GridCoordinate, Tile] = {}
    for x_value, y_value, path in parsed:
        coord = _normalize_filename_coord(x_value, y_value, one_based=one_based)
        if coord in tiles:
            raise GridLoadError(f"duplicate coordinate {coord} from {path.name}")
        width, height = _probe_size(path)
        tiles[coord] = Tile(
            coord=coord,
            source_id=str(path),
            width=width,
            height=height,
            path=path,
            stage=_load_stage_sidecar(path),
        )
    _LOGGER.info("loaded %s tiles from directory %s", len(tiles), folder)
    return TileGrid(tiles=tiles, overlap_x=overlap_x, overlap_y=overlap_y, name=folder.name)


def _normalize_filename_coord(x_value: int, y_value: int, *, one_based: bool) -> GridCoordinate:
    """Filename ``x_y`` follows Kraken: x is column, y is row."""

    if one_based:
        return GridCoordinate.from_kraken_xy(x_value, y_value)
    return GridCoordinate(row=y_value, col=x_value)


def _parse_xy_stem(stem: str) -> tuple[int, int] | None:
    match = _XY_FILENAME.search(stem)
    if match is None:
        return None
    return int(match["x"]), int(match["y"])


def _optional_stage(payload: object) -> StageCoordinate | None:
    if payload is None:
        return None
    if not isinstance(payload, Mapping) or "x" not in payload or "y" not in payload:
        raise GridLoadError("stage must be an object with x and y")
    return StageCoordinate(float(payload["x"]), float(payload["y"]))


def _load_stage_sidecar(image_path: Path) -> StageCoordinate | None:
    sidecar = image_path.with_suffix(image_path.suffix + ".stage.json")
    if not sidecar.is_file():
        sidecar = image_path.with_name(image_path.stem + ".stage.json")
    if not sidecar.is_file():
        return None
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GridLoadError(f"cannot read stage sidecar {sidecar}: {exc}") from exc
    return _optional_stage(payload)


def _probe_size(path: Path) -> tuple[int, int]:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise GridLoadError(f"cannot decode image: {path}")
    height, width = image.shape[:2]
    return int(width), int(height)
