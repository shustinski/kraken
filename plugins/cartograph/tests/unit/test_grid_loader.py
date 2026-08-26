from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from cartograph.infrastructure.grid_loader import load_tile_grid

from .helpers import unique_scene


def test_directory_loader_maps_kraken_xy_filenames(tmp_path: Path) -> None:
    image = np.clip(unique_scene(32, 40, seed=3), 0, 255).astype(np.uint8)
    cv2.imwrite(str(tmp_path / "frame_1_2.png"), image)
    cv2.imwrite(str(tmp_path / "frame_2_2.png"), image)
    grid = load_tile_grid(tmp_path, overlap_x=0.1, overlap_y=0.1)
    assert grid.col_count == 2
    assert grid.row_count == 2
    assert grid.get(grid.tiles[next(iter(grid.tiles))].coord) is not None
    coords = set(grid.tiles)
    # 1_2 → x=1,y=2 → col=0,row=1; 2_2 → col=1,row=1
    from cartograph.domain.coordinates import GridCoordinate

    assert GridCoordinate(1, 0) in coords
    assert GridCoordinate(1, 1) in coords


def test_manifest_loader_reads_optional_stage(tmp_path: Path) -> None:
    image = np.clip(unique_scene(16, 16, seed=5), 0, 255).astype(np.uint8)
    cv2.imwrite(str(tmp_path / "a.png"), image)
    manifest = {
        "schema": "cartograph.grid.v1",
        "overlap_x": 0.2,
        "overlap_y": 0.2,
        "tiles": [{"path": "a.png", "row": 0, "col": 0, "stage": {"x": 1.5, "y": 2.5}}],
    }
    path = tmp_path / "grid.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    grid = load_tile_grid(path)
    tile = next(iter(grid.tiles.values()))
    assert tile.stage is not None
    assert tile.stage.x == 1.5
    assert tile.width == 16
