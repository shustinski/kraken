from __future__ import annotations

import numpy as np
import pytest

from cartograph.domain.coordinates import GridCoordinate, Translation2D
from cartograph.infrastructure.render import BlendMode, render_local_mosaic

from .helpers import buffer, tile_at


def test_renderer_places_originals_without_chaining() -> None:
    left = np.zeros((20, 20), dtype=np.float32)
    right = np.zeros((20, 20), dtype=np.float32)
    left[:, :] = 50
    right[:, :] = 200
    tiles = {
        GridCoordinate(0, 0): tile_at(0, 0, 20, 20),
        GridCoordinate(0, 1): tile_at(0, 1, 20, 20),
    }
    images = {
        GridCoordinate(0, 0): buffer(left),
        GridCoordinate(0, 1): buffer(right),
    }
    poses = {
        GridCoordinate(0, 0): Translation2D(0.0, 0.0),
        GridCoordinate(0, 1): Translation2D(20.0, 0.0),
    }
    mosaic = render_local_mosaic(tiles, images, poses, blend=BlendMode.HARD_SEAM)
    assert mosaic.width >= 40
    assert mosaic.pixels[10, 5] == pytest.approx(50, abs=1.0)
    assert mosaic.pixels[10, 30] == pytest.approx(200, abs=1.0)
