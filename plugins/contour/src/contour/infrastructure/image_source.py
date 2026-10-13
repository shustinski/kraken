"""Adapt an explicit asset stream to Contour's existing image-loader port."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from kraken_core.asset_source import AssetSource

from contour.utils import load_image_color


@dataclass(frozen=True)
class AssetImageLoader:
    source: AssetSource
    logical_root: Path

    def __call__(self, path: str) -> np.ndarray:
        relative = Path(path).relative_to(self.logical_root).as_posix()
        with self.source.open(relative) as stream:
            return load_image_color(stream)
