"""Preparing a 20 000-frame matrix stays fast (writes 80 000 files: about two minutes on Windows)."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import pytest

from karakal.core.analytics import collect_frame_records
from karakal.core.domain import BuildOptions, ModelSpec


def _write_layers(root: Path, frames: int) -> tuple[Path, Path, Path, Path]:
    folders = [root / "m1", root / "m2", root / "c1", root / "c2"]
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)
    blob = cv2.imencode(".png", np.zeros((8, 8), dtype=np.uint8))[1].tobytes()
    for index in range(frames):
        name = f"frame_{index:05d}.png"
        for folder in folders:
            (folder / name).write_bytes(blob)
    return folders[0], folders[1], folders[2], folders[3]


@pytest.mark.performance
def test_prepare_20000_frames_with_confidence_stays_fast(tmp_path: Path) -> None:
    m1, m2, c1, c2 = _write_layers(tmp_path, 20_000)
    specs = (
        ModelSpec(model_id="m1", display_name="M1", mask_folder=m1, prob_folder=c1),
        ModelSpec(model_id="m2", display_name="M2", mask_folder=m2, prob_folder=c2),
    )
    started = perf_counter()
    result = collect_frame_records(specs, BuildOptions())
    elapsed = perf_counter() - started
    assert len(result.records) == 20_000
    # Local SSD budget: previously ~40s due to per-key Path.resolve(); target is seconds.
    assert elapsed < 25.0, f"prepare took {elapsed:.2f}s"
