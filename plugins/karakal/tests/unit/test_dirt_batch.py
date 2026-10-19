"""Dirty regions of many frames: cached per frame, cancellable."""

from __future__ import annotations

import os

import cv2
import pytest

import karakal.core.dirt_batch as dirt_batch
from karakal.core.dirt_batch import DirtCancelled, DirtFrame, analyze_dirt_frames
from karakal.core.dirt_regions import DirtConfig

from test_dirt_regions import lattice, smear


def _write(path, image) -> str:
    cv2.imencode(".png", image)[1].tofile(str(path))
    return str(path)


@pytest.fixture
def frames(tmp_path):
    folder = tmp_path / "Маски"
    folder.mkdir()
    return [
        DirtFrame("A", _write(folder / "A.png", smear(lattice(), 380, 350, 70, 90))),
        DirtFrame("B", _write(folder / "B.png", lattice())),
        DirtFrame("C", str(folder / "missing.png")),
    ]


def test_regions_of_all_frames_and_the_cache(tmp_path, frames, monkeypatch) -> None:
    cache = tmp_path / "cache"
    seen = []
    results = analyze_dirt_frames(
        frames, use_processes=False, cache_dir=cache, progress=lambda done, total: seen.append((done, total))
    )
    assert set(results) == {"A", "B", "C"}
    assert len(results["A"].regions) == 1 and results["B"].regions == ()
    assert results["C"].error
    assert seen[-1] == (3, 3)

    # A second run reads A and B from the cache; the unreadable C is tried again.
    calls = []
    original = dirt_batch._analyze_one
    monkeypatch.setattr(dirt_batch, "_analyze_one", lambda frame, config: calls.append(frame.key) or original(frame, config))
    again = analyze_dirt_frames(frames, use_processes=False, cache_dir=cache)
    assert calls == ["C"]
    assert again["A"].regions == results["A"].regions

    # Other settings or a changed mask are analysed anew.
    calls.clear()
    analyze_dirt_frames(frames, DirtConfig(merge_gap_cells=1.0), use_processes=False, cache_dir=cache)
    assert sorted(calls) == ["A", "B", "C"]
    calls.clear()
    stat = os.stat(frames[1].mask_path)
    os.utime(frames[1].mask_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10_000_000))
    analyze_dirt_frames(frames, use_processes=False, cache_dir=cache)
    assert sorted(calls) == ["B", "C"]


def test_cancel_stops_the_run(frames) -> None:
    with pytest.raises(DirtCancelled):
        analyze_dirt_frames(frames, use_processes=False, cache_dir=None, cancelled=lambda: True)
