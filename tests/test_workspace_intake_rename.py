from __future__ import annotations

import os
from pathlib import Path

import pytest

from kraken_manager.infrastructure import workspace_files


@pytest.mark.skipif(os.name != "nt", reason="Windows sharing violations only")
def test_intake_rename_waits_out_scanners_holding_new_files(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "frame.png").write_bytes(b"png")
    destination = tmp_path / "project" / "m1"
    real_replace = os.replace
    failures = [3]

    def flaky(src, dst):
        if failures[0]:
            failures[0] -= 1
            error = PermissionError(13, "Отказано в доступе")
            error.winerror = 5
            raise error
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", flaky)

    workspace_files._copy_directory(source, destination)

    assert (destination / "frame.png").read_bytes() == b"png"
    assert not list(destination.parent.glob(".intake-*"))
