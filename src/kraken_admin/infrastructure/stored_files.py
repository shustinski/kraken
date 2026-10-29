"""Move or remove a stored path only after deletion is authorized."""

from __future__ import annotations

import shutil
from pathlib import Path

from kraken_admin.domain.deletion import DeletionAuthorization


def stage_stored_path(source: Path, trash: Path, authorization: DeletionAuthorization) -> None:
    authorization.require()
    trash.parent.mkdir(parents=True, exist_ok=True)
    source.rename(trash)


def remove_stored_path(path: Path, authorization: DeletionAuthorization) -> None:
    authorization.require()
    if not path.exists():
        return
    if path.is_symlink() or not path.is_dir():
        path.unlink()
        return
    shutil.rmtree(path)
