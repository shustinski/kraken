"""Move or remove a stored path only after deletion is authorized."""

from __future__ import annotations

from pathlib import Path

from kraken_admin.domain.deletion import DeletionAuthorization
from kraken_manager.infrastructure.workspace_files import remove_tree


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
    # Project folders contain an ``aux`` directory; only a DOS-name aware
    # removal (trailing-separator targets) can delete it on Windows.
    remove_tree(path)
