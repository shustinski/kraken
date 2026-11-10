"""Karakal's own folder: projects, profiles and exports in one place the user chooses. Qt-free.

    <data folder>\\
      Проекты\\<name>.karakal\\            a project (core.project_store)
        Выгрузки\\<run>\\                  what was exported from it
      Профили\\<name>.json                 saved analysis and export settings (core.profiles)
      Выгрузки\\<run>\\                    exports of runs not saved as a project

The snapshot folders themselves lie anywhere (another disk, the network): projects keep their
paths. The folder is set in Settings → Data folder; by default ``Documents\\Каракал``. Projects of
older versions lay in ``projects`` beside karakal.exe and are moved here once.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

PROJECTS = "Проекты"
PROFILES = "Профили"
EXPORTS = "Выгрузки"

_root: Path | None = None


def default_data_root() -> Path:
    override = str(os.environ.get("KARAKAL_DATA_DIR", "") or "").strip()
    if override:
        return Path(override)
    home = Path(os.environ.get("USERPROFILE") or Path.home())
    documents = home / "Documents"
    return (documents if documents.is_dir() else home) / "Каракал"


def set_data_root(path: Path | str | None) -> None:
    """The folder chosen in the settings; None (or empty) goes back to the default."""

    global _root
    _root = Path(path) if path else None


def _writable(folder: Path) -> bool:
    try:
        folder.mkdir(parents=True, exist_ok=True)
        probe = folder / ".write_probe"
        probe.write_text("", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


def data_root() -> Path:
    root = _root or default_data_root()
    if _writable(root):
        return root
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
    fallback = Path(base) / "Karakal" if base else root
    return fallback if _writable(fallback) else root


def _sub(name: str) -> Path:
    folder = data_root() / name
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return folder


def projects_dir() -> Path:
    return _sub(PROJECTS)


def profiles_dir() -> Path:
    return _sub(PROFILES)


def exports_dir(project: Path | str | None = None) -> Path:
    """Where exports go: into the project's own folder, else the data folder's."""

    if project:
        return Path(project) / EXPORTS
    return _sub(EXPORTS)


def legacy_projects_dir() -> Path | None:
    """``projects`` beside the installed karakal.exe, where versions before 0.4.98 kept projects."""

    if not getattr(sys, "frozen", False):
        return None
    folder = Path(sys.executable).resolve().parent / "projects"
    return folder if folder.is_dir() else None


def move_projects(source: Path, target: Path) -> dict[str, str]:
    """Move every project folder of ``source`` into ``target``; old path → new path of the moved ones.

    A project that cannot be moved (open elsewhere, no rights) stays where it is.
    """

    from .project_store import PROJECT_SUFFIX, find_projects, project_folder

    moved: dict[str, str] = {}
    if source is None or not Path(source).is_dir():
        return moved
    try:
        if Path(source).resolve() == Path(target).resolve():
            return moved
    except OSError:
        return moved
    Path(target).mkdir(parents=True, exist_ok=True)
    for folder in find_projects(source):
        destination = project_folder(target, folder.name[: -len(PROJECT_SUFFIX)])
        try:
            shutil.move(str(folder), str(destination))
        except (OSError, shutil.Error):
            continue
        moved[str(folder)] = str(destination)
    return moved
