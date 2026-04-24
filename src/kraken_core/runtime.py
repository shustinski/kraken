from __future__ import annotations

import os
import sys
from pathlib import Path


def current_platform() -> str:
    if sys.platform.startswith("win"):
        return "windows"
    if sys.platform.startswith("linux"):
        return "linux"
    if sys.platform == "darwin":
        return "macos"
    return sys.platform


def workspace_root(start: Path | None = None) -> Path:
    path = (start or Path.cwd()).resolve()
    for candidate in (path, *path.parents):
        if (candidate / "plugins").is_dir() and (candidate / "src").is_dir():
            return candidate
    return path


def package_runtime_root(package_file: str | Path, *, plugin_root_name: str | None = None) -> Path:
    if bool(getattr(sys, "frozen", False)):
        executable_dir = Path(sys.executable).resolve().parent
        internal_dir = executable_dir / "_internal"
        if internal_dir.exists():
            return internal_dir
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            return Path(meipass)
        return executable_dir

    path = Path(package_file).resolve()
    if plugin_root_name:
        for parent in path.parents:
            if parent.name == plugin_root_name and parent.parent.name == "plugins":
                return parent
    for parent in path.parents:
        if parent.name == "src":
            return parent.parent
    return path.parent


def env_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return Path(value).expanduser().resolve() if value else default
