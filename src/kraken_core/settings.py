"""Qt settings with an explicit, opt-in directory for isolated GUI instances."""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QSettings as NativeQSettings

_directory: Path | None = None


def configure_settings_directory(directory: str | Path) -> Path:
    global _directory
    root = Path(directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    _directory = root
    return root


class QSettings(NativeQSettings):
    def __init__(self, *args, **kwargs):
        if (_directory is not None and len(args) in (2, 3)
                and isinstance(args[0], str) and isinstance(args[1], str)):
            organization, application = args[:2]
            if any(part in {"", ".", ".."} or "/" in part or "\\" in part for part in (organization, application)):
                raise ValueError("Settings organization/application must be plain names")
            path = _directory / organization / (application + ".ini")
            path.parent.mkdir(parents=True, exist_ok=True)
            parent = args[2] if len(args) == 3 else kwargs.pop("parent", None)
            if kwargs:
                raise TypeError("Unsupported settings constructor keyword")
            super().__init__(str(path), NativeQSettings.Format.IniFormat, parent)
            self.setFallbacksEnabled(False)
        else:
            super().__init__(*args, **kwargs)
