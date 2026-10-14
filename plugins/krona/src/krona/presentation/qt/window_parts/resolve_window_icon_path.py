"""window: resolve window icon path."""
from __future__ import annotations
from pathlib import Path
import sys
from kraken_core.qt import resolve_icon_path

def _resolve_window_icon_path() -> Path | None:
    candidates: list[Path] = []
    shared_icon = resolve_icon_path("krona")
    if shared_icon is not None:
        candidates.append(shared_icon)
    frozen_base = getattr(sys, "_MEIPASS", None)
    if frozen_base:
        candidates.append(Path(frozen_base) / "resources" / "icons" / "krona.ico")
    candidates.append(Path(__file__).resolve().parents[4] / "resources" / "icons" / "app_icon.ico")
    candidates.append(Path.cwd() / "resources" / "icons" / "app_icon.ico")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None
