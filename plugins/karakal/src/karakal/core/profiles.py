"""Profiles: the analysis and export settings saved under a name, to take up again at once. Qt-free.

A profile is ``<data folder>\\Профили\\<name>.json`` (core.data_folder). It keeps what decides the
result and what is exported: the sensitivities, the defect types, the debris size, the sparse cell
distance, the ones/zeros conflict thresholds, the format of the error pictures and the error types
of the export. Not the snapshot: which folders are what is told by the snapshot itself.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .project_store import safe_project_name

PROFILE_SCHEMA = "karakal.profile.v1"
PROFILE_SUFFIX = ".json"


def profile_path(name: str, folder: Path | None = None) -> Path:
    from .data_folder import profiles_dir

    return Path(folder or profiles_dir()) / f"{safe_project_name(name, 'Профиль')}{PROFILE_SUFFIX}"


def list_profiles(folder: Path | None = None) -> list[Path]:
    from .data_folder import profiles_dir

    root = Path(folder or profiles_dir())
    if not root.is_dir():
        return []
    return sorted((item for item in root.glob(f"*{PROFILE_SUFFIX}") if item.is_file()), key=lambda item: item.stem.lower())


def save_profile(name: str, settings: dict, folder: Path | None = None) -> Path:
    """Write the profile (over one of the same name)."""

    from ..version import APP_VERSION

    path = profile_path(name, folder)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": PROFILE_SCHEMA,
        "name": path.stem,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "app_version": APP_VERSION,
        "settings": dict(settings),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)
    return path


def read_profile(path: Path | str) -> dict:
    """The settings of a profile; ValueError when the file is not one."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema") != PROFILE_SCHEMA or not isinstance(data.get("settings"), dict):
        raise ValueError(f"not a Karakal profile: {path}")
    return dict(data["settings"])
