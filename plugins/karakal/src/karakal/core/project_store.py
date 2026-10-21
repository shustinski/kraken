"""Karakal projects: a computed run saved as it is, opened again without recomputing. Qt-free.

A project is a folder ``<name>.karakal``:

- ``project.json``: what the run is (layers and their folders, settings, algorithm, frame count)
  and how it was looked at (matrix layer, drop rules, frame sets, dirt settings). Small; written
  on every save, also when the app closes.
- ``results.pickle``: the computed run (frames of the run, cell defects of every layer, pair
  conflicts, the normal-cell banks, dirty regions, the memory map). Written after a run.
- ``view.pickle``: how the run is looked at (layer of the matrix, drop rules, frame sets, dropped
  frames, block roles). Small; written with every save and when the app closes.

Projects travel between machines. Every folder of the run is kept twice: as it was and relative
to the project folder. On another machine whichever of the two exists is used; when neither
does, the run still opens (the frames' masks are just not there to look at). A layer's id is
made from its folder path, so when the folders move the ids change too: ``relocate_state``
moves paths and ids together.

The results file is a pickle: open only projects that come from people you trust.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
import sys
from collections.abc import Callable
from dataclasses import fields, is_dataclass, replace
from datetime import datetime
from pathlib import Path

PROJECT_SUFFIX = ".karakal"
PROJECT_SCHEMA = "karakal.project.v1"
MANIFEST_FILE = "project.json"
RESULTS_FILE = "results.pickle"
VIEW_FILE = "view.pickle"


# Folders --------------------------------------------------------------------------------------


def default_projects_dir() -> Path:
    """``projects`` beside the installed karakal.exe; from source, ``KARAKAL_PROJECTS_DIR`` or the
    working folder. Falls back to the user's local app data when the install folder is read-only."""

    override = str(os.environ.get("KARAKAL_PROJECTS_DIR", "") or "").strip()
    if override:
        candidates = [Path(override)]
    elif getattr(sys, "frozen", False):
        candidates = [Path(sys.executable).resolve().parent / "projects"]
    else:
        candidates = [Path.cwd() / "projects"]
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
    if base:
        candidates.append(Path(base) / "Karakal" / "projects")
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / ".write_probe"
            probe.write_text("", encoding="utf-8")
            probe.unlink()
            return candidate
        except OSError:
            continue
    return candidates[0]


def safe_project_name(name: str, fallback: str = "Проект") -> str:
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', " ", str(name or ""))
    text = re.sub(r"\s+", " ", text).strip().strip(".")
    return text or fallback


def project_folder(parent: Path | str, name: str) -> Path:
    """A new project folder under ``parent``: ``<name>.karakal``, numbered when taken."""

    stem = safe_project_name(name)
    candidate = Path(parent) / f"{stem}{PROJECT_SUFFIX}"
    index = 2
    while candidate.exists():
        candidate = Path(parent) / f"{stem} ({index}){PROJECT_SUFFIX}"
        index += 1
    return candidate


def is_project(path: Path | str) -> bool:
    return (Path(path) / MANIFEST_FILE).is_file()


def find_projects(folder: Path | str) -> list[Path]:
    root = Path(folder)
    if not root.is_dir():
        return []
    return sorted((item for item in root.iterdir() if item.suffix == PROJECT_SUFFIX and is_project(item)), key=str)


# Paths ----------------------------------------------------------------------------------------


def relative_path(path: str, project_dir: Path) -> str:
    """``path`` relative to the project folder; empty across drives."""

    if not path:
        return ""
    try:
        return os.path.relpath(str(path), str(project_dir))
    except ValueError:
        return ""


def stable_model_id(mask_folder: Path | str | None, confidence_folder: Path | str | None = None) -> str:
    """A layer's id from its absolute folder path (the presenter makes ids the same way)."""

    seed = ""
    if mask_folder:
        seed = str(Path(mask_folder).resolve())
    elif confidence_folder:
        seed = str(Path(confidence_folder).resolve())
    seed = seed.strip().lower().replace("\\", "/")
    if not seed:
        return ""
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:10]
    leaf = Path(seed).name
    leaf = re.sub(r"[^a-zA-Z0-9_]+", "_", leaf).strip("_") or "model"
    return f"{leaf}_{digest}"


def resolve_folder(stored: str, relative: str, project_dir: Path) -> str:
    """Where a folder of the run is now: as stored, else relative to the project, else as stored."""

    if stored and Path(stored).exists():
        return stored
    if relative:
        candidate = os.path.normpath(os.path.join(str(project_dir), relative))
        if Path(candidate).exists():
            return candidate
    return stored


# Writing and reading --------------------------------------------------------------------------


def _write_atomic(path: Path, write: Callable[[object], None], mode: str = "wb") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    encoding = None if "b" in mode else "utf-8"
    with open(temporary, mode, encoding=encoding) as handle:
        write(handle)
    os.replace(temporary, path)


def write_manifest(folder: Path | str, manifest: dict) -> Path:
    path = Path(folder) / MANIFEST_FILE
    payload = dict(manifest)
    payload["schema"] = PROJECT_SCHEMA
    payload["saved_at"] = datetime.now().isoformat(timespec="seconds")
    _write_atomic(path, lambda handle: json.dump(payload, handle, ensure_ascii=False, indent=1), "w")
    return path


def write_results(folder: Path | str, results: dict, name: str = RESULTS_FILE) -> Path:
    path = Path(folder) / name
    _write_atomic(path, lambda handle: pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL))
    return path


def read_manifest(folder: Path | str) -> dict:
    payload = json.loads((Path(folder) / MANIFEST_FILE).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != PROJECT_SCHEMA:
        raise ValueError(f"not a Karakal project: {folder}")
    return payload


def read_results(folder: Path | str, name: str = RESULTS_FILE) -> dict:
    path = Path(folder) / name
    if name != RESULTS_FILE and not path.is_file():
        return {}
    with path.open("rb") as handle:
        results = pickle.load(handle)
    if not isinstance(results, dict):
        raise ValueError(f"damaged project results: {folder}")
    return results


# Moving paths and ids -------------------------------------------------------------------------


def _replace_text(text: str, path_moves: list[tuple[str, str]], id_moves: dict[str, str]) -> str:
    for old, new in path_moves:
        if text == old or text.startswith(old + "\\") or text.startswith(old + "/"):
            text = new + text[len(old) :]
            break
    for old, new in id_moves.items():
        if old in text:
            text = text.replace(old, new)
    return text


# Dataclasses that carry folder paths or layer ids. Computed results (cell defects, banks, maps)
# keep their own copies only for information and are not walked: that would cost seconds on a
# big run and could re-run their packing.
_RELOCATED_TYPES = frozenset(
    {"BuildResult", "FrameRecord", "ModelSpec", "FrameDirtResult", "FrameSetModel", "FrameFilterRule", "FrameSet"}
)


def relocate_state(value, path_moves: list[tuple[str, str]], id_moves: dict[str, str], _seen: dict | None = None):
    """``value`` with folder prefixes moved (old -> new) and layer ids renamed, through dicts, lists,
    tuples, sets and dataclasses. Objects of other kinds (numpy arrays, images) are kept as they are."""

    if not path_moves and not id_moves:
        return value
    seen = {} if _seen is None else _seen
    marker = id(value)
    if marker in seen:
        return seen[marker]

    def move(item):
        return relocate_state(item, path_moves, id_moves, seen)

    if isinstance(value, str):
        return _replace_text(value, path_moves, id_moves)
    if isinstance(value, Path):
        return Path(_replace_text(str(value), path_moves, id_moves))
    if isinstance(value, dict):
        moved = {move(key): move(item) for key, item in value.items()}
    elif isinstance(value, list):
        moved = [move(item) for item in value]
    elif isinstance(value, tuple) and not hasattr(value, "_fields"):
        moved = tuple(move(item) for item in value)
    elif isinstance(value, (set, frozenset)):
        moved = type(value)(move(item) for item in value)
    elif is_dataclass(value) and not isinstance(value, type) and type(value).__name__ in _RELOCATED_TYPES:
        changes = {}
        for item in fields(value):
            if not item.init:
                continue
            current = getattr(value, item.name)
            if isinstance(current, (str, Path, dict, list, tuple, set, frozenset)) or is_dataclass(current):
                updated = move(current)
                if updated is not current:
                    changes[item.name] = updated
        try:
            moved = replace(value, **changes) if changes else value
        except (TypeError, ValueError):
            moved = value
    else:
        return value
    seen[marker] = moved
    return moved


# Slim results -----------------------------------------------------------------------------------


def is_slim(result) -> bool:
    """A frame result saved by a project with its normal cells left out."""

    return bool(getattr(result, "cells_trimmed", False))


def slim_results(value, _seen: dict | None = None):
    """``value`` with every frame result (GridFrameAnalysisResult) keeping only its defect cells.

    A frame of the memory array has a thousand and more normal cells, each a Python object: read
    back, they cost ~13 ms a frame (minutes for a big run). The matrix, the error list, the drop
    rules, the export and the memory map read only the defects and the frame's counts; the frame
    window analyses its frame again when it needs the normal cells. Shared results stay shared."""

    seen = {} if _seen is None else _seen
    marker = id(value)
    if marker in seen:
        return seen[marker]
    if type(value).__name__ == "GridFrameAnalysisResult":
        cells = tuple(getattr(value, "per_cell_results", ()) or ())
        kept = tuple(cell for cell in cells if str(getattr(cell, "status", "")) != "normal")
        slim = value if len(kept) == len(cells) else replace(value, per_cell_results=kept, debug=None, cells_trimmed=True)
        seen[marker] = slim
        return slim
    if isinstance(value, dict):
        moved = {key: slim_results(item, seen) for key, item in value.items()}
    elif isinstance(value, list):
        moved = [slim_results(item, seen) for item in value]
    elif isinstance(value, tuple) and not hasattr(value, "_fields"):
        moved = tuple(slim_results(item, seen) for item in value)
    else:
        return value
    seen[marker] = moved
    return moved
