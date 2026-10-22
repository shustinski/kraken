"""Recent selections of the home page: snapshot folders computed and exported, with how far their
matrix is. Qt-free.

A selection is a snapshot folder with the format and the folder its error maps go to, and what its
last run left: done (a project that opens at once), stopped part way, failed, or not run yet. A
project saved from the matrix counts too: its layers' snapshot folder is a selection with that
project. Picking a selection on the home page puts its paths back; its matrix button tells how
ready the matrix is.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path

from .snapshot_layout import discover_snapshot

RECENT_LIMIT = 20

RUN_DONE = "done"
RUN_STOPPED = "stopped"
RUN_FAILED = "failed"

# How ready the matrix of a selection is.
MATRIX_EMPTY = "empty"  # no snapshot chosen: an empty matrix
MATRIX_NONE = "none"  # nothing computed: the matrix computes on opening
MATRIX_RUNNING = "running"
MATRIX_PARTIAL = "partial"  # stopped or failed part way: the rest is computed on opening
MATRIX_READY = "ready"  # a project: opens at once


@dataclass(slots=True)
class Selection:
    snapshot: str
    errors_format: str = "png"
    output_dir: str = ""
    project: str = ""
    run: str = ""  # RUN_DONE / RUN_STOPPED / RUN_FAILED, empty before the first run
    share: float = 0.0
    stage: str = ""  # the stage a stopped run was at
    frames: int = 0
    written: int = 0
    problems: int = 0
    defects: int = -1  # defect cells of a done run, -1 when not counted
    seconds: float = 0.0
    at: str = ""  # when the last run ended (ISO)

    @classmethod
    def from_dict(cls, payload: dict) -> "Selection | None":
        if not isinstance(payload, dict) or not str(payload.get("snapshot") or ""):
            return None
        known = {item.name for item in fields(cls)}
        values = {key: value for key, value in payload.items() if key in known}
        try:
            return cls(**values)
        except TypeError:
            return None

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def name(self) -> str:
        return Path(self.snapshot).name or self.snapshot


def same_folder(a: str | Path, b: str | Path) -> bool:
    return _folder_key(a) == _folder_key(b)


def _folder_key(path: str | Path) -> str:
    return str(path or "").strip().replace("\\", "/").rstrip("/").lower()


def load(payload: dict) -> list[Selection]:
    items = payload.get("selections") if isinstance(payload, dict) else None
    selections = [Selection.from_dict(item) for item in items or () if isinstance(item, dict)]
    return [selection for selection in selections if selection is not None]


def dump(selections: list[Selection], seconds_per_frame: float = 0.0) -> dict:
    return {"selections": [selection.to_dict() for selection in selections], "seconds_per_frame": float(seconds_per_frame)}


def find(selections: list[Selection], snapshot: str | Path) -> Selection | None:
    return next((selection for selection in selections if same_folder(selection.snapshot, snapshot)), None)


def remember(selections: list[Selection], selection: Selection, limit: int = RECENT_LIMIT) -> list[Selection]:
    """``selection`` first, its older copy gone, the list cut to ``limit``."""

    rest = [item for item in selections if not same_folder(item.snapshot, selection.snapshot)]
    return [selection] + rest[: max(0, limit - 1)]


def forget(selections: list[Selection], snapshot: str | Path) -> list[Selection]:
    return [item for item in selections if not same_folder(item.snapshot, snapshot)]


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def snapshot_root(mask_folders: list[str | Path]) -> Path | None:
    """The snapshot folder of a matrix's layers: the folder holding them when it reads as a
    snapshot with these layers (``binary_cif`` and ``binary_cif_inv`` in one folder), else the
    layer's own folder."""

    folders = [Path(folder) for folder in mask_folders if str(folder or "")]
    if not folders:
        return None
    parents = {_folder_key(folder.parent) for folder in folders}
    if len(parents) == 1:
        parent = folders[0].parent
        found = {_folder_key(layer.mask_folder) for layer in discover_snapshot(parent).layers}
        if found and all(_folder_key(folder) in found for folder in folders):
            return parent
    return folders[0]


def matrix_state(selection: Selection | None, *, running: bool = False, project_exists: bool = False) -> str:
    if selection is None:
        return MATRIX_EMPTY
    if running:
        return MATRIX_RUNNING
    if selection.run == RUN_DONE and selection.project and project_exists:
        return MATRIX_READY
    if selection.run in (RUN_STOPPED, RUN_FAILED) and selection.share > 0:
        return MATRIX_PARTIAL
    return MATRIX_NONE
