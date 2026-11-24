"""Local pause/resume state for a managed Contour batch.

The resume cursor is the set of valid CIFs in the job's output directory:
parallel workers finish out of frame order, so no numeric position is kept.
The state file only records which settings produced which frames. Settings
are captured before workers start; any valid CIF that appears afterwards
and was not attributed earlier came from the active snapshot. Frames whose
settings were never recorded (output written before this feature) keep an
explicit unknown provenance instead of borrowing the current settings.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

STATE_NAME = "contour-resume.json"
_VERSION = 1
_MAX_STATE_BYTES = 64 * 1024 * 1024


class LocalBatchResumeError(RuntimeError):
    """The saved local run cannot be trusted; original files are untouched."""


def is_complete_cif(path: Path) -> bool:
    """True for a regular CIF file written to its final ``E`` line."""
    try:
        if path.is_symlink() or not path.is_file():
            return False
        with path.open("rb") as stream:
            head = stream.read(512)
            size = stream.seek(0, os.SEEK_END)
            stream.seek(max(0, size - 16))
            tail = stream.read()
    except OSError:
        return False
    # Text-mode writers on Windows emit CRLF; accept either line ending.
    tail = tail.replace(b"\r\n", b"\n")
    return head.startswith(b"DS 1 1 1;") and b"( S " in head and tail.rstrip().endswith(b"DF;\nE")


def _canonical(settings: Mapping[str, Any]) -> str:
    return json.dumps(settings, sort_keys=True, separators=(",", ":"), default=str)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.part")
    try:
        with temporary.open("xb") as stream:
            stream.write(json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class LocalBatchResume:
    """Settings snapshots and frame provenance for one Kraken job on this machine."""

    def __init__(self, path: Path, data: Mapping[str, Any] | None = None) -> None:
        data = dict(data or {})
        self.path = Path(path)
        self.snapshots: list[dict[str, Any]] = [dict(item) for item in data.get("snapshots", [])]
        self.current: str | None = data.get("current")
        self.frames: dict[str, str | None] = dict(data.get("frames", {}))
        self.unknown_outputs: set[str] = set(data.get("unknown_outputs", []))
        self.complete = bool(data.get("complete", False))

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> LocalBatchResume:
        path = Path(path)
        if not os.path.lexists(path):
            return cls(path)
        try:
            if path.is_symlink() or not path.is_file() or path.stat().st_size > _MAX_STATE_BYTES:
                raise ValueError("not a regular, reasonably sized file")
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or data.get("version") != _VERSION:
                raise ValueError("unsupported format")
            state = cls(path, data)
            ids = {snapshot["id"] for snapshot in state.snapshots}
            if any(not isinstance(snapshot.get("settings"), dict) for snapshot in state.snapshots):
                raise ValueError("snapshot without settings")
            if state.current is not None and state.current not in ids:
                raise ValueError("unknown current snapshot")
            if any(value is not None and value not in ids for value in state.frames.values()):
                raise ValueError("frame refers to an unknown snapshot")
        except (OSError, UnicodeError, ValueError, KeyError, TypeError) as exc:
            raise LocalBatchResumeError(f"Local Contour recovery state is damaged ({path.name}): {exc}") from exc
        return state

    @property
    def current_settings(self) -> dict[str, Any] | None:
        snapshot = self._snapshot(self.current)
        return None if snapshot is None else dict(snapshot["settings"])

    def _snapshot(self, snapshot_id: str | None) -> dict[str, Any] | None:
        return next((item for item in self.snapshots if item["id"] == snapshot_id), None)

    def _snapshot_id_for(self, settings: Mapping[str, Any]) -> str | None:
        key = _canonical(settings)
        return next((item["id"] for item in self.snapshots if _canonical(item["settings"]) == key), None)

    def provenance(self, completed: set[str]) -> dict[str, str | None]:
        """Snapshot id (``None`` = unknown) that produced each completed frame."""
        result: dict[str, str | None] = {}
        for stem in completed:
            if stem in self.frames:
                result[stem] = self.frames[stem]
            elif stem in self.unknown_outputs:
                result[stem] = None
            else:
                result[stem] = self.current
        return result

    def has_unknown(self, completed: set[str]) -> bool:
        return any(value is None for value in self.provenance(completed).values())

    def needs_mixed_settings_confirmation(self, settings: Mapping[str, Any], completed: set[str]) -> bool:
        if not completed:
            return False
        target = self._snapshot_id_for(settings)
        return any(value is None or value != target for value in self.provenance(completed).values())

    def begin(self, settings: Mapping[str, Any], completed: set[str]) -> None:
        """Seal existing frames to their producer, then make *settings* active."""
        sealed = self.provenance(completed)
        self.frames = sealed
        self.unknown_outputs &= completed
        snapshot_id = self._snapshot_id_for(settings)
        if snapshot_id is None:
            snapshot_id = uuid4().hex
            self.snapshots.append({"id": snapshot_id, "settings": json.loads(_canonical(settings))})
        self.current = snapshot_id
        self.complete = False
        self.save()

    def record(self, stem: str, settings: Mapping[str, Any]) -> None:
        """Attribute one frame saved outside a batch to the settings it used."""
        snapshot_id = self._snapshot_id_for(settings)
        if snapshot_id is None:
            snapshot_id = uuid4().hex
            self.snapshots.append({"id": snapshot_id, "settings": json.loads(_canonical(settings))})
        self.frames[stem] = snapshot_id
        self.unknown_outputs.discard(stem)
        self.save()

    def groups(self, frame_ids: Mapping[str, str]) -> list[dict[str, Any]]:
        """Completed frame ids grouped by the settings that produced them."""
        provenance = self.provenance(set(frame_ids))
        ordered: list[dict[str, Any]] = []
        for snapshot_id in [item["id"] for item in self.snapshots] + [None]:
            members = sorted(frame_ids[stem] for stem, value in provenance.items() if value == snapshot_id)
            if members:
                snapshot = self._snapshot(snapshot_id)
                ordered.append(
                    {"settings": None if snapshot is None else dict(snapshot["settings"]), "frame_ids": members}
                )
        return ordered

    def mark_complete(self) -> None:
        self.complete = True
        self.save()

    def save(self) -> None:
        _atomic_write_json(
            self.path,
            {
                "version": _VERSION,
                "snapshots": self.snapshots,
                "current": self.current,
                "frames": self.frames,
                "unknown_outputs": sorted(self.unknown_outputs),
                "complete": self.complete,
            },
        )


def result_parameters(current: Mapping[str, Any], groups: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """``applied_parameters`` for a result; single-settings runs keep their shape."""
    if not groups:
        return dict(current)
    if len(groups) == 1 and groups[0]["settings"] is not None:
        settings = dict(groups[0]["settings"])
        settings.pop("save_options", None)
        return {**dict(current), **settings}
    return {
        **dict(current),
        "settings_history": [
            {
                "settings": None if group["settings"] is None else dict(group["settings"]),
                "provenance": "unknown" if group["settings"] is None else "recorded",
                "frame_ids": list(group["frame_ids"]),
            }
            for group in groups
        ],
    }


class ManagedBatchResume:
    """Qt-free view of a managed job's outputs and its local resume state."""

    def __init__(
        self,
        input_paths: Sequence[Path],
        frame_ids: Sequence[str],
        output_directory: Path,
        state: LocalBatchResume,
    ) -> None:
        self.input_paths = [Path(path) for path in input_paths]
        self.frame_ids = {path.stem: frame_id for path, frame_id in zip(self.input_paths, frame_ids, strict=True)}
        self.output_directory = Path(output_directory)
        self.state = state
        self.offset = 0
        self.running = False

    @classmethod
    def open(cls, session: Any) -> ManagedBatchResume:
        state = LocalBatchResume.load(Path(session.staging_root) / STATE_NAME)
        return cls(
            session.input_paths, [item.frame_id for item in session.manifest.inputs], session.output_directory, state
        )

    @property
    def total(self) -> int:
        return len(self.input_paths)

    def output_for(self, stem: str) -> Path:
        return self.output_directory / f"{stem}.cif"

    def completed(self) -> set[str]:
        return {path.stem for path in self.input_paths if is_complete_cif(self.output_for(path.stem))}

    def completed_cifs(self) -> list[Path]:
        done = self.completed()
        return [self.output_for(path.stem) for path in self.input_paths if path.stem in done]

    def remaining_paths(self) -> list[str]:
        done = self.completed()
        return [str(path) for path in self.input_paths if path.stem not in done]

    def needs_confirmation(self, settings: Mapping[str, Any]) -> bool:
        return self.state.needs_mixed_settings_confirmation(settings, self.completed())

    def has_unknown(self) -> bool:
        return self.state.has_unknown(self.completed())

    def begin(self, settings: Mapping[str, Any]) -> list[str]:
        """Persist *settings* before workers start; return the frames to process."""
        completed = self.completed()
        self.state.begin(settings, completed)
        self.offset = len(completed)
        self.running = True
        return [str(path) for path in self.input_paths if path.stem not in completed]

    def overall_progress(self, current: int) -> tuple[int, int]:
        return min(self.total, self.offset + max(0, int(current))), self.total

    def finish(self) -> None:
        self.running = False

    def record_manual(self, cif_path: str | os.PathLike[str], settings: Mapping[str, Any]) -> None:
        """Record a "process/save current frame" CIF written into the job outputs."""
        path = Path(cif_path)
        if path.stem not in self.frame_ids or os.path.normcase(os.path.abspath(path)) != os.path.normcase(
            os.path.abspath(self.output_for(path.stem))
        ):
            return
        self.state.record(path.stem, settings)

    def groups(self) -> list[dict[str, Any]]:
        return self.state.groups({stem: self.frame_ids[stem] for stem in self.completed()})


__all__ = [
    "STATE_NAME",
    "LocalBatchResume",
    "LocalBatchResumeError",
    "ManagedBatchResume",
    "is_complete_cif",
    "result_parameters",
]
