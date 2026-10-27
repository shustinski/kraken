from __future__ import annotations

import re
import os
import stat
from collections.abc import Iterable
from pathlib import Path

from kraken_core.safe_files import ensure_regular_directory

from .models import FileOperation, MissingFrameSet, OperationPlan, PlannedOperation, ProcessingConfig, SourceFileSnapshot
from .ranges import expand_frames, parse_frame_ranges

_FRAME_SUFFIX = re.compile(r"^(?:.+_)?(?P<frame>\d+)$")


class PlanningError(ValueError):
    """Raised when a processing plan cannot be built from the given config."""


def normalize_extension(extension: str) -> str:
    value = extension.strip().lower()
    if not value:
        raise PlanningError("Extension cannot be empty.")
    return value if value.startswith(".") else f".{value}"


def discover_extensions(folder: Path, *, defaults: Iterable[str] = (".jpg", ".bmp", ".cif")) -> tuple[str, ...]:
    if not folder.is_dir():
        return tuple(normalize_extension(item) for item in defaults)
    extensions = {path.suffix.lower() for path in folder.iterdir() if path.is_file() and path.suffix}
    return tuple(sorted(extensions or {normalize_extension(item) for item in defaults}))


def build_operation_plan(config: ProcessingConfig) -> OperationPlan:
    validate_processing_config(config)
    frame_ranges = parse_frame_ranges(config.frame_expression)
    frames = expand_frames(frame_ranges, mode=config.selection_mode, frames_per_row=config.frames_per_row)
    plan = OperationPlan(config=config)
    requested_frames = set(frames)
    destinations = set()

    for source in config.sources:
        try:
            source_root = ensure_regular_directory(source.path.absolute())
        except (OSError, ValueError) as exc:
            raise PlanningError(str(exc)) from exc
        for extension in source.extensions:
            normalized_extension = normalize_extension(extension)
            index = _index_files_by_frame(source_root, normalized_extension, requested_frames)
            available_frames = requested_frames.intersection(index)
            if not available_frames:
                plan.skipped_sources.append((source.path, normalized_extension))
                continue

            missing = tuple(frame for frame in frames if frame not in index)
            if missing:
                plan.missing_frames.append(
                    MissingFrameSet(source_folder=source.path, extension=normalized_extension, frames=missing)
                )

            for frame in frames:
                source_file = index.get(frame)
                if source_file is None:
                    continue
                destination = _destination_for(config, source.path, source_file, normalized_extension)
                if destination is not None:
                    destination = destination.absolute()
                    key = os.path.normcase(str(destination))
                    if key in destinations:
                        raise PlanningError(f"Destination collision in this plan: {destination}")
                    destinations.add(key)
                snapshot = SourceFileSnapshot.from_stat(source_file.stat(follow_symlinks=False))
                plan.operations.append(
                    PlannedOperation(
                        source=source_file,
                        destination=destination,
                        frame=frame,
                        extension=normalized_extension,
                        source_folder=source_root,
                        source_snapshot=snapshot,
                    )
                )
                plan.total_bytes += snapshot.size_bytes

    return plan


def validate_processing_config(config: ProcessingConfig) -> None:
    if not config.sources:
        raise PlanningError("At least one source folder must be selected.")
    for source in config.sources:
        if not source.path.is_dir():
            raise PlanningError(f"Source folder does not exist: {source.path}")
        if not source.extensions:
            raise PlanningError(f"No file extensions selected for source folder: {source.path}")
    if config.operation != FileOperation.DELETE:
        if config.destination is None:
            raise PlanningError("Destination folder is required for copy and move operations.")
        if config.destination.exists() and not config.destination.is_dir():
            raise PlanningError(f"Destination is not a folder: {config.destination}")


def _index_files_by_frame(folder: Path, extension: str, selected: set[int]) -> dict[int, Path]:
    index: dict[int, Path] = {}
    for path in sorted(folder.iterdir(), key=lambda item: item.name.lower()):
        if path.suffix.lower() != extension:
            continue
        match = _FRAME_SUFFIX.match(path.stem)
        if match is None:
            continue
        frame = int(match.group("frame"))
        if frame not in selected:
            continue
        value = path.stat(follow_symlinks=False)
        if stat.S_ISLNK(value.st_mode) or getattr(value, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400):
            raise PlanningError(f"Selected source is a link or reparse point: {path}")
        if not stat.S_ISREG(value.st_mode):
            continue
        if frame in index:
            raise PlanningError(f"Ambiguous frame {frame}: {index[frame].name}, {path.name}")
        index[frame] = path
    return index


def _destination_for(config: ProcessingConfig, source_folder: Path, source_file: Path, extension: str) -> Path | None:
    if config.destination is None:
        return None
    prefix = f"{extension.removeprefix('.')}_" if config.add_extension_prefix else ""
    return config.destination / f"{prefix}{source_folder.name}" / source_file.name
