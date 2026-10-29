"""Validated CLI-to-window context; configuring a window never executes a plan."""
from dataclasses import dataclass
from pathlib import Path

from kraken_core.safe_files import ensure_regular_directory
from csliser.domain.models import FileOperation, SelectionMode, SourceFolder
from csliser.domain.planner import normalize_extension
from csliser.domain.ranges import parse_frame_ranges


@dataclass(frozen=True, slots=True)
class LaunchContext:
    sources: tuple[SourceFolder, ...] = ()
    frame_expression: str | None = None
    frames_per_row: int | None = None
    selection_mode: SelectionMode | None = None
    operation: FileOperation | None = None
    destination: str | None = None
    add_extension_prefix: bool | None = None


def context_from_args(args) -> LaunchContext:
    sources = []
    extensions = tuple(dict.fromkeys(normalize_extension(value) for value in args.extension))
    for value in args.source:
        if not value.strip():
            raise ValueError("Source folder is empty.")
        sources.append(SourceFolder(ensure_regular_directory(Path(value).expanduser().absolute()), extensions))
    if args.frames is not None:
        parse_frame_ranges(args.frames)
    if args.frames_per_row is not None and args.frames_per_row <= 0:
        raise ValueError("GUI frames per row must be greater than zero.")
    return LaunchContext(tuple(sources), args.frames, args.frames_per_row,
        None if args.selection_mode is None else SelectionMode(args.selection_mode),
        None if args.operation is None else FileOperation(args.operation),
        args.destination, False if args.no_extension_prefix else None)
