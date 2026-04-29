from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from csliser import __version__
from csliser.domain.models import FileOperation, ProcessingConfig, SelectionMode, SourceFolder
from csliser.domain.planner import build_operation_plan


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.plan:
        plan = build_operation_plan(_config_from_args(args))
        payload = {
            "files": len(plan.operations),
            "total_bytes": plan.total_bytes,
            "skipped_sources": [
                {"folder": str(folder), "extension": extension} for folder, extension in plan.skipped_sources
            ],
            "missing_frame_sets": len(plan.missing_frames),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2 if args.pretty else None))
        return

    from .bootstrap import build_application

    components = build_application([], apply_qss=not args.no_qss)
    components.window.show()
    raise SystemExit(components.app.exec())


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="csliser", description="Select frame files and copy, move, or delete them.")
    parser.add_argument("--source", action="append", default=[], help="Source folder. Can be passed multiple times.")
    parser.add_argument(
        "--extension",
        action="append",
        default=[],
        help="Extension to use for all CLI sources, for example .jpg. Can be passed multiple times.",
    )
    parser.add_argument("--frames", default="", help="Frame expression, for example 10-20;100:300,500.")
    parser.add_argument("--frames-per-row", type=int, default=135, help="Grid width for rectangle selection.")
    parser.add_argument(
        "--selection-mode",
        choices=[item.value for item in SelectionMode],
        default=SelectionMode.RECTANGLE.value,
        help="How frame ranges are expanded.",
    )
    parser.add_argument(
        "--operation",
        choices=[item.value for item in FileOperation],
        default=FileOperation.COPY.value,
        help="Operation used when planning.",
    )
    parser.add_argument("--destination", help="Destination folder for copy or move plans.")
    parser.add_argument("--no-extension-prefix", action="store_true", help="Do not prefix destination folders.")
    parser.add_argument("--plan", action="store_true", help="Build a plan in CLI mode and print JSON.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print CLI JSON output.")
    parser.add_argument("--no-qss", action="store_true", help="Do not apply the shared Kraken stylesheet.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def _config_from_args(args: argparse.Namespace) -> ProcessingConfig:
    extensions = tuple(args.extension or [".jpg", ".bmp", ".cif"])
    return ProcessingConfig(
        sources=tuple(SourceFolder(path=Path(item), extensions=extensions) for item in args.source),
        frame_expression=args.frames,
        selection_mode=SelectionMode(args.selection_mode),
        frames_per_row=args.frames_per_row,
        operation=FileOperation(args.operation),
        destination=Path(args.destination) if args.destination else None,
        add_extension_prefix=not args.no_extension_prefix,
    )
