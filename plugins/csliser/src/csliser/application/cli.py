from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from csliser import __version__
from csliser.domain.models import FileOperation, ProcessingConfig, SelectionMode, SourceFolder
from csliser.domain.planner import build_operation_plan
from csliser.domain.planner import PlanningError
from csliser.domain.ranges import FrameRangeError


def main(argv: Sequence[str] | None = None) -> None:
    from kraken_core.process_lifetime import bind_child_process_lifetime

    bind_child_process_lifetime()
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.self_test_removal_crash_dir:
        if args.source or args.destination or args.frames or args.operation or args.settings_file:
            parser.error("Crash probe generates only its own diagnostic file; source/operation options are forbidden.")
        from .removal_crash_probe import run_removal_crash_probe

        run_removal_crash_probe(args.self_test_removal_crash_dir)
        return

    if args.recover_source:
        if args.source or args.destination or args.frames or args.operation or args.settings_file:
            parser.error("Recovery uses only explicit --recover-source folders; it does not execute a new transfer plan.")
        from .removal_recovery import RecoveryPlan, RecoverRemovals, discover_removal_receipts

        try:
            paths = tuple(dict.fromkeys(path for folder in args.recover_source
                         for path in discover_removal_receipts(Path(folder)).operations))
            if len(paths) > 10000:
                raise ValueError("At most 10000 removal receipts can be recovered in one operation")
        except (OSError, ValueError) as error:
            parser.error(str(error))
        executor = RecoverRemovals()
        result = executor.execute(RecoveryPlan(paths))
        print(json.dumps({"ok": result.ok, "requested": result.requested,
            "completed": result.completed, "outcomes": executor.outcomes,
            "errors": [{"receipt": str(e.source), "message": e.message} for e in result.errors]},
            ensure_ascii=True, indent=2 if args.pretty else None))
        raise SystemExit(0 if result.ok else 1)

    if args.get_capabilities:
        from csliser.actions import create_action_registry
        from kraken_core.ipc import ActionRequest

        print(create_action_registry().dispatch(ActionRequest("get_capabilities")).to_json())
        return
    if args.self_test_dir:
        if args.operation not in (None, FileOperation.COPY.value) or args.destination or args.settings_file:
            parser.error("Self-test copies only into its own fresh directory and owns its settings file.")
        from .self_test import run_self_test

        raise SystemExit(run_self_test(args))

    if args.plan:
        try:
            plan = build_operation_plan(_config_from_args(args))
        except (FrameRangeError, PlanningError, OSError, ValueError) as exc:
            parser.error(str(exc))
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
    from .launch_context import context_from_args

    try:
        context = context_from_args(args)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    settings_store = None
    if args.settings_file:
        from PyQt6.QtCore import QSettings
        from csliser.infrastructure.settings_store import WindowSettingsStore

        path = Path(args.settings_file).expanduser().absolute()
        settings_store = WindowSettingsStore(lambda: QSettings(str(path), QSettings.Format.IniFormat))
    components = build_application([], apply_qss=not args.no_qss, settings_store=settings_store)
    components.window.apply_launch_context(context)
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
    parser.add_argument("--frames", help="Frame expression, for example 10-20;100:300,500.")
    parser.add_argument("--frames-per-row", type=int, help="Grid width for rectangle selection (plan default: 135).")
    parser.add_argument(
        "--selection-mode",
        choices=[item.value for item in SelectionMode],
        default=None,
        help="How frame ranges are expanded.",
    )
    parser.add_argument(
        "--operation",
        choices=[item.value for item in FileOperation],
        default=None,
        help="Operation used when planning.",
    )
    parser.add_argument("--destination", help="Destination folder for copy or move plans.")
    parser.add_argument("--no-extension-prefix", action="store_true", help="Do not prefix destination folders.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true", help="Build a plan in CLI mode and print JSON.")
    mode.add_argument("--get-capabilities", action="store_true", help="Print supported legacy discovery actions.")
    mode.add_argument("--self-test-dir", help="Run a bounded real Qt copy probe in a fresh child directory.")
    mode.add_argument("--recover-source", action="append", help="Restore interrupted source quarantine in this folder. Repeat for multiple folders.")
    mode.add_argument("--self-test-removal-crash-dir", help="Create an isolated diagnostic file and exit at the source-quarantine boundary.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print CLI JSON output.")
    parser.add_argument("--no-qss", action="store_true", help="Do not apply the shared Kraken stylesheet.")
    parser.add_argument("--settings-file", help="Explicit INI for an isolated GUI instance.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def _config_from_args(args: argparse.Namespace) -> ProcessingConfig:
    extensions = tuple(args.extension or [".jpg", ".bmp", ".cif"])
    return ProcessingConfig(
        sources=tuple(SourceFolder(path=Path(item), extensions=extensions) for item in args.source),
        frame_expression=args.frames or "",
        selection_mode=SelectionMode(args.selection_mode or SelectionMode.RECTANGLE.value),
        frames_per_row=135 if args.frames_per_row is None else args.frames_per_row,
        operation=FileOperation(args.operation or FileOperation.COPY.value),
        destination=Path(args.destination) if args.destination else None,
        add_extension_prefix=not args.no_extension_prefix,
    )
