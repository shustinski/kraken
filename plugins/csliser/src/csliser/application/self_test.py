"""Bounded deployment proof: actual configured window, Qt worker and safe copy."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from unittest.mock import patch
from uuid import uuid4


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run_self_test(args) -> int:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtCore import QSettings
    from PyQt6.QtWidgets import QApplication, QMessageBox
    from PyQt6.QtGui import QFont, QFontDatabase, QFontMetrics
    from csliser.application.bootstrap import build_application
    from csliser.application.launch_context import context_from_args
    from csliser.domain.models import FileOperation
    from csliser.domain.planner import build_operation_plan
    from csliser.infrastructure.settings_store import WindowSettingsStore

    run = Path(args.self_test_dir).expanduser().resolve() / ("csliser-self-test-" + uuid4().hex)
    run.mkdir(parents=True)
    report = {"status": "running", "training_started": False, "directory": str(run),
              "scope": "Qt/runtime copy and own diagnostic source-removal crash recovery; no external desktop input, user move/delete or power-cut certification",
              "notification_dialogs": "noninteractive collector; file backend and Qt worker are real"}
    (run / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    window = None
    components = None
    try:
        values = vars(args).copy()
        if not args.source:
            source = run / "source"
            source.mkdir()
            for index in range(2):
                (source / f"frame_{index}.txt").write_bytes(f"diagnostic transfer {index}".encode())
            values.update(source=[str(source)], extension=[".txt"], frames="0-1", selection_mode="full_range")
        elif args.frames is None:
            raise ValueError("An explicit frame expression is required for reference sources")
        values.update(destination=str(run / "copies"), operation="copy")
        from argparse import Namespace

        context = context_from_args(Namespace(**values))
        store = WindowSettingsStore(lambda: QSettings(str(run / "settings.ini"), QSettings.Format.IniFormat))
        app = QApplication.instance() or QApplication([])
        if not QFontDatabase.families() and os.name == "nt":
            font = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts/segoeui.ttf"
            if font.is_file() and QFontDatabase.addApplicationFont(str(font)) >= 0:
                app.setFont(QFont("Segoe UI", 9))
        components = build_application([], apply_qss=not args.no_qss, settings_store=store)
        window = components.window
        metrics = QFontMetrics(window.font)
        if not all(metrics.inFontUcs4(ord(character)) for character in "CSliser0123456789Копировать"):
            raise RuntimeError("The deployment probe cannot render its Latin/Cyrillic UI text")
        report["ui_glyphs_available"] = True
        window.apply_launch_context(context)
        window.show()
        components.app.processEvents()
        actual = window._config(FileOperation.COPY)
        plan = build_operation_plan(actual)
        if not 1 <= len(plan.operations) <= 64 or plan.total_bytes > 512 * 1024 * 1024:
            raise ValueError("Deployment probe requires 1..64 files and at most 512 MiB")
        report["configured"] = {"sources": [str(s.path) for s in actual.sources],
            "extensions": [list(s.extensions) for s in actual.sources], "frames": actual.frame_expression,
            "frames_per_row": actual.frames_per_row, "selection_mode": actual.selection_mode.value,
            "destination": str(actual.destination)}
        before = {str(op.source): _digest(op.source) for op in plan.operations}
        notices = []

        def execute():
            window._current_plan = plan
            window._start_worker(plan)
            deadline = time.monotonic() + 60
            while window._worker_thread is not None and time.monotonic() < deadline:
                components.app.processEvents()
                time.sleep(0.005)
            if window._worker_thread is not None:
                raise TimeoutError("Qt transfer did not reach a stopped-thread state")
            return window._last_result

        with patch.object(QMessageBox, "information", side_effect=lambda *_args: notices.append("complete")), \
             patch.object(QMessageBox, "warning", side_effect=lambda *_args: notices.append("warning")):
            first = execute()
            if not first.ok or first.completed != len(plan.operations):
                raise RuntimeError("Real Qt copy failed")
            for op in plan.operations:
                if _digest(op.destination) != before[str(op.source)]:
                    raise RuntimeError("Copied bytes differ from the source")
            repeated = execute()
            if repeated.ok or repeated.completed != 0:
                raise RuntimeError("Existing destinations were not refused")
        for op in plan.operations:
            if _digest(op.source) != before[str(op.source)] or _digest(op.destination) != before[str(op.source)]:
                raise RuntimeError("Source or destination changed after overwrite refusal")
        components.app.processEvents()
        if not window.grab().save(str(run / "window.png")):
            raise RuntimeError("Could not save CSliser rendering")
        report.update(status="passed", files=len(plan.operations), bytes=plan.total_bytes,
            completed=first.completed, overwrite_refused=True, source_unchanged=True,
            worker_stopped=window._worker_thread is None, notifications=notices)
        command = ([sys.executable] if getattr(sys, "frozen", False) else
                   [sys.executable, "-m", "csliser"])
        crash_directory = run / "removal-crash"
        with (run / "removal-crash.log").open("wb") as log:
            child = subprocess.run([*command, "--self-test-removal-crash-dir", str(crash_directory)],
                stdout=log, stderr=subprocess.STDOUT, timeout=20,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        if child.returncode != 73:
            raise RuntimeError(f"Source-removal crash probe exited at the wrong boundary: {child.returncode}")
        controls = list(crash_directory.glob("csliser-removal-crash-*/control.json"))
        if len(controls) != 1:
            raise RuntimeError("Crash probe did not leave a unique control receipt")
        control = json.loads(controls[0].read_text(encoding="utf-8"))
        for role in ("source", "receipt", "quarantine"):
            if not Path(control[role]).resolve().is_relative_to(crash_directory.resolve()):
                raise RuntimeError("Crash control escaped its diagnostic directory")
        if Path(control["source"]).exists() or _digest(Path(control["quarantine"])) != control["sha256"]:
            raise RuntimeError("Crash did not preserve the expected quarantine bytes")
        from .removal_recovery import RecoveryPlan, RecoverRemovals

        recovery = RecoverRemovals()
        recovered = recovery.execute(RecoveryPlan((Path(control["receipt"]),)))
        if not recovered.ok or recovered.completed != 1 or recovery.outcomes[0]["status"] != "restored":
            raise RuntimeError("Actual interrupted source removal did not recover")
        if _digest(Path(control["source"])) != control["sha256"] or Path(control["receipt"]).exists():
            raise RuntimeError("Recovered source bytes or receipt cleanup differ")
        report["source_removal_crash_recovery"] = dict(status="passed", crash_exit=child.returncode,
            sha256=control["sha256"], restored=True, scope="Own diagnostic source, real process exit; no power-cut certification")
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
    finally:
        if window is not None:
            if window._worker_thread is not None:
                window._cancel_worker()
                deadline = time.monotonic() + 20
                while window._worker_thread is not None and time.monotonic() < deadline:
                    components.app.processEvents()
                    time.sleep(0.01)
            if window._worker_thread is not None:
                report.update(status="failed", cleanup_error="Worker remains alive")
            else:
                window.close()
                components.app.processEvents()
        (run / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0 if report["status"] == "passed" else 1
