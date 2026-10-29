"""Bounded deployment fault injection; creates only its own diagnostic file."""
import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4


def run_removal_crash_probe(output_directory):
    from csliser.application import file_transfer
    from csliser.application.removal_recovery import discover_removal_receipts
    from csliser.application.use_cases import ExecuteTransferPlan
    from csliser.domain.models import FileOperation, ProcessingConfig, SelectionMode, SourceFolder
    from csliser.domain.planner import build_operation_plan
    from csliser.infrastructure.removal_receipt import sync_directory

    run = Path(output_directory).expanduser().resolve() / ("csliser-removal-crash-" + uuid4().hex)
    run.mkdir(parents=True)
    source = run / "frame_1.txt"
    payload = b"CSliser own deployment crash-recovery diagnostic\n" * 1000
    with source.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    plan = build_operation_plan(ProcessingConfig((SourceFolder(run, (".txt",)),), "1",
        selection_mode=SelectionMode.FULL_RANGE, operation=FileOperation.DELETE))
    rename = file_transfer._rename_new

    def fault_after_quarantine(a, b):
        rename(a, b)
        if a == source:
            receipt, = discover_removal_receipts(run).operations
            control = dict(source=str(source), receipt=str(receipt), quarantine=str(b),
                           sha256=hashlib.sha256(payload).hexdigest(), fault="process exit after source quarantine")
            with (run / "control.json").open("x", encoding="utf-8") as stream:
                json.dump(control, stream)
                stream.flush()
                os.fsync(stream.fileno())
            sync_directory(run)
            # Deliberately bypass finally/atexit, leaving the real immutable
            # receipt and quarantine behind. OS process locks must be released.
            os._exit(73)

    file_transfer._rename_new = fault_after_quarantine
    result = ExecuteTransferPlan().execute(plan)
    raise RuntimeError(f"Crash injection did not exit at the required boundary: {result}")
