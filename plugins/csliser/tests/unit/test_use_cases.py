from __future__ import annotations

import os
from pathlib import Path

from csliser.application.use_cases import ExecuteTransferPlan, TransferExecutionOptions, resolve_transfer_workers
from csliser.domain.models import FileOperation, ProcessingConfig, SelectionMode, SourceFolder
from csliser.domain.planner import build_operation_plan


def _build_plan(
    source: Path,
    destination: Path | None,
    *,
    extension: str,
    frames: str,
    operation: FileOperation,
):
    return build_operation_plan(
        ProcessingConfig(
            sources=(SourceFolder(source, (extension,)),),
            frame_expression=frames,
            selection_mode=SelectionMode.FULL_RANGE,
            operation=operation,
            destination=destination,
        )
    )


def test_execute_copy_plan_copies_files_in_fast_mode(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "out"
    source.mkdir()
    source_file = source / "chip_000001.jpg"
    source_file.write_text("payload", encoding="utf-8")
    os.utime(source_file, (946684800, 946684800))
    plan = _build_plan(source, destination, extension=".jpg", frames="1", operation=FileOperation.COPY)

    result = ExecuteTransferPlan().execute(plan)
    copied = destination / "jpg_source" / "chip_000001.jpg"

    assert result.ok
    assert copied.read_text(encoding="utf-8") == "payload"
    assert int(copied.stat().st_mtime) != 946684800


def test_execute_copy_plan_can_preserve_metadata_when_requested(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "out"
    source.mkdir()
    source_file = source / "chip_000001.jpg"
    source_file.write_text("payload", encoding="utf-8")
    os.utime(source_file, (946684800, 946684800))
    plan = _build_plan(source, destination, extension=".jpg", frames="1", operation=FileOperation.COPY)

    result = ExecuteTransferPlan().execute(plan, options=TransferExecutionOptions(copy_metadata=True))

    assert result.ok
    assert int((destination / "jpg_source" / "chip_000001.jpg").stat().st_mtime) == 946684800


def test_execute_copy_plan_creates_destination_directories_for_multiple_files(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "out"
    source.mkdir()
    for frame in range(1, 5):
        (source / f"chip_{frame:06d}.jpg").write_text(str(frame), encoding="utf-8")
    plan = _build_plan(source, destination, extension=".jpg", frames="1-4", operation=FileOperation.COPY)

    result = ExecuteTransferPlan().execute(plan, options=TransferExecutionOptions(max_workers=2, progress_batch_size=1))

    assert result.ok
    assert sorted(path.name for path in (destination / "jpg_source").iterdir()) == [
        "chip_000001.jpg",
        "chip_000002.jpg",
        "chip_000003.jpg",
        "chip_000004.jpg",
    ]


def test_execute_plan_keeps_running_after_file_error(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "out"
    source.mkdir()
    missing = source / "chip_000001.jpg"
    survivor = source / "chip_000002.jpg"
    missing.write_text("missing", encoding="utf-8")
    survivor.write_text("survivor", encoding="utf-8")
    plan = _build_plan(source, destination, extension=".jpg", frames="1-2", operation=FileOperation.COPY)
    missing.unlink()

    result = ExecuteTransferPlan().execute(plan, options=TransferExecutionOptions(max_workers=2, progress_batch_size=1))

    assert not result.ok
    assert result.completed == 1
    assert len(result.errors) == 1
    assert (destination / "jpg_source" / "chip_000002.jpg").read_text(encoding="utf-8") == "survivor"


def test_execute_plan_cancelled_before_start_does_not_copy(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "out"
    source.mkdir()
    (source / "chip_000001.jpg").write_text("payload", encoding="utf-8")
    plan = _build_plan(source, destination, extension=".jpg", frames="1", operation=FileOperation.COPY)

    result = ExecuteTransferPlan().execute(plan, cancelled=lambda: True)

    assert result.cancelled
    assert result.completed == 0
    assert not destination.exists()


def test_execute_delete_plan_removes_files(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    file_path = source / "chip_000001.cif"
    file_path.write_text("payload", encoding="utf-8")
    plan = _build_plan(source, None, extension=".cif", frames="1", operation=FileOperation.DELETE)

    result = ExecuteTransferPlan().execute(plan)

    assert result.ok
    assert not file_path.exists()


def test_resolve_transfer_workers_uses_auto_cap() -> None:
    assert resolve_transfer_workers(0) == 0
    assert resolve_transfer_workers(1) == 1
    assert resolve_transfer_workers(100) <= 8
    assert resolve_transfer_workers(100, max_workers=3) == 3
    assert resolve_transfer_workers(2, max_workers=100) == 2
