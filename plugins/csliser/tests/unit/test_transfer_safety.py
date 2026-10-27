
import pytest

from csliser.application.use_cases import ExecuteTransferPlan, TransferExecutionOptions
from csliser.domain.models import FileOperation, ProcessingConfig, SelectionMode, SourceFolder
from csliser.domain.planner import PlanningError, build_operation_plan


def config(source, destination, operation=FileOperation.COPY):
    return ProcessingConfig(sources=(SourceFolder(source, (".jpg",)),), frame_expression="1",
        selection_mode=SelectionMode.FULL_RANGE, operation=operation, destination=destination)


@pytest.mark.parametrize("operation", [FileOperation.COPY, FileOperation.MOVE])
def test_existing_destination_is_never_overwritten_and_source_is_preserved(tmp_path, operation):
    source, destination = tmp_path / "source", tmp_path / "out"
    source.mkdir()
    image = source / "chip_000001.jpg"
    image.write_bytes(b"incoming image")
    target = destination / "jpg_source" / image.name
    target.parent.mkdir(parents=True)
    target.write_bytes(b"existing engineering result")
    plan = build_operation_plan(config(source, destination, operation))
    result = ExecuteTransferPlan().execute(plan)
    assert not result.ok and result.completed == 0 and result.errors
    assert target.read_bytes() == b"existing engineering result"
    assert image.read_bytes() == b"incoming image"


def test_two_names_with_the_same_selected_frame_are_explicitly_ambiguous(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for prefix in ("chip", "other"):
        (source / f"{prefix}_000001.jpg").write_bytes(prefix.encode())
    with pytest.raises(PlanningError, match="[Aa]mbiguous"):
        build_operation_plan(config(source, tmp_path / "out"))


def test_duplicate_destination_from_different_sources_is_rejected_before_execution(tmp_path):
    sources = [tmp_path / group / "source" for group in ("a", "b")]
    for source in sources:
        source.mkdir(parents=True)
        (source / "chip_000001.jpg").write_bytes(source.parent.name.encode())
    settings = ProcessingConfig(sources=tuple(SourceFolder(p, (".jpg",)) for p in sources), frame_expression="1",
        selection_mode=SelectionMode.FULL_RANGE, destination=tmp_path / "out")
    with pytest.raises(PlanningError, match="[Dd]estination"):
        build_operation_plan(settings)


@pytest.mark.parametrize("operation", [FileOperation.COPY, FileOperation.MOVE, FileOperation.DELETE])
def test_changed_source_after_preview_is_not_copied_moved_or_deleted(tmp_path, operation):
    source = tmp_path / "source"
    source.mkdir()
    image = source / "chip_000001.jpg"
    image.write_bytes(b"previewed image")
    plan = build_operation_plan(config(source, tmp_path / "out", operation))
    image.unlink()
    image.write_bytes(b"new replacement image, not in the reviewed plan")
    result = ExecuteTransferPlan().execute(plan)
    assert not result.ok and result.errors and result.completed == 0
    assert image.read_bytes() == b"new replacement image, not in the reviewed plan"
    assert not (tmp_path / "out/jpg_source/chip_000001.jpg").exists()


def test_numeric_corpus_filenames_are_indexed_without_renaming(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "0001.jpg").write_bytes(b"real corpus naming")
    plan = build_operation_plan(config(source, tmp_path / "out"))
    assert len(plan.operations) == 1 and plan.operations[0].source.name == "0001.jpg"


def test_mid_file_cancel_never_publishes_partial_destination(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    image = source / "chip_000001.jpg"
    image.write_bytes(b"a" * (4*1024*1024))
    plan = build_operation_plan(config(source, tmp_path / "out"))
    calls = []
    def cancel():
        calls.append(True)
        return len(calls) >= 5
    result = ExecuteTransferPlan().execute(plan, options=TransferExecutionOptions(max_workers=1), cancelled=cancel)
    assert result.cancelled and result.completed == 0
    assert not (tmp_path / "out/jpg_source/chip_000001.jpg").exists()
    assert not list((tmp_path / "out").rglob("*.part-*"))


def test_unselected_duplicate_frames_do_not_block_other_frames(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for name in ("chip_000001.jpg", "chip_000002.jpg", "other_000002.jpg"):
        (source / name).write_bytes(name.encode())
    plan = build_operation_plan(config(source, tmp_path / "out"))
    assert [p.frame for p in plan.operations] == [1]


@pytest.mark.parametrize("operation", [FileOperation.MOVE, FileOperation.DELETE])
def test_replacement_at_native_rename_is_restored_and_never_deleted(tmp_path, monkeypatch, operation):
    from csliser.application import file_transfer
    source = tmp_path / "source"
    source.mkdir()
    image = source / "chip_000001.jpg"
    image.write_bytes(b"reviewed original")
    plan = build_operation_plan(config(source, tmp_path / "out", operation))
    original_rename = file_transfer._rename_new
    saved = source / "original-preserved.jpg"
    replaced = []
    def rename(a, b):
        if a == image and not replaced:
            original_rename(image, saved)
            image.write_bytes(b"concurrent replacement image")
            replaced.append(True)
        original_rename(a, b)
    monkeypatch.setattr(file_transfer, "_rename_new", rename)
    result = ExecuteTransferPlan().execute(plan)
    assert result.errors and result.completed == 0
    assert image.read_bytes() == b"concurrent replacement image"
    assert saved.read_bytes() == b"reviewed original"
    assert not list(source.glob("*.csliser-remove-*"))


def test_parallel_publish_of_same_destination_has_one_winner(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    source = tmp_path / "source"
    source.mkdir()
    image = source / "chip_000001.jpg"
    image.write_bytes(b"verified payload" * 10000)
    plans = [build_operation_plan(config(source, tmp_path / "out")) for _ in range(2)]
    barrier = Barrier(2)
    def execute(plan):
        barrier.wait(timeout=5)
        return ExecuteTransferPlan().execute(plan)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(execute, plans))
    assert sum(r.completed for r in results) == 1
    assert sum(bool(r.errors) for r in results) == 1
    assert (tmp_path / "out/jpg_source/chip_000001.jpg").read_bytes() == image.read_bytes()
    assert not list((tmp_path / "out").rglob("*.part-*"))


def test_readonly_source_can_move_on_same_filesystem_without_changing_metadata(tmp_path):
    import os
    import stat
    source = tmp_path / "source"
    source.mkdir()
    image = source / "chip_000001.jpg"
    image.write_bytes(b"readonly source")
    os.chmod(image, stat.S_IREAD)
    before = image.stat().st_mtime_ns
    plan = build_operation_plan(config(source, tmp_path / "out", FileOperation.MOVE))
    target = plan.operations[0].destination
    try:
        result = ExecuteTransferPlan().execute(plan)
        assert result.ok and not image.exists()
        assert target.read_bytes() == b"readonly source" and target.stat().st_mtime_ns == before
    finally:
        for path in (image, target):
            if path.exists():
                os.chmod(path, stat.S_IREAD | stat.S_IWRITE)


def test_cancel_after_metadata_copy_cleans_only_owned_readonly_staging(tmp_path, monkeypatch):
    import os
    import stat
    from threading import Event
    from csliser.application import file_transfer
    source = tmp_path / "source"
    source.mkdir()
    image = source / "chip_000001.jpg"
    image.write_bytes(b"readonly source")
    os.chmod(image, stat.S_IREAD)
    plan = build_operation_plan(config(source, tmp_path / "out"))
    cancel = Event()
    copystat = file_transfer.shutil.copystat
    def metadata(*args, **kwargs):
        copystat(*args, **kwargs)
        cancel.set()
    monkeypatch.setattr(file_transfer.shutil, "copystat", metadata)
    try:
        result = ExecuteTransferPlan().execute(plan, options=TransferExecutionOptions(copy_metadata=True, max_workers=1), cancelled=cancel.is_set)
        assert result.cancelled and not result.errors and not plan.operations[0].destination.exists()
        assert image.read_bytes() == b"readonly source" and not list((tmp_path / "out").rglob("*.part-*"))
    finally:
        os.chmod(image, stat.S_IREAD | stat.S_IWRITE)
