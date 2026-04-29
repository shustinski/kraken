from __future__ import annotations

from pathlib import Path

from csliser.domain.models import FileOperation, ProcessingConfig, SelectionMode, SourceFolder
from csliser.domain.planner import build_operation_plan, discover_extensions


def _write(path: Path, text: str = "data") -> None:
    path.write_text(text, encoding="utf-8")


def test_discover_extensions_returns_sorted_existing_extensions(tmp_path: Path) -> None:
    _write(tmp_path / "frame_001.jpg")
    _write(tmp_path / "frame_001.cif")
    _write(tmp_path / "README")

    assert discover_extensions(tmp_path) == (".cif", ".jpg")


def test_build_operation_plan_indexes_trailing_numeric_frame_suffix(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "out"
    source.mkdir()
    _write(source / "chip_000001.jpg", "a")
    _write(source / "chip_000002.jpg", "bb")
    _write(source / "chip_notes.jpg", "ignored")

    plan = build_operation_plan(
        ProcessingConfig(
            sources=(SourceFolder(source, (".jpg",)),),
            frame_expression="1-3",
            selection_mode=SelectionMode.FULL_RANGE,
            operation=FileOperation.COPY,
            destination=destination,
            add_extension_prefix=True,
        )
    )

    assert [item.source.name for item in plan.operations] == ["chip_000001.jpg", "chip_000002.jpg"]
    assert [item.destination for item in plan.operations] == [
        destination / "jpg_source" / "chip_000001.jpg",
        destination / "jpg_source" / "chip_000002.jpg",
    ]
    assert plan.missing_frames[0].frames == (3,)
    assert plan.total_bytes == 3


def test_build_delete_plan_does_not_require_destination(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _write(source / "chip_000001.cif")

    plan = build_operation_plan(
        ProcessingConfig(
            sources=(SourceFolder(source, (".cif",)),),
            frame_expression="1",
            selection_mode=SelectionMode.FULL_RANGE,
            operation=FileOperation.DELETE,
            destination=None,
        )
    )

    assert len(plan.operations) == 1
    assert plan.operations[0].destination is None
