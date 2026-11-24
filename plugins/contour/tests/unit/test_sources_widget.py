from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

from contour.application.sources import SOURCE_IMAGES, SOURCE_VECTORS


@pytest.fixture
def widget():
    from contour.widget import PolygonExtractionWidget

    instance = PolygonExtractionWidget()
    instance.set_ui_language("en")
    try:
        yield instance
    finally:
        instance.close()
        instance.deleteLater()


def _frames(directory: Path, stems: list[str]) -> list[str]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for stem in stems:
        path = directory / f"{stem}.png"
        cv2.imwrite(str(path), np.zeros((8, 8), dtype=np.uint8))
        paths.append(str(path))
    return paths


def _cifs(directory: Path, stems: list[str]) -> list[str]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for stem in stems:
        path = directory / f"{stem}.cif"
        path.write_text("DS 1 1 1;\nL NM;\nB 2 2 4 4;\nDF;\nE\n", encoding="utf-8")
        paths.append(str(path))
    return paths


def _wait_for_scans(widget, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        QApplication.processEvents()
        if not widget.scanning_source_ids() and not widget._image_list_rebuild_in_progress:
            QApplication.processEvents()
            return
        time.sleep(0.01)
    raise AssertionError("source scans did not finish")


def test_folder_and_file_sources_combine_and_pair_with_vectors(widget, tmp_path: Path) -> None:
    _frames(tmp_path / "lot", ["f1", "f2", "f3"])
    extra = _frames(tmp_path / "extra", ["g4"])
    _cifs(tmp_path / "cif", ["f1", "f2", "orphan"])

    widget.add_image_folder(str(tmp_path / "lot"))
    _wait_for_scans(widget)
    widget.add_image_files(extra)
    widget.add_vector_folder(str(tmp_path / "cif"))
    _wait_for_scans(widget)

    assert [Path(path).stem for path in widget._workspace.image_paths] == ["f1", "f2", "f3", "g4"]
    assert set(widget._workspace.cif_paths_by_stem) == {"f1", "f2", "orphan"}
    report = widget.source_registry.pairing()
    (pair,) = report.pairs
    assert (pair.matched, pair.total) == (2, 3)
    tree = widget.sources_panel.tree
    assert tree.topLevelItemCount() == 2
    assert tree.topLevelItem(0).child(0).text(1) == "⚠ 2/3"


def test_removing_a_source_unloads_only_its_frames_and_vectors(widget, tmp_path: Path) -> None:
    _frames(tmp_path / "lot", ["f1", "f2"])
    extra = _frames(tmp_path / "extra", ["g3"])
    fix = _cifs(tmp_path / "fix", ["f1"])
    widget.add_image_folder(str(tmp_path / "lot"))
    _wait_for_scans(widget)
    widget.add_image_files(extra)
    widget.add_vector_files(fix)
    _wait_for_scans(widget)

    files_source = widget.source_registry.sources(SOURCE_IMAGES)[1]
    vector_source = widget.source_registry.sources(SOURCE_VECTORS)[0]
    assert widget.remove_source(files_source.id)
    assert widget.remove_source(vector_source.id)
    _wait_for_scans(widget)

    assert [Path(path).stem for path in widget._workspace.image_paths] == ["f1", "f2"]
    assert widget._workspace.cif_paths_by_stem == {}


def test_vector_overlap_asks_and_keeps_current_on_request(widget, tmp_path: Path, monkeypatch) -> None:
    _frames(tmp_path / "lot", ["f1"])
    first = _cifs(tmp_path / "a", ["f1"])
    second = _cifs(tmp_path / "b", ["f1"])
    widget.add_image_folder(str(tmp_path / "lot"))
    _wait_for_scans(widget)
    widget.add_vector_files(first)
    asked: list[tuple[str, int]] = []
    monkeypatch.setattr(widget, "_ask_source_overlap", lambda kind, count: asked.append((kind, count)) or "keep")

    widget.add_vector_files(second)
    _wait_for_scans(widget)

    assert asked == [(SOURCE_VECTORS, 1)]
    assert widget._workspace.cif_paths_by_stem["f1"] == first[0]

    monkeypatch.setattr(widget, "_ask_source_overlap", lambda kind, count: "replace")
    widget.add_vector_files(_cifs(tmp_path / "c", ["f1"]))
    assert widget._workspace.cif_paths_by_stem["f1"] == str(tmp_path / "c" / "f1.cif")


def test_duplicate_frames_can_be_skipped(widget, tmp_path: Path, monkeypatch) -> None:
    _frames(tmp_path / "lot", ["f1", "f2"])
    copies = _frames(tmp_path / "copy", ["f1", "n3"])
    widget.add_image_folder(str(tmp_path / "lot"))
    _wait_for_scans(widget)
    monkeypatch.setattr(widget, "_ask_source_overlap", lambda kind, count: "skip")

    widget.add_image_files(copies)

    assert [Path(path).stem for path in widget._workspace.image_paths] == ["f1", "f2", "n3"]


def test_removing_a_frame_excludes_it_from_its_folder_source(widget, tmp_path: Path) -> None:
    paths = _frames(tmp_path / "lot", ["f1", "f2"])
    widget.add_image_folder(str(tmp_path / "lot"))
    _wait_for_scans(widget)

    widget.remove_images([paths[0]])

    (source,) = widget.source_registry.sources(SOURCE_IMAGES)
    assert source.excluded == frozenset({paths[0]})
    assert [Path(path).stem for path in widget._workspace.image_paths] == ["f2"]


def test_sources_are_saved_and_restored(widget, tmp_path: Path) -> None:
    from contour.widget import PolygonExtractionWidget

    _frames(tmp_path / "lot", ["f1", "f2"])
    _cifs(tmp_path / "cif", ["f2"])
    widget.add_image_folder(str(tmp_path / "lot"))
    widget.add_vector_folder(str(tmp_path / "cif"))
    _wait_for_scans(widget)

    restored = PolygonExtractionWidget()
    try:
        restored._restore_persisted_session_selection()
        _wait_for_scans(restored)
        assert [Path(path).stem for path in restored._workspace.image_paths] == ["f1", "f2"]
        assert set(restored._workspace.cif_paths_by_stem) == {"f2"}
    finally:
        restored.close()
        restored.deleteLater()


def test_legacy_directory_settings_are_migrated(widget, tmp_path: Path) -> None:
    from contour.application.dto import PersistedPaths
    from contour.infrastructure.settings_store import IMAGE_LIST_MODE_DIRECTORY

    _frames(tmp_path / "lot", ["f1"])
    _cifs(tmp_path / "cif", ["f1"])
    widget._path_settings.save(
        PersistedPaths(input_directory=str(tmp_path / "lot"), cif_directory=str(tmp_path / "cif"))
    )
    widget._session_settings_store.save_image_list_mode(IMAGE_LIST_MODE_DIRECTORY)

    widget._restore_persisted_session_selection()
    _wait_for_scans(widget)

    assert [source.origin for source in widget.source_registry.sources()] == ["folder", "folder"]
    assert set(widget._workspace.cif_paths_by_stem) == {"f1"}


def test_kraken_lock_locks_sources_and_keeps_suggested_destination(widget, tmp_path: Path) -> None:
    _frames(tmp_path / "lot", ["f1"])
    widget.add_image_folder(str(tmp_path / "lot"))
    _wait_for_scans(widget)
    widget.output_dir_edit.setText(str(tmp_path))

    widget.lock_paths_for_kraken(lock_destinations=False)

    assert all(source.locked for source in widget.source_registry.sources())
    assert not widget.remove_source(widget.source_registry.sources()[0].id)
    assert not widget.output_dir_edit.is_locked()
    assert widget.output_dir_edit.badge_label.text() == "Kraken"
