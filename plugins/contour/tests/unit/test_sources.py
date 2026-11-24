from __future__ import annotations

from pathlib import Path

from contour.application.sources import (
    ORIGIN_FOLDER,
    SOURCE_IMAGES,
    SOURCE_VECTORS,
    FrameSource,
    SourceRegistry,
    files_source,
    folder_source,
)


def _scanned_folder(registry: SourceRegistry, kind: str, root: Path, names: list[str]) -> FrameSource:
    source = registry.add(folder_source(kind, root))
    updated = registry.set_scanned_paths(source.id, [root / name for name in names])
    assert updated is not None
    return updated


def test_image_paths_keep_source_order_and_skip_duplicates(tmp_path: Path) -> None:
    registry = SourceRegistry()
    _scanned_folder(registry, SOURCE_IMAGES, tmp_path / "a", ["f1.png", "f2.png"])
    registry.add(files_source(SOURCE_IMAGES, [tmp_path / "b" / "g1.png", tmp_path / "a" / "f2.png"]))

    assert registry.image_paths() == [
        str(tmp_path / "a" / "f1.png"),
        str(tmp_path / "a" / "f2.png"),
        str(tmp_path / "b" / "g1.png"),
    ]


def test_later_vector_source_wins_unless_added_with_low_priority(tmp_path: Path) -> None:
    registry = SourceRegistry()
    folder = _scanned_folder(registry, SOURCE_VECTORS, tmp_path / "cif", ["f1.cif", "f2.cif"])
    fix = files_source(SOURCE_VECTORS, [tmp_path / "fix" / "F2.cif"])

    assert registry.vector_conflicts(fix) == {"f2"}
    registry.add(fix)
    assert registry.cif_index()["f2"] == str(tmp_path / "fix" / "F2.cif")
    assert registry.vector_source_for_stem("f2").id == fix.id

    registry.remove(fix.id)
    low = registry.add(files_source(SOURCE_VECTORS, [tmp_path / "fix" / "F2.cif"]), low_priority=True)
    assert registry.cif_index()["f2"] == str(tmp_path / "cif" / "f2.cif")
    assert registry.vector_source_for_stem("f2").id == folder.id
    assert registry.pairing().overridden_counts[low.id] == 1


def test_pairing_reports_matches_per_image_source_and_unpaired_vectors(tmp_path: Path) -> None:
    registry = SourceRegistry()
    lot = _scanned_folder(registry, SOURCE_IMAGES, tmp_path / "lot", ["f1.png", "f2.png", "f3.png"])
    extra = registry.add(files_source(SOURCE_IMAGES, [tmp_path / "x" / "g1.png"]))
    cif = _scanned_folder(registry, SOURCE_VECTORS, tmp_path / "cif", ["f1.cif", "f2.cif", "g1.cif", "orphan.cif"])

    report = registry.pairing()

    assert [(p.image_source_id, p.matched, p.total) for p in report.pairs_for_vector_source(cif.id)] == [
        (lot.id, 2, 3),
        (extra.id, 1, 1),
    ]
    assert report.unpaired_vector_counts[cif.id] == 1


def test_exclude_images_removes_files_and_excludes_from_folders(tmp_path: Path) -> None:
    registry = SourceRegistry()
    folder = _scanned_folder(registry, SOURCE_IMAGES, tmp_path / "lot", ["f1.png", "f2.png"])
    files = registry.add(files_source(SOURCE_IMAGES, [tmp_path / "x" / "g1.png"]))

    registry.exclude_images([tmp_path / "lot" / "f1.png", tmp_path / "x" / "g1.png"])

    assert registry.image_paths() == [str(tmp_path / "lot" / "f2.png")]
    assert registry.get(files.id) is None
    assert registry.get(folder.id).excluded == frozenset({str(tmp_path / "lot" / "f1.png")})


def test_removing_a_source_drops_only_its_frames(tmp_path: Path) -> None:
    registry = SourceRegistry()
    _scanned_folder(registry, SOURCE_IMAGES, tmp_path / "lot", ["f1.png"])
    extra = registry.add(files_source(SOURCE_IMAGES, [tmp_path / "x" / "g1.png"]))

    registry.remove(extra.id)

    assert registry.image_paths() == [str(tmp_path / "lot" / "f1.png")]


def test_image_duplicates_detect_same_stem_from_another_folder(tmp_path: Path) -> None:
    registry = SourceRegistry()
    _scanned_folder(registry, SOURCE_IMAGES, tmp_path / "lot", ["f1.png"])
    candidate = files_source(SOURCE_IMAGES, [tmp_path / "copy" / "F1.tif", tmp_path / "copy" / "n.png"])

    assert registry.image_duplicates(candidate) == [str(tmp_path / "copy" / "F1.tif")]


def test_payload_round_trip_rescans_folders_and_keeps_colors(tmp_path: Path) -> None:
    registry = SourceRegistry()
    folder = _scanned_folder(registry, SOURCE_IMAGES, tmp_path / "lot", ["f1.png", "f2.png"])
    registry.exclude_images([tmp_path / "lot" / "f2.png"])
    files = registry.add(files_source(SOURCE_VECTORS, [tmp_path / "v" / "f1.cif"]))
    registry.add(files_source(SOURCE_IMAGES, [tmp_path / "k" / "k.png"], locked=True))

    restored = SourceRegistry.from_payload(registry.to_payload())

    assert [source.id for source in restored.sources()] == [folder.id, files.id]
    restored_folder = restored.get(folder.id)
    assert restored_folder.origin == ORIGIN_FOLDER
    assert restored_folder.scanned is False
    assert restored_folder.paths == ()
    assert restored_folder.excluded == frozenset({str(tmp_path / "lot" / "f2.png")})
    assert restored.get(files.id).paths == (str(tmp_path / "v" / "f1.cif"),)
    assert {s.color_index for s in restored.sources()} == {folder.color_index, files.color_index}


def test_new_sources_get_distinct_colors(tmp_path: Path) -> None:
    registry = SourceRegistry()
    first = registry.add(files_source(SOURCE_IMAGES, [tmp_path / "a.png"]))
    second = registry.add(files_source(SOURCE_VECTORS, [tmp_path / "a.cif"]))
    assert first.color_index != second.color_index


def test_migration_from_directory_mode_settings(tmp_path: Path) -> None:
    from contour.application.sources import migrate_legacy_sources

    registry = migrate_legacy_sources(
        input_directory=str(tmp_path / "lot"),
        cif_directory=str(tmp_path / "cif"),
        explicit_image_list=False,
        image_paths=[str(tmp_path / "lot" / "f1.png")],
        vector_paths=[str(tmp_path / "cif" / "f1.cif"), str(tmp_path / "fix" / "f2.cif")],
    )

    kinds = [(s.kind, s.origin, s.root) for s in registry.sources()]
    assert kinds == [
        (SOURCE_IMAGES, ORIGIN_FOLDER, str(tmp_path / "lot")),
        (SOURCE_VECTORS, ORIGIN_FOLDER, str(tmp_path / "cif")),
        (SOURCE_VECTORS, "files", str(tmp_path / "fix")),
    ]


def test_migration_from_explicit_files(tmp_path: Path) -> None:
    from contour.application.sources import migrate_legacy_sources

    registry = migrate_legacy_sources(
        input_directory=str(tmp_path / "lot"),
        cif_directory="",
        explicit_image_list=True,
        image_paths=[str(tmp_path / "lot" / "a.png")],
        vector_paths=[],
    )

    (source,) = registry.sources()
    assert source.origin == "files"
    assert source.paths == (str(tmp_path / "lot" / "a.png"),)
