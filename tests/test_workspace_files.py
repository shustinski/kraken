from __future__ import annotations

import os
from pathlib import Path

import pytest
from PIL import Image

from kraken_manager.workspace import (
    DerivedRunKind,
    ImageConversionSettings,
    LayerSourceMode,
    WorkspaceValidationError,
    map_frame_positions,
    scan_layer_source,
    validate_workspace_name,
)
from kraken_manager.infrastructure.workspace_files import (
    WorkspaceFileService,
    WorkspaceRegistry,
    validate_workspace_roots,
)


def _service(tmp_path: Path) -> WorkspaceFileService:
    return WorkspaceFileService(WorkspaceRegistry(tmp_path / "catalog"))


def _roots(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "source"
    derived = tmp_path / "derived"
    source.mkdir()
    derived.mkdir()
    return source, derived


@pytest.mark.parametrize("name", ["", " layer", "layer.", "A/B", "CON", "Lpt1.txt"])
def test_windows_unsafe_workspace_names_are_rejected(name: str) -> None:
    with pytest.raises(WorkspaceValidationError):
        validate_workspace_name(name, field_name="Layer name")


def test_roots_must_be_distinct_and_not_nested(tmp_path: Path) -> None:
    source, derived = _roots(tmp_path)
    assert validate_workspace_roots(source, derived) == (source.resolve(), derived.resolve())
    nested = source / "nested"
    nested.mkdir()
    with pytest.raises(WorkspaceValidationError):
        validate_workspace_roots(source, nested)


def test_scan_selects_densest_then_nearest_directory_and_maps_last_number(
    tmp_path: Path,
) -> None:
    root = tmp_path / "microscope"
    nearer = root / "nearer"
    deeper = root / "other" / "deeper"
    nearer.mkdir(parents=True)
    deeper.mkdir(parents=True)
    for directory in (nearer, deeper):
        (directory / "notes.txt").write_text("aux", encoding="utf-8")
        (directory / "tile_row9_0.jpg").write_bytes(b"jpg")
        (directory / "tile_row9_2.jpg").write_bytes(b"jpg")
    (nearer / "layout.ssc").write_text("ssc", encoding="utf-8")
    (deeper / "layout.ssc").write_text("ssc", encoding="utf-8")

    scan = scan_layer_source(root, maximum_frames=4)

    assert scan.ready
    assert Path(scan.working_directory) == nearer.resolve()
    assert scan.frame_positions == {"tile_row9_0.jpg": 1, "tile_row9_2.jpg": 3}
    assert len(scan.ssc_files) == 1


def test_mixed_jpg_and_bmp_and_duplicate_frames_block_scan(tmp_path: Path) -> None:
    mixed = tmp_path / "mixed"
    mixed.mkdir()
    (mixed / "frame0.jpg").write_bytes(b"jpg")
    (mixed / "frame1.bmp").write_bytes(b"bmp")
    assert not scan_layer_source(mixed, maximum_frames=4).ready

    with pytest.raises(WorkspaceValidationError, match="повторяется"):
        map_frame_positions(
            (Path("microscope_0.jpg"), Path("alternate_0.jpg")),
            maximum=4,
        )

    with pytest.raises(WorkspaceValidationError, match="от 0 до 3"):
        map_frame_positions((Path("frame_4.jpg"),), maximum=4)


def test_project_layout_and_versioned_run_paths(tmp_path: Path) -> None:
    source, derived = _roots(tmp_path)
    service = _service(tmp_path)
    binding = service.create_project(
        project_id="project-1",
        project_name="Chip A",
        source_root=source,
        derived_root=derived,
    )

    for category in ("img", "ssc", "prv", "aux"):
        assert (Path(binding.source_project_dir) / category).is_dir()
    for category in ("dataset", "result", "vector"):
        assert (Path(binding.derived_project_dir) / category).is_dir()

    first = service.begin_run(
        project_id="project-1",
        layer_id="layer-1",
        layer_name="Metal",
        kind=DerivedRunKind.VECTOR,
        plugin_id="contour",
        operation="vectorize",
    )
    second = service.begin_run(
        project_id="project-1",
        layer_id="layer-1",
        layer_name="Metal",
        kind=DerivedRunKind.VECTOR,
        plugin_id="contour",
        operation="vectorize",
    )
    assert first.path != second.path
    assert Path(first.path).parent == Path(binding.derived_project_dir) / "vector" / "Metal"
    assert Path(first.path).is_dir()
    assert Path(second.path).is_dir()


def test_bmp_import_converts_images_and_preserves_originals_in_aux(tmp_path: Path) -> None:
    source_root, derived_root = _roots(tmp_path)
    microscope = tmp_path / "microscope"
    working = microscope / "capture"
    nested = microscope / "metadata" / "calibration"
    empty = microscope / "metadata" / "empty-folder"
    working.mkdir(parents=True)
    nested.mkdir(parents=True)
    empty.mkdir(parents=True)
    Image.new("RGB", (2, 1), (255, 0, 0)).save(working / "tile_0.bmp")
    Image.new("RGB", (2, 1), (0, 255, 0)).save(working / "tile_1.bmp")
    (working / "layout.ssc").write_text("ssc", encoding="utf-8")
    (working / "preview.prv").write_text("prv", encoding="utf-8")
    (nested / "lens.txt").write_text("keep", encoding="utf-8")
    scan = scan_layer_source(microscope, maximum_frames=4)
    service = _service(tmp_path)
    project = service.create_project(
        project_id="project-1",
        project_name="Chip",
        source_root=source_root,
        derived_root=derived_root,
    )

    binding = service.import_layer(
        project=project,
        layer_id="layer-1",
        layer_name="Metal",
        scan=scan,
        conversion=ImageConversionSettings(
            target_format="png",
            flip_horizontal=True,
            png_compression=6,
        ),
    )

    assert binding.mode is LayerSourceMode.MANAGED_COPY
    assert sorted(path.name for path in Path(binding.image_directory).iterdir()) == [
        "tile_0.png",
        "tile_1.png",
    ]
    assert (Path(binding.ssc_directory) / "layout.ssc").is_file()
    assert (Path(binding.prv_directory) / "preview.prv").is_file()
    assert (Path(binding.aux_directory) / "capture" / "tile_0.bmp").is_file()
    assert (Path(binding.aux_directory) / "metadata" / "calibration" / "lens.txt").is_file()
    assert (Path(binding.aux_directory) / "metadata" / "empty-folder").is_dir()
    assert (working / "tile_0.bmp").is_file()


def test_changed_source_is_rejected_without_partial_layer(tmp_path: Path) -> None:
    source_root, derived_root = _roots(tmp_path)
    microscope = tmp_path / "microscope"
    microscope.mkdir()
    image = microscope / "frame_0.jpg"
    image.write_bytes(b"first")
    scan = scan_layer_source(microscope, maximum_frames=1)
    image.write_bytes(b"changed")
    os.utime(image, None)
    service = _service(tmp_path)
    project = service.create_project(
        project_id="project-1",
        project_name="Chip",
        source_root=source_root,
        derived_root=derived_root,
    )

    with pytest.raises(WorkspaceValidationError, match="После сканирования"):
        service.import_layer(
            project=project,
            layer_id="layer-1",
            layer_name="Metal",
            scan=scan,
            conversion=ImageConversionSettings(),
        )

    assert not (Path(project.source_project_dir) / "img" / "Metal").exists()
    assert not list(Path(project.source_project_dir).glob(".import-*"))


def test_external_layer_binding_keeps_absolute_paths_without_copying(tmp_path: Path) -> None:
    images = tmp_path / "external-images"
    images.mkdir()
    Image.new("RGB", (1, 1), "white").save(images / "frame_0.png")
    service = _service(tmp_path)

    binding = service.bind_external_layer(
        project_id="project-1",
        layer_id="layer-1",
        layer_name="Metal",
        image_directory=images,
        ssc_directory=None,
        prv_directory=None,
        maximum_frames=1,
    )

    assert binding.mode is LayerSourceMode.EXTERNAL
    assert binding.image_directory == str(images.resolve())
    assert binding.frame_positions == {"frame_0.png": 1}
    assert (images / "frame_0.png").is_file()


def test_external_plugin_output_is_copied_into_reserved_run(tmp_path: Path) -> None:
    source, derived = _roots(tmp_path)
    service = _service(tmp_path)
    service.create_project(
        project_id="project-1",
        project_name="Chip",
        source_root=source,
        derived_root=derived,
    )
    run = service.begin_run(
        project_id="project-1",
        layer_id="layer-1",
        layer_name="Metal",
        kind=DerivedRunKind.RESULT,
        plugin_id="neuralimage",
        operation="dataset.model.train.v1",
    )
    external = tmp_path / "external-result"
    external.mkdir()
    (external / "model.pt").write_bytes(b"model")

    published = service.publish_run(
        project_id="project-1",
        run_id=run.run_id,
        output_directory=external,
        provenance={"version": "1"},
    )

    assert published.state.value == "succeeded"
    assert Path(published.path) == Path(run.path)
    assert (Path(published.path) / "model.pt").read_bytes() == b"model"
    assert (external / "model.pt").is_file()


def test_managed_deletion_moves_to_trash_and_can_roll_back(tmp_path: Path) -> None:
    source, derived = _roots(tmp_path)
    service = _service(tmp_path)
    project = service.create_project(
        project_id="project-1",
        project_name="Chip",
        source_root=source,
        derived_root=derived,
    )
    for category in ("img", "ssc", "prv", "aux"):
        target = Path(project.source_project_dir) / category / "Metal"
        target.mkdir()
        (target / "file.dat").write_bytes(b"data")
    binding = service.registry.get_layer("project-1", "layer-1")
    assert binding is None
    from kraken_manager.workspace import LayerFileBinding

    binding = LayerFileBinding(
        layer_id="layer-1",
        layer_name="Metal",
        mode=LayerSourceMode.MANAGED_COPY,
        image_directory=str(Path(project.source_project_dir) / "img" / "Metal"),
        ssc_directory=str(Path(project.source_project_dir) / "ssc" / "Metal"),
        prv_directory=str(Path(project.source_project_dir) / "prv" / "Metal"),
        aux_directory=str(Path(project.source_project_dir) / "aux" / "Metal"),
    )
    service.registry.save_layer("project-1", binding)

    stage = service.stage_layer_deletion(
        project=project,
        binding=binding,
        delete_id="delete-1",
    )
    assert not Path(binding.image_directory).exists()
    assert any(path.exists() for _source, path in stage.moves)

    service.rollback_layer_deletion(stage)
    assert Path(binding.image_directory).is_dir()
    assert not any(path.exists() for path in stage.trash_roots)


def test_headless_run_failure_is_persisted_with_error_provenance(
    tmp_path: Path,
) -> None:
    source, derived = _roots(tmp_path)
    service = _service(tmp_path)
    service.create_project(
        project_id="project-1",
        project_name="Chip",
        source_root=source,
        derived_root=derived,
    )
    run = service.begin_run(
        project_id="project-1",
        layer_id="layer-1",
        layer_name="Metal",
        kind=DerivedRunKind.RESULT,
        plugin_id="neuralimage",
        operation="frames.binary-segment.v2",
    )

    failed = service.fail_run(
        project_id="project-1",
        run_id=run.run_id,
        error="model is unavailable",
    )

    assert failed.state.value == "failed"
    assert failed.provenance["error"] == "model is unavailable"
