from __future__ import annotations

from pathlib import Path

import pytest

from contour.kraken_bridge import ContourWorkspaceSession, prepare_contour_launch
from kraken_core.plugin_protocol import (
    WorkspacePluginContextV1,
    WorkspacePluginResultV1,
)
from neuralimage.kraken_bridge import NeuralImageWorkspaceSession


def _context(
    tmp_path: Path,
    *,
    plugin_id: str,
    operation: str,
    with_cif: bool = False,
) -> tuple[WorkspacePluginContextV1, Path]:
    images = tmp_path / "images"
    output = tmp_path / "output"
    images.mkdir()
    output.mkdir()
    inputs = {"images": str(images.resolve())}
    if with_cif:
        cif = tmp_path / "cif"
        cif.mkdir()
        inputs["cif"] = str(cif.resolve())
    context = WorkspacePluginContextV1(
        project_id="project-1",
        project_name="Chip",
        layer_id="layer-1",
        layer_name="Metal",
        operation=operation,
        plugin_id=plugin_id,
        run_id="run-1",
        input_directories=inputs,
        proposed_output_directory=str(output.resolve()),
        result_manifest_path=str((output / ".kraken-result.json").resolve()),
    )
    path = tmp_path / "context.json"
    context.write(path)
    return context, path


def test_workspace_contract_round_trip_and_absolute_path_policy(tmp_path: Path) -> None:
    context, path = _context(
        tmp_path,
        plugin_id="contour",
        operation="frames.vectorize.v1",
    )
    assert WorkspacePluginContextV1.read(path) == context

    result = WorkspacePluginResultV1(
        run_id=context.run_id,
        plugin_id=context.plugin_id,
        operation=context.operation,
        outcome="succeeded",
        output_directory=context.proposed_output_directory,
        provenance={"file_count": 2},
    )
    result_path = tmp_path / "result.json"
    result.write(result_path)
    assert WorkspacePluginResultV1.read(result_path) == result

    with pytest.raises(ValueError, match="absolute"):
        WorkspacePluginContextV1(
            project_id="project-1",
            project_name="Chip",
            layer_id="layer-1",
            layer_name="Metal",
            operation="frames.vectorize.v1",
            plugin_id="contour",
            run_id="run-1",
            input_directories={"images": "relative/images"},
            proposed_output_directory=context.proposed_output_directory,
            result_manifest_path=context.result_manifest_path,
        )


def test_contour_direct_workspace_launch_controls_input_and_output(tmp_path: Path) -> None:
    context, path = _context(
        tmp_path,
        plugin_id="contour",
        operation="frames.dataset.prepare.v1",
    )

    session, arguments = prepare_contour_launch(
        ["--kraken-workspace-context", str(path), "--language", "ru"]
    )

    assert isinstance(session, ContourWorkspaceSession)
    assert arguments[arguments.index("--input-dir") + 1] == context.input_directories["images"]
    assert arguments[arguments.index("--dataset-dir") + 1] == context.proposed_output_directory
    assert "--kraken-workspace-context" not in arguments


def test_neuralimage_training_workspace_requires_images_and_cif(tmp_path: Path) -> None:
    context, path = _context(
        tmp_path,
        plugin_id="neuralimage",
        operation="dataset.model.train.v1",
        with_cif=True,
    )
    session = NeuralImageWorkspaceSession.load(path)
    assert session.context == context
    assert session.images_directory == Path(context.input_directories["images"])
    assert session.cif_directory == Path(context.input_directories["cif"])
