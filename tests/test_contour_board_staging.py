"""Multi-layer review must never modify the original image directory."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from kraken_core.plugin_protocol import WorkspacePluginContextV1

from kraken_hub.presentation.qt.parts.contour_launch_arguments import ContourLaunchArgumentsMixin
from kraken_hub.presentation.qt.parts.layer_board_actions import LayerBoardActionsMixin
from kraken_manager.domain.project import RepresentationKind


@pytest.mark.parametrize("workspace_mode", [False, True])
def test_board_selection_uses_staged_images_and_matching_cif(tmp_path, workspace_mode):
    source = tmp_path / "Original images"
    source.mkdir()
    (source / "original.png").write_bytes(b"original")
    primary = SimpleNamespace(kind="source", representation_id="primary", node_id="primary")
    second = SimpleNamespace(kind="binary", representation_id="secondary-uuid", node_id="second")
    vector = SimpleNamespace(kind="vector", representation_id="vectors-uuid", node_id="vector")
    blobs = {"primary": b"primary", "secondary-uuid": b"secondary", "vectors-uuid": b"CIF"}

    class Controller(LayerBoardActionsMixin, ContourLaunchArgumentsMixin):
        def _layer_id_for_representation(self, _identifier):
            return "layer"

        def _frames_per_row_arguments(self):
            return ()

        def _error(self, text):
            raise AssertionError(text)

        def _launch_managed_plugin(self, _plugin, *, arguments):
            self.arguments = arguments
            return None

    controller = Controller()
    controller._project_id = "project"
    controller.service = SimpleNamespace(
        data_dir=tmp_path,
        list_representations=lambda *_args: (
            SimpleNamespace(id="primary", kind=RepresentationKind.IMAGE, source=str(source)),
        ),
        frame_cells=lambda _pid, _lid, rid: (SimpleNamespace(x=1, y=1, sha256=rid),),
        read_project_blob=lambda _pid, digest: blobs[digest],
    )
    monitored = []
    if workspace_mode:
        output = tmp_path / "result"
        output.mkdir()
        context_path = output / ".kraken-context.json"
        WorkspacePluginContextV1(
            project_id="project",
            project_name="Project",
            layer_id="layer",
            layer_name="Layer",
            operation="frames.vectorize.v1",
            plugin_id="contour",
            run_id="run",
            input_directories={"images": str(source)},
            proposed_output_directory=str(output),
            result_manifest_path=str(output / ".kraken-result.json"),
        ).write(context_path)
        controller._contour_launch_arguments = lambda **_kwargs: (
            ("--kraken-workspace-context", str(context_path)),
            {
                "result_directory": str(output),
                "workspace_result_manifest": str(output / ".kraken-result.json"),
                "workspace_run_id": "run",
            },
        )

        def launch(_plugin, *, arguments):
            controller.arguments = arguments
            return "process"

        controller._launch_managed_plugin = launch
        controller._monitor_workspace_result = lambda process, **kwargs: monitored.append((process, kwargs))
        controller.shell = SimpleNamespace(statusBar=lambda: SimpleNamespace(showMessage=lambda *_args: None))
    controller._open_board_selection_in_contour("layer", (primary, second, vector))
    assert [p.name for p in source.iterdir()] == ["original.png"]
    args = controller.arguments
    if workspace_mode:
        context = WorkspacePluginContextV1.read(context_path)
        images, vectors = Path(context.input_directories["images"]), Path(context.input_directories["vectors"])
        assert monitored == [
            (
                "process",
                {
                    "result_manifest": str(output / ".kraken-result.json"),
                    "workspace_run_id": "run",
                    "layer_id": "layer",
                },
            )
        ]
    else:
        images = Path(args[args.index("--input-dir") + 1])
        vectors = Path(args[args.index("--cif-dir") + 1])
    assert images != source
    image_files = tuple(images.glob("*.png"))
    assert {p.read_bytes() for p in image_files} == {b"primary", b"secondary"}
    assert {p.stem for p in image_files} == {p.stem for p in vectors.glob("*.cif")}
    assert all(p.read_bytes() == b"CIF" for p in vectors.glob("*.cif"))
    assert all("uuid" not in p.name for p in (*image_files, *vectors.glob("*.cif")))
