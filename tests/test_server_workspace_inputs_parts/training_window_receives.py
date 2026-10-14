"""test_server_workspace_inputs: test training window receives."""
from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
import pytest

@pytest.mark.parametrize("protocol", ["v1", "v2"])
def test_training_window_receives_samples_and_returns_model(qapp, tmp_path, monkeypatch, protocol):
    import hashlib
    import sys
    from uuid import uuid4
    from zipfile import ZipFile
    from PyQt6.QtWidgets import QMainWindow, QMessageBox
    from kraken_core.plugin_protocol import PluginAsset, PluginFrameInput, PluginJobManifest, PluginJobManifestV2
    from neuralimage.kraken_bridge import NeuralImageTrainSession

    archive = tmp_path / "dataset.zip"
    with ZipFile(archive, "w") as stream:
        stream.writestr("images/0000.jpg", b"sample")
        stream.writestr("cif/0000.cif", b"L M1; E;")
    identity = str(uuid4())
    input_values = dict(
        frame_id=identity,
        x=1,
        y=1,
        relative_path="dataset.zip",
        sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        media_type="application/zip",
    )
    common = dict(
        job_id=identity, project_id=identity, layer_id=identity, actor_id=identity, operation="dataset.model.train.v1"
    )
    if protocol == "v1":
        manifest = PluginJobManifest(
            **common,
            target_representation_id=identity,
            inputs=(PluginFrameInput(artifact_version_id=identity, **input_values),),
        )
    else:
        manifest = PluginJobManifestV2(
            **common, inputs=(PluginAsset(asset_id=identity, scope="frame", role="input", **input_values),)
        )
    session = NeuralImageTrainSession(manifest, tmp_path, tmp_path / "result.json")
    session.output_directory.mkdir()
    window = QMainWindow()
    values = {}
    window.set_jpg_path = lambda value: values.update(images=value)
    window.set_label_path = lambda value: values.update(cif=value)
    window.set_result_path = lambda value: values.update(result=value)
    state = SimpleNamespace()

    def execute():
        assert Path(values["images"]).is_dir()
        assert Path(values["cif"]).is_dir()
        assert state.sample_folder == values["images"]
        assert state.label_folder == values["cif"]
        assert state.result_folder == values["result"]
        (session.output_directory / "weights.pth").write_bytes(b"test-model")
        window._kraken_return_action.trigger()

    monkeypatch.setitem(
        sys.modules,
        "neuralimage.controller",
        SimpleNamespace(
            AppController=lambda **_kwargs: SimpleNamespace(
                main_window_presenter=SimpleNamespace(view=window, main_window_state=state), exec=execute
            )
        ),
    )
    monkeypatch.setattr(QMessageBox, "information", lambda *_args: QMessageBox.StandardButton.Ok)
    session.run_interactive()
    assert session.result_manifest_path.is_file()
