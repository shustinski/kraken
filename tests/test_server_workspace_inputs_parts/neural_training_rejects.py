"""test_server_workspace_inputs: test neural training rejects."""
from __future__ import annotations
from types import SimpleNamespace
import pytest

@pytest.mark.parametrize("filename", ["../escape.jpg", "/absolute.jpg", "images/a.jpg:stream"])
def test_neural_training_rejects_unsafe_dataset_paths(tmp_path, filename):
    import hashlib
    from zipfile import ZipFile
    from neuralimage.kraken_bridge import NeuralImageTrainSession

    archive = tmp_path / "dataset.zip"
    with ZipFile(archive, "w") as stream:
        stream.writestr(filename, b"unsafe")
    manifest = SimpleNamespace(
        inputs=(
            SimpleNamespace(
                relative_path="dataset.zip",
                media_type="application/zip",
                sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
            ),
        )
    )
    session = NeuralImageTrainSession(manifest, tmp_path, tmp_path / "result.json")
    with pytest.raises(ValueError):
        session.prepare_dataset()
    assert not (tmp_path.parent / "escape.jpg").exists()
