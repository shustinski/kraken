"""Headless Contour prepares a dataset archive from paired images and CIF files."""

from __future__ import annotations

import hashlib
from pathlib import Path
from zipfile import ZipFile

from kraken_core.plugin_protocol import PluginFrameInput, PluginJobManifest, PluginResultManifest

from contour.worker import run


def test_headless_dataset_prepare_archives_image_and_cif(tmp_path: Path) -> None:
    image = b"png-bytes"
    label = b"DS 1 1 1;\nDF;\nE\n"
    staged = tmp_path / "inputs" / "1_1"
    staged.mkdir(parents=True)
    (staged / "frame.png").write_bytes(image)
    vectors = tmp_path / "inputs" / "vectors"
    vectors.mkdir()
    (vectors / "frame.cif").write_bytes(label)
    manifest = PluginJobManifest(
        job_id="dataset-job",
        operation="frames.dataset.prepare.v1",
        project_id="project",
        layer_id="layer",
        actor_id="actor",
        target_representation_id="target",
        inputs=(
            PluginFrameInput(
                frame_id="frame",
                x=1,
                y=1,
                artifact_version_id="image-version",
                sha256=hashlib.sha256(image).hexdigest(),
                media_type="image/png",
                relative_path="inputs/1_1/frame.png",
            ),
        ),
        parameters={
            "vector_inputs": [
                {
                    "frame_id": "frame",
                    "artifact_version_id": "cif-version",
                    "sha256": hashlib.sha256(label).hexdigest(),
                    "relative_path": "inputs/vectors/frame.cif",
                }
            ]
        },
    )
    job = tmp_path / "job.json"
    result_path = tmp_path / "result.json"
    job.write_text(manifest.to_json(), encoding="utf-8")

    result = run(job, result_path, tmp_path)

    published = PluginResultManifest.from_json(result_path.read_text(encoding="utf-8"))
    assert result.outcome == published.outcome == "succeeded"
    assert published.outputs[0].role == "dataset"
    assert published.outputs[0].media_type == "application/zip"
    with ZipFile(tmp_path / "outputs" / "dataset.zip") as archive:
        assert archive.read("images/frame.png") == image
        assert archive.read("cif/frame.cif") == label
