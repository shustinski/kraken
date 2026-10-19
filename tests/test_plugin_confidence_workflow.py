from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from PIL import Image

from kraken_core.plugin_protocol import PluginFrameInput, PluginJobManifest


@pytest.mark.parametrize("operation", ["frames.binary-segment.v2", "frames.confidence.v1"])
def test_neuralimage_runs_real_model_and_publishes_the_requested_output(tmp_path, monkeypatch, operation):
    import torch
    from neuralimage.model.NeuralNetwork import create_model
    from neuralimage.model.NeuralNetwork.model_io import save_model_artifact
    from neuralimage.kraken_bridge import NeuralImageKrakenSession

    monkeypatch.setenv("NEURALIMAGE_TORCH_COMPILE", "0")
    monkeypatch.delenv("NEURALIMAGE_SERVER_URL", raising=False)
    monkeypatch.delenv("NEURALIMAGE_REMOTE_ONLY", raising=False)
    torch.set_num_threads(1)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    model = inputs / "model.pth"
    save_model_artifact(create_model("S 660k", 1), model, model_name="S 660k", input_channels=1)
    image = inputs / "0000.png"
    Image.linear_gradient("L").resize((32, 32)).save(image)
    manifest = PluginJobManifest(
        job_id="confidence-test",
        actor_id="owner",
        project_id="project",
        layer_id="layer",
        target_representation_id="target",
        operation=operation,
        inputs=(
            PluginFrameInput(
                frame_id="frame",
                artifact_version_id="input",
                relative_path="inputs/0000.png",
                x=1,
                y=1,
                sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
                media_type="image/png",
            ),
        ),
        parameters={
            "model_relative_path": "inputs/model.pth",
            "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
            "patch_size": 32,
            "batch_size": 1,
            "overlap": 0,
        },
    )
    path = tmp_path / "job.json"
    path.write_text(manifest.to_json(), encoding="utf-8")
    session = NeuralImageKrakenSession.load(
        job_manifest=path, result_manifest=tmp_path / "result.json", staging_root=tmp_path
    )
    result = session.run_headless()
    assert result.outcome == "succeeded", result.errors
    assert len(result.outputs) == 1
    output = result.outputs[0]
    with Image.open(tmp_path / output.relative_path) as saved:
        assert saved.size == (32, 32)
        assert saved.mode == "L"
        assert output.role == ("confidence" if operation == "frames.confidence.v1" else "binary-image")
        if operation != "frames.confidence.v1":
            assert set(saved.getdata()) <= {0, 255}
    assert result.applied_parameters["format"] == "neuralimage-inference-settings"
    if operation == "frames.confidence.v1":
        assert result.applied_parameters["confidence_save_mode"] == "separate_grayscale"


def test_karakal_parameter_file_round_trip_uses_real_controls(tmp_path):
    import json
    from PyQt6.QtCore import QSettings
    from PyQt6.QtWidgets import QApplication
    from karakal.app.main_window import KarakalWidget

    app = QApplication.instance() or QApplication([])
    widget = KarakalWidget(settings=QSettings(str(tmp_path / "karakal.ini"), QSettings.Format.IniFormat))
    try:
        widget.point_match_radius_spin.setValue(7)
        payload = widget.parameter_payload()
        path = tmp_path / "parameters.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        widget.point_match_radius_spin.setValue(3)
        widget.load_parameter_file(path)
        assert widget.point_match_radius_spin.value() == 7
        for key, invalid in (("mask_threshold", 2.0), ("boundary_radius", -1), ("analysis_profile", "unknown")):
            invalid_payload = widget.parameter_payload()
            invalid_payload["settings"][key] = invalid
            path.write_text(json.dumps(invalid_payload), encoding="utf-8")
            with pytest.raises(ValueError):
                widget.load_parameter_file(path)
            assert widget.point_match_radius_spin.value() == 7
        payload["settings"]["point_match_radius"] = "invalid"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError):
            widget.load_parameter_file(path)
        assert widget.point_match_radius_spin.value() == 7
    finally:
        widget.close()
        app.processEvents()


@pytest.mark.parametrize("changed", ["none", "geometry", "disk"])
def test_dataset_export_reuses_only_verified_unchanged_cif(tmp_path, changed):
    import numpy as np
    from contour.domain import PolygonData
    from contour.application.processing import ImageProcessingState
    from contour.application.services.dataset_exporter import export_frame_to_dataset

    image = tmp_path / "frame.png"
    Image.new("L", (20, 20)).save(image)
    cif = tmp_path / "frame.cif"
    original = b"DS 1 1 1;\nL NM;\n( R frame.png );\n( S 20 20 );\nP 1 19 5 19 5 15 1 19;\nDF;\nE\n"
    cif.write_bytes(original)
    polygon = PolygonData(id=1, points=[(1, 1), (5, 1), (5, 5)])
    state = ImageProcessingState(
        image_path=str(image),
        source_image=np.zeros((20, 20), dtype=np.uint8),
        polygons=[polygon],
        reference_polygons=[polygon.clone()],
        loaded_cif_path=str(cif),
        loaded_cif_sha256=hashlib.sha256(original).hexdigest(),
    )
    if changed == "geometry":
        polygon.points[0] = (2, 2)
    elif changed == "disk":
        cif.write_bytes(b"external modification")
    result = export_frame_to_dataset(
        dataset_directory=str(tmp_path / "dataset"), image_path=str(image), state=state, polygons=[polygon]
    )
    assert result.saved_files
    saved = Path(result.saved_files["cif"]).read_bytes()
    assert saved != b"external modification"
    if changed == "none":
        assert saved == original
    elif changed == "geometry":
        assert saved != original
