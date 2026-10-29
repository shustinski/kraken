import hashlib
import io
from zipfile import ZipFile

import numpy as np
import pytest
from PIL import Image

from kraken_core.asset_source import ServerAssetSource
from kraken_core.plugin_protocol import PluginFrameInput
from neuralimage.infrastructure.dataset_source import DatasetStreamSource
from neuralimage.lib.data_interfaces import SamplePrepareSettings
from neuralimage.lib.images import ImagePreparator
from neuralimage.model.NeuralNetwork.recognition_pipeline import cut_image_prepare


def make_source(monkeypatch, content: bytes, key: str, media_type: str):
    item = PluginFrameInput("frame", 1, 1, "version", hashlib.sha256(content).hexdigest(), media_type, "inputs/data")
    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout: io.BytesIO(content))
    return ServerAssetSource("http://localhost", "job", {key: item}, "test-token")


def image_bytes():
    with io.BytesIO() as output:
        Image.new("L", (16, 16), color=100).save(output, "PNG")
        return output.getvalue()


def dataset_bytes(*, filename="images/a.png", dimensions=(16, 16)):
    with io.BytesIO() as output:
        with ZipFile(output, "w") as archive:
            archive.writestr(filename, image_bytes())
            archive.writestr("cif/a.cif", f"9 S {dimensions[0]} {dimensions[1]};\nL M1;\nP 1 1 8 1 8 8 1 8;\nE;\n")
        return output.getvalue()


def test_server_dataset_generates_masks_without_local_input_files(monkeypatch, tmp_path):
    source = make_source(monkeypatch, dataset_bytes(), "dataset.zip", "application/zip")
    dataset = DatasetStreamSource.load(source, "dataset.zip", tmp_path / "virtual")
    image, label = dataset.sample_pairs[0]
    assert not image.exists() and not label.exists()
    prepared = ImagePreparator(image, SamplePrepareSettings(), asset_source=dataset)
    assert prepared.size == (16, 16)
    with dataset.open(str(label)) as stream, Image.open(stream) as mask:
        assert mask.size == (16, 16)
        assert np.asarray(mask).max() > 0
    assert not (tmp_path / "virtual").exists()


@pytest.mark.parametrize("filename,dimensions", [("../escape.png", (16, 16)), ("images/a.png", (32, 32))])
def test_server_dataset_rejects_unsafe_paths_and_bad_cif_dimensions(monkeypatch, tmp_path, filename, dimensions):
    source = make_source(
        monkeypatch, dataset_bytes(filename=filename, dimensions=dimensions), "dataset.zip", "application/zip"
    )
    with pytest.raises(ValueError):
        DatasetStreamSource.load(source, "dataset.zip", tmp_path / "virtual")
    assert not (tmp_path / "virtual").exists()


def test_recognition_cuts_server_image_without_local_file(monkeypatch, tmp_path):
    path = tmp_path / "virtual" / "a.png"
    source = make_source(monkeypatch, image_bytes(), str(path), "image/png")
    result = cut_image_prepare(path, (1, 8, 8), 0, asset_source=source)
    assert result["baseim_size"] == (16, 16)
    assert result["cutted_image"].shape == (9, 1, 8, 8)
    assert not path.exists()


def test_contour_preview_and_dataset_export_use_server_stream(monkeypatch, tmp_path):
    from contour.application.frame_lod import PyramidFrameStore
    from contour.domain import PolygonData
    from contour.infrastructure.image_source import AssetImageLoader
    from contour.serializers import export_dataset_frame, load_polygons_cif

    path = tmp_path / "virtual" / "inputs" / "data"
    source = make_source(monkeypatch, image_bytes(), "inputs/data", "image/png")
    loader = AssetImageLoader(source, tmp_path / "virtual")
    store = PyramidFrameStore([path], image_loader=loader)
    assert store.has_lod()
    assert store.get_thumbnail(0).shape == (16, 16, 3)
    polygons = [PolygonData(1, [(1, 1), (8, 1), (8, 8), (1, 8)])]
    saved = export_dataset_frame(tmp_path / "output", str(path), polygons, store.get_frame(0))
    with Image.open(saved["image"]) as image:
        assert image.size == (16, 16)
    _image_name, dimensions, loaded = load_polygons_cif(saved["cif"])
    assert dimensions == (16, 16) and len(loaded) == 1
    assert not path.exists()


def test_contour_batch_request_preserves_server_loader_across_serialization(monkeypatch, tmp_path):
    import pickle
    from contour.batch_worker import BatchChunkRequest, process_batch_chunk
    from contour.infrastructure.image_source import AssetImageLoader

    path = tmp_path / "virtual" / "inputs" / "data"
    source = make_source(monkeypatch, image_bytes(), "inputs/data", "image/png")
    request = BatchChunkRequest(
        chunk_id=1,
        image_paths=(str(path),),
        pipeline_config={},
        contour_settings={},
        output_directory=str(tmp_path / "outputs"),
        save_options={"save_cif": True},
        display_settings={},
        image_loader=AssetImageLoader(source, tmp_path / "virtual"),
    )
    restored = pickle.loads(pickle.dumps(request))
    result = process_batch_chunk(restored)
    assert len(result.metadata) == 1
    assert result.metadata[0].error is None
    assert result.metadata[0].saved_files["cif"]
    assert not path.exists()


def test_agent_preserves_username_for_torch_optimizer(monkeypatch):
    from kraken_agent.runner import SubprocessPluginRunner

    monkeypatch.setenv("USERNAME", "audit-user")
    monkeypatch.setenv("PRIVATE_TOKEN", "secret")
    environment = SubprocessPluginRunner._plugin_environment()
    assert environment["USERNAME"] == "audit-user"
    assert "PRIVATE_TOKEN" not in environment


def test_remote_hub_does_not_create_local_analysis_store(monkeypatch):
    from types import SimpleNamespace
    from kraken_hub.presentation.qt import manager_app

    controller = object.__new__(manager_app.DesktopController)
    controller.service = SimpleNamespace(is_remote_project=lambda _id: True)
    callbacks = []
    workspace = SimpleNamespace(analysisRequested=SimpleNamespace(connect=callbacks.append))
    remote_controllers = []

    def remote_controller(parent, service, project_id, **options):
        instance = SimpleNamespace(timer=object())
        remote_controllers.append((parent, service, project_id, options, instance))
        return instance

    monkeypatch.setattr("kraken_hub.presentation.qt.remote_analysis.RemoteAnalysisController", remote_controller)
    monkeypatch.setattr(
        manager_app, "FilesystemAnalysisStore", lambda *_args: pytest.fail("Local project store opened")
    )
    controller._configure_analysis(workspace, "server-project")
    assert workspace._analysis_store is None
    assert len(remote_controllers) == 1
    parent, service, project_id, _options, instance = remote_controllers[0]
    assert (parent, service, project_id) == (workspace, controller.service, "server-project")
    assert workspace._analysis_coordinator is instance
    assert workspace._analysis_timer is instance.timer
    assert len(callbacks) == 1


def test_contour_window_keeps_all_100_server_frames_after_refresh(tmp_path):
    import os
    import subprocess
    import sys
    from kraken_core.plugin_protocol import PluginJobManifest

    inputs = tuple(
        PluginFrameInput(
            f"frame-{index}",
            index % 10 + 1,
            index // 10 + 1,
            f"version-{index}",
            hashlib.sha256(image_bytes()).hexdigest(),
            "image/png",
            f"inputs/{index % 10 + 1}_{index // 10 + 1}.png",
        )
        for index in range(100)
    )
    manifest = PluginJobManifest(
        "job",
        "frames.vectorize.v1",
        "project",
        "layer",
        "actor",
        "target",
        inputs,
        {"input_transport": "server", "server_url": "http://localhost", "project_width": 10},
    )
    job_path = tmp_path / "job.json"
    job_path.write_text(manifest.to_json(), encoding="utf-8")
    script = tmp_path / "check_window.py"
    script.write_text(
        """
import os,io,sys,traceback,urllib.request
os.environ['QT_QPA_PLATFORM']='offscreen'
if os.name=='nt':
    import ctypes
    ctypes.windll.kernel32.SetErrorMode(3)
from pathlib import Path
from PIL import Image
from PyQt6.QtCore import QSettings,QTimer
from contour.application import bootstrap
from contour.application.cli import main
root=Path(sys.argv[1])
QSettings.setDefaultFormat(QSettings.Format.IniFormat)
QSettings.setPath(QSettings.Format.IniFormat,QSettings.Scope.UserScope,str(root/'settings'))
payload=io.BytesIO()
Image.new('L',(16,16),color=100).save(payload,'PNG')
urllib.request.urlopen=lambda request,timeout: io.BytesIO(payload.getvalue())
original=bootstrap.build_application
def build(argv):
    global app,window
    app,window=original([*argv,'--no-qss','--log-file',str(root/'contour.log')])
    QTimer.singleShot(1000,check)
    return app,window
def check():
    try:
        assert len(window._widget._workspace.image_paths)==100
        assert window._widget._image_list_model.rowCount()==100
        assert len(set(window._widget._base_frame_numbers))==100
        window._widget.refresh_image_list()
        QTimer.singleShot(500,finish)
    except BaseException:
        traceback.print_exc()
        app.exit(1)
def finish():
    try:
        assert len(window._widget._workspace.image_paths)==100
        assert window._widget._image_list_model.rowCount()==100
        assert window._widget._workspace.current_state is not None
        app.exit(0)
    except BaseException:
        traceback.print_exc()
        app.exit(1)
bootstrap.build_application=build
main(['--kraken-job-manifest',str(root/'job.json'),'--kraken-result-manifest',str(root/'result.json'),'--kraken-staging-root',str(root)])
""",
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, str(script), str(tmp_path)],
        env={**os.environ, "KRAKEN_ASSET_TOKEN": "test-token"},
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
