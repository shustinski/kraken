from dataclasses import replace
import hashlib
from io import BytesIO
import os
import subprocess
import sys
import zipfile

from PIL import Image, ImageDraw
import pytest

from kraken_core.plugin_protocol import PluginResultManifest
from neuralimage.kraken_bridge import KrakenBridgeError
from neuralimage.managed_training import training_configuration, validate_external_partition
from tests.test_plugin_app_v1_bridges import _manifest, _workspace


@pytest.mark.parametrize(
    "options",
    [
        dict(epochs=True),
        dict(epochs=0),
        dict(seed=-1),
        dict(settings={"model_path": "outside.pth"}),
        dict(settings={"learning_rate": float("nan")}),
        dict(validation_relative_root=12),
        dict(unknown=True),
        dict(settings={"early_stopping_patience": 1}),
        dict(settings={"early_stopping_restore_best_weights": False}),
    ],
    ids=["boolean-epoch", "zero-epoch", "seed", "foreign-path", "nan", "bad-root", "unknown",
         "unsupported-automatic-patience", "unsupported-automatic-restore-policy"],
)
def test_headless_training_configuration_rejects_unsafe_or_invalid_values(options):
    with pytest.raises(KrakenBridgeError):
        training_configuration({"training": options})


def test_renaming_training_image_does_not_create_an_independent_validation_set(tmp_path):
    train, validation = tmp_path / "train", tmp_path / "val"
    train.mkdir()
    validation.mkdir()
    (train / "one.jpg").write_bytes(b"identical image content")
    (validation / "renamed.jpg").write_bytes(b"identical image content")
    with pytest.raises(KrakenBridgeError, match="overlap"):
        validate_external_partition(train, validation)
    (validation / "renamed.jpg").write_bytes(b"independent image bytes")
    result = validate_external_partition(train, validation)
    assert result["train_images"] == result["validation_images"] == 1


@pytest.mark.parametrize(
    "entries,limits,message",
    [
        ([("images/a.png", b"image"), ("cif/a.cif", b"label")], {"max_members": 1}, "limits"),
        ([("images/a.png", b"x" * 100)], {"max_unpacked_bytes": 99}, "limits"),
        ([("images", b"file"), ("images/a.png", b"image")], {}, "conflict"),
        ([("images/A.png", b"one"), ("images/a.png", b"two")], {}, "duplicate"),
    ],
    ids=["member-limit", "expanded-byte-limit", "file-directory-conflict", "case-collision"],
)
def test_training_archive_preflight_rejects_before_extracting_files(tmp_path, entries, limits, message):
    from neuralimage.kraken_bridge import NeuralImageTrainSession

    encoded = BytesIO()
    with zipfile.ZipFile(encoded, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in entries:
            archive.writestr(name, content)
    payload = encoded.getvalue()
    manifest = _manifest(operation="dataset.model.train.v1", payload=payload)
    manifest = replace(manifest, inputs=(replace(manifest.inputs[0], media_type="application/zip"),))
    job, result = _workspace(tmp_path, manifest, payload)
    session = NeuralImageTrainSession.load(job_manifest=job, result_manifest=result, staging_root=tmp_path)
    with pytest.raises(KrakenBridgeError, match=message):
        session.prepare_dataset(**limits)
    assert not any(path.is_file() for path in (tmp_path / "dataset").rglob("*"))


@pytest.mark.skipif(
    os.environ.get("KRAKEN_RUN_TRAINING_LIVE") != "1", reason="Explicit actual NeuralImage training subprocess gate"
)
def test_actual_headless_training_publishes_real_model_and_applied_workflow(tmp_path):
    from contour.domain import PolygonData
    from contour.serializers import save_polygons_cif

    dataset = tmp_path / "source"
    for split in ("", "validation"):
        for kind in ("images", "cif"):
            (dataset / split / kind).mkdir(parents=True)
    for index in range(3):
        split = "" if index < 2 else "validation"
        image = Image.new("RGB", (128, 128), (20 + index, 20, 20))
        ImageDraw.Draw(image).rectangle((20 + index, 30, 100, 90), fill=(200, 200, 200))
        source = dataset / split / "images" / f"{index}.png"
        image.save(source)
        save_polygons_cif(
            dataset / split / "cif" / f"{index}.cif",
            str(source),
            [PolygonData(1, [(20 + index, 30), (100, 30), (100, 90), (20 + index, 90)])],
            (128, 128),
        )
    encoded = BytesIO()
    with zipfile.ZipFile(encoded, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in dataset.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(dataset).as_posix())
    payload = encoded.getvalue()
    parameters = {
        "execution_mode": "headless",
        "training": {
            "epochs": 1,
            "seed": 1234,
            "validation_relative_root": "validation",
            "settings": dict(
                model="S 660k",
                sample_size=[128, 128],
                train_patch_size=[128, 128],
                step=128,
                batch_size=2,
                train_batch_size=2,
                vertical_rotation=False,
                horizontal_rotation=False,
                use_validation=True,
                validation_source="external",
                dataloader_num_workers=0,
                early_stopping_enabled=True,
            ),
        },
    }
    manifest = _manifest(operation="dataset.model.train.v1", payload=payload, parameters=parameters)
    manifest = replace(manifest, inputs=(replace(manifest.inputs[0], media_type="application/zip"),))
    workspace = tmp_path / "job"
    job, result = _workspace(workspace, manifest, payload)
    environment = dict(
        os.environ,
        KRAKEN_JOB_MANIFEST=str(job),
        KRAKEN_RESULT_MANIFEST=str(result),
        KRAKEN_STAGING_ROOT=str(workspace),
        QT_QPA_PLATFORM="offscreen",
    )
    with (tmp_path / "training-subprocess.log").open("wb") as log:
        process = subprocess.run(
            [sys.executable, "-m", "neuralimage"], env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=180
        )
    assert process.returncode == 0, (tmp_path / "training-subprocess.log").read_text(
        encoding="utf-8", errors="replace"
    )[-6000:]
    publication = PluginResultManifest.from_json(result.read_text(encoding="utf-8"))
    assert publication.outcome == "succeeded" and len(publication.outputs) == 1
    output = publication.outputs[0]
    model = workspace / output.relative_path
    assert model.stat().st_size > 100000 and hashlib.sha256(model.read_bytes()).hexdigest() == output.sha256
    execution = publication.applied_parameters["managed_execution"]
    assert execution["dataset_sha256"] == hashlib.sha256(payload).hexdigest()
    assert execution["train_images"] == 2 and execution["validation_images"] == 1
    assert execution["epoch_upper_bound"] == 1 and execution["early_stopping_policy"] == "automatic_dataset_size"
    assert publication.applied_parameters["main_window_state"]["epochs"] == 1
    assert len(list(workspace.rglob("neuralimage_workflow.json"))) == 1
    training_log = (workspace / "outputs/training-run.log").read_text(encoding="utf-8")
    assert "Automatic early stopping enabled:" in training_log
    assert "max_epochs=1." in training_log
    from neuralimage.model.NeuralNetwork.model_io import load_model_artifact

    loaded = load_model_artifact(model, map_location="cpu")
    assert sum(parameter.numel() for parameter in loaded.parameters()) > 100000
