"""Opt-in real GPU/HTTP integration: NEURALIMAGE_RUN_GPU_TESTS=1."""

from __future__ import annotations
import os
import time
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from neuralimage.application.dto import MainWindowState, SettingsState
from neuralimage.remote.contracts import package_inputs
from neuralimage.remote.server import create_app

pytestmark = pytest.mark.skipif(os.environ.get("NEURALIMAGE_RUN_GPU_TESTS") != "1", reason="opt-in GPU integration")


@pytest.mark.parametrize("pause_and_resume", [False, True])
def test_train_then_recognize_over_http(tmp_path, pause_and_resume, monkeypatch):
    images, labels = tmp_path / "images", tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    for i in range(2):
        image = np.zeros((64, 64), dtype=np.uint8)
        image[8 + i : 48 + i, 16:40] = 200
        Image.fromarray(image).save(images / f"frame{i}.png")
        Image.fromarray((image > 0).astype(np.uint8) * 255).save(labels / f"frame{i}.png")
    main = MainWindowState(
        work_mode="train_and_recognition",
        sample_folder=str(images),
        label_folder=str(labels),
        source_folder=str(images),
        result_folder=str(tmp_path / "results"),
        epochs=30 if pause_and_resume else 1,
    )
    settings = SettingsState(
        model="M 720k",
        sample_size=(64, 64),
        train_patch_size=(64, 64),
        recognition_patch_size=(64, 64),
        step=64,
        batch_size=2,
        train_batch_size=2,
        recognition_batch_size=2,
        vertical_rotation=False,
        horizontal_rotation=False,
        color_mode="L",
        dataloader_num_workers=0,
        recognition_multiprocessing_enabled=False,
        show_batch_preview=False,
        use_validation=False,
    )
    body, sources = package_inputs(main, settings)
    app = create_app(tmp_path / "server")
    with TestClient(app) as client:
        response = client.post("/api/v1/jobs", json={"request_key": "gpu-smoke", "payload": body})
        assert response.status_code == 200, response.text
        job = response.json()["id"]
        for name, source in sources.items():
            response = client.put(f"/api/v1/jobs/{job}/inputs/{name}", content=source.read_bytes())
            assert response.status_code == 200, response.text
        client.post(f"/api/v1/jobs/{job}/submit")
        deadline = time.monotonic() + 240
        cursor = 0
        pause_requested = False
        resumed = False
        while time.monotonic() < deadline:
            for event in client.get(f"/api/v1/jobs/{job}/events?after={cursor}").json():
                cursor = event["seq"]
                if event["topic"] == "question":
                    client.post(f"/api/v1/jobs/{job}/answer", json={"question": event["payload"]["id"], "value": True})
                elif event["topic"] in {"error", "logging"}:
                    print(event["payload"], flush=True)
            status = client.get(f"/api/v1/jobs/{job}").json()
            checkpoints = list((app.state.store.root / job / "outputs" / "training").glob("*.ckpt"))
            if pause_and_resume and not pause_requested and checkpoints and status["status"] == "running":
                response = client.post(f"/api/v1/jobs/{job}/pause")
                assert response.status_code == 200, response.text
                pause_requested = True
            if status["status"] == "paused":
                assert checkpoints, "Pause must persist a training checkpoint"
                response = client.post(f"/api/v1/jobs/{job}/resume")
                assert response.status_code == 200, response.text
                resumed = True
            if status["status"] in {"succeeded", "failed"}:
                break
            time.sleep(0.2)
        assert status["status"] == "succeeded", status
        assert resumed == pause_and_resume
        manifest = client.get(f"/api/v1/jobs/{job}/artifacts").json()
        assert any(item["path"].endswith(".pth") for item in manifest)
        outputs = [
            item
            for item in manifest
            if item["path"].startswith("recognition/") and item["path"].endswith((".png", ".jpg", ".jpeg", ".bmp"))
        ]
        assert len(outputs) >= 2, manifest
        # Execute the same trained model locally and compare the actual rasters.
        from neuralimage.application.services.workflow_mapper import build_workflow_parameters
        from neuralimage.model.general_neural_handler import GeneralNeuralHandler
        from neuralimage.lib.message_bus import MessageBus

        model = next(
            app.state.store.root / job / "outputs" / item["path"] for item in manifest if item["path"].endswith(".pth")
        )
        local_main = replace(
            main, work_mode="recognition_only", model_path=str(model), result_folder=str(tmp_path / "local")
        )
        mode, training, recognition = build_workflow_parameters(local_main, settings)
        errors = []
        bus = MessageBus()
        bus.subscribe("error", errors.append)
        GeneralNeuralHandler(mode, lambda *args, **kwargs: True, bus, recognition, training).start()
        assert not errors
        import io

        for item in outputs:
            raw = client.get(f"/api/v1/jobs/{job}/artifacts/{item['path']}").content
            remote_image = np.asarray(Image.open(io.BytesIO(raw)))
            local_file = tmp_path / "local" / Path(item["path"]).relative_to("recognition")
            assert np.array_equal(remote_image, np.asarray(Image.open(local_file)))

        if not pause_and_resume:
            # Exercise the lightweight Kraken Agent adapter against the real worker.
            from neuralimage.remote import managed
            from neuralimage.remote.client import RemoteClient
            from neuralimage.remote.contracts import safe_path

            class InProcessClient(RemoteClient):
                def request(self, method, route, payload=None, *, raw=None):
                    response = client.request(method, "/api/v1" + route, json=payload, content=raw)
                    response.raise_for_status()
                    return response.json()

                def download(self, job, destination, cancelled=None):
                    result = destination / job
                    for item in self.request("GET", f"/jobs/{job}/artifacts"):
                        path = safe_path(result, item["path"])
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(client.get(f"/api/v1/jobs/{job}/artifacts/{item['path']}").content)
                    return result

            monkeypatch.setattr(managed, "RemoteClient", InProcessClient)
            recognition.result_folder = tmp_path / "managed"
            recognition.source_files = sorted(images.glob("*.png"))
            completed = []
            managed_bus = MessageBus()
            managed_bus.subscribe("metrics", completed.append)
            managed.run_recognition(recognition, managed_bus, "http://testserver", "managed-gpu")
            frames = [value for value in completed if value.get("type") == "recognition_completed"]
            assert len(frames) == 2
            for value in frames:
                with Image.open(value["output_path"]) as result:
                    assert result.format == "PNG"
                    assert result.mode == "L"
                    assert set(np.unique(np.asarray(result))) <= {0, 255}
