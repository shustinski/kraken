"""HTTP execution for staged Kraken Agent recognition jobs, without ML imports."""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

from neuralimage.application.dto import MainWindowState, SettingsState
from neuralimage.lib.data_interfaces import RecognitionParameters
from neuralimage.lib.message_bus import MessageBus

from .client import RemoteClient
from .contracts import API_VERSION, digest, safe_path


def run_recognition(parameters: RecognitionParameters, bus: MessageBus, url: str, request_key: str) -> float | None:
    client = RemoteClient(url)
    client.capabilities()
    model = Path(parameters.model)
    sources = {f"frames/{index:06d}_{path.name}": path for index, path in enumerate(parameters.source_files)}
    if not sources:
        raise ValueError("Managed recognition requires at least one staged image")
    original = dict(sources)
    sources[f"model/{model.name}"] = model
    sidecar = model.with_suffix(".json")
    if sidecar.is_file():
        sources[f"model/{sidecar.name}"] = sidecar
    main = MainWindowState(work_mode="recognition_only", source_folder="frames", model_path=f"model/{model.name}")
    settings = SettingsState(
        recognition_patch_size=parameters.part_size,
        recognition_batch_size=parameters.batch_size,
        overlap=parameters.overlap,
        recognition_binarize_output=True,
        recognition_use_auto_threshold=parameters.use_auto_threshold,
        recognition_threshold=parameters.threshold,
        recognition_postprocess=parameters.postprocess_enabled,
        recognition_postprocess_kernel_size=parameters.postprocess_kernel_size,
        recognition_tta_enabled=parameters.recognition_tta_enabled,
        recognition_multiprocessing_enabled=False,
        confidence_save_mode="off",
    )
    payload = {
        "version": API_VERSION,
        "managed_recognition": True,
        "main": asdict(main),
        "settings": asdict(settings),
        "files": [
            {"path": name, "size": path.stat().st_size, "sha256": digest(path)} for name, path in sources.items()
        ],
    }
    job = client.request("POST", "/jobs", {"request_key": f"kraken:{request_key}", "payload": payload})["id"]
    status = client.request("GET", f"/jobs/{job}")["status"]
    if status == "uploading":
        client.upload(job, sources)
        client.request("POST", f"/jobs/{job}/submit")
    cursor = 0
    completed = []
    threshold = parameters.threshold
    while True:
        events = client.request("GET", f"/jobs/{job}/events?after={cursor}")
        for event in events:
            value = event["payload"]
            if event["topic"] == "metrics" and isinstance(value, dict) and value.get("type") == "recognition_completed":
                completed.append(value)
            elif event["topic"] == "managed_threshold":
                threshold = value
            elif event["topic"] in {"logging", "error"}:
                bus.publish(event["topic"], value)
            cursor = event["seq"]
        state = client.request("GET", f"/jobs/{job}")
        if not events and state["status"] == "succeeded":
            break
        if state["status"] in {"failed", "cancelled", "paused", "interrupted"}:
            raise RuntimeError(f"Remote Kraken job {job}: {state['status']}: {state['error']}")
        if not events:
            time.sleep(0.5)
    output = client.download(job, parameters.result_folder)
    for value in completed:
        bus.publish(
            "metrics",
            {
                "type": "recognition_completed",
                "source_path": str(original[value["source_input"]]),
                "output_path": str(safe_path(output, value["output_artifact"])),
            },
        )
    return threshold
