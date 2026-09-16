"""Spawned compute process. No Qt imports or interactive stdin."""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
import uuid
from pathlib import Path

from .contracts import input_bindings, safe_path
from .store import JobStore


def run_job(root: str, job: str) -> None:
    if os.name != "nt":
        os.setsid()
    # Uploads must never opt into unrestricted pickle loading.
    os.environ.pop("NEURALIMAGE_ALLOW_UNSAFE_MODEL_LOAD", None)
    from neuralimage.application.dto import MainWindowState, SettingsState
    from neuralimage.application.services.workflow_mapper import build_workflow_parameters
    from neuralimage.lib.message_bus import MessageBus
    from neuralimage.model.general_neural_handler import GeneralNeuralHandler

    store = JobStore(Path(root))
    payload = json.loads(store.get(job)["payload"])
    folder = store.root / job
    output = folder / "outputs"
    errors = []
    finished = threading.Event()

    def encode(value):
        if isinstance(value, Path):
            if value.resolve().is_relative_to(output):
                return {"artifact": value.resolve().relative_to(output).as_posix()}
            return str(value)
        if isinstance(value, dict):
            return {str(k): encode(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [encode(v) for v in value]
        if hasattr(value, "shape") and hasattr(value, "dtype"):
            import numpy as np

            name = f"previews/{uuid.uuid4().hex}.npy"
            target = safe_path(output, name)
            target.parent.mkdir(exist_ok=True)
            with target.open("wb") as stream:
                np.save(stream, value, allow_pickle=False)
            return {"array_artifact": name}
        if hasattr(value, "item"):
            return encode(value.item())
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        raise TypeError(f"Unsupported event payload: {type(value).__name__}")

    class RemoteBus(MessageBus):
        def publish(self, topic, payload=None):
            if topic in {"logging", "error"}:
                with (output / "run.log").open("a", encoding="utf-8") as log:
                    log.write(f"[{topic}] {payload}\n")
            if topic == "error":
                errors.append(str(payload))
            if topic == "metrics" and isinstance(payload, dict) and payload.get("type") == "training_completed":
                (output / "training-completed.json").write_text(json.dumps(payload), encoding="utf-8")
            if topic == "metrics" and isinstance(payload, dict) and payload.get("type") == "recognition_completed":
                payload = dict(payload)
                payload["source_input"] = Path(payload["source_path"]).relative_to(folder / "inputs").as_posix()
                payload["output_artifact"] = Path(payload["output_path"]).relative_to(output).as_posix()
            store.event(job, topic, encode(payload))
            super().publish(topic, payload)

    def question(text, header, default_answer=False, timeout_seconds=None):
        question_id = uuid.uuid4().hex
        store.event(
            job,
            "question",
            {
                "id": question_id,
                "text": str(text),
                "title": str(header),
                "default": bool(default_answer),
                "timeout": timeout_seconds,
            },
        )
        deadline = time.monotonic() + timeout_seconds if timeout_seconds else None
        while not finished.wait(0.2):
            answer = store.read_answer(job, question_id)
            if answer is not None:
                return answer
            if store.get(job)["command"]:
                store.answer(job, question_id, False)
                return False
            if deadline and time.monotonic() >= deadline:
                store.answer(job, question_id, bool(default_answer))
                return bool(default_answer)
        return False

    try:
        import torch
        import numpy as np

        # Existing checkpoints contain NumPy RNG state. Validate uploaded files
        # with a narrow allowlist before the legacy trainer reads them.
        numpy_globals = [
            np.ndarray,
            np.dtype,
            np.dtypes.UInt32DType,
            np._core.multiarray._reconstruct,
            (np._core.multiarray._reconstruct, "numpy.core.multiarray._reconstruct"),
        ]
        for entry in payload["files"]:
            if entry["path"].endswith(".ckpt"):
                with torch.serialization.safe_globals(numpy_globals):
                    torch.load(safe_path(folder / "inputs", entry["path"]), map_location="cpu", weights_only=True)
        for container, name, prefix in input_bindings(payload):
            if container.get(name):
                container[name] = str(safe_path(folder / "inputs", container[name]))
            elif name != "offline_manifest":
                missing = folder / "inputs" / prefix
                if name != "model_path":
                    missing.mkdir(parents=True, exist_ok=True)
                container[name] = str(missing)
        payload["main"]["result_folder"] = str(output / "recognition")
        mode, training, recognition = build_workflow_parameters(
            MainWindowState(**payload["main"]), SettingsState(**payload["settings"])
        )
        training.artifact_dir = output / "training"
        training.resume_from_checkpoint = any(training.artifact_dir.glob("*.ckpt"))
        from neuralimage.lib.data_interfaces import WorkMode

        if mode in {WorkMode.continue_training, WorkMode.further_training}:
            source_model = Path(recognition.model)
            checkpoint = source_model.with_suffix(".ckpt")
            training.artifact_dir.mkdir(parents=True, exist_ok=True)
            target_checkpoint = training.artifact_dir / checkpoint.name
            if checkpoint.is_file() and not target_checkpoint.exists():
                shutil.copy2(checkpoint, target_checkpoint)
        completed = output / "training-completed.json"
        if completed.exists() and mode in {WorkMode.train_and_recognition, WorkMode.further_training}:
            mode = WorkMode.recognition_only
            recognition.model = Path(json.loads(completed.read_text(encoding="utf-8"))["model_path"])
        recognition.resume_manifest_path = output / "recognition" / ".neuralimage-recognition-progress.json"
        recognition.resume_from_manifest = recognition.resume_manifest_path.exists()
        os.environ["NEURALIMAGE_TORCH_COMPILE"] = "1" if payload["settings"].get("torch_compile_enabled") else "0"
        managed = payload.get("managed_recognition", False)
        if managed:
            from neuralimage.model.NeuralNetwork.model_train_and_recognition import NeuralRecognizer

            recognition.lossless_binary_png = True
            recognition.recognition_multiprocessing_enabled = False
            handler = NeuralRecognizer(recognition, RemoteBus())
        else:
            handler = GeneralNeuralHandler(mode, question, RemoteBus(), recognition, training)

        def monitor():
            while not finished.wait(0.2):
                command = store.get(job)["command"]
                if command:
                    if command == "pause":
                        handler.stop() if managed else handler.pause_execution()
                    else:
                        handler.stop() if managed else handler.stop_execution()
                    return

        control = threading.Thread(target=monitor, daemon=True)
        control.start()
        if managed:
            handler.run(multithreading=False)
            store.event(job, "managed_threshold", getattr(handler, "_resolved_output_threshold", recognition.threshold))
        else:
            handler.start()
        # Handler may return after its timeout; do not free the queue while its
        # non-daemon worker is still alive (process exit is the final barrier).
        command = store.get(job)["command"]
        status = "failed" if errors else "paused" if command == "pause" else "cancelled" if command else "succeeded"
        store.finish_compute(job, status, "\n".join(errors))
    except Exception as error:
        import traceback

        store.event(job, "error", traceback.format_exc())
        store.finish_compute(job, "failed", str(error))
    finally:
        finished.set()
