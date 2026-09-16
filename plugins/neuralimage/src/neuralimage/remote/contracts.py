"""Wire format independent of Qt, NumPy and the compute runtime."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, fields
from pathlib import Path, PurePosixPath
from typing import Any
from threading import Event

from neuralimage.application.dto import MainWindowState, SettingsState
from neuralimage.application.services.workflow_mapper import build_workflow_parameters

API_VERSION = 1
CHUNK_SIZE = 4 * 1024 * 1024
TERMINAL = frozenset({"succeeded", "failed", "cancelled", "interrupted"})
INPUT_FIELDS = {
    "main": ("source_folder", "label_folder", "sample_folder", "model_path"),
    "settings": ("validation_image_folder", "validation_label_folder"),
}


def input_bindings(payload: dict):
    """Yield mutable settings locations and their portable upload namespaces."""
    for section, names in INPUT_FIELDS.items():
        for name in names:
            yield payload[section], name, f"{section}/{name}"
    config = payload["settings"].get("sem_segmentation_config", {})
    mining = config.get("hard_mining", {})
    if "offline_manifest" in mining:
        yield mining, "offline_manifest", "settings/sem_segmentation_config/hard_mining/offline_manifest"


def safe_path(root: Path, relative: str) -> Path:
    parts = PurePosixPath(relative)
    if (
        not relative
        or "\\" in relative
        or ":" in relative
        or parts.is_absolute()
        or any(p in {"..", ".", ""} for p in relative.split("/"))
    ):
        raise ValueError("Expected a relative portable file path")
    result = root.joinpath(*parts.parts).resolve()
    if not result.is_relative_to(root.resolve()) or result == root.resolve():
        raise ValueError("Path escapes the job directory")
    return result


def digest(path: Path, cancelled: Event | None = None) -> str:
    with path.open("rb") as stream:
        if cancelled is None:
            return hashlib.file_digest(stream, "sha256").hexdigest()
        result = hashlib.sha256()
        while chunk := stream.read(CHUNK_SIZE):
            if cancelled.is_set():
                raise InterruptedError("Transfer cancelled")
            result.update(chunk)
        return result.hexdigest()


def validate_payload(payload: dict[str, Any]) -> None:
    if payload.get("version") != API_VERSION:
        raise ValueError("Unsupported protocol version")
    for key, cls in (("main", MainWindowState), ("settings", SettingsState)):
        raw = payload[key]
        if not isinstance(raw, dict) or set(raw) - {f.name for f in fields(cls)}:
            raise ValueError(f"Invalid {key} settings")
    main = MainWindowState(**payload["main"])
    settings = SettingsState(**payload["settings"])
    mode, _, _ = build_workflow_parameters(main, settings)
    if mode is None:
        raise ValueError("Unknown work mode")
    if type(payload.get("managed_recognition", False)) is not bool:
        raise ValueError("Invalid managed recognition flag")
    if payload.get("managed_recognition") and main.work_mode != "recognition_only":
        raise ValueError("Managed jobs only support recognition")
    if not isinstance(payload.get("client_id", ""), str) or len(payload.get("client_id", "")) > 128:
        raise ValueError("Invalid client identifier")
    seen = set()
    for item in payload["files"]:
        name = item["path"]
        safe_path(Path.cwd(), name)
        if name in seen or type(item["size"]) is not int or item["size"] < 0:
            raise ValueError("Invalid or duplicate file")
        if len(item["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in item["sha256"]):
            raise ValueError("Invalid SHA-256")
        seen.add(name)
    for container, name, _ in input_bindings(payload):
        value = container.get(name)
        if value:
            safe_path(Path.cwd(), value)

    # Fail early if this snapshot contains an unsupported external resource.
    def check(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if child and (key.endswith("_path") or key.endswith("_folder") or key.endswith("_manifest")):
                    if key not in sum((list(v) for v in INPUT_FIELDS.values()), []) + [
                        "result_folder",
                        "offline_manifest",
                    ]:
                        raise ValueError(f"External resource requires explicit packaging: {key}")
                check(child)
        elif isinstance(value, list):
            for child in value:
                check(child)

    check(payload["settings"])


def package_inputs(
    main: MainWindowState, settings: SettingsState, cancelled: Event | None = None
) -> tuple[dict, dict[str, Path]]:
    payload = {"version": API_VERSION, "main": asdict(main), "settings": asdict(settings), "files": []}
    payload["main"]["mode_state"] = {}
    payload["main"]["result_folder"] = "results"
    sources = {}
    mode = main.work_mode
    if mode == "recognition_only":
        for name in ("sample_folder", "label_folder"):
            payload["main"][name] = ""
        for name in INPUT_FIELDS["settings"]:
            payload["settings"][name] = ""
    if mode in {"train_only", "continue_training"}:
        payload["main"]["source_folder"] = ""
    if mode in {"train_only", "train_and_recognition"}:
        payload["main"]["model_path"] = ""
    if settings.validation_source != "external" or not settings.use_validation:
        for name in INPUT_FIELDS["settings"]:
            payload["settings"][name] = ""
    for container, name, prefix in input_bindings(payload):
        value = container.get(name)
        if not value:
            continue
        source = Path(value).resolve(strict=True)
        candidates = sorted(source.rglob("*")) if source.is_dir() else [source]
        if name == "model_path" and source.is_file():
            candidates += [
                sidecar for sidecar in (source.with_suffix(".ckpt"), source.with_suffix(".json")) if sidecar.is_file()
            ]
        container[name] = prefix if source.is_dir() else f"{prefix}/{source.name}"
        for file in candidates:
            if file.name.startswith(".neuralimage-remote-"):
                continue
            if file.is_symlink() or (source.is_dir() and not file.resolve().is_relative_to(source)):
                raise ValueError(f"Symbolic links are not accepted: {file}")
            if not file.is_file():
                continue
            relative = file.relative_to(source).as_posix() if source.is_dir() else file.name
            remote = f"{prefix}/{relative}"
            sources[remote] = file
            payload["files"].append({"path": remote, "size": file.stat().st_size, "sha256": digest(file, cancelled)})
    validate_payload(payload)
    return json.loads(json.dumps(payload)), sources
