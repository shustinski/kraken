"""Headless Kraken Agent bridge for NeuralImage protocol v1 jobs.

Only staged inputs and a staged, hash-pinned model are accepted. The bridge
uses NeuralImage's existing recognition runtime and returns immutable,
lossless one-channel PNG masks containing only values 0 and 255.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from kraken_core.plugin_protocol import (
    PluginFrameOutput,
    PluginJobManifest,
    PluginJobOutcome,
    PluginOperation,
    PluginResultManifest,
    safe_relative_path,
)


try:
    APP_VERSION = version("neuralimage")
except PackageNotFoundError:
    # Source checkout fallback; the installed package metadata is authoritative
    # in production builds.
    APP_VERSION = "6.3.0"


JOB_ENV = "KRAKEN_JOB_MANIFEST"
RESULT_ENV = "KRAKEN_RESULT_MANIFEST"
STAGING_ENV = "KRAKEN_STAGING_ROOT"
_MAX_MANIFEST_BYTES = 16 * 1024 * 1024


class KrakenBridgeError(RuntimeError):
    """Raised when a NeuralImage managed job is unsafe or malformed."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_sha256(value: object, field_name: str) -> str:
    result = str(value or "").strip().lower()
    if len(result) != 64 or any(char not in "0123456789abcdef" for char in result):
        raise KrakenBridgeError(f"{field_name} must contain 64 hexadecimal characters")
    return result


def _root(value: str | os.PathLike[str]) -> Path:
    raw = Path(value)
    if raw.is_symlink():
        raise KrakenBridgeError("Kraken staging root must not be a symbolic link")
    try:
        resolved = raw.resolve(strict=True)
    except OSError as exc:
        raise KrakenBridgeError(f"Kraken staging root is unavailable: {raw}") from exc
    if not resolved.is_dir():
        raise KrakenBridgeError("Kraken staging root is not a directory")
    return resolved


def _direct_child(root: Path, value: str | os.PathLike[str], *, exists: bool) -> Path:
    raw = Path(value)
    if raw.is_symlink():
        raise KrakenBridgeError(f"Symbolic links are not accepted: {raw}")
    try:
        resolved = raw.resolve(strict=exists)
    except OSError as exc:
        raise KrakenBridgeError(f"Kraken bridge path is unavailable: {raw}") from exc
    if resolved.parent != root:
        raise KrakenBridgeError(f"Kraken bridge path must be a direct staging child: {raw}")
    return resolved


def _contained_file(root: Path, relative_path: str) -> Path:
    normalized = safe_relative_path(relative_path)
    cursor = root
    for part in normalized.split("/"):
        cursor = cursor / part
        if cursor.is_symlink():
            raise KrakenBridgeError(f"Staged path contains a symbolic link: {relative_path}")
    try:
        resolved = cursor.resolve(strict=True)
    except OSError as exc:
        raise KrakenBridgeError(f"Staged file is missing: {relative_path}") from exc
    if root not in resolved.parents or not resolved.is_file():
        raise KrakenBridgeError(f"Staged path is not a contained regular file: {relative_path}")
    return resolved


def _atomic_text(path: Path, payload: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise KrakenBridgeError(f"Temporary path already exists: {temporary.name}")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload.encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_copy(source: Path, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    if destination.exists() or destination.is_symlink() or temporary.exists() or temporary.is_symlink():
        raise KrakenBridgeError(f"Refusing to overwrite staged output: {destination.name}")
    try:
        with source.open("rb") as reader, temporary.open("xb") as writer:
            shutil.copyfileobj(reader, writer, length=1024 * 1024)
            writer.flush()
            os.fsync(writer.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _bool_parameter(parameters: Mapping[str, Any], name: str, default: bool) -> bool:
    value = parameters.get(name, default)
    if not isinstance(value, bool):
        raise KrakenBridgeError(f"{name} must be a boolean")
    return value


def _positive_int(parameters: Mapping[str, Any], name: str, default: int, *, maximum: int) -> int:
    value = parameters.get(name, default)
    if isinstance(value, bool):
        raise KrakenBridgeError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise KrakenBridgeError(f"{name} must be an integer") from exc
    if result < 1 or result > maximum:
        raise KrakenBridgeError(f"{name} must be between 1 and {maximum}")
    return result


def _patch_size(parameters: Mapping[str, Any]) -> tuple[int, int]:
    value = parameters.get("patch_size", 256)
    if isinstance(value, bool):
        raise KrakenBridgeError("patch_size must be an integer or a two-item array")
    if isinstance(value, int):
        pair = (value, value)
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            pair = (int(value[0]), int(value[1]))
        except (TypeError, ValueError) as exc:
            raise KrakenBridgeError("patch_size values must be integers") from exc
    else:
        raise KrakenBridgeError("patch_size must be an integer or a two-item array")
    if any(item < 16 or item > 8192 for item in pair):
        raise KrakenBridgeError("patch_size values must be between 16 and 8192")
    return pair


@dataclass(frozen=True, slots=True)
class HeadlessOptions:
    model_path: Path
    model_sha256: str
    model_version: str
    patch_size: tuple[int, int]
    batch_size: int
    overlap: int
    use_auto_threshold: bool
    threshold: float
    tta: bool
    postprocess_enabled: bool
    postprocess_kernel_size: int

    @classmethod
    def from_session(cls, session: "NeuralImageKrakenSession") -> "HeadlessOptions":
        parameters = session.manifest.parameters
        relative_model = str(parameters.get("model_relative_path", "")).strip()
        if not relative_model:
            raise KrakenBridgeError(
                "model_relative_path is required; V1 does not read models outside Kraken staging"
            )
        model = _contained_file(session.staging_root, relative_model)
        model_hash = _expected_sha256(parameters.get("model_sha256"), "model_sha256")
        if _sha256(model) != model_hash:
            raise KrakenBridgeError("Staged model checksum does not match model_sha256")
        patch_size = _patch_size(parameters)
        overlap_raw = parameters.get("overlap", 0)
        if isinstance(overlap_raw, bool):
            raise KrakenBridgeError("overlap must be an integer")
        try:
            overlap = int(overlap_raw)
        except (TypeError, ValueError) as exc:
            raise KrakenBridgeError("overlap must be an integer") from exc
        if overlap < 0 or overlap >= min(patch_size):
            raise KrakenBridgeError("overlap must be non-negative and smaller than patch_size")
        try:
            threshold = float(parameters.get("threshold", 0.5))
        except (TypeError, ValueError) as exc:
            raise KrakenBridgeError("threshold must be a number") from exc
        if not 0.0 <= threshold <= 1.0:
            raise KrakenBridgeError("threshold must be between 0 and 1")
        model_version = str(parameters.get("model_version", "")).strip() or f"sha256:{model_hash[:12]}"
        return cls(
            model_path=model,
            model_sha256=model_hash,
            model_version=model_version,
            patch_size=patch_size,
            batch_size=_positive_int(parameters, "batch_size", 1, maximum=1024),
            overlap=overlap,
            use_auto_threshold=_bool_parameter(parameters, "use_auto_threshold", False),
            threshold=threshold,
            tta=_bool_parameter(parameters, "tta", False),
            postprocess_enabled=_bool_parameter(parameters, "postprocess_enabled", False),
            postprocess_kernel_size=_positive_int(
                parameters,
                "postprocess_kernel_size",
                3,
                maximum=255,
            ),
        )

    def provenance(self) -> dict[str, Any]:
        return {
            "model_sha256": self.model_sha256,
            "model_version": self.model_version,
            "patch_size": list(self.patch_size),
            "batch_size": self.batch_size,
            "overlap": self.overlap,
            "use_auto_threshold": self.use_auto_threshold,
            "threshold": self.threshold,
            "tta": self.tta,
            "postprocess_enabled": self.postprocess_enabled,
            "postprocess_kernel_size": self.postprocess_kernel_size,
            "lossless_binary_png": True,
        }


@dataclass(frozen=True, slots=True)
class NeuralImageKrakenSession:
    manifest: PluginJobManifest
    staging_root: Path
    result_manifest_path: Path
    input_paths: tuple[Path, ...]
    output_directory: Path

    @classmethod
    def load(
        cls,
        *,
        job_manifest: str | os.PathLike[str],
        result_manifest: str | os.PathLike[str],
        staging_root: str | os.PathLike[str],
    ) -> "NeuralImageKrakenSession":
        root = _root(staging_root)
        job_path = _direct_child(root, job_manifest, exists=True)
        result_path = _direct_child(root, result_manifest, exists=False)
        if result_path.exists() or result_path.is_symlink():
            raise KrakenBridgeError("Kraken result manifest already exists")
        if not job_path.is_file() or job_path.stat().st_size > _MAX_MANIFEST_BYTES:
            raise KrakenBridgeError("Kraken job manifest is not a regular, reasonably sized file")
        try:
            manifest = PluginJobManifest.from_json(job_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError) as exc:
            raise KrakenBridgeError(f"Invalid Kraken job manifest: {exc}") from exc
        if manifest.operation != PluginOperation.BINARY_SEGMENT_FRAMES.value:
            raise KrakenBridgeError(f"NeuralImage does not support operation {manifest.operation!r}")
        if len(manifest.inputs) > 10_000:
            raise KrakenBridgeError("NeuralImage protocol v1 accepts at most 10000 frames per job")
        inputs: list[Path] = []
        stems: set[str] = set()
        for item in manifest.inputs:
            if item.media_type not in {"image/png", "image/tiff", "image/jpeg"}:
                raise KrakenBridgeError(f"Unsupported NeuralImage input media type: {item.media_type}")
            path = _contained_file(root, item.relative_path)
            if _sha256(path) != item.sha256:
                raise KrakenBridgeError(f"Staged input checksum mismatch: {item.relative_path}")
            if path.stem.casefold() in stems:
                raise KrakenBridgeError(
                    "NeuralImage job contains duplicate input stems; native output mapping would be ambiguous"
                )
            stems.add(path.stem.casefold())
            inputs.append(path)
        output_directory = root / "outputs"
        if output_directory.is_symlink():
            raise KrakenBridgeError("Kraken output directory must not be a symbolic link")
        output_directory.mkdir(exist_ok=True)
        output_directory = output_directory.resolve(strict=True)
        if output_directory.parent != root or not output_directory.is_dir():
            raise KrakenBridgeError("Kraken output directory escapes staging")
        return cls(manifest, root, result_path, tuple(inputs), output_directory)

    def write_result(self, result: PluginResultManifest) -> PluginResultManifest:
        if result.job_id != self.manifest.job_id:
            raise KrakenBridgeError("Refusing to write a result for another Kraken job")
        if self.result_manifest_path.exists() or self.result_manifest_path.is_symlink():
            raise KrakenBridgeError("Kraken result manifest already exists")
        _atomic_text(self.result_manifest_path, result.to_json())
        return result

    def failed_result(self, error: Exception | str) -> PluginResultManifest:
        message = str(error).strip() or type(error).__name__
        return PluginResultManifest(
            job_id=self.manifest.job_id,
            outcome=PluginJobOutcome.FAILED.value,
            plugin_id="neuralimage",
            plugin_version=APP_VERSION,
            errors=(message[:10_000],),
        )

    def run_headless(self) -> PluginResultManifest:
        try:
            options = HeadlessOptions.from_session(self)
            outputs, applied_threshold = _run_recognition(self, options)
            provenance = options.provenance()
            provenance["applied_threshold"] = applied_threshold
            result = PluginResultManifest(
                job_id=self.manifest.job_id,
                outcome=PluginJobOutcome.SUCCEEDED.value,
                plugin_id="neuralimage",
                plugin_version=APP_VERSION,
                outputs=tuple(outputs),
                applied_parameters=provenance,
            )
        except Exception as exc:
            result = self.failed_result(exc)
        return self.write_result(result)


def _validate_binary_png(path: Path) -> None:
    from PIL import Image

    if path.is_symlink() or not path.is_file():
        raise KrakenBridgeError(f"NeuralImage output is not a regular file: {path.name}")
    with Image.open(path) as image:
        image.load()
        if image.format != "PNG" or image.mode != "L":
            raise KrakenBridgeError("NeuralImage managed output must be a one-channel PNG")
        histogram = image.histogram()
    if any(histogram[index] for index in range(1, 255)):
        raise KrakenBridgeError("NeuralImage managed output contains values other than 0 and 255")


def _run_recognition(
    session: NeuralImageKrakenSession,
    options: HeadlessOptions,
) -> tuple[list[PluginFrameOutput], float | None]:
    # Heavy ML imports remain behind the headless execution boundary so CLI
    # validation and standalone UI startup do not eagerly load torch.
    from .lib.data_interfaces import RecognitionParameters
    from .lib.message_bus import MessageBus
    from .model.NeuralNetwork.model_train_and_recognition import NeuralRecognizer

    native_directory = session.output_directory / ".neuralimage-native"
    if native_directory.is_symlink():
        raise KrakenBridgeError("Native output directory must not be a symbolic link")
    native_directory.mkdir(exist_ok=True)
    native_directory = native_directory.resolve(strict=True)
    if native_directory.parent != session.output_directory:
        raise KrakenBridgeError("Native output directory escapes staging")

    completed: dict[Path, Path] = {}
    bus = MessageBus()

    def capture_metrics(payload: object) -> None:
        if not isinstance(payload, dict) or payload.get("type") != "recognition_completed":
            return
        try:
            source = Path(str(payload["source_path"])).resolve(strict=True)
            output = Path(str(payload["output_path"])).resolve(strict=True)
        except (KeyError, OSError):
            return
        completed[source] = output

    bus.subscribe("metrics", capture_metrics)
    parameters = RecognitionParameters(
        source_files=list(session.input_paths),
        result_folder=native_directory,
        model=options.model_path,
        part_size=options.patch_size,
        batch_size=options.batch_size,
        overlap=options.overlap,
        recognition_multiprocessing_enabled=False,
        binarize_output=True,
        use_auto_threshold=options.use_auto_threshold,
        threshold=options.threshold,
        postprocess_enabled=options.postprocess_enabled,
        postprocess_kernel_size=options.postprocess_kernel_size,
        recognition_tta_enabled=options.tta,
        confidence_tta_enabled=False,
        confidence_save_mode="off",
        lossless_binary_png=True,
    )
    recognizer = NeuralRecognizer(parameters, bus)
    recognizer.run(multithreading=False)

    outputs: list[PluginFrameOutput] = []
    for index, (frame, source) in enumerate(zip(session.manifest.inputs, session.input_paths, strict=True), start=1):
        native = completed.get(source.resolve(strict=True))
        if native is None:
            raise KrakenBridgeError(f"Inference did not produce a result for frame {frame.frame_id}")
        if native_directory not in native.parents:
            raise KrakenBridgeError("Inference returned an output outside its staging directory")
        _validate_binary_png(native)
        destination = session.output_directory / f"{index:06d}_{frame.x}_{frame.y}.png"
        _atomic_copy(native, destination)
        _validate_binary_png(destination)
        outputs.append(
            PluginFrameOutput(
                output_id=f"{session.manifest.job_id}:neuralimage:{index}",
                frame_id=frame.frame_id,
                relative_path=destination.relative_to(session.staging_root).as_posix(),
                sha256=_sha256(destination),
                media_type="image/png",
                role="binary-image",
            )
        )
    applied_threshold = getattr(recognizer, "_resolved_output_threshold", options.threshold)
    return outputs, None if applied_threshold is None else float(applied_threshold)


def load_session_from_values(
    *,
    job_manifest: str | None,
    result_manifest: str | None,
    staging_root: str | None,
    environ: Mapping[str, str] | None = None,
) -> NeuralImageKrakenSession | None:
    environment = os.environ if environ is None else environ
    values = {
        "job_manifest": job_manifest or environment.get(JOB_ENV),
        "result_manifest": result_manifest or environment.get(RESULT_ENV),
        "staging_root": staging_root or environment.get(STAGING_ENV),
    }
    if not any(values.values()):
        return None
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise KrakenBridgeError("Incomplete Kraken bridge configuration: " + ", ".join(missing))
    return NeuralImageKrakenSession.load(
        job_manifest=str(values["job_manifest"]),
        result_manifest=str(values["result_manifest"]),
        staging_root=str(values["staging_root"]),
    )


__all__ = [
    "HeadlessOptions",
    "KrakenBridgeError",
    "NeuralImageKrakenSession",
    "load_session_from_values",
]
