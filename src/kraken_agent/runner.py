"""Crash-aware subprocess execution against isolated staging workspaces."""

from __future__ import annotations

import json
import os
import subprocess
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping

from kraken_core.analysis_protocol import AnalysisOutcome
from kraken_core.analysis_run_protocol import AnalysisPartitionJobManifest, AnalysisPartitionResultManifest
from kraken_core.plugin_protocol import (
    PluginAssetScope,
    PluginJobManifest,
    PluginJobManifestV2,
    PluginJobOutcome,
    PluginResultManifest,
    PluginResultPublicationV2,
)
from kraken_core.safe_files import open_regular_append, open_regular_read

from .jobs import AgentJobState, DurableJobStore, JobStateError, StagingWorkspace, TERMINAL_STATES
from .protocols import parse_result_json


ANALYSIS_OPERATION = "karakal.analyze.v1"


@dataclass(frozen=True, slots=True)
class PluginProcessSpec:
    operation: str
    command: tuple[str, ...]
    working_directory: Path | None = None
    interactive: bool = False
    modes: frozenset[str] | None = None

    def __post_init__(self) -> None:
        if not self.operation or not self.command:
            raise ValueError("Plugin process operation and command are required")
        modes = (
            frozenset(self.modes)
            if self.modes is not None
            else frozenset(("interactive" if self.interactive else "headless",))
        )
        if not modes or not modes <= {"interactive", "headless"}:
            raise ValueError("Agent process modes must be interactive/headless")
        if ("interactive" if self.interactive else "headless") not in modes:
            raise ValueError("Agent process default mode must be supported")
        object.__setattr__(self, "modes", modes)

    def for_execution(self, parameters: Mapping[str, object]) -> PluginProcessSpec:
        mode = parameters.get("execution_mode")
        if mode is None:
            return self
        if not isinstance(mode, str) or mode not in self.modes:
            raise ValueError("Requested execution mode is not supported by the registered process")
        return replace(self, interactive=mode == "interactive")


class PluginRegistry:
    def __init__(self, specs: Mapping[str, PluginProcessSpec] | None = None) -> None:
        self._specs = dict(specs or {})

    def get(self, operation: str) -> PluginProcessSpec | None:
        return self._specs.get(operation)

    def operations(self) -> frozenset[str]:
        return frozenset(self._specs)

    @classmethod
    def from_json(cls, path: Path | str) -> "PluginRegistry":
        payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        if not isinstance(payload, dict) or not isinstance(payload.get("plugins"), list):
            raise ValueError("Agent plugin config must contain a plugins array")
        specs: dict[str, PluginProcessSpec] = {}
        for item in payload["plugins"]:
            if not isinstance(item, dict) or not isinstance(item.get("command"), list):
                raise ValueError("Invalid plugin process specification")
            operation = str(item.get("operation", ""))
            spec = PluginProcessSpec(
                operation=operation,
                command=tuple(str(part) for part in item["command"]),
                working_directory=None
                if item.get("working_directory") is None
                else Path(str(item["working_directory"])).resolve(strict=True),
                interactive=bool(item.get("interactive", False)),
                modes=None if item.get("modes") is None else frozenset(item["modes"]),
            )
            if operation in specs:
                raise ValueError(f"Duplicate agent operation: {operation}")
            specs[operation] = spec
        return cls(specs)


class SubprocessPluginRunner:
    _ENV_ALLOWLIST = frozenset(
        {
            "PATH",
            "PATHEXT",
            "SYSTEMROOT",
            "WINDIR",
            "COMSPEC",
            "TEMP",
            "TMP",
            "TMPDIR",
            "HOME",
            "USERPROFILE",
            "USERNAME",
            "USER",
            "LOGNAME",
            "HOMEDRIVE",
            "HOMEPATH",
            "LOCALAPPDATA",
            "APPDATA",
            "XDG_CONFIG_HOME",
            "XDG_DATA_HOME",
            "XDG_CACHE_HOME",
            "XDG_STATE_HOME",
            "LANG",
            "LC_ALL",
            "PYTHONPATH",
            "PYTHONHOME",
            "VIRTUAL_ENV",
            "KRAKEN_PROCESS_LIFETIME_ROOT_PID",
            "DISPLAY",
            "WAYLAND_DISPLAY",
            "XAUTHORITY",
            "XDG_RUNTIME_DIR",
            "QT_PLUGIN_PATH",
            "QT_QPA_PLATFORM",
            "CUDA_VISIBLE_DEVICES",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
        }
    )
    _MAX_RESULT_MANIFEST_BYTES = 16 * 1024 * 1024

    def __init__(
        self,
        store: DurableJobStore,
        staging_root: Path | str,
        registry: PluginRegistry,
        *,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        self.store = store
        self.staging_root = Path(staging_root).resolve()
        self.staging_root.mkdir(parents=True, exist_ok=True)
        self.registry = registry
        self.environment = dict(environment or {})
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    @classmethod
    def _plugin_environment(cls) -> dict[str, str]:
        return {key: value for key, value in os.environ.items() if key.upper() in cls._ENV_ALLOWLIST}

    @staticmethod
    def _terminate(process: subprocess.Popen[bytes]) -> None:
        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)

    @staticmethod
    def _validate_result_contract(
        manifest: PluginJobManifest | PluginJobManifestV2,
        result: PluginResultManifest | PluginResultPublicationV2,
    ) -> None:
        if isinstance(manifest, PluginJobManifestV2):
            if not isinstance(result, PluginResultPublicationV2):
                raise ValueError("A V2 job requires a V2 result publication")
            input_frames = {
                str(item.frame_id)
                for item in manifest.inputs
                if item.scope is PluginAssetScope.FRAME and item.frame_id is not None
            }
            for output in result.outputs:
                if output.scope is PluginAssetScope.FRAME and output.frame_id not in input_frames:
                    raise ValueError(f"Plugin returned an unknown frame: {output.frame_id}")
            if not result.final:
                raise ValueError("A terminating V2 process must mark its publication final")
            if result.outcome in {"failed", "cancelled"} and result.outputs:
                raise ValueError("Failed or cancelled V2 results cannot publish outputs")
            if manifest.operation == "layer.stitch.v1":
                if result.outcome not in {"succeeded", "failed", "cancelled"}:
                    raise ValueError("A mosaic cannot be partially published")
                if result.outcome == "succeeded" and (len(result.outputs) != 1
                        or result.outputs[0].scope is not PluginAssetScope.LAYER
                        or result.outputs[0].role != "mosaic"
                        or result.outputs[0].media_type != "application/vnd.kraken.mosaic+sqlite3"):
                    raise ValueError("Stitching must return exactly one complete layer mosaic")
            return
        if not isinstance(result, PluginResultManifest):
            raise ValueError("A V1 job requires a V1 result manifest")
        inputs = {item.frame_id: item for item in manifest.inputs}
        output_frames: set[str] = set()
        for output in result.outputs:
            if output.frame_id not in inputs:
                raise ValueError(f"Plugin returned an unknown frame: {output.frame_id}")
            if output.frame_id in output_frames:
                raise ValueError(f"Plugin returned multiple outputs for frame: {output.frame_id}")
            output_frames.add(output.frame_id)
            if manifest.operation == "frames.vectorize.v1":
                if output.role != "vector" or output.media_type != "application/x-cif":
                    raise ValueError("Vectorization outputs must be CIF vectors")
            elif manifest.operation in {"frames.binary-segment.v1", "frames.binary-segment.v2"}:
                if output.role != "binary-image" or output.media_type != "image/png":
                    raise ValueError("Binary segmentation outputs must be lossless PNG binary images")
            elif manifest.operation == "frames.dataset.prepare.v1":
                if output.role != "dataset" or output.media_type != "application/zip":
                    raise ValueError("Dataset preparation must return a dataset ZIP archive")
            elif manifest.operation == "frames.confidence.v1":
                if output.role != "confidence" or output.media_type != "image/jpeg":
                    raise ValueError("Confidence generation must return grayscale confidence images")
        outcome = PluginJobOutcome(result.outcome)
        if outcome is PluginJobOutcome.SUCCEEDED and manifest.operation == "frames.dataset.prepare.v1":
            if len(result.outputs) != 1:
                raise ValueError("Dataset preparation must return exactly one archive for the selection")
        elif outcome is PluginJobOutcome.SUCCEEDED and manifest.operation == "dataset.model.train.v1":
            if not result.outputs or any(output.role != "model" for output in result.outputs):
                raise ValueError("Training must return model artifacts")
        elif outcome is PluginJobOutcome.SUCCEEDED and output_frames != set(inputs):
            raise ValueError("A succeeded plugin result must contain exactly one output per input frame")
        if outcome in {PluginJobOutcome.FAILED, PluginJobOutcome.CANCELLED} and result.outputs:
            raise ValueError("Failed or cancelled plugin results cannot publish outputs")

    @staticmethod
    def _validate_analysis_result_contract(
        manifest: AnalysisPartitionJobManifest,
        result: AnalysisPartitionResultManifest,
    ) -> None:
        if result.job_id != manifest.job_id or result.run_id != manifest.run_id:
            raise ValueError("Karakal Worker returned a result for another analysis run")
        if result.partition_id != manifest.partition_id or result.project_id != manifest.project_id:
            raise ValueError("Karakal Worker returned a result for another partition")
        from kraken_core.analysis_runtime import validate_worker_runtime
        validate_worker_runtime(manifest, result)
        if result.bundle is not None and result.bundle.frame_count != len(manifest.frames):
            raise ValueError("Analysis result bundle must contain exactly the partition frames")
        requested = next((p.value for p in manifest.parameters if p.key == "_render_frame_id"), None)
        if result.rendered_map is not None and (requested != result.rendered_map.frame_id or len(manifest.frames) != 1
                                                or requested != manifest.frames[0].frame_id):
            raise ValueError("Worker returned an unrequested rendered map")
        if requested is not None and result.outcome.value == "succeeded" and result.rendered_map is None:
            raise ValueError("Worker did not return the requested rendered map")

    def _capture_progress(self, workspace: StagingWorkspace, job_id: str) -> None:
        checkpoint_path = workspace.path / "checkpoint.json"
        try:
            with open_regular_read(checkpoint_path, root=workspace.path) as stream:
                checkpoint = json.loads(stream.read(1024 * 1024 + 1).decode("utf-8"))
            if (isinstance(checkpoint, dict) and checkpoint.get("job_id") == job_id
                    and checkpoint != self.store.get(job_id).checkpoint):
                from kraken_core.job_recovery import execution_state
                execution_state({"checkpoint": checkpoint})
                for item in checkpoint.get("files", []):
                    if workspace.digest(item["relative_path"]) != item["sha256"]:
                        raise ValueError("Incomplete checkpoint file")
                self.store.record_checkpoint(job_id, checkpoint)
        except (OSError, ValueError, TypeError):
            pass  # An atomic snapshot is picked up on the next poll.
        progress_path = workspace.path / "progress.json"
        if not progress_path.is_file():
            return
        try:
            with open_regular_read(progress_path, root=workspace.path) as stream:
                payload = json.loads(stream.read(1024 * 1024).decode("utf-8"))
            if isinstance(payload, dict) and payload.get("job_id") == job_id:
                self.store.record_progress(job_id, payload)
        except OSError, ValueError, json.JSONDecodeError:
            return

    def run_once(self, *, job_id: str | None = None, cancel_event: threading.Event | None = None) -> bool:
        if self._stop.is_set() or (cancel_event is not None and cancel_event.is_set()):
            return False
        job = self.store.claim_next(job_id=job_id)
        if job is None:
            return False
        try:
            manifest = job.manifest
            workspace = StagingWorkspace(self.staging_root, job.job_id)
            workspace.write_manifest(manifest)
            # The plugin decides its own resume granularity; Agent supplies the
            # last durable state and files rather than keeping Python memory alive.
            resume_path = workspace.path / "resume.json"
            with open_regular_append(resume_path, root=workspace.path) as stream:
                stream.seek(0)
                stream.truncate()
                stream.write(json.dumps({"progress": job.progress, "checkpoint": job.checkpoint},
                                        allow_nan=False).encode("utf-8"))
            for item in (job.checkpoint or {}).get("files", []):
                if workspace.digest(item["relative_path"]) != item["sha256"]:
                    raise ValueError("Saved checkpoint is missing or damaged")
            if isinstance(manifest, AnalysisPartitionJobManifest):
                input_artifacts = tuple(artifact for frame in manifest.frames for artifact in frame.artifacts)
                operation = ANALYSIS_OPERATION
            else:
                input_artifacts = manifest.inputs
                operation = manifest.operation
            stream_inputs = (
                isinstance(manifest, PluginJobManifest) and manifest.parameters.get("input_transport") == "server"
            )
            if stream_inputs:
                supported = operation in {
                    "frames.vectorize.v1",
                    "frames.dataset.prepare.v1",
                    "dataset.model.train.v1",
                    "frames.binary-segment.v1",
                    "frames.binary-segment.v2",
                }
                if not supported or not self.environment.get("KRAKEN_ASSET_TOKEN"):
                    raise ValueError("Server input streams require a supported operation and authenticated runner")
                # The stream source verifies each immutable input before exposing
                # bytes to the plugin. No staged image files exist in this mode.
                input_artifacts = ()
            for item in input_artifacts:
                if workspace.digest(item.relative_path) != item.sha256:
                    raise ValueError(f"Missing or damaged staged input: {item.relative_path}")
            spec = self.registry.get(operation)
            if spec is None:
                raise ValueError(f"No plugin registered for {operation}")
            if isinstance(manifest, (PluginJobManifest, PluginJobManifestV2)):
                spec = spec.for_execution(manifest.parameters)
            job = self.store.transition(job.job_id, AgentJobState.RUNNING, expected_revision=job.revision)
            if spec.interactive:
                job = self.store.transition(
                    job.job_id,
                    AgentJobState.WAITING_FOR_USER,
                    expected_revision=job.revision,
                )
            result_path = workspace.path / "result.json"
            environment = self._plugin_environment()
            environment.update(self.environment)
            environment.update(
                {
                    "KRAKEN_JOB_MANIFEST": str(workspace.path / "job.json"),
                    "KRAKEN_RESULT_MANIFEST": str(result_path),
                    "KRAKEN_STAGING_ROOT": str(workspace.path),
                    "KRAKEN_PROGRESS_PATH": str(workspace.path / "progress.json"),
                    "KRAKEN_CHECKPOINT_PATH": str(workspace.path / "checkpoint.json"),
                    "KRAKEN_RESUME_PATH": str(resume_path),
                    "KRAKEN_CANCEL_PATH": str(workspace.path / "cancel.request"),
                }
            )
            log_path = workspace.path / "plugin.log"
            with open_regular_append(log_path, root=workspace.path) as log:
                process = subprocess.Popen(
                    list(spec.command),
                    cwd=None if spec.working_directory is None else str(spec.working_directory),
                    env=environment,
                    stdin=None if spec.interactive else subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                while process.poll() is None:
                    self._capture_progress(workspace, job.job_id)
                    if self._stop.wait(0.25):
                        self._terminate(process)
                        self._capture_progress(workspace, job.job_id)
                        current = self.store.get(job.job_id)
                        if current.state in {AgentJobState.RUNNING, AgentJobState.WAITING_FOR_USER}:
                            self.store.transition(
                                current.job_id,
                                AgentJobState.RECOVERY_REQUIRED,
                                expected_revision=current.revision,
                                error="Agent stopped during plugin execution",
                            )
                        return True
                    current = self.store.get(job.job_id)
                    if current.state is AgentJobState.CANCELLED or (cancel_event is not None and cancel_event.is_set()):
                        self._terminate(process)
                        if current.state in {AgentJobState.RUNNING, AgentJobState.WAITING_FOR_USER}:
                            self.store.transition(
                                current.job_id,
                                AgentJobState.CANCELLED,
                                expected_revision=current.revision,
                                error="Execution lease lost",
                            )
                        return True
            self._capture_progress(workspace, job.job_id)
            allowed_return_codes = {0, 3, 130} if isinstance(manifest, AnalysisPartitionJobManifest) else {0}
            if process.returncode not in allowed_return_codes:
                raise RuntimeError(f"Plugin exited with code {process.returncode}")
            with open_regular_read(result_path, root=workspace.path) as result_stream:
                raw_result = result_stream.read(self._MAX_RESULT_MANIFEST_BYTES + 1)
            if len(raw_result) > self._MAX_RESULT_MANIFEST_BYTES:
                raise ValueError("Plugin result manifest is too large")
            result = parse_result_json(raw_result.decode("utf-8"))
            if result.job_id != job.job_id:
                raise ValueError("Plugin returned a result for another job")
            if isinstance(manifest, AnalysisPartitionJobManifest):
                if not isinstance(result, AnalysisPartitionResultManifest):
                    raise ValueError("Karakal Worker returned a legacy plugin result")
                self._validate_analysis_result_contract(manifest, result)
            else:
                self._validate_result_contract(manifest, result)
            workspace.verify_result(result, manifest)
            outcome_value = result.outcome.value if isinstance(result.outcome, AnalysisOutcome) else result.outcome
            if outcome_value == PluginJobOutcome.SUCCEEDED.value:
                target = AgentJobState.IMPORTING
            elif outcome_value == PluginJobOutcome.PARTIAL.value:
                target = AgentJobState.PARTIAL
            elif outcome_value == PluginJobOutcome.CANCELLED.value:
                target = AgentJobState.CANCELLED
            else:
                target = AgentJobState.FAILED
            current = self.store.get(job.job_id)
            self.store.record_result(
                result,
                callback_key=(
                    f"publication:{result.publication_id}"
                    if isinstance(result, PluginResultPublicationV2)
                    else f"analysis:{result.partition_id}:{result.bundle.sha256 if result.bundle else result.outcome.value}"
                    if isinstance(result, AnalysisPartitionResultManifest)
                    else f"process:{result.completed_at}"
                ),
                expected_revision=current.revision,
                target_state=target,
                allowed_states=(AgentJobState.RUNNING, AgentJobState.WAITING_FOR_USER),
            )
        except JobStateError, KeyError:
            return True
        except Exception as exc:
            current = self.store.get(job.job_id)
            if current.state not in TERMINAL_STATES:
                try:
                    self.store.transition(
                        current.job_id,
                        AgentJobState.FAILED,
                        expected_revision=current.revision,
                        error=str(exc)[:10_000],
                    )
                except JobStateError:
                    pass
        return True

    def run_forever(self, *, idle_seconds: float = 0.5) -> None:
        while not self._stop.is_set():
            if not self.run_once():
                self._stop.wait(idle_seconds)


__all__ = ["PluginProcessSpec", "PluginRegistry", "SubprocessPluginRunner"]
