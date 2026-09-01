"""Crash-aware subprocess execution against isolated staging workspaces."""

from __future__ import annotations

import json
import os
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from kraken_core.plugin_protocol import PluginJobManifest, PluginJobOutcome, PluginResultManifest
from kraken_core.analysis_protocol import AnalysisOutcome
from kraken_core.analysis_run_protocol import AnalysisPartitionJobManifest, AnalysisPartitionResultManifest
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

    def __post_init__(self) -> None:
        if not self.operation or not self.command:
            raise ValueError("Plugin process operation and command are required")


class PluginRegistry:
    def __init__(self, specs: Mapping[str, PluginProcessSpec] | None = None) -> None:
        self._specs = dict(specs or {})

    def get(self, operation: str) -> PluginProcessSpec | None:
        return self._specs.get(operation)

    @classmethod
    def from_json(cls, path: Path | str) -> "PluginRegistry":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
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
            "LANG",
            "LC_ALL",
            "PYTHONPATH",
            "PYTHONHOME",
            "VIRTUAL_ENV",
            "DISPLAY",
            "WAYLAND_DISPLAY",
            "XAUTHORITY",
            "XDG_RUNTIME_DIR",
            "QT_PLUGIN_PATH",
            "QT_QPA_PLATFORM",
        }
    )
    _MAX_RESULT_MANIFEST_BYTES = 16 * 1024 * 1024

    def __init__(self, store: DurableJobStore, staging_root: Path | str, registry: PluginRegistry) -> None:
        self.store = store
        self.staging_root = Path(staging_root).resolve()
        self.staging_root.mkdir(parents=True, exist_ok=True)
        self.registry = registry
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
    def _validate_result_contract(manifest: PluginJobManifest, result: PluginResultManifest) -> None:
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
            elif manifest.operation == "frames.binary-segment.v1":
                if output.role != "binary-image" or output.media_type != "image/png":
                    raise ValueError("Binary segmentation outputs must be lossless PNG binary images")
        outcome = PluginJobOutcome(result.outcome)
        if outcome is PluginJobOutcome.SUCCEEDED and output_frames != set(inputs):
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
        if result.bundle is not None and result.bundle.frame_count != len(manifest.frames):
            raise ValueError("Analysis result bundle must contain exactly the partition frames")

    def _capture_progress(self, workspace: StagingWorkspace, job_id: str) -> None:
        progress_path = workspace.path / "progress.json"
        if not progress_path.is_file():
            return
        try:
            with open_regular_read(progress_path, root=workspace.path) as stream:
                payload = json.loads(stream.read(1024 * 1024).decode("utf-8"))
            if isinstance(payload, dict) and payload.get("job_id") == job_id:
                self.store.record_progress(job_id, payload)
        except (OSError, ValueError, json.JSONDecodeError):
            return

    def run_once(self) -> bool:
        jobs = self.store.list(states=(AgentJobState.QUEUED,), limit=1)
        if not jobs:
            return False
        job = jobs[0]
        try:
            job = self.store.transition(job.job_id, AgentJobState.STAGING, expected_revision=job.revision)
            manifest = job.manifest
            workspace = StagingWorkspace(self.staging_root, job.job_id)
            workspace.write_manifest(manifest)
            if isinstance(manifest, AnalysisPartitionJobManifest):
                input_artifacts = tuple(artifact for frame in manifest.frames for artifact in frame.artifacts)
                operation = ANALYSIS_OPERATION
            else:
                input_artifacts = manifest.inputs
                operation = manifest.operation
            for item in input_artifacts:
                if workspace.digest(item.relative_path) != item.sha256:
                    raise ValueError(f"Missing or damaged staged input: {item.relative_path}")
            spec = self.registry.get(operation)
            if spec is None:
                raise ValueError(f"No plugin registered for {operation}")
            job = self.store.transition(job.job_id, AgentJobState.RUNNING, expected_revision=job.revision)
            result_path = workspace.path / "result.json"
            environment = self._plugin_environment()
            environment.update(
                {
                    "KRAKEN_JOB_MANIFEST": str(workspace.path / "job.json"),
                    "KRAKEN_RESULT_MANIFEST": str(result_path),
                    "KRAKEN_STAGING_ROOT": str(workspace.path),
                    "KRAKEN_PROGRESS_PATH": str(workspace.path / "progress.json"),
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
                        return True
                    current = self.store.get(job.job_id)
                    if current.state is AgentJobState.CANCELLED:
                        self._terminate(process)
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
                if not isinstance(result, PluginResultManifest):
                    raise ValueError("Legacy plugin returned an analysis result")
                self._validate_result_contract(manifest, result)
            workspace.verify_result(result, manifest)
            current = self.store.get(job.job_id)
            current, _ = self.store.record_result(
                result,
                callback_key=(
                    f"analysis:{result.partition_id}:{result.bundle.sha256 if result.bundle else result.outcome.value}"
                    if isinstance(result, AnalysisPartitionResultManifest)
                    else f"process:{result.completed_at}"
                ),
                expected_revision=current.revision,
            )
            outcome_value = result.outcome.value if isinstance(result.outcome, AnalysisOutcome) else result.outcome
            if outcome_value == PluginJobOutcome.SUCCEEDED.value:
                target = AgentJobState.IMPORTING
            elif outcome_value == PluginJobOutcome.PARTIAL.value:
                target = AgentJobState.PARTIAL
            elif outcome_value == PluginJobOutcome.CANCELLED.value:
                target = AgentJobState.CANCELLED
            else:
                target = AgentJobState.FAILED
            self.store.transition(current.job_id, target, expected_revision=current.revision)
        except (JobStateError, KeyError):
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
