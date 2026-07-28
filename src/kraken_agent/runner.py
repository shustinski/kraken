"""Crash-aware subprocess execution against isolated staging workspaces."""

from __future__ import annotations

import json
import os
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from kraken_core.plugin_protocol import (
    PluginAssetScope,
    PluginJobManifest,
    PluginJobManifestV2,
    PluginJobOutcome,
    PluginResultManifest,
    PluginResultPublicationV2,
    parse_plugin_result_json,
)
from kraken_core.safe_files import open_regular_append, open_regular_read

from .jobs import AgentJobState, DurableJobStore, JobStateError, StagingWorkspace, TERMINAL_STATES


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
                if (
                    output.scope is PluginAssetScope.FRAME
                    and output.frame_id not in input_frames
                ):
                    raise ValueError(f"Plugin returned an unknown frame: {output.frame_id}")
            if not result.final:
                raise ValueError("A terminating V2 process must mark its publication final")
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
            elif manifest.operation == "frames.binary-segment.v1":
                if output.role != "binary-image" or output.media_type != "image/png":
                    raise ValueError("Binary segmentation outputs must be lossless PNG binary images")
        outcome = PluginJobOutcome(result.outcome)
        if outcome is PluginJobOutcome.SUCCEEDED and output_frames != set(inputs):
            raise ValueError("A succeeded plugin result must contain exactly one output per input frame")
        if outcome in {PluginJobOutcome.FAILED, PluginJobOutcome.CANCELLED} and result.outputs:
            raise ValueError("Failed or cancelled plugin results cannot publish outputs")

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
            for item in manifest.inputs:
                if workspace.digest(item.relative_path) != item.sha256:
                    raise ValueError(f"Missing or damaged staged input: {item.relative_path}")
            spec = self.registry.get(manifest.operation)
            if spec is None:
                raise ValueError(f"No plugin registered for {manifest.operation}")
            job = self.store.transition(job.job_id, AgentJobState.RUNNING, expected_revision=job.revision)
            result_path = workspace.path / "result.json"
            environment = self._plugin_environment()
            environment.update(
                {
                    "KRAKEN_JOB_MANIFEST": str(workspace.path / "job.json"),
                    "KRAKEN_RESULT_MANIFEST": str(result_path),
                    "KRAKEN_STAGING_ROOT": str(workspace.path),
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
                    if self._stop.wait(0.25):
                        self._terminate(process)
                        return True
                    current = self.store.get(job.job_id)
                    if current.state is AgentJobState.CANCELLED:
                        self._terminate(process)
                        return True
            if process.returncode != 0:
                raise RuntimeError(f"Plugin exited with code {process.returncode}")
            with open_regular_read(result_path, root=workspace.path) as result_stream:
                raw_result = result_stream.read(self._MAX_RESULT_MANIFEST_BYTES + 1)
            if len(raw_result) > self._MAX_RESULT_MANIFEST_BYTES:
                raise ValueError("Plugin result manifest is too large")
            result = parse_plugin_result_json(raw_result.decode("utf-8"))
            if result.job_id != job.job_id:
                raise ValueError("Plugin returned a result for another job")
            self._validate_result_contract(manifest, result)
            workspace.verify_result(result)
            current = self.store.get(job.job_id)
            callback_key = (
                f"publication:{result.publication_id}"
                if isinstance(result, PluginResultPublicationV2)
                else f"process:{result.completed_at}"
            )
            current, _ = self.store.record_result(
                result,
                callback_key=callback_key,
                expected_revision=current.revision,
            )
            if isinstance(result, PluginResultPublicationV2):
                target = {
                    "succeeded": AgentJobState.IMPORTING,
                    "partial": AgentJobState.PARTIAL,
                    "failed": AgentJobState.FAILED,
                    "cancelled": AgentJobState.CANCELLED,
                }[result.outcome]
            elif result.outcome == PluginJobOutcome.SUCCEEDED.value:
                target = AgentJobState.IMPORTING
            elif result.outcome == PluginJobOutcome.PARTIAL.value:
                target = AgentJobState.PARTIAL
            elif result.outcome == PluginJobOutcome.CANCELLED.value:
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
