from __future__ import annotations

import gzip
import hashlib
import sys
import threading
from pathlib import Path

import cv2
import numpy as np
import pytest

from kraken_agent.runner import ANALYSIS_OPERATION, PluginProcessSpec, PluginRegistry, SubprocessPluginRunner
from kraken_agent.service import AgentControlServer
from kraken_core.analysis_protocol import (
    AnalysisArtifactInput,
    AnalysisFrameInput,
    AnalysisFrameResult,
    AnalysisMetricValue,
    AnalysisOutcome,
    AnalysisSourceRole,
)
from kraken_core.analysis_run_protocol import (
    AnalysisExpression,
    AnalysisPartitionJobManifest,
    AnalysisPartitionResultManifest,
    AnalysisRecipe,
    AnalysisRecordBundle,
    AnalysisRuntimeIdentity,
    AnalysisSourceBinding,
    canonical_json,
)
from kraken_manager.application.analysis_runs import AnalysisGatewayJob, AnalysisRunCoordinator
from kraken_manager.infrastructure.analysis import FilesystemAnalysisStore
from kraken_manager.infrastructure.plugin import AgentAnalysisGateway


class FakeAnalysisGateway:
    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root
        self.jobs: dict[str, AnalysisGatewayJob] = {}
        self.manifests: dict[str, AnalysisPartitionJobManifest] = {}
        self.results: dict[str, tuple[AnalysisPartitionResultManifest, Path]] = {}
        self.reject_submissions = False

    def submit(self, manifest, source_paths):
        if self.reject_submissions:
            raise RuntimeError("agent unavailable")
        assert {artifact.artifact_version_id for frame in manifest.frames for artifact in frame.artifacts} <= set(
            source_paths
        )
        job = AnalysisGatewayJob(manifest.job_id, "importing", 2)
        self.jobs[manifest.job_id] = job
        self.manifests[manifest.job_id] = manifest
        frames = tuple(
            AnalysisFrameResult(
                frame.frame_id,
                frame.x,
                frame.y,
                "ready",
                (AnalysisMetricValue("iou", 1.0, 1.0),),
            )
            for frame in manifest.frames
        )
        bundle_path = self.output_root / f"{manifest.job_id}.jsonl.gz"
        uncompressed_size = 0
        with bundle_path.open("wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as archive:
                for frame in frames:
                    encoded = (canonical_json(frame.to_payload()) + "\n").encode()
                    archive.write(encoded)
                    uncompressed_size += len(encoded)
        bundle = AnalysisRecordBundle(
            f"outputs/{bundle_path.name}",
            hashlib.sha256(bundle_path.read_bytes()).hexdigest(),
            bundle_path.stat().st_size,
            uncompressed_size,
            len(frames),
        )
        result = AnalysisPartitionResultManifest(
            manifest.job_id,
            manifest.run_id,
            manifest.partition_id,
            manifest.project_id,
            AnalysisOutcome.SUCCEEDED,
            bundle,
        )
        self.results[manifest.job_id] = (result, bundle_path)
        return job

    def get(self, job_id):
        return self.jobs[job_id]

    def result(self, manifest):
        return self.results[manifest.job_id]

    def confirm_partial(self, job):
        return AnalysisGatewayJob(job.job_id, "importing", job.revision + 1)

    def complete_import(self, job):
        completed = AnalysisGatewayJob(job.job_id, "succeeded", job.revision + 1)
        self.jobs[job.job_id] = completed
        return completed

    def cancel(self, job):
        cancelled = AnalysisGatewayJob(job.job_id, "cancelled", job.revision + 1)
        self.jobs[job.job_id] = cancelled
        return cancelled


def _input_frames(source: Path, count: int) -> tuple[AnalysisFrameInput, ...]:
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    return tuple(
        AnalysisFrameInput(
            f"frame-{index:04d}",
            index + 1,
            1,
            tuple(
                AnalysisArtifactInput(
                    key,
                    AnalysisSourceRole.MODEL_OUTPUT,
                    f"series-{key}",
                    f"version-{key}",
                    f"inputs/{key}/frame-{index:04d}.png",
                    "image/png",
                    digest,
                )
                for key in ("A", "B")
            ),
        )
        for index in range(count)
    )


def _coordinator(tmp_path: Path, *, count: int = 1001):
    source = tmp_path / "source.png"
    source.write_bytes(b"immutable")
    store = FilesystemAnalysisStore(tmp_path / "catalog", "project-1")
    gateway = FakeAnalysisGateway(tmp_path)
    coordinator = AnalysisRunCoordinator(store, gateway, lambda _version: source)
    recipe = AnalysisRecipe(
        AnalysisExpression.binary("compare", AnalysisExpression.source("A"), AnalysisExpression.source("B"))
    )
    run = coordinator.start(
        project_id="project-1",
        frames=_input_frames(source, count),
        source_bindings=(
            AnalysisSourceBinding("A", "model-a", "v1"),
            AnalysisSourceBinding("B", "model-b", "v1"),
        ),
        recipe=recipe,
        runtime=AnalysisRuntimeIdentity("1", "build", "3.14", "2", "5", "windows"),
    )
    return coordinator, store, gateway, run


def test_coordinator_dispatches_and_progressively_imports_all_partitions(tmp_path) -> None:
    coordinator, store, gateway, run = _coordinator(tmp_path)

    assert len(gateway.jobs) == 2
    assert store.get_run(run.run_id).state == "running"
    refreshed = coordinator.refresh(run.run_id)

    assert refreshed.state == "completed"
    assert refreshed.completed_frames == 1001
    assert len(store.frame_results(run.run_id, "iou")) == 1001
    assert {job.state for job in gateway.jobs.values()} == {"succeeded"}


def test_repeat_has_new_run_id_but_same_fingerprint(tmp_path) -> None:
    coordinator, store, _gateway, run = _coordinator(tmp_path, count=2)
    coordinator.refresh(run.run_id)

    repeated = coordinator.repeat(run.run_id)

    assert repeated.run_id != run.run_id
    assert repeated.fingerprint == run.fingerprint
    assert len(store.list_runs()) == 2


def test_failed_submission_is_reissued_with_a_new_agent_job_id(tmp_path) -> None:
    coordinator, store, gateway, completed_run = _coordinator(tmp_path, count=1)
    coordinator.refresh(completed_run.run_id)
    source = tmp_path / "source.png"
    frames = _input_frames(source, 1)
    recipe = completed_run.recipe
    gateway.reject_submissions = True
    with pytest.raises(RuntimeError, match="agent unavailable"):
        coordinator.start(
            project_id="project-1",
            frames=frames,
            source_bindings=completed_run.source_bindings,
            recipe=recipe,
            runtime=completed_run.runtime,
        )
    failed = next(item for item in store.list_runs() if item.state == "failed")
    old_job_id = store.failed_partitions(failed.run_id)[0].job_id
    gateway.reject_submissions = False

    coordinator.retry_failed(failed.run_id)
    replacement = store.retryable_partitions(failed.run_id)[0]

    assert replacement.job_id != old_job_id
    assert replacement.job_id in gateway.jobs


def test_local_agent_worker_gateway_and_manager_import_smoke(tmp_path) -> None:
    source = tmp_path / "mask.png"
    assert cv2.imwrite(str(source), np.asarray([[0, 255]], dtype=np.uint8))
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    artifacts = tuple(
        AnalysisArtifactInput(
            key,
            AnalysisSourceRole.MODEL_OUTPUT,
            f"series-{key}",
            f"version-{key}",
            f"inputs/{key}/frame.png",
            "image/png",
            digest,
        )
        for key in ("A", "B")
    )
    control = AgentControlServer.create(tmp_path / "agent.sqlite3", token="test-token")
    httpd = control.build_http_server()
    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()
    try:
        gateway = AgentAnalysisGateway(
            base_url=f"http://127.0.0.1:{control.port}",
            token="test-token",
            staging_root=tmp_path / "staging",
        )
        store = FilesystemAnalysisStore(tmp_path / "catalog", "project-1")
        coordinator = AnalysisRunCoordinator(store, gateway, lambda _version: source)
        run = coordinator.start(
            project_id="project-1",
            frames=(AnalysisFrameInput("frame-1", 1, 1, artifacts),),
            source_bindings=(
                AnalysisSourceBinding("A", "model-a", "v1"),
                AnalysisSourceBinding("B", "model-b", "v1"),
            ),
            recipe=AnalysisRecipe(
                AnalysisExpression.binary(
                    "compare", AnalysisExpression.source("A"), AnalysisExpression.source("B")
                )
            ),
            runtime=AnalysisRuntimeIdentity("1", "build", "3.14", "2", "5", "windows"),
        )
        runner = SubprocessPluginRunner(
            control.store,
            tmp_path / "staging",
            PluginRegistry(
                {
                    ANALYSIS_OPERATION: PluginProcessSpec(
                        ANALYSIS_OPERATION,
                        (sys.executable, "-m", "karakal.worker"),
                    )
                }
            ),
        )

        assert runner.run_once() is True
        completed = coordinator.refresh(run.run_id)

        assert completed.state == "completed"
        assert completed.completed_frames == 1
        assert len(store.frame_results(run.run_id, "iou")) == 1
    finally:
        httpd.shutdown()
        httpd.server_close()
        server_thread.join(timeout=5)
