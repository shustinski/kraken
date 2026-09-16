"""Application workflow for partitioned Karakal analysis runs."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Protocol
from uuid import uuid4

from kraken_core.analysis_protocol import AnalysisFrameInput, AnalysisParameter
from kraken_core.analysis_run_protocol import (
    AnalysisPartitionJobManifest,
    AnalysisPartitionResultManifest,
    AnalysisRecipe,
    AnalysisRunManifest,
    AnalysisRuntimeIdentity,
    AnalysisSourceBinding,
)


@dataclass(frozen=True, slots=True)
class AnalysisGatewayJob:
    job_id: str
    state: str
    revision: int
    progress: Mapping[str, object] | None = None
    error: str = ""


class AnalysisPartitionGateway(Protocol):
    def submit(
        self,
        manifest: AnalysisPartitionJobManifest,
        source_paths: Mapping[str, os.PathLike[str]],
    ) -> AnalysisGatewayJob: ...

    def get(self, job_id: str) -> AnalysisGatewayJob: ...

    def result(
        self, manifest: AnalysisPartitionJobManifest
    ) -> tuple[AnalysisPartitionResultManifest, os.PathLike[str]]: ...

    def confirm_partial(self, job: AnalysisGatewayJob) -> AnalysisGatewayJob: ...

    def complete_import(self, job: AnalysisGatewayJob) -> AnalysisGatewayJob: ...

    def cancel(self, job: AnalysisGatewayJob) -> AnalysisGatewayJob: ...


class AnalysisRunStore(Protocol):
    def create_run(
        self,
        manifest: AnalysisRunManifest,
        partitions: tuple[AnalysisPartitionJobManifest, ...],
    ) -> object: ...

    def get_run(self, run_id: str) -> object | None: ...

    def list_runs(self) -> tuple[object, ...]: ...

    def retryable_partitions(self, run_id: str) -> tuple[AnalysisPartitionJobManifest, ...]: ...

    def all_partitions(self, run_id: str) -> tuple[AnalysisPartitionJobManifest, ...]: ...

    def failed_partitions(self, run_id: str) -> tuple[AnalysisPartitionJobManifest, ...]: ...

    def requeue_partition(self, manifest: AnalysisPartitionJobManifest) -> None: ...

    def import_partition(
        self, result: AnalysisPartitionResultManifest, bundle_path: str | os.PathLike[str]
    ) -> bool: ...

    def mark_partition_state(self, partition_id: str, state: str, *, error: str = "") -> None: ...


class AnalysisRunCoordinator:
    """Create reproducible runs, dispatch partitions and import completed bundles."""

    def __init__(
        self,
        store: AnalysisRunStore,
        gateway: AnalysisPartitionGateway,
        source_path_for_version: Callable[[str], os.PathLike[str]],
    ) -> None:
        self.store = store
        self.gateway = gateway
        self.source_path_for_version = source_path_for_version

    @staticmethod
    def build_contracts(
        *,
        project_id: str,
        frames: tuple[AnalysisFrameInput, ...],
        source_bindings: tuple[AnalysisSourceBinding, ...],
        recipe: AnalysisRecipe,
        runtime: AnalysisRuntimeIdentity,
        parameters: tuple[AnalysisParameter, ...] = (),
        run_id: str | None = None,
    ) -> tuple[AnalysisRunManifest, tuple[AnalysisPartitionJobManifest, ...]]:
        if not frames:
            raise ValueError("Analysis selection is empty")
        ordered = tuple(sorted(frames, key=lambda item: (item.y, item.x, item.frame_id)))
        identifier = run_id or str(uuid4())
        run = AnalysisRunManifest(
            run_id=identifier,
            project_id=project_id,
            frame_ids=tuple(frame.frame_id for frame in ordered),
            source_bindings=source_bindings,
            recipe=recipe,
            runtime=runtime,
            parameters=parameters,
        )
        chunks = tuple(
            ordered[offset : offset + 1000]
            for offset in range(0, len(ordered), 1000)
        )
        partitions = tuple(
            AnalysisPartitionJobManifest(
                job_id=str(uuid4()),
                run_id=run.run_id,
                partition_id=str(uuid4()),
                project_id=run.project_id,
                partition_index=index,
                partition_count=len(chunks),
                run_fingerprint=run.fingerprint,
                recipe=recipe,
                frames=chunk,
                parameters=parameters,
            )
            for index, chunk in enumerate(chunks)
        )
        return run, partitions

    def _source_paths(self, partition: AnalysisPartitionJobManifest) -> dict[str, os.PathLike[str]]:
        version_ids = {
            artifact.artifact_version_id
            for frame in partition.frames
            for artifact in frame.artifacts
        }
        return {identifier: self.source_path_for_version(identifier) for identifier in version_ids}

    def start(
        self,
        *,
        project_id: str,
        frames: tuple[AnalysisFrameInput, ...],
        source_bindings: tuple[AnalysisSourceBinding, ...],
        recipe: AnalysisRecipe,
        runtime: AnalysisRuntimeIdentity,
        parameters: tuple[AnalysisParameter, ...] = (),
    ) -> AnalysisRunManifest:
        run, partitions = self.build_contracts(
            project_id=project_id,
            frames=frames,
            source_bindings=source_bindings,
            recipe=recipe,
            runtime=runtime,
            parameters=parameters,
        )
        self.store.create_run(run, partitions)
        for partition in partitions:
            try:
                self.gateway.submit(partition, self._source_paths(partition))
                self.store.mark_partition_state(partition.partition_id, "running")
            except Exception as exc:
                self.store.mark_partition_state(partition.partition_id, "failed", error=str(exc))
                raise
        return run

    def refresh(self, run_id: str) -> object | None:
        for partition in self.store.retryable_partitions(run_id):
            try:
                job = self.gateway.get(partition.job_id)
                if job.state in {"queued", "staging", "running"}:
                    self.store.mark_partition_state(partition.partition_id, "running")
                    continue
                if job.state == "partial":
                    job = self.gateway.confirm_partial(job)
                if job.state == "importing":
                    result, bundle_path = self.gateway.result(partition)
                    self.store.import_partition(result, bundle_path)
                    self.gateway.complete_import(job)
                elif job.state in {"failed", "cancelled", "recovery_required"}:
                    self.store.mark_partition_state(
                        partition.partition_id,
                        "cancelled" if job.state == "cancelled" else "failed",
                        error=job.error,
                    )
            except Exception as exc:
                self.store.mark_partition_state(partition.partition_id, "failed", error=str(exc))
        return self.store.get_run(run_id)

    def retry_failed(self, run_id: str) -> None:
        for partition in self.store.failed_partitions(run_id):
            replacement = replace(partition, job_id=str(uuid4()))
            self.store.requeue_partition(replacement)
            self.gateway.submit(replacement, self._source_paths(replacement))
            self.store.mark_partition_state(replacement.partition_id, "running")

    def cancel(self, run_id: str) -> None:
        for partition in self.store.retryable_partitions(run_id):
            job = self.gateway.get(partition.job_id)
            if job.state not in {"failed", "cancelled", "succeeded"}:
                self.gateway.cancel(job)
                self.store.mark_partition_state(partition.partition_id, "cancelled")

    def repeat(self, run_id: str) -> AnalysisRunManifest:
        previous = self.store.get_run(run_id)
        if previous is None:
            raise KeyError(f"Unknown analysis run: {run_id}")
        manifest = previous.manifest
        frames = tuple(frame for partition in self.store.all_partitions(run_id) for frame in partition.frames)
        return self.start(
            project_id=manifest.project_id,
            frames=frames,
            source_bindings=manifest.source_bindings,
            recipe=manifest.recipe,
            runtime=manifest.runtime,
            parameters=manifest.parameters,
        )


__all__ = [
    "AnalysisGatewayJob",
    "AnalysisPartitionGateway",
    "AnalysisRunCoordinator",
    "AnalysisRunStore",
]
