"""Standalone orchestration over the shared headless engine and private SQLite."""

from __future__ import annotations

from pathlib import Path

from kraken_core.analysis_protocol import AnalysisOutcome
from kraken_core.analysis_run_protocol import AnalysisPartitionJobManifest, AnalysisRunManifest

from .core.headless import CancellationCheck, ProgressCallback, run_analysis
from .storage import AnalysisHistoryStore


class StandaloneAnalysisService:
    def __init__(self, history: AnalysisHistoryStore | None = None) -> None:
        self.history = history or AnalysisHistoryStore()

    def start(
        self,
        run: AnalysisRunManifest,
        partitions: tuple[AnalysisPartitionJobManifest, ...],
        *,
        workspace: str | Path,
        output_dir: str | Path,
        progress: ProgressCallback | None = None,
        cancellation: CancellationCheck | None = None,
    ):
        existing = self.history.get_run(run.run_id)
        if existing is None:
            self.history.create_run(run)
        elif existing.fingerprint != run.fingerprint:
            raise ValueError("Standalone analysis run id belongs to another manifest")
        for partition in partitions:
            self.history.save_partition(partition)
        return self.resume(
            run.run_id,
            workspace=workspace,
            output_dir=output_dir,
            progress=progress,
            cancellation=cancellation,
        )

    def resume(
        self,
        run_id: str,
        *,
        workspace: str | Path,
        output_dir: str | Path,
        progress: ProgressCallback | None = None,
        cancellation: CancellationCheck | None = None,
    ):
        for stored in self.history.incomplete_partitions(run_id):
            if cancellation is not None and cancellation():
                break
            self.history.mark_partition_running(stored.partition_id)
            result = run_analysis(stored.manifest, workspace, output_dir, progress, cancellation)
            if result.outcome == AnalysisOutcome.CANCELLED:
                break
            self.history.import_partition(stored.partition_id, result.frames)
        return self.history.get_run(run_id)


__all__ = ["StandaloneAnalysisService"]
