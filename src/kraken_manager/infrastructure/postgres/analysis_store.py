"""PostgreSQL projection adapter for partitioned analysis results."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any, Iterator

from kraken_core.analysis_protocol import AnalysisFrameResult
from kraken_core.analysis_run_protocol import (
    AnalysisPartitionJobManifest,
    AnalysisPartitionResultManifest,
    AnalysisRunManifest,
)

from .event_store import _sqlalchemy


def _analysis_tables() -> tuple[Any, ...]:
    sa, _ = _sqlalchemy()
    metadata = sa.MetaData()
    runs = sa.Table(
        "analysis_runs",
        metadata,
        sa.Column("run_id", sa.Text, primary_key=True),
        sa.Column("project_id", sa.Uuid(as_uuid=False), nullable=False),
        sa.Column("fingerprint", sa.String(64), nullable=False),
        sa.Column("manifest", sa.JSON, nullable=False),
        sa.Column("state", sa.Text, nullable=False),
        sa.Column("total_frames", sa.BigInteger, nullable=False),
        sa.Column("completed_frames", sa.BigInteger, nullable=False, server_default="0"),
        sa.Column("failed_frames", sa.BigInteger, nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    sources = sa.Table(
        "analysis_sources",
        metadata,
        sa.Column("run_id", sa.Text, sa.ForeignKey("analysis_runs.run_id", ondelete="CASCADE"), primary_key=True),
        sa.Column("binding_key", sa.Text, primary_key=True),
        sa.Column("source_id", sa.Text, nullable=False),
        sa.Column("source_version", sa.Text, nullable=False),
        sa.Column("display_name", sa.Text, nullable=False),
    )
    partitions = sa.Table(
        "analysis_partitions",
        metadata,
        sa.Column("partition_id", sa.Text, primary_key=True),
        sa.Column("run_id", sa.Text, sa.ForeignKey("analysis_runs.run_id", ondelete="CASCADE"), nullable=False),
        sa.Column("partition_index", sa.Integer, nullable=False),
        sa.Column("job_id", sa.Text, nullable=False, unique=True),
        sa.Column("manifest", sa.JSON, nullable=False),
        sa.Column("state", sa.Text, nullable=False),
        sa.Column("attempt", sa.Integer, nullable=False, server_default="0"),
        sa.Column("completed_frames", sa.Integer, nullable=False, server_default="0"),
        sa.Column("failed_frames", sa.Integer, nullable=False, server_default="0"),
        sa.Column("bundle_sha256", sa.String(64)),
        sa.Column("result", sa.JSON),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("run_id", "partition_index"),
    )
    frames = sa.Table(
        "analysis_frame_results",
        metadata,
        sa.Column("run_id", sa.Text, sa.ForeignKey("analysis_runs.run_id", ondelete="CASCADE"), primary_key=True),
        sa.Column("frame_id", sa.Text, primary_key=True),
        sa.Column(
            "partition_id", sa.Text, sa.ForeignKey("analysis_partitions.partition_id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("x", sa.Integer, nullable=False),
        sa.Column("y", sa.Integer, nullable=False),
        sa.Column("status", sa.Text, nullable=False),
        sa.Column("message", sa.Text, nullable=False),
    )
    metrics = sa.Table(
        "analysis_metric_values",
        metadata,
        sa.Column("run_id", sa.Text, primary_key=True),
        sa.Column("frame_id", sa.Text, primary_key=True),
        sa.Column("metric_key", sa.Text, primary_key=True),
        sa.Column("raw_value", sa.Float, nullable=False),
        sa.Column("goodness", sa.Float, nullable=False),
        sa.Column("percentile", sa.Float),
        sa.Column("unit", sa.Text, nullable=False),
        sa.Column("higher_is_better", sa.Boolean, nullable=False),
        sa.ForeignKeyConstraint(
            ["run_id", "frame_id"], ["analysis_frame_results.run_id", "analysis_frame_results.frame_id"], ondelete="CASCADE"
        ),
    )
    scales = sa.Table(
        "analysis_metric_scales",
        metadata,
        sa.Column("run_id", sa.Text, sa.ForeignKey("analysis_runs.run_id", ondelete="CASCADE"), primary_key=True),
        sa.Column("metric_key", sa.Text, primary_key=True),
        sa.Column("mode", sa.Text, nullable=False),
        sa.Column("low", sa.Float, nullable=False),
        sa.Column("high", sa.Float, nullable=False),
        sa.Column("p05", sa.Float),
        sa.Column("p50", sa.Float),
        sa.Column("p95", sa.Float),
        sa.Column("clipped_low", sa.BigInteger, nullable=False),
        sa.Column("clipped_high", sa.BigInteger, nullable=False),
    )
    artifacts = sa.Table(
        "analysis_artifacts",
        metadata,
        sa.Column("artifact_key", sa.String(64), primary_key=True),
        sa.Column("run_id", sa.Text, sa.ForeignKey("analysis_runs.run_id", ondelete="CASCADE"), nullable=False),
        sa.Column("frame_id", sa.Text, nullable=False),
        sa.Column("kind", sa.Text, nullable=False),
        sa.Column("recipe_fingerprint", sa.String(64), nullable=False),
        sa.Column("blob_sha256", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    return metadata, runs, sources, partitions, frames, metrics, scales, artifacts


class PostgresAnalysisProjectionStore:
    def __init__(self, engine: Any, *, connection: Any | None = None, create_schema_for_tests: bool = False) -> None:
        self.engine = engine
        self.connection = connection
        (
            self.metadata,
            self.runs,
            self.sources,
            self.partitions,
            self.frames,
            self.metrics,
            self.scales,
            self.artifacts,
        ) = _analysis_tables()
        if create_schema_for_tests:
            self.metadata.create_all(engine)

    @contextmanager
    def _scope(self, *, write: bool = False) -> Iterator[Any]:
        if self.connection is not None:
            yield self.connection
            return
        with (self.engine.begin() if write else self.engine.connect()) as connection:
            yield connection

    def create_run(
        self,
        manifest: AnalysisRunManifest,
        partitions: tuple[AnalysisPartitionJobManifest, ...],
    ) -> None:
        sa, _ = _sqlalchemy()
        now = datetime.now(UTC)
        with self._scope(write=True) as connection:
            existing = connection.execute(
                sa.select(self.runs.c.fingerprint).where(self.runs.c.run_id == manifest.run_id)
            ).scalar_one_or_none()
            if existing is not None:
                if existing != manifest.fingerprint:
                    raise ValueError("Analysis run id already belongs to another manifest")
                return
            connection.execute(
                sa.insert(self.runs),
                {
                    "run_id": manifest.run_id,
                    "project_id": manifest.project_id,
                    "fingerprint": manifest.fingerprint,
                    "manifest": manifest.to_payload(),
                    "state": "queued",
                    "total_frames": len(manifest.frame_ids),
                    "completed_frames": 0,
                    "failed_frames": 0,
                    "created_at": now,
                    "updated_at": now,
                },
            )
            connection.execute(
                sa.insert(self.sources),
                [
                    {
                        "run_id": manifest.run_id,
                        "binding_key": source.binding_key,
                        "source_id": source.source_id,
                        "source_version": source.source_version,
                        "display_name": source.display_name,
                    }
                    for source in manifest.source_bindings
                ],
            )
            connection.execute(
                sa.insert(self.partitions),
                [
                    {
                        "partition_id": partition.partition_id,
                        "run_id": partition.run_id,
                        "partition_index": partition.partition_index,
                        "job_id": partition.job_id,
                        "manifest": partition.to_payload(),
                        "state": "queued",
                        "attempt": 0,
                        "completed_frames": 0,
                        "failed_frames": 0,
                        "updated_at": now,
                    }
                    for partition in partitions
                ],
            )

    def import_partition(
        self,
        result: AnalysisPartitionResultManifest,
        frames: tuple[AnalysisFrameResult, ...],
    ) -> bool:
        if result.bundle is None:
            raise ValueError("Analysis partition import requires a bundle")
        sa, _ = _sqlalchemy()
        now = datetime.now(UTC)
        with self._scope(write=True) as connection:
            partition = connection.execute(
                sa.select(self.partitions).where(self.partitions.c.partition_id == result.partition_id)
            ).mappings().one_or_none()
            if partition is None:
                raise KeyError(result.partition_id)
            if partition["state"] == "imported":
                if partition["bundle_sha256"] != result.bundle.sha256:
                    raise ValueError("Imported partition was repeated with different content")
                return False
            for frame in frames:
                connection.execute(
                    sa.insert(self.frames),
                    {
                        "run_id": result.run_id,
                        "frame_id": frame.frame_id,
                        "partition_id": result.partition_id,
                        "x": frame.x,
                        "y": frame.y,
                        "status": frame.status,
                        "message": frame.message,
                    },
                )
                if frame.metrics:
                    connection.execute(
                        sa.insert(self.metrics),
                        [
                            {
                                "run_id": result.run_id,
                                "frame_id": frame.frame_id,
                                "metric_key": metric.key,
                                "raw_value": metric.raw_value,
                                "goodness": metric.goodness,
                                "percentile": metric.percentile,
                                "unit": metric.unit,
                                "higher_is_better": metric.higher_is_better,
                            }
                            for metric in frame.metrics
                        ],
                    )
            completed = sum(frame.status == "ready" for frame in frames)
            connection.execute(
                sa.update(self.partitions)
                .where(self.partitions.c.partition_id == result.partition_id)
                .values(
                    state="imported",
                    completed_frames=completed,
                    failed_frames=len(frames) - completed,
                    bundle_sha256=result.bundle.sha256,
                    result=result.to_payload(),
                    updated_at=now,
                )
            )
            totals = connection.execute(
                sa.select(
                    sa.func.count(self.partitions.c.partition_id),
                    sa.func.sum(sa.case((self.partitions.c.state == "imported", 1), else_=0)),
                    sa.func.sum(self.partitions.c.completed_frames),
                    sa.func.sum(self.partitions.c.failed_frames),
                ).where(self.partitions.c.run_id == result.run_id)
            ).one()
            total, imported, completed_count, failed_count = (int(value or 0) for value in totals)
            state = "partial" if imported == total and failed_count else "completed" if imported == total else "running"
            connection.execute(
                sa.update(self.runs)
                .where(self.runs.c.run_id == result.run_id)
                .values(
                    state=state,
                    completed_frames=completed_count,
                    failed_frames=failed_count,
                    updated_at=now,
                )
            )
        return True

    def get_run_payload(self, run_id: str) -> dict[str, object] | None:
        sa, _ = _sqlalchemy()
        with self._scope() as connection:
            row = connection.execute(sa.select(self.runs).where(self.runs.c.run_id == run_id)).mappings().one_or_none()
        return None if row is None else dict(row)


__all__ = ["PostgresAnalysisProjectionStore"]
