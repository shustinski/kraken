"""Transaction-bound projection/rebuild of canonical server AnalysisRun events."""

from __future__ import annotations

from kraken_core.analysis_bundle import sha256_stream, stream_bundle_records
from kraken_core.analysis_run_protocol import (
    AnalysisRunManifest,
    AnalysisPartitionJobManifest,
    AnalysisPartitionResultManifest,
)
from kraken_manager.application.errors import ConflictError
from kraken_manager.domain.artifacts import BlobRef
from .analysis_store import PostgresAnalysisProjectionStore


class PostgresAnalysisWorkflowStore:
    def __init__(self, unit, engine):
        self.unit = unit
        self.projection = PostgresAnalysisProjectionStore(engine, connection=unit._connection)

    def sync(self, state):
        import sqlalchemy as sa

        projection, connection = self.projection, self.unit._connection
        run = AnalysisRunManifest.from_payload(state["manifest"])
        partitions = tuple(
            AnalysisPartitionJobManifest.from_payload(p["manifest"]) for p in state["partitions"].values()
        )
        self.projection.create_run(run, partitions)
        existing = {
            row["partition_id"]: row
            for row in connection.execute(
                sa.select(projection.partitions).where(projection.partitions.c.run_id == run.run_id)
            ).mappings()
        }
        for identifier, partition in state["partitions"].items():
            row = existing[identifier]
            if row["manifest"] == partition["manifest"] and row["state"] == partition["state"]:
                continue
            replacement = row["job_id"] != partition["manifest"]["job_id"]
            values = {
                "manifest": partition["manifest"],
                "job_id": partition["manifest"]["job_id"],
                "state": partition["state"],
            }
            if replacement:
                connection.execute(sa.delete(projection.frames).where(projection.frames.c.partition_id == identifier))
                values.update(
                    attempt=row["attempt"] + 1, bundle_sha256=None, result=None, completed_frames=0, failed_frames=0
                )
            connection.execute(
                sa.update(projection.partitions)
                .where(projection.partitions.c.partition_id == identifier)
                .values(**values)
            )
        connection.execute(
            sa.update(projection.runs).where(projection.runs.c.run_id == run.run_id).values(state=state["state"])
        )

    def records(self, partition, result, reference):
        if (result.job_id, result.run_id, result.partition_id, result.project_id) != (
            partition.job_id,
            partition.run_id,
            partition.partition_id,
            partition.project_id,
        ):
            raise ConflictError("Analysis result does not match its exact partition attempt")
        if (
            result.bundle is None
            or reference.sha256 != result.bundle.sha256
            or reference.size_bytes != result.bundle.compressed_size
        ):
            raise ConflictError("Analysis bundle hash/size differs from its registered output")
        with self.unit.blobs.open(reference) as stream:
            digest, size = sha256_stream(stream)
        if (digest, size) != (reference.sha256, reference.size_bytes):
            raise ConflictError("Stored analysis bundle failed checksum validation")
        with self.unit.blobs.open(reference) as stream:
            records = tuple(
                stream_bundle_records(
                    stream, result.bundle, expected_frame_ids=(frame.frame_id for frame in partition.frames)
                )
            )
        expected = {frame.frame_id: (frame.x, frame.y) for frame in partition.frames}
        if any((frame.x, frame.y) != expected[frame.frame_id] for frame in records):
            raise ConflictError("Analysis records changed their immutable frame coordinates")
        return records

    def rebuild(self, state):
        import sqlalchemy as sa

        projection = self.projection
        identifier = state["manifest"]["run_id"]
        self.unit._connection.execute(sa.delete(projection.runs).where(projection.runs.c.run_id == identifier))
        # Create first, import canonical bundles second, restore terminal states last.
        base = {
            **state,
            "state": "queued",
            "partitions": {key: {**part, "state": "queued"} for key, part in state["partitions"].items()},
        }
        self.sync(base)
        for partition in state["partitions"].values():
            if partition["state"] != "imported":
                continue
            result = AnalysisPartitionResultManifest.from_payload(partition["result"])
            manifest = AnalysisPartitionJobManifest.from_payload(partition["manifest"])
            reference = BlobRef(**partition["blob"])
            projection.import_partition(result, self.records(manifest, result, reference))
        self.sync(state)
