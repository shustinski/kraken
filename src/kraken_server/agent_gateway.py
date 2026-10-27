"""PostgreSQL-backed gateway and lease protocol for remote Kraken agents."""

from __future__ import annotations

import hashlib
import json
import secrets
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

from kraken_manager.domain.workflows import PluginJobManifestV1
from kraken_manager.domain.artifacts import BlobRef
from kraken_manager.application.dto import StoredContent
from kraken_manager.infrastructure.filesystem._codec import decode_model, encode_model

from .agent_auth import AgentIdentity, PostgresAgentTokenStore
from .agent_resources import QueueAdmission, QueueResourceBudget, requested_resources, validate_telemetry

DESKTOP_CAPABILITIES = frozenset(
    {
        "frames.vectorize.v1",
        "frames.dataset.prepare.v1",
        "dataset.model.train.v1",
        "frames.binary-segment.v2",
        "frames.confidence.v1",
    }
)


class PostgresAgentGateway:
    def __init__(
        self,
        engine: Any,
        tokens: PostgresAgentTokenStore,
        blobs: Any,
        *,
        resource_budget: QueueResourceBudget | None = None,
    ) -> None:
        import sqlalchemy as sa

        self.engine = engine
        self.tokens = tokens
        self.blobs = blobs
        self.metadata = sa.MetaData()
        self.jobs = sa.Table(
            "background_jobs",
            self.metadata,
            sa.Column("job_id", sa.Uuid(as_uuid=False), primary_key=True),
            sa.Column("project_id", sa.Uuid(as_uuid=False), nullable=False),
            sa.Column("state", sa.Text, nullable=False),
            sa.Column("payload", sa.JSON, nullable=False),
            sa.Column("lease_owner", sa.Text),
            sa.Column("lease_claim", sa.Text),
            sa.Column("telemetry", sa.JSON),
            sa.Column("lease_until", sa.DateTime(timezone=True)),
            sa.Column("lease_attempts", sa.Integer, nullable=False),
            sa.Column("last_error", sa.Text),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        )
        self.admission = QueueAdmission(self.jobs, resource_budget or QueueResourceBudget.from_environment())
        self.publications = sa.Table(
            "agent_job_publications",
            self.metadata,
            sa.Column("publication_id", sa.Text, primary_key=True),
            sa.Column("job_id", sa.Uuid(as_uuid=False), primary_key=True),
            sa.Column("payload", sa.JSON, nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        )
        self.outputs = sa.Table(
            "agent_job_outputs",
            self.metadata,
            sa.Column("job_id", sa.Uuid(as_uuid=False), primary_key=True),
            sa.Column("output_id", sa.Text, primary_key=True),
            sa.Column("sha256", sa.Text, nullable=False),
            sa.Column("size_bytes", sa.BigInteger, nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        )

    def is_available(self, capability: str, protocol_version: str) -> bool:
        return protocol_version == "1.0" and (
            capability in DESKTOP_CAPABILITIES or self.tokens.has_capability(capability)
        )

    def desktop_identity(self, job_id: str, principal_id: str) -> tuple[AgentIdentity, str]:
        import sqlalchemy as sa

        with self.engine.connect() as connection:
            payload = connection.execute(
                sa.select(self.jobs.c.payload).where(self.jobs.c.job_id == job_id)
            ).scalar_one_or_none()
        if (
            payload is None
            or str(payload.get("actor_principal_id", "")) != principal_id
            or payload.get("parameters", {}).get("execution_host") != "desktop"
            or payload.get("capability") not in DESKTOP_CAPABILITIES
        ):
            raise PermissionError("Only the initiating user can run this interactive desktop job")
        return (
            AgentIdentity(f"desktop:{principal_id}", "Kraken desktop", frozenset({payload["capability"]})),
            str(payload["project_id"]),
        )

    def submit(self, manifest: PluginJobManifestV1) -> None:
        self._submit_payload(str(manifest.job_id), str(manifest.project_id), encode_model(manifest))

    def submit_analysis(self, partition, *, actor_id, resources=None) -> None:
        from kraken_core.analysis_run_protocol import AnalysisPartitionJobManifest
        from kraken_manager.domain.common import validate_uuid

        if not isinstance(partition, AnalysisPartitionJobManifest):
            raise TypeError("Analysis queue requires the native partition manifest")
        parameters = {}
        if resources is not None:
            parameters["execution_resources"] = (
                resources.to_dict() if hasattr(resources, "to_dict") else dict(resources)
            )
        payload = dict(
            kind="analysis",
            capability="karakal.analyze.v1",
            project_id=partition.project_id,
            actor_principal_id=validate_uuid(str(actor_id)),
            parameters=parameters,
            native_manifest=partition.to_payload(),
        )
        self._submit_payload(str(partition.job_id), str(partition.project_id), payload)

    def submit_stitch(self, manifest) -> None:
        from kraken_core.plugin_protocol import PluginJobManifestV2
        if not isinstance(manifest, PluginJobManifestV2) or manifest.operation != "layer.stitch.v1":
            raise ValueError("Stitch queue requires a native layer.stitch.v1 manifest")
        owner = manifest.parameters.get("_pipeline")
        resources = {"cpu": 1, "gpu": 0, "memory_mb": 512} if owner is None else owner["resources"]
        self._submit_payload(manifest.job_id, manifest.project_id, {
            "kind": "stitch", "capability": manifest.operation, "project_id": manifest.project_id,
            "actor_principal_id": manifest.actor_id,
            "parameters": {"execution_resources": resources},
            "native_manifest": manifest.to_dict(),
        })

    def reconcile_imported_stitch(self, request) -> None:
        """Canonical accepted receipt wins after a lost queue completion commit."""
        import sqlalchemy as sa
        from kraken_core.plugin_protocol import PluginJobManifestV2
        if request.state != "imported" or request.output is None or request.receipt_sha256 is None:
            raise ValueError("Queue settlement requires an accepted canonical mosaic receipt")
        with self.engine.begin() as connection:
            self.admission.lock(connection)
            row = connection.execute(sa.select(self.jobs).where(self.jobs.c.job_id == request.job_id).with_for_update()).mappings().one_or_none()
            if row is None:
                return
            payload = row["payload"]
            if payload.get("kind") != "stitch" or PluginJobManifestV2.from_dict(payload["native_manifest"]).to_json() != request.manifest_json:
                raise ValueError("Queue job differs from the canonical accepted stitch request")
            if row["state"] != "completed":
                connection.execute(sa.update(self.jobs).where(self.jobs.c.job_id == request.job_id).values(
                    state="completed", lease_owner=None, lease_claim=None, lease_until=None, last_error=None, updated_at=datetime.now(UTC)))

    def _submit_payload(self, job_id, project_id, payload) -> None:
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        requested_resources(payload.get("parameters", {}))
        now = datetime.now(UTC)
        with self.engine.begin() as connection:
            connection.execute(
                pg_insert(self.jobs)
                .values(
                    job_id=job_id,
                    project_id=project_id,
                    state="queued",
                    payload=payload,
                    lease_owner=None,
                    lease_until=None,
                    lease_attempts=0,
                    last_error=None,
                    created_at=now,
                    updated_at=now,
                )
                .on_conflict_do_nothing(index_elements=[self.jobs.c.job_id])
            )
            import sqlalchemy as sa

            stored = connection.execute(sa.select(self.jobs.c.payload).where(self.jobs.c.job_id == job_id)).scalar_one()
            if json.dumps(stored, sort_keys=True, separators=(",", ":")) != json.dumps(
                payload, sort_keys=True, separators=(",", ":")
            ):
                raise ValueError("Job ID was already enqueued with a different immutable manifest")

    def cancel(self, job_id: object) -> None:
        """Stop a queued or leased agent, and ignore a queue row that is already finished.

        Domain jobs can stay open after the worker publishes a partial result and
        the queue row moves to completed, failed, or cancelled. Cancelling that
        domain job must not fail just because there is nothing left to stop.
        """
        import sqlalchemy as sa

        terminal = {"cancelled", "completed", "failed"}
        with self.engine.begin() as connection:
            state = connection.execute(
                sa.select(self.jobs.c.state).where(self.jobs.c.job_id == str(job_id))
            ).scalar_one_or_none()
            if state is None or state in terminal:
                return
            result = connection.execute(
                sa.update(self.jobs)
                .where(
                    self.jobs.c.job_id == str(job_id),
                    self.jobs.c.state.in_(("queued", "leased")),
                )
                .values(
                    state="cancelled",
                    lease_owner=None,
                    lease_until=None,
                    updated_at=datetime.now(UTC),
                )
            )
            if result.rowcount == 1:
                return
            state = connection.execute(
                sa.select(self.jobs.c.state).where(self.jobs.c.job_id == str(job_id))
            ).scalar_one_or_none()
        if state is None or state in terminal:
            return
        raise RuntimeError("Agent job is not cancellable")

    def lease(self, agent: AgentIdentity, *, seconds: int = 60, job_id: str | None = None) -> dict[str, Any] | None:
        import sqlalchemy as sa

        with self.engine.begin() as connection:
            self.admission.lock(connection)
            now = datetime.now(UTC)
            lease_until = now + timedelta(seconds=max(15, min(int(seconds), 300)))
            active, used, _legacy = self.admission.usage(connection, now)
            host = self.jobs.c.payload["parameters"]["execution_host"].as_string()
            actor = self.jobs.c.payload["actor_principal_id"].as_string()
            statement = sa.select(self.jobs).where(
                self.jobs.c.payload["capability"].as_string().in_(agent.capabilities),
                sa.or_(
                    host.is_(None),
                    host != "desktop",
                    actor == agent.token_id.removeprefix("desktop:")
                    if agent.token_id.startswith("desktop:")
                    else sa.false(),
                ),
                sa.or_(
                    sa.and_(
                        self.jobs.c.state == "queued",
                        sa.or_(self.jobs.c.lease_until.is_(None), self.jobs.c.lease_until < now),
                    ),
                    sa.and_(self.jobs.c.state == "leased", self.jobs.c.lease_until < now),
                ),
            )
            if job_id is not None:
                statement = statement.where(self.jobs.c.job_id == job_id)
            cursor = None
            row = None
            while row is None:
                page = statement
                if cursor is not None:
                    page = page.where(sa.tuple_(self.jobs.c.created_at, self.jobs.c.job_id) > cursor)
                candidates = (
                    connection.execute(
                        page.order_by(self.jobs.c.created_at, self.jobs.c.job_id)
                        .with_for_update(skip_locked=True)
                        .limit(100)
                    )
                    .mappings()
                    .all()
                )
                if not candidates:
                    break
                for candidate in candidates:
                    try:
                        reason = self.admission.decision(candidate["payload"], active, used)
                    except ValueError:
                        reason = "admission:invalid_resource_request"
                    if reason is None:
                        row = candidate
                        break
                    if candidate["last_error"] != reason:
                        connection.execute(
                            sa.update(self.jobs)
                            .where(self.jobs.c.job_id == candidate["job_id"])
                            .values(last_error=reason, updated_at=now)
                        )
                cursor = (candidates[-1]["created_at"], candidates[-1]["job_id"])
                if len(candidates) < 100 or job_id is not None:
                    break
            if row is None:
                return None
            connection.execute(
                sa.update(self.jobs)
                .where(self.jobs.c.job_id == row["job_id"])
                .values(
                    state="leased",
                    lease_owner=agent.token_id,
                    lease_claim=(claim := secrets.token_urlsafe(32)),
                    lease_until=lease_until,
                    lease_attempts=self.jobs.c.lease_attempts + 1,
                    last_error=None,
                    updated_at=now,
                )
            )
        return {
            "job_id": str(row["job_id"]),
            "lease_until": lease_until.isoformat(),
            "lease_claim": claim,
            "manifest": dict(row["payload"].get("native_manifest", row["payload"])),
        }

    def heartbeat(self, job_id: str, agent: AgentIdentity, *, seconds: int = 60, telemetry: Any = None) -> str:
        import sqlalchemy as sa

        with self.engine.begin() as connection:
            self.admission.lock(connection)
            self._require_lease(connection, job_id, agent)
            now = datetime.now(UTC)
            lease_until = now + timedelta(seconds=max(15, min(int(seconds), 300)))
            values = {"lease_until": lease_until, "updated_at": now}
            if telemetry is not None:
                values["telemetry"] = {"received_at": now.isoformat(), "sample": validate_telemetry(telemetry)}
            result = connection.execute(
                sa.update(self.jobs)
                .where(
                    self.jobs.c.job_id == job_id,
                    self.jobs.c.lease_owner == agent.token_id,
                    self.jobs.c.lease_claim == agent.lease_claim,
                    self.jobs.c.state == "leased",
                    self.jobs.c.lease_until > now,
                )
                .values(**values)
            )
        if result.rowcount != 1:
            raise PermissionError("Agent does not own an active lease for this job")
        return lease_until.isoformat()

    def resource_status(self) -> dict[str, Any]:
        with self.engine.begin() as connection:
            self.admission.lock(connection)
            return self.admission.status(connection, datetime.now(UTC))

    def host_observations(self) -> tuple[dict[str, Any], ...]:
        """Latest report per anonymous host, with receipt age and explicit scope.

        Retained on the existing queue row; this is a latest observation API,
        not a utilization time series or per-job billing record.
        """
        import sqlalchemy as sa

        now = datetime.now(UTC)
        with self.engine.connect() as connection:
            packets = (
                connection.execute(sa.select(self.jobs.c.telemetry).where(self.jobs.c.telemetry.is_not(None)))
                .scalars()
                .all()
            )
        hosts = {}
        for packet in packets:
            if not packet:
                continue
            received = datetime.fromisoformat(packet["received_at"])
            identifier = packet["sample"]["host_id"]
            if identifier not in hosts or received > hosts[identifier][0]:
                age = max(0, (now - received).total_seconds())
                hosts[identifier] = (
                    received,
                    {
                        **packet,
                        "age_seconds": age,
                        "stale": age > 120,
                        "attribution": "authenticated_agent_report_whole_host",
                    },
                )
        return tuple(hosts[key][1] for key in sorted(hosts))

    def admission_reason(self, job_id: str) -> str | None:
        import sqlalchemy as sa

        with self.engine.connect() as connection:
            value = connection.execute(
                sa.select(self.jobs.c.last_error).where(self.jobs.c.job_id == job_id)
            ).scalar_one_or_none()
        return str(value) if value and str(value).startswith("admission:") else None

    def manifest(self, job_id: str, agent: AgentIdentity):
        with self.engine.connect() as connection:
            payload = self._require_lease(connection, job_id, agent, lock=False)
        if payload.get("kind") == "analysis":
            from kraken_core.analysis_run_protocol import AnalysisPartitionJobManifest

            return AnalysisPartitionJobManifest.from_payload(payload["native_manifest"])
        if payload.get("kind") == "stitch":
            from kraken_core.plugin_protocol import PluginJobManifestV2
            return PluginJobManifestV2.from_dict(payload["native_manifest"])
        return decode_model(PluginJobManifestV1, payload)

    def _require_lease(
        self, connection: Any, job_id: str, agent: AgentIdentity, *, lock: bool = True
    ) -> dict[str, Any]:
        import sqlalchemy as sa

        if not agent.lease_claim:
            raise PermissionError("Lease claim required: update the Agent/Hub client and acquire a new lease")
        statement = sa.select(self.jobs).where(self.jobs.c.job_id == job_id)
        if lock:
            statement = statement.with_for_update()
        row = connection.execute(statement).mappings().one_or_none()
        expiry = None if row is None else row["lease_until"]
        if expiry is not None and expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=UTC)
        # Capture time after acquiring the row lock; a blocked caller must not
        # extend or publish through a lease which expired while it was waiting.
        if (
            row is None
            or row["state"] != "leased"
            or row["lease_owner"] != agent.token_id
            or row["lease_claim"] != agent.lease_claim
            or expiry is None
            or expiry <= datetime.now(UTC)
        ):
            raise PermissionError("Agent lease expired or claim superseded; acquire a new lease")
        return dict(row["payload"])

    def publish(self, job_id: str, agent: AgentIdentity, payload: Mapping[str, Any]) -> bool:
        manifest = self.manifest(job_id, agent)
        del manifest
        canonical = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), default=str)
        publication_id = str(payload.get("publication_id") or hashlib.sha256(canonical.encode()).hexdigest())
        import sqlalchemy as sa
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        with self.engine.begin() as connection:
            self._require_lease(connection, job_id, agent)
            result = connection.execute(
                pg_insert(self.publications)
                .values(
                    publication_id=publication_id,
                    job_id=job_id,
                    payload=dict(payload),
                    created_at=datetime.now(UTC),
                )
                .on_conflict_do_nothing(
                    index_elements=[
                        self.publications.c.job_id,
                        self.publications.c.publication_id,
                    ]
                )
                .returning(self.publications.c.publication_id)
            ).scalar_one_or_none()
            stored_payload = connection.execute(
                sa.select(self.publications.c.payload).where(
                    self.publications.c.job_id == job_id,
                    self.publications.c.publication_id == publication_id,
                )
            ).scalar_one()
            if json.dumps(stored_payload, sort_keys=True, separators=(",", ":"), default=str) != canonical:
                raise RuntimeError("A publication ID was already used with a different payload")
        return result is not None

    def upload_output(
        self,
        job_id: str,
        output_id: str,
        agent: AgentIdentity,
        chunks: Any,
        *,
        expected_sha256: str,
    ) -> dict[str, Any]:
        self.manifest(job_id, agent)
        stored = self.blobs.put(chunks, expected_sha256=expected_sha256)
        return self._record_output(job_id, output_id, stored, agent)

    def register_output(
        self,
        job_id: str,
        output_id: str,
        agent: AgentIdentity,
        *,
        expected_sha256: str,
        expected_size: int,
    ) -> dict[str, Any]:
        self.manifest(job_id, agent)
        reference = self.blobs.stat(BlobRef(expected_sha256, expected_size))
        if reference.size_bytes != expected_size:
            raise RuntimeError("Stored Agent output size does not match the completed upload")
        return self._record_output(job_id, output_id, StoredContent(reference, already_existed=True), agent)

    def _record_output(
        self, job_id: str, output_id: str, stored: StoredContent, agent: AgentIdentity
    ) -> dict[str, Any]:
        import sqlalchemy as sa
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        with self.engine.begin() as connection:
            self._require_lease(connection, job_id, agent)
            existing = (
                connection.execute(
                    sa.select(self.outputs.c.sha256, self.outputs.c.size_bytes).where(
                        self.outputs.c.job_id == job_id,
                        self.outputs.c.output_id == output_id,
                    )
                )
                .mappings()
                .one_or_none()
            )
            if existing is not None:
                if (
                    str(existing["sha256"]) != stored.blob.sha256
                    or int(existing["size_bytes"]) != stored.blob.size_bytes
                ):
                    raise RuntimeError("An output ID was already published with different content")
                return {
                    "sha256": str(existing["sha256"]),
                    "size_bytes": int(existing["size_bytes"]),
                }
            connection.execute(
                pg_insert(self.outputs)
                .values(
                    job_id=job_id,
                    output_id=output_id,
                    sha256=stored.blob.sha256,
                    size_bytes=stored.blob.size_bytes,
                    created_at=datetime.now(UTC),
                )
                .on_conflict_do_nothing(index_elements=[self.outputs.c.job_id, self.outputs.c.output_id])
            )
            persisted = (
                connection.execute(
                    sa.select(self.outputs.c.sha256, self.outputs.c.size_bytes).where(
                        self.outputs.c.job_id == job_id,
                        self.outputs.c.output_id == output_id,
                    )
                )
                .mappings()
                .one()
            )
            if str(persisted["sha256"]) != stored.blob.sha256 or int(persisted["size_bytes"]) != stored.blob.size_bytes:
                raise RuntimeError("An output ID was already published with different content")
        return {"sha256": stored.blob.sha256, "size_bytes": stored.blob.size_bytes}

    def finish(
        self, job_id: str, agent: AgentIdentity, *, failed: bool, error: str = "", import_result: Any = None
    ) -> Any:
        import sqlalchemy as sa

        with self.engine.begin() as connection:
            self.admission.lock(connection)
            self._require_lease(connection, job_id, agent)
            imported = import_result() if import_result is not None else None
            now = datetime.now(UTC)
            result = connection.execute(
                sa.update(self.jobs)
                .where(self.jobs.c.job_id == job_id, self.jobs.c.lease_owner == agent.token_id)
                .values(
                    state="failed" if failed else "completed",
                    lease_owner=None,
                    lease_claim=None,
                    lease_until=None,
                    last_error=error or None,
                    updated_at=now,
                )
            )
        if result.rowcount != 1:
            raise PermissionError("Agent does not own this job")
        return imported

    def publications_for(self, job_id: str, agent: AgentIdentity) -> tuple[dict[str, Any], ...]:
        self.manifest(job_id, agent)
        import sqlalchemy as sa

        with self.engine.connect() as connection:
            rows = (
                connection.execute(
                    sa.select(self.publications.c.payload)
                    .where(self.publications.c.job_id == job_id)
                    .order_by(self.publications.c.created_at)
                )
                .scalars()
                .all()
            )
        return tuple(dict(item) for item in rows)

    def analysis_job_manifest(self, job_id: str):
        """Internal lookup after the calling Agent route has checked its claim."""
        import sqlalchemy as sa
        from kraken_core.analysis_run_protocol import AnalysisPartitionJobManifest

        with self.engine.connect() as connection:
            payload = connection.execute(sa.select(self.jobs.c.payload).where(self.jobs.c.job_id == job_id)).scalar_one_or_none()
        if not payload or payload.get("kind") != "analysis":
            return None
        return AnalysisPartitionJobManifest.from_payload(payload["native_manifest"])

    def output_blob(self, job_id: str, output_id: str) -> BlobRef:
        import sqlalchemy as sa

        with self.engine.connect() as connection:
            row = (
                connection.execute(
                    sa.select(self.outputs.c.sha256, self.outputs.c.size_bytes).where(
                        self.outputs.c.job_id == job_id,
                        self.outputs.c.output_id == output_id,
                    )
                )
                .mappings()
                .one_or_none()
            )
        if row is None:
            raise FileNotFoundError(f"Agent output {output_id} was not uploaded")
        return BlobRef(str(row["sha256"]), int(row["size_bytes"]))

    def stitch_job_manifest(self, job_id):
        """Internal recovery lookup; external Agent routes first require a lease."""
        import sqlalchemy as sa
        from kraken_core.plugin_protocol import PluginJobManifestV2
        with self.engine.connect() as connection:
            payload = connection.execute(sa.select(self.jobs.c.payload).where(self.jobs.c.job_id == job_id)).scalar_one_or_none()
        if not payload or payload.get("kind") != "stitch":
            return None
        return PluginJobManifestV2.from_dict(payload["native_manifest"])


__all__ = ["PostgresAgentGateway"]
