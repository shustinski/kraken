"""persistent_services: import agent result."""
from __future__ import annotations
import base64
import json
from typing import Any
from kraken_manager.application.dto import (
    CancelPluginJobCommand,
    ImportPluginResultCommand,
)
from kraken_manager.application.dto import (
    CommandContext as ApplicationCommandContext,
)
from kraken_manager.application.errors import (
    ApplicationError,
    AuthorizationError as ApplicationAuthorizationError,
    NotFoundError as ApplicationNotFoundError,
)
from kraken_manager.application.plugin_jobs import (
    ImportPluginResultHandler,
)
from kraken_manager.domain.common import (
    PluginJobId,
    validate_uuid,
)
from kraken_manager.domain.workflows import (
    PluginResultManifestV1,
    PluginResultOutcome,
)
from kraken_core.plugin_protocol import PluginResultManifest
from kraken_manager.infrastructure.plugin.result_reader import domain_result_from_transport
from kraken_manager.infrastructure.filesystem._codec import encode_model
from kraken_server.services import CommandContext, ConflictError, ValidationError

from kraken_server.persistent_services_parts.layer_source_scan import (
    SystemClock,
    _AgentBlobResultReader,
    _project_id,
)


class ImportAgentResultMixin:
    def cancel_plugin_job(
        self,
        project_id: str,
        job_id: str,
        context: CommandContext,
    ) -> dict[str, Any]:
        if self._cancel_plugin_job is None:
            raise ConflictError("Server agent queue is not configured")
        try:
            value = self._cancel_plugin_job(
                CancelPluginJobCommand(
                    context=self._application_context(context),
                    project_id=_project_id(project_id),
                    job_id=PluginJobId(validate_uuid(job_id, field="job_id")),
                    expected_revision=self._require_revision(context),
                )
            )
            return encode_model(value)
        except Exception as exc:
            if isinstance(exc, RuntimeError) and not isinstance(exc, ApplicationError):
                raise ConflictError(str(exc)) from exc
            self._translate_lifecycle_error(exc)
            raise AssertionError("unreachable")

    def import_agent_result(self, job_id: str, agent: Any) -> dict[str, Any]:
        if self.agent_gateway is None:
            raise ConflictError("Server agent queue is not configured")
        publications = self.agent_gateway.publications_for(job_id, agent)
        if len(publications) != 1:
            raise ConflictError("Exactly one final V1 publication is required")
        try:
            transport = PluginResultManifest.from_dict(publications[0])
            result_manifest = domain_result_from_transport(transport)
            job = self.projections.get_plugin_job(
                PluginJobId(validate_uuid(job_id, field="job_id"))
            )
            if job is None:
                raise ApplicationNotFoundError("Plugin job was not found")
            actor = self.identities.get(job.actor_principal_id)
            if actor is None or not actor.active:
                raise ApplicationAuthorizationError("Initiating principal is inactive")
            references = {
                output.relative_path: self.agent_gateway.output_blob(job_id, output.output_id)
                for output in transport.outputs
            }
            imported = ImportPluginResultHandler(
                self.uow_factory,
                self.profiles,
                SystemClock(),
                _AgentBlobResultReader(self.uow_factory.blobs, references),
            )(
                ImportPluginResultCommand(
                    context=ApplicationCommandContext(
                        actor=actor,
                        idempotency_key=f"server-agent-import:{job_id}:{transport.completed_at}",
                        gitlab_identity_verified=True,
                    ),
                    manifest=result_manifest,
                    confirm_partial=False,
                )
            )
            return {
                "job": encode_model(imported.job),
                "versions": [encode_model(item) for item in imported.versions],
                "requires_partial_confirmation": imported.requires_partial_confirmation,
                "already_imported": imported.already_imported,
            }
        except Exception as exc:
            self._translate_lifecycle_error(exc)
            raise AssertionError("unreachable")

    def fail_agent_job(self, job_id: str, error: str) -> dict[str, Any]:
        try:
            job = self.projections.get_plugin_job(
                PluginJobId(validate_uuid(job_id, field="job_id"))
            )
            if job is None:
                raise ApplicationNotFoundError("Plugin job was not found")
            actor = self.identities.get(job.actor_principal_id)
            if actor is None or not actor.active:
                raise ApplicationAuthorizationError("Initiating principal is inactive")
            manifest = PluginResultManifestV1(
                job_id=job.id,
                plugin_name="kraken-agent",
                plugin_version="1",
                results=(),
                parameters_applied={"error": error},
                protocol_version="1.0",
                outcome=PluginResultOutcome.FAILED,
            )
            imported = ImportPluginResultHandler(
                self.uow_factory,
                self.profiles,
                SystemClock(),
                _AgentBlobResultReader(self.uow_factory.blobs, {}),
            )(
                ImportPluginResultCommand(
                    context=ApplicationCommandContext(
                        actor=actor,
                        idempotency_key=f"server-agent-failed:{job_id}:{job.revision}",
                        gitlab_identity_verified=True,
                    ),
                    manifest=manifest,
                    confirm_partial=False,
                )
            )
            return {"job": encode_model(imported.job)}
        except Exception as exc:
            self._translate_lifecycle_error(exc)
            raise AssertionError("unreachable")

    @staticmethod
    def _decode_cursor(cursor: str | None) -> int:
        if not cursor:
            return 0
        try:
            raw = base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4))
            value = int(json.loads(raw)["position"])
        except Exception as exc:
            raise ValidationError("Invalid history cursor") from exc
        if value < 0:
            raise ValidationError("Invalid history cursor")
        return value

    @staticmethod
    def _encode_cursor(position: int) -> str:
        raw = json.dumps({"position": position}, separators=(",", ":")).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    def history(self, project_id: str, *, cursor: str | None, limit: int) -> dict[str, Any]:
        self.get_project(project_id)
        if limit < 1 or limit > 500:
            raise ValidationError("History limit must be between 1 and 500")
        after = self._decode_cursor(cursor)
        rows = self.events.list_project_events(_project_id(project_id), after_position=after, limit=limit + 1)
        has_more = len(rows) > limit
        page = rows[:limit]
        items = [
            {
                "position": position,
                "event_id": event.event_id,
                "stream_id": event.stream_id,
                "revision": event.revision,
                "event_type": event.event_type,
                "schema_version": event.schema_version,
                "recorded_at": event.recorded_at.isoformat(),
                "effective_at": None if event.effective_at is None else event.effective_at.isoformat(),
                "actor": {
                    "principal_id": str(event.actor.principal_id),
                    "provider": event.actor.provider.value,
                    "subject": event.actor.subject,
                    "display_name": event.actor.display_name,
                },
                "performer_id": None if event.performer_id is None else str(event.performer_id),
                "payload": encode_model(event.payload),
            }
            for position, event in page
        ]
        next_cursor = self._encode_cursor(page[-1][0]) if has_more and page else None
        return {"items": items, "next_cursor": next_cursor}
