"""Versioned template definitions stored in Kraken's existing event log."""

from __future__ import annotations

from kraken_manager.application.authorization import AuthorizationPolicy
from kraken_manager.application.errors import ConcurrencyError, ConflictError, NotFoundError
from kraken_manager.application.planning import WorkPlanningService
from kraken_manager.domain.common import DomainValidationError, require_non_empty, validate_uuid
from kraken_manager.domain.events import ActorSnapshot, EventEnvelope
from kraken_manager.domain.identity import Permission
from kraken_manager.domain.planning_templates import WorkPlanTemplate


class WorkTemplateService:
    def __init__(self, events, authorization=None):
        self.events = events
        self.authorization = authorization or AuthorizationPolicy()

    @staticmethod
    def stream_id(template_id):
        return f"work-template:{validate_uuid(str(template_id))}"

    def load(self, project_id, template_id, *, version=None):
        events = self.events.load_stream(self.stream_id(template_id))
        if version is not None and (isinstance(version, bool) or not isinstance(version, int) or version < 1):
            raise DomainValidationError("requested template version must be a positive integer")
        event = next((e for e in reversed(events) if version is None or e.revision == version), None)
        if event is None or str(event.project_id) != str(project_id):
            raise NotFoundError("Work template version not found in this project")
        template = WorkPlanTemplate.from_dict(event.payload)
        if (
            template.id != validate_uuid(template_id)
            or template.project_id != str(project_id)
            or template.version != event.revision
        ):
            raise DomainValidationError("template stream identity or version mismatch")
        return template

    def publish(
        self,
        template,
        *,
        expected_version,
        principal,
        storage,
        roles=(),
        gitlab_identity_verified=False,
        idempotency_key,
    ):
        self.authorization.require(
            principal=principal,
            storage=storage,
            roles=roles,
            permission=Permission.ASSIGN_WORK,
            gitlab_identity_verified=gitlab_identity_verified,
        )
        require_non_empty(idempotency_key, field="idempotency_key")
        if isinstance(expected_version, bool) or not isinstance(expected_version, int) or expected_version < 0:
            raise DomainValidationError("expected template version must be a nonnegative integer")
        if not isinstance(template, WorkPlanTemplate) or template.version != expected_version + 1:
            raise DomainValidationError("new template version must immediately follow the expected version")
        command = WorkPlanningService._command("publish_template", template.to_dict())
        prior = self.events.find_by_idempotency_key(template.project_id, idempotency_key)
        if prior:
            event = prior[-1]
            if event.stream_id != self.stream_id(template.id) or event.payload.get("command") != command:
                raise ConflictError("Idempotency key belongs to a different template command")
            return WorkPlanTemplate.from_dict(event.payload)
        history = self.events.load_stream(self.stream_id(template.id))
        if history and str(history[-1].project_id) != template.project_id:
            raise NotFoundError("Work template does not belong to this project")
        if self.events.current_revision(self.stream_id(template.id)) != expected_version:
            raise ConcurrencyError("Work template version changed")
        payload = template.to_dict()
        payload["command"] = command
        event = EventEnvelope.create(
            stream_id=self.stream_id(template.id),
            project_id=template.project_id,
            revision=template.version,
            event_type="work_template.published",
            payload=payload,
            actor=ActorSnapshot.from_principal(principal),
            idempotency_key=idempotency_key,
        )
        self.events.append(event.stream_id, expected_revision=expected_version, events=(event,))
        return template
