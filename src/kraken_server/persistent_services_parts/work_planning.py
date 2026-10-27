"""Server transport for durable engineering plans."""

from __future__ import annotations

from kraken_manager.application.errors import ConflictError, NotFoundError
from kraken_manager.application.planning import WorkPlanningService
from kraken_manager.domain.common import validate_uuid
from kraken_manager.domain.planning import WorkPlan
from kraken_manager.domain.project import ProjectState
from kraken_manager.infrastructure.planning_commands import command_id, decode_arguments, list_plans


class WorkPlanningMixin:
    def _planning_project(self, projections, project_id, mutate=False):
        project = projections.get_project(validate_uuid(str(project_id)))
        if project is None:
            raise NotFoundError("Project was not found")
        if mutate and project.state is not ProjectState.ACTIVE:
            raise ConflictError("Archived projects cannot change work plans")
        return project

    def list_work_plans(self, project_id):
        try:
            self._planning_project(self.projections, project_id)
            return [p.to_dict()["plan"] for p in list_plans(self.events, str(project_id))]
        except Exception as exc:
            self._translate_planning_error(exc)
            raise

    def get_work_plan(self, project_id, plan_id):
        try:
            self._planning_project(self.projections, project_id)
            plan = WorkPlanningService(self.events).load(str(plan_id))
            if plan.project_id != str(project_id):
                raise NotFoundError("Plan does not belong to the project")
            return plan.to_dict()["plan"]
        except Exception as exc:
            self._translate_planning_error(exc)
            raise

    def transport_create_work_plan(self, project_id, payload, context):
        actor = self._actor(context.actor_id)
        try:
            if set(payload) - {"title"}:
                raise ValueError("Only a plan title can be supplied")
            if context.expected_revision not in {None, 0}:
                raise ConflictError("New plans require revision zero")
            with self.uow_factory() as unit:
                project = self._planning_project(unit.projections, project_id, mutate=True)
                plan = WorkPlan.create(
                    project.id, payload["title"], plan_id=command_id(project.id, context.idempotency_key, "plan")
                )
                result = WorkPlanningService(unit.event_store, projections=unit.projections).create(
                    plan,
                    principal=actor,
                    storage=self.profiles.get(project.storage_profile),
                    roles=unit.acl.roles_for(project.id, actor.id),
                    gitlab_identity_verified=True,
                    idempotency_key=context.idempotency_key,
                )
                unit.commit()
                return result.to_dict()["plan"]
        except Exception as exc:
            self._translate_planning_error(exc)
            raise

    def transport_mutate_work_plan(self, project_id, plan_id, payload, context):
        actor = self._actor(context.actor_id)
        try:
            if set(payload) - {"operation", "arguments"}:
                raise ValueError("Invalid work plan mutation fields")
            if context.expected_revision is None:
                raise ConflictError("Expected plan revision is required")
            with self.uow_factory() as unit:
                project = self._planning_project(unit.projections, project_id, mutate=True)
                service = WorkPlanningService(unit.event_store, projections=unit.projections)
                plan = service.load(str(plan_id))
                if plan.project_id != str(project.id):
                    raise NotFoundError("Plan does not belong to the project")
                operation = payload["operation"]
                args = decode_arguments(
                    operation,
                    payload.get("arguments", {}),
                    unit.projections,
                    project,
                    self.performer_store,
                    context.idempotency_key,
                    plan=plan,
                    events=unit.event_store,
                )
                result = service.execute(
                    str(plan.id),
                    expected_revision=context.expected_revision,
                    principal=actor,
                    storage=self.profiles.get(project.storage_profile),
                    roles=unit.acl.roles_for(project.id, actor.id),
                    gitlab_identity_verified=True,
                    operation=operation,
                    arguments=args,
                    idempotency_key=context.idempotency_key,
                )
                unit.commit()
                return result.to_dict()["plan"]
        except Exception as exc:
            self._translate_planning_error(exc)
            raise

    def _translate_planning_error(self, exc):
        from kraken_manager.domain.common import InvalidStateTransition
        from kraken_manager.infrastructure.filesystem.event_store import EventStreamConflict
        from kraken_server.services import ConflictError as TransportConflictError, ValidationError

        if isinstance(exc, (InvalidStateTransition, EventStreamConflict)):
            raise TransportConflictError(str(exc)) from exc
        if isinstance(exc, KeyError):
            raise ValidationError(f"Required planning field is missing: {exc}") from exc
        self._translate_lifecycle_error(exc)

    def list_work_templates(self, project_id):
        from kraken_manager.infrastructure.planning_templates import list_templates

        try:
            self._planning_project(self.projections, project_id)
            return [t.to_dict()["template"] for t in list_templates(self.events, str(project_id))]
        except Exception as exc:
            self._translate_planning_error(exc)
            raise

    def get_work_template(self, project_id, template_id, *, version=None):
        from kraken_manager.application.planning_templates import WorkTemplateService

        try:
            self._planning_project(self.projections, project_id)
            template = WorkTemplateService(self.events).load(str(project_id), str(template_id), version=version)
            return template.to_dict()["template"]
        except Exception as exc:
            self._translate_planning_error(exc)
            raise

    def transport_save_work_template(self, project_id, payload, context, *, template_id=None):
        from kraken_manager.application.planning_templates import WorkTemplateService
        from kraken_manager.infrastructure.planning_templates import decode_template_definition

        actor = self._actor(context.actor_id)
        try:
            expected = context.expected_revision
            if expected is None or isinstance(expected, bool) or not isinstance(expected, int) or expected < 0:
                raise ConflictError("Expected template version is required")
            if template_id is None and expected != 0:
                raise ConflictError("New templates require version zero")
            with self.uow_factory() as unit:
                project = self._planning_project(unit.projections, project_id, mutate=True)
                identifier = template_id or command_id(project.id, context.idempotency_key, "template")
                template = decode_template_definition(
                    payload, template_id=identifier, project_id=project.id, version=expected + 1
                )
                result = WorkTemplateService(unit.event_store).publish(
                    template,
                    expected_version=expected,
                    principal=actor,
                    storage=self.profiles.get(project.storage_profile),
                    roles=unit.acl.roles_for(project.id, actor.id),
                    gitlab_identity_verified=True,
                    idempotency_key=context.idempotency_key,
                )
                unit.commit()
                return result.to_dict()["template"]
        except Exception as exc:
            self._translate_planning_error(exc)
            raise

    def transport_work_plan_analytics(self, project_id, request, actor_id):
        from kraken_manager.infrastructure.work_plan_reports import planning_report
        actor = self._actor(actor_id)
        try:
            with self.uow_factory() as unit:
                project = self._planning_project(unit.projections, project_id)
                report = planning_report(unit, project, principal=actor,
                                        profile=self.profiles.get(project.storage_profile), request=request, performers=self.performer_store)
            gateway = getattr(self, "agent_gateway", None)
            report["current_compute_resources"] = {
                "admission": gateway.resource_status() if gateway is not None else {
                    "state": "unavailable", "admission_ready": False,
                },
                "host_observations": list(gateway.host_observations()) if gateway is not None else [],
                "time_basis": "current_observation_not_report_as_of",
            }
            return report
        except Exception as exc:
            self._translate_planning_error(exc)
            raise
