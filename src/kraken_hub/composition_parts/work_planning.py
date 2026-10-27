"""Local Hub engineering planning, persisted in the project event log."""

from __future__ import annotations

from kraken_manager.application.errors import NotFoundError
from kraken_manager.application.planning import WorkPlanningService
from kraken_manager.domain.common import validate_uuid
from kraken_manager.domain.planning import WorkPlan
from kraken_manager.domain.project import ProjectState
from kraken_manager.infrastructure.filesystem import FilesystemEventStore
from kraken_manager.infrastructure.planning_commands import command_id, decode_arguments, list_plans


class WorkPlanningMixin:
    def create_work_plan(self, *, project_id, title, principal, idempotency_key):
        with self._uow(str(project_id))() as unit:
            project = self._planning_project(unit.projections, project_id, mutate=True)
            plan = WorkPlan.create(
                project.id, title, at=self.clock.now(), plan_id=command_id(project.id, idempotency_key, "plan")
            )
            result = WorkPlanningService(unit.event_store, projections=unit.projections).create(
                plan,
                principal=principal,
                storage=self.profiles.get(project.storage_profile),
                roles=unit.acl.roles_for(project.id, principal.id),
                idempotency_key=idempotency_key,
            )
            unit.commit()
            return result

    def list_work_plans(self, project_id):
        self._planning_project(self._projection(project_id), project_id)
        return list_plans(FilesystemEventStore(self.catalog_root, str(project_id)), str(project_id))

    def get_work_plan(self, project_id, plan_id):
        self._planning_project(self._projection(project_id), project_id)
        plan = WorkPlanningService(FilesystemEventStore(self.catalog_root, str(project_id))).load(str(plan_id))
        if plan.project_id != str(project_id):
            raise NotFoundError("Plan does not belong to the project")
        return plan

    def mutate_work_plan(
        self, *, project_id, plan_id, principal, expected_revision, operation, arguments, idempotency_key
    ):
        with self._uow(str(project_id))() as unit:
            project = self._planning_project(unit.projections, project_id, mutate=True)
            service = WorkPlanningService(unit.event_store, projections=unit.projections)
            plan = service.load(str(plan_id))
            if plan.project_id != str(project.id):
                raise NotFoundError("Plan does not belong to the project")
            args = decode_arguments(
                operation,
                arguments,
                unit.projections,
                project,
                self.performers,
                idempotency_key,
                plan=plan,
                events=unit.event_store,
            )
            result = service.execute(
                str(plan.id),
                expected_revision=expected_revision,
                principal=principal,
                storage=self.profiles.get(project.storage_profile),
                roles=unit.acl.roles_for(project.id, principal.id),
                operation=operation,
                arguments=args,
                idempotency_key=idempotency_key,
            )
            unit.commit()
            return result

    @staticmethod
    def _planning_project(projections, project_id, mutate=False):
        project = projections.get_project(validate_uuid(str(project_id)))
        if project is None:
            raise NotFoundError("Project was not found")
        if mutate and project.state is not ProjectState.ACTIVE:
            from kraken_manager.application.errors import ConflictError

            raise ConflictError("Archived projects cannot change work plans")
        return project

    def list_work_templates(self, project_id):
        from kraken_manager.infrastructure.planning_templates import list_templates

        self._planning_project(self._projection(project_id), project_id)
        return list_templates(FilesystemEventStore(self.catalog_root, str(project_id)), str(project_id))

    def get_work_template(self, project_id, template_id, *, version=None):
        from kraken_manager.application.planning_templates import WorkTemplateService

        self._planning_project(self._projection(project_id), project_id)
        return WorkTemplateService(FilesystemEventStore(self.catalog_root, str(project_id))).load(
            str(project_id), str(template_id), version=version
        )

    def save_work_template(
        self, *, project_id, principal, definition, idempotency_key, template_id=None, expected_version=0
    ):
        from kraken_manager.application.planning_templates import WorkTemplateService
        from kraken_manager.infrastructure.planning_templates import decode_template_definition

        with self._uow(str(project_id))() as unit:
            project = self._planning_project(unit.projections, project_id, mutate=True)
            identifier = template_id or command_id(project.id, idempotency_key, "template")
            template = decode_template_definition(
                definition, template_id=identifier, project_id=project.id, version=expected_version + 1
            )
            result = WorkTemplateService(unit.event_store).publish(
                template,
                expected_version=expected_version,
                principal=principal,
                storage=self.profiles.get(project.storage_profile),
                roles=unit.acl.roles_for(project.id, principal.id),
                idempotency_key=idempotency_key,
            )
            unit.commit()
            return result

    def work_plan_analytics(self, project_id, *, principal, **request):
        from kraken_manager.infrastructure.work_plan_reports import planning_report
        with self._uow(str(project_id))() as unit:
            project = self._planning_project(unit.projections, project_id)
            report = planning_report(unit, project, principal=principal,
                                    profile=self.profiles.get(project.storage_profile), request=request, performers=self.performers)
        from kraken_agent.telemetry import sample_host
        report["current_compute_resources"] = {
            "admission": {"state": "unenforced", "admission_ready": False, "scope": "local_agent",
                          "os_enforced": False, "per_host_capacity_enforced": False},
            "host_observations": [{"sample": sample_host(), "stale": False, "age_seconds": 0}],
            "time_basis": "current_observation_not_report_as_of",
        }
        return report
