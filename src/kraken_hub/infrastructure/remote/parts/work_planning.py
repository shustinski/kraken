"""Hub client for durable engineering plans."""

from __future__ import annotations

from kraken_manager.domain.planning import WorkPlan


def _plan(payload):
    return WorkPlan.from_dict({"schema_version": 1, "plan": payload})


class WorkPlanningMixin:
    def list_work_plans(self, project_id):
        payload = self._call("GET", f"/api/v1/projects/{project_id}/work-plans")
        return tuple(_plan(item) for item in payload["items"])

    def get_work_plan(self, project_id, plan_id):
        return _plan(self._call("GET", f"/api/v1/projects/{project_id}/work-plans/{plan_id}"))

    def create_work_plan(self, *, project_id, title, principal, idempotency_key):
        return _plan(
            self._call(
                "POST",
                f"/api/v1/projects/{project_id}/work-plans",
                payload={"title": title},
                idempotency_key=idempotency_key,
            )
        )

    def mutate_work_plan(
        self,
        *,
        project_id,
        plan_id,
        principal,
        expected_revision,
        operation,
        arguments,
        idempotency_key,
    ):
        return _plan(
            self._call(
                "POST",
                f"/api/v1/projects/{project_id}/work-plans/{plan_id}/commands",
                payload={"operation": operation, "arguments": arguments},
                idempotency_key=idempotency_key,
                if_match=expected_revision,
            )
        )

    def list_work_templates(self, project_id):
        from kraken_manager.domain.planning_templates import WorkPlanTemplate

        payload = self._call("GET", f"/api/v1/projects/{project_id}/work-plan-templates")
        return tuple(WorkPlanTemplate.from_dict({"schema_version": 1, "template": item}) for item in payload["items"])

    def get_work_template(self, project_id, template_id, *, version=None):
        from kraken_manager.domain.planning_templates import WorkPlanTemplate

        suffix = "" if version is None else f"?version={int(version)}"
        payload = self._call("GET", f"/api/v1/projects/{project_id}/work-plan-templates/{template_id}{suffix}")
        return WorkPlanTemplate.from_dict({"schema_version": 1, "template": payload})

    def save_work_template(
        self, *, project_id, principal, definition, idempotency_key, template_id=None, expected_version=0
    ):
        from kraken_manager.domain.planning_templates import WorkPlanTemplate

        base = f"/api/v1/projects/{project_id}/work-plan-templates"
        payload = self._call(
            "POST" if template_id is None else "PUT",
            base if template_id is None else f"{base}/{template_id}",
            payload=definition,
            idempotency_key=idempotency_key,
            if_match=expected_version,
        )
        return WorkPlanTemplate.from_dict({"schema_version": 1, "template": payload})

    def work_plan_analytics(self, project_id, *, principal, **request):
        from datetime import datetime
        payload = {key: value.isoformat() if isinstance(value, datetime) else value for key, value in request.items()}
        return self._call("POST", f"/api/v1/projects/{project_id}/work-planning/analytics", payload=payload)
