"""Transport for executable layer graph definitions and commands."""


class PipelineExecutionMixin:
    def get_layer_pipeline(self, project_id, layer_id):
        return self._call("GET", f"/api/v1/projects/{project_id}/layers/{layer_id}/pipeline/workflow")

    def define_layer_pipeline(self, *, project_id, layer_id, definition, principal, expected_revision, idempotency_key):
        return self._call(
            "PUT",
            f"/api/v1/projects/{project_id}/layers/{layer_id}/pipeline/workflow",
            payload=definition,
            if_match=expected_revision,
            idempotency_key=idempotency_key,
        )

    def command_layer_pipeline(
        self, *, project_id, layer_id, operation, principal, expected_revision, idempotency_key, node_id=None
    ):
        return self._call(
            "POST",
            f"/api/v1/projects/{project_id}/layers/{layer_id}/pipeline/workflow/commands",
            payload={"operation": operation, "node_id": node_id},
            if_match=expected_revision,
            idempotency_key=idempotency_key,
        )
