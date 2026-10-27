"""Execute the layer graph through the existing local Agent handlers."""

from kraken_manager.application.plugin_jobs import SubmitPluginJobHandler, CancelPluginJobHandler
from kraken_manager.infrastructure.pipeline_runtime import PipelineRuntime


class PipelineExecutionMixin:
    def _layer_runtime(self, project_id, gateway=None, stitch_gateway=None):
        factory = self._uow(str(project_id))
        from kraken_manager.application.errors import ConflictError
        from kraken_manager.application.stitch_results import fail_stitch_result
        from kraken_manager.infrastructure.plugin.stitch_gateway import AgentStitchGateway
        if stitch_gateway is None and gateway is not None and all(hasattr(gateway, key) for key in ("base_url", "token", "staging_root")):
            stitch_gateway = AgentStitchGateway(base_url=gateway.base_url, token=gateway.token, staging_root=gateway.staging_root)

        def run_stitch(project_id, job_id, principal, *, dispatch=False):
            try:
                method = self.dispatch_stitch if dispatch else self.refresh_stitch
                return method(project_id=project_id, job_id=job_id, principal=principal, gateway=stitch_gateway)
            except ConflictError as exc:
                with factory() as unit:
                    project = unit.projections.get_project(project_id)
                    fail_stitch_result(unit, project_id=project_id, job_id=job_id, principal=principal,
                        storage=self.profiles.get(project.storage_profile), roles=unit.acl.roles_for(project_id, principal.id), error=exc)
                    unit.commit()
        return PipelineRuntime(
            factory,
            self.profiles,
            self.clock,
            None if gateway is None else SubmitPluginJobHandler(factory, self.profiles, self.clock, gateway),
            None if gateway is None else CancelPluginJobHandler(factory, self.profiles, self.clock, gateway),
            stitch_dispatch=None if stitch_gateway is None else lambda p, j, a: run_stitch(p, j, a, dispatch=True),
            stitch_refresh=None if stitch_gateway is None else run_stitch,
            stitch_cancel=None if stitch_gateway is None else run_stitch,
        )

    def get_layer_pipeline(self, project_id, layer_id):
        return self._layer_runtime(project_id).get(str(project_id), str(layer_id))

    def define_layer_pipeline(self, *, project_id, layer_id, definition, principal, expected_revision, idempotency_key):
        return self._layer_runtime(project_id).define(
            str(project_id),
            str(layer_id),
            definition,
            principal=principal,
            expected_revision=expected_revision,
            idempotency_key=idempotency_key,
        )

    def command_layer_pipeline(
        self, *, project_id, layer_id, operation, principal, expected_revision, idempotency_key, node_id=None
    ):
        return self._layer_runtime(project_id).command(
            str(project_id),
            str(layer_id),
            operation,
            principal=principal,
            expected_revision=expected_revision,
            idempotency_key=idempotency_key,
            node_id=node_id,
        )

    def tick_layer_pipeline(self, *, project_id, layer_id, principal, gateway=None, stitch_gateway=None):
        return self._layer_runtime(project_id, gateway, stitch_gateway).tick(str(project_id), str(layer_id), principal=principal)
