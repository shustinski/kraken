"""Server adapter for physical model commands and their transactional event stream."""
from __future__ import annotations

from kraken_manager.application.multilayer import MultilayerService
from kraken_manager.infrastructure.registration import fit_affine
from .stitch_execution import StitchExecutionMixin


class MultilayerMixin(StitchExecutionMixin):
    def commit_stitch_publication(self, project_id, publication, source_path, context):
        """Internal queue-import adapter; the request is already project-bound."""
        from kraken_manager.application.stitch_results import import_stitch_result
        from kraken_manager.infrastructure.mosaic import MosaicReader

        def validate(path, plan, job_id):
            with MosaicReader(path) as reader:
                return reader.validate(expected_plan=plan, expected_job_id=job_id)

        try:
            with self.uow_factory() as unit:
                actor = self._actor(context.actor_id)
                service = MultilayerService(unit.event_store, unit.projections)
                project = service._project(project_id)
                model = import_stitch_result(unit, project_id=str(project_id), principal=actor,
                    storage=self.profiles.get(project.storage_profile), roles=unit.acl.roles_for(project.id, actor.id),
                    publication=publication, source_path=source_path, validate_mosaic=validate, verified=True)
                unit.commit()
                return service.describe(model)
        except Exception as exc:
            self._translate_multilayer_error(exc)
            raise

    def get_multilayer_model(self, project_id):
        try:
            service = MultilayerService(self.events, self.projections)
            return service.describe(service.load(str(project_id)))
        except Exception as exc:
            self._translate_multilayer_error(exc)
            raise

    def transport_mutate_multilayer_model(self, project_id, payload, context):
        try:
            if set(payload) - {"operation", "arguments"}:
                raise ValueError("Unknown multilayer command fields")
            with self.uow_factory() as unit:
                service = MultilayerService(unit.event_store, unit.projections, fit_affine)
                actor = self._actor(context.actor_id)
                project = service._project(project_id)
                model = service.mutate(
                    project_id=str(project_id), principal=actor, storage=self.profiles.get(project.storage_profile),
                    roles=unit.acl.roles_for(project.id, actor.id), expected_revision=context.expected_revision,
                    operation=payload["operation"], arguments=payload.get("arguments", {}),
                    idempotency_key=context.idempotency_key, gitlab_identity_verified=True,
                )
                unit.commit()
                result = service.describe(model)
            if payload["operation"] in {"request_layer_stitch", "cancel_layer_stitch"}:
                try:
                    self.dispatch_stitch_project(str(project_id))
                except Exception:
                    import logging
                    logging.getLogger(__name__).exception("Committed stitch intent awaits dispatch: %s", project_id)
            return result
        except Exception as exc:
            self._translate_multilayer_error(exc)
            raise

    def _translate_multilayer_error(self, exc):
        from kraken_manager.infrastructure.filesystem.event_store import EventStreamConflict
        from kraken_server.services import ConflictError, ValidationError

        if isinstance(exc, EventStreamConflict):
            raise ConflictError(str(exc)) from exc
        if isinstance(exc, (TypeError, KeyError)):
            raise ValidationError(str(exc)) from exc
        self._translate_lifecycle_error(exc)
