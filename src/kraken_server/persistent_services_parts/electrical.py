"""Persistent Server transport for reviewed conductor/terminal evidence."""
from __future__ import annotations

from kraken_manager.application.electrical import ElectricalService
from kraken_manager.infrastructure.conductor_masks import extract_conductors


class ElectricalMixin:
    def get_electrical_model(self, project_id):
        try:
            service = ElectricalService(self.events, self.projections)
            return service.describe(service.load(str(project_id)))
        except Exception as exc:
            self._translate_multilayer_error(exc)
            raise

    def export_electrical_model(self, project_id):
        try:
            return ElectricalService(self.events, self.projections).export(str(project_id))
        except Exception as exc:
            self._translate_multilayer_error(exc)
            raise

    def transport_mutate_electrical_model(self, project_id, payload, context):
        try:
            if set(payload) - {"operation", "arguments", "expected_physical_revision"}:
                raise ValueError("unknown electrical command fields")
            with self.uow_factory() as unit:
                service = ElectricalService(unit.event_store, unit.projections, extractor=extract_conductors, blob_store=unit.blobs)
                actor = self._actor(context.actor_id)
                project = service.physical._project(project_id)
                model = service.mutate(project_id=str(project_id), principal=actor,
                    storage=self.profiles.get(project.storage_profile), roles=unit.acl.roles_for(project.id, actor.id),
                    expected_revision=context.expected_revision, expected_physical_revision=payload["expected_physical_revision"],
                    operation=payload["operation"], arguments=payload.get("arguments", {}),
                    idempotency_key=context.idempotency_key, gitlab_identity_verified=True)
                unit.commit()
                return service.describe(model)
        except Exception as exc:
            self._translate_multilayer_error(exc)
            raise
