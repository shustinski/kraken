"""Local artifact-backed conductor evidence and explicit netlist export."""
from __future__ import annotations

from kraken_manager.application.electrical import ElectricalService
from kraken_manager.infrastructure.conductor_masks import extract_conductors
from kraken_manager.infrastructure.filesystem import FilesystemEventStore


class ElectricalMixin:
    def _electrical_reader(self, project_id):
        return ElectricalService(FilesystemEventStore(self.catalog_root, str(project_id)), self._projection(project_id))

    def get_electrical_model(self, project_id):
        service = self._electrical_reader(project_id)
        return service.describe(service.load(str(project_id)))

    def export_electrical_model(self, project_id):
        return self._electrical_reader(project_id).export(str(project_id))

    def mutate_electrical_model(self, *, project_id, principal, expected_revision, expected_physical_revision,
                                operation, arguments, idempotency_key):
        with self._uow(str(project_id))() as unit:
            service = ElectricalService(unit.event_store, unit.projections, extractor=extract_conductors, blob_store=unit.blobs)
            project = service.physical._project(project_id)
            model = service.mutate(project_id=str(project_id), principal=principal,
                storage=self.profiles.get(project.storage_profile), roles=unit.acl.roles_for(project.id, principal.id),
                expected_revision=expected_revision, expected_physical_revision=expected_physical_revision,
                operation=operation, arguments=arguments, idempotency_key=idempotency_key)
            unit.commit()
            return service.describe(model)
