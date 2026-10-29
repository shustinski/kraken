"""Local durable physical model; existing stitching artifacts remain untouched."""
from __future__ import annotations

from kraken_manager.application.multilayer import MultilayerService
from kraken_manager.infrastructure.filesystem import FilesystemEventStore
from kraken_manager.infrastructure.registration import fit_affine
from .stitch_computations import StitchComputationsMixin


class MultilayerMixin(StitchComputationsMixin):
    def get_multilayer_model(self, project_id):
        service = MultilayerService(FilesystemEventStore(self.catalog_root, str(project_id)), self._projection(project_id))
        return service.describe(service.load(str(project_id)))

    def mutate_multilayer_model(self, *, project_id, principal, expected_revision, operation, arguments, idempotency_key):
        with self._uow(str(project_id))() as unit:
            service = MultilayerService(unit.event_store, unit.projections, fit_affine)
            project = service._project(project_id)
            model = service.mutate(
                project_id=str(project_id), principal=principal, storage=self.profiles.get(project.storage_profile),
                roles=unit.acl.roles_for(project.id, principal.id), expected_revision=expected_revision,
                operation=operation, arguments=arguments, idempotency_key=idempotency_key,
            )
            unit.commit()
            return service.describe(model)
