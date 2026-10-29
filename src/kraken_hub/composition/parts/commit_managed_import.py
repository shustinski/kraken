"""Import frame files into a local project."""
from kraken_manager.application.managed_import import commit_managed_import
from kraken_manager.infrastructure.filesystem import FilesystemEventStore
from kraken_manager.application.results import ManagedImportResult
from kraken_manager.infrastructure.import_content import FilesystemImportContentReader


class CommitManagedImportMixin:
    def commit_managed_import(self, *, principal, project, layer, representation, plan, idempotency_key):
        versions = commit_managed_import(
            principal=principal, project=project, layer=layer, representation=representation,
            plan=plan, idempotency_key=idempotency_key, projections=self._projection(project.id),
            events=FilesystemEventStore(self.catalog_root, str(project.id)),
            uow_factory=self._uow(str(project.id)), profiles=self.profiles, clock=self.clock,
            content_reader=FilesystemImportContentReader(),
        )
        return ManagedImportResult(plan, versions)
