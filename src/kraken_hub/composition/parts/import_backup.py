"""composition: import backup."""
from __future__ import annotations
import json
from pathlib import Path
from kraken_manager.application.authorization import AuthorizationPolicy
from kraken_manager.domain.common import (
    ProjectId,
)
from kraken_manager.domain.identity import (
    Permission,
    Principal,
    ProjectRole,
    ProjectRoleAssignment,
)
from kraken_manager.domain.project import (
    Project,
)
from kraken_manager.infrastructure.filesystem import (
    FileProjectLayout,
    FilesystemEventStore,
    SQLiteProjectionStore,
)
from kraken_manager.infrastructure.blob.filesystem import FilesystemBlobStore
from kraken_manager.infrastructure.migration import (
    BundleVerifier,
    CanonicalBundleExporter,
    CanonicalBundleImporter,
    KrakenMigrationBundleV1,
    load_bundle_manifest,
)
from kraken_manager.infrastructure.projections import rebuild_filesystem_index

from kraken_hub.composition.parts.frame_cell_snapshot import IntegrityScanResult


class ImportBackupMixin:
    def scan_integrity(self) -> IntegrityScanResult:
        event_count = 0
        blob_count = 0
        errors: list[str] = []
        projects = self.list_projects(include_archived=True)
        for project in projects:
            try:
                events = FilesystemEventStore(self.catalog_root, str(project.id))
                verified_events, _ = events.verify()
                event_count += verified_events
                blobs = FilesystemBlobStore.for_project(self.catalog_root, str(project.id))
                blob_count += len(blobs.verify_all())
                self._projection(project.id)
            except Exception as exc:
                errors.append(f"{project.name} ({project.id}): {exc}")
        return IntegrityScanResult(len(projects), event_count, blob_count, tuple(errors))

    def export_backup(
        self,
        project_id: ProjectId | str,
        destination: Path | str,
        *,
        principal: Principal | None = None,
    ) -> KrakenMigrationBundleV1:
        project = self.get_project(project_id)
        if project is None:
            raise ValueError("Project was not found")
        if principal is not None:
            AuthorizationPolicy().require(
                principal=principal,
                storage=self.profile,
                permission=Permission.MIGRATE_PROJECT,
                roles=self.project_roles(project.id, principal.id),
            )
        events = FilesystemEventStore(self.catalog_root, str(project.id))
        blobs = FilesystemBlobStore.for_project(self.catalog_root, str(project.id))
        return CanonicalBundleExporter(
            events, blobs, source_profile=self.profile.id
        ).export(destination)

    def import_backup(
        self,
        bundle_root: Path | str,
        *,
        principal: Principal | None = None,
        take_ownership: bool = False,
    ) -> Project:
        root = Path(bundle_root).resolve(strict=True)
        report = BundleVerifier().verify(root)
        report.raise_for_errors()
        manifest = load_bundle_manifest(root)
        available_owners = tuple(
            candidate
            for candidate in self.identities.list()
            if ProjectRole.MAINTAINER
            in self.project_roles(manifest.project_id, candidate.id)
        )
        if principal is not None:
            if available_owners:
                AuthorizationPolicy().require(
                    principal=principal,
                    storage=self.profile,
                    permission=Permission.MIGRATE_PROJECT,
                    roles=self.project_roles(manifest.project_id, principal.id),
                )
            elif not take_ownership:
                raise ValueError(
                    "В локальном хранилище нет доступного сопровождающего. "
                    "Для восстановления явно подтвердите принятие проекта."
                )
        events = FilesystemEventStore(self.catalog_root, manifest.project_id)
        blobs = FilesystemBlobStore.for_project(self.catalog_root, manifest.project_id)
        CanonicalBundleImporter(events, blobs).import_bundle(root, verify_first=False)
        rebuild_filesystem_index(events, SQLiteProjectionStore(self.catalog_root, manifest.project_id), acl=self.identities)
        project = self.get_project(manifest.project_id)
        if project is None:
            raise RuntimeError("Imported backup has no project projection")
        if principal is not None and not available_owners:
            self.identities.assign(
                ProjectRoleAssignment.create(
                    project_id=project.id,
                    principal_id=principal.id,
                    role=ProjectRole.MAINTAINER,
                    assigned_by=principal.id,
                    assigned_at=self.clock.now(),
                )
            )
        return project

    def attach_project(
        self,
        project_directory: Path | str,
        *,
        principal: Principal | None = None,
        take_ownership: bool = False,
    ) -> Project:
        source = Path(project_directory).resolve(strict=True)
        descriptor = json.loads((source / "project.json").read_text(encoding="utf-8"))
        project_id = str(descriptor["project_id"])
        expected = FileProjectLayout(self.catalog_root, project_id).project_dir.resolve()
        if source != expected:
            raise ValueError("v1 attaches only projects already located in the configured local catalog")
        project = self.get_project(project_id)
        if project is None:
            raise ValueError("Project cannot be reconstructed from its authoritative event log")
        if principal is not None:
            available_owners = tuple(
                candidate
                for candidate in self.identities.list()
                if ProjectRole.MAINTAINER in self.project_roles(project.id, candidate.id)
            )
            if available_owners:
                AuthorizationPolicy().require(
                    principal=principal,
                    storage=self.profile,
                    permission=Permission.MIGRATE_PROJECT,
                    roles=self.project_roles(project.id, principal.id),
                )
            elif not take_ownership:
                raise ValueError(
                    "В локальном хранилище нет доступного сопровождающего. "
                    "Для подключения явно подтвердите принятие проекта."
                )
            else:
                self.identities.assign(
                    ProjectRoleAssignment.create(
                        project_id=project.id,
                        principal_id=principal.id,
                        role=ProjectRole.MAINTAINER,
                        assigned_by=principal.id,
                        assigned_at=self.clock.now(),
                    )
                )
        return project
