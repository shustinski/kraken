"""review_workflow: review return inspector."""
from __future__ import annotations
import json
from dataclasses import replace
from hashlib import sha256
from typing import Any
from kraken_manager.application.authorization import AuthorizationPolicy
from kraken_manager.application.dto import (
    ReturnedFileDigest,
    ReviewReturnPlan,
)
from kraken_manager.application.errors import ConflictError
from kraken_manager.application.ports import (
    Clock,
    ReviewPackageReader,
    StorageProfileCatalog,
    UnitOfWork,
    UnitOfWorkFactory,
)
from kraken_manager.application.use_cases import (
    ReviewReturnComparator,
)
from kraken_manager.domain.workflows import (
    ReviewBatch,
)

from kraken_manager.application.review_workflow_parts.review_handler import _ReviewHandler

class _ReviewReturnInspector(_ReviewHandler):
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        storage_profiles: StorageProfileCatalog,
        clock: Clock,
        reader: ReviewPackageReader,
        authorization: AuthorizationPolicy | None = None,
    ) -> None:
        super().__init__(uow_factory, storage_profiles, clock, authorization)
        self._reader = reader
        self._comparator = ReviewReturnComparator()

    def _scan_return_package(
        self, source: str
    ) -> tuple[Any, tuple[ReturnedFileDigest, ...], list[tuple[str, str | None, bool]]]:
        manifest = self._reader.read_manifest(source)
        ignored_paths = {
            item.relative_path.casefold() for item in manifest.files if item.role == "image"
        }
        returned: list[ReturnedFileDigest] = []
        fingerprint_rows: list[tuple[str, str | None, bool]] = []
        for relative_path in self._reader.list_relative_paths(source):
            try:
                digest = sha256()
                size = 0
                for chunk in self._reader.iter_file(source, relative_path):
                    if not isinstance(chunk, bytes):
                        raise TypeError("Review package readers must yield bytes")
                    digest.update(chunk)
                    size += len(chunk)
                value = digest.hexdigest()
                valid = True
            except (OSError, TypeError, ValueError):
                value = None
                size = None
                valid = False
            fingerprint_rows.append((relative_path.casefold(), value, valid))
            if relative_path.casefold() not in ignored_paths:
                returned.append(ReturnedFileDigest(relative_path, value, size, valid))
        return manifest, tuple(returned), fingerprint_rows

    def _inspect(
        self,
        uow: UnitOfWork,
        batch: ReviewBatch,
        source: str,
        prepared: tuple[Any, tuple[ReturnedFileDigest, ...], list[tuple[str, str | None, bool]]] | None = None,
    ) -> ReviewReturnPlan:
        prepared_scan = self._scan_return_package(source) if prepared is None else prepared
        manifest, returned, fingerprint_rows = prepared_scan
        manifest_batch_id = manifest.batch_id or manifest.package_id
        if (
            manifest_batch_id != batch.id
            or manifest.project_id != batch.project_id
            or manifest.layer_id != batch.layer_id
        ):
            raise ConflictError("Returned package does not belong to this review batch")
        vector_files = tuple(item for item in manifest.files if item.role == "vector")
        if not vector_files:
            raise ConflictError("Returned package has no vector files")
        vector_manifest = replace(manifest, files=vector_files)
        active_versions = {}
        for expected in vector_files:
            version = uow.projections.get_artifact_version(expected.artifact_version_id)
            series = None if version is None else uow.projections.get_artifact_series(version.series_id)
            active = None if series is None else uow.projections.get_active_artifact_version(series.id)
            # An absent active pointer is also a stale base, not permission to
            # silently treat the issued version as active.
            active_versions[expected.frame_id] = None if active is None else active.id
        report = self._comparator.compare(
            manifest=vector_manifest,
            returned_files=tuple(returned),
            active_versions=active_versions,
        )
        fingerprint_payload = json.dumps(
            sorted(fingerprint_rows), ensure_ascii=True, separators=(",", ":")
        ).encode("utf-8")
        return ReviewReturnPlan(
            batch_id=batch.id,
            package_id=manifest.package_id,
            report=report,
            return_fingerprint=sha256(fingerprint_payload).hexdigest(),
        )
