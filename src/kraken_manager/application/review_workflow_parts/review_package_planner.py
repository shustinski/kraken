"""review_workflow: review package planner."""
from __future__ import annotations
from collections.abc import Callable, Iterator
from kraken_manager.application.dto import (
    ReviewPackagePlan,
)
from kraken_manager.application.errors import ConflictError
from kraken_manager.application.ports import (
    UnitOfWork,
)
from kraken_manager.domain.common import ReviewBatchId
from kraken_manager.domain.workflows import (
    ReviewBatch,
    ReviewPackageFileV1,
    ReviewPackageManifestV1,
)

from kraken_manager.application.review_workflow_parts.review_handler import _ReviewHandler

class _ReviewPackagePlanner(_ReviewHandler):
    def _build_plan(
        self,
        uow: UnitOfWork,
        batch: ReviewBatch,
        *,
        package_id: ReviewBatchId,
        issued_by,
        include_images: bool,
        issued_at,
    ) -> tuple[ReviewPackagePlan, dict[str, Callable[[], Iterator[bytes]]]]:
        package_files: list[ReviewPackageFileV1] = []
        readers: dict[str, Callable[[], Iterator[bytes]]] = {}
        issues: list[str] = []
        total_size = 0
        coordinates = {
            coordinate.frame_id(batch.project_id): coordinate
            for coordinate in batch.selection.iter_coordinates()
        }
        for item in batch.items:
            coordinate = coordinates[item.frame_id]
            roles = (("vector", item.vector_version_id),)
            if include_images and item.image_version_id is not None:
                roles += (("image", item.image_version_id),)
            for role, version_id in roles:
                version = uow.projections.get_artifact_version(version_id)
                if version is None:
                    issues.append(f"Missing {role} version {version_id}")
                    continue
                if role == "vector" and version.sha256 != item.vector_sha256:
                    issues.append(f"Vector hash snapshot differs for frame {item.frame_id}")
                    continue
                if role == "vector":
                    series = uow.projections.get_artifact_series(version.series_id)
                    active = (
                        None
                        if series is None
                        else uow.projections.get_active_artifact_version(series.id)
                    )
                    if active is None or active.id != version.id:
                        issues.append(f"Vector base is no longer active for frame {item.frame_id}")
                        continue
                if version.blob is None:
                    issues.append(f"External {role} file cannot be placed into a review package: {version.filename}")
                    continue
                if not uow.blobs.exists(version.blob):
                    issues.append(f"Managed blob is missing for {role} file {version.filename}")
                    continue
                relative_path = f"{role}s/{item.frame_id}_{version.filename}"
                package_files.append(
                    ReviewPackageFileV1(
                        frame_id=item.frame_id,
                        artifact_version_id=version.id,
                        sha256=version.sha256,
                        relative_path=relative_path,
                        x=coordinate.x,
                        y=coordinate.y,
                        role=role,
                    )
                )
                readers[relative_path] = lambda blob=version.blob, store=uow.blobs: store.iter_bytes(blob)
                total_size += version.size_bytes
        if not package_files:
            raise ConflictError("Review package has no exportable files")
        manifest = ReviewPackageManifestV1(
            package_id=package_id,
            batch_id=batch.id,
            project_id=batch.project_id,
            layer_id=batch.layer_id,
            issued_at=issued_at,
            performer_id=batch.assignee_id,
            issued_by=issued_by,
            due_at=batch.due_at,
            instructions=batch.instructions,
            files=tuple(package_files),
        )
        return ReviewPackagePlan(manifest, total_size, tuple(issues)), readers
