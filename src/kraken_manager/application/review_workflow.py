"""Application orchestration for signed review packages and immutable returns."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import replace
from hashlib import sha256
import json

from kraken_manager.application.dto import (
    AcceptReviewCommand,
    CancelReviewBatchCommand,
    CommitReviewReturnCommand,
    CreateReviewBatchCommand,
    DryRunReviewReturnCommand,
    ExportReviewPackageCommand,
    PlanReviewPackageCommand,
    RequestReviewChangesCommand,
    ReturnedFileDigest,
    ReviewPackagePlan,
    ReviewReturnCommitResult,
    ReviewReturnPlan,
)
from kraken_manager.application.authorization import AuthorizationPolicy
from kraken_manager.application.errors import ConcurrencyError, ConflictError, NotFoundError
from kraken_manager.application.ports import (
    Clock,
    PerformerStore,
    ReviewPackageReader,
    ReviewPackageWriter,
    StorageProfileCatalog,
    UnitOfWork,
    UnitOfWorkFactory,
)
from kraken_manager.application.use_cases import (
    ReviewReturnComparator,
    _ProjectHandler,
    _layer_stream,
    _prior_entity_id,
    _series_stream,
)
from kraken_manager.domain.artifacts import ArtifactVersion
from kraken_manager.domain.common import ArtifactVersionId, ReviewBatchId, require_non_empty
from kraken_manager.domain.events import ActorSnapshot, EventEnvelope
from kraken_manager.domain.identity import Permission
from kraken_manager.domain.project import RepresentationKind, StructureState
from kraken_manager.domain.workflows import (
    ReviewBatch,
    ReviewBatchState,
    ReviewComparisonReport,
    ReviewFileComparison,
    ReviewFileStatus,
    ReviewItem,
    ReviewPackageFileV1,
    ReviewPackageManifestV1,
)


def _review_stream(batch_id: ReviewBatchId | str) -> str:
    return f"review-batch:{batch_id}"


def _prior_command_event(
    uow: UnitOfWork,
    *,
    project_id,
    idempotency_key: str,
    event_type: str,
    batch_id: ReviewBatchId | None = None,
) -> EventEnvelope | None:
    events = uow.event_store.find_by_idempotency_key(project_id, idempotency_key)
    matches = [
        event
        for event in events
        if event.event_type == event_type
        and (batch_id is None or str(event.payload.get("review_batch_id")) == str(batch_id))
    ]
    if matches:
        return matches[-1]
    if events:
        raise ConflictError("Idempotency key was already used by another command")
    return None


def _review_item_payload(item: ReviewItem) -> dict[str, object]:
    return {
        "frame_id": item.frame_id,
        "vector_version_id": item.vector_version_id,
        "vector_sha256": item.vector_sha256,
        "image_version_id": item.image_version_id,
    }


def _batch_payload(batch: ReviewBatch) -> dict[str, object]:
    return {
        "review_batch_id": batch.id,
        "layer_id": batch.layer_id,
        "selection": batch.selection.to_dict(),
        "items": [_review_item_payload(item) for item in batch.items],
        "assignee_id": batch.assignee_id,
        "created_by": batch.created_by,
        "instructions": batch.instructions,
        "state": batch.state.value,
        "batch_revision": batch.revision,
        "created_at": batch.created_at.isoformat(),
        "updated_at": batch.updated_at.isoformat(),
        "due_at": None if batch.due_at is None else batch.due_at.isoformat(),
    }


def _comparison_payload(comparison: ReviewFileComparison) -> dict[str, object]:
    return {
        "status": comparison.status.value,
        "relative_path": comparison.relative_path,
        "frame_id": comparison.frame_id,
        "expected_version_id": comparison.expected_version_id,
        "expected_sha256": comparison.expected_sha256,
        "returned_sha256": comparison.returned_sha256,
    }


def _report_from_payload(payload: object) -> ReviewComparisonReport:
    if not isinstance(payload, (tuple, list)):
        raise ConflictError("Stored review return has an invalid comparison report")
    return ReviewComparisonReport(
        tuple(
            ReviewFileComparison(
                status=ReviewFileStatus(str(item["status"])),
                relative_path=str(item["relative_path"]),
                frame_id=None if item.get("frame_id") is None else str(item["frame_id"]),
                expected_version_id=None
                if item.get("expected_version_id") is None
                else ArtifactVersionId(str(item["expected_version_id"])),
                expected_sha256=None if item.get("expected_sha256") is None else str(item["expected_sha256"]),
                returned_sha256=None if item.get("returned_sha256") is None else str(item["returned_sha256"]),
            )
            for item in payload
            if isinstance(item, Mapping)
        )
    )


def _artifact_created_payload(version: ArtifactVersion, *, activated: bool, candidate: bool) -> dict[str, object]:
    if version.blob is None:
        raise ConflictError("A managed artifact version is required")
    return {
        "artifact_version_id": version.id,
        "series_id": version.series_id,
        "sha256": version.sha256,
        "size_bytes": version.size_bytes,
        "media_type": version.media_type,
        "filename": version.filename,
        "blob": {"sha256": version.blob.sha256, "size_bytes": version.blob.size_bytes},
        "parent_version_id": version.parent_version_id,
        "input_version_ids": version.input_version_ids,
        "tool_name": version.tool_name,
        "tool_version": version.tool_version,
        "parameters": version.parameters,
        "author_principal_id": version.author_principal_id,
        "created_at": version.created_at.isoformat(),
        "activated": activated,
        "candidate": candidate,
    }


class _ReviewHandler(_ProjectHandler):
    def _load_batch(
        self,
        uow: UnitOfWork,
        *,
        project_id,
        batch_id: ReviewBatchId,
        context,
        permission: Permission,
    ) -> ReviewBatch:
        project, _ = self._load_project_and_authorize(
            uow,
            project_id=project_id,
            actor=context.actor,
            permission=permission,
            gitlab_identity_verified=context.gitlab_identity_verified,
        )
        batch = uow.projections.get_review_batch(batch_id)
        if batch is None or batch.project_id != project.id:
            raise NotFoundError("Review batch was not found in the project")
        return batch

    @staticmethod
    def _require_revision(uow: UnitOfWork, batch: ReviewBatch, expected: int) -> None:
        if batch.revision != expected:
            raise ConcurrencyError(f"Expected review batch revision {expected}, found {batch.revision}")
        stream_revision = uow.event_store.current_revision(_review_stream(batch.id))
        if stream_revision != batch.revision:
            raise ConcurrencyError(
                f"Review batch projection is at revision {batch.revision}, stream is at {stream_revision}"
            )


class CreateReviewBatchHandler(_ReviewHandler):
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        storage_profiles: StorageProfileCatalog,
        clock: Clock,
        performer_store: PerformerStore,
        *,
        authorization: AuthorizationPolicy | None = None,
    ) -> None:
        super().__init__(uow_factory, storage_profiles, clock, authorization)
        self._performers = performer_store

    def __call__(self, command: CreateReviewBatchCommand) -> ReviewBatch:
        with self._uow_factory() as uow:
            project, _ = self._load_project_and_authorize(
                uow,
                project_id=command.project_id,
                actor=command.context.actor,
                permission=Permission.MANAGE_REVIEW,
                gitlab_identity_verified=command.context.gitlab_identity_verified,
            )
            previous_id = _prior_entity_id(
                uow,
                project.id,
                command.context.idempotency_key,
                "ReviewBatchCreated",
                "review_batch_id",
            )
            if previous_id is not None:
                previous = uow.projections.get_review_batch(ReviewBatchId(previous_id))
                if previous is not None:
                    return previous
            _prior_command_event(
                uow,
                project_id=project.id,
                idempotency_key=command.context.idempotency_key,
                event_type="ReviewBatchCreated",
            )
            layer = uow.projections.get_layer(command.layer_id)
            if layer is None or layer.project_id != project.id:
                raise NotFoundError("Layer was not found in the project")
            if layer.state is StructureState.ARCHIVED:
                raise ConflictError("Archived layer is read-only")
            if layer.revision != command.expected_layer_revision:
                raise ConcurrencyError(
                    f"Expected layer revision {command.expected_layer_revision}, found {layer.revision}"
                )
            command.selection.validate_bounds(width=project.width, height=project.height)
            now = self._clock.now()
            batch = ReviewBatch.create(
                project_id=project.id,
                layer_id=layer.id,
                selection=command.selection,
                items=tuple(command.items),
                assignee_id=command.assignee_id,
                created_by=command.context.actor.id,
                instructions=command.instructions,
                due_at=command.due_at,
                batch_id=command.batch_id,
                created_at=now,
            )
            performer = self._performers.get(batch.assignee_id)
            if performer is None or not performer.active:
                raise ConflictError("Review assignee is missing or archived")

            if batch.selection.cardinality() != len(batch.items):
                raise ConflictError("Every selected frame must have exactly one vector version")
            selected_ids = {
                coordinate.frame_id(project.id) for coordinate in batch.selection.iter_coordinates()
            }
            if selected_ids != {item.frame_id for item in batch.items}:
                raise ConflictError("Review items do not exactly match the immutable frame selection")

            for item in batch.items:
                self._validate_item(uow, batch, item)
            requested = {(item.frame_id, item.vector_version_id) for item in batch.items}
            for active in uow.projections.list_active_review_batches(project.id, layer.id):
                occupied = {(item.frame_id, item.vector_version_id) for item in active.items}
                if requested & occupied:
                    raise ConflictError("A selected vector version is already in an active review batch")

            next_layer = replace(layer, revision=layer.revision + 1)
            event = EventEnvelope.create(
                stream_id=_layer_stream(str(layer.id)),
                project_id=project.id,
                revision=next_layer.revision,
                event_type="ReviewBatchCreated",
                payload=_batch_payload(batch),
                actor=ActorSnapshot.from_principal(command.context.actor),
                recorded_at=now,
                effective_at=command.context.effective_at,
                performer_id=command.context.performer_id,
                correlation_id=command.context.correlation_id,
                idempotency_key=command.context.idempotency_key,
            )
            uow.event_store.append(
                _layer_stream(str(layer.id)), expected_revision=layer.revision, events=(event,)
            )
            uow.projections.save_review_batch(batch)
            uow.projections.save_layer(next_layer)
            uow.commit()
            return batch

    @staticmethod
    def _validate_item(uow: UnitOfWork, batch: ReviewBatch, item: ReviewItem) -> None:
        vector = uow.projections.get_artifact_version(item.vector_version_id)
        if vector is None or vector.sha256 != item.vector_sha256:
            raise ConflictError("Review vector version is missing or its hash snapshot differs")
        series = uow.projections.get_artifact_series(vector.series_id)
        if (
            series is None
            or series.archived
            or series.project_id != batch.project_id
            or series.layer_id != batch.layer_id
            or series.frame_id != item.frame_id
        ):
            raise ConflictError("Review vector does not belong to the selected layer and frame")
        representation = uow.projections.get_representation(series.representation_id)
        if (
            representation is None
            or representation.kind is not RepresentationKind.VECTOR
            or representation.state is StructureState.ARCHIVED
        ):
            raise ConflictError("Review item must reference a vector representation")
        if item.image_version_id is None:
            return
        image = uow.projections.get_artifact_version(item.image_version_id)
        image_series = None if image is None else uow.projections.get_artifact_series(image.series_id)
        image_representation = (
            None
            if image_series is None
            else uow.projections.get_representation(image_series.representation_id)
        )
        if (
            image is None
            or image_series is None
            or image_series.archived
            or image_series.project_id != batch.project_id
            or image_series.layer_id != batch.layer_id
            or image_series.frame_id != item.frame_id
            or image_representation is None
            or image_representation.kind is not RepresentationKind.IMAGE
            or image_representation.state is StructureState.ARCHIVED
        ):
            raise ConflictError("Review image does not belong to the selected layer and frame")


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
                readers[relative_path] = lambda blob=version.blob: uow.blobs.iter_bytes(blob)
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


class PlanReviewPackageHandler(_ReviewPackagePlanner):
    def __call__(self, command: PlanReviewPackageCommand) -> ReviewPackagePlan:
        with self._uow_factory() as uow:
            batch = self._load_batch(
                uow,
                project_id=command.project_id,
                batch_id=command.batch_id,
                context=command.context,
                permission=Permission.MANAGE_REVIEW,
            )
            self._require_revision(uow, batch, command.expected_batch_revision)
            if batch.state not in {ReviewBatchState.DRAFT, ReviewBatchState.CHANGES_REQUESTED}:
                raise ConflictError("Only a draft or changes-requested batch can be exported")
            plan, _ = self._build_plan(
                uow,
                batch,
                package_id=command.package_id,
                issued_by=command.context.actor.id,
                include_images=command.include_images,
                issued_at=self._clock.now(),
            )
            return plan


class ExportReviewPackageHandler(_ReviewPackagePlanner):
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        storage_profiles: StorageProfileCatalog,
        clock: Clock,
        writer: ReviewPackageWriter,
        authorization: AuthorizationPolicy | None = None,
    ) -> None:
        super().__init__(uow_factory, storage_profiles, clock, authorization)
        self._writer = writer

    def __call__(self, command: ExportReviewPackageCommand) -> ReviewBatch:
        with self._uow_factory() as uow:
            batch = self._load_batch(
                uow,
                project_id=command.project_id,
                batch_id=command.batch_id,
                context=command.context,
                permission=Permission.MANAGE_REVIEW,
            )
            prior_events = uow.event_store.find_by_idempotency_key(
                batch.project_id,
                command.context.idempotency_key,
            )
            if any(
                event.event_type in {"ReviewBatchIssued", "ReviewBatchReexported"}
                and str(event.payload.get("review_batch_id")) == str(batch.id)
                for event in prior_events
            ):
                return batch
            if prior_events:
                raise ConflictError("Idempotency key was already used by another command")
            self._require_revision(uow, batch, command.expected_batch_revision)
            if batch.state in {ReviewBatchState.COMPLETED, ReviewBatchState.CANCELLED}:
                raise ConflictError("A closed review batch cannot be exported")
            now = self._clock.now()
            plan, readers = self._build_plan(
                uow,
                batch,
                package_id=command.package_id,
                issued_by=command.context.actor.id,
                include_images=command.include_images,
                issued_at=now,
            )
            if not plan.can_export:
                raise ConflictError("Review package preflight failed: " + "; ".join(plan.issues))

            # This side effect deliberately happens before the domain state
            # changes. A failed/partial writer must never mark frames issued.
            self._writer.write(command.destination, plan.manifest, readers)
            first_issue = batch.state in {
                ReviewBatchState.DRAFT,
                ReviewBatchState.CHANGES_REQUESTED,
            }
            issued = (
                batch.issue(at=now)
                if first_issue
                else replace(
                    batch,
                    revision=batch.revision + 1,
                    updated_at=now,
                )
            )
            event_type = "ReviewBatchIssued" if first_issue else "ReviewBatchReexported"
            event = EventEnvelope.create(
                stream_id=_review_stream(batch.id),
                project_id=batch.project_id,
                revision=issued.revision,
                event_type=event_type,
                payload={
                    "review_batch_id": batch.id,
                    "package_id": plan.manifest.package_id,
                    "state": issued.state.value,
                    "batch_revision": issued.revision,
                    "updated_at": issued.updated_at.isoformat(),
                    "file_count": len(plan.manifest.files),
                    "total_size_bytes": plan.total_size_bytes,
                },
                actor=ActorSnapshot.from_principal(command.context.actor),
                recorded_at=now,
                effective_at=command.context.effective_at,
                performer_id=batch.assignee_id,
                correlation_id=command.context.correlation_id,
                idempotency_key=command.context.idempotency_key,
            )
            uow.event_store.append(
                _review_stream(batch.id), expected_revision=batch.revision, events=(event,)
            )
            uow.projections.save_review_batch(issued)
            uow.commit()
            return issued


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

    def _inspect(self, uow: UnitOfWork, batch: ReviewBatch, source: str) -> ReviewReturnPlan:
        manifest = self._reader.read_manifest(source)
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


class DryRunReviewReturnHandler(_ReviewReturnInspector):
    def __call__(self, command: DryRunReviewReturnCommand) -> ReviewReturnPlan:
        with self._uow_factory() as uow:
            batch = self._load_batch(
                uow,
                project_id=command.project_id,
                batch_id=command.batch_id,
                context=command.context,
                permission=Permission.RETURN_REVIEW,
            )
            self._require_revision(uow, batch, command.expected_batch_revision)
            if batch.state not in {ReviewBatchState.ISSUED, ReviewBatchState.PARTIALLY_RETURNED}:
                raise ConflictError("Only an issued review batch can receive returned files")
            return self._inspect(uow, batch, command.source)


class CommitReviewReturnHandler(_ReviewReturnInspector):
    def __call__(self, command: CommitReviewReturnCommand) -> ReviewReturnCommitResult:
        with self._uow_factory() as uow:
            batch = self._load_batch(
                uow,
                project_id=command.project_id,
                batch_id=command.batch_id,
                context=command.context,
                permission=Permission.RETURN_REVIEW,
            )
            committed = _prior_command_event(
                uow,
                project_id=batch.project_id,
                idempotency_key=command.context.idempotency_key,
                event_type="ReviewReturnCommitted",
                batch_id=batch.id,
            )
            if committed is not None:
                return self._result_from_event(uow, batch, committed)

            plan = self._inspect(uow, batch, command.source)
            review_events = uow.event_store.load_stream(_review_stream(batch.id))
            for event in reversed(review_events):
                if (
                    event.event_type == "ReviewReturnCommitted"
                    and event.payload.get("return_fingerprint") == plan.return_fingerprint
                    and str(event.payload.get("package_id")) == str(plan.package_id)
                ):
                    return self._result_from_event(uow, batch, event)

            self._require_revision(uow, batch, command.expected_batch_revision)
            if batch.state not in {ReviewBatchState.ISSUED, ReviewBatchState.PARTIALLY_RETURNED}:
                raise ConflictError("Only an issued review batch can receive returned files")
            if not plan.report.can_commit:
                raise ConflictError("Duplicate or invalid returned files must be resolved before commit")

            manifest = self._reader.read_manifest(command.source)
            if (
                manifest.package_id != plan.package_id
                or (manifest.batch_id or manifest.package_id) != batch.id
                or manifest.project_id != batch.project_id
                or manifest.layer_id != batch.layer_id
            ):
                raise ConflictError("Review manifest changed during return preflight")
            expected_by_path = {
                item.relative_path.casefold(): item for item in manifest.files if item.role == "vector"
            }
            prior_candidates: dict[tuple[str, str], ArtifactVersionId] = {}
            for review_event in review_events:
                if review_event.event_type != "ReviewReturnCommitted":
                    continue
                candidate_files = review_event.payload.get("candidate_files", ())
                if isinstance(candidate_files, (tuple, list)):
                    for candidate_file in candidate_files:
                        if not isinstance(candidate_file, Mapping):
                            continue
                        path = str(candidate_file.get("relative_path", "")).casefold()
                        digest = str(candidate_file.get("returned_sha256", ""))
                        version_id = candidate_file.get("version_id")
                        if path and digest and version_id is not None:
                            prior_candidates[(path, digest)] = ArtifactVersionId(str(version_id))
            now = self._clock.now()
            candidates: list[ArtifactVersion] = []
            candidate_files: list[dict[str, object]] = []
            quarantined_extras: list[dict[str, object]] = []
            for comparison in plan.report.items:
                if comparison.status is not ReviewFileStatus.EXTRA or comparison.returned_sha256 is None:
                    continue
                quarantined = uow.blobs.put(
                    self._reader.iter_file(command.source, comparison.relative_path),
                    expected_sha256=comparison.returned_sha256,
                )
                quarantined_extras.append(
                    {
                        "relative_path": comparison.relative_path,
                        "sha256": quarantined.blob.sha256,
                        "size_bytes": quarantined.blob.size_bytes,
                    }
                )
            for comparison in plan.report.items:
                if comparison.status not in {
                    ReviewFileStatus.CHANGED,
                    ReviewFileStatus.STALE_BASE_CONFLICT,
                }:
                    continue
                if comparison.returned_sha256 is None:
                    continue
                if (
                    comparison.status is ReviewFileStatus.CHANGED
                    and comparison.returned_sha256 == comparison.expected_sha256
                ):
                    continue
                expected_file = expected_by_path.get(comparison.relative_path.casefold())
                if expected_file is None:
                    continue
                if (
                    expected_file.artifact_version_id != comparison.expected_version_id
                    or expected_file.sha256 != comparison.expected_sha256
                ):
                    raise ConflictError("Review manifest changed during return preflight")
                existing_id = prior_candidates.get(
                    (comparison.relative_path.casefold(), comparison.returned_sha256)
                )
                if existing_id is not None:
                    existing = uow.projections.get_artifact_version(existing_id)
                    if existing is not None:
                        candidates.append(existing)
                        candidate_files.append(
                            {
                                "relative_path": comparison.relative_path,
                                "returned_sha256": comparison.returned_sha256,
                                "version_id": existing.id,
                            }
                        )
                        continue
                base = uow.projections.get_artifact_version(expected_file.artifact_version_id)
                if base is None:
                    raise ConflictError("The review base version no longer exists")
                stored = uow.blobs.put(
                    self._reader.iter_file(command.source, comparison.relative_path),
                    expected_sha256=comparison.returned_sha256,
                )
                candidate = ArtifactVersion.managed(
                    series_id=base.series_id,
                    blob=stored.blob,
                    media_type=base.media_type,
                    filename=base.filename,
                    author_principal_id=command.context.actor.id,
                    created_at=now,
                    parent_version_id=base.id,
                    input_version_ids=(base.id,),
                    tool_name="Kraken Review Return",
                    parameters={
                        "review_batch_id": str(batch.id),
                        "package_id": str(plan.package_id),
                        "comparison": comparison.status.value,
                    },
                )
                series_stream = _series_stream(str(base.series_id))
                series_revision = uow.event_store.current_revision(series_stream)
                candidate_event = EventEnvelope.create(
                    stream_id=series_stream,
                    project_id=batch.project_id,
                    revision=series_revision + 1,
                    event_type="ArtifactVersionCreated",
                    payload=_artifact_created_payload(candidate, activated=False, candidate=True),
                    actor=ActorSnapshot.from_principal(command.context.actor),
                    recorded_at=now,
                    effective_at=command.context.effective_at,
                    performer_id=batch.assignee_id,
                    correlation_id=command.context.correlation_id,
                    # Only the aggregate-closing ReviewReturnCommitted event
                    # owns the command idempotency key. This permits one atomic
                    # command to append to several semantic streams.
                    idempotency_key=None,
                )
                uow.event_store.append(
                    series_stream, expected_revision=series_revision, events=(candidate_event,)
                )
                uow.projections.save_artifact_version(candidate, activate=False)
                candidates.append(candidate)
                candidate_files.append(
                    {
                        "relative_path": comparison.relative_path,
                        "returned_sha256": comparison.returned_sha256,
                        "version_id": candidate.id,
                    }
                )

            statuses = {item.status for item in plan.report.items}
            returned = batch.register_return(
                has_missing=ReviewFileStatus.MISSING in statuses,
                has_changed=bool(
                    statuses & {ReviewFileStatus.CHANGED, ReviewFileStatus.STALE_BASE_CONFLICT}
                ),
                at=now,
            )
            event = EventEnvelope.create(
                stream_id=_review_stream(batch.id),
                project_id=batch.project_id,
                revision=returned.revision,
                event_type="ReviewReturnCommitted",
                payload={
                    "review_batch_id": batch.id,
                    "package_id": plan.package_id,
                    "return_fingerprint": plan.return_fingerprint,
                    "state": returned.state.value,
                    "batch_revision": returned.revision,
                    "updated_at": returned.updated_at.isoformat(),
                    "comparisons": [_comparison_payload(item) for item in plan.report.items],
                    "candidate_version_ids": [item.id for item in candidates],
                    "candidate_files": candidate_files,
                    "quarantined_extras": quarantined_extras,
                },
                actor=ActorSnapshot.from_principal(command.context.actor),
                recorded_at=now,
                effective_at=command.context.effective_at,
                performer_id=batch.assignee_id,
                correlation_id=command.context.correlation_id,
                idempotency_key=command.context.idempotency_key,
            )
            uow.event_store.append(
                _review_stream(batch.id), expected_revision=batch.revision, events=(event,)
            )
            uow.projections.save_review_batch(returned)
            uow.commit()
            return ReviewReturnCommitResult(returned, plan.report, tuple(candidates))

    @staticmethod
    def _result_from_event(
        uow: UnitOfWork, batch: ReviewBatch, event: EventEnvelope
    ) -> ReviewReturnCommitResult:
        candidate_ids = event.payload.get("candidate_version_ids", ())
        candidates = tuple(
            version
            for identifier in candidate_ids
            if (version := uow.projections.get_artifact_version(ArtifactVersionId(str(identifier))))
            is not None
        )
        return ReviewReturnCommitResult(
            batch,
            _report_from_payload(event.payload.get("comparisons", ())),
            candidates,
        )


class AcceptReviewHandler(_ReviewHandler):
    def __call__(self, command: AcceptReviewCommand) -> ReviewBatch:
        with self._uow_factory() as uow:
            batch = self._load_batch(
                uow,
                project_id=command.project_id,
                batch_id=command.batch_id,
                context=command.context,
                permission=Permission.ACCEPT_REVIEW,
            )
            if _prior_command_event(
                uow,
                project_id=batch.project_id,
                idempotency_key=command.context.idempotency_key,
                event_type="ReviewBatchAccepted",
                batch_id=batch.id,
            ) is not None:
                return batch
            self._require_revision(uow, batch, command.expected_batch_revision)
            if batch.state is not ReviewBatchState.AWAITING_ACCEPTANCE:
                raise ConflictError("Review batch is not awaiting acceptance")
            requested_ids = tuple(dict.fromkeys(command.candidate_version_ids))
            if not requested_ids:
                raise ConflictError("Acceptance requires candidate versions")

            latest_by_series: dict[str, ArtifactVersionId] = {}
            recorded_ids: set[ArtifactVersionId] = set()
            for event in uow.event_store.load_stream(_review_stream(batch.id)):
                if event.event_type != "ReviewReturnCommitted":
                    continue
                for value in event.payload.get("candidate_version_ids", ()):
                    identifier = ArtifactVersionId(str(value))
                    candidate = uow.projections.get_artifact_version(identifier)
                    if candidate is not None:
                        recorded_ids.add(identifier)
                        latest_by_series[str(candidate.series_id)] = identifier
            required_ids = set(latest_by_series.values())
            if set(requested_ids) != required_ids or not set(requested_ids) <= recorded_ids:
                raise ConflictError("All latest candidates from this review batch must be accepted together")

            base_ids = {item.vector_version_id for item in batch.items}
            candidates: list[ArtifactVersion] = []
            series_ids: set[str] = set()
            for identifier in requested_ids:
                candidate = uow.projections.get_artifact_version(identifier)
                if candidate is None or candidate.parent_version_id not in base_ids:
                    raise ConflictError("Candidate does not belong to this review batch")
                if str(candidate.series_id) in series_ids:
                    raise ConflictError("Only one candidate per artifact series may be accepted")
                series_ids.add(str(candidate.series_id))
                candidates.append(candidate)

            now = self._clock.now()
            for candidate in candidates:
                stream_id = _series_stream(str(candidate.series_id))
                revision = uow.event_store.current_revision(stream_id)
                activation = EventEnvelope.create(
                    stream_id=stream_id,
                    project_id=batch.project_id,
                    revision=revision + 1,
                    event_type="ArtifactVersionActivated",
                    payload={
                        "artifact_version_id": candidate.id,
                        "series_id": candidate.series_id,
                        "review_batch_id": batch.id,
                    },
                    actor=ActorSnapshot.from_principal(command.context.actor),
                    recorded_at=now,
                    effective_at=command.context.effective_at,
                    performer_id=batch.assignee_id,
                    correlation_id=command.context.correlation_id,
                )
                uow.event_store.append(stream_id, expected_revision=revision, events=(activation,))
                uow.projections.save_artifact_version(candidate, activate=True)

            accepted = batch.transition(ReviewBatchState.COMPLETED, at=now)
            event = EventEnvelope.create(
                stream_id=_review_stream(batch.id),
                project_id=batch.project_id,
                revision=accepted.revision,
                event_type="ReviewBatchAccepted",
                payload={
                    "review_batch_id": batch.id,
                    "state": accepted.state.value,
                    "batch_revision": accepted.revision,
                    "updated_at": accepted.updated_at.isoformat(),
                    "candidate_version_ids": list(requested_ids),
                },
                actor=ActorSnapshot.from_principal(command.context.actor),
                recorded_at=now,
                effective_at=command.context.effective_at,
                performer_id=batch.assignee_id,
                correlation_id=command.context.correlation_id,
                idempotency_key=command.context.idempotency_key,
            )
            uow.event_store.append(
                _review_stream(batch.id), expected_revision=batch.revision, events=(event,)
            )
            uow.projections.save_review_batch(accepted)
            uow.commit()
            return accepted


class RequestReviewChangesHandler(_ReviewHandler):
    def __call__(self, command: RequestReviewChangesCommand) -> ReviewBatch:
        reason = require_non_empty(command.reason, field="review_changes.reason", maximum=100_000)
        with self._uow_factory() as uow:
            batch = self._load_batch(
                uow,
                project_id=command.project_id,
                batch_id=command.batch_id,
                context=command.context,
                permission=Permission.ACCEPT_REVIEW,
            )
            if _prior_command_event(
                uow,
                project_id=batch.project_id,
                idempotency_key=command.context.idempotency_key,
                event_type="ReviewChangesRequested",
                batch_id=batch.id,
            ) is not None:
                return batch
            self._require_revision(uow, batch, command.expected_batch_revision)
            if batch.state is not ReviewBatchState.AWAITING_ACCEPTANCE:
                raise ConflictError("Review batch is not awaiting acceptance")
            now = self._clock.now()
            changed = batch.transition(ReviewBatchState.CHANGES_REQUESTED, at=now)
            event = EventEnvelope.create(
                stream_id=_review_stream(batch.id),
                project_id=batch.project_id,
                revision=changed.revision,
                event_type="ReviewChangesRequested",
                payload={
                    "review_batch_id": batch.id,
                    "state": changed.state.value,
                    "batch_revision": changed.revision,
                    "updated_at": changed.updated_at.isoformat(),
                    "reason": reason,
                },
                actor=ActorSnapshot.from_principal(command.context.actor),
                recorded_at=now,
                effective_at=command.context.effective_at,
                performer_id=batch.assignee_id,
                correlation_id=command.context.correlation_id,
                idempotency_key=command.context.idempotency_key,
            )
            uow.event_store.append(
                _review_stream(batch.id), expected_revision=batch.revision, events=(event,)
            )
            uow.projections.save_review_batch(changed)
            uow.commit()
            return changed


class CancelReviewBatchHandler(_ReviewHandler):
    def __call__(self, command: CancelReviewBatchCommand) -> ReviewBatch:
        with self._uow_factory() as uow:
            batch = self._load_batch(
                uow,
                project_id=command.project_id,
                batch_id=command.batch_id,
                context=command.context,
                permission=Permission.MANAGE_REVIEW,
            )
            if _prior_command_event(
                uow,
                project_id=batch.project_id,
                idempotency_key=command.context.idempotency_key,
                event_type="ReviewBatchCancelled",
                batch_id=batch.id,
            ) is not None:
                return batch
            self._require_revision(uow, batch, command.expected_batch_revision)
            if batch.state in {ReviewBatchState.COMPLETED, ReviewBatchState.CANCELLED}:
                raise ConflictError("Review batch is already closed")
            now = self._clock.now()
            cancelled = batch.transition(ReviewBatchState.CANCELLED, at=now)
            event = EventEnvelope.create(
                stream_id=_review_stream(batch.id),
                project_id=batch.project_id,
                revision=cancelled.revision,
                event_type="ReviewBatchCancelled",
                payload={
                    "review_batch_id": batch.id,
                    "state": cancelled.state.value,
                    "batch_revision": cancelled.revision,
                    "updated_at": cancelled.updated_at.isoformat(),
                },
                actor=ActorSnapshot.from_principal(command.context.actor),
                recorded_at=now,
                effective_at=command.context.effective_at,
                performer_id=batch.assignee_id,
                correlation_id=command.context.correlation_id,
                idempotency_key=command.context.idempotency_key,
            )
            uow.event_store.append(
                _review_stream(batch.id),
                expected_revision=batch.revision,
                events=(event,),
            )
            uow.projections.save_review_batch(cancelled)
            uow.commit()
            return cancelled


__all__ = [
    "AcceptReviewHandler",
    "CancelReviewBatchHandler",
    "CommitReviewReturnHandler",
    "CreateReviewBatchHandler",
    "DryRunReviewReturnHandler",
    "ExportReviewPackageHandler",
    "PlanReviewPackageHandler",
    "RequestReviewChangesHandler",
]
