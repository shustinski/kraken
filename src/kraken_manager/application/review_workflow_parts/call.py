"""review_workflow: call."""
from __future__ import annotations
from collections.abc import Mapping
from kraken_manager.application.dto import (
    CommitReviewReturnCommand,
    ReviewReturnCommitResult,
)
from kraken_manager.application.errors import ConflictError
from kraken_manager.application.use_cases import (
    _series_stream,
)
from kraken_manager.domain.artifacts import ArtifactVersion
from kraken_manager.domain.common import ArtifactVersionId
from kraken_manager.domain.events import ActorSnapshot, EventEnvelope
from kraken_manager.domain.identity import Permission
from kraken_manager.domain.workflows import (
    ReviewBatchState,
    ReviewFileStatus,
)

from kraken_manager.application.review_workflow_parts.review_handler import (
    _artifact_created_payload,
    _comparison_payload,
    _prior_command_event,
    _review_stream,
)


class CallMixin:
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
        prepared = self._scan_return_package(command.source)
        preview = self._preview_return(command, prepared)
        if isinstance(preview, ReviewReturnCommitResult):
            return preview
        stored_blobs = self._store_return_blobs(command.source, preview)
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

            plan = self._inspect(uow, batch, command.source, prepared)
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

            manifest = prepared[0]
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
                quarantined = self._stored_return_blob(
                    stored_blobs, comparison.relative_path, comparison.returned_sha256
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
                stored = self._stored_return_blob(
                    stored_blobs, comparison.relative_path, comparison.returned_sha256
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
                    payload={
                        **_artifact_created_payload(candidate, activated=False, candidate=True),
                        "file_history": "batch",
                        "batch_key": str(batch.id),
                    },
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
