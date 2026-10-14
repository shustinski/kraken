"""review_workflow: result event."""
from __future__ import annotations
from collections.abc import Mapping
from kraken_manager.application.dto import (
    ReviewReturnCommitResult,
    StoredContent,
)
from kraken_manager.application.errors import ConflictError
from kraken_manager.application.ports import (
    UnitOfWork,
)
from kraken_manager.application.use_cases import (
    _blob_store,
)
from kraken_manager.domain.common import ArtifactVersionId
from kraken_manager.domain.events import EventEnvelope
from kraken_manager.domain.workflows import (
    ReviewBatch,
)

from kraken_manager.application.review_workflow_parts.review_handler import _report_from_payload


class ResultEventMixin:
    def _store_return_blobs(
        self, source: str, writes: tuple[tuple[str, str], ...]
    ) -> dict[tuple[str, str], StoredContent]:
        blobs = _blob_store(self._uow_factory)
        stored: dict[tuple[str, str], StoredContent] = {}
        for relative_path, digest in writes:
            stored[(relative_path.casefold(), digest)] = blobs.put(
                self._reader.iter_file(source, relative_path),
                expected_sha256=digest,
            )
        return stored

    @staticmethod
    def _stored_return_blob(
        stored: Mapping[tuple[str, str], StoredContent], relative_path: str, digest: str
    ) -> StoredContent:
        found = stored.get((relative_path.casefold(), digest))
        if found is None:
            raise ConflictError("Returned file was not stored before the review transaction")
        return found

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
