"""remote_client: export review batch."""
from __future__ import annotations
import os
import tempfile
import threading
from collections.abc import Mapping
from pathlib import Path
from kraken_manager.application.dto import ReviewReturnCommitResult, ReviewReturnPlan
from kraken_manager.domain.identity import (
    Principal,
)
from kraken_manager.domain.workflows import (
    ReviewBatch,
)
from kraken_manager.infrastructure.filesystem._codec import decode_model
from kraken_manager.infrastructure.review.manifest import manifest_from_json

from kraken_hub.infrastructure.remote.parts.extract_review_archive import (
    RemoteServerError,
    _extract_review_archive,
    _review_archive,
)


class ExportReviewBatchMixin:
    def accept_review(
        self,
        *,
        principal: Principal,
        batch: ReviewBatch,
        candidate_version_ids: object,
        idempotency_key: str,
    ) -> ReviewBatch:
        del principal
        payload = self._call(
            "PATCH",
            f"/api/v1/projects/{batch.project_id}/reviews/{batch.id}",
            payload={
                "action": "accept",
                "candidate_version_ids": [str(item) for item in candidate_version_ids],
            },
            idempotency_key=idempotency_key,
            if_match=batch.revision,
        )
        return decode_model(ReviewBatch, payload)

    def request_review_changes(
        self,
        *,
        principal: Principal,
        batch: ReviewBatch,
        reason: str,
        idempotency_key: str,
    ) -> ReviewBatch:
        del principal
        payload = self._call(
            "PATCH",
            f"/api/v1/projects/{batch.project_id}/reviews/{batch.id}",
            payload={"action": "request_changes", "reason": reason},
            idempotency_key=idempotency_key,
            if_match=batch.revision,
        )
        return decode_model(ReviewBatch, payload)

    def cancel_review_batch(
        self,
        *,
        principal: Principal,
        batch: ReviewBatch,
        idempotency_key: str,
    ) -> ReviewBatch:
        del principal
        payload = self._call(
            "PATCH",
            f"/api/v1/projects/{batch.project_id}/reviews/{batch.id}",
            payload={"action": "cancel"},
            idempotency_key=idempotency_key,
            if_match=batch.revision,
        )
        return decode_model(ReviewBatch, payload)

    def review_candidate_version_ids(self, batch: ReviewBatch) -> tuple[object, ...]:
        latest_by_series: dict[str, object] = {}
        for event in self.history(batch.project_id):
            if event.event_type != "ReviewReturnCommitted" or str(
                event.payload.get("review_batch_id", "")
            ) != str(batch.id):
                continue
            for value in event.payload.get("candidate_version_ids", ()):
                version = self.artifact_version(batch.project_id, value)
                if version is not None:
                    latest_by_series[str(version.series_id)] = version.id
        return tuple(latest_by_series[key] for key in sorted(latest_by_series))

    def export_review_batch(
        self,
        *,
        principal: Principal,
        batch: ReviewBatch,
        destination: Path | str,
        idempotency_key: str,
    ) -> ReviewBatch:
        del principal
        target = Path(destination).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}-{threading.get_ident()}.zip")
        temporary.unlink(missing_ok=True)
        try:
            self._http.download(
                f"/api/v1/projects/{batch.project_id}/reviews/{batch.id}/package",
                token=self.auth.access_token,
                destination=temporary,
                method="POST",
                headers={
                    "Idempotency-Key": idempotency_key,
                    "If-Match": str(batch.revision),
                },
            )
            _extract_review_archive(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        return next(
            (item for item in self.review_batches() if str(item.id) == str(batch.id)),
            batch,
        )

    def _review_source(self, source: Path | str) -> tuple[Path, ReviewBatch]:
        folder = Path(source).resolve(strict=True)
        raw = (folder / "kraken-review.json").read_bytes()
        if len(raw) > 16 * 1024**2:
            raise ValueError("Review manifest exceeds the size limit")
        manifest = manifest_from_json(raw.decode("utf-8"))
        batch_id = manifest.batch_id or manifest.package_id
        batch = next(
            (
                item
                for item in self.review_batches()
                if str(item.project_id) == str(manifest.project_id)
                and str(item.id) == str(batch_id)
            ),
            None,
        )
        if batch is None:
            raise ValueError("Manifest does not match a known server review batch")
        return folder, batch

    def _upload_review_return(
        self,
        source: Path | str,
        batch: ReviewBatch,
        *,
        operation: str,
        idempotency_key: str,
    ) -> Mapping[str, object]:
        folder = Path(source).resolve(strict=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix="kraken-review-return-", suffix=".zip"
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        temporary.unlink(missing_ok=True)
        try:
            _review_archive(folder, temporary)
            payload = self._http.upload(
                f"/api/v1/projects/{batch.project_id}/reviews/{batch.id}/return/{operation}",
                token=self.auth.access_token,
                source=temporary,
                headers={
                    "Idempotency-Key": idempotency_key,
                    "If-Match": str(batch.revision),
                },
            )
        finally:
            temporary.unlink(missing_ok=True)
        if not isinstance(payload, Mapping):
            raise RemoteServerError("Kraken Server returned an invalid review response")
        return payload

    def review_return_preflight(
        self,
        *,
        principal: Principal,
        source: Path | str,
        idempotency_key: str,
    ) -> tuple[ReviewBatch, ReviewReturnPlan]:
        del principal
        folder, batch = self._review_source(source)
        payload = self._upload_review_return(
            folder,
            batch,
            operation="preflight",
            idempotency_key=idempotency_key,
        )
        return batch, decode_model(ReviewReturnPlan, payload)

    def commit_review_return(
        self,
        *,
        principal: Principal,
        batch: ReviewBatch,
        source: Path | str,
        idempotency_key: str,
    ) -> ReviewReturnCommitResult:
        del principal
        payload = self._upload_review_return(
            source,
            batch,
            operation="commit",
            idempotency_key=idempotency_key,
        )
        return decode_model(ReviewReturnCommitResult, payload)
