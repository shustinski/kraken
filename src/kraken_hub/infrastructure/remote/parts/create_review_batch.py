"""remote_client: create review batch."""
from __future__ import annotations
import hashlib
import os
import shutil
import threading
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from kraken_core.safe_files import open_regular_read
from urllib.parse import unquote, urlparse
from kraken_manager.domain.artifacts import (
    ArtifactVersion,
)
from kraken_manager.domain.identity import (
    Principal,
)
from kraken_manager.domain.workflows import (
    ReviewBatch,
)
from kraken_manager.infrastructure.filesystem._codec import decode_model


class CreateReviewBatchMixin:
    def managed_artifact_path(self, project_id: object, version_id: object) -> Path:
        version = self.artifact_version(project_id, version_id)
        if version is None or version.blob is None:
            raise ValueError("Для запуска нужны файлы, сохранённые в проекте")
        target = self.data_dir / "remote-cache" / str(project_id) / version.blob.sha256
        if target.is_file() and target.stat().st_size == version.blob.size_bytes:
            with open_regular_read(target, root=self.data_dir) as stream:
                cached_digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if cached_digest == version.blob.sha256:
                return target
        target.unlink(missing_ok=True)
        temporary = target.with_name(f"{target.name}.part-{threading.get_ident()}")
        temporary.unlink(missing_ok=True)
        self._download_managed_artifact(project_id, version.id, temporary)
        digest = hashlib.sha256()
        size = 0
        with temporary.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
        if digest.hexdigest() != version.blob.sha256 or size != version.blob.size_bytes:
            temporary.unlink(missing_ok=True)
            raise ValueError(
                "Сервер вернул blob с неверной контрольной суммой или размером"
            )
        temporary.replace(target)
        return target

    def read_project_blob(self, project_id: object, source_key: str) -> bytes:
        version = self.artifact_version(project_id, source_key)
        if version is None or version.blob is None:
            raise ValueError("Для чтения нужны файлы, сохранённые на сервере")
        digest = hashlib.sha256()
        content = bytearray()
        for chunk in self._http.iter_bytes(
            f"/api/v1/projects/{project_id}/artifacts/versions/{version.id}", token=self.auth.access_token
        ):
            content.extend(chunk)
            if len(content) > version.blob.size_bytes:
                raise ValueError("Сервер вернул blob с неверным размером")
            digest.update(chunk)
        if digest.hexdigest() != version.blob.sha256 or len(content) != version.blob.size_bytes:
            raise ValueError("Сервер вернул blob с неверной контрольной суммой или размером")
        return bytes(content)

    def external_artifact_changed(self, version: ArtifactVersion) -> bool:
        if version.external is None:
            return False
        parsed = urlparse(version.external.uri)
        if parsed.scheme != "file":
            raise ValueError(
                f"Внешний файл недоступен на этой рабочей станции: {version.external.uri}"
            )
        path = Path(
            unquote(parsed.path.lstrip("/") if os.name == "nt" else parsed.path)
        )
        if os.name == "nt" and parsed.netloc:
            path = Path(f"//{parsed.netloc}/{unquote(parsed.path.lstrip('/'))}")
        try:
            resolved = path.resolve(strict=True)
        except OSError as exc:
            raise FileNotFoundError(
                f"Внешний файл недоступен на этой рабочей станции: {path}"
            ) from exc
        digest = hashlib.sha256()
        size = 0
        with resolved.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
        return (
            size != version.external.observed_size_bytes
            or digest.hexdigest() != version.external.fingerprint_sha256
        )

    def export_artifact_version(
        self, project_id: object, version: ArtifactVersion, destination: Path | str
    ) -> Path:
        if version.external is None:
            return self.export_managed_artifact(project_id, version, destination)
        if self.external_artifact_changed(version):
            raise ValueError(
                f"Внешний файл изменён после регистрации: {version.external.uri}"
            )
        parsed = urlparse(version.external.uri)
        source = Path(
            unquote(parsed.path.lstrip("/") if os.name == "nt" else parsed.path)
        )
        if os.name == "nt" and parsed.netloc:
            source = Path(f"//{parsed.netloc}/{unquote(parsed.path.lstrip('/'))}")
        target = Path(destination).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(target)
        shutil.copy2(source.resolve(strict=True), target)
        return target

    def review_batches(self) -> tuple[ReviewBatch, ...]:
        batches: list[ReviewBatch] = []
        for project in self.list_projects(include_archived=True):
            payload = self._call("GET", f"/api/v1/projects/{project.id}/reviews")
            items = payload.get("items", ()) if isinstance(payload, Mapping) else ()
            batches.extend(
                decode_model(ReviewBatch, item)
                for item in items
                if isinstance(item, Mapping)
            )
        return tuple(
            sorted(
                batches, key=lambda item: (item.updated_at, str(item.id)), reverse=True
            )
        )

    def active_review_batches(self) -> tuple[ReviewBatch, ...]:
        batches: list[ReviewBatch] = []
        for project in self.list_projects(include_archived=True):
            payload = self._call(
                "GET", f"/api/v1/projects/{project.id}/reviews?active_only=true"
            )
            items = payload.get("items", ()) if isinstance(payload, Mapping) else ()
            batches.extend(
                decode_model(ReviewBatch, item)
                for item in items
                if isinstance(item, Mapping)
            )
        return tuple(
            sorted(
                batches, key=lambda item: (item.updated_at, str(item.id)), reverse=True
            )
        )

    def create_review_batch(
        self,
        *,
        principal: Principal,
        project_id: object,
        layer_id: object,
        image_representation_id: object,
        vector_representation_id: object,
        coordinates: object,
        assignee_id: object,
        instructions: str = "",
        due_at: datetime | None = None,
        idempotency_key: str,
    ) -> ReviewBatch:
        del principal
        layer = next(
            (
                item
                for item in self.list_layers(project_id)
                if str(item.id) == str(layer_id)
            ),
            None,
        )
        if layer is None:
            raise ValueError("Layer was not found")
        payload = self._call(
            "POST",
            f"/api/v1/projects/{project_id}/reviews",
            payload={
                "layer_id": str(layer_id),
                "image_representation_id": str(image_representation_id),
                "vector_representation_id": str(vector_representation_id),
                "coordinates": [[int(x), int(y)] for x, y in coordinates],
                "assignee_id": str(assignee_id),
                "instructions": instructions,
                "due_at": None if due_at is None else due_at.isoformat(),
            },
            idempotency_key=idempotency_key,
            if_match=layer.revision,
        )
        return decode_model(ReviewBatch, payload)
