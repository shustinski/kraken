"""composition: domain selection."""
from __future__ import annotations
from collections.abc import Iterable
from pathlib import Path
from kraken_manager.domain.artifacts import (
    ArtifactVersion,
)
from kraken_manager.domain.common import (
    ArtifactSeriesId,
    ArtifactVersionId,
    ProjectId,
)
from kraken_manager.domain.selection import FrameRowRange, FrameSelectionV1
from kraken_manager.infrastructure.blob.filesystem import FilesystemBlobStore
from kraken_manager.infrastructure.review.crypto import Ed25519KeyPair


class DomainSelectionMixin:
    def artifact_versions(
        self,
        project_id: ProjectId | str,
        series_id: ArtifactSeriesId | str,
    ) -> tuple[ArtifactVersion, ...]:
        return self._projection(project_id).list_artifact_versions(
            ArtifactSeriesId(str(series_id))
        )

    def active_artifact_version(
        self,
        project_id: ProjectId | str,
        series_id: ArtifactSeriesId | str,
    ) -> ArtifactVersion | None:
        return self._projection(project_id).get_active_artifact_version(
            ArtifactSeriesId(str(series_id))
        )

    def artifact_version(
        self,
        project_id: ProjectId | str,
        version_id: ArtifactVersionId | str,
    ) -> ArtifactVersion | None:
        return self._projection(project_id).get_artifact_version(
            ArtifactVersionId(str(version_id))
        )

    def managed_artifact_path(
        self,
        project_id: ProjectId | str,
        version_id: ArtifactVersionId | str,
    ) -> Path:
        version = self.artifact_version(project_id, version_id)
        if version is None or version.blob is None:
            raise ValueError("Для запуска нужны файлы, сохранённые в проекте")
        store = FilesystemBlobStore.for_project(self.catalog_root, str(project_id))
        path = store._path(version.blob.sha256)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def _review_keys(self) -> Ed25519KeyPair:
        private_name = f"review-signing:{self.profile.id}:private"
        public_name = f"review-signing:{self.profile.id}:public"
        get_secret = getattr(self.secrets, "get")
        set_secret = getattr(self.secrets, "set")
        private_key = get_secret(private_name)
        public_key = get_secret(public_name)
        if private_key is None or public_key is None:
            pair = Ed25519KeyPair.generate()
            set_secret(private_name, pair.private_key)
            set_secret(public_name, pair.public_key)
            return pair
        return Ed25519KeyPair(private_key=private_key, public_key=public_key)

    @staticmethod
    def _domain_selection(coordinates: Iterable[tuple[int, int]]) -> FrameSelectionV1:
        unique = sorted(
            {(int(x), int(y)) for x, y in coordinates},
            key=lambda item: (item[1], item[0]),
        )
        if not unique:
            raise ValueError("Select at least one frame")
        return FrameSelectionV1(
            row_ranges=tuple(
                FrameRowRange(y=y, x_start=x, x_end=x)
                for x, y in unique
            )
        )
