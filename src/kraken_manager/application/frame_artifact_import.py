"""Import one image per frame as a single authorized action.

Preparing a layer used to run several commands per frame, each opening its own
transactions and re-checking the actor's rights. This handler checks the right
once for the whole action, then writes frames in bounded transactions so a
large layer neither holds one transaction for minutes nor pays per-frame
overhead. Events and idempotency keys stay per frame, so an interrupted import
resumes where it stopped.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from contextlib import AbstractContextManager
from dataclasses import dataclass, replace

from kraken_manager.application.dto import (
    AddArtifactVersionCommand,
    CommandContext,
    CreateArtifactSeriesCommand,
)
from kraken_manager.application.errors import ConflictError
from kraken_manager.application.use_cases import (
    AddArtifactVersionHandler,
    CreateArtifactSeriesHandler,
    _blob_store,
    _ProjectHandler,
    _series_stream,
)
from kraken_manager.domain.common import ProjectId
from kraken_manager.domain.identity import Permission

TRANSACTION_BATCH_SIZE = 250


@dataclass(frozen=True, slots=True)
class FrameArtifactImport:
    series: CreateArtifactSeriesCommand
    # ``expected_series_revision`` is ignored: the import only adds a first version.
    version: AddArtifactVersionCommand
    open_content: Callable[[], AbstractContextManager[Iterable[bytes]]]


@dataclass(frozen=True, slots=True)
class ImportFrameArtifactsCommand:
    context: CommandContext
    project_id: ProjectId
    items: tuple[FrameArtifactImport, ...]


class ImportFrameArtifactsHandler(_ProjectHandler):
    def __init__(self, uow_factory, storage_profiles, clock, authorization=None) -> None:
        super().__init__(uow_factory, storage_profiles, clock, authorization)
        self._series = CreateArtifactSeriesHandler(uow_factory, storage_profiles, clock, authorization)
        self._versions = AddArtifactVersionHandler(uow_factory, storage_profiles, clock, authorization)

    def __call__(self, command: ImportFrameArtifactsCommand) -> int:
        for item in command.items:
            if item.series.project_id != command.project_id or item.version.project_id != command.project_id:
                raise ConflictError("Every imported frame must belong to the authorized project")
            if item.version.series_id != item.series.series_id:
                raise ConflictError("Imported version must target its own frame series")
        project = None
        blobs = None
        for start in range(0, len(command.items), TRANSACTION_BATCH_SIZE):
            chunk = command.items[start:start + TRANSACTION_BATCH_SIZE]
            with self._uow_factory() as uow:
                if project is None:
                    # The only rights check of the whole action.
                    project, _ = self._load_project_and_authorize(
                        uow,
                        project_id=command.project_id,
                        actor=command.context.actor,
                        permission=Permission.IMPORT_ARTIFACT,
                        gitlab_identity_verified=command.context.gitlab_identity_verified,
                    )
                pending = []
                for item in chunk:
                    series, _created = self._series.record(uow, project, item.series)
                    if uow.projections.get_active_artifact_version(series.id) is None:
                        pending.append(item)
                uow.commit()
            if not pending:
                continue
            if blobs is None:
                blobs = _blob_store(self._uow_factory)
            stored = []
            for item in pending:
                with item.open_content() as chunks:
                    stored.append(blobs.put(chunks, expected_sha256=item.version.expected_sha256))
            with self._uow_factory() as uow:
                for item, content in zip(pending, stored, strict=True):
                    revision = uow.event_store.current_revision(_series_stream(str(item.version.series_id)))
                    version = replace(item.version, expected_series_revision=revision)
                    self._versions.record(uow, version, content, project=project)
                uow.commit()
        return len(command.items)
