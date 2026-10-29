"""Values returned by application operations to any presentation or transport."""

from dataclasses import dataclass

from kraken_manager.application.imports import ImportPlan
from kraken_manager.domain.artifacts import ArtifactVersion
from kraken_manager.domain.identity import Principal


@dataclass(frozen=True, slots=True)
class UserSession:
    token: str
    principal: Principal
    expires_at: str


@dataclass(frozen=True, slots=True)
class ManagedImportResult:
    plan: ImportPlan
    versions: tuple[ArtifactVersion, ...]


@dataclass(frozen=True, slots=True)
class IntegrityScanResult:
    projects: int
    events: int
    blobs: int
    errors: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.errors


@dataclass(frozen=True, slots=True)
class FrameCellSnapshot:
    x: int
    y: int
    status: str
    frame_id: str
    artifact_version_id: str
    sha256: str
    modified_at: str = ""
    performer_color: str = ""
    performer_initials: str = ""
    review_status: str = "not_checked"
    quality: float | None = None


@dataclass(frozen=True, slots=True)
class FrameManagementState:
    frame_id: str
    artifact_version_id: str
    modified_at: str
    performer_color: str
    performer_initials: str
    review_status: str


@dataclass(frozen=True, slots=True)
class ProjectDeletionResult:
    project_id: str
    catalog_cache_removed: bool
    thumbnail_cache_removed: bool
    staging_directories_removed: int


@dataclass(frozen=True, slots=True)
class KarakalAnalysisRun:
    run_id: str
    project_id: str
    layer_id: str
    publication_sequence: int
    created_at: str
    frame_confidence: dict[str, float]
    report: dict[str, object]
    parameters: dict[str, object]
    plugin_version: str
