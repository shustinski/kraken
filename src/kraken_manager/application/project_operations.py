"""Public application ports. Frontends depend on these contracts, adapters implement them.

Operation declarations are explicit: an implementation missing an operation is
rejected at assembly time. This module never imports a UI, transport, or store.
"""

from __future__ import annotations
from kraken_manager.application.results import KarakalAnalysisRun

from collections.abc import Iterable, Mapping
from datetime import datetime
from os import PathLike
from typing import Any, Protocol, runtime_checkable

from kraken_manager.application.imports import ImportMappingMode, ImportPlan
from kraken_manager.application.ports import ProjectCommand, StorageProfile
from kraken_manager.application.reporting import ActivityRecord
from kraken_manager.application.results import FrameCellSnapshot, FrameManagementState, ManagedImportResult
from kraken_manager.domain.artifacts import ArtifactScope, ArtifactSeries, ArtifactVersion, NoteRevision
from kraken_manager.domain.common import (
    ArtifactSeriesId,
    ArtifactVersionId,
    LayerId,
    PerformerId,
    PrincipalId,
    ProjectId,
    RepresentationId,
)
from kraken_manager.domain.events import EventEnvelope
from kraken_manager.domain.identity import Performer, Permission, Principal, ProjectRole
from kraken_manager.domain.project import (
    GridOrientation,
    Layer,
    LayerType,
    Project,
    Representation,
    RepresentationKind,
    RepresentationPurpose,
)
from kraken_manager.domain.work_orders import WorkOrder
from kraken_manager.domain.workflows import ReviewBatch
from kraken_manager.workspace import ImageConversionSettings, LayerFileBinding, LayerSourceScan, ProjectWorkspaceBinding


class CatalogOperations(Protocol):
    def execute_project_command(self, command: ProjectCommand) -> object: ...

    def archive_project(self, *, principal: Principal, project: Project, idempotency_key: str) -> Project: ...

    def create_project(
        self,
        *,
        principal: Principal,
        name: str,
        width: int,
        height: int,
        orientation: GridOrientation,
        idempotency_key: str,
        layer_template: bool = ...,
        source_root: PathLike[str] | str | None = ...,
        derived_root: PathLike[str] | str | None = ...,
        storage_profile_id: str | None = ...,
    ) -> Project: ...

    def get_project(self, project_id: ProjectId | str, *, as_of: datetime | None = ...) -> Project | None: ...

    def is_remote_project(self, project_id: ProjectId | str) -> bool: ...

    def list_projects(self, *, include_archived: bool = ...) -> tuple[Project, ...]: ...

    def list_storage_profiles(self) -> tuple[StorageProfile, ...]: ...

    def project_storage_label(self, project_id: ProjectId | str) -> str: ...

    def rename_project(self, *, principal: Principal, project: Project, name: str, idempotency_key: str) -> Project: ...

    def restore_project(self, *, principal: Principal, project: Project, idempotency_key: str) -> Project: ...


class LayersOperations(Protocol):
    def archive_layer(self, *, principal: Principal, project: Project, layer: Layer, idempotency_key: str) -> Layer: ...

    def create_external_layer(
        self,
        *,
        principal: Principal,
        project: Project,
        name: str,
        layer_type: LayerType,
        order: int,
        image_directory: PathLike[str] | str,
        ssc_directory: PathLike[str] | str | None,
        prv_directory: PathLike[str] | str | None,
        idempotency_key: str,
    ) -> tuple[Layer, LayerFileBinding, Representation]: ...

    def create_layer(
        self,
        *,
        principal: Principal,
        project: Project,
        name: str,
        layer_type: LayerType,
        order: int,
        idempotency_key: str,
        layer_id: LayerId | str | None = ...,
    ) -> Layer: ...

    def create_layer_from_disk(
        self,
        *,
        principal: Principal,
        project: Project,
        name: str,
        layer_type: LayerType,
        order: int,
        scan: LayerSourceScan,
        conversion: ImageConversionSettings,
        idempotency_key: str,
        progress: object = ...,
        cancelled: object = ...,
    ) -> tuple[Layer, LayerFileBinding, Representation]: ...

    def delete_layer(
        self, *, principal: Principal, project: Project, layer: Layer, confirmation_name: str, idempotency_key: str
    ) -> Layer: ...

    def list_layers(
        self, project_id: ProjectId | str, *, as_of: datetime | None = ..., include_archived: bool = ...
    ) -> tuple[Layer, ...]: ...

    def rename_layer(
        self, *, principal: Principal, project: Project, layer: Layer, name: str, idempotency_key: str
    ) -> Layer: ...

    def reorder_layer(
        self, *, principal: Principal, project: Project, layer: Layer, order: int, idempotency_key: str
    ) -> Layer: ...

    def reorder_layers(
        self,
        *,
        principal: Principal,
        project: Project,
        layers: Iterable[Layer],
        layer_ids: Iterable[LayerId | str],
        idempotency_key: str,
    ) -> tuple[Layer, ...]: ...


class RepresentationsOperations(Protocol):
    def activate_representation(
        self,
        *,
        principal: Principal,
        project: Project,
        layer: Layer,
        representation: Representation,
        idempotency_key: str,
    ) -> Representation: ...

    def archive_representation(
        self,
        *,
        principal: Principal,
        project: Project,
        layer: Layer,
        representation: Representation,
        idempotency_key: str,
    ) -> Representation: ...

    def create_representation(
        self,
        *,
        principal: Principal,
        project: Project,
        layer: Layer,
        name: str,
        kind: RepresentationKind,
        idempotency_key: str,
        note: str = ...,
        source: str | None = ...,
        source_image_representation_id: object = ...,
        active: bool = ...,
        purpose: RepresentationPurpose = ...,
    ) -> Representation: ...

    def deactivate_representation(
        self,
        *,
        principal: Principal,
        project: Project,
        layer: Layer,
        representation: Representation,
        idempotency_key: str,
    ) -> Representation: ...

    def list_representations(
        self,
        project_id: ProjectId | str,
        layer_id: LayerId | str,
        *,
        include_archived: bool = ...,
        as_of: datetime | None = ...,
    ) -> tuple[Representation, ...]: ...

    def materialize_representation_inputs(
        self, *, principal: Principal, project: Project, layer: Layer, representation: Representation
    ) -> tuple[FrameCellSnapshot, ...]: ...

    def rename_representation(
        self,
        *,
        principal: Principal,
        project: Project,
        layer: Layer,
        representation: Representation,
        name: str,
        idempotency_key: str,
    ) -> Representation: ...

    def update_representation_note(
        self,
        *,
        principal: Principal,
        project: Project,
        layer: Layer,
        representation: Representation,
        note: str,
        idempotency_key: str,
    ) -> Representation: ...


class ArtifactsOperations(Protocol):
    def activate_artifact_version(
        self,
        *,
        principal: Principal,
        project_id: ProjectId | str,
        series_id: ArtifactSeriesId | str,
        version_id: ArtifactVersionId | str,
        idempotency_key: str,
    ) -> ArtifactVersion: ...

    def active_artifact_version(
        self, project_id: ProjectId | str, series_id: ArtifactSeriesId | str
    ) -> ArtifactVersion | None: ...

    def add_external_artifact_version(
        self,
        *,
        principal: Principal,
        project_id: ProjectId | str,
        series_id: ArtifactSeriesId | str,
        source: PathLike[str] | str,
        parent_version_id: ArtifactVersionId | str | None = ...,
        idempotency_key: str,
    ) -> ArtifactVersion: ...

    def add_managed_artifact_version(
        self,
        *,
        principal: Principal,
        project_id: ProjectId | str,
        series_id: ArtifactSeriesId | str,
        source: PathLike[str] | str,
        parent_version_id: ArtifactVersionId | str | None = ...,
        idempotency_key: str,
    ) -> ArtifactVersion: ...

    def archive_artifact_series(
        self, *, principal: Principal, series: ArtifactSeries, idempotency_key: str
    ) -> ArtifactSeries: ...

    def artifact_source_path(
        self, project_id: ProjectId | str, artifact_version_id: ArtifactVersionId | str
    ) -> PathLike[str]: ...

    def artifact_stream_revision(self, project_id: ProjectId | str, series_id: ArtifactSeriesId | str) -> int: ...

    def artifact_version(
        self, project_id: ProjectId | str, version_id: ArtifactVersionId | str
    ) -> ArtifactVersion | None: ...

    def artifact_versions(
        self, project_id: ProjectId | str, series_id: ArtifactSeriesId | str
    ) -> tuple[ArtifactVersion, ...]: ...

    def create_artifact_series(
        self,
        *,
        principal: Principal,
        project_id: ProjectId | str,
        scope: ArtifactScope,
        name: str,
        layer_id: LayerId | str | None = ...,
        representation_id: RepresentationId | str | None = ...,
        frame_id: str | None = ...,
        series_id: ArtifactSeriesId | str | None = ...,
        idempotency_key: str,
    ) -> ArtifactSeries: ...

    def create_note(
        self,
        *,
        principal: Principal,
        project_id: ProjectId | str,
        body: str,
        layer_id: LayerId | str | None = ...,
        frame_id: str | None = ...,
        idempotency_key: str,
    ) -> NoteRevision: ...

    def export_artifact_version(
        self, project_id: ProjectId | str, version: ArtifactVersion, destination: PathLike[str] | str
    ) -> PathLike[str]: ...

    def export_managed_artifact(
        self, project_id: ProjectId | str, version: ArtifactVersion, destination: PathLike[str] | str
    ) -> PathLike[str]: ...

    def external_artifact_changed(self, version: ArtifactVersion) -> bool: ...

    def frame_cells(
        self,
        project_id: ProjectId | str,
        layer_id: LayerId | str,
        representation_id: RepresentationId | str,
        *,
        as_of: datetime | None = ...,
        bounds: tuple[int, int, int, int] | None = ...,
    ) -> tuple[FrameCellSnapshot, ...]: ...

    def frame_management_states(
        self,
        project_id: ProjectId | str,
        layer_id: LayerId | str,
        representation_id: RepresentationId | str,
        *,
        as_of: datetime | None = ...,
    ) -> tuple[FrameManagementState, ...]: ...

    def get_artifact_version(
        self, project_id: ProjectId | str, artifact_version_id: ArtifactVersionId | str
    ) -> ArtifactVersion | None: ...

    def list_artifact_series(
        self,
        project_id: ProjectId | str,
        *,
        layer_id: LayerId | str | None = ...,
        representation_id: RepresentationId | str | None = ...,
        include_archived: bool = ...,
    ) -> tuple[ArtifactSeries, ...]: ...

    def list_notes(
        self, project_id: ProjectId | str, *, layer_id: LayerId | str | None = ..., frame_id: str | None = ...
    ) -> tuple[NoteRevision, ...]: ...

    def managed_artifact_path(
        self, project_id: ProjectId | str, version_id: ArtifactVersionId | str
    ) -> PathLike[str]: ...

    def matrix_viewport(
        self,
        project_id: ProjectId | str,
        *,
        layer_id: LayerId | str,
        representation_ids: Iterable[RepresentationId | str] = ...,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        lod: int = ...,
        include_missing: bool = ...,
    ) -> dict[str, Any]: ...

    def read_project_blob(self, project_id: ProjectId | str, sha256: str) -> bytes: ...

    def rename_artifact_series(
        self, *, principal: Principal, series: ArtifactSeries, name: str, idempotency_key: str
    ) -> ArtifactSeries: ...

    def revise_note(
        self, *, principal: Principal, note: NoteRevision, body: str, idempotency_key: str
    ) -> NoteRevision: ...


class ReviewsOperations(Protocol):
    def accept_review(
        self,
        *,
        principal: Principal,
        batch: ReviewBatch,
        candidate_version_ids: Iterable[ArtifactVersionId | str],
        idempotency_key: str,
    ) -> ReviewBatch: ...

    def active_review_batches(self) -> tuple[ReviewBatch, ...]: ...

    def cancel_review_batch(self, *, principal: Principal, batch: ReviewBatch, idempotency_key: str) -> ReviewBatch: ...

    def commit_review_return(
        self, *, principal: Principal, batch: ReviewBatch, source: PathLike[str] | str, idempotency_key: str
    ) -> object: ...

    def create_review_batch(
        self,
        *,
        principal: Principal,
        project_id: ProjectId | str,
        layer_id: LayerId | str,
        image_representation_id: RepresentationId | str,
        vector_representation_id: RepresentationId | str,
        coordinates: Iterable[tuple[int, int]],
        assignee_id: PerformerId | str,
        instructions: str = ...,
        due_at: datetime | None = ...,
        idempotency_key: str,
    ) -> ReviewBatch: ...

    def export_review_batch(
        self, *, principal: Principal, batch: ReviewBatch, destination: PathLike[str] | str, idempotency_key: str
    ) -> ReviewBatch: ...

    def request_review_changes(
        self, *, principal: Principal, batch: ReviewBatch, reason: str, idempotency_key: str
    ) -> ReviewBatch: ...

    def review_batches(self) -> tuple[ReviewBatch, ...]: ...

    def review_candidate_version_ids(self, batch: ReviewBatch) -> tuple[ArtifactVersionId, ...]: ...

    def review_return_preflight(
        self, *, principal: Principal, source: PathLike[str] | str, idempotency_key: str
    ) -> object: ...


class WorkOperations(Protocol):
    def cancel_work_order(self, order: WorkOrder, *, idempotency_key: str, principal: object = ...) -> WorkOrder: ...

    def complete_work_order(self, order: WorkOrder, *, idempotency_key: str, principal: object = ...) -> WorkOrder: ...

    def create_work_plan(
        self, *, project_id: object, title: object, principal: object, idempotency_key: object
    ) -> object: ...

    def get_work_plan(self, project_id: object, plan_id: object) -> object: ...

    def get_work_template(self, project_id: object, template_id: object, *, version: object = ...) -> object: ...

    def issue_work(
        self,
        *,
        principal: Principal,
        project_id: object,
        layer_id: object,
        kind: str,
        placement: str,
        assignee_id: object,
        coordinates: tuple[tuple[int, int], ...] = ...,
        image_representation_id: str = ...,
        vector_representation_id: str = ...,
        instructions: str = ...,
        due_at: datetime | None = ...,
        destination_parent: PathLike[str] | str | None = ...,
        idempotency_key: str,
        transport: str | None = ...,
        companion_representations: tuple[tuple[str, str], ...] | None = ...,
    ) -> WorkOrder: ...

    def list_work_plans(self, project_id: object) -> object: ...

    def list_work_templates(self, project_id: object) -> object: ...

    def mark_work_running(
        self, order: WorkOrder, *, job_id: str = ..., idempotency_key: str, principal: object = ...
    ) -> WorkOrder: ...

    def migrate_legacy_work_orders(
        self, *, principal: object, project_id: object, expected_plan_digest: object, idempotency_key: object
    ) -> object: ...

    def mutate_work_plan(
        self,
        *,
        project_id: object,
        plan_id: object,
        principal: object,
        expected_revision: object,
        operation: object,
        arguments: object,
        idempotency_key: object,
    ) -> object: ...

    def plan_legacy_work_order_migration(self, *, principal: object, project_id: object) -> object: ...

    def save_work_template(
        self,
        *,
        project_id: object,
        principal: object,
        definition: object,
        idempotency_key: object,
        template_id: object = ...,
        expected_version: object = ...,
    ) -> object: ...

    def work_orders(self) -> tuple[WorkOrder, ...]: ...

    def work_plan_analytics(self, project_id: object, *, principal: object, **request) -> object: ...


class PluginsOperations(Protocol):
    def cancel_plugin_job(
        self, *, principal: Principal, gateway: object, job: object, idempotency_key: str
    ) -> object: ...

    def plugin_jobs(self) -> tuple[object, ...]: ...

    def retry_plugin_job(
        self, *, principal: Principal, gateway: object, job: object, idempotency_key: str
    ) -> object: ...

    def submit_plugin_job(
        self,
        *,
        principal: Principal,
        gateway: object,
        project_id: ProjectId | str,
        layer_id: LayerId | str,
        source_representation_id: RepresentationId | str,
        target_representation_id: RepresentationId | str,
        coordinates: Iterable[tuple[int, int]],
        capability: str,
        parameters: Mapping[str, object],
        idempotency_key: str,
    ) -> object: ...

    def synchronize_plugin_jobs(self, *, principal: Principal, gateway: object) -> tuple[object, ...]: ...


class PipelinesOperations(Protocol):
    def command_layer_pipeline(
        self,
        *,
        project_id: object,
        layer_id: object,
        operation: object,
        principal: object,
        expected_revision: object,
        idempotency_key: object,
        node_id: object = ...,
    ) -> object: ...

    def define_layer_pipeline(
        self,
        *,
        project_id: object,
        layer_id: object,
        definition: object,
        principal: object,
        expected_revision: object,
        idempotency_key: object,
    ) -> object: ...

    def get_layer_pipeline(self, project_id: object, layer_id: object) -> object: ...

    def record_layer_pipeline_action(
        self,
        *,
        principal: Principal,
        project_id: ProjectId | str,
        layer_id: LayerId | str,
        action: str,
        node_id: str,
        plugin_id: str,
        capability: str,
        mode: str,
        parameters: dict[str, object] | None = ...,
    ) -> EventEnvelope: ...

    def remove_layer_pipeline_action(
        self,
        *,
        principal: Principal,
        project_id: ProjectId | str,
        layer_id: LayerId | str,
        action_event_id: str,
        remove_task: bool = ...,
    ) -> EventEnvelope: ...


class QualityOperations(Protocol):
    def create_quality_from_analysis(
        self,
        *,
        project_id: object,
        principal: object,
        run_id: object,
        rule: object,
        severity: object,
        idempotency_key: object,
    ) -> object: ...

    def create_quality_issue(
        self, *, project_id: object, principal: object, arguments: object, idempotency_key: object
    ) -> object: ...

    def get_quality_issue(self, project_id: object, issue_id: object) -> object: ...

    def list_quality_issues(
        self, project_id: object, *, artifact_version_id: object = ..., frame_id: object = ...
    ) -> object: ...

    def mutate_quality_issue(
        self,
        *,
        project_id: object,
        principal: object,
        issue_id: object,
        expected_revision: object,
        operation: object,
        arguments: object,
        idempotency_key: object,
    ) -> object: ...

    def quality_analysis_readiness(self, project_id: object, run_id: object) -> object: ...

    def quality_analysis_runs(self, project_id: object) -> object: ...

    def quality_frame_versions(self, project_id: object, layer_id: object, x: object, y: object) -> object: ...

    def quality_issue_history(self, project_id: object, issue_id: object) -> object: ...

    def recheck_quality_from_analysis(
        self,
        *,
        project_id: object,
        principal: object,
        issue_id: object,
        run_id: object,
        expected_revision: object,
        idempotency_key: object,
        note: object,
    ) -> object: ...


class PhysicalModelsOperations(Protocol):
    def stitch_block(
        self, project_id: object, job_id: object, column: int, row: int, *, coverage: bool = False
    ) -> tuple: ...

    def export_electrical_model(self, project_id: object) -> object: ...

    def export_stitch_vectors(
        self, *, project_id: object, job_id: object, destination: object, cancelled: object = ...
    ) -> object: ...

    def get_electrical_model(self, project_id: object) -> object: ...

    def get_multilayer_model(self, project_id: object) -> object: ...

    def import_stitch_observations(
        self, *, project_id: object, layer_id: object, principal: object, source_path: object, idempotency_key: object
    ) -> object: ...

    def mutate_electrical_model(
        self,
        *,
        project_id: object,
        principal: object,
        expected_revision: object,
        expected_physical_revision: object,
        operation: object,
        arguments: object,
        idempotency_key: object,
    ) -> object: ...

    def mutate_multilayer_model(
        self,
        *,
        project_id: object,
        principal: object,
        expected_revision: object,
        operation: object,
        arguments: object,
        idempotency_key: object,
    ) -> object: ...

    def set_layer_stitching_status(
        self,
        *,
        principal: Principal,
        project_id: object,
        layer_id: object,
        kind: str,
        completed: bool,
        version_id: str = ...,
    ) -> EventEnvelope: ...

    def stitch_vector_block(self, project_id: object, job_id: object, column: object, row: object) -> object: ...

    def stitch_vector_header(self, project_id: object, job_id: object) -> object: ...


class IdentitiesOperations(Protocol):
    def assign_project_role(
        self,
        *,
        principal: Principal,
        project: Project,
        target_principal_id: PrincipalId,
        role: ProjectRole,
        expected_revision: int,
        idempotency_key: str,
    ) -> frozenset[ProjectRole]: ...

    def list_performers(self, *, include_archived: bool = ...) -> tuple[Performer, ...]: ...

    def list_principals(self, *, include_inactive: bool = ...) -> tuple[Principal, ...]: ...

    def list_project_principals(
        self, project_id: ProjectId | str, *, include_inactive: bool = ...
    ) -> tuple[Principal, ...]: ...

    def project_permissions(self, project_id: ProjectId | str, principal: Principal) -> frozenset[Permission]: ...

    def project_role_revision(self, project_id: ProjectId | str, principal_id: PrincipalId | str) -> int: ...

    def project_roles(self, project_id: ProjectId | str, principal_id: PrincipalId | str) -> frozenset[ProjectRole]: ...

    def revoke_project_role(
        self,
        *,
        principal: Principal,
        project: Project,
        target_principal_id: PrincipalId,
        role: ProjectRole,
        expected_revision: int,
        idempotency_key: str,
    ) -> frozenset[ProjectRole]: ...


class WorkspaceOperations(Protocol):
    def commit_managed_import(
        self,
        *,
        principal: Principal,
        project: Project,
        layer: Layer,
        representation: Representation,
        plan: ImportPlan,
        idempotency_key: str,
    ) -> ManagedImportResult: ...

    def layer_file_binding(self, project_id: ProjectId | str, layer_id: LayerId | str) -> LayerFileBinding | None: ...

    def plan_import_directory(
        self,
        *,
        project: Project,
        directory: PathLike[str] | str,
        mode: ImportMappingMode = ...,
        regex: str | None = ...,
    ) -> ImportPlan: ...

    def project_workspace(self, project_id: ProjectId | str) -> ProjectWorkspaceBinding | None: ...

    def scan_layer_source(
        self, directory: PathLike[str] | str, *, maximum_frames: int, project_id: ProjectId | str
    ) -> LayerSourceScan: ...


class HistoryOperations(Protocol):
    def activity_records(self) -> tuple[ActivityRecord, ...]: ...

    def history(
        self, project_id: ProjectId | str, *, as_of: datetime | None = ..., event_types: Iterable[str] | None = ...
    ) -> tuple[object, ...]: ...

    def latest_karakal_analysis(
        self, project_id: ProjectId | str, layer_id: LayerId | str, *, as_of: datetime | None = ...
    ) -> KarakalAnalysisRun | None: ...

    def publish_karakal_analysis(
        self,
        *,
        principal: Principal,
        project_id: ProjectId | str,
        layer_id: LayerId | str,
        frame_confidence: dict[str, float],
        report: dict[str, object],
        parameters: dict[str, object],
        plugin_version: str,
        idempotency_key: str,
    ) -> KarakalAnalysisRun: ...


@runtime_checkable
class ProjectOperations(
    CatalogOperations,
    LayersOperations,
    RepresentationsOperations,
    ArtifactsOperations,
    ReviewsOperations,
    WorkOperations,
    PluginsOperations,
    PipelinesOperations,
    QualityOperations,
    PhysicalModelsOperations,
    IdentitiesOperations,
    WorkspaceOperations,
    HistoryOperations,
    Protocol,
):
    """Complete shared contract of project operations."""
