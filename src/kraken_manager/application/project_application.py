"""Storage-independent project application boundary for Qt, CLI and other callers.

The resolver is injected by the composition root. No operation inspects storage
profiles, knows an HTTP client, or falls back to another catalog.
"""

from __future__ import annotations
from kraken_manager.application.results import KarakalAnalysisRun

from collections.abc import Callable, Iterable, Mapping
from datetime import datetime
from os import PathLike
from typing import Any

from kraken_manager.application.imports import ImportMappingMode, ImportPlan
from kraken_manager.application.project_operations import ProjectOperations
from kraken_manager.application.ports import ProjectCommand
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
from kraken_manager.domain.identity import Permission, Principal, ProjectRole
from kraken_manager.domain.project import (
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


class ProjectApplication:
    def execute_project_command(self, command: ProjectCommand) -> object:
        return self._project_operations(command.project_id).execute_project_command(command)

    def stitch_block(
        self, project_id: object, job_id: object, column: int, row: int, *, coverage: bool = False
    ) -> tuple:
        return self._project_operations(project_id).stitch_block(project_id, job_id, column, row, coverage=coverage)

    def __init__(self, resolve: Callable[[object], ProjectOperations]):
        self._resolve = resolve

    def _project_operations(self, project_id: object) -> ProjectOperations:
        return self._resolve(project_id)

    def accept_review(
        self,
        *,
        principal: Principal,
        batch: ReviewBatch,
        candidate_version_ids: Iterable[ArtifactVersionId | str],
        idempotency_key: str,
    ) -> ReviewBatch:
        return self._project_operations(batch.project_id).accept_review(
            principal=principal,
            batch=batch,
            candidate_version_ids=candidate_version_ids,
            idempotency_key=idempotency_key,
        )

    def activate_artifact_version(
        self,
        *,
        principal: Principal,
        project_id: ProjectId | str,
        series_id: ArtifactSeriesId | str,
        version_id: ArtifactVersionId | str,
        idempotency_key: str,
    ) -> ArtifactVersion:
        return self._project_operations(project_id).activate_artifact_version(
            principal=principal,
            project_id=project_id,
            series_id=series_id,
            version_id=version_id,
            idempotency_key=idempotency_key,
        )

    def activate_representation(
        self,
        *,
        principal: Principal,
        project: Project,
        layer: Layer,
        representation: Representation,
        idempotency_key: str,
    ) -> Representation:
        return self._project_operations(project.id).activate_representation(
            principal=principal,
            project=project,
            layer=layer,
            representation=representation,
            idempotency_key=idempotency_key,
        )

    def active_artifact_version(
        self, project_id: ProjectId | str, series_id: ArtifactSeriesId | str
    ) -> ArtifactVersion | None:
        return self._project_operations(project_id).active_artifact_version(project_id, series_id)

    def add_external_artifact_version(
        self,
        *,
        principal: Principal,
        project_id: ProjectId | str,
        series_id: ArtifactSeriesId | str,
        source: PathLike[str] | str,
        parent_version_id: ArtifactVersionId | str | None = None,
        idempotency_key: str,
    ) -> ArtifactVersion:
        return self._project_operations(project_id).add_external_artifact_version(
            principal=principal,
            project_id=project_id,
            series_id=series_id,
            source=source,
            parent_version_id=parent_version_id,
            idempotency_key=idempotency_key,
        )

    def add_managed_artifact_version(
        self,
        *,
        principal: Principal,
        project_id: ProjectId | str,
        series_id: ArtifactSeriesId | str,
        source: PathLike[str] | str,
        parent_version_id: ArtifactVersionId | str | None = None,
        idempotency_key: str,
    ) -> ArtifactVersion:
        return self._project_operations(project_id).add_managed_artifact_version(
            principal=principal,
            project_id=project_id,
            series_id=series_id,
            source=source,
            parent_version_id=parent_version_id,
            idempotency_key=idempotency_key,
        )

    def archive_artifact_series(
        self, *, principal: Principal, series: ArtifactSeries, idempotency_key: str
    ) -> ArtifactSeries:
        return self._project_operations(series.project_id).archive_artifact_series(
            principal=principal, series=series, idempotency_key=idempotency_key
        )

    def archive_layer(self, *, principal: Principal, project: Project, layer: Layer, idempotency_key: str) -> Layer:
        return self._project_operations(project.id).archive_layer(
            principal=principal, project=project, layer=layer, idempotency_key=idempotency_key
        )

    def archive_project(self, *, principal: Principal, project: Project, idempotency_key: str) -> Project:
        return self._project_operations(project.id).archive_project(
            principal=principal, project=project, idempotency_key=idempotency_key
        )

    def archive_representation(
        self,
        *,
        principal: Principal,
        project: Project,
        layer: Layer,
        representation: Representation,
        idempotency_key: str,
    ) -> Representation:
        return self._project_operations(project.id).archive_representation(
            principal=principal,
            project=project,
            layer=layer,
            representation=representation,
            idempotency_key=idempotency_key,
        )

    def artifact_source_path(
        self, project_id: ProjectId | str, artifact_version_id: ArtifactVersionId | str
    ) -> PathLike[str]:
        return self._project_operations(project_id).artifact_source_path(project_id, artifact_version_id)

    def artifact_stream_revision(self, project_id: ProjectId | str, series_id: ArtifactSeriesId | str) -> int:
        return self._project_operations(project_id).artifact_stream_revision(project_id, series_id)

    def artifact_version(
        self, project_id: ProjectId | str, version_id: ArtifactVersionId | str
    ) -> ArtifactVersion | None:
        return self._project_operations(project_id).artifact_version(project_id, version_id)

    def artifact_versions(
        self, project_id: ProjectId | str, series_id: ArtifactSeriesId | str
    ) -> tuple[ArtifactVersion, ...]:
        return self._project_operations(project_id).artifact_versions(project_id, series_id)

    def assign_project_role(
        self,
        *,
        principal: Principal,
        project: Project,
        target_principal_id: PrincipalId,
        role: ProjectRole,
        expected_revision: int,
        idempotency_key: str,
    ) -> frozenset[ProjectRole]:
        return self._project_operations(project.id).assign_project_role(
            principal=principal,
            project=project,
            target_principal_id=target_principal_id,
            role=role,
            expected_revision=expected_revision,
            idempotency_key=idempotency_key,
        )

    def cancel_plugin_job(self, *, principal: Principal, gateway: object, job: object, idempotency_key: str) -> object:
        return self._project_operations(job.project_id).cancel_plugin_job(
            principal=principal, gateway=gateway, job=job, idempotency_key=idempotency_key
        )

    def cancel_review_batch(self, *, principal: Principal, batch: ReviewBatch, idempotency_key: str) -> ReviewBatch:
        return self._project_operations(batch.project_id).cancel_review_batch(
            principal=principal, batch=batch, idempotency_key=idempotency_key
        )

    def cancel_work_order(self, order: WorkOrder, *, idempotency_key: str, principal: object = None) -> WorkOrder:
        return self._project_operations(order.project_id).cancel_work_order(
            order, idempotency_key=idempotency_key, principal=principal
        )

    def command_layer_pipeline(
        self,
        *,
        project_id: object,
        layer_id: object,
        operation: object,
        principal: object,
        expected_revision: object,
        idempotency_key: object,
        node_id: object = None,
    ) -> object:
        return self._project_operations(project_id).command_layer_pipeline(
            project_id=project_id,
            layer_id=layer_id,
            operation=operation,
            principal=principal,
            expected_revision=expected_revision,
            idempotency_key=idempotency_key,
            node_id=node_id,
        )

    def commit_managed_import(
        self,
        *,
        principal: Principal,
        project: Project,
        layer: Layer,
        representation: Representation,
        plan: ImportPlan,
        idempotency_key: str,
    ) -> ManagedImportResult:
        return self._project_operations(project.id).commit_managed_import(
            principal=principal,
            project=project,
            layer=layer,
            representation=representation,
            plan=plan,
            idempotency_key=idempotency_key,
        )

    def commit_review_return(
        self, *, principal: Principal, batch: ReviewBatch, source: PathLike[str] | str, idempotency_key: str
    ) -> object:
        return self._project_operations(batch.project_id).commit_review_return(
            principal=principal, batch=batch, source=source, idempotency_key=idempotency_key
        )

    def complete_work_order(self, order: WorkOrder, *, idempotency_key: str, principal: object = None) -> WorkOrder:
        return self._project_operations(order.project_id).complete_work_order(
            order, idempotency_key=idempotency_key, principal=principal
        )

    def create_artifact_series(
        self,
        *,
        principal: Principal,
        project_id: ProjectId | str,
        scope: ArtifactScope,
        name: str,
        layer_id: LayerId | str | None = None,
        representation_id: RepresentationId | str | None = None,
        frame_id: str | None = None,
        series_id: ArtifactSeriesId | str | None = None,
        idempotency_key: str,
    ) -> ArtifactSeries:
        return self._project_operations(project_id).create_artifact_series(
            principal=principal,
            project_id=project_id,
            scope=scope,
            name=name,
            layer_id=layer_id,
            representation_id=representation_id,
            frame_id=frame_id,
            series_id=series_id,
            idempotency_key=idempotency_key,
        )

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
    ) -> tuple[Layer, LayerFileBinding, Representation]:
        return self._project_operations(project.id).create_external_layer(
            principal=principal,
            project=project,
            name=name,
            layer_type=layer_type,
            order=order,
            image_directory=image_directory,
            ssc_directory=ssc_directory,
            prv_directory=prv_directory,
            idempotency_key=idempotency_key,
        )

    def create_layer(
        self,
        *,
        principal: Principal,
        project: Project,
        name: str,
        layer_type: LayerType,
        order: int,
        idempotency_key: str,
        layer_id: LayerId | str | None = None,
    ) -> Layer:
        return self._project_operations(project.id).create_layer(
            principal=principal,
            project=project,
            name=name,
            layer_type=layer_type,
            order=order,
            idempotency_key=idempotency_key,
            layer_id=layer_id,
        )

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
        progress: object = None,
        cancelled: object = None,
    ) -> tuple[Layer, LayerFileBinding, Representation]:
        return self._project_operations(project.id).create_layer_from_disk(
            principal=principal,
            project=project,
            name=name,
            layer_type=layer_type,
            order=order,
            scan=scan,
            conversion=conversion,
            idempotency_key=idempotency_key,
            progress=progress,
            cancelled=cancelled,
        )

    def create_note(
        self,
        *,
        principal: Principal,
        project_id: ProjectId | str,
        body: str,
        layer_id: LayerId | str | None = None,
        frame_id: str | None = None,
        idempotency_key: str,
    ) -> NoteRevision:
        return self._project_operations(project_id).create_note(
            principal=principal,
            project_id=project_id,
            body=body,
            layer_id=layer_id,
            frame_id=frame_id,
            idempotency_key=idempotency_key,
        )

    def create_quality_from_analysis(
        self,
        *,
        project_id: object,
        principal: object,
        run_id: object,
        rule: object,
        severity: object,
        idempotency_key: object,
    ) -> object:
        return self._project_operations(project_id).create_quality_from_analysis(
            project_id=project_id,
            principal=principal,
            run_id=run_id,
            rule=rule,
            severity=severity,
            idempotency_key=idempotency_key,
        )

    def create_quality_issue(
        self, *, project_id: object, principal: object, arguments: object, idempotency_key: object
    ) -> object:
        return self._project_operations(project_id).create_quality_issue(
            project_id=project_id, principal=principal, arguments=arguments, idempotency_key=idempotency_key
        )

    def create_representation(
        self,
        *,
        principal: Principal,
        project: Project,
        layer: Layer,
        name: str,
        kind: RepresentationKind,
        idempotency_key: str,
        note: str = "",
        source: str | None = None,
        source_image_representation_id: object = None,
        active: bool = False,
        purpose: RepresentationPurpose = RepresentationPurpose.SOURCE,
    ) -> Representation:
        return self._project_operations(project.id).create_representation(
            principal=principal,
            project=project,
            layer=layer,
            name=name,
            kind=kind,
            idempotency_key=idempotency_key,
            note=note,
            source=source,
            source_image_representation_id=source_image_representation_id,
            active=active,
            purpose=purpose,
        )

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
        instructions: str = "",
        due_at: datetime | None = None,
        idempotency_key: str,
    ) -> ReviewBatch:
        return self._project_operations(project_id).create_review_batch(
            principal=principal,
            project_id=project_id,
            layer_id=layer_id,
            image_representation_id=image_representation_id,
            vector_representation_id=vector_representation_id,
            coordinates=coordinates,
            assignee_id=assignee_id,
            instructions=instructions,
            due_at=due_at,
            idempotency_key=idempotency_key,
        )

    def create_work_plan(
        self, *, project_id: object, title: object, principal: object, idempotency_key: object
    ) -> object:
        return self._project_operations(project_id).create_work_plan(
            project_id=project_id, title=title, principal=principal, idempotency_key=idempotency_key
        )

    def deactivate_representation(
        self,
        *,
        principal: Principal,
        project: Project,
        layer: Layer,
        representation: Representation,
        idempotency_key: str,
    ) -> Representation:
        return self._project_operations(project.id).deactivate_representation(
            principal=principal,
            project=project,
            layer=layer,
            representation=representation,
            idempotency_key=idempotency_key,
        )

    def define_layer_pipeline(
        self,
        *,
        project_id: object,
        layer_id: object,
        definition: object,
        principal: object,
        expected_revision: object,
        idempotency_key: object,
    ) -> object:
        return self._project_operations(project_id).define_layer_pipeline(
            project_id=project_id,
            layer_id=layer_id,
            definition=definition,
            principal=principal,
            expected_revision=expected_revision,
            idempotency_key=idempotency_key,
        )

    def delete_layer(
        self, *, principal: Principal, project: Project, layer: Layer, confirmation_name: str, idempotency_key: str
    ) -> Layer:
        return self._project_operations(project.id).delete_layer(
            principal=principal,
            project=project,
            layer=layer,
            confirmation_name=confirmation_name,
            idempotency_key=idempotency_key,
        )

    def export_artifact_version(
        self, project_id: ProjectId | str, version: ArtifactVersion, destination: PathLike[str] | str
    ) -> PathLike[str]:
        return self._project_operations(project_id).export_artifact_version(project_id, version, destination)

    def export_electrical_model(self, project_id: object) -> object:
        return self._project_operations(project_id).export_electrical_model(project_id)

    def export_managed_artifact(
        self, project_id: ProjectId | str, version: ArtifactVersion, destination: PathLike[str] | str
    ) -> PathLike[str]:
        return self._project_operations(project_id).export_managed_artifact(project_id, version, destination)

    def export_review_batch(
        self, *, principal: Principal, batch: ReviewBatch, destination: PathLike[str] | str, idempotency_key: str
    ) -> ReviewBatch:
        return self._project_operations(batch.project_id).export_review_batch(
            principal=principal, batch=batch, destination=destination, idempotency_key=idempotency_key
        )

    def export_stitch_vectors(
        self, *, project_id: object, job_id: object, destination: object, cancelled: object = lambda: False
    ) -> object:
        return self._project_operations(project_id).export_stitch_vectors(
            project_id=project_id, job_id=job_id, destination=destination, cancelled=cancelled
        )

    def frame_cells(
        self,
        project_id: ProjectId | str,
        layer_id: LayerId | str,
        representation_id: RepresentationId | str,
        *,
        as_of: datetime | None = None,
        bounds: tuple[int, int, int, int] | None = None,
    ) -> tuple[FrameCellSnapshot, ...]:
        return self._project_operations(project_id).frame_cells(
            project_id, layer_id, representation_id, as_of=as_of, bounds=bounds
        )

    def frame_management_states(
        self,
        project_id: ProjectId | str,
        layer_id: LayerId | str,
        representation_id: RepresentationId | str,
        *,
        as_of: datetime | None = None,
    ) -> tuple[FrameManagementState, ...]:
        return self._project_operations(project_id).frame_management_states(
            project_id, layer_id, representation_id, as_of=as_of
        )

    def get_artifact_version(
        self, project_id: ProjectId | str, artifact_version_id: ArtifactVersionId | str
    ) -> ArtifactVersion | None:
        return self._project_operations(project_id).get_artifact_version(project_id, artifact_version_id)

    def get_electrical_model(self, project_id: object) -> object:
        return self._project_operations(project_id).get_electrical_model(project_id)

    def get_layer_pipeline(self, project_id: object, layer_id: object) -> object:
        return self._project_operations(project_id).get_layer_pipeline(project_id, layer_id)

    def get_multilayer_model(self, project_id: object) -> object:
        return self._project_operations(project_id).get_multilayer_model(project_id)

    def get_project(self, project_id: ProjectId | str, *, as_of: datetime | None = None) -> Project | None:
        return self._project_operations(project_id).get_project(project_id, as_of=as_of)

    def get_quality_issue(self, project_id: object, issue_id: object) -> object:
        return self._project_operations(project_id).get_quality_issue(project_id, issue_id)

    def get_work_plan(self, project_id: object, plan_id: object) -> object:
        return self._project_operations(project_id).get_work_plan(project_id, plan_id)

    def get_work_template(self, project_id: object, template_id: object, *, version: object = None) -> object:
        return self._project_operations(project_id).get_work_template(project_id, template_id, version=version)

    def history(
        self, project_id: ProjectId | str, *, as_of: datetime | None = None, event_types: Iterable[str] | None = None
    ) -> tuple[object, ...]:
        return self._project_operations(project_id).history(project_id, as_of=as_of, event_types=event_types)

    def import_stitch_observations(
        self, *, project_id: object, layer_id: object, principal: object, source_path: object, idempotency_key: object
    ) -> object:
        return self._project_operations(project_id).import_stitch_observations(
            project_id=project_id,
            layer_id=layer_id,
            principal=principal,
            source_path=source_path,
            idempotency_key=idempotency_key,
        )

    def issue_work(
        self,
        *,
        principal: Principal,
        project_id: object,
        layer_id: object,
        kind: str,
        placement: str,
        assignee_id: object,
        coordinates: tuple[tuple[int, int], ...] = (),
        image_representation_id: str = "",
        vector_representation_id: str = "",
        instructions: str = "",
        due_at: datetime | None = None,
        destination_parent: PathLike[str] | str | None = None,
        idempotency_key: str,
        transport: str | None = None,
        companion_representations: tuple[tuple[str, str], ...] | None = None,
    ) -> WorkOrder:
        return self._project_operations(project_id).issue_work(
            principal=principal,
            project_id=project_id,
            layer_id=layer_id,
            kind=kind,
            placement=placement,
            assignee_id=assignee_id,
            coordinates=coordinates,
            image_representation_id=image_representation_id,
            vector_representation_id=vector_representation_id,
            instructions=instructions,
            due_at=due_at,
            destination_parent=destination_parent,
            idempotency_key=idempotency_key,
            transport=transport,
            companion_representations=companion_representations,
        )

    def latest_karakal_analysis(
        self, project_id: ProjectId | str, layer_id: LayerId | str, *, as_of: datetime | None = None
    ) -> KarakalAnalysisRun | None:
        return self._project_operations(project_id).latest_karakal_analysis(project_id, layer_id, as_of=as_of)

    def layer_file_binding(self, project_id: ProjectId | str, layer_id: LayerId | str) -> LayerFileBinding | None:
        return self._project_operations(project_id).layer_file_binding(project_id, layer_id)

    def list_artifact_series(
        self,
        project_id: ProjectId | str,
        *,
        layer_id: LayerId | str | None = None,
        representation_id: RepresentationId | str | None = None,
        include_archived: bool = False,
    ) -> tuple[ArtifactSeries, ...]:
        return self._project_operations(project_id).list_artifact_series(
            project_id, layer_id=layer_id, representation_id=representation_id, include_archived=include_archived
        )

    def list_layers(
        self, project_id: ProjectId | str, *, as_of: datetime | None = None, include_archived: bool = False
    ) -> tuple[Layer, ...]:
        return self._project_operations(project_id).list_layers(
            project_id, as_of=as_of, include_archived=include_archived
        )

    def list_notes(
        self, project_id: ProjectId | str, *, layer_id: LayerId | str | None = None, frame_id: str | None = None
    ) -> tuple[NoteRevision, ...]:
        return self._project_operations(project_id).list_notes(project_id, layer_id=layer_id, frame_id=frame_id)

    def list_project_principals(
        self, project_id: ProjectId | str, *, include_inactive: bool = False
    ) -> tuple[Principal, ...]:
        return self._project_operations(project_id).list_project_principals(
            project_id, include_inactive=include_inactive
        )

    def list_quality_issues(
        self, project_id: object, *, artifact_version_id: object = None, frame_id: object = None
    ) -> object:
        return self._project_operations(project_id).list_quality_issues(
            project_id, artifact_version_id=artifact_version_id, frame_id=frame_id
        )

    def list_representations(
        self,
        project_id: ProjectId | str,
        layer_id: LayerId | str,
        *,
        include_archived: bool = False,
        as_of: datetime | None = None,
    ) -> tuple[Representation, ...]:
        return self._project_operations(project_id).list_representations(
            project_id, layer_id, include_archived=include_archived, as_of=as_of
        )

    def list_work_plans(self, project_id: object) -> object:
        return self._project_operations(project_id).list_work_plans(project_id)

    def list_work_templates(self, project_id: object) -> object:
        return self._project_operations(project_id).list_work_templates(project_id)

    def managed_artifact_path(self, project_id: ProjectId | str, version_id: ArtifactVersionId | str) -> PathLike[str]:
        return self._project_operations(project_id).managed_artifact_path(project_id, version_id)

    def mark_work_running(
        self, order: WorkOrder, *, job_id: str = "", idempotency_key: str, principal: object = None
    ) -> WorkOrder:
        return self._project_operations(order.project_id).mark_work_running(
            order, job_id=job_id, idempotency_key=idempotency_key, principal=principal
        )

    def materialize_representation_inputs(
        self, *, principal: Principal, project: Project, layer: Layer, representation: Representation
    ) -> tuple[FrameCellSnapshot, ...]:
        return self._project_operations(project.id).materialize_representation_inputs(
            principal=principal, project=project, layer=layer, representation=representation
        )

    def matrix_viewport(
        self,
        project_id: ProjectId | str,
        *,
        layer_id: LayerId | str,
        representation_ids: Iterable[RepresentationId | str] = (),
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        lod: int = 0,
        include_missing: bool = True,
    ) -> dict[str, Any]:
        return self._project_operations(project_id).matrix_viewport(
            project_id,
            layer_id=layer_id,
            representation_ids=representation_ids,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            lod=lod,
            include_missing=include_missing,
        )

    def migrate_legacy_work_orders(
        self, *, principal: object, project_id: object, expected_plan_digest: object, idempotency_key: object
    ) -> object:
        return self._project_operations(project_id).migrate_legacy_work_orders(
            principal=principal,
            project_id=project_id,
            expected_plan_digest=expected_plan_digest,
            idempotency_key=idempotency_key,
        )

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
    ) -> object:
        return self._project_operations(project_id).mutate_electrical_model(
            project_id=project_id,
            principal=principal,
            expected_revision=expected_revision,
            expected_physical_revision=expected_physical_revision,
            operation=operation,
            arguments=arguments,
            idempotency_key=idempotency_key,
        )

    def mutate_multilayer_model(
        self,
        *,
        project_id: object,
        principal: object,
        expected_revision: object,
        operation: object,
        arguments: object,
        idempotency_key: object,
    ) -> object:
        return self._project_operations(project_id).mutate_multilayer_model(
            project_id=project_id,
            principal=principal,
            expected_revision=expected_revision,
            operation=operation,
            arguments=arguments,
            idempotency_key=idempotency_key,
        )

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
    ) -> object:
        return self._project_operations(project_id).mutate_quality_issue(
            project_id=project_id,
            principal=principal,
            issue_id=issue_id,
            expected_revision=expected_revision,
            operation=operation,
            arguments=arguments,
            idempotency_key=idempotency_key,
        )

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
    ) -> object:
        return self._project_operations(project_id).mutate_work_plan(
            project_id=project_id,
            plan_id=plan_id,
            principal=principal,
            expected_revision=expected_revision,
            operation=operation,
            arguments=arguments,
            idempotency_key=idempotency_key,
        )

    def plan_import_directory(
        self,
        *,
        project: Project,
        directory: PathLike[str] | str,
        mode: ImportMappingMode = ImportMappingMode.XY_FILENAME,
        regex: str | None = None,
    ) -> ImportPlan:
        return self._project_operations(project.id).plan_import_directory(
            project=project, directory=directory, mode=mode, regex=regex
        )

    def plan_legacy_work_order_migration(self, *, principal: object, project_id: object) -> object:
        return self._project_operations(project_id).plan_legacy_work_order_migration(
            principal=principal, project_id=project_id
        )

    def project_permissions(self, project_id: ProjectId | str, principal: Principal) -> frozenset[Permission]:
        return self._project_operations(project_id).project_permissions(project_id, principal)

    def project_role_revision(self, project_id: ProjectId | str, principal_id: PrincipalId | str) -> int:
        return self._project_operations(project_id).project_role_revision(project_id, principal_id)

    def project_roles(self, project_id: ProjectId | str, principal_id: PrincipalId | str) -> frozenset[ProjectRole]:
        return self._project_operations(project_id).project_roles(project_id, principal_id)

    def project_workspace(self, project_id: ProjectId | str) -> ProjectWorkspaceBinding | None:
        return self._project_operations(project_id).project_workspace(project_id)

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
    ) -> KarakalAnalysisRun:
        return self._project_operations(project_id).publish_karakal_analysis(
            principal=principal,
            project_id=project_id,
            layer_id=layer_id,
            frame_confidence=frame_confidence,
            report=report,
            parameters=parameters,
            plugin_version=plugin_version,
            idempotency_key=idempotency_key,
        )

    def quality_analysis_readiness(self, project_id: object, run_id: object) -> object:
        return self._project_operations(project_id).quality_analysis_readiness(project_id, run_id)

    def quality_analysis_runs(self, project_id: object) -> object:
        return self._project_operations(project_id).quality_analysis_runs(project_id)

    def quality_frame_versions(self, project_id: object, layer_id: object, x: object, y: object) -> object:
        return self._project_operations(project_id).quality_frame_versions(project_id, layer_id, x, y)

    def quality_issue_history(self, project_id: object, issue_id: object) -> object:
        return self._project_operations(project_id).quality_issue_history(project_id, issue_id)

    def read_project_blob(self, project_id: ProjectId | str, sha256: str) -> bytes:
        return self._project_operations(project_id).read_project_blob(project_id, sha256)

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
    ) -> object:
        return self._project_operations(project_id).recheck_quality_from_analysis(
            project_id=project_id,
            principal=principal,
            issue_id=issue_id,
            run_id=run_id,
            expected_revision=expected_revision,
            idempotency_key=idempotency_key,
            note=note,
        )

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
        parameters: dict[str, object] | None = None,
    ) -> EventEnvelope:
        return self._project_operations(project_id).record_layer_pipeline_action(
            principal=principal,
            project_id=project_id,
            layer_id=layer_id,
            action=action,
            node_id=node_id,
            plugin_id=plugin_id,
            capability=capability,
            mode=mode,
            parameters=parameters,
        )

    def remove_layer_pipeline_action(
        self,
        *,
        principal: Principal,
        project_id: ProjectId | str,
        layer_id: LayerId | str,
        action_event_id: str,
        remove_task: bool = False,
    ) -> EventEnvelope:
        return self._project_operations(project_id).remove_layer_pipeline_action(
            principal=principal,
            project_id=project_id,
            layer_id=layer_id,
            action_event_id=action_event_id,
            remove_task=remove_task,
        )

    def rename_artifact_series(
        self, *, principal: Principal, series: ArtifactSeries, name: str, idempotency_key: str
    ) -> ArtifactSeries:
        return self._project_operations(series.project_id).rename_artifact_series(
            principal=principal, series=series, name=name, idempotency_key=idempotency_key
        )

    def rename_layer(
        self, *, principal: Principal, project: Project, layer: Layer, name: str, idempotency_key: str
    ) -> Layer:
        return self._project_operations(project.id).rename_layer(
            principal=principal, project=project, layer=layer, name=name, idempotency_key=idempotency_key
        )

    def rename_project(self, *, principal: Principal, project: Project, name: str, idempotency_key: str) -> Project:
        return self._project_operations(project.id).rename_project(
            principal=principal, project=project, name=name, idempotency_key=idempotency_key
        )

    def rename_representation(
        self,
        *,
        principal: Principal,
        project: Project,
        layer: Layer,
        representation: Representation,
        name: str,
        idempotency_key: str,
    ) -> Representation:
        return self._project_operations(project.id).rename_representation(
            principal=principal,
            project=project,
            layer=layer,
            representation=representation,
            name=name,
            idempotency_key=idempotency_key,
        )

    def reorder_layer(
        self, *, principal: Principal, project: Project, layer: Layer, order: int, idempotency_key: str
    ) -> Layer:
        return self._project_operations(project.id).reorder_layer(
            principal=principal, project=project, layer=layer, order=order, idempotency_key=idempotency_key
        )

    def reorder_layers(
        self,
        *,
        principal: Principal,
        project: Project,
        layers: Iterable[Layer],
        layer_ids: Iterable[LayerId | str],
        idempotency_key: str,
    ) -> tuple[Layer, ...]:
        return self._project_operations(project.id).reorder_layers(
            principal=principal, project=project, layers=layers, layer_ids=layer_ids, idempotency_key=idempotency_key
        )

    def request_review_changes(
        self, *, principal: Principal, batch: ReviewBatch, reason: str, idempotency_key: str
    ) -> ReviewBatch:
        return self._project_operations(batch.project_id).request_review_changes(
            principal=principal, batch=batch, reason=reason, idempotency_key=idempotency_key
        )

    def restore_project(self, *, principal: Principal, project: Project, idempotency_key: str) -> Project:
        return self._project_operations(project.id).restore_project(
            principal=principal, project=project, idempotency_key=idempotency_key
        )

    def retry_plugin_job(self, *, principal: Principal, gateway: object, job: object, idempotency_key: str) -> object:
        return self._project_operations(job.project_id).retry_plugin_job(
            principal=principal, gateway=gateway, job=job, idempotency_key=idempotency_key
        )

    def review_candidate_version_ids(self, batch: ReviewBatch) -> tuple[ArtifactVersionId, ...]:
        return self._project_operations(batch.project_id).review_candidate_version_ids(batch)

    def revise_note(self, *, principal: Principal, note: NoteRevision, body: str, idempotency_key: str) -> NoteRevision:
        return self._project_operations(note.project_id).revise_note(
            principal=principal, note=note, body=body, idempotency_key=idempotency_key
        )

    def revoke_project_role(
        self,
        *,
        principal: Principal,
        project: Project,
        target_principal_id: PrincipalId,
        role: ProjectRole,
        expected_revision: int,
        idempotency_key: str,
    ) -> frozenset[ProjectRole]:
        return self._project_operations(project.id).revoke_project_role(
            principal=principal,
            project=project,
            target_principal_id=target_principal_id,
            role=role,
            expected_revision=expected_revision,
            idempotency_key=idempotency_key,
        )

    def save_work_template(
        self,
        *,
        project_id: object,
        principal: object,
        definition: object,
        idempotency_key: object,
        template_id: object = None,
        expected_version: object = 0,
    ) -> object:
        return self._project_operations(project_id).save_work_template(
            project_id=project_id,
            principal=principal,
            definition=definition,
            idempotency_key=idempotency_key,
            template_id=template_id,
            expected_version=expected_version,
        )

    def scan_layer_source(
        self, directory: PathLike[str] | str, *, maximum_frames: int, project_id: ProjectId | str
    ) -> LayerSourceScan:
        return self._project_operations(project_id).scan_layer_source(
            directory, maximum_frames=maximum_frames, project_id=project_id
        )

    def set_layer_stitching_status(
        self,
        *,
        principal: Principal,
        project_id: object,
        layer_id: object,
        kind: str,
        completed: bool,
        version_id: str = "",
    ) -> EventEnvelope:
        return self._project_operations(project_id).set_layer_stitching_status(
            principal=principal,
            project_id=project_id,
            layer_id=layer_id,
            kind=kind,
            completed=completed,
            version_id=version_id,
        )

    def stitch_vector_block(self, project_id: object, job_id: object, column: object, row: object) -> object:
        return self._project_operations(project_id).stitch_vector_block(project_id, job_id, column, row)

    def stitch_vector_header(self, project_id: object, job_id: object) -> object:
        return self._project_operations(project_id).stitch_vector_header(project_id, job_id)

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
    ) -> object:
        return self._project_operations(project_id).submit_plugin_job(
            principal=principal,
            gateway=gateway,
            project_id=project_id,
            layer_id=layer_id,
            source_representation_id=source_representation_id,
            target_representation_id=target_representation_id,
            coordinates=coordinates,
            capability=capability,
            parameters=parameters,
            idempotency_key=idempotency_key,
        )

    def update_representation_note(
        self,
        *,
        principal: Principal,
        project: Project,
        layer: Layer,
        representation: Representation,
        note: str,
        idempotency_key: str,
    ) -> Representation:
        return self._project_operations(project.id).update_representation_note(
            principal=principal,
            project=project,
            layer=layer,
            representation=representation,
            note=note,
            idempotency_key=idempotency_key,
        )

    def work_plan_analytics(self, project_id: object, *, principal: object, **request) -> object:
        return self._project_operations(project_id).work_plan_analytics(project_id, principal=principal, **request)
