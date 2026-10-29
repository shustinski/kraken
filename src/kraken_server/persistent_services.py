"""Production service facade backed by the clean-architecture application API.

Resource modules own the operations. This class installs them in one place so
callers still use a single service object, without a thirty-class inheritance list.
"""

from __future__ import annotations

from .persistent_services_parts.active_artifact_versions import ActiveArtifactVersionsMixin
from .persistent_services_parts.analysis_maps import AnalysisMapsMixin
from .persistent_services_parts.analysis_runs import AnalysisRunsMixin
from .persistent_services_parts.append_auxiliary_event import AppendAuxiliaryEventMixin
from .persistent_services_parts.append_pipeline_event import AppendPipelineEventMixin
from .persistent_services_parts.attach_workspace_layer import AttachWorkspaceLayerMixin
from .persistent_services_parts.construction import ConstructionMixin
from .persistent_services_parts.create_artifact_series import CreateArtifactSeriesMixin
from .persistent_services_parts.create_external_layer import CreateExternalLayerMixin
from .persistent_services_parts.create_project import CreateProjectMixin
from .persistent_services_parts.create_representation import CreateRepresentationMixin
from .persistent_services_parts.create_review_batch import CreateReviewBatchMixin
from .persistent_services_parts.discard_workspace_layer import DiscardWorkspaceLayerMixin
from .persistent_services_parts.electrical import ElectricalMixin
from .persistent_services_parts.import_agent_result import ImportAgentResultMixin
from .persistent_services_parts.issue_work import IssueWorkMixin
from .persistent_services_parts.layer_source_scan import ServerStorageProfiles, SystemClock
from .persistent_services_parts.materialize_representation_inputs import MaterializeRepresentationInputsMixin
from .persistent_services_parts.matrix_viewport import MatrixViewportMixin
from .persistent_services_parts.multilayer import MultilayerMixin
from .persistent_services_parts.mutate_artifact_series import MutateArtifactSeriesMixin
from .persistent_services_parts.mutate_review_batch import MutateReviewBatchMixin
from .persistent_services_parts.pipeline_execution import PipelineExecutionMixin
from .persistent_services_parts.quality import QualityMixin
from .persistent_services_parts.request_project_deletion import RequestProjectDeletionMixin
from .persistent_services_parts.start_desktop_job import StartDesktopJobMixin
from .persistent_services_parts.statistics import StatisticsMixin
from .persistent_services_parts.submit_plugin_job import SubmitPluginJobMixin
from .persistent_services_parts.update_representation import UpdateRepresentationMixin
from .persistent_services_parts.work_planning import WorkPlanningMixin

_RESOURCES = (
    RequestProjectDeletionMixin,
    ConstructionMixin,
    CreateProjectMixin,
    CreateExternalLayerMixin,
    MaterializeRepresentationInputsMixin,
    AttachWorkspaceLayerMixin,
    DiscardWorkspaceLayerMixin,
    CreateRepresentationMixin,
    UpdateRepresentationMixin,
    ActiveArtifactVersionsMixin,
    MatrixViewportMixin,
    CreateArtifactSeriesMixin,
    MutateArtifactSeriesMixin,
    CreateReviewBatchMixin,
    IssueWorkMixin,
    MutateReviewBatchMixin,
    StartDesktopJobMixin,
    SubmitPluginJobMixin,
    ImportAgentResultMixin,
    StatisticsMixin,
    AppendAuxiliaryEventMixin,
    AppendPipelineEventMixin,
    WorkPlanningMixin,
    QualityMixin,
    PipelineExecutionMixin,
    MultilayerMixin,
    ElectricalMixin,
    AnalysisRunsMixin,
    AnalysisMapsMixin,
)


def _contribute(facade: type, resources: tuple[type, ...]) -> None:
    """Install resource operations. Earlier resources win, as the old base list did."""
    seen: set[type] = set()
    for resource in reversed(resources):
        for cls in reversed(resource.__mro__):
            if cls is object or cls in seen:
                continue
            seen.add(cls)
            for name, value in cls.__dict__.items():
                if name.startswith("__") and name != "__init__":
                    continue
                setattr(facade, name, value)


class PostgresServerServices:
    """Transport facade. Mutations execute application command handlers."""


_contribute(PostgresServerServices, _RESOURCES)

__all__ = ["PostgresServerServices", "ServerStorageProfiles", "SystemClock"]
