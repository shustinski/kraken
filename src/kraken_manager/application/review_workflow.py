"""Application orchestration for signed review packages and immutable returns."""

from __future__ import annotations
import json
from collections.abc import Callable, Iterator, Mapping
from dataclasses import replace
from hashlib import sha256
from typing import Any
from kraken_manager.application.authorization import AuthorizationPolicy
from kraken_manager.application.dto import (
    AcceptReviewCommand,
    CancelReviewBatchCommand,
    CommitReviewReturnCommand,
    CreateReviewBatchCommand,
    DryRunReviewReturnCommand,
    ExportReviewPackageCommand,
    PlanReviewPackageCommand,
    RequestReviewChangesCommand,
    ReturnedFileDigest,
    ReviewPackagePlan,
    ReviewReturnCommitResult,
    ReviewReturnPlan,
    StoredContent,
)
from kraken_manager.application.errors import ConcurrencyError, ConflictError, NotFoundError
from kraken_manager.application.ports import (
    Clock,
    PerformerStore,
    ReviewPackageReader,
    ReviewPackageWriter,
    StorageProfileCatalog,
    UnitOfWork,
    UnitOfWorkFactory,
)
from kraken_manager.application.use_cases import (
    ReviewReturnComparator,
    _blob_store,
    _layer_stream,
    _prior_entity_id,
    _ProjectHandler,
    _series_stream,
)
from kraken_manager.domain.artifacts import ArtifactVersion
from kraken_manager.domain.common import ArtifactVersionId, ReviewBatchId, require_non_empty
from kraken_manager.domain.events import ActorSnapshot, EventEnvelope
from kraken_manager.domain.identity import Permission
from kraken_manager.domain.project import RepresentationKind, StructureState
from kraken_manager.domain.workflows import (
    ReviewBatch,
    ReviewBatchState,
    ReviewComparisonReport,
    ReviewFileComparison,
    ReviewFileStatus,
    ReviewItem,
    ReviewPackageFileV1,
    ReviewPackageManifestV1,
)

from .review_workflow_parts.call import CallMixin
from .review_workflow_parts.preview_return import PreviewReturnMixin
from .review_workflow_parts.result_event import ResultEventMixin
from .review_workflow_parts.review_handler import (
    _review_stream,
    _prior_command_event,
    _review_item_payload,
    _batch_payload,
    _comparison_payload,
    _report_from_payload,
    _artifact_created_payload,
    _ReviewHandler,
)
from .review_workflow_parts.create_review_batch_handler import CreateReviewBatchHandler
from .review_workflow_parts.review_package_planner import _ReviewPackagePlanner
from .review_workflow_parts.plan_review_package_handler import PlanReviewPackageHandler
from .review_workflow_parts.export_review_package_handler import ExportReviewPackageHandler
from .review_workflow_parts.review_return_inspector import _ReviewReturnInspector
from .review_workflow_parts.dry_run_review_return import DryRunReviewReturnHandler
from .review_workflow_parts.accept_review_handler import AcceptReviewHandler
from .review_workflow_parts.request_review_changes_handler import (
    RequestReviewChangesHandler,
    CancelReviewBatchHandler,
)

class CommitReviewReturnHandler(CallMixin, PreviewReturnMixin, ResultEventMixin, _ReviewReturnInspector):
    pass

# Imported so existing callers and tests can patch or import these names
# from this module after the implementation moved into focused units.
_FACADE_EXPORTS = (
    _review_stream,
    _prior_command_event,
    _review_item_payload,
    _batch_payload,
    _comparison_payload,
    _report_from_payload,
    _artifact_created_payload,
    _ReviewHandler,
    CreateReviewBatchHandler,
    _ReviewPackagePlanner,
    PlanReviewPackageHandler,
    ExportReviewPackageHandler,
    _ReviewReturnInspector,
    DryRunReviewReturnHandler,
    AcceptReviewHandler,
    RequestReviewChangesHandler,
    CancelReviewBatchHandler,
)

__all__ = [
    "AcceptReviewHandler",
    "CancelReviewBatchHandler",
    "CommitReviewReturnHandler",
    "CreateReviewBatchHandler",
    "DryRunReviewReturnHandler",
    "ExportReviewPackageHandler",
    "PlanReviewPackageHandler",
    "RequestReviewChangesHandler",
]
