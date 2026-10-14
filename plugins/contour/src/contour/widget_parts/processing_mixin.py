from __future__ import annotations
from .host_contract import WidgetMixinHost

from .processing_mixin_parts.module_state import _LOGGER
from .processing_mixin_parts.start_image_recognition_profile import StartImageRecognitionProfileMixin
from .processing_mixin_parts.contact_preview_profile import ContactPreviewProfileMixin
from .processing_mixin_parts.image_list_model_row import ImageListModelRowMixin
from .processing_mixin_parts.image_list_current_path import ImageListCurrentPathMixin
from .processing_mixin_parts.persist_current_overlay_changes import PersistCurrentOverlayChangesMixin
from .processing_mixin_parts.discard_current_vector_changes import DiscardCurrentVectorChangesMixin
from .processing_mixin_parts.prompt_transition_vector_save import PromptTransitionVectorSaveMixin
from .processing_mixin_parts.try_leave_current_frame import TryLeaveCurrentFrameMixin
from .processing_mixin_parts.suspend_editor_viewport_updates import SuspendEditorViewportUpdatesMixin
from .processing_mixin_parts.queue_editor_display_pixmap import QueueEditorDisplayPixmapMixin
from .processing_mixin_parts.apply_display_image_editor import ApplyDisplayImageEditorMixin
from .processing_mixin_parts.flush_pending_editor_frame import FlushPendingEditorFrameMixin
from .processing_mixin_parts.apply_frame_editor import ApplyFrameEditorMixin
from .processing_mixin_parts.queue_neighbor_frame_load import QueueNeighborFrameLoadMixin
from .processing_mixin_parts.apply_cached_neighbor_frames import ApplyCachedNeighborFramesMixin
from .processing_mixin_parts.sync_neighbor_frames import SyncNeighborFramesMixin
from .processing_mixin_parts.queue_preview_processing import QueuePreviewProcessingMixin
from .processing_mixin_parts.start_pending_preview_processing import StartPendingPreviewProcessingMixin
from .processing_mixin_parts.refresh_busy_indicator import RefreshBusyIndicatorMixin
from .processing_mixin_parts.preview_processing_result import PreviewProcessingResultMixin
from .processing_mixin_parts.polygons_edited import PolygonsEditedMixin
from .processing_mixin_parts.heuristic_contact_feedback_context import HeuristicContactFeedbackContextMixin
from .processing_mixin_parts.contact_center import ContactCenterMixin
from .processing_mixin_parts.manual_via_added import ManualViaAddedMixin
from .processing_mixin_parts.recognized_vias_deleted import RecognizedViasDeletedMixin
from .processing_mixin_parts.apply_contact_feedback_adjustments import ApplyContactFeedbackAdjustmentsMixin
from .processing_mixin_parts.highlight_contact_feedback_widget import HighlightContactFeedbackWidgetMixin
from .processing_mixin_parts.antialias_opened_cif_files import AntialiasOpenedCifFilesMixin
from .processing_mixin_parts.antialias_finished import AntialiasFinishedMixin
from .processing_mixin_parts.fix_internal_contours_all import FixInternalContoursAllMixin
from .processing_mixin_parts.begin_fix_internal_contours import BeginFixInternalContoursMixin
from .processing_mixin_parts.step_fix_internal_contours import StepFixInternalContoursMixin
from .processing_mixin_parts.complete_fix_internal_contours import CompleteFixInternalContoursMixin
from .processing_mixin_parts.fix_internal_contours_finished import FixInternalContoursFinishedMixin
from .processing_mixin_parts.restore_persisted_session_selection import RestorePersistedSessionSelectionMixin
from .processing_mixin_parts.apply_image_paths_workspace import ApplyImagePathsWorkspaceMixin
from .processing_mixin_parts.reset_project import ResetProjectMixin
from .processing_mixin_parts.load_cif_overlay_polygons import LoadCifOverlayPolygonsMixin
from .processing_mixin_parts.load_image import LoadImageMixin
from .processing_mixin_parts.begin_frame_load import BeginFrameLoadMixin
from .processing_mixin_parts.reload_current_frame_vectors import ReloadCurrentFrameVectorsMixin
from .processing_mixin_parts.begin_frame_vectors_reload import BeginFrameVectorsReloadMixin
from .processing_mixin_parts.drain_pending_frame_load import DrainPendingFrameLoadMixin
from .processing_mixin_parts.finish_frame_load_ui import FinishFrameLoadUiMixin
from .processing_mixin_parts.schedule_deferred_geometry_validation import ScheduleDeferredGeometryValidationMixin
from .processing_mixin_parts.start_deferred_geometry_validation import StartDeferredGeometryValidationMixin
from .processing_mixin_parts.schedule_invalid_polygon_description import ScheduleInvalidPolygonDescriptionMixin
from .processing_mixin_parts.format_invalid_polygon_description import FormatInvalidPolygonDescriptionMixin
from .processing_mixin_parts.offer_repair_invalid_polygon import OfferRepairInvalidPolygonMixin
from .processing_mixin_parts.prefetch_frame_neighborhood import PrefetchFrameNeighborhoodMixin
from .processing_mixin_parts.enqueue_prefetch_frame import EnqueuePrefetchFrameMixin
from .processing_mixin_parts.poll_frame_switch_profile import PollFrameSwitchProfileMixin
from .processing_mixin_parts.emit_cif_open_profile import EmitCifOpenProfileMixin
from .processing_mixin_parts.export_dataset_frame_state import ExportDatasetFrameStateMixin
from .processing_mixin_parts.save_current_result import SaveCurrentResultMixin
from .processing_mixin_parts.start_batch_processing import StartBatchProcessingMixin
from .processing_mixin_parts.handle_gamification_after_save import HandleGamificationAfterSaveMixin

class WidgetProcessingMixin(
    StartImageRecognitionProfileMixin,
    ContactPreviewProfileMixin,
    ImageListModelRowMixin,
    ImageListCurrentPathMixin,
    PersistCurrentOverlayChangesMixin,
    DiscardCurrentVectorChangesMixin,
    PromptTransitionVectorSaveMixin,
    TryLeaveCurrentFrameMixin,
    SuspendEditorViewportUpdatesMixin,
    QueueEditorDisplayPixmapMixin,
    ApplyDisplayImageEditorMixin,
    FlushPendingEditorFrameMixin,
    ApplyFrameEditorMixin,
    QueueNeighborFrameLoadMixin,
    ApplyCachedNeighborFramesMixin,
    SyncNeighborFramesMixin,
    QueuePreviewProcessingMixin,
    StartPendingPreviewProcessingMixin,
    RefreshBusyIndicatorMixin,
    PreviewProcessingResultMixin,
    PolygonsEditedMixin,
    HeuristicContactFeedbackContextMixin,
    ContactCenterMixin,
    ManualViaAddedMixin,
    RecognizedViasDeletedMixin,
    ApplyContactFeedbackAdjustmentsMixin,
    HighlightContactFeedbackWidgetMixin,
    AntialiasOpenedCifFilesMixin,
    AntialiasFinishedMixin,
    FixInternalContoursAllMixin,
    BeginFixInternalContoursMixin,
    StepFixInternalContoursMixin,
    CompleteFixInternalContoursMixin,
    FixInternalContoursFinishedMixin,
    RestorePersistedSessionSelectionMixin,
    ApplyImagePathsWorkspaceMixin,
    ResetProjectMixin,
    LoadCifOverlayPolygonsMixin,
    LoadImageMixin,
    BeginFrameLoadMixin,
    ReloadCurrentFrameVectorsMixin,
    BeginFrameVectorsReloadMixin,
    DrainPendingFrameLoadMixin,
    FinishFrameLoadUiMixin,
    ScheduleDeferredGeometryValidationMixin,
    StartDeferredGeometryValidationMixin,
    ScheduleInvalidPolygonDescriptionMixin,
    FormatInvalidPolygonDescriptionMixin,
    OfferRepairInvalidPolygonMixin,
    PrefetchFrameNeighborhoodMixin,
    EnqueuePrefetchFrameMixin,
    PollFrameSwitchProfileMixin,
    EmitCifOpenProfileMixin,
    ExportDatasetFrameStateMixin,
    SaveCurrentResultMixin,
    StartBatchProcessingMixin,
    HandleGamificationAfterSaveMixin,
    WidgetMixinHost,
):
    pass

# Imported so existing callers and tests can patch or import these names
# from this module after the implementation moved into focused units.
_FACADE_EXPORTS = (
    _LOGGER,
)
