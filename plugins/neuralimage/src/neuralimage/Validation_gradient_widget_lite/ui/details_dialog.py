"""Details dialog for the extended validation gradient widget."""

from __future__ import annotations
import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QImage, QKeySequence, QPainter, QPen, QPixmap, QShortcut
from PyQt6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QFormLayout,
    QFrame,
    QGraphicsScene,
    QGraphicsView,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from ..core.analysis_modes import metric_level_key, metric_visual_ratio
from ..core.backend_constants import BCE_SCORE_CAP, MODEL_CONFIDENCE_UNCERTAIN_DELTA, POINT_SUPPORT_THRESHOLD, POLYGON_SUPPORT_THRESHOLD
from ..core.domain import BuildResult, ComparisonMode, FrameRecord
from ..core.subpixel_grid import (
    SubpixelGrid,
    SubpixelGridSpec,
    SubpixelSelection,
    build_subpixel_grid_from_pair,
    subpixel_bounds_for_index,
    subpixel_spec_from_options,
)
from ..core.repository import (
    _boundary_mask,
    _confidence_map_from_probability,
    _confidence_display_map_from_probability,
    _frame_uncertainty_components_from_probability,
    metric_higher_is_better,
    _paint_disk,
    _point_map_from_view,
    _support_weights_from_probability,
    _uncertainty_map_from_probability,
    compute_comparison,
    compute_comparison_score,
)
from ..core.confidence_maps import (
    DEFAULT_CONFIDENCE_BAD_AREA_THRESHOLD,
    build_algorithmic_uncertainty,
    build_model_uncertainty,
    combine_uncertainty_maps,
    confidence_bad_area_intensity,
)
from ..core.tile_grid import TileGridPlan, plan_tile_grid, tile_bounds_for_index
from ..core.workers import DetailConfidenceWorker, DetailPayloadWorker
from ..ui.i18n import Translator
from ..ui.ui_constants import DEFAULT_SUBPIXEL_AGGREGATION

from .details_dialog_parts.module_state import DETAIL_PIXEL_VIEW_THRESHOLD
from .details_dialog_parts.construction import ConstructionMixin
from .details_dialog_parts.build_ui import BuildUiMixin
from .details_dialog_parts.connect_signals import ConnectSignalsMixin
from .details_dialog_parts.populate_model_combos import PopulateModelCombosMixin
from .details_dialog_parts.component_name_label import ComponentNameLabelMixin
from .details_dialog_parts.result_kind_hint import ResultKindHintMixin
from .details_dialog_parts.comparison_presets import ComparisonPresetsMixin
from .details_dialog_parts.step_subpixel_selection import StepSubpixelSelectionMixin
from .details_dialog_parts.start_confidence_loading import StartConfidenceLoadingMixin
from .details_dialog_parts.confidence_heatmap import ConfidenceHeatmapMixin
from .details_dialog_parts.algorithmic_uncertainty_map import AlgorithmicUncertaintyMapMixin
from .details_dialog_parts.confidence_mix_pixmap import ConfidenceMixPixmapMixin
from .details_dialog_parts.point_object_confidence_overlay import PointObjectConfidenceOverlayMixin
from .details_dialog_parts.confidence_branch_debug_pixmap import ConfidenceBranchDebugPixmapMixin
from .details_dialog_parts.selected_comparison_score_info import SelectedComparisonScoreInfoMixin
from .details_dialog_parts.source_mask import SourceMaskMixin
from .details_dialog_parts.refresh_result_kind_options import RefreshResultKindOptionsMixin
from .details_dialog_parts.point_matches_pixmap import PointMatchesPixmapMixin
from .details_dialog_parts.source_pixmap import SourcePixmapMixin
from .details_dialog_parts.refresh_scene import RefreshSceneMixin
from .details_dialog_parts.refresh_result_layer import RefreshResultLayerMixin
from .details_dialog_parts.update_tile_lod import UpdateTileLodMixin
from .details_dialog_parts.quick_confidence_score import QuickConfidenceScoreMixin
from .details_dialog_parts.comparison_score_style import ComparisonScoreStyleMixin
from .details_dialog_parts.refresh_info import RefreshInfoMixin
from .details_dialog_parts.build_view_settings_payload import BuildViewSettingsPayloadMixin
from .details_dialog_parts.restore_view_settings import RestoreViewSettingsMixin
from .details_dialog_parts.overlay_graphics_view import _OverlayGraphicsView
from .details_dialog_parts.expandable_detail_score_card import _ExpandableDetailScoreCard

class ExtendFrameDetailsDialog(
    ConstructionMixin,
    BuildUiMixin,
    ConnectSignalsMixin,
    PopulateModelCombosMixin,
    ComponentNameLabelMixin,
    ResultKindHintMixin,
    ComparisonPresetsMixin,
    StepSubpixelSelectionMixin,
    StartConfidenceLoadingMixin,
    ConfidenceHeatmapMixin,
    AlgorithmicUncertaintyMapMixin,
    ConfidenceMixPixmapMixin,
    PointObjectConfidenceOverlayMixin,
    ConfidenceBranchDebugPixmapMixin,
    SelectedComparisonScoreInfoMixin,
    SourceMaskMixin,
    RefreshResultKindOptionsMixin,
    PointMatchesPixmapMixin,
    SourcePixmapMixin,
    RefreshSceneMixin,
    RefreshResultLayerMixin,
    UpdateTileLodMixin,
    QuickConfidenceScoreMixin,
    ComparisonScoreStyleMixin,
    RefreshInfoMixin,
    BuildViewSettingsPayloadMixin,
    RestoreViewSettingsMixin,
    QDialog,
):
    """Show standard frame comparisons for two selected models."""

# Backward-compatible alias for legacy lite imports.
LiteFrameDetailsDialog = ExtendFrameDetailsDialog

# Imported so existing callers and tests can patch or import these names
# from this module after the implementation moved into focused units.
_FACADE_EXPORTS = (
    DETAIL_PIXEL_VIEW_THRESHOLD,
    _OverlayGraphicsView,
    _ExpandableDetailScoreCard,
)
