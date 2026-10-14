from __future__ import annotations
from .host_contract import WidgetMixinHost

from .navigation_mixin_parts.module_state import _LOGGER
from .navigation_mixin_parts.refresh_asset_filter_list import RefreshAssetFilterListMixin
from .navigation_mixin_parts.rebuild_asset_filter_lists import RebuildAssetFilterListsMixin
from .navigation_mixin_parts.take_scrollable_control_panel import TakeScrollableControlPanelMixin
from .navigation_mixin_parts.configure_thumbnail_grid_geometry import ConfigureThumbnailGridGeometryMixin
from .navigation_mixin_parts.rebuild_thumbnail_grid import RebuildThumbnailGridMixin
from .navigation_mixin_parts.reseed_thumbnail_radial_fill import ReseedThumbnailRadialFillMixin
from .navigation_mixin_parts.resume_thumbnail_radial_fill import ResumeThumbnailRadialFillMixin
from .navigation_mixin_parts.flush_thumbnail_icon_batch import FlushThumbnailIconBatchMixin
from .navigation_mixin_parts.queue_thumbnail_load import QueueThumbnailLoadMixin
from .navigation_mixin_parts.schedule_visible_thumbnail_loads import ScheduleVisibleThumbnailLoadsMixin
from .navigation_mixin_parts.update_thumbnail_grid_selection import UpdateThumbnailGridSelectionMixin
from .navigation_mixin_parts.scroll_thumbnail_grid_row import ScrollThumbnailGridRowMixin
from .navigation_mixin_parts.handle_frame_matrix_arrow import HandleFrameMatrixArrowMixin
from .navigation_mixin_parts.activate_frame_path_matrix import ActivateFramePathMatrixMixin
from .navigation_mixin_parts.sync_after_cif_index import SyncAfterCifIndexMixin
from .navigation_mixin_parts.refresh_extra_layers_list import RefreshExtraLayersListMixin
from .navigation_mixin_parts.build_extra_layer_row import BuildExtraLayerRowMixin
from .navigation_mixin_parts.extra_layer_directory import ExtraLayerDirectoryMixin

class WidgetNavigationMixin(
    RefreshAssetFilterListMixin,
    RebuildAssetFilterListsMixin,
    TakeScrollableControlPanelMixin,
    ConfigureThumbnailGridGeometryMixin,
    RebuildThumbnailGridMixin,
    ReseedThumbnailRadialFillMixin,
    ResumeThumbnailRadialFillMixin,
    FlushThumbnailIconBatchMixin,
    QueueThumbnailLoadMixin,
    ScheduleVisibleThumbnailLoadsMixin,
    UpdateThumbnailGridSelectionMixin,
    ScrollThumbnailGridRowMixin,
    HandleFrameMatrixArrowMixin,
    ActivateFramePathMatrixMixin,
    SyncAfterCifIndexMixin,
    RefreshExtraLayersListMixin,
    BuildExtraLayerRowMixin,
    ExtraLayerDirectoryMixin,
    WidgetMixinHost,
):
    pass

# Imported so existing callers and tests can patch or import these names
# from this module after the implementation moved into focused units.
_FACADE_EXPORTS = (
    _LOGGER,
)
