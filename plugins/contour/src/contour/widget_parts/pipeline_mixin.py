from __future__ import annotations
from .host_contract import WidgetMixinHost

from .pipeline_mixin_parts.populate_pipeline_operations import PopulatePipelineOperationsMixin
from .pipeline_mixin_parts.render_pipeline_parameters import RenderPipelineParametersMixin
from .pipeline_mixin_parts.color_selection_entries import ColorSelectionEntriesMixin
from .pipeline_mixin_parts.render_color_binarize_parameters import RenderColorBinarizeParametersMixin
from .pipeline_mixin_parts.remove_selected_color_entry import RemoveSelectedColorEntryMixin
from .pipeline_mixin_parts.refresh_via_template_list import RefreshViaTemplateListMixin
from .pipeline_mixin_parts.editor_image_region_selected import EditorImageRegionSelectedMixin
from .pipeline_mixin_parts.apply_via_preset_payload import ApplyViaPresetPayloadMixin
from .pipeline_mixin_parts.current_metal_preset_payload import CurrentMetalPresetPayloadMixin
from .pipeline_mixin_parts.export_selected_metal_preset import ExportSelectedMetalPresetMixin
from .pipeline_mixin_parts.import_metal_preset import ImportMetalPresetMixin
from .pipeline_mixin_parts.heuristic_default_values import HeuristicDefaultValuesMixin
from .pipeline_mixin_parts.reset_bright_via_parameters import ResetBrightViaParametersMixin
from .pipeline_mixin_parts.editor_image_clicked import EditorImageClickedMixin
from .pipeline_mixin_parts.color_selection_entry import ColorSelectionEntry

class WidgetPipelineMixin(
    PopulatePipelineOperationsMixin,
    RenderPipelineParametersMixin,
    ColorSelectionEntriesMixin,
    RenderColorBinarizeParametersMixin,
    RemoveSelectedColorEntryMixin,
    RefreshViaTemplateListMixin,
    EditorImageRegionSelectedMixin,
    ApplyViaPresetPayloadMixin,
    CurrentMetalPresetPayloadMixin,
    ExportSelectedMetalPresetMixin,
    ImportMetalPresetMixin,
    HeuristicDefaultValuesMixin,
    ResetBrightViaParametersMixin,
    EditorImageClickedMixin,
    WidgetMixinHost,
):
    pass

# Imported so existing callers and tests can patch or import these names
# from this module after the implementation moved into focused units.
_FACADE_EXPORTS = (
    ColorSelectionEntry,
)
