from __future__ import annotations
from .host_contract import WidgetMixinHost

from .extraction_controls_mixin_parts.delete_fixed_via_row import DeleteFixedViaRowMixin
from .extraction_controls_mixin_parts.add_fixed_via_row import AddFixedViaRowMixin
from .extraction_controls_mixin_parts.apply_extraction_tooltips import ApplyExtractionTooltipsMixin
from .extraction_controls_mixin_parts.apply_bright_via_tooltips import ApplyBrightViaTooltipsMixin
from .extraction_controls_mixin_parts.update_via_size_controls import UpdateViaSizeControlsMixin
from .extraction_controls_mixin_parts.update_via_threshold_controls import UpdateViaThresholdControlsMixin
from .extraction_controls_mixin_parts.update_extraction_profile_controls import UpdateExtractionProfileControlsMixin

class WidgetExtractionControlsMixin(
    DeleteFixedViaRowMixin,
    AddFixedViaRowMixin,
    ApplyExtractionTooltipsMixin,
    ApplyBrightViaTooltipsMixin,
    UpdateViaSizeControlsMixin,
    UpdateViaThresholdControlsMixin,
    UpdateExtractionProfileControlsMixin,
    WidgetMixinHost,
):
    pass
