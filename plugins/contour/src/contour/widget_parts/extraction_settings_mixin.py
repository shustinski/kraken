from __future__ import annotations
from .host_contract import WidgetMixinHost

from .extraction_settings_mixin_parts.try_extract_if_recognition import TryExtractIfRecognitionMixin
from .extraction_settings_mixin_parts.start_auto_tune_reference import StartAutoTuneReferenceMixin
from .extraction_settings_mixin_parts.apply_metal_strategy_combo import ApplyMetalStrategyComboMixin
from .extraction_settings_mixin_parts.extraction_settings import ExtractionSettingsMixin
from .extraction_settings_mixin_parts.bright_via_diameter_bounds import BrightViaDiameterBoundsMixin
from .extraction_settings_mixin_parts.current_contour_settings import CurrentContourSettingsMixin
from .extraction_settings_mixin_parts.recognition_mode_changed import RecognitionModeChangedMixin
from .extraction_settings_mixin_parts.restore_cif_overlay_after import RestoreCifOverlayAfterMixin
from .extraction_settings_mixin_parts.reset_metal_parameters import ResetMetalParametersMixin
from .extraction_settings_mixin_parts.recognition_status import RecognitionStatusMixin

class WidgetExtractionSettingsMixin(
    TryExtractIfRecognitionMixin,
    StartAutoTuneReferenceMixin,
    ApplyMetalStrategyComboMixin,
    ExtractionSettingsMixin,
    BrightViaDiameterBoundsMixin,
    CurrentContourSettingsMixin,
    RecognitionModeChangedMixin,
    RestoreCifOverlayAfterMixin,
    ResetMetalParametersMixin,
    RecognitionStatusMixin,
    WidgetMixinHost,
):
    pass
