"""i18n_content: module state."""

from __future__ import annotations

from .module_state_parts.module_state import (
    LocalizedTextMap,
    EXTRACTION_HELP_TEXTS,
    PIPELINE_PARAMETER_HELP_TEXTS,
    PIPELINE_CONTROL_TOOLTIPS,
    EDITOR_TOOL_TOOLTIPS,
    EDITOR_ACTION_TOOLTIPS,
    GENERAL_CONTROL_TOOLTIPS,
    PIPELINE_OPERATION_GROUPS,
    PIPELINE_OPERATION_HELP_TEXTS,
)

# Imported so existing callers and tests can patch or import these names
# from this module after the implementation moved into focused units.
_FACADE_EXPORTS = (
    LocalizedTextMap,
    EXTRACTION_HELP_TEXTS,
    PIPELINE_PARAMETER_HELP_TEXTS,
    PIPELINE_CONTROL_TOOLTIPS,
    EDITOR_TOOL_TOOLTIPS,
    EDITOR_ACTION_TOOLTIPS,
    GENERAL_CONTROL_TOOLTIPS,
    PIPELINE_OPERATION_GROUPS,
    PIPELINE_OPERATION_HELP_TEXTS,
)
