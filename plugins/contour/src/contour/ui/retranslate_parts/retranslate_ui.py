"""retranslate: retranslate ui."""

from __future__ import annotations

from .retranslate_ui_parts.retranslate_ui import retranslate_ui

# Imported so existing callers and tests can patch or import these names
# from this module after the implementation moved into focused units.
_FACADE_EXPORTS = (
    retranslate_ui,
)
