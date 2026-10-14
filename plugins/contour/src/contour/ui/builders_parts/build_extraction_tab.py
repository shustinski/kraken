"""builders: build extraction tab."""

from __future__ import annotations

from .build_extraction_tab_parts.build_extraction_tab import build_extraction_tab

# Imported so existing callers and tests can patch or import these names
# from this module after the implementation moved into focused units.
_FACADE_EXPORTS = (
    build_extraction_tab,
)
