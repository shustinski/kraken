from __future__ import annotations
from dataclasses import dataclass
import math
import re
from PyQt6.QtGui import QBrush, QColor, QFont, QPen
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QGraphicsScene
from krona.application.ports import SceneData
from krona.infrastructure.edif_parser import EdifTextParser

from .scene_renderer_parts.parse_color import ParseColorMixin
from .scene_renderer_parts.draw_polyline import DrawPolylineMixin
from .scene_renderer_parts.draw_arc import DrawArcMixin
from .scene_renderer_parts.draw_text import DrawTextMixin
from .scene_renderer_parts.draw_instance_overlay_fallback import DrawInstanceOverlayFallbackMixin
from .scene_renderer_parts.pmos_instance import PmosInstanceMixin
from .scene_renderer_parts.draw_mos_contrast_overlay import DrawMosContrastOverlayMixin
from .scene_renderer_parts.draw_pmos_gate_bubble import DrawPmosGateBubbleMixin
from .scene_renderer_parts.draw_port_glyph_at import DrawPortGlyphAtMixin
from .scene_renderer_parts.draw_ground_glyph_at import DrawGroundGlyphAtMixin
from .scene_renderer_parts.draw_port_implementation import DrawPortImplementationMixin
from .scene_renderer_parts.draw_string_displays import DrawStringDisplaysMixin
from .scene_renderer_parts.draw_container import DrawContainerMixin
from .scene_renderer_parts.draw_top_page import DrawTopPageMixin
from .scene_renderer_parts.draw_logic_highlight_overlay import DrawLogicHighlightOverlayMixin
from .scene_renderer_parts.draw_instance_hitbox import DrawInstanceHitboxMixin
from .scene_renderer_parts.collect_drawable_points import CollectDrawablePointsMixin
from .scene_renderer_parts.affine2_d import (
    Affine2D,
    RenderStats,
    re_search_first_quoted,
)

class EdifSceneRenderer(
    ParseColorMixin,
    DrawPolylineMixin,
    DrawArcMixin,
    DrawTextMixin,
    DrawInstanceOverlayFallbackMixin,
    PmosInstanceMixin,
    DrawMosContrastOverlayMixin,
    DrawPmosGateBubbleMixin,
    DrawPortGlyphAtMixin,
    DrawGroundGlyphAtMixin,
    DrawPortImplementationMixin,
    DrawStringDisplaysMixin,
    DrawContainerMixin,
    DrawTopPageMixin,
    DrawLogicHighlightOverlayMixin,
    DrawInstanceHitboxMixin,
    CollectDrawablePointsMixin,
):
    pass

# Imported so existing callers and tests can patch or import these names
# from this module after the implementation moved into focused units.
_FACADE_EXPORTS = (
    Affine2D,
    RenderStats,
    re_search_first_quoted,
)
