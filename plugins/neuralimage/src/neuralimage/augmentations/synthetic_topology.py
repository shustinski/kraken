from __future__ import annotations
import math
import random
from dataclasses import dataclass, replace
import cv2
import numpy as np

from .synthetic_topology_parts.generate import GenerateMixin
from .synthetic_topology_parts.build_pcb_tree_family import BuildPcbTreeFamilyMixin
from .synthetic_topology_parts.build_ic_mixed_family import BuildIcMixedFamilyMixin
from .synthetic_topology_parts.build_ic_tree_family import BuildIcTreeFamilyMixin
from .synthetic_topology_parts.build_ic_cell_array import BuildIcCellArrayMixin
from .synthetic_topology_parts.build_pcb_route_polyline import BuildPcbRoutePolylineMixin
from .synthetic_topology_parts.resolve_irregular_axis_positions import ResolveIrregularAxisPositionsMixin
from .synthetic_topology_parts.pattern_route import PatternRouteMixin
from .synthetic_topology_parts.place_ic_channel_blockages import PlaceIcChannelBlockagesMixin
from .synthetic_topology_parts.build_ic_channel_row import BuildIcChannelRowMixin
from .synthetic_topology_parts.build_ic_cell_row import BuildIcCellRowMixin
from .synthetic_topology_parts.build_ic_tree_component import BuildIcTreeComponentMixin
from .synthetic_topology_parts.plan_ic_window_slots import PlanIcWindowSlotsMixin
from .synthetic_topology_parts.build_ic_tree_backbone import BuildIcTreeBackboneMixin
from .synthetic_topology_parts.resolve_polyline_y_at import ResolvePolylineYAtMixin
from .synthetic_topology_parts.build_ic_serpentine_rect import BuildIcSerpentineRectMixin
from .synthetic_topology_parts.decorate_pcb_terminals import DecoratePcbTerminalsMixin
from .synthetic_topology_parts.allow_candidate_overlap import AllowCandidateOverlapMixin
from .synthetic_topology_parts.build_trace_polyline import BuildTracePolylineMixin
from .synthetic_topology_parts.segments_intersect import SegmentsIntersectMixin
from .synthetic_topology_parts.render_ic_grayscale import RenderIcGrayscaleMixin
from .synthetic_topology_parts.render_pcb_rgb import RenderPcbRgbMixin
from .synthetic_topology_parts.channel_first import ChannelFirstMixin
from .synthetic_topology_parts.synthetic_topology_parameters import SyntheticTopologyParameters

class SyntheticTopologyGenerator(
    GenerateMixin,
    BuildPcbTreeFamilyMixin,
    BuildIcMixedFamilyMixin,
    BuildIcTreeFamilyMixin,
    BuildIcCellArrayMixin,
    BuildPcbRoutePolylineMixin,
    ResolveIrregularAxisPositionsMixin,
    PatternRouteMixin,
    PlaceIcChannelBlockagesMixin,
    BuildIcChannelRowMixin,
    BuildIcCellRowMixin,
    BuildIcTreeComponentMixin,
    PlanIcWindowSlotsMixin,
    BuildIcTreeBackboneMixin,
    ResolvePolylineYAtMixin,
    BuildIcSerpentineRectMixin,
    DecoratePcbTerminalsMixin,
    AllowCandidateOverlapMixin,
    BuildTracePolylineMixin,
    SegmentsIntersectMixin,
    RenderIcGrayscaleMixin,
    RenderPcbRgbMixin,
    ChannelFirstMixin,
):
    """Generate a consistent synthetic topology pair: grayscale image + binary mask."""
    _MIXED_FAMILIES: tuple[tuple[str, float], ...] = (
        ('independent', 0.28),
        ('parallel', 0.28),
        ('tree', 0.22),
        ('cell_array', 0.22),
    )
    _PCB_FAMILIES: dict[str, str] = {
        'pcb_parallel': 'parallel',
        'pcb_independent': 'independent',
        'pcb_fanout': 'tree',
    }
    _IC_FAMILIES: dict[str, str] = {
        'ic_cell_array': 'cell_array',
        'ic_channels': 'parallel',
        'ic_tree': 'tree',
    }
    _PCB_MIXED_FAMILIES: tuple[tuple[str, float], ...] = (
        ('independent', 0.36),
        ('parallel', 0.34),
        ('tree', 0.30),
    )
    _IC_MIXED_FAMILIES: tuple[tuple[str, float], ...] = (
        ('cell_array', 0.42),
        ('parallel', 0.35),
        ('tree', 0.23),
    )

# Imported so existing callers and tests can patch or import these names
# from this module after the implementation moved into focused units.
_FACADE_EXPORTS = (
    SyntheticTopologyParameters,
)
