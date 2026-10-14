"""synthetic_topology: synthetic topology parameters."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class SyntheticTopologyParameters:
    trace_count: int = 5
    segment_count: int = 4
    trace_half_width: int = 2
    segment_count_range: tuple[int, int] | None = None
    trace_half_width_range: tuple[int, int] | None = None
    margin: int = 12
    background_noise_sigma: float = 0.02
    trace_noise_sigma: float = 0.01
    trace_clearance: int = 2
    topology_domain: str = 'pcb'
    topology_family: str = 'independent'
    via_count_range: tuple[int, int] | None = None
