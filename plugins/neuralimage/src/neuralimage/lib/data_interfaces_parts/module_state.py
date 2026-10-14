"""data_interfaces: module state."""
from __future__ import annotations

RANDOM_ARTIFACT_TYPES: tuple[str, ...] = (
    'dust',
    'resist_residue',
    'etch_residue',
    'particle_cluster',
    'flake',
)

_PCB_DEFECT_TYPE_ALIASES: dict[str, str] = {
    'break': 'break',
    'break_defect': 'break',
    'short': 'short',
    'short_circuit': 'short',
    'missing': 'missing_copper',
    'missing_copper': 'missing_copper',
    'excess': 'excess_copper',
    'excess_copper': 'excess_copper',
    'pinhole': 'pinhole',
    'small_hole': 'pinhole',
    'spurious': 'spurious_copper',
    'spurious_copper': 'spurious_copper',
    'via': 'via',
    'via_defect': 'via',
    'via_defects': 'via',
    'misalignment': 'misalignment',
}

SYNTHETIC_TOPOLOGY_DOMAINS: tuple[str, str] = ('pcb', 'ic')

PCB_TOPOLOGY_FAMILIES: tuple[str, ...] = (
    'pcb_mixed',
    'pcb_parallel',
    'pcb_independent',
    'pcb_fanout',
)

IC_TOPOLOGY_FAMILIES: tuple[str, ...] = (
    'ic_mixed',
    'ic_cell_array',
    'ic_channels',
    'ic_tree',
)

_IC_DEFECT_TYPE_ALIASES: dict[str, str] = {
    'line_break': 'line_break',
    'bridge': 'bridge',
    'necking': 'necking',
    'missing_metal': 'missing_metal',
    'spur': 'spur',
    'pinhole': 'pinhole',
    'via_open': 'via_open',
    'line_shift': 'line_shift',
}
