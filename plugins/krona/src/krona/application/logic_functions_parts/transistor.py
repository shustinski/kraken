"""logic_functions: transistor."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class _Transistor:
    instance_name: str
    kind: str  # "nmos" or "pmos"
    gate: str
    source: str
    drain: str
