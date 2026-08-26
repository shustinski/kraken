"""Extension points for later interlayer alignment. Not used in Cartograph v1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .coordinates import Translation2D
from .tiles import TileGrid


@dataclass(frozen=True, slots=True)
class LayerAlignment:
    translation: Translation2D
    confidence: float
    message: str = ""


class LayerAligner(Protocol):
    """Align a moving process layer to a reference layer. Iteration 2+."""

    def align(self, reference: TileGrid, moving: TileGrid) -> LayerAlignment:
        """Return a coarse interlayer translation. Topology constraints come later."""


class TopologyConstraintProvider(Protocol):
    """Future source of vectorized topology constraints (gates/contacts)."""

    def constraints_for(self, grid: TileGrid) -> tuple[object, ...]:
        """Return topology constraints used by a later LayerAligner."""
