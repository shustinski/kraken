"""Placeholder interfaces for future graph and GNN scoring extensions."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class StructuralFeatureBundle:
    """Store scalar structural features prepared for future graph scoring."""

    geometry_mode: str
    frame_key: str
    features: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GraphBuildArtifact:
    """Describe one graph artifact without committing to a concrete GNN backend."""

    node_count: int = 0
    edge_count: int = 0
    node_feature_names: tuple[str, ...] = ()
    edge_feature_names: tuple[str, ...] = ()


@runtime_checkable
class GraphScorerProtocol(Protocol):
    """Future GNN scorer contract kept separate from the current analytics pipeline."""

    def score_frame(self, artifact: GraphBuildArtifact, features: StructuralFeatureBundle) -> float:
        """Return one scalar score for the frame graph."""
