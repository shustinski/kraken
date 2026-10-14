"""patterns: storage phase profile."""
from __future__ import annotations
from dataclasses import dataclass, field
from krona.analysis.structural.model import (
    BistableCoreEvidence,
    ConditionalEdge,
    LogicLevel,
)
from krona.analysis.structural.phases import ClockChainAnalysis

@dataclass(frozen=True)
class BistableCoreCandidate:
    nodes: tuple[str, ...]
    evidence: BistableCoreEvidence
    incoming_edges: tuple[ConditionalEdge, ...]
    outgoing_edges: tuple[ConditionalEdge, ...]
    storage_outputs: tuple[str, ...]
    cross_coupled_pair: tuple[str, str] | None = None

@dataclass(frozen=True)
class InputMuxCandidate:
    storage_node: str
    select_net: str | None
    data_sources: tuple[str, ...]
    feedback_sources: tuple[str, ...]
    edges: tuple[ConditionalEdge, ...]
    enable_net: str | None = None

    def as_dict(self) -> dict:
        return {
            "storage_node": self.storage_node,
            "select_net": self.select_net,
            "data_sources": list(self.data_sources),
            "feedback_sources": list(self.feedback_sources),
            "edges": [edge.as_dict() for edge in self.edges],
            "enable_net": self.enable_net,
        }

@dataclass(frozen=True)
class AsyncControlCandidate:
    signal: str
    role: str
    active_level: LogicLevel
    target_nodes: tuple[str, ...]
    path_edges: tuple[ConditionalEdge, ...]

    def as_dict(self) -> dict:
        return {
            "signal": self.signal,
            "role": self.role,
            "active_level": self.active_level.value,
            "asynchronous": True,
            "target_nodes": list(self.target_nodes),
            "path_edges": [edge.as_dict() for edge in self.path_edges],
        }

@dataclass(frozen=True)
class StoragePhaseProfile:
    storage_nodes: tuple[str, ...]
    data_sources: tuple[str, ...]
    feedback_sources: tuple[str, ...]
    control_signals: tuple[str, ...]
    transparency: dict[str, dict]
    hold_checks: list[dict]

    def inferred_enable(self) -> tuple[str | None, str | None]:
        if not self.transparency:
            return None, None
        counts: dict[tuple[str, bool], int] = {}
        for info in self.transparency.values():
            if not info.get("transparent"):
                continue
            assignment = info.get("assignment", {})
            for signal in self.control_signals:
                if signal in assignment:
                    counts[(signal, bool(assignment[signal]))] = counts.get((signal, bool(assignment[signal])), 0) + 1
        if not counts:
            return None, None
        (signal, level), _ = max(counts.items(), key=lambda item: (item[1], item[0][0]))
        return signal, ("high" if level else "low")

@dataclass(frozen=True)
class PatternDatabase:
    scc_summaries: list[dict]
    bistable_cores: list[BistableCoreCandidate]
    input_muxes: list[InputMuxCandidate]
    async_controls: list[AsyncControlCandidate]
    clock_analysis: ClockChainAnalysis
    clock_gating_features: list[dict]
    keeper_features: list[dict]
    dynamic_features: list[dict]
    xor_toggle_features: list[dict]
    toggle_chains: list[dict]
    diagnostics: list[str] = field(default_factory=list)
