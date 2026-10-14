"""patterns: run."""
from __future__ import annotations
from krona.analysis.structural.graphs import GraphBundle
from krona.analysis.structural.model import (
    CircuitModel,
)
from krona.analysis.structural.phases import PhaseAnalyzer

from krona.analysis.structural.patterns_parts.storage_phase_profile import PatternDatabase


class RunMixin:
    def __init__(self, circuit: CircuitModel, graphs: GraphBundle, phase_analyzer: PhaseAnalyzer):
        self._circuit = circuit
        self._graphs = graphs
        self._phase = phase_analyzer
        self._power = circuit.power_nets()
        self._ground = circuit.ground_nets()

    def run(self) -> PatternDatabase:
        clock_analysis = self._phase.analyze_clock_chains()
        scc_summaries, bistable_cores = self._detect_bistable_cores()
        input_muxes = self._detect_input_muxes(bistable_cores, clock_analysis)
        async_controls = self._detect_async_controls(bistable_cores)
        clock_gating = self._detect_clock_gating(clock_analysis)
        dynamic_features = self._detect_dynamic_precharge_evaluate(clock_analysis)
        keeper_features = self._detect_keepers(dynamic_features, bistable_cores)
        xor_toggle = self._detect_xor_toggle_feedback()
        toggle_chains = self._detect_toggle_chains(xor_toggle)

        diagnostics = list(self._graphs.diagnostics)
        return PatternDatabase(
            scc_summaries=scc_summaries,
            bistable_cores=bistable_cores,
            input_muxes=input_muxes,
            async_controls=async_controls,
            clock_analysis=clock_analysis,
            clock_gating_features=clock_gating,
            keeper_features=keeper_features,
            dynamic_features=dynamic_features,
            xor_toggle_features=xor_toggle,
            toggle_chains=toggle_chains,
            diagnostics=diagnostics,
        )
