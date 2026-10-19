"""patterns: analyze core phases."""
from __future__ import annotations
from typing import TYPE_CHECKING
from krona.analysis.structural.phases import ClockChainAnalysis

from krona.analysis.structural.patterns_parts.storage_phase_profile import StoragePhaseProfile


class AnalyzeCorePhasesMixin:
    def _analyze_core_phases(
        self,
        core: BistableCoreCandidate,
        muxes: list[InputMuxCandidate],
        clock_analysis: ClockChainAnalysis,
    ) -> list[StoragePhaseProfile]:
        if not muxes:
            return []
        profiles: list[StoragePhaseProfile] = []
        for mux in muxes:
            controls = []
            if mux.select_net:
                controls.append(mux.select_net)
            for a, b in clock_analysis.inverse_pairs:
                if mux.select_net == a and b not in controls:
                    controls.append(b)
                if mux.select_net == b and a not in controls:
                    controls.append(a)
            if not controls:
                continue
            transparency = self._phase.transparency_by_phase(
                data_sources=list(mux.data_sources),
                storage_nodes=[mux.storage_node],
                control_signals=controls,
                max_combinations=8,
            )
            hold_assignments = [
                {k: bool(v) for k, v in info.get("assignment", {}).items()}
                for info in transparency.values()
                if not info.get("transparent")
            ][:4]
            hold_checks = self._phase.hold_break_check(
                data_sources=list(mux.data_sources),
                storage_nodes=[mux.storage_node],
                hold_assignments=hold_assignments,
            )
            profiles.append(
                StoragePhaseProfile(
                    storage_nodes=(mux.storage_node,),
                    data_sources=tuple(mux.data_sources),
                    feedback_sources=tuple(mux.feedback_sources),
                    control_signals=tuple(controls),
                    transparency=transparency,
                    hold_checks=hold_checks,
                )
            )
        return profiles

if TYPE_CHECKING:
    from krona.analysis.structural.patterns_parts.storage_phase_profile import BistableCoreCandidate, InputMuxCandidate
