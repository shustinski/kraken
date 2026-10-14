"""patterns: classify bistable cores."""
from __future__ import annotations
from krona.analysis.structural.model import (
    CompositeFeature,
    RecognizedCellKind,
    RecognizedStructure,
)


class ClassifyBistableCoresMixin:
    def _classify_bistable_cores(self, patterns: PatternDatabase) -> list[RecognizedStructure]:
        recognized: list[RecognizedStructure] = []
        clock_set = set(patterns.clock_analysis.clock_candidates)

        for core in patterns.bistable_cores:
            core_set = set(core.nodes)
            core_muxes = [mux for mux in patterns.input_muxes if mux.storage_node in core_set]
            core_async = [ctrl for ctrl in patterns.async_controls if core_set.intersection(ctrl.target_nodes)]
            data_sources = sorted({src for mux in core_muxes for src in mux.data_sources})
            controls = self._controls_from_async(core_async)
            features: list[CompositeFeature] = []
            if core.evidence.cross_coupled_inverters:
                features.append(CompositeFeature.CROSS_COUPLED_INVERTERS)

            phase_profiles = self._analyze_core_phases(core, core_muxes, patterns.clock_analysis)
            enable_net = None
            enable_level = None
            if phase_profiles:
                enable_net, enable_level = phase_profiles[0].inferred_enable()

            if core_muxes:
                features.append(CompositeFeature.INPUT_MUX)
                if any(mux.feedback_sources for mux in core_muxes):
                    features.append(CompositeFeature.FEEDBACK_MUX)

            if enable_net and enable_net not in clock_set:
                controls.append(
                    {
                        "role": "enable",
                        "net": enable_net,
                        "activation": {"mode": "level", "level": enable_level or "unknown"},
                    }
                )
                features.append(CompositeFeature.CLOCK_ENABLE)

            if any(ctrl.role == "reset" for ctrl in core_async):
                features.append(CompositeFeature.ASYNC_RESET)
            if any(ctrl.role == "set" for ctrl in core_async):
                features.append(CompositeFeature.ASYNC_SET)

            inferred_clocks = []
            for mux in core_muxes:
                if mux.select_net and mux.select_net in clock_set:
                    inferred_clocks.append(mux.select_net)
            if phase_profiles and enable_net and enable_net in clock_set:
                inferred_clocks.append(enable_net)
            inferred_clocks = self._unique(inferred_clocks)

            kind = RecognizedCellKind.RS_LATCH
            confidence = 0.62
            if core_muxes and data_sources:
                kind = RecognizedCellKind.GATED_D_LATCH
                confidence = 0.8 if inferred_clocks else 0.72

            recognized.append(
                RecognizedStructure(
                    kind=kind,
                    confidence=confidence,
                    storage_nodes=list(core.nodes),
                    outputs=list(core.storage_outputs) or list(core.nodes),
                    inputs=data_sources,
                    clocks=inferred_clocks,
                    controls=controls,
                    features=self._unique_features(features),
                    evidence={
                        "bistable_core": {
                            "nodes": list(core.nodes),
                            "positive_feedback": core.evidence.positive_feedback,
                            "stable_states": core.evidence.stable_states,
                            "cross_coupled_inverters": [list(pair) for pair in core.evidence.cross_coupled_inverters],
                        },
                        "input_muxes": [mux.as_dict() for mux in core_muxes],
                        "async_controls": [ctrl.as_dict() for ctrl in core_async],
                        "phase_analysis": [self._phase_profile_to_dict(item) for item in phase_profiles],
                    },
                    components=self._core_component_devices(core),
                )
            )

        recognized.extend(self._compose_master_slave_ffs(recognized, patterns))
        return recognized
