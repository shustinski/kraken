"""patterns: compose master slave ffs."""
from __future__ import annotations
from itertools import combinations
from krona.analysis.structural.model import (
    CompositeFeature,
    RecognizedCellKind,
    RecognizedStructure,
)


class ComposeMasterSlaveFfsMixin:
    def _compose_master_slave_ffs(
        self,
        recognized: list[RecognizedStructure],
        patterns: PatternDatabase,
    ) -> list[RecognizedStructure]:
        latches = [item for item in recognized if item.kind == RecognizedCellKind.GATED_D_LATCH]
        if len(latches) < 2:
            return []
        inverse_pairs = {tuple(sorted(pair)) for pair in patterns.clock_analysis.inverse_pairs}
        composed: list[RecognizedStructure] = []
        for a, b in combinations(latches, 2):
            a_clk = a.clocks[0] if a.clocks else None
            b_clk = b.clocks[0] if b.clocks else None
            if not a_clk or not b_clk or a_clk == b_clk:
                continue
            if tuple(sorted((a_clk, b_clk))) not in inverse_pairs and not self._is_likely_complement_name_pair(a_clk, b_clk):
                continue
            if not a.outputs or not b.storage_nodes:
                continue
            a_to_b = bool(set(a.outputs).intersection(b.inputs))
            b_to_a = bool(set(b.outputs).intersection(a.inputs))
            if not a_to_b and not b_to_a:
                a_to_b = any(self._influence_path_exists(src, dst) for src in a.outputs for dst in b.storage_nodes)
                b_to_a = any(self._influence_path_exists(src, dst) for src in b.outputs for dst in a.storage_nodes)
            if a_to_b == b_to_a:
                continue
            master, slave = (a, b) if a_to_b else (b, a)
            base_clock = self._base_clock_name(master.clocks[0], slave.clocks[0]) or master.clocks[0]
            edge = self._infer_edge_polarity(master, slave, base_clock)
            features = self._unique_features([*master.features, *slave.features, CompositeFeature.CROSS_COUPLED_INVERTERS])
            controls = [*master.controls, *slave.controls, {"role": "clock", "net": base_clock, "activation": {"mode": "edge", "edge": edge}}]
            common_payload = {
                "storage_nodes": self._unique([*master.storage_nodes, *slave.storage_nodes]),
                "outputs": list(slave.outputs or slave.storage_nodes),
                "inputs": list(master.inputs),
                "clocks": [base_clock],
                "controls": controls,
                "features": features,
                "components": self._unique([*master.components, *slave.components]),
                "evidence": {
                    "master": master.as_dict(),
                    "slave": slave.as_dict(),
                    "phase_relation": "complementary_latches",
                },
            }
            composed.append(
                RecognizedStructure(
                    kind=RecognizedCellKind.MASTER_SLAVE_DFF,
                    confidence=0.88,
                    **common_payload,
                )
            )
            composed.append(
                RecognizedStructure(
                    kind=RecognizedCellKind.EDGE_TRIGGERED_DFF,
                    confidence=0.84,
                    **common_payload,
                )
            )
        return composed
