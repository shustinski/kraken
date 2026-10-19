"""patterns: upgrade jk t counter."""
from __future__ import annotations
from typing import TYPE_CHECKING
from krona.analysis.structural.model import (
    CompositeFeature,
    RecognizedCellKind,
    RecognizedStructure,
)


class UpgradeJkTCounterMixin:
    def _upgrade_jk_t_counter(self, recognized: list[RecognizedStructure], patterns: PatternDatabase) -> list[RecognizedStructure]:
        upgraded = list(recognized)
        xor_toggle_by_ff = {str(item.get("ff_device")): item for item in patterns.xor_toggle_features}
        for item in list(recognized):
            if item.kind not in {RecognizedCellKind.D_FLIP_FLOP, RecognizedCellKind.MASTER_SLAVE_DFF, RecognizedCellKind.EDGE_TRIGGERED_DFF}:
                continue
            macro_device = str(item.evidence.get("macro_device", "")) if isinstance(item.evidence, dict) else ""
            xor_toggle = xor_toggle_by_ff.get(macro_device)
            if xor_toggle:
                upgraded.append(
                    RecognizedStructure(
                        kind=RecognizedCellKind.T_FLIP_FLOP,
                        confidence=min(0.95, item.confidence + 0.1),
                        storage_nodes=list(item.storage_nodes),
                        outputs=list(item.outputs),
                        inputs=list(item.inputs),
                        clocks=list(item.clocks),
                        controls=list(item.controls),
                        features=self._unique_features([*item.features, CompositeFeature.XOR_TOGGLE_FEEDBACK]),
                        evidence={**item.evidence, "decoded_from": item.kind.value, "xor_toggle_feedback": xor_toggle},
                        components=self._unique([*item.components, str(xor_toggle.get("xor_device"))]),
                    )
                )
            jk_evidence = self._detect_jk_feedback_equation(item)
            if jk_evidence:
                upgraded.append(
                    RecognizedStructure(
                        kind=RecognizedCellKind.JK_FLIP_FLOP,
                        confidence=min(0.9, item.confidence + 0.08),
                        storage_nodes=list(item.storage_nodes),
                        outputs=list(item.outputs),
                        inputs=jk_evidence.get("inputs", item.inputs),
                        clocks=list(item.clocks),
                        controls=list(item.controls),
                        features=self._unique_features(item.features),
                        evidence={**item.evidence, "decoded_from": item.kind.value, "jk_decode": jk_evidence},
                        components=list(item.components),
                    )
                )
        for chain in patterns.toggle_chains:
            upgraded.append(
                RecognizedStructure(
                    kind=RecognizedCellKind.COUNTER_CELL,
                    confidence=0.78,
                    storage_nodes=[],
                    outputs=[],
                    inputs=[],
                    clocks=[],
                    controls=[],
                    features=[CompositeFeature.TOGGLE_CHAIN],
                    evidence={"toggle_chain": chain},
                    components=list(chain.get("stages", [])),
                )
            )
        return upgraded

if TYPE_CHECKING:
    from krona.analysis.structural.patterns_parts.storage_phase_profile import PatternDatabase
