"""patterns: classify."""
from __future__ import annotations
from typing import TYPE_CHECKING
from krona.analysis.structural.graphs import GraphBundle
from krona.analysis.structural.model import (
    CircuitModel,
    RecognizedStructure,
)
from krona.analysis.structural.phases import PhaseAnalyzer


class ClassifyMixin:
    def __init__(self, circuit: CircuitModel, graphs: GraphBundle, phase_analyzer: PhaseAnalyzer):
        self._circuit = circuit
        self._graphs = graphs
        self._phase = phase_analyzer

    def classify(self, patterns: PatternDatabase) -> list[RecognizedStructure]:
        recognized: list[RecognizedStructure] = []
        recognized.extend(self._classify_sequential_macros(patterns))
        recognized.extend(self._classify_bistable_cores(patterns))
        recognized = self._upgrade_jk_t_counter(recognized, patterns)
        return self._dedupe_recognized(recognized)

if TYPE_CHECKING:
    from krona.analysis.structural.patterns_parts.storage_phase_profile import PatternDatabase
