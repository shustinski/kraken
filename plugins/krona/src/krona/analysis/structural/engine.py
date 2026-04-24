from __future__ import annotations

from dataclasses import dataclass, field

from .graphs import GraphBuilder
from .model import CircuitModel, StructuralAnalysisReport
from .patterns import PatternMatcher, SemanticClassifier
from .phases import PhaseAnalyzer
from .reporting import ReportingLayer


@dataclass
class StructuralSequentialAnalyzer:
    """
    Multi-level structural sequential analyzer.

    Pipeline:
    1. Build switch/gate conditional graphs
    2. Run SCC (Tarjan) + pattern extraction
    3. Enumerate clock/reset phases for transparency/hold checks
    4. Classify semantic storage cells
    5. Produce structured JSON-ready report
    """

    graph_builder: GraphBuilder = field(default_factory=GraphBuilder)
    reporting: ReportingLayer = field(default_factory=ReportingLayer)

    def analyze(self, circuit: CircuitModel) -> StructuralAnalysisReport:
        graphs = self.graph_builder.build(circuit)
        phase = PhaseAnalyzer(circuit, graphs)
        patterns = PatternMatcher(circuit, graphs, phase).run()
        recognized = SemanticClassifier(circuit, graphs, phase).classify(patterns)
        return self.reporting.build_report(
            design_name=circuit.design_name,
            source_format=circuit.source_format,
            graphs=graphs,
            patterns=patterns,
            recognized=recognized,
            metadata=circuit.metadata,
        )
