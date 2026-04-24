from __future__ import annotations

from pathlib import Path

from krona.analysis.structural import EdifStructuralParser, StructuralSequentialAnalyzer
from krona.application.ports import NetlistRepository


class AnalyzeSequentialStructure:
    """
    High-level use case: parse EDIF into device abstraction and run multi-level
    structural sequential analysis with JSON-friendly reporting.
    """

    def __init__(self, repository: NetlistRepository):
        self._parser = EdifStructuralParser(repository)
        self._analyzer = StructuralSequentialAnalyzer()

    def execute(self, path: str | Path) -> dict:
        circuit = self._parser.parse(Path(path))
        report = self._analyzer.analyze(circuit)
        return report.as_dict()
