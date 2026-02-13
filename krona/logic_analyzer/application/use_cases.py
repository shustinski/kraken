from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from logic_analyzer.application.ports import NetlistRepository, SceneData


class LoadTopLevelNetlist:
    def __init__(self, repository: NetlistRepository):
        self._repository = repository

    def execute(self, path: str | Path) -> dict:
        netlist = self._repository.read_top_level_netlist(Path(path))
        return asdict(netlist)


class LoadSceneData:
    def __init__(self, repository: NetlistRepository):
        self._repository = repository

    def execute(self, path: str | Path) -> SceneData:
        source_path = Path(path)
        return SceneData(
            netlist=self._repository.read_top_level_netlist(source_path),
            view_index=self._repository.read_view_index(source_path),
            top_page_block=self._repository.read_top_page_block(source_path),
            source_text=self._repository.read_file_text(source_path),
            diagnostics=self._repository.read_diagnostics(source_path),
        )
