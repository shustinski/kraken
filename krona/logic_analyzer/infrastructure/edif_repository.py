from __future__ import annotations

from pathlib import Path

from logic_analyzer.domain.netlist import TopLevelNetlist
from logic_analyzer.infrastructure.edif_parser import EdifTextParser


class EdifRepository:
    def __init__(self):
        self._parser_cache: dict[Path, EdifTextParser] = {}

    def _parser_for(self, path: Path) -> EdifTextParser:
        key = path.resolve()
        parser = self._parser_cache.get(key)
        if parser is None:
            parser = EdifTextParser(key)
            self._parser_cache[key] = parser
        return parser

    def read_top_level_netlist(self, path: Path) -> TopLevelNetlist:
        return self._parser_for(path).parse_top_level_netlist()

    def read_view_index(self, path: Path) -> dict[tuple[str, str, str], str]:
        return self._parser_for(path).build_view_index()

    def read_top_page_block(self, path: Path) -> str:
        return self._parser_for(path).get_top_page_block()

    def read_file_text(self, path: Path) -> str:
        return self._parser_for(path).file_text
