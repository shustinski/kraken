from __future__ import annotations

from pathlib import Path
from hashlib import sha256

from krona.application.ports import DiagnosticItem
from krona.domain.netlist import TopLevelNetlist
from krona.infrastructure.edif_parser import EdifTextParser


class EdifRepository:
    def __init__(self):
        self._parser_cache: dict[Path, tuple[bytes, EdifTextParser]] = {}

    def _parser_for(self, path: Path) -> EdifTextParser:
        key = path.resolve()
        # Size and mtime alone miss rewrites by editors and restored revisions.
        digest = sha256(key.read_bytes()).digest()
        cached = self._parser_cache.get(key)
        if cached is None or cached[0] != digest:
            parser = EdifTextParser(key)
            self._parser_cache[key] = (digest, parser)
            return parser
        return cached[1]

    def read_top_level_netlist(self, path: Path) -> TopLevelNetlist:
        return self._parser_for(path).parse_top_level_netlist()

    def read_view_index(self, path: Path) -> dict[tuple[str, str, str], str]:
        return self._parser_for(path).build_view_index()

    def read_top_page_block(self, path: Path) -> str:
        return self._parser_for(path).get_top_page_block()

    def read_file_text(self, path: Path) -> str:
        return self._parser_for(path).file_text

    def read_diagnostics(self, path: Path) -> list[DiagnosticItem]:
        return self._parser_for(path).collect_diagnostics()
