from __future__ import annotations

from pathlib import Path

from logic_analyzer.domain.netlist import Connection, Instance, Net, Point, TopLevelNetlist, Wire
from logic_analyzer.infrastructure.edif_parser import EdifTextParser


def remove_edif_header(edif_list: str) -> str:
    end_header_index = edif_list.index("\n")
    return edif_list[end_header_index + 1 :].lstrip()


class EdifParser:
    """
    Backward-compatible facade around the infrastructure parser.
    """

    def __init__(self, path: str | Path):
        self._parser = EdifTextParser(path)
        self.file_text = self._parser.file_text

    @staticmethod
    def find_classes(where: str, what: str) -> list[str]:
        return EdifTextParser.find_classes(where, what)

    @staticmethod
    def find_direct_classes(where: str, what: str) -> list[str]:
        return EdifTextParser.find_direct_classes(where, what)

    @staticmethod
    def _parse_header_name(block: str, keyword: str) -> str:
        return EdifTextParser.parse_header_name(block, keyword)

    @staticmethod
    def _first_any_or_none(block: str, what: str) -> str | None:
        return EdifTextParser.first_any_or_none(block, what)

    @staticmethod
    def _first_direct_or_none(block: str, what: str) -> str | None:
        return EdifTextParser.first_direct_or_none(block, what)

    @staticmethod
    def _parse_points(block: str) -> list[Point]:
        return EdifTextParser.parse_points(block)

    def build_view_index(self) -> dict[tuple[str, str, str], str]:
        return self._parser.build_view_index()

    def get_top_page_block(self) -> str:
        return self._parser.get_top_page_block()

    def extract_top_level_netlist(self) -> TopLevelNetlist:
        return self._parser.parse_top_level_netlist()


def find_net_name(text: str) -> str:
    return EdifTextParser.parse_header_name(text, "net")
