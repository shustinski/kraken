from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from krona.domain.netlist import TopLevelNetlist


@dataclass(frozen=True)
class DiagnosticItem:
    severity: str
    message: str


class NetlistRepository(Protocol):
    def read_top_level_netlist(self, path: Path) -> TopLevelNetlist:
        ...

    def read_view_index(self, path: Path) -> dict[tuple[str, str, str], str]:
        ...

    def read_top_page_block(self, path: Path) -> str:
        ...

    def read_file_text(self, path: Path) -> str:
        ...

    def read_diagnostics(self, path: Path) -> list[DiagnosticItem]:
        ...


@dataclass(frozen=True)
class SceneData:
    netlist: TopLevelNetlist
    view_index: dict[tuple[str, str, str], str]
    top_page_block: str
    source_text: str
    diagnostics: list[DiagnosticItem]
