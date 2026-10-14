"""scene_renderer: pmos instance."""
from __future__ import annotations
from krona.infrastructure.edif_parser import EdifTextParser


class PmosInstanceMixin:
    @staticmethod
    def _is_pmos_instance(instance_block: str) -> bool:
        cell_ref = EdifTextParser.first_any_or_none(instance_block, "(cellRef ")
        view_ref = EdifTextParser.first_direct_or_none(instance_block, "(viewRef ")
        cell_name = EdifTextParser.parse_header_name(cell_ref, "cellRef") if cell_ref else ""
        view_name = EdifTextParser.parse_header_name(view_ref, "viewRef") if view_ref else ""
        if "PMOS" in cell_name.upper() or "PMOS" in view_name.upper():
            return True
        return "PMOS" in instance_block.upper()

    @staticmethod
    def _is_mos_instance(instance_block: str) -> bool:
        upper = instance_block.upper()
        return "NMOS" in upper or "PMOS" in upper
