from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from logic_analyzer.domain.netlist import Connection, Instance, Net, Point, TopLevelNetlist, Wire


@dataclass(frozen=True)
class _TopContext:
    design_name: str
    library_name: str
    cell_name: str
    view_name: str
    top_page_block: str


class EdifTextParser:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.file_text = self.path.read_text(encoding="utf-8", errors="ignore")

    @staticmethod
    def _extract_balanced(text: str, start_index: int) -> tuple[str, int]:
        depth = 0
        for idx in range(start_index, len(text)):
            char = text[idx]
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    return text[start_index:idx + 1], idx + 1
        raise ValueError("Unbalanced EDIF block")

    @staticmethod
    def find_classes(where: str, what: str) -> list[str]:
        classes = []
        cursor = 0
        while True:
            try:
                start = where.index(what, cursor)
            except ValueError:
                break
            block, end = EdifTextParser._extract_balanced(where, start)
            classes.append(block)
            cursor = end
        return classes

    @staticmethod
    def find_direct_classes(where: str, what: str) -> list[str]:
        classes = []
        stripped = where.lstrip()
        if not stripped.startswith("("):
            return classes
        base_offset = len(where) - len(stripped)
        depth = 0
        idx = 0
        while idx < len(stripped):
            char = stripped[idx]
            if char == "(":
                depth += 1
                absolute_idx = base_offset + idx
                if depth == 2 and where.startswith(what, absolute_idx):
                    block, end = EdifTextParser._extract_balanced(where, absolute_idx)
                    classes.append(block)
                    idx = end - base_offset
                    depth = 1
                    continue
            elif char == ")":
                depth -= 1
            idx += 1
        return classes

    @staticmethod
    def _parse_rename_expr(expr: str) -> str | None:
        match = re.match(r'^\(rename\s+[^\s)]+\s+"([^"]+)"\)$', expr.strip(), re.DOTALL)
        return match.group(1) if match else None

    @classmethod
    def parse_header_name(cls, block: str, keyword: str) -> str:
        prefix = f"({keyword}"
        start = block.find(prefix)
        if start == -1:
            raise ValueError(f"Block does not start with {prefix}")
        pos = start + len(prefix)
        while pos < len(block) and block[pos].isspace():
            pos += 1
        if pos < len(block) and block[pos] == "(":
            expr, _ = cls._extract_balanced(block, pos)
            renamed = cls._parse_rename_expr(expr)
            return renamed if renamed is not None else expr.strip()
        end = pos
        while end < len(block) and not block[end].isspace() and block[end] != ")":
            end += 1
        return block[pos:end]

    @classmethod
    def first_direct_or_none(cls, block: str, what: str) -> str | None:
        blocks = cls.find_direct_classes(block, what)
        return blocks[0] if blocks else None

    @classmethod
    def first_any_or_none(cls, block: str, what: str) -> str | None:
        blocks = cls.find_classes(block, what)
        return blocks[0] if blocks else None

    @classmethod
    def parse_points(cls, block: str) -> list[Point]:
        points: list[Point] = []
        for x_str, y_str in re.findall(r"\(pt\s+(-?\d+)\s+(-?\d+)\s*\)", block):
            points.append(Point(x=int(x_str), y=int(y_str)))
        return points

    def _parse_designator(self, instance_block: str) -> str | None:
        designator_block = self.first_direct_or_none(instance_block, "(designator")
        if not designator_block:
            return None
        match = re.search(r'"([^"]+)"', designator_block)
        return match.group(1) if match else None

    def _parse_origin_from_transform(self, instance_block: str) -> Point | None:
        transform_block = self.first_direct_or_none(instance_block, "(transform")
        if not transform_block:
            return None
        origin_block = self.first_any_or_none(transform_block, "(origin")
        if not origin_block:
            return None
        points = self.parse_points(origin_block)
        return points[0] if points else None

    def _parse_orientation_from_transform(self, instance_block: str) -> str | None:
        transform_block = self.first_direct_or_none(instance_block, "(transform")
        if not transform_block:
            return None
        match = re.search(r"\(orientation\s+([A-Z0-9]+)\s*\)", transform_block)
        return match.group(1) if match else None

    def _build_top_context(self) -> _TopContext:
        design_block = self.first_any_or_none(self.file_text, "(design ")
        if not design_block:
            raise ValueError("No design block found in EDIF")
        design_name = self.parse_header_name(design_block, "design")
        cell_ref = self.first_any_or_none(design_block, "(cellRef ")
        if not cell_ref:
            raise ValueError("Top design does not contain cellRef")
        top_cell_name = self.parse_header_name(cell_ref, "cellRef")
        library_ref = self.first_any_or_none(cell_ref, "(libraryRef ")
        if not library_ref:
            raise ValueError("Top design cellRef does not contain libraryRef")
        top_library_name = self.parse_header_name(library_ref, "libraryRef")

        libraries = self.find_direct_classes(self.file_text, "(library ")
        top_library_block = next(
            (lib for lib in libraries if self.parse_header_name(lib, "library") == top_library_name),
            None,
        )
        if not top_library_block:
            raise ValueError(f"Library '{top_library_name}' not found")

        cell_blocks = self.find_direct_classes(top_library_block, "(cell ")
        top_cell_block = next(
            (cell for cell in cell_blocks if self.parse_header_name(cell, "cell") == top_cell_name),
            None,
        )
        if not top_cell_block:
            raise ValueError(f"Cell '{top_cell_name}' not found in library '{top_library_name}'")

        view_blocks = self.find_direct_classes(top_cell_block, "(view ")
        if not view_blocks:
            raise ValueError(f"Cell '{top_cell_name}' does not have a view")
        top_view_block = next((view for view in view_blocks if "(viewType SCHEMATIC)" in view), view_blocks[0])
        top_view_name = self.parse_header_name(top_view_block, "view")
        contents = self.first_direct_or_none(top_view_block, "(contents")
        if not contents:
            raise ValueError(f"View '{top_view_name}' does not contain contents")
        page_blocks = self.find_direct_classes(contents, "(page ")
        top_page = page_blocks[0] if page_blocks else contents
        return _TopContext(
            design_name=design_name,
            library_name=top_library_name,
            cell_name=top_cell_name,
            view_name=top_view_name,
            top_page_block=top_page,
        )

    def build_view_index(self) -> dict[tuple[str, str, str], str]:
        index: dict[tuple[str, str, str], str] = {}
        for lib_block in self.find_direct_classes(self.file_text, "(library "):
            lib_name = self.parse_header_name(lib_block, "library")
            for cell_block in self.find_direct_classes(lib_block, "(cell "):
                cell_name = self.parse_header_name(cell_block, "cell")
                for view_block in self.find_direct_classes(cell_block, "(view "):
                    view_name = self.parse_header_name(view_block, "view")
                    index[(lib_name, cell_name, view_name)] = view_block
        return index

    def get_top_page_block(self) -> str:
        return self._build_top_context().top_page_block

    def parse_top_level_netlist(self) -> TopLevelNetlist:
        ctx = self._build_top_context()
        instances: list[Instance] = []
        for instance_block in self.find_direct_classes(ctx.top_page_block, "(instance "):
            name = self.parse_header_name(instance_block, "instance")
            view_ref = self.first_direct_or_none(instance_block, "(viewRef ")
            view_name = self.parse_header_name(view_ref, "viewRef") if view_ref else None
            cell_ref = self.first_any_or_none(instance_block, "(cellRef ")
            cell_name = self.parse_header_name(cell_ref, "cellRef") if cell_ref else None
            lib_ref = self.first_any_or_none(instance_block, "(libraryRef ")
            lib_name = self.parse_header_name(lib_ref, "libraryRef") if lib_ref else None
            origin = self._parse_origin_from_transform(instance_block)
            orientation = self._parse_orientation_from_transform(instance_block)
            instances.append(
                Instance(
                    name=name,
                    view=view_name,
                    cell=cell_name,
                    library=lib_name,
                    designator=self._parse_designator(instance_block),
                    x=origin.x if origin else None,
                    y=origin.y if origin else None,
                    orientation=orientation,
                )
            )

        nets: list[Net] = []
        for net_block in self.find_direct_classes(ctx.top_page_block, "(net "):
            net_name = self.parse_header_name(net_block, "net")
            connections: list[Connection] = []
            wires: list[Wire] = []
            for joined in self.find_direct_classes(net_block, "(joined"):
                for port_ref in self.find_direct_classes(joined, "(portRef "):
                    port_name = self.parse_header_name(port_ref, "portRef")
                    instance_ref = self.first_any_or_none(port_ref, "(instanceRef ")
                    instance_name = self.parse_header_name(instance_ref, "instanceRef") if instance_ref else None
                    connections.append(Connection(port=port_name, instance=instance_name))
            for figure_block in self.find_direct_classes(net_block, "(figure "):
                for path_block in self.find_classes(figure_block, "(path"):
                    for point_list_block in self.find_classes(path_block, "(pointList"):
                        points = self.parse_points(point_list_block)
                        if points:
                            wires.append(Wire(points=points))
            nets.append(Net(name=net_name, connections=connections, wires=wires))

        return TopLevelNetlist(
            design=ctx.design_name,
            library=ctx.library_name,
            cell=ctx.cell_name,
            view=ctx.view_name,
            instances=instances,
            nets=nets,
        )
