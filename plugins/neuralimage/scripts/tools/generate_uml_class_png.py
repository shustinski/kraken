from __future__ import annotations

import ast
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


SKIP_DIRS = {
    ".venv",
    ".git",
    "dist",
    "dist_tmp",
    "build",
    "build_tmp",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".idea",
    ".vscode",
    ".test_runtime",
    ".tmp",
    ".tracecov",
    ".tracecov_new",
    ".codex_tmp_scratch",
    ".codex_pytest_tmp",
    ".pycache_tmp",
    ".pytest_tmp",
    ".uv-cache",
    "_internal",
    "tests",
    "tmpt3zb14ax",
    "tmp_check",
    "tmp_dynamic_len_check",
    "tmp_preview_pcb_check",
}

SKIP_ROOTS = {
    "test",
}

PACKAGE_LAYERS: list[list[str]] = [
    ["main", "manage", "main_code_version"],
    ["controller", "bootstrap", "webui_project", "neuralimage.webui"],
    ["presenter", "view", "UI", "application"],
    ["model", "augmentations", "infrastructure", "Validation_gradient_widget_lite"],
    ["lib"],
    ["hooks", "tools"],
]

PACKAGE_COLORS = {
    "entry": (233, 244, 255),
    "orchestration": (241, 236, 255),
    "presentation": (234, 248, 239),
    "core": (255, 244, 229),
    "support": (243, 243, 243),
    "tooling": (249, 240, 229),
    "fallback": (250, 250, 250),
}

GROUP_BY_ROOT = {
    "main": "entry",
    "manage": "entry",
    "main_code_version": "entry",
    "controller": "orchestration",
    "bootstrap": "orchestration",
    "webui_project": "orchestration",
    "neuralimage.webui": "orchestration",
    "presenter": "presentation",
    "view": "presentation",
    "UI": "presentation",
    "application": "presentation",
    "model": "core",
    "augmentations": "core",
    "infrastructure": "core",
    "Validation_gradient_widget_lite": "core",
    "lib": "support",
    "hooks": "tooling",
    "tools": "tooling",
}


@dataclass
class ClassInfo:
    fqcn: str
    module: str
    root: str
    name: str
    bases_raw: list[str] = field(default_factory=list)
    attributes: list[str] = field(default_factory=list)
    methods: list[str] = field(default_factory=list)
    inherits: set[str] = field(default_factory=set)
    associations: set[str] = field(default_factory=set)
    dependencies: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class ClassBoxLayout:
    width: int
    height: int
    title_height: int
    attr_height: int
    method_height: int


@dataclass(frozen=True)
class ModuleLayout:
    width: int
    height: int
    header_height: int


@dataclass(frozen=True)
class PackageLayout:
    width: int
    height: int
    header_height: int
    columns: int
    rows: int


def should_skip(path: Path) -> bool:
    return any(part in SKIP_DIRS for part in path.parts)


def module_name(root: Path, path: Path) -> str:
    rel = path.relative_to(root)
    parts = list(rel.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def root_name(module: str) -> str:
    return module.split(".")[0] if module else ""


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/calibri.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top


def iter_type_names(node: ast.AST | None) -> list[str]:
    if node is None:
        return []
    names: list[str] = []
    stack: list[ast.AST] = [node]
    while stack:
        cur = stack.pop()
        if isinstance(cur, ast.Name):
            names.append(cur.id)
        elif isinstance(cur, ast.Attribute):
            parts: list[str] = []
            ref: ast.AST | None = cur
            while isinstance(ref, ast.Attribute):
                parts.append(ref.attr)
                ref = ref.value
            if isinstance(ref, ast.Name):
                parts.append(ref.id)
                names.append(".".join(reversed(parts)))
            else:
                stack.append(cur.value)
        elif isinstance(cur, ast.Constant):
            if isinstance(cur.value, str):
                token = cur.value.strip().strip("\"'")
                if token:
                    names.append(token)
        else:
            for child in ast.iter_child_nodes(cur):
                stack.append(child)
    return names


def format_annotation(node: ast.AST | None) -> str:
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except Exception:
        names = iter_type_names(node)
        return " | ".join(names[:3])


def format_value(node: ast.AST | None) -> str:
    if node is None:
        return ""
    try:
        raw = ast.unparse(node)
    except Exception:
        return ""
    raw = " ".join(raw.split())
    return raw[:40] + "..." if len(raw) > 43 else raw


def visibility_prefix(name: str) -> str:
    if name.startswith("__") and not name.endswith("__"):
        return "-"
    if name.startswith("_") and not name.startswith("__"):
        return "#"
    return "+"


def format_attribute_line(name: str, annotation: str = "", value: str = "") -> str:
    line = f"{visibility_prefix(name)} {name}"
    if annotation:
        line += f": {annotation}"
    if value:
        line += f" = {value}"
    return line


def format_arg(arg: ast.arg) -> str:
    if arg.annotation is not None:
        return f"{arg.arg}: {format_annotation(arg.annotation)}"
    return arg.arg


def format_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    params: list[str] = []
    all_args = node.args.posonlyargs + node.args.args
    defaults = [None] * (len(all_args) - len(node.args.defaults)) + list(node.args.defaults)
    for arg, default in zip(all_args, defaults):
        if arg.arg == "self":
            continue
        param = format_arg(arg)
        if default is not None:
            default_str = format_value(default)
            if default_str:
                param += f" = {default_str}"
        params.append(param)

    if node.args.vararg is not None:
        params.append(f"*{format_arg(node.args.vararg)}")
    elif node.args.kwonlyargs:
        params.append("*")

    for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults):
        param = format_arg(arg)
        if default is not None:
            default_str = format_value(default)
            if default_str:
                param += f" = {default_str}"
        params.append(param)

    if node.args.kwarg is not None:
        params.append(f"**{format_arg(node.args.kwarg)}")

    prefix = visibility_prefix(node.name)
    signature = f"{prefix} {node.name}({', '.join(params)})"
    if node.returns is not None:
        signature += f" -> {format_annotation(node.returns)}"
    return signature


def resolve_ref(
    raw: str,
    current_module: str,
    classes_by_name: dict[str, set[str]],
    class_names: set[str],
) -> str | None:
    token = raw.strip()
    if not token:
        return None
    token = token.replace("|", " ")
    for chunk in token.replace(",", " ").split():
        candidate = chunk.strip()
        if not candidate:
            continue
        if candidate in class_names:
            return candidate
        if "." in candidate:
            parts = candidate.split(".")
            for idx in range(len(parts) - 1):
                mod = ".".join(parts[idx:-1])
                cls = parts[-1]
                fqcn = f"{mod}.{cls}" if mod else cls
                if fqcn in class_names:
                    return fqcn
            cls = parts[-1]
            if cls in classes_by_name:
                possible = classes_by_name[cls]
                if len(possible) == 1:
                    return next(iter(possible))
        else:
            if candidate in classes_by_name:
                possible = classes_by_name[candidate]
                if len(possible) == 1:
                    return next(iter(possible))
                top = current_module.split(".")[0] if current_module else ""
                same_top = [fq for fq in possible if fq.split(".")[0] == top]
                if len(same_top) == 1:
                    return same_top[0]
    return None


def collect_attributes(class_node: ast.ClassDef) -> list[str]:
    attrs: list[str] = []
    seen: set[str] = set()

    def add(line: str) -> None:
        if line not in seen:
            seen.add(line)
            attrs.append(line)

    for stmt in class_node.body:
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            add(
                format_attribute_line(
                    stmt.target.id,
                    format_annotation(stmt.annotation),
                    format_value(stmt.value),
                )
            )
        elif isinstance(stmt, ast.Assign):
            value_str = format_value(stmt.value)
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    add(format_attribute_line(target.id, value=value_str))
        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for inner in ast.walk(stmt):
                if isinstance(inner, ast.Assign):
                    value_str = format_value(inner.value)
                    for target in inner.targets:
                        if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
                            add(format_attribute_line(target.attr, value=value_str))
                elif isinstance(inner, ast.AnnAssign):
                    target = inner.target
                    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
                        add(
                            format_attribute_line(
                                target.attr,
                                format_annotation(inner.annotation),
                                format_value(inner.value),
                            )
                        )
    return attrs


def collect_methods(class_node: ast.ClassDef) -> list[str]:
    methods: list[str] = []
    for stmt in class_node.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            methods.append(format_signature(stmt))
    return methods


def extract_relations(class_node: ast.ClassDef) -> tuple[list[str], list[str], list[str]]:
    inherits_raw: list[str] = []
    assoc_raw: list[str] = []
    dep_raw: list[str] = []

    for base in class_node.bases:
        inherits_raw.extend(iter_type_names(base))

    for stmt in class_node.body:
        if isinstance(stmt, ast.AnnAssign):
            assoc_raw.extend(iter_type_names(stmt.annotation))
        elif isinstance(stmt, ast.Assign):
            if isinstance(stmt.value, ast.Call):
                assoc_raw.extend(iter_type_names(stmt.value.func))
        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for arg in list(stmt.args.args) + list(stmt.args.kwonlyargs):
                if arg.arg == "self":
                    continue
                dep_raw.extend(iter_type_names(arg.annotation))
            dep_raw.extend(iter_type_names(stmt.returns))
            for inner in ast.walk(stmt):
                if isinstance(inner, ast.Call):
                    dep_raw.extend(iter_type_names(inner.func))
    return inherits_raw, assoc_raw, dep_raw


def class_box_layout(
    draw: ImageDraw.ImageDraw,
    info: ClassInfo,
    *,
    title_font: ImageFont.ImageFont,
    body_font: ImageFont.ImageFont,
    meta_font: ImageFont.ImageFont,
) -> ClassBoxLayout:
    lines = [info.name, f"<<{info.module}>>", *info.attributes, *info.methods]
    widths = []
    for index, line in enumerate(lines):
        font = title_font if index == 0 else meta_font if index == 1 else body_font
        widths.append(text_size(draw, line, font)[0])
    width = max(260, max(widths, default=0) + 28)

    title_height = text_size(draw, info.name, title_font)[1] + text_size(draw, f"<<{info.module}>>", meta_font)[1] + 20
    attr_line_height = max(text_size(draw, "Ag", body_font)[1], 16)
    method_line_height = max(text_size(draw, "Ag", body_font)[1], 16)
    attr_height = max(24, len(info.attributes) * (attr_line_height + 2) + 12)
    method_height = max(24, len(info.methods) * (method_line_height + 2) + 12)
    height = title_height + attr_height + method_height + 4
    return ClassBoxLayout(width=width, height=height, title_height=title_height, attr_height=attr_height, method_height=method_height)


def module_layout(
    draw: ImageDraw.ImageDraw,
    module: str,
    classes: list[ClassInfo],
    box_layouts: dict[str, ClassBoxLayout],
    *,
    header_font: ImageFont.ImageFont,
) -> ModuleLayout:
    title_w, title_h = text_size(draw, module, header_font)
    content_width = max([title_w] + [box_layouts[cls.fqcn].width for cls in classes])
    content_height = sum(box_layouts[cls.fqcn].height for cls in classes)
    content_height += max(0, len(classes) - 1) * 16
    width = content_width + 24
    height = content_height + title_h + 34
    return ModuleLayout(width=width, height=height, header_height=title_h + 18)


def choose_package_columns(module_count: int) -> int:
    if module_count <= 2:
        return 1
    if module_count <= 6:
        return 2
    if module_count <= 12:
        return 3
    return 4


def package_layout(
    modules: list[str],
    module_layouts: dict[str, ModuleLayout],
) -> PackageLayout:
    cols = choose_package_columns(len(modules))
    rows = max(1, math.ceil(len(modules) / cols))
    gap_x = 26
    gap_y = 22
    inner_pad = 22
    header_height = 46

    row_widths: list[int] = []
    row_heights: list[int] = []
    for row_index in range(rows):
        row_modules = modules[row_index * cols:(row_index + 1) * cols]
        row_width = sum(module_layouts[module].width for module in row_modules) + max(0, len(row_modules) - 1) * gap_x
        row_height = max(module_layouts[module].height for module in row_modules)
        row_widths.append(row_width)
        row_heights.append(row_height)

    width = max(row_widths, default=0) + inner_pad * 2
    height = sum(row_heights) + max(0, rows - 1) * gap_y + inner_pad * 2 + header_height
    return PackageLayout(width=width, height=height, header_height=header_height, columns=cols, rows=rows)


def ordered_roots(roots: set[str]) -> list[str]:
    known = [name for layer in PACKAGE_LAYERS for name in layer if name in roots]
    known_set = set(known)
    return known + [name for name in sorted(roots) if name not in known_set]


def package_group(root: str) -> str:
    return GROUP_BY_ROOT.get(root, "fallback")


def build_layer_rows(roots: list[str]) -> list[list[str]]:
    rows: list[list[str]] = []
    assigned: set[str] = set()
    for layer in PACKAGE_LAYERS:
        members = [name for name in layer if name in roots]
        if members:
            rows.append(members)
            assigned.update(members)
    extras = [name for name in roots if name not in assigned]
    if extras:
        rows.append(extras)
    return rows


def center(rect: tuple[int, int, int, int]) -> tuple[int, int]:
    x1, y1, x2, y2 = rect
    return (x1 + x2) // 2, (y1 + y2) // 2


def edge_anchor(
    source_rect: tuple[int, int, int, int],
    target_rect: tuple[int, int, int, int],
) -> tuple[tuple[int, int], tuple[int, int]]:
    sx1, sy1, sx2, sy2 = source_rect
    tx1, ty1, tx2, ty2 = target_rect
    scx, scy = center(source_rect)
    tcx, tcy = center(target_rect)
    dx = tcx - scx
    dy = tcy - scy
    if abs(dx) >= abs(dy):
        start = (sx2, scy) if dx >= 0 else (sx1, scy)
        end = (tx1, tcy) if dx >= 0 else (tx2, tcy)
    else:
        start = (scx, sy2) if dy >= 0 else (scx, sy1)
        end = (tcx, ty1) if dy >= 0 else (tcx, ty2)
    return start, end


def draw_polyline_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    color: tuple[int, int, int, int],
    width: int,
    dashed: bool = False,
) -> None:
    sx, sy = start
    ex, ey = end
    if abs(ex - sx) >= abs(ey - sy):
        mid_x = (sx + ex) // 2
        points = [(sx, sy), (mid_x, sy), (mid_x, ey), (ex, ey)]
    else:
        mid_y = (sy + ey) // 2
        points = [(sx, sy), (sx, mid_y), (ex, mid_y), (ex, ey)]

    if dashed:
        for p1, p2 in zip(points, points[1:]):
            draw_dashed_segment(draw, p1, p2, color=color, width=width)
    else:
        draw.line(points, fill=color, width=width, joint="curve")

    last_x, last_y = points[-2]
    dx = ex - last_x
    dy = ey - last_y
    dist = max(1.0, (dx * dx + dy * dy) ** 0.5)
    ux = dx / dist
    uy = dy / dist
    base_x = ex - ux * 12
    base_y = ey - uy * 12
    perp_x = -uy
    perp_y = ux
    left = (int(base_x + perp_x * 5), int(base_y + perp_y * 5))
    right = (int(base_x - perp_x * 5), int(base_y - perp_y * 5))
    draw.polygon([(ex, ey), left, right], fill=color)


def draw_dashed_segment(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    color: tuple[int, int, int, int],
    width: int,
) -> None:
    sx, sy = start
    ex, ey = end
    dx = ex - sx
    dy = ey - sy
    dist = max(1.0, (dx * dx + dy * dy) ** 0.5)
    dash = 10.0
    gap = 7.0
    step = dash + gap
    offset = 0.0
    while offset < dist:
        start_ratio = offset / dist
        end_ratio = min(dist, offset + dash) / dist
        x1 = int(sx + dx * start_ratio)
        y1 = int(sy + dy * start_ratio)
        x2 = int(sx + dx * end_ratio)
        y2 = int(sy + dy * end_ratio)
        draw.line((x1, y1, x2, y2), fill=color, width=width)
        offset += step


def main() -> None:
    root = Path(".").resolve()
    output = root / "project_uml_class_diagram.png"

    py_files = [path for path in sorted(root.rglob("*.py")) if not should_skip(path)]

    class_nodes: dict[str, tuple[str, ast.ClassDef]] = {}
    classes_by_name: dict[str, set[str]] = defaultdict(set)
    module_to_classes: dict[str, list[str]] = defaultdict(list)
    package_to_modules: dict[str, set[str]] = defaultdict(set)

    for path in py_files:
        module = module_name(root, path)
        if not module:
            continue
        pkg = root_name(module)
        if pkg in SKIP_ROOTS:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except Exception:
            continue
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            fqcn = f"{module}.{node.name}"
            class_nodes[fqcn] = (module, node)
            classes_by_name[node.name].add(fqcn)
            module_to_classes[module].append(fqcn)
            package_to_modules[pkg].add(module)

    class_names = set(class_nodes.keys())
    infos: dict[str, ClassInfo] = {}

    for fqcn, (module, class_node) in class_nodes.items():
        pkg = root_name(module)
        inherits_raw, assoc_raw, dep_raw = extract_relations(class_node)
        attributes = collect_attributes(class_node)
        methods = collect_methods(class_node)
        base_names = [format_annotation(base) for base in class_node.bases]

        inherits: set[str] = set()
        associations: set[str] = set()
        dependencies: set[str] = set()

        for raw in inherits_raw:
            ref = resolve_ref(raw, module, classes_by_name, class_names)
            if ref and ref != fqcn:
                inherits.add(ref)
        for raw in assoc_raw:
            ref = resolve_ref(raw, module, classes_by_name, class_names)
            if ref and ref != fqcn and ref not in inherits:
                associations.add(ref)
        for raw in dep_raw:
            ref = resolve_ref(raw, module, classes_by_name, class_names)
            if ref and ref != fqcn and ref not in inherits and ref not in associations:
                dependencies.add(ref)

        infos[fqcn] = ClassInfo(
            fqcn=fqcn,
            module=module,
            root=pkg,
            name=class_node.name,
            bases_raw=base_names,
            attributes=attributes or ["(no attributes detected)"],
            methods=methods or ["(no methods detected)"],
            inherits=inherits,
            associations=associations,
            dependencies=dependencies,
        )

    if not infos:
        raise SystemExit("No classes found.")

    roots = ordered_roots({info.root for info in infos.values()})
    measure = Image.new("RGB", (10, 10), "white")
    measure_draw = ImageDraw.Draw(measure)
    title_font = load_font(26)
    package_font = load_font(20)
    module_font = load_font(16)
    class_font = load_font(17)
    body_font = load_font(13)
    meta_font = load_font(12)
    small_font = load_font(14)

    class_layouts = {
        fqcn: class_box_layout(measure_draw, info, title_font=class_font, body_font=body_font, meta_font=meta_font)
        for fqcn, info in infos.items()
    }

    module_layouts: dict[str, ModuleLayout] = {}
    package_modules_sorted: dict[str, list[str]] = {}
    for pkg in roots:
        modules = sorted(package_to_modules[pkg])
        package_modules_sorted[pkg] = modules
        for module in modules:
            fqcn_list = sorted(module_to_classes[module], key=lambda key: infos[key].name)
            module_layouts[module] = module_layout(measure_draw, module, [infos[fqcn] for fqcn in fqcn_list], class_layouts, header_font=module_font)

    package_layouts = {
        pkg: package_layout(package_modules_sorted[pkg], module_layouts)
        for pkg in roots
    }

    layer_rows = build_layer_rows(roots)
    outer_margin_x = 90
    outer_margin_y = 100
    row_gap = 80
    package_gap = 50

    row_widths: list[int] = []
    row_heights: list[int] = []
    for row in layer_rows:
        row_width = sum(package_layouts[pkg].width for pkg in row) + max(0, len(row) - 1) * package_gap
        row_height = max(package_layouts[pkg].height for pkg in row)
        row_widths.append(row_width)
        row_heights.append(row_height)

    image_width = max(4200, max(row_widths, default=0) + outer_margin_x * 2)
    image_height = max(3600, sum(row_heights) + max(0, len(row_heights) - 1) * row_gap + outer_margin_y * 2 + 140)

    image = Image.new("RGBA", (image_width, image_height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    edge_draw = ImageDraw.Draw(image, "RGBA")

    package_rects: dict[str, tuple[int, int, int, int]] = {}
    module_rects: dict[str, tuple[int, int, int, int]] = {}
    class_rects: dict[str, tuple[int, int, int, int]] = {}

    current_y = outer_margin_y + 90
    for row, row_height, row_width in zip(layer_rows, row_heights, row_widths):
        current_x = (image_width - row_width) // 2
        for pkg in row:
            layout = package_layouts[pkg]
            package_rect = (current_x, current_y, current_x + layout.width, current_y + layout.height)
            package_rects[pkg] = package_rect

            inner_pad = 22
            module_gap_x = 26
            module_gap_y = 22
            px1, py1, _, _ = package_rect
            start_x = px1 + inner_pad
            start_y = py1 + layout.header_height + inner_pad

            for index, module in enumerate(package_modules_sorted[pkg]):
                row_index = index // layout.columns
                col_index = index % layout.columns
                row_modules = package_modules_sorted[pkg][row_index * layout.columns:(row_index + 1) * layout.columns]
                offset_x = start_x + sum(module_layouts[name].width for name in row_modules[:col_index]) + module_gap_x * col_index
                offset_y = start_y
                for previous_row in range(row_index):
                    previous_row_modules = package_modules_sorted[pkg][previous_row * layout.columns:(previous_row + 1) * layout.columns]
                    previous_height = max(module_layouts[name].height for name in previous_row_modules)
                    offset_y += previous_height + module_gap_y

                module_layout_info = module_layouts[module]
                module_rect = (offset_x, offset_y, offset_x + module_layout_info.width, offset_y + module_layout_info.height)
                module_rects[module] = module_rect

                mx1, my1, _, _ = module_rect
                current_class_y = my1 + module_layout_info.header_height + 8
                for fqcn in sorted(module_to_classes[module], key=lambda key: infos[key].name):
                    class_layout_info = class_layouts[fqcn]
                    class_rect = (
                        mx1 + 12,
                        current_class_y,
                        mx1 + 12 + class_layout_info.width,
                        current_class_y + class_layout_info.height,
                    )
                    class_rects[fqcn] = class_rect
                    current_class_y += class_layout_info.height + 16

            current_x += layout.width + package_gap
        current_y += row_height + row_gap

    edge_specs: list[tuple[str, str, str]] = []
    for source, info in infos.items():
        for target in sorted(info.inherits):
            edge_specs.append((source, target, "inherit"))
        for target in sorted(info.associations):
            edge_specs.append((source, target, "assoc"))
        for target in sorted(info.dependencies):
            edge_specs.append((source, target, "dep"))

    edge_style = {
        "inherit": ((30, 30, 30, 170), 3, False),
        "assoc": ((32, 93, 173, 135), 2, False),
        "dep": ((130, 130, 130, 90), 1, True),
    }

    for source, target, rel in edge_specs:
        start, end = edge_anchor(class_rects[source], class_rects[target])
        color, width, dashed = edge_style[rel]
        draw_polyline_arrow(edge_draw, start, end, color=color, width=width, dashed=dashed)

    for pkg in roots:
        x1, y1, x2, y2 = package_rects[pkg]
        draw.rounded_rectangle((x1, y1, x2, y2), radius=18, fill=PACKAGE_COLORS[package_group(pkg)], outline=(70, 70, 70), width=2)
        draw.line((x1, y1 + package_layouts[pkg].header_height, x2, y1 + package_layouts[pkg].header_height), fill=(150, 150, 150), width=1)
        draw.text((x1 + 18, y1 + 12), pkg, fill=(20, 20, 20), font=package_font)
        summary = f"{len(package_to_modules[pkg])} modules / {sum(len(module_to_classes[m]) for m in package_modules_sorted[pkg])} classes"
        summary_w, _ = text_size(draw, summary, small_font)
        draw.text((x2 - summary_w - 18, y1 + 15), summary, fill=(78, 78, 78), font=small_font)

    for module, rect in module_rects.items():
        x1, y1, x2, y2 = rect
        draw.rounded_rectangle((x1, y1, x2, y2), radius=10, fill=(255, 255, 255), outline=(130, 130, 130), width=1)
        draw.line((x1, y1 + module_layouts[module].header_height, x2, y1 + module_layouts[module].header_height), fill=(180, 180, 180), width=1)
        label = module
        draw.text((x1 + 10, y1 + 8), label, fill=(38, 38, 38), font=module_font)

    for fqcn, rect in class_rects.items():
        info = infos[fqcn]
        layout = class_layouts[fqcn]
        x1, y1, x2, y2 = rect
        draw.rounded_rectangle((x1, y1, x2, y2), radius=8, fill=(252, 252, 252), outline=(55, 55, 55), width=1)
        draw.line((x1, y1 + layout.title_height, x2, y1 + layout.title_height), fill=(135, 135, 135), width=1)
        draw.line((x1, y1 + layout.title_height + layout.attr_height, x2, y1 + layout.title_height + layout.attr_height), fill=(165, 165, 165), width=1)

        current_y = y1 + 8
        title_w, title_h = text_size(draw, info.name, class_font)
        draw.text((x1 + (layout.width - title_w) // 2, current_y), info.name, fill=(18, 18, 18), font=class_font)
        current_y += title_h + 2

        if info.bases_raw:
            bases_line = " : " + ", ".join(info.bases_raw[:3])
            if len(info.bases_raw) > 3:
                bases_line += ", ..."
            draw.text((x1 + 10, current_y), bases_line, fill=(78, 78, 78), font=meta_font)
            current_y += text_size(draw, bases_line, meta_font)[1] + 2
        else:
            module_line = f"<<{info.module}>>"
            draw.text((x1 + 10, current_y), module_line, fill=(78, 78, 78), font=meta_font)
            current_y += text_size(draw, module_line, meta_font)[1] + 2

        attr_y = y1 + layout.title_height + 6
        for line in info.attributes:
            draw.text((x1 + 8, attr_y), line, fill=(25, 25, 25), font=body_font)
            attr_y += text_size(draw, line, body_font)[1] + 2

        method_y = y1 + layout.title_height + layout.attr_height + 6
        for line in info.methods:
            draw.text((x1 + 8, method_y), line, fill=(25, 25, 25), font=body_font)
            method_y += text_size(draw, line, body_font)[1] + 2

    draw.text((outer_margin_x, 26), "UML Class Diagram", fill=(0, 0, 0), font=title_font)
    subtitle = "All detected project classes with attributes and methods. Relations: inheritance, association, dependency."
    draw.text((outer_margin_x, 58), subtitle, fill=(72, 72, 72), font=body_font)

    legend_x = outer_margin_x
    legend_y = image_height - 92
    legend_items = [
        ("Inheritance", (30, 30, 30, 220), 3),
        ("Association", (32, 93, 173, 220), 2),
        ("Dependency", (130, 130, 130, 220), 1),
    ]
    for label, color, width in legend_items:
        if label == "Dependency":
            draw_dashed_segment(edge_draw, (legend_x, legend_y + 9), (legend_x + 34, legend_y + 9), color=color, width=width)
        else:
            edge_draw.line((legend_x, legend_y + 9, legend_x + 34, legend_y + 9), fill=color, width=width)
        draw.text((legend_x + 44, legend_y - 1), label, fill=(35, 35, 35), font=body_font)
        legend_x += 210

    rgb = image.convert("RGB")
    rgb.save(output, format="PNG")
    print(f"Saved: {output}")
    print(f"Classes: {len(infos)}")
    print(f"Modules: {len(module_to_classes)}")
    print(f"Relations: {len(edge_specs)}")


if __name__ == "__main__":
    main()
