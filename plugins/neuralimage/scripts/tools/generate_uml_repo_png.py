from __future__ import annotations

import ast
import math
from collections import defaultdict
from dataclasses import dataclass
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


@dataclass(frozen=True)
class ModuleInfo:
    module: str
    root: str
    path: Path
    label_lines: tuple[str, ...]


@dataclass(frozen=True)
class PackageLayout:
    width: int
    height: int
    header_height: int
    columns: int
    rows: int
    cell_width: int
    cell_height: int


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


def relative_label(module: str, root: str) -> str:
    if module == root:
        return "__init__"
    prefix = f"{root}."
    suffix = module[len(prefix):] if module.startswith(prefix) else module
    return suffix.replace(".", "/")


def wrap_label(label: str, max_len: int = 24) -> tuple[str, ...]:
    if len(label) <= max_len:
        return (label,)

    parts: list[str] = []
    for chunk in label.replace("/", "/ ").replace("_", "_ ").split():
        parts.append(chunk)

    lines: list[str] = []
    current = ""
    for part in parts:
        token = part.strip()
        if not token:
            continue
        candidate = token if not current else f"{current}{token}"
        if len(candidate) <= max_len:
            current = candidate
            continue
        if current:
            lines.append(current.rstrip("_/"))
        current = token
    if current:
        lines.append(current.rstrip("_/"))

    if not lines:
        return (label[:max_len], label[max_len:])
    return tuple(lines[:3])


def discover_modules(root: Path) -> dict[str, ModuleInfo]:
    modules: dict[str, ModuleInfo] = {}
    for path in sorted(root.rglob("*.py")):
        if should_skip(path):
            continue
        module = module_name(root, path)
        if not module:
            continue
        pkg = root_name(module)
        if pkg in SKIP_ROOTS:
            continue
        label = relative_label(module, pkg)
        modules[module] = ModuleInfo(
            module=module,
            root=pkg,
            path=path,
            label_lines=wrap_label(label),
        )
    return modules


def resolve_relative_module(module: str, imported_module: str | None, level: int) -> str:
    if level <= 0:
        return imported_module or ""
    base_parts = module.split(".")[:-1]
    drop = max(0, level - 1)
    prefix = base_parts[: len(base_parts) - drop] if drop <= len(base_parts) else []
    if imported_module:
        prefix.extend(imported_module.split("."))
    return ".".join(part for part in prefix if part)


def best_existing_module(target: str, module_map: dict[str, ModuleInfo]) -> str | None:
    parts = target.split(".")
    for size in range(len(parts), 0, -1):
        candidate = ".".join(parts[:size])
        if candidate in module_map:
            return candidate
    return None


def resolve_import_targets(
    *,
    current_module: str,
    node: ast.AST,
    module_map: dict[str, ModuleInfo],
) -> set[str]:
    targets: set[str] = set()

    if isinstance(node, ast.Import):
        for alias in node.names:
            target = best_existing_module(alias.name, module_map)
            if target:
                targets.add(target)
        return targets

    if not isinstance(node, ast.ImportFrom):
        return targets

    base = resolve_relative_module(current_module, node.module, node.level)
    if not base:
        return targets

    for alias in node.names:
        if alias.name == "*":
            target = best_existing_module(base, module_map)
            if target:
                targets.add(target)
            continue

        exact_submodule = f"{base}.{alias.name}"
        if exact_submodule in module_map:
            targets.add(exact_submodule)
            continue

        base_target = best_existing_module(base, module_map)
        if base_target:
            targets.add(base_target)

    return targets


def build_edges(root: Path, module_map: dict[str, ModuleInfo]) -> set[tuple[str, str]]:
    edges: set[tuple[str, str]] = set()

    for module, info in module_map.items():
        try:
            tree = ast.parse(info.path.read_text(encoding="utf-8"), filename=str(info.path))
        except Exception:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            for target in resolve_import_targets(current_module=module, node=node, module_map=module_map):
                if target != module:
                    edges.add((module, target))

    return edges


def ordered_roots(module_map: dict[str, ModuleInfo]) -> list[str]:
    found = sorted({info.root for info in module_map.values()})
    known = [name for layer in PACKAGE_LAYERS for name in layer if name in found]
    known_set = set(known)
    return known + [name for name in found if name not in known_set]


def choose_columns(count: int) -> int:
    if count <= 4:
        return 1
    if count <= 10:
        return 2
    if count <= 18:
        return 3
    return 4


def compute_package_layout(
    draw: ImageDraw.ImageDraw,
    modules: list[ModuleInfo],
    *,
    title_font: ImageFont.ImageFont,
    body_font: ImageFont.ImageFont,
) -> PackageLayout:
    columns = choose_columns(len(modules))
    rows = max(1, math.ceil(len(modules) / columns))

    max_line_width = 0
    max_line_height = 0
    max_lines = 1
    for module in modules:
        max_lines = max(max_lines, len(module.label_lines))
        for line in module.label_lines:
            line_w, line_h = text_size(draw, line, body_font)
            max_line_width = max(max_line_width, line_w)
            max_line_height = max(max_line_height, line_h)

    cell_width = max(150, max_line_width + 28)
    cell_height = max(42, max_lines * max_line_height + 20)

    title_height = text_size(draw, modules[0].root if modules else "", title_font)[1]
    header_height = max(56, title_height + 24)
    inner_pad = 18
    gap_x = 14
    gap_y = 10

    width = inner_pad * 2 + columns * cell_width + max(0, columns - 1) * gap_x
    height = header_height + inner_pad + rows * cell_height + max(0, rows - 1) * gap_y + inner_pad

    return PackageLayout(
        width=width,
        height=height,
        header_height=header_height,
        columns=columns,
        rows=rows,
        cell_width=cell_width,
        cell_height=cell_height,
    )


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


def edge_anchor(
    source_rect: tuple[int, int, int, int],
    target_rect: tuple[int, int, int, int],
) -> tuple[tuple[int, int], tuple[int, int]]:
    sx1, sy1, sx2, sy2 = source_rect
    tx1, ty1, tx2, ty2 = target_rect
    scx = (sx1 + sx2) // 2
    scy = (sy1 + sy2) // 2
    tcx = (tx1 + tx2) // 2
    tcy = (ty1 + ty2) // 2

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
) -> None:
    sx, sy = start
    ex, ey = end
    points: list[tuple[int, int]]

    if abs(ex - sx) >= abs(ey - sy):
        mid_x = (sx + ex) // 2
        points = [(sx, sy), (mid_x, sy), (mid_x, ey), (ex, ey)]
    else:
        mid_y = (sy + ey) // 2
        points = [(sx, sy), (sx, mid_y), (ex, mid_y), (ex, ey)]

    draw.line(points, fill=color, width=width, joint="curve")

    last_x, last_y = points[-2]
    dx = ex - last_x
    dy = ey - last_y
    dist = max(1.0, (dx * dx + dy * dy) ** 0.5)
    ux = dx / dist
    uy = dy / dist
    base_x = ex - ux * 10
    base_y = ey - uy * 10
    perp_x = -uy
    perp_y = ux
    left = (int(base_x + perp_x * 4), int(base_y + perp_y * 4))
    right = (int(base_x - perp_x * 4), int(base_y - perp_y * 4))
    draw.polygon([(ex, ey), left, right], fill=color)


def main() -> None:
    root = Path(".").resolve()
    output_main = root / "project_uml_repository.png"
    output_alias = root / "project_uml_file_imports.png"

    module_map = discover_modules(root)
    roots = ordered_roots(module_map)
    modules_by_root: dict[str, list[ModuleInfo]] = defaultdict(list)
    for info in module_map.values():
        modules_by_root[info.root].append(info)
    for info_list in modules_by_root.values():
        info_list.sort(key=lambda item: (item.path.as_posix(), item.module))

    measure_image = Image.new("RGB", (10, 10), "white")
    measure_draw = ImageDraw.Draw(measure_image)
    title_font = load_font(28)
    package_font = load_font(20)
    body_font = load_font(15)
    small_font = load_font(14)

    layouts = {
        pkg: compute_package_layout(measure_draw, modules_by_root[pkg], title_font=package_font, body_font=body_font)
        for pkg in roots
    }

    layer_rows = build_layer_rows(roots)
    outer_margin_x = 70
    outer_margin_y = 90
    row_gap = 90
    panel_gap = 55

    row_widths: list[int] = []
    row_heights: list[int] = []
    for row in layer_rows:
        row_width = sum(layouts[pkg].width for pkg in row) + max(0, len(row) - 1) * panel_gap
        row_height = max(layouts[pkg].height for pkg in row)
        row_widths.append(row_width)
        row_heights.append(row_height)

    image_width = max(2600, max(row_widths, default=0) + outer_margin_x * 2)
    image_height = max(2200, sum(row_heights) + max(0, len(row_heights) - 1) * row_gap + outer_margin_y * 2 + 120)

    image = Image.new("RGBA", (image_width, image_height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    edge_draw = ImageDraw.Draw(image, "RGBA")

    package_rects: dict[str, tuple[int, int, int, int]] = {}
    module_rects: dict[str, tuple[int, int, int, int]] = {}

    current_y = outer_margin_y + 90
    for row, row_height, row_width in zip(layer_rows, row_heights, row_widths):
        current_x = (image_width - row_width) // 2
        for pkg in row:
            layout = layouts[pkg]
            package_rect = (current_x, current_y, current_x + layout.width, current_y + layout.height)
            package_rects[pkg] = package_rect

            px1, py1, _, _ = package_rect
            inner_x = px1 + 18
            inner_y = py1 + layout.header_height + 18
            gap_x = 14
            gap_y = 10

            for index, info in enumerate(modules_by_root[pkg]):
                row_index = index // layout.columns
                col_index = index % layout.columns
                x1 = inner_x + col_index * (layout.cell_width + gap_x)
                y1 = inner_y + row_index * (layout.cell_height + gap_y)
                x2 = x1 + layout.cell_width
                y2 = y1 + layout.cell_height
                module_rects[info.module] = (x1, y1, x2, y2)

            current_x += layout.width + panel_gap
        current_y += row_height + row_gap

    edges = sorted(build_edges(root, module_map))
    for source, target in edges:
        start, end = edge_anchor(module_rects[source], module_rects[target])
        same_root = module_map[source].root == module_map[target].root
        color = (75, 75, 75, 120) if same_root else (35, 86, 160, 105)
        width = 1 if same_root else 2
        draw_polyline_arrow(edge_draw, start, end, color=color, width=width)

    for pkg in roots:
        x1, y1, x2, y2 = package_rects[pkg]
        fill = PACKAGE_COLORS[package_group(pkg)]
        draw.rounded_rectangle((x1, y1, x2, y2), radius=18, fill=fill, outline=(80, 80, 80), width=2)
        draw.line((x1, y1 + layouts[pkg].header_height, x2, y1 + layouts[pkg].header_height), fill=(145, 145, 145), width=1)
        draw.text((x1 + 18, y1 + 12), pkg, fill=(18, 18, 18), font=package_font)
        meta = f"{len(modules_by_root[pkg])} files"
        meta_w, _ = text_size(draw, meta, small_font)
        draw.text((x2 - meta_w - 18, y1 + 16), meta, fill=(80, 80, 80), font=small_font)

    for info in module_map.values():
        x1, y1, x2, y2 = module_rects[info.module]
        draw.rounded_rectangle((x1, y1, x2, y2), radius=8, fill=(255, 255, 255), outline=(115, 115, 115), width=1)

        line_sizes = [text_size(draw, line, body_font) for line in info.label_lines]
        total_text_height = sum(size[1] for size in line_sizes) + max(0, len(line_sizes) - 1) * 2
        current_y = y1 + (y2 - y1 - total_text_height) // 2 - 1
        for line, (_, line_h) in zip(info.label_lines, line_sizes):
            line_w, _ = text_size(draw, line, body_font)
            line_x = x1 + (x2 - x1 - line_w) // 2
            draw.text((line_x, current_y), line, fill=(20, 20, 20), font=body_font)
            current_y += line_h + 2

    draw.text((outer_margin_x, 28), "Repository UML File Import Diagram", fill=(0, 0, 0), font=title_font)
    subtitle = "Production modules only; every arrow is an internal Python file import relation resolved from AST."
    draw.text((outer_margin_x, 60), subtitle, fill=(78, 78, 78), font=body_font)

    legend_y = image_height - 95
    legend_items = [
        ("Cross-package import", (35, 86, 160, 170)),
        ("Intra-package import", (75, 75, 75, 170)),
    ]
    legend_x = outer_margin_x
    for label, color in legend_items:
        edge_draw.line((legend_x, legend_y + 10, legend_x + 32, legend_y + 10), fill=color, width=3)
        draw.text((legend_x + 42, legend_y), label, fill=(40, 40, 40), font=body_font)
        legend_x += 270

    rgb_image = image.convert("RGB")
    rgb_image.save(output_main, format="PNG")
    rgb_image.save(output_alias, format="PNG")
    print(f"Saved: {output_main}")
    print(f"Saved: {output_alias}")
    print(f"Modules: {len(module_map)}")
    print(f"Import relations: {len(edges)}")


if __name__ == "__main__":
    main()
