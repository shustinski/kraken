from __future__ import annotations

import ast
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
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
    "_internal",
}


@dataclass
class ClassInfo:
    fqcn: str
    module: str
    name: str
    inherits: set[str]
    associations: set[str]
    dependencies: set[str]


def should_skip(path: Path) -> bool:
    return any(part in SKIP_DIRS for part in path.parts)


def module_name(root: Path, path: Path) -> str:
    rel = path.relative_to(root)
    parts = list(rel.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


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


def extract_init_self_annotations(class_node: ast.ClassDef) -> list[str]:
    refs: list[str] = []
    for node in class_node.body:
        if not isinstance(node, ast.FunctionDef) or node.name != "__init__":
            continue
        arg_ann: dict[str, ast.AST] = {}
        args = list(node.args.args) + list(node.args.kwonlyargs)
        for arg in args:
            if arg.arg == "self":
                continue
            if arg.annotation is not None:
                arg_ann[arg.arg] = arg.annotation

        for stmt in ast.walk(node):
            if not isinstance(stmt, ast.Assign):
                continue
            if len(stmt.targets) != 1:
                continue
            target = stmt.targets[0]
            if not isinstance(target, ast.Attribute):
                continue
            if not isinstance(target.value, ast.Name) or target.value.id != "self":
                continue
            if isinstance(stmt.value, ast.Name):
                pname = stmt.value.id
                ann = arg_ann.get(pname)
                if ann is not None:
                    refs.extend(iter_type_names(ann))
    return refs


def resolve_ref(
    raw: str,
    current_module: str,
    classes_by_name: dict[str, set[str]],
    class_names: set[str],
    modules: set[str],
) -> str | None:
    token = raw.strip()
    if not token:
        return None
    token = token.replace("|", " ")
    for chunk in token.replace(",", " ").split():
        candidate = chunk.strip()
        if not candidate:
            continue
        # Direct fully-qualified class reference.
        if candidate in class_names:
            return candidate
        # Module.Class style.
        if "." in candidate:
            parts = candidate.split(".")
            for i in range(len(parts) - 1):
                mod = ".".join(parts[i:-1])
                cls = parts[-1]
                fqcn = f"{mod}.{cls}" if mod else cls
                if fqcn in class_names:
                    return fqcn
            # Relative module in same package.
            cls = parts[-1]
            if cls in classes_by_name:
                possible = classes_by_name[cls]
                if len(possible) == 1:
                    return next(iter(possible))
        else:
            # Simple class name.
            if candidate in classes_by_name:
                possible = classes_by_name[candidate]
                if len(possible) == 1:
                    return next(iter(possible))
                # Bias to same top-level package.
                top = current_module.split(".")[0] if current_module else ""
                same_top = [fq for fq in possible if fq.split(".")[0] == top]
                if len(same_top) == 1:
                    return same_top[0]
    return None


def dashed_line(draw: ImageDraw.ImageDraw, p1: tuple[int, int], p2: tuple[int, int], fill: tuple[int, int, int]) -> None:
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    dist = max(1.0, (dx * dx + dy * dy) ** 0.5)
    dash = 8.0
    gap = 6.0
    step = dash + gap
    t = 0.0
    while t < dist:
        a = t / dist
        b = min(dist, t + dash) / dist
        xa = int(x1 + dx * a)
        ya = int(y1 + dy * a)
        xb = int(x1 + dx * b)
        yb = int(y1 + dy * b)
        draw.line((xa, ya, xb, yb), fill=fill, width=1)
        t += step


def main() -> None:
    root = Path(".").resolve()
    output = root / "project_uml_class_diagram.png"

    py_files = [p for p in root.rglob("*.py") if not should_skip(p)]
    modules = {module_name(root, p) for p in py_files}

    class_nodes: dict[str, tuple[str, ast.ClassDef]] = {}
    classes_by_name: dict[str, set[str]] = defaultdict(set)

    for path in py_files:
        mod = module_name(root, path)
        if not mod:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except Exception:
            continue
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                fqcn = f"{mod}.{node.name}"
                class_nodes[fqcn] = (mod, node)
                classes_by_name[node.name].add(fqcn)

    class_names = set(class_nodes.keys())
    infos: dict[str, ClassInfo] = {}

    for fqcn, (mod, class_node) in class_nodes.items():
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
            elif isinstance(stmt, ast.FunctionDef):
                for arg in list(stmt.args.args) + list(stmt.args.kwonlyargs):
                    if arg.arg == "self":
                        continue
                    dep_raw.extend(iter_type_names(arg.annotation))
                dep_raw.extend(iter_type_names(stmt.returns))
                for inner in ast.walk(stmt):
                    if isinstance(inner, ast.Call):
                        dep_raw.extend(iter_type_names(inner.func))
            elif isinstance(stmt, ast.AsyncFunctionDef):
                for arg in list(stmt.args.args) + list(stmt.args.kwonlyargs):
                    if arg.arg == "self":
                        continue
                    dep_raw.extend(iter_type_names(arg.annotation))
                dep_raw.extend(iter_type_names(stmt.returns))
                for inner in ast.walk(stmt):
                    if isinstance(inner, ast.Call):
                        dep_raw.extend(iter_type_names(inner.func))

        assoc_raw.extend(extract_init_self_annotations(class_node))

        inherits: set[str] = set()
        associations: set[str] = set()
        dependencies: set[str] = set()

        for raw in inherits_raw:
            ref = resolve_ref(raw, mod, classes_by_name, class_names, modules)
            if ref and ref != fqcn:
                inherits.add(ref)
        for raw in assoc_raw:
            ref = resolve_ref(raw, mod, classes_by_name, class_names, modules)
            if ref and ref != fqcn and ref not in inherits:
                associations.add(ref)
        for raw in dep_raw:
            ref = resolve_ref(raw, mod, classes_by_name, class_names, modules)
            if ref and ref != fqcn and ref not in inherits and ref not in associations:
                dependencies.add(ref)

        infos[fqcn] = ClassInfo(
            fqcn=fqcn,
            module=mod,
            name=class_node.name,
            inherits=inherits,
            associations=associations,
            dependencies=dependencies,
        )

    g = nx.MultiDiGraph()
    for fqcn in sorted(infos):
        g.add_node(fqcn)
    for src, info in infos.items():
        for dst in info.inherits:
            g.add_edge(src, dst, rel="inherit")
        for dst in info.associations:
            g.add_edge(src, dst, rel="assoc")
        for dst in info.dependencies:
            g.add_edge(src, dst, rel="dep")

    if g.number_of_nodes() == 0:
        raise SystemExit("No classes found.")

    ug = g.to_undirected()
    k = 2.0 / math.sqrt(max(1, ug.number_of_nodes()))
    pos = nx.spring_layout(ug, seed=42, k=k, iterations=350)

    width, height = 16000, 12000
    margin = 280
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    def to_px(x: float, y: float) -> tuple[int, int]:
        nxv = (x - min_x) / (max_x - min_x + 1e-9)
        nyv = (y - min_y) / (max_y - min_y + 1e-9)
        return (
            int(margin + nxv * (width - 2 * margin)),
            int(margin + nyv * (height - 2 * margin)),
        )

    node_rects: dict[str, tuple[int, int, int, int]] = {}
    for n in g.nodes:
        x, y = to_px(*pos[n])
        cls = infos[n]
        label1 = cls.name
        label2 = cls.module
        w = max(140, int(max(len(label1), len(label2)) * 7 + 20))
        h = 42
        node_rects[n] = (x - w // 2, y - h // 2, x + w // 2, y + h // 2)

    def center(rect: tuple[int, int, int, int]) -> tuple[int, int]:
        x1, y1, x2, y2 = rect
        return (x1 + x2) // 2, (y1 + y2) // 2

    # Draw edges first.
    edge_colors = {
        "inherit": (33, 33, 33),
        "assoc": (26, 115, 232),
        "dep": (120, 120, 120),
    }
    for u, v, key, data in g.edges(keys=True, data=True):
        rel = str(data.get("rel", "dep"))
        p1 = center(node_rects[u])
        p2 = center(node_rects[v])
        color = edge_colors.get(rel, (120, 120, 120))
        if rel == "dep":
            dashed_line(draw, p1, p2, color)
        else:
            draw.line((p1[0], p1[1], p2[0], p2[1]), fill=color, width=1)

    # Draw nodes.
    for n in g.nodes:
        x1, y1, x2, y2 = node_rects[n]
        draw.rectangle((x1, y1, x2, y2), fill=(252, 252, 252), outline=(40, 40, 40), width=1)
        cls = infos[n]
        draw.text((x1 + 6, y1 + 5), cls.name, fill=(0, 0, 0), font=font)
        draw.line((x1, y1 + 20, x2, y1 + 20), fill=(160, 160, 160), width=1)
        draw.text((x1 + 6, y1 + 24), cls.module, fill=(70, 70, 70), font=font)

    title = "UML Class Diagram (Project Internal Classes Only)"
    draw.text((margin, 70), title, fill=(0, 0, 0), font=font)
    legend_y = 100
    legends = [
        ("Inheritance", edge_colors["inherit"]),
        ("Association", edge_colors["assoc"]),
        ("Dependency", edge_colors["dep"]),
    ]
    for text, color in legends:
        draw.line((margin, legend_y + 8, margin + 26, legend_y + 8), fill=color, width=2)
        draw.text((margin + 36, legend_y), text, fill=(0, 0, 0), font=font)
        legend_y += 22

    img.save(output, format="PNG")
    print(f"Saved: {output}")
    print(f"Classes: {g.number_of_nodes()}")
    print(f"Relations: {g.number_of_edges()}")


if __name__ == "__main__":
    main()
