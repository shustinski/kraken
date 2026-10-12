"""Memory map files: bits as text (whole field, by blocks or in the user's own format), a picture
of the map, a table of cells and a summary. Qt-free."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - OpenCV is a hard dependency of the app
    cv2 = None

from .memory_map import (
    CONFLICT,
    DEFECT,
    EMPTY,
    MODE_PAIR,
    ONE,
    OUTSIDE,
    PARTIAL,
    SEAM_BITS,
    STATE_NAMES,
    ZERO,
    MemoryMap,
)

LAYOUT_FIELD = "field"
LAYOUT_BLOCKS = "blocks"


def default_symbols(mode: str) -> dict[str, str]:
    """One character per state. EMPTY is a missing cell with two networks, «not painted» with one."""

    return {
        "one": "1",
        "zero": "0",
        "empty": "?" if mode == MODE_PAIR else ".",
        "conflict": "X",
        "defect": "!",
        "partial": "~",
        "outside": " ",
    }


@dataclass(frozen=True, slots=True)
class TxtFormat:
    """How the bits are written. The two presets are field_format() and blocks_format(); any field
    can be changed for the user's own variant."""

    layout: str = LAYOUT_FIELD
    symbols: dict[str, str] = field(default_factory=lambda: default_symbols(MODE_PAIR))
    cell_separator: str = ""
    row_numbers: bool = False
    column_ruler: bool = False
    header: bool = True
    # Blocks layout: a blank line between blocks.
    block_gap: bool = True
    # Leave service blocks out (their sites become OUTSIDE).
    main_only: bool = False

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_payload(payload: dict[str, Any] | None, mode: str = MODE_PAIR) -> "TxtFormat":
        base = TxtFormat(symbols=default_symbols(mode))
        if not isinstance(payload, dict):
            return base
        symbols = dict(base.symbols)
        for key, value in dict(payload.get("symbols") or {}).items():
            if key in symbols and isinstance(value, str):
                symbols[key] = value
        layout = str(payload.get("layout") or LAYOUT_FIELD)
        return TxtFormat(
            layout=layout if layout in (LAYOUT_FIELD, LAYOUT_BLOCKS) else LAYOUT_FIELD,
            symbols=symbols,
            cell_separator=str(payload.get("cell_separator") or ""),
            row_numbers=bool(payload.get("row_numbers", False)),
            column_ruler=bool(payload.get("column_ruler", False)),
            header=bool(payload.get("header", True)),
            block_gap=bool(payload.get("block_gap", True)),
            main_only=bool(payload.get("main_only", False)),
        )


def field_format(mode: str) -> TxtFormat:
    return TxtFormat(layout=LAYOUT_FIELD, symbols=default_symbols(mode))


def blocks_format(mode: str) -> TxtFormat:
    return TxtFormat(layout=LAYOUT_BLOCKS, symbols=default_symbols(mode), row_numbers=True)


def _lookup(fmt: TxtFormat) -> np.ndarray:
    table = np.full(max(STATE_NAMES) + 1, " ", dtype=object)
    for code, name in STATE_NAMES.items():
        table[code] = fmt.symbols.get(name, " ")
    return table


def _header_lines(mmap: MemoryMap, fmt: TxtFormat) -> list[str]:
    counts = mmap.counts()
    seams = mmap.seam_counts()
    legend = ", ".join(f"{fmt.symbols.get(name, '?')!r} {name}" for name in ("one", "zero", "empty", "conflict", "defect", "partial", "outside"))
    rows, cols = mmap.shape
    lines = [
        "# Karakal memory map",
        f"# networks: {mmap.mode}" + ("" if mmap.mode == MODE_PAIR else " (one network: 'empty' is a zero or a missing cell)"),
        f"# legend: {legend}",
        f"# field: {rows} rows x {cols} columns, {len(mmap.blocks)} blocks ("
        + (", ".join(str(b.index) for b in mmap.blocks if b.service) or "none")
        + " of them service"
        + ("; service blocks left out" if fmt.main_only else "")
        + f"), pitch {mmap.pitch.x:.3f} x {mmap.pitch.y:.3f} px",
        f"# frames: {len(mmap.frames)}; seams: " + ", ".join(f"{k} {v}" for k, v in sorted(seams.items())),
        "# cells: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items()) if k != "outside"),
        f"# disagreements between frames: {mmap.disagreements}; collisions: {mmap.collisions}",
        f"# layout: {fmt.layout}; first row and column are numbered 0",
    ]
    return lines


def _ruler(cols: int, offset: int, fmt: TxtFormat, label_width: int) -> list[str]:
    width = 1 + len(fmt.cell_separator)
    tens = [" "] * (cols * width)
    for col in range(cols):
        if (offset + col) % 10 == 0:
            text = str(offset + col)
            for i, ch in enumerate(text):
                if col * width + i < len(tens):
                    tens[col * width + i] = ch
    prefix = " " * label_width
    return [("# " if fmt.header else "") + prefix + "".join(tens).rstrip()]


def _rows_text(grid: np.ndarray, fmt: TxtFormat, *, first_row: int, first_col: int) -> list[str]:
    table = _lookup(fmt)
    chars = table[grid]
    label_width = len(str(first_row + max(0, grid.shape[0] - 1))) + 1 if fmt.row_numbers else 0
    lines = _ruler(grid.shape[1], first_col, fmt, label_width) if fmt.column_ruler else []
    sep = fmt.cell_separator
    for index, row in enumerate(chars):
        text = sep.join(row.tolist())
        if fmt.row_numbers:
            text = f"{first_row + index:>{label_width - 1}} " + text
        lines.append(text.rstrip() if not sep else text)
    return lines


def main_field(mmap: MemoryMap) -> np.ndarray:
    """The field without service blocks (their sites OUTSIDE)."""

    service = [b.index for b in mmap.blocks if b.service]
    if not service:
        return mmap.field
    return np.where(np.isin(mmap.block_index, service), OUTSIDE, mmap.field).astype(np.uint8)


def memory_map_text(mmap: MemoryMap, fmt: TxtFormat) -> str:
    lines = _header_lines(mmap, fmt) if fmt.header else []
    if fmt.layout == LAYOUT_BLOCKS:
        for block in mmap.blocks:
            if fmt.main_only and block.service:
                continue
            box = (slice(block.row, block.row + block.rows), slice(block.col, block.col + block.cols))
            grid = np.where(mmap.block_index[box] == block.index, mmap.field[box], OUTSIDE)
            if fmt.block_gap and lines:
                lines.append("")
            lines.append(
                f"# block {block.index}{' (service)' if block.service else ''}: field rows {block.row}..{block.row + block.rows - 1}, "
                f"columns {block.col}..{block.col + block.cols - 1} ({block.rows} x {block.cols})"
            )
            lines.extend(_rows_text(grid, fmt, first_row=0, first_col=0))
    else:
        lines.extend(_rows_text(main_field(mmap) if fmt.main_only else mmap.field, fmt, first_row=0, first_col=0))
    return "\n".join(lines) + "\n"


def export_txt(mmap: MemoryMap, path: Path | str, fmt: TxtFormat) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(memory_map_text(mmap, fmt), encoding="utf-8", newline="\n")
    return target


# --- Picture ------------------------------------------------------------------------------------

# BGR colours of the states on the map (OUTSIDE is the background).
STATE_COLORS: dict[int, tuple[int, int, int]] = {
    OUTSIDE: (40, 40, 40),
    EMPTY: (39, 159, 239),  # amber: missing / not painted
    ONE: (221, 138, 55),  # blue
    ZERO: (89, 196, 151),  # green
    CONFLICT: (74, 75, 226),  # red
    DEFECT: (126, 83, 212),  # pink
    PARTIAL: (128, 128, 128),
}
SINGLE_EMPTY_COLOR = (96, 96, 96)


def memory_map_image(
    mmap: MemoryMap,
    *,
    cell_px: int = 1,
    block_outlines: bool = False,
    frame_outlines: bool = False,
    colors: dict[int, tuple[int, int, int]] | None = None,
) -> np.ndarray:
    """BGR picture of the map: one square of cell_px pixels per site."""

    palette = dict(STATE_COLORS)
    if mmap.mode != MODE_PAIR:
        palette[EMPTY] = SINGLE_EMPTY_COLOR
    palette.update(colors or {})
    table = np.zeros((max(STATE_NAMES) + 1, 3), np.uint8)
    for code, bgr in palette.items():
        table[code] = bgr
    image = table[mmap.field]
    scale = max(1, int(cell_px))
    if scale > 1:
        image = np.repeat(np.repeat(image, scale, axis=0), scale, axis=1)
    if block_outlines:
        for block in mmap.blocks:
            cv2.rectangle(
                image,
                (block.col * scale, block.row * scale),
                ((block.col + block.cols) * scale - 1, (block.row + block.rows) * scale - 1),
                (220, 220, 220),
                1,
            )
    if frame_outlines:
        for frame in mmap.frames:
            # Frame corners in field sites (the frame edge lies half a site outside its last centres).
            x0 = (frame.x - mmap.origin[0]) / mmap.pitch.x + 0.5
            y0 = (frame.y - mmap.origin[1]) / mmap.pitch.y + 0.5
            x1 = x0 + frame.width / mmap.pitch.x
            y1 = y0 + frame.height / mmap.pitch.y
            cv2.rectangle(
                image,
                (int(round(x0 * scale)), int(round(y0 * scale))),
                (int(round(x1 * scale)) - 1, int(round(y1 * scale)) - 1),
                (74, 216, 255),
                1,
            )
    return image


def export_png(mmap: MemoryMap, path: Path | str, **options) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    ok, data = cv2.imencode(".png", memory_map_image(mmap, **options))
    if not ok:
        raise OSError(f"could not encode {target}")
    data.tofile(str(target))
    return target


# --- Table and summary --------------------------------------------------------------------------


def export_csv(mmap: MemoryMap, path: Path | str, *, chunk_rows: int = 64) -> Path:
    """One line per memory site: field row and column, block and place in it, state, and the frame
    (with the pixel) the state was read from."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    names = np.array([STATE_NAMES[code] for code in range(max(STATE_NAMES) + 1)], dtype=object)
    frame_ids = np.array([frame.frame_id for frame in mmap.frames] + [""], dtype=object)
    block_row0 = np.array([b.row for b in mmap.blocks] + [0])
    block_col0 = np.array([b.col for b in mmap.blocks] + [0])
    block_dx = np.array([b.offset[0] for b in mmap.blocks] + [0.0])
    block_dy = np.array([b.offset[1] for b in mmap.blocks] + [0.0])
    frame_x = np.array([f.x for f in mmap.frames] + [0.0])
    frame_y = np.array([f.y for f in mmap.frames] + [0.0])
    with target.open("w", encoding="utf-8", newline="\n") as out:
        out.write("row,col,block,block_row,block_col,state,frame,frame_x,frame_y\n")
        rows = mmap.field.shape[0]
        for start in range(0, rows, chunk_rows):
            sub = mmap.field[start : start + chunk_rows]
            r, c = np.nonzero(sub != OUTSIDE)
            r = r + start
            block = mmap.block_index[r, c]
            source = mmap.source[r, c]
            b = np.where(block >= 0, block, len(mmap.blocks))
            s = np.where(source >= 0, source, len(mmap.frames))
            px = mmap.origin[0] + c * mmap.pitch.x + block_dx[b] - frame_x[s]
            py = mmap.origin[1] + r * mmap.pitch.y + block_dy[b] - frame_y[s]
            lines = [
                f"{rr},{cc},{bb},{rr - block_row0[bi]},{cc - block_col0[bi]},{names[st]},{frame_ids[si]},{x:.1f},{y:.1f}"
                for rr, cc, bb, bi, st, si, x, y in zip(
                    r.tolist(), c.tolist(), block.tolist(), b.tolist(), mmap.field[r, c].tolist(), s.tolist(), px.tolist(), py.tolist()
                )
            ]
            if lines:
                out.write("\n".join(lines) + "\n")
    return target


def memory_map_summary(mmap: MemoryMap) -> dict[str, Any]:
    frames = mmap.frames
    return {
        "networks": mmap.mode,
        "field": {"rows": mmap.shape[0], "cols": mmap.shape[1]},
        "pitch": {"x": round(mmap.pitch.x, 4), "y": round(mmap.pitch.y, 4)},
        "cells": mmap.counts(),
        "blocks": [
            {"index": b.index, "row": b.row, "col": b.col, "rows": b.rows, "cols": b.cols, "sites": b.sites, "service": b.service}
            for b in mmap.blocks
        ],
        "frames": len(frames),
        "frames_placed_by_estimate": [f.frame_id for f in frames if f.method != SEAM_BITS],
        "seams": mmap.seam_counts(),
        "untrusted_seams": [
            {
                "a": frames[s.a].frame_id,
                "b": frames[s.b].frame_id,
                "kind": s.kind,
                "status": s.status,
                "agreement": round(s.agreement, 4),
                "compared": s.compared,
            }
            for s in mmap.seams
            if s.status != SEAM_BITS
        ],
        "disagreements": mmap.disagreements,
        "collisions": mmap.collisions,
    }


def export_summary(mmap: MemoryMap, path: Path | str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(memory_map_summary(mmap), ensure_ascii=False, indent=2), encoding="utf-8")
    return target


# --- One export, several files ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MemoryMapExport:
    """What the export window asked for."""

    folder: str
    name: str = "memory_map"
    txt_field: bool = True
    txt_blocks: bool = False
    txt_custom: bool = False
    custom: TxtFormat = field(default_factory=TxtFormat)
    png: bool = True
    png_cell_px: int = 2
    png_blocks: bool = True
    png_frames: bool = False
    csv: bool = False
    summary: bool = True
    main_only: bool = False


def main_only_map(mmap: MemoryMap) -> MemoryMap:
    """The map without its service blocks (for pictures and tables)."""

    from dataclasses import replace

    service = [b.index for b in mmap.blocks if b.service]
    if not service:
        return mmap
    gone = np.isin(mmap.block_index, service)
    return replace(
        mmap,
        field=np.where(gone, OUTSIDE, mmap.field).astype(np.uint8),
        source=np.where(gone, -1, mmap.source).astype(np.int32),
        block_index=np.where(gone, -1, mmap.block_index).astype(np.int32),
        blocks=tuple(b for b in mmap.blocks if not b.service),
    )


def write_memory_map_files(mmap: MemoryMap, choice: MemoryMapExport) -> list[Path]:
    from dataclasses import replace

    folder = Path(choice.folder)
    folder.mkdir(parents=True, exist_ok=True)
    base = choice.name.strip() or "memory_map"
    shown = main_only_map(mmap) if choice.main_only else mmap
    written: list[Path] = []
    if choice.txt_field:
        fmt = replace(field_format(mmap.mode), main_only=choice.main_only)
        written.append(export_txt(mmap, folder / f"{base}_field.txt", fmt))
    if choice.txt_blocks:
        fmt = replace(blocks_format(mmap.mode), main_only=choice.main_only)
        written.append(export_txt(mmap, folder / f"{base}_blocks.txt", fmt))
    if choice.txt_custom:
        written.append(export_txt(mmap, folder / f"{base}_custom.txt", replace(choice.custom, main_only=choice.main_only)))
    if choice.png:
        written.append(
            export_png(
                shown,
                folder / f"{base}.png",
                cell_px=choice.png_cell_px,
                block_outlines=choice.png_blocks,
                frame_outlines=choice.png_frames,
            )
        )
    if choice.csv:
        written.append(export_csv(shown, folder / f"{base}_cells.csv"))
    if choice.summary:
        written.append(export_summary(mmap, folder / f"{base}_summary.json"))
    return written
