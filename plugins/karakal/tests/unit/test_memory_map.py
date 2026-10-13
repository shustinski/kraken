"""Memory map: frames cut from a known memory are read back into the same bits."""

from __future__ import annotations

import numpy as np
import pytest

from karakal.core.memory_map import (
    CONFLICT,
    DEFECT,
    EMPTY,
    MODE_ONES,
    ONE,
    OUTSIDE,
    SEAM_BITS,
    SEAM_EMPTY,
    ZERO,
    MemoryMapConfig,
    Pitch,
    assemble_memory_map,
    estimate_pitch,
    frame_neighbours,
    read_frame_sites,
)

PX, PY = 37.75, 62.9
CELL_W, CELL_H = 22, 26
FRAME_W, FRAME_H = 760, 700
# Small synthetic frames share 2-3 known columns of 10 rows at a seam (real frames: ~96 sites).
CFG = MemoryMapConfig(min_compared=16)


class Memory:
    """Bits on a lattice: 1, 0 or -1 (no site); an optional second block shifted by part of a pitch."""

    def __init__(self, rows: int, cols: int, *, seed: int = 0, gap_col: int | None = None, shift_from: int | None = None, shift: float = 0.0):
        rng = np.random.default_rng(seed)
        self.bits = (rng.random((rows, cols)) < 0.67).astype(np.int8)
        if gap_col is not None:
            self.bits[:, gap_col] = -1
        # Inside the frames (they start left of it and the odd frame rows drift right by a site).
        self.origin = (75.0, 18.0)
        self.shift_from = shift_from
        self.shift = shift
        self.missing: set[tuple[int, int]] = set()
        self.conflict: set[tuple[int, int]] = set()
        self.merged: set[tuple[int, int]] = set()

    def centre(self, r: int, c: int) -> tuple[float, float]:
        dy = self.shift if self.shift_from is not None and c >= self.shift_from else 0.0
        return self.origin[0] + c * PX, self.origin[1] + r * PY + dy

    def render(self, fx: float, fy: float, value: int | None) -> np.ndarray:
        """Mask of a frame at (fx, fy) in the plane: cells holding value (None: no network)."""

        image = np.zeros((FRAME_H, FRAME_W), np.uint8)
        if value is None:
            return image
        rows, cols = self.bits.shape
        for r in range(rows):
            for c in range(cols):
                bit = int(self.bits[r, c])
                if bit < 0 or (r, c) in self.missing:
                    continue
                paint = bit == value or (r, c) in self.conflict
                if not paint:
                    continue
                x, y = self.centre(r, c)
                x0, y0 = int(round(x - fx - CELL_W / 2)), int(round(y - fy - CELL_H / 2))
                image[max(0, y0) : max(0, y0 + CELL_H), max(0, x0) : max(0, x0 + CELL_W)] = 255
                if (r, c) in self.merged:
                    image[max(0, y0 + 8) : max(0, y0 + 16), max(0, x0 + CELL_W) : max(0, x0 + int(PX) + 3)] = 255
        return image


def frames_of(memory: Memory, frame_rows: int, per_row: int, *, overlap_cols: int = 4, overlap_rows: int = 3):
    """Frame positions row by row, overlapping, with a sideways shift of one site every other row."""

    step_x = FRAME_W - overlap_cols * PX
    step_y = FRAME_H - overlap_rows * PY
    positions = []
    for fr in range(frame_rows):
        drift = PX if fr % 2 else 0.0
        for fc in range(per_row):
            positions.append((fc * step_x + drift - 10.0, fr * step_y - 5.0))
    return positions


def read_all(memory: Memory, positions, *, ones=True, zeros=True, pitch=Pitch(PX, PY)):
    return [
        read_frame_sites(
            memory.render(x, y, 1 if ones else None) if ones else None,
            memory.render(x, y, 0) if zeros else None,
            pitch,
            frame_id=f"F{index:03d}",
            config=CFG,
        )
        for index, (x, y) in enumerate(positions)
    ]


def _expected(memory: Memory, r: int, c: int, *, single: bool = False) -> int:
    bit = int(memory.bits[r, c])
    if bit < 0:
        return OUTSIDE
    if (r, c) in memory.merged or (r, c - 1) in memory.merged:
        return DEFECT
    if (r, c) in memory.missing:
        return EMPTY
    if (r, c) in memory.conflict:
        return CONFLICT
    if single:
        return ONE if bit == 1 else EMPTY
    return ONE if bit == 1 else ZERO


def _field_site(mmap, x: float, y: float) -> tuple[int, int] | None:
    """Field site of a plane point: the block whose shifted lattice puts a site there."""

    for block in mmap.blocks:
        fc = (x - mmap.origin[0] - block.offset[0]) / mmap.pitch.x
        fr = (y - mmap.origin[1] - block.offset[1]) / mmap.pitch.y
        col, row = int(round(fc)), int(round(fr))
        if abs(fc - col) < 0.2 and abs(fr - row) < 0.2:
            if block.row <= row < block.row + block.rows and block.col <= col < block.col + block.cols:
                return row, col
    return None


def _field_vs_memory(memory: Memory, mmap, positions, *, single: bool = False):
    """Compare every memory site with the field: returns (wrong, checked)."""

    # The map's plane is the memory plane moved by where the first frame was placed.
    frame = mmap.frames[0]
    shift_x, shift_y = frame.x - positions[0][0], frame.y - positions[0][1]
    wrong = checked = 0
    rows, cols = memory.bits.shape
    for r in range(rows):
        for c in range(cols):
            expect = _expected(memory, r, c, single=single)
            x, y = memory.centre(r, c)
            site = _field_site(mmap, x + shift_x, y + shift_y)
            got = OUTSIDE if site is None else int(mmap.field[site])
            checked += 1
            wrong += int(got != expect)
    return wrong, checked


def test_pitch_is_found_from_the_masks() -> None:
    memory = Memory(10, 22, seed=1)
    pitch = estimate_pitch([memory.render(0, 0, 1), memory.render(200, 100, 0)])
    assert pitch.x == pytest.approx(PX, abs=0.1) and pitch.y == pytest.approx(PY, abs=0.15)


def test_frames_are_read_back_into_the_memory_bits() -> None:
    memory = Memory(26, 64, seed=2, gap_col=30)
    memory.missing = {(5, 7), (12, 40)}
    memory.conflict = {(3, 50)}
    memory.merged = {(9, 12)}  # bridged to (9, 13), both ones
    memory.bits[9, 12] = memory.bits[9, 13] = 1
    positions = frames_of(memory, frame_rows=3, per_row=4)
    sites = read_all(memory, positions)
    mmap = assemble_memory_map(sites, Pitch(PX, PY), per_row=4, config=CFG)

    statuses = {seam.status for seam in mmap.seams}
    assert statuses == {SEAM_BITS}, mmap.seam_counts()
    assert len(mmap.seams) == len(frame_neighbours(12, 4))
    wrong, checked = _field_vs_memory(memory, mmap, positions)
    assert checked == 26 * 64 and wrong == 0
    assert len(mmap.blocks) == 2  # the gap column splits the memory
    assert mmap.disagreements == 0 and mmap.collisions == 0
    counts = mmap.counts()
    assert counts["empty"] == 2 and counts["conflict"] == 1
    # A field site leads back to a frame pixel on a cell of the right value.
    row, col = np.argwhere(mmap.field == ONE)[0]
    index, px, py = mmap.frame_point(int(row), int(col))
    x, y = positions[index]
    assert memory.render(x, y, 1)[int(round(py)), int(round(px))] == 255


def test_one_network_reads_ones_and_leaves_the_rest_empty() -> None:
    memory = Memory(26, 64, seed=3)
    memory.missing = {(4, 4)}
    positions = frames_of(memory, frame_rows=3, per_row=4)
    sites = read_all(memory, positions, zeros=False)
    mmap = assemble_memory_map(sites, Pitch(PX, PY), per_row=4, mode=MODE_ONES, config=CFG)
    assert {seam.status for seam in mmap.seams} == {SEAM_BITS}
    # Zeros and the missing cell alike read as «not painted».
    wrong, checked = _field_vs_memory(memory, mmap, positions, single=True)
    assert checked == 26 * 64 and wrong == 0


def test_a_block_shifted_by_half_a_pitch_is_its_own_block() -> None:
    memory = Memory(26, 64, seed=4, gap_col=31, shift_from=32, shift=PY / 2)
    positions = frames_of(memory, frame_rows=3, per_row=4)
    sites = read_all(memory, positions)
    assert any(len(s.phases) == 2 for s in sites)
    mmap = assemble_memory_map(sites, Pitch(PX, PY), per_row=4, config=CFG)
    assert {seam.status for seam in mmap.seams} == {SEAM_BITS}
    assert len(mmap.blocks) == 2 and mmap.collisions == 0
    wrong, checked = _field_vs_memory(memory, mmap, positions)
    assert checked == 26 * 64 and wrong == 0
    # A field site of the shifted block leads to the centre of its cell in a frame.
    row, col = np.argwhere((mmap.block_index == 1) & (mmap.field == ONE))[0]
    index, px, py = mmap.frame_point(int(row), int(col))
    x, y = positions[index]
    assert memory.render(x, y, 1)[int(round(py)), int(round(px))] == 255


def test_a_frame_without_cells_is_placed_by_the_usual_offset() -> None:
    memory = Memory(26, 64, seed=5)
    positions = frames_of(memory, frame_rows=3, per_row=4)
    sites = read_all(memory, positions)
    blank = np.zeros((FRAME_H, FRAME_W), np.uint8)
    sites[5] = read_frame_sites(blank, blank, Pitch(PX, PY), frame_id="F005", config=CFG)
    mmap = assemble_memory_map(sites, Pitch(PX, PY), per_row=4, config=CFG)
    empty = [seam for seam in mmap.seams if 5 in (seam.a, seam.b)]
    assert empty and all(seam.status == SEAM_EMPTY for seam in empty)
    assert mmap.frames[5].method == "estimated"
    # The others still read back right; frame 5's area is seen by its overlapping neighbours only in part.
    assert all(seam.status == SEAM_BITS for seam in mmap.seams if 5 not in (seam.a, seam.b))


def _write_run(tmp_path, memory: Memory, positions, *, ones=True, zeros=True):
    import cv2

    from karakal.core.memory_map import MapFrame

    frames = []
    for index, (x, y) in enumerate(positions):
        paths = {}
        for name, value, wanted in (("ones", 1, ones), ("zeros", 0, zeros)):
            if not wanted:
                continue
            folder = tmp_path / name
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"F{index:03d}.png"
            cv2.imwrite(str(path), memory.render(x, y, value))
            paths[name] = str(path)
        frames.append(MapFrame(f"F{index:03d}", paths.get("ones", ""), paths.get("zeros", "")))
    return frames


def test_a_run_of_mask_files_builds_the_map_with_progress(tmp_path) -> None:
    from karakal.core.memory_map import MODE_PAIR, BuildCancelled, build_memory_map

    memory = Memory(26, 64, seed=6, gap_col=30)
    positions = frames_of(memory, frame_rows=3, per_row=4)
    frames = _write_run(tmp_path, memory, positions)
    stages = []
    mmap = build_memory_map(
        frames, per_row=4, config=CFG, progress=lambda stage, done, total: stages.append(stage), workers=2, cache_dir=None
    )
    assert mmap.mode == MODE_PAIR
    assert {"pitch", "frames", "seams", "field"} <= set(stages)
    assert mmap.pitch.x == pytest.approx(PX, abs=0.1)
    assert {seam.status for seam in mmap.seams} == {SEAM_BITS}
    wrong, _checked = _field_vs_memory(memory, mmap, positions)
    assert wrong == 0
    with pytest.raises(BuildCancelled):
        build_memory_map(frames, per_row=4, config=CFG, cancelled=lambda: True, cache_dir=None)


def test_zeros_only_reads_zeros(tmp_path) -> None:
    from karakal.core.memory_map import MODE_ZEROS, build_memory_map

    memory = Memory(26, 64, seed=7)
    positions = frames_of(memory, frame_rows=3, per_row=4)
    mmap = build_memory_map(_write_run(tmp_path, memory, positions, ones=False), per_row=4, config=CFG, cache_dir=None)
    assert mmap.mode == MODE_ZEROS
    counts = mmap.counts()
    assert counts["zero"] == int((memory.bits == 0).sum())
    assert counts["empty"] == int((memory.bits == 1).sum())


def test_text_picture_table_and_summary(tmp_path) -> None:
    import csv
    import json

    import cv2

    from karakal.core.memory_map_export import (
        TxtFormat,
        blocks_format,
        export_csv,
        export_png,
        export_summary,
        export_txt,
        field_format,
        memory_map_image,
        memory_map_text,
    )

    memory = Memory(26, 64, seed=8, gap_col=30)
    memory.missing = {(2, 2)}
    positions = frames_of(memory, frame_rows=3, per_row=4)
    mmap = assemble_memory_map(read_all(memory, positions), Pitch(PX, PY), per_row=4, config=CFG)
    rows, cols = mmap.shape

    text = memory_map_text(mmap, field_format(mmap.mode)).splitlines()
    body = [line for line in text if not line.startswith("#")]
    assert len(body) == rows and any("?" in line for line in body)
    assert set("".join(body)) <= set("10? X!~")
    assert body[0].rstrip() == "".join({2: "1", 3: "0", 1: "?", 0: " "}[int(v)] for v in mmap.field[0]).rstrip()

    blocks = memory_map_text(mmap, blocks_format(mmap.mode)).splitlines()
    assert sum(line.startswith("# block ") for line in blocks) == 2
    first = next(i for i, line in enumerate(blocks) if line.startswith("# block 0"))
    assert blocks[first + 1].startswith(" 0 ")  # rows numbered inside the block

    own = TxtFormat(symbols={**field_format(mmap.mode).symbols, "one": "#", "zero": "-"}, cell_separator=",", row_numbers=True, header=False)
    own_lines = memory_map_text(mmap, own).splitlines()
    assert len(own_lines) == rows and "#" in own_lines[0] and "," in own_lines[0]
    assert TxtFormat.from_payload(own.to_payload()).symbols["one"] == "#"

    image = memory_map_image(mmap, cell_px=3, block_outlines=True, frame_outlines=True)
    assert image.shape == (rows * 3, cols * 3, 3)
    png = export_png(mmap, tmp_path / "карта.png", cell_px=2)
    assert cv2.imdecode(np.fromfile(str(png), np.uint8), cv2.IMREAD_COLOR).shape[:2] == (rows * 2, cols * 2)

    export_txt(mmap, tmp_path / "bits.txt", field_format(mmap.mode))
    assert (tmp_path / "bits.txt").read_text(encoding="utf-8").startswith("# Karakal memory map")

    table = list(csv.DictReader((export_csv(mmap, tmp_path / "cells.csv")).open(encoding="utf-8")))
    assert len(table) == int(np.count_nonzero(mmap.field != OUTSIDE))
    one = next(row for row in table if row["state"] == "one")
    frame_index = int(one["frame"][1:])
    x, y = positions[frame_index]
    assert memory.render(x, y, 1)[int(round(float(one["frame_y"]))), int(round(float(one["frame_x"])))] == 255

    summary = json.loads(export_summary(mmap, tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["cells"]["empty"] == 1 and len(summary["blocks"]) == 2 and summary["untrusted_seams"] == []


def test_frames_are_read_in_processes_and_kept_in_the_cache(tmp_path, monkeypatch) -> None:
    import karakal.core.memory_map as mm

    memory = Memory(26, 64, seed=9)
    positions = frames_of(memory, frame_rows=3, per_row=4)
    frames = _write_run(tmp_path / "masks", memory, positions)
    cache = tmp_path / "cache"
    first = mm.build_memory_map(frames, per_row=4, config=CFG, pitch=Pitch(PX, PY), workers=2, cache_dir=cache)
    assert len(list(cache.glob("*.pickle"))) == len(frames)

    # A second build reads no mask: every frame comes from the cache.
    def no_reading(*_args, **_kwargs):
        raise AssertionError("a cached frame was read again")

    monkeypatch.setattr(mm, "_read_batch", no_reading)
    second = mm.build_memory_map(frames, per_row=4, config=CFG, pitch=Pitch(PX, PY), workers=2, cache_dir=cache)
    assert np.array_equal(first.field, second.field)

    # A changed mask is read again, the rest still come from the cache.
    import os

    os.utime(frames[3].ones_path, ns=(1, 1))
    monkeypatch.undo()
    read = []
    original = mm._read_batch
    monkeypatch.setattr(mm, "_read_batch", lambda batch, pitch, cfg: read.extend(i for i, _f in batch) or original(batch, pitch, cfg))
    mm.build_memory_map(frames, per_row=4, config=CFG, pitch=Pitch(PX, PY), workers=1, cache_dir=cache)
    assert read == [3]


def test_threads_take_over_when_processes_cannot_start(tmp_path, monkeypatch) -> None:
    import concurrent.futures

    import karakal.core.memory_map as mm

    class Broken:
        def __init__(self, *args, **kwargs):
            raise OSError("no processes here")

    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", Broken)
    memory = Memory(26, 64, seed=10)
    positions = frames_of(memory, frame_rows=3, per_row=4)
    mmap = mm.build_memory_map(_write_run(tmp_path, memory, positions), per_row=4, config=CFG, workers=3, cache_dir=None)
    assert {seam.status for seam in mmap.seams} == {SEAM_BITS}


def test_one_network_keeps_an_all_zero_line_inside_a_block() -> None:
    memory = Memory(26, 64, seed=11)
    memory.bits[10, :] = 0  # a whole row of zeros: data, not a gap
    positions = frames_of(memory, frame_rows=3, per_row=4)
    mmap = assemble_memory_map(read_all(memory, positions, zeros=False), Pitch(PX, PY), per_row=4, mode=MODE_ONES, config=CFG)
    wrong, checked = _field_vs_memory(memory, mmap, positions, single=True)
    assert wrong == 0 and len(mmap.blocks) == 1


def test_orientations_turn_and_mirror_like_numpy() -> None:
    from karakal.core.memory_map_export import ORIENTATIONS, orient_grid

    grid = np.arange(12).reshape(3, 4)
    expected = {
        "none": grid,
        "rot90": np.rot90(grid, -1),
        "rot180": np.rot90(grid, 2),
        "rot270": np.rot90(grid, 1),
        "transpose": grid.T,
        "antitranspose": np.rot90(grid, 2).T,
        "flip_h": grid[:, ::-1],
        "flip_v": grid[::-1],
    }
    assert set(expected) == set(ORIENTATIONS)
    for name, want in expected.items():
        assert np.array_equal(orient_grid(grid, name), want), name


def _two_block_map():
    memory = Memory(26, 64, seed=31, gap_col=30)
    positions = frames_of(memory, frame_rows=3, per_row=4)
    return memory, positions, assemble_memory_map(read_all(memory, positions), Pitch(PX, PY), per_row=4, config=CFG)


def test_a_turned_field_keeps_every_site_and_its_frame_pixel() -> None:
    from karakal.core.memory_map_export import oriented_map

    _memory, _positions, mmap = _two_block_map()
    turned = oriented_map(mmap, "rot90")
    assert np.array_equal(turned.field, np.rot90(mmap.field, -1))
    rows, cols = mmap.shape
    for block, new in zip(mmap.blocks, turned.blocks):
        # rot90 clockwise: (r, c) -> (c, rows - 1 - r)
        assert (new.row, new.col, new.rows, new.cols) == (block.col, rows - (block.row + block.rows), block.cols, block.rows)
        assert np.all(turned.block_index[new.row : new.row + new.rows, new.col : new.col + new.cols][turned.field[new.row : new.row + new.rows, new.col : new.col + new.cols] != OUTSIDE] == block.index)
    row, col = (int(v) for v in np.argwhere(mmap.field == ONE)[5])
    new_row, new_col = col, rows - 1 - row
    assert turned.frame_point(new_row, new_col) == mmap.frame_point(row, col)


def test_blocks_turned_in_place_move_apart_and_keep_their_gap() -> None:
    from karakal.core.memory_map_export import oriented_map

    _memory, _positions, mmap = _two_block_map()
    left, right = sorted(mmap.blocks, key=lambda b: b.col)
    gap = right.col - (left.col + left.cols)
    flipped = oriented_map(mmap, block_orientation="flip_h")
    for block in mmap.blocks:
        box = (slice(block.row, block.row + block.rows), slice(block.col, block.col + block.cols))
        assert np.array_equal(flipped.field[box], mmap.field[box][:, ::-1])
    turned = oriented_map(mmap, block_orientation="rot90")
    new_left, new_right = sorted(turned.blocks, key=lambda b: b.col)
    assert (new_left.rows, new_left.cols) == (left.cols, left.rows)
    assert new_right.col - (new_left.col + new_left.cols) == gap
    box = (slice(new_left.row, new_left.row + new_left.rows), slice(new_left.col, new_left.col + new_left.cols))
    original = mmap.field[left.row : left.row + left.rows, left.col : left.col + left.cols]
    assert np.array_equal(turned.field[box], np.rot90(original, -1))
    # Turned in place and then the whole field: still every site once.
    both = oriented_map(mmap, "flip_v", "transpose")
    assert int((both.field != OUTSIDE).sum()) == int((mmap.field != OUTSIDE).sum())


def test_numbering_from_one_reverse_block_order_and_turned_table(tmp_path) -> None:
    import csv

    from karakal.core.memory_map_export import MemoryMapExport, memory_map_text, write_memory_map_files
    from karakal.core.memory_map_export import blocks_format
    from dataclasses import replace as dc_replace

    memory, positions, mmap = _two_block_map()
    text = memory_map_text(mmap, dc_replace(blocks_format(mmap.mode), number_from=1, block_order="reverse")).splitlines()
    headers = [line for line in text if line.startswith("# block ")]
    assert headers[0].startswith("# block 2") and headers[1].startswith("# block 1")
    first = text.index(headers[0])
    assert text[first + 1].split()[0] == "1"  # rows numbered from 1
    written = write_memory_map_files(
        mmap,
        MemoryMapExport(folder=str(tmp_path), name="t", txt_field=True, png=True, csv=True, summary=True,
                        field_orientation="rot180", block_orientation="flip_h", number_from=1),
    )
    rows = list(csv.DictReader((tmp_path / "t_cells.csv").open(encoding="utf-8")))
    assert "built_row" in rows[0]
    one = next(row for row in rows if row["state"] == "one")
    x, y = positions[int(one["frame"][1:])]
    assert memory.render(x, y, 1)[int(round(float(one["frame_y"]))), int(round(float(one["frame_x"])))] == 255
    assert min(int(row["row"]) for row in rows) == 1
    assert "turned or mirrored" in (tmp_path / "t_field.txt").read_text(encoding="utf-8")
    assert len(written) == 4


def test_defects_found_by_the_cell_analysis_mark_their_sites() -> None:
    memory = Memory(26, 64, seed=41)
    x, y = frames_of(memory, frame_rows=3, per_row=4)[0]
    ones, zeros = memory.render(x, y, 1), memory.render(x, y, 0)
    plain = read_frame_sites(ones, zeros, Pitch(PX, PY), config=CFG)
    target = int(np.flatnonzero(plain.state == ONE)[10])
    cx, cy = plain.x[target], plain.y[target]
    marked = read_frame_sites(ones, zeros, Pitch(PX, PY), defect_boxes=[(int(cx) - 11, int(cy) - 13, 22, 26)], config=CFG)
    assert marked.state[target] == DEFECT
    assert int((marked.state == DEFECT).sum()) == int((plain.state == DEFECT).sum()) + 1


def test_frames_per_row_are_found_from_the_bits(tmp_path) -> None:
    from karakal.core.memory_map import build_memory_map, detect_frames_per_row

    memory = Memory(26, 64, seed=51)
    positions = frames_of(memory, frame_rows=3, per_row=4)
    sites = read_all(memory, positions)
    assert detect_frames_per_row(sites, Pitch(PX, PY), CFG) == 4
    frames = _write_run(tmp_path, memory, positions)
    mmap = build_memory_map(frames, per_row=0, pitch=Pitch(PX, PY), config=CFG, cache_dir=None, use_processes=False)
    assert mmap.per_row == 4 and {s.status for s in mmap.seams} == {SEAM_BITS}
    # Frames that do not overlap downwards: no answer rather than a guess.
    assert detect_frames_per_row(read_all(memory, frames_of(memory, frame_rows=1, per_row=4)), Pitch(PX, PY), CFG) is None


def test_a_defect_marked_in_one_frame_only_does_not_break_its_seam() -> None:
    from karakal.core.memory_map import match_frames

    memory = Memory(26, 64, seed=61)
    positions = frames_of(memory, frame_rows=1, per_row=2)
    (ax, ay), (bx, by) = positions
    a = read_frame_sites(memory.render(ax, ay, 1), memory.render(ax, ay, 0), Pitch(PX, PY), config=CFG)
    # The left frame's analysis marks every cell of the overlap broken; the right frame does not.
    overlap = [(int(x) - 11, int(y) - 13, 22, 26) for x, y in zip(a.x, a.y) if x > FRAME_W - 4 * PX]
    a_marked = read_frame_sites(memory.render(ax, ay, 1), memory.render(ax, ay, 0), Pitch(PX, PY), defect_boxes=overlap, config=CFG)
    b = read_frame_sites(memory.render(bx, by, 1), memory.render(bx, by, 0), Pitch(PX, PY), config=CFG)
    true = (bx - ax, by - ay)
    assert match_frames(a, b, "right", Pitch(PX, PY), CFG).status == SEAM_BITS
    # Every third cell of the overlap marked in one frame only: the seam still matches.
    some = read_frame_sites(memory.render(ax, ay, 1), memory.render(ax, ay, 0), Pitch(PX, PY), defect_boxes=overlap[::3], config=CFG)
    seam = match_frames(some, b, "right", Pitch(PX, PY), CFG)
    assert seam.status == SEAM_BITS and abs(seam.offset[0] - true[0]) < 2 and abs(seam.offset[1] - true[1]) < 2
    # All of it marked: nothing to compare, and the seam is not trusted (never a wrong translation).
    assert match_frames(a_marked, b, "right", Pitch(PX, PY), CFG).status != SEAM_BITS


def test_turning_blocks_never_loses_a_site_in_any_layout() -> None:
    from karakal.core.memory_map import MapBlock, MemoryMap
    from karakal.core.memory_map_export import ORIENTATIONS, oriented_map

    # Like the real memory: a top band whose halves start at different rows, a wide band under it,
    # three small blocks side by side in the periphery band, a bottom band.
    rects = [(2, 0, 10, 30), (0, 34, 12, 26), (14, 1, 15, 28), (14, 33, 15, 27), (32, 5, 2, 6), (32, 13, 2, 5), (32, 40, 2, 7), (37, 0, 6, 60)]
    rows, cols = 43, 61
    field_ = np.zeros((rows, cols), np.uint8)
    block = np.full((rows, cols), -1, np.int32)
    rng = np.random.default_rng(3)
    blocks = []
    for index, (r, c, h, w) in enumerate(rects):
        field_[r : r + h, c : c + w] = rng.integers(2, 4, size=(h, w))
        block[r : r + h, c : c + w] = index
        blocks.append(MapBlock(index, r, c, h, w, h * w))
    mmap = MemoryMap("pair", Pitch(PX, PY), field_, np.zeros((rows, cols), np.int32), tuple(blocks), (), (), block_index=block)
    for field_orientation in ORIENTATIONS:
        for block_orientation in ORIENTATIONS:
            turned = oriented_map(mmap, field_orientation, block_orientation)
            assert int((turned.field != OUTSIDE).sum()) == int((field_ != OUTSIDE).sum()), (field_orientation, block_orientation)
            for b in turned.blocks:
                box = turned.block_index[b.row : b.row + b.rows, b.col : b.col + b.cols]
                assert int((box == b.index).sum()) == b.sites and set(np.unique(box)) <= {-1, b.index}
