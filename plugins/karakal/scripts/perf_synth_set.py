"""Large synthetic memory-array run with an answer key, for profiling and quality checks.

    python scripts/perf_synth_set.py OUT_DIR [--cols 100] [--rows 100] [--workers N]

Two layers like a real export (Snapshot 3): the ones network (TEST_00000.jpg) and the zeros network
(TEST_inv_00000.jpg) of one memory array, cols x rows frames each (100 x 100 = 2 x 10 000 frames),
2000 x 2000 JPEG masks with confidence maps beside them:

- OUT/ones/Весь прогон/Маски/TEST_00000.jpg,   .../Confidence/TEST_00000_confidence.jpg
- OUT/zeros/Весь прогон/Маски/TEST_inv_00000.jpg, .../Confidence/TEST_inv_00000_confidence.jpg
- OUT/answers.jsonl  one line per frame position: planted cell defects per layer, dirt regions,
  uncertain-only regions (confidence drops, mask intact), ones/zeros conflicts; boxes in frame px.

Frames outside the array (the border ring) are empty or hold a few stray blobs, as on a real run.
The ones and zeros layers are complementary on one lattice (a bit is a cell in exactly one layer),
so the ones/zeros pair has conflicts only where they were planted (and where a cell was removed).
Confidence maps are certainty maps (255 = sure), darker on cell edges, like the real network's.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parents[1] / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from karakal.core.synthetic_masks import SyntheticFrame, _draw_cell, add_defect  # noqa: E402

SIZE = 2000
PITCH_X = 56
PITCH_Y = 84
CELL_W = 27
ONES_H = 74
ZEROS_H = 52
# Share of bits that are ones (ones cells are sparse on the real array).
ONES_SHARE = 0.2
MARGIN_FRAMES = 10
DEFECT_KINDS = ("bite", "cut_corner", "short", "wrong_size", "debris", "bridge", "split", "removed")
JPEG_QUALITY = 90


def _layer_dirs(out: Path) -> dict[str, tuple[Path, Path, str]]:
    return {
        "ones": (out / "ones" / "Весь прогон" / "Маски", out / "ones" / "Весь прогон" / "Confidence", "TEST_"),
        "zeros": (out / "zeros" / "Весь прогон" / "Маски", out / "zeros" / "Весь прогон" / "Confidence", "TEST_inv_"),
    }


def _write(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise RuntimeError(f"cannot encode {path}")
    encoded.tofile(str(path))


def _array_bounds(cols: int, rows: int) -> tuple[int, int, int, int]:
    """Global pixel box of the memory array: a ring of frames around it stays outside."""

    margin_x = min(MARGIN_FRAMES, max(0, cols // 5))
    margin_y = min(MARGIN_FRAMES, max(0, rows // 5))
    return (
        margin_x * SIZE + 300,
        margin_y * SIZE + 500,
        (cols - margin_x) * SIZE - 700,
        (rows - margin_y) * SIZE - 400,
    )


def _slots(col: int, row: int, bounds: tuple[int, int, int, int]) -> list[tuple[int, int, int, int]]:
    """Lattice slots (global i, j, frame x, frame y) whose cell lies inside this frame and the array."""

    gx0, gy0 = col * SIZE, row * SIZE
    ax0, ay0, ax1, ay1 = bounds
    slots = []
    i0 = max(0, (max(gx0, ax0) - ax0) // PITCH_X - 1)
    j0 = max(0, (max(gy0, ay0) - ay0) // PITCH_Y - 1)
    i = i0
    while ax0 + i * PITCH_X < min(gx0 + SIZE, ax1):
        gx = ax0 + i * PITCH_X
        j = j0
        while ay0 + j * PITCH_Y < min(gy0 + SIZE, ay1):
            gy = ay0 + j * PITCH_Y
            if gx + CELL_W <= ax1 and gy + ONES_H <= ay1:
                slots.append((i, j, gx - gx0, gy - gy0))
            j += 1
        i += 1
    return slots


def _bit(i: int, j: int, seed: int) -> bool:
    value = (i * 73856093) ^ (j * 19349663) ^ (seed * 83492791)
    value = (value * 2654435761) & 0xFFFFFFFF
    return (value / 0xFFFFFFFF) < ONES_SHARE


def _random_box(rng: np.random.Generator, radius: int) -> tuple[int, int, int, int]:
    cx = int(rng.integers(radius, SIZE - radius))
    cy = int(rng.integers(radius, SIZE - radius))
    return cx, cy, radius, radius


def _blob(rng: np.random.Generator, cx: int, cy: int, radius: int) -> np.ndarray:
    """Irregular filled region (union of ellipses) around a centre, as a bool image."""

    region = np.zeros((SIZE, SIZE), dtype=np.uint8)
    for _ in range(int(rng.integers(4, 9))):
        ox = int(cx + rng.normal(0, radius * 0.35))
        oy = int(cy + rng.normal(0, radius * 0.35))
        axes = (int(rng.uniform(0.3, 0.7) * radius), int(rng.uniform(0.3, 0.7) * radius))
        cv2.ellipse(region, (ox, oy), axes, float(rng.uniform(0, 180)), 0, 360, 255, -1)
    return region > 0


def _box_of(region: np.ndarray) -> list[int]:
    ys, xs = np.nonzero(region)
    if xs.size == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]


def _junk(rng: np.random.Generator, region: np.ndarray, density: float) -> np.ndarray:
    """Dirt drawn by a network: grains and smears inside the region."""

    junk = np.zeros((SIZE, SIZE), dtype=np.uint8)
    ys, xs = np.nonzero(region)
    count = int(xs.size * density / 60.0)
    if count == 0:
        return junk > 0
    picks = rng.integers(0, xs.size, size=count)
    for index in picks:
        if rng.random() < 0.6:
            cv2.circle(junk, (int(xs[index]), int(ys[index])), int(rng.integers(2, 9)), 255, -1)
        else:
            axes = (int(rng.integers(4, 20)), int(rng.integers(2, 8)))
            cv2.ellipse(junk, (int(xs[index]), int(ys[index])), axes, float(rng.uniform(0, 180)), 0, 360, 255, -1)
    return (junk > 0) & region


def _certainty(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Certainty map of a clean network output: sure everywhere, less sure on object edges."""

    binary = (mask > 127).astype(np.uint8)
    edge = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    certainty = np.full(mask.shape, 255.0, dtype=np.float32)
    certainty -= edge.astype(np.float32) * float(rng.uniform(70, 100))
    certainty = cv2.GaussianBlur(certainty, (3, 3), 0)
    return certainty


def _lower(certainty: np.ndarray, region: np.ndarray, rng: np.random.Generator, low: float, high: float) -> None:
    """Uncertain patch: certainty between low and high, smooth, with grain."""

    soft = cv2.GaussianBlur(region.astype(np.float32), (0, 0), 12)
    noise = rng.uniform(low, high, size=(SIZE // 16 + 1, SIZE // 16 + 1)).astype(np.float32)
    noise = cv2.resize(noise, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)[:SIZE, :SIZE]
    np.minimum(certainty, certainty * (1.0 - soft) + noise * soft, out=certainty)


WINDOW = 400


def _defect_in_window(frame: SyntheticFrame, kind: str, rng: np.random.Generator, taken: list) -> dict | None:
    """One defect drawn by add_defect in a window of the frame (add_defect is quadratic in cells).

    The frame's mask is changed in place; the defect's box is returned in frame pixels."""

    x0 = int(rng.integers(0, SIZE - WINDOW))
    y0 = int(rng.integers(0, SIZE - WINDOW))
    cells = tuple(
        (x - x0, y - y0, w, h)
        for x, y, w, h in frame.cells
        if x >= x0 + 4 and y >= y0 + 4 and x + w <= x0 + WINDOW - 4 and y + h <= y0 + WINDOW - 4
    )
    if len(cells) < 3:
        return None
    window = frame.mask[y0 : y0 + WINDOW, x0 : x0 + WINDOW]
    local = SyntheticFrame(window.copy(), window.copy(), np.zeros(window.shape, dtype=bool), cells)
    # Cells already hit keep their defect: a window never reuses a changed cell.
    changed = add_defect(local, kind, seed=int(rng.integers(0, 2**31)))
    if not changed.defects:
        return None
    defect = changed.defects[-1]
    x, y, w, h = defect.bbox
    box = [x + x0, y + y0, w, h]
    gap = PITCH_X
    for tx, ty, tw, th in taken:
        if box[0] < tx + tw + gap and tx < box[0] + w + gap and box[1] < ty + th + gap and ty < box[1] + h + gap:
            return None
    window[:, :] = changed.mask
    taken.append(box)
    return {"kind": defect.kind, "class": int(defect.error_class), "box": box}


def make_position(task: tuple[int, int, int, int, str, int]) -> dict:
    col, row, cols, rows, out_text, seed = task
    out = Path(out_text)
    index = row * cols + col
    rng = np.random.default_rng((seed, index))
    bounds = _array_bounds(cols, rows)
    slots = _slots(col, row, bounds)
    answer: dict = {"index": index, "col": col, "row": row, "in_array": bool(slots), "layers": {}}

    layers: dict[str, SyntheticFrame] = {}
    for layer, height, want in (("ones", ONES_H, True), ("zeros", ZEROS_H, False)):
        mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
        boxes = []
        for i, j, x, y in slots:
            if _bit(i, j, seed) == want:
                box = (x, y, CELL_W, height)
                _draw_cell(mask, box, 3)
                boxes.append(box)
        layers[layer] = SyntheticFrame(mask, mask.copy(), np.zeros(mask.shape, dtype=bool), tuple(boxes))

    if not slots:
        # Periphery: empty frames, some with a few stray blobs.
        for layer in layers:
            frame = layers[layer]
            if rng.random() < 0.3:
                for _ in range(int(rng.integers(1, 6))):
                    cx, cy, r, _ = _random_box(rng, int(rng.integers(8, 20)))
                    cv2.circle(frame.mask, (cx, cy), r, 255, -1)
        answer["layers"] = {layer: {"defects": []} for layer in layers}
    else:
        # Cell defects, drawn by the library that the unit tests use (its answers are known).
        for layer in layers:
            frame = layers[layer]
            defects = []
            taken: list = []
            if len(frame.cells) >= 6 and rng.random() < 0.35:
                for _step in range(int(rng.integers(1, 4))):
                    kind = str(rng.choice(DEFECT_KINDS))
                    found = _defect_in_window(frame, kind, rng, taken)
                    if found is not None:
                        defects.append(found)
            answer["layers"][layer] = {"defects": defects}

    certainty = {layer: _certainty(frame.mask, rng) for layer, frame in layers.items()}
    for layer in layers:
        for defect in answer["layers"][layer]["defects"]:
            x, y, w, h = defect["box"]
            patch = certainty[layer][max(0, y - 3) : y + h + 3, max(0, x - 3) : x + w + 3]
            np.minimum(patch, rng.uniform(150, 215), out=patch)

    dirt: list[dict] = []
    uncertain: list[dict] = []
    conflicts: list[dict] = []
    if slots:
        roll = rng.random()
        if roll < 0.05:
            # Large dirt: grains and smears over many cells, cells partly lost, low certainty.
            cx, cy, radius, _ = _random_box(rng, int(rng.integers(120, 320)))
            region = _blob(rng, cx, cy, radius)
            for layer, frame in layers.items():
                junk = _junk(rng, region, float(rng.uniform(0.6, 1.0)))
                frame.mask[junk] = 255
                eat = _junk(rng, region, 0.4)
                frame.mask[eat & ~junk] = 0
                _lower(certainty[layer], region, rng, 60, 170)
            dirt.append({"kind": "large", "box": _box_of(region)})
        elif roll < 0.08:
            # Small dirt: a particle cluster about a cell pitch across.
            cx, cy, radius, _ = _random_box(rng, int(rng.integers(45, 80)))
            region = _blob(rng, cx, cy, radius)
            for layer, frame in layers.items():
                junk = _junk(rng, region, 1.2)
                frame.mask[junk] = 255
                _lower(certainty[layer], region, rng, 80, 190)
            dirt.append({"kind": "small", "box": _box_of(region)})
        elif roll < 0.11:
            # The network is unsure but drew the mask right.
            cx, cy, radius, _ = _random_box(rng, int(rng.integers(150, 300)))
            region = _blob(rng, cx, cy, radius)
            for layer in layers:
                _lower(certainty[layer], region, rng, 40, 140)
            uncertain.append({"kind": "uncertain", "box": _box_of(region)})
        elif roll < 0.14:
            # Both networks painted the same cells (a ones/zeros conflict).
            ones = layers["ones"]
            if ones.cells:
                pick = ones.cells[int(rng.integers(0, len(ones.cells)))]
                x, y, _w, _h = pick
                for bx, by, bw, bh in ones.cells:
                    if abs(bx - x) <= 2 * PITCH_X and abs(by - y) <= PITCH_Y:
                        _draw_cell(layers["zeros"].mask, (bx, by + (ONES_H - ZEROS_H) // 2, CELL_W, ZEROS_H), 3)
                        conflicts.append({"kind": "both", "box": [bx, by, bw, bh]})

    answer["dirt"] = dirt
    answer["uncertain"] = uncertain
    answer["conflicts"] = conflicts

    for layer, (mask_dir, confidence_dir, prefix) in _layer_dirs(out).items():
        name = f"{prefix}{index:05d}"
        _write(mask_dir / f"{name}.jpg", layers[layer].mask)
        noisy = certainty[layer] + rng.normal(0, 1.2, size=(SIZE, SIZE)).astype(np.float32)
        _write(confidence_dir / f"{name}_confidence.jpg", np.clip(noisy, 0, 255).astype(np.uint8))
    return answer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", type=Path)
    parser.add_argument("--cols", type=int, default=100)
    parser.add_argument("--rows", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args(argv)

    for mask_dir, confidence_dir, _prefix in _layer_dirs(args.out).values():
        mask_dir.mkdir(parents=True, exist_ok=True)
        confidence_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(col, row, args.cols, args.rows, str(args.out), args.seed) for row in range(args.rows) for col in range(args.cols)]
    started = time.perf_counter()
    answers: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for number, answer in enumerate(pool.map(make_position, tasks, chunksize=16), start=1):
            answers.append(answer)
            if number % 500 == 0:
                print(f"{number}/{len(tasks)} positions, {time.perf_counter() - started:.0f} s", flush=True)
    answers.sort(key=lambda item: item["index"])
    with (args.out / "answers.jsonl").open("w", encoding="utf-8") as handle:
        for answer in answers:
            handle.write(json.dumps(answer, ensure_ascii=False) + "\n")
    meta = {
        "cols": args.cols,
        "rows": args.rows,
        "frames_per_layer": len(tasks),
        "seed": args.seed,
        "in_array": sum(1 for a in answers if a["in_array"]),
        "dirt": sum(len(a["dirt"]) for a in answers),
        "uncertain": sum(len(a["uncertain"]) for a in answers),
        "conflict_frames": sum(1 for a in answers if a["conflicts"]),
        "defects": sum(len(layer["defects"]) for a in answers for layer in a["layers"].values()),
        "seconds": round(time.perf_counter() - started, 1),
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
