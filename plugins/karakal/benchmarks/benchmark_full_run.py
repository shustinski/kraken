"""Whole Karakal run in the real window (offscreen): cell check, dirt, validation; timings and results.

    python benchmarks/benchmark_full_run.py DATASET OUT.json [--phases cells,validation]
        [--limit N] [--profile] [--work-dir DIR] [--keep-cache]

DATASET is a folder made by scripts/perf_synth_set.py (ones/ and zeros/ layers like a real export,
answers.jsonl). The window is driven the way the operator does it: both layers with their confidence
folders, «Проверка ячеек» -> run (dirt is looked for with it), then «Валидация» -> run.

OUT.json gets per phase the wall time, the longest stall of the UI thread (the window's event loop
is ticked every 20 ms; a long gap is a frozen window) and, with --profile, the top functions of the
UI thread (cProfile). OUT.json also keeps a digest of every result (cells per frame, dirt regions,
validation scores and findings), so two runs can be compared exactly:

    python benchmarks/benchmark_full_run.py --compare BEFORE.json AFTER.json

Caches, projects and settings go to --work-dir (wiped first unless --keep-cache), never to the
user's profile.
"""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import io
import json
import os
import pickle
import pstats
import shutil
import sys
import time
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parents[1]


def _prepare_environment(work_dir: Path, keep_cache: bool, source_root: Path | None = None) -> None:
    if work_dir.exists() and not keep_cache:
        shutil.rmtree(work_dir, ignore_errors=True)
    for name in ("cache", "data", "projects", "local"):
        (work_dir / name).mkdir(parents=True, exist_ok=True)
    os.environ["KARAKAL_CACHE_DIR"] = str(work_dir / "cache")
    os.environ["KARAKAL_DATA_DIR"] = str(work_dir / "data")
    os.environ["KARAKAL_PROJECTS_DIR"] = str(work_dir / "projects")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("KARAKAL_HOME_PAGE", "0")
    if os.name == "nt" and os.path.isdir(r"C:\Windows\Fonts"):
        os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")
    # --src runs another copy of the code (a baseline exported from git) the same way.
    root = Path(source_root) if source_root else PLUGIN_ROOT.parents[1]
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root / "plugins" / "karakal" / "src"))
    os.environ["PYTHONPATH"] = os.pathsep.join([str(root / "plugins" / "karakal" / "src"), str(root / "src")])


def _limit_dataset(dataset: Path, limit: int, work_dir: Path, stride: int = 1, offset: int = 0) -> Path:
    """Every `stride`-th frame of each layer, at most `limit` of them, as hard links (the same
    files, a smaller run spread over the whole array)."""

    stride = max(1, int(stride))
    if limit <= 0 and stride == 1 and offset <= 0:
        return dataset
    target = work_dir / "dataset"
    for layer in ("ones", "zeros"):
        for sub in ("Маски", "Confidence"):
            source = dataset / layer / "Весь прогон" / sub
            out = target / layer / "Весь прогон" / sub
            out.mkdir(parents=True, exist_ok=True)
            chosen = sorted(source.iterdir())[max(0, int(offset)) :: stride]
            for path in chosen[:limit] if limit > 0 else chosen:
                link = out / path.name
                if not link.exists():
                    try:
                        os.link(path, link)
                    except OSError:
                        shutil.copy2(path, link)
    return target


# Roots of the run's files (set in run()): paths inside results are hashed without them, so two
# runs from different work folders compare.
_ROOTS: list[bytes] = []
# Model id -> "ones" / "zeros" (ids are hashes of folder paths).
_LAYER_NAMES: dict[str, str] = {}


def _without_root(value):
    """A frame result with its frame path relative to the run's folder (pickle keeps string lengths,
    so a path cannot be replaced in the bytes afterwards)."""

    path = getattr(value, "frame_path", None)
    if isinstance(path, str) and _ROOTS:
        import dataclasses

        text = path.replace("\\", "/")
        for root in _ROOTS:
            prefix = root.decode("utf-8").replace("\\", "/")
            if text.startswith(prefix):
                text = "<ROOT>" + text[len(prefix):]
                break
        try:
            return dataclasses.replace(value, frame_path=text)
        except (TypeError, ValueError):
            return value
    return value


def _digest(value) -> str:
    # repr, not pickle: pickle also encodes which equal objects are shared (one result object for
    # both layers of a frame, say), which says nothing about the values. repr of floats is exact.
    data = repr(_without_root(value)).encode("utf-8")
    for root in _ROOTS:
        data = data.replace(root, b"<ROOT>")
    for model_id, name in _LAYER_NAMES.items():
        data = data.replace(model_id.encode("utf-8"), name.encode("utf-8"))
    return hashlib.sha256(data).hexdigest()[:16]


def _name(model_id: str) -> str:
    text = str(model_id)
    for key, name in _LAYER_NAMES.items():
        text = text.replace(key, name)
    return text


class _Stalls:
    """Longest gaps between ticks of a 20 ms timer in the UI thread."""

    def __init__(self) -> None:
        from PyQt6.QtCore import QTimer

        self.last = time.perf_counter()
        self.gaps: list[float] = []
        self.timer = QTimer()
        self.timer.setInterval(20)
        self.timer.timeout.connect(self._tick)

    def _tick(self) -> None:
        now = time.perf_counter()
        self.gaps.append(now - self.last)
        self.last = now

    def start(self) -> None:
        self.gaps.clear()
        self.last = time.perf_counter()
        self.timer.start()

    def stop(self) -> dict:
        self.timer.stop()
        gaps = sorted(self.gaps)
        if not gaps:
            return {"max_ms": 0.0, "p99_ms": 0.0, "over_200ms": 0, "over_1s": 0}
        return {
            "max_ms": round(gaps[-1] * 1000, 1),
            "p99_ms": round(gaps[int(len(gaps) * 0.99) - 1 if len(gaps) > 1 else 0] * 1000, 1),
            "over_200ms": sum(1 for gap in gaps if gap > 0.2),
            "over_1s": sum(1 for gap in gaps if gap > 1.0),
            "stalled_s": round(sum(gap - 0.02 for gap in gaps if gap > 0.1), 1),
        }



class _UiSampler:
    """Where the UI thread is, sampled every 5 ms from another thread (cProfile of Python 3.12+
    sees every thread at once, so it cannot tell what froze the window).

    Each sample is weighted by the time since the previous one: a long Qt (C++) call holds the
    interpreter lock, the sampler cannot run during it, and the gap is put on the stack the UI
    thread shows right after it (the Python caller of that call). The longest gaps keep the stacks
    of every thread.
    """

    def __init__(self, interval: float = 0.005) -> None:
        import threading

        self.interval = interval
        self.thread_id = threading.main_thread().ident
        self.leaf: dict[str, float] = {}
        self.inclusive: dict[str, float] = {}
        self.busy = 0.0
        self.gaps: list[tuple[float, float, dict[str, list[str]]]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._started = 0.0

    @staticmethod
    def _stack(frame) -> list[str]:
        names = []
        while frame is not None:
            code = frame.f_code
            names.append(f"{Path(code.co_filename).name}:{frame.f_lineno}({code.co_name})")
            frame = frame.f_back
        return names

    def _run(self) -> None:
        own = Path(__file__).name
        last = time.perf_counter()
        while not self._stop.wait(self.interval):
            now = time.perf_counter()
            elapsed, last = now - last, now
            frames = sys._current_frames()
            frame = frames.get(self.thread_id)
            if frame is None:
                continue
            names = self._stack(frame)
            # Idle is the harness sleeping between event loop passes; time inside processEvents
            # with no Python frame above it is Qt's own work and counts as busy.
            idle = bool(names) and names[0].startswith(own) and names[0].endswith("(_wait)") and self._sleep_line(frame)
            if elapsed > 0.25:
                stacks = {str(ident): self._stack(item)[:14] for ident, item in frames.items()}
                self.gaps.append((round(elapsed, 3), round(now - self._started, 2), stacks))
                self.gaps.sort(key=lambda item: -item[0])
                del self.gaps[12:]
            if idle:
                continue
            self.busy += elapsed
            leaf = names[0]
            self.leaf[leaf] = self.leaf.get(leaf, 0.0) + elapsed
            for name in set(item.split(":", 1)[0] + ":" + item.split("(", 1)[-1] for item in names):
                if not name.startswith(own):
                    self.inclusive[name] = self.inclusive.get(name, 0.0) + elapsed

    @staticmethod
    def _sleep_line(frame) -> bool:
        import linecache

        return "sleep" in linecache.getline(frame.f_code.co_filename, frame.f_lineno)

    def start(self) -> None:
        self._started = time.perf_counter()
        self._thread.start()

    def stop(self) -> dict:
        self._stop.set()
        self._thread.join()

        def top(table: dict[str, float]) -> list[dict]:
            return [
                {"function": name, "busy_s": round(value, 2)}
                for name, value in sorted(table.items(), key=lambda item: -item[1])[:45]
            ]

        return {
            "busy_s": round(self.busy, 1),
            "by_leaf": top(self.leaf),
            "by_inclusive": top(self.inclusive),
            "longest_gaps": [{"gap_s": gap, "at_s": at, "stacks": stacks} for gap, at, stacks in self.gaps],
        }


def _wait(app, condition, timeout: float) -> bool:
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        app.processEvents()
        if condition():
            return True
        time.sleep(0.005)
    return False


def _top_functions(profile: cProfile.Profile, count: int = 40) -> list[dict]:
    stats = pstats.Stats(profile, stream=io.StringIO())
    rows = []
    for (filename, line, name), (cc, nc, tt, ct, _callers) in stats.stats.items():
        rows.append({"function": f"{Path(filename).name}:{line}({name})", "calls": nc, "self_s": round(tt, 3), "total_s": round(ct, 3)})
    rows.sort(key=lambda row: row["self_s"], reverse=True)
    by_self = rows[:count]
    rows.sort(key=lambda row: row["total_s"], reverse=True)
    return {"by_self": by_self, "by_total": rows[:count]}


def _cell_layers(state):
    """(model, layer, frame key, result) of every cell-check result: model -> layer -> frame."""

    for model_id, layers in sorted((getattr(state, "grid_inspection_payloads_by_model", {}) or {}).items()):
        for layer, frames in sorted((layers or {}).items()):
            for key, result in sorted((frames or {}).items()):
                yield str(model_id), str(layer), str(key), result


def _cell_digests(state) -> dict:
    out: dict[str, dict[str, str]] = {}
    for model_id, layer, key, result in _cell_layers(state):
        out.setdefault(f"{_name(model_id)}/{layer}", {})[key] = _digest(result)
    for pair, frames in sorted((getattr(state, "grid_inspection_payloads_by_pair", {}) or {}).items()):
        for key, result in sorted((frames or {}).items()):
            out.setdefault(f"pair/{_name(pair)}", {})[key] = _digest(result)
    return out


def _cell_summary(state) -> dict:
    summary: dict[str, dict] = {}
    for model_id, layer, _key, result in _cell_layers(state):
        item = summary.setdefault(f"{model_id}/{layer}", {"frames": 0, "broken": 0, "suspicious": 0, "artifact": 0, "missing": 0})
        item["frames"] += 1
        for name in ("broken", "suspicious", "artifact", "missing"):
            item[name] += int(getattr(result, f"{name}_cells", 0) or 0)
    for pair, frames in sorted((getattr(state, "grid_inspection_payloads_by_pair", {}) or {}).items()):
        summary[f"pair/{pair}"] = {"frames": len(frames or {})}
    return summary


def _defect_boxes(state) -> dict:
    """Boxes of every cell that is not normal, per model and frame (binary layer, for the scorecard)."""

    out: dict[str, dict[str, list]] = {}
    for model_id, layer, key, result in _cell_layers(state):
        if layer != "binary":
            continue
        boxes = []
        for cell in getattr(result, "per_cell_results", ()) or ():
            status = str(getattr(cell, "status", "") or "")
            if status and status not in ("normal", "ok"):
                bbox = getattr(cell, "bbox", None)
                if bbox:
                    boxes.append([int(v) for v in bbox[:4]] + [status, list(getattr(cell, "reasons", ()) or ())])
        out.setdefault(_name(model_id), {})[key] = boxes
    return out


def _dirt_payload(presenter) -> dict:
    controller = getattr(presenter, "_dirt_controller", None)
    results = dict(getattr(controller, "_results", {}) or {})
    digests: dict[str, dict[str, str]] = {}
    regions: dict[str, dict[str, list]] = {}
    for model_id, frames in sorted(results.items()):
        for key, result in sorted(frames.items()):
            # Settings and paths are the same in both runs; the regions are what counts.
            digests.setdefault(_name(model_id), {})[key] = _digest((result.width, result.height, result.used_confidence, result.regions, result.error))
            if result.regions:
                regions.setdefault(_name(model_id), {})[key] = [
                    [*map(int, region.bbox), region.level, round(float(region.dirtiness), 4)] for region in result.regions
                ]
    return {"digests": digests, "regions": regions}


def _validation_payload(state) -> dict:
    build = getattr(state, "build_result", None)
    records = tuple(getattr(build, "records", ()) or ())
    scores = {}
    findings = {}
    counts = {}
    digests = {}
    for record in records:
        scores[record.key] = None if not record.score_ready else round(float(record.score), 6)
        summary = getattr(record, "summary", None)
        items = tuple(getattr(summary, "findings", ()) or ()) if summary is not None else ()
        digests[record.key] = _digest((record.score_ready, record.score, record.absolute_score, record.score_percentile, summary))
        if items:
            # Worst first (summaries keep them sorted); a few dozen tell where the frame is bad.
            findings[record.key] = [
                [item.kind, item.detector, item.model_id, round(float(item.severity), 4), list(item.box) if item.box else None]
                for item in items[:50]
            ]
            counts[record.key] = len(items)
    return {
        "metric_key": str(getattr(state, "metric_key", "")),
        "scores": scores,
        "findings": findings,
        "finding_counts": counts,
        "digests": digests,
    }


def run(args) -> dict:
    work_dir = args.work_dir.resolve()
    _prepare_environment(work_dir, args.keep_cache, args.src)
    dataset = _limit_dataset(args.dataset.resolve(), args.limit, work_dir, args.stride, args.offset)

    from PyQt6.QtCore import QSettings
    from PyQt6.QtWidgets import QApplication, QMessageBox

    class _TimedApp(QApplication):
        """Time of every top-level Qt event by receiver class and event type (--qt-events): the
        part of a frozen window spent in Qt itself (painting, layout), which a Python sampler
        does not see."""

        def __init__(self, argv) -> None:
            super().__init__(argv)
            self.depth = 0
            self.timing: dict[str, list[float]] = {}
            self.enabled = False

        def notify(self, receiver, event):  # noqa: N802 - Qt name
            if not self.enabled or self.depth:
                return super().notify(receiver, event)
            self.depth += 1
            started = time.perf_counter()
            try:
                return super().notify(receiver, event)
            finally:
                self.depth -= 1
                spent = time.perf_counter() - started
                if spent > 0.002:
                    try:
                        key = f"{type(receiver).__name__}:{event.type().name}"
                    except Exception:
                        key = "?"
                    entry = self.timing.setdefault(key, [0.0, 0.0, 0])
                    entry[0] += spent
                    entry[1] = max(entry[1], spent)
                    entry[2] += 1

        def take(self) -> list[dict]:
            rows = [
                {"event": key, "total_s": round(total, 2), "max_s": round(longest, 3), "count": count}
                for key, (total, longest, count) in self.timing.items()
            ]
            self.timing = {}
            return sorted(rows, key=lambda row: -row["total_s"])[:30]

    app = QApplication.instance() or (_TimedApp(sys.argv[:1]) if args.qt_events else QApplication(sys.argv[:1]))
    shown: list[str] = []
    for name in ("information", "warning", "critical", "question"):
        setattr(
            QMessageBox,
            name,
            staticmethod(lambda *a, **_k: shown.append(str(a[2]) if len(a) > 2 else "") or QMessageBox.StandardButton.Ok),
        )

    from karakal.app.main_window import KarakalWidget
    from karakal.ui.ui_constants import FOLDER_CELL_CLASS_ROLE, FOLDER_CONFIDENCE_ROLE

    started = time.perf_counter()
    widget = KarakalWidget(settings=QSettings(str(work_dir / "karakal.ini"), QSettings.Format.IniFormat))
    widget.resize(1600, 1000)
    widget.show()
    presenter = widget._presenter
    for layer in args.layers:
        base = dataset / layer / "Весь прогон"
        item = presenter._append_folder_item(base / "Маски", checked=True)
        item.setData(FOLDER_CONFIDENCE_ROLE, str(base / "Confidence"))
        # The ones and zeros layers of one array: the cell check also computes their conflicts.
        item.setData(FOLDER_CELL_CLASS_ROLE, layer)
    presenter._refresh_folder_rows()
    presenter._sync_action_buttons()
    app.processEvents()
    for layer in args.layers:
        folder = dataset / layer / "Весь прогон" / "Маски"
        _LAYER_NAMES[presenter._stable_model_id_for_folders(folder, dataset / layer / "Весь прогон" / "Confidence")] = layer
    for root in {str(dataset), str(dataset).replace("\\", "/"), str(dataset.resolve())}:
        _ROOTS.extend({root.encode("utf-8"), root.replace("\\", "/").encode("utf-8"), root.replace("/", "\\").encode("utf-8")})
    import karakal

    report: dict = {"source": str(Path(karakal.__file__).parent), "dataset": str(dataset), "limit": args.limit, "stride": args.stride, "phases": {}, "messages": shown, "startup_s": round(time.perf_counter() - started, 2)}

    stalls = _Stalls()
    panel = widget.analysis_setup_panel
    dirt = getattr(presenter, "_dirt_controller", None)

    def idle() -> bool:
        return presenter._worker_thread is None and not (dirt is not None and dirt.is_running())

    for phase in args.phases:
        mode = {"cells": "grid_inspection", "validation": "validation"}[phase]
        panel._mode_buttons[mode].click()
        app.processEvents()
        profile = cProfile.Profile() if args.profile else None
        sampler = _UiSampler() if args.sample else None
        if sampler is not None:
            sampler.start()
        stalls.start()
        phase_started = time.perf_counter()
        if profile is not None:
            profile.enable()
        if args.qt_events and hasattr(app, "take"):
            app.take()
            app.enabled = True
        presenter._on_run_button_clicked()
        # A run may chain several workers (index, analysis, dirt): idle for a while means done.
        done_at = None
        while True:
            _wait(app, lambda: False, 0.05)
            if idle():
                if done_at is None:
                    done_at = time.perf_counter()
                elif time.perf_counter() - done_at > 1.5:
                    break
            else:
                done_at = None
            if time.perf_counter() - phase_started > args.timeout:
                report["phases"][phase] = {"error": "timeout"}
                break
        if profile is not None:
            profile.disable()
        sampled = sampler.stop() if sampler is not None else None
        wall = (done_at or time.perf_counter()) - phase_started
        state = presenter._current_tab_state()
        entry = {"wall_s": round(wall, 2), "ui": stalls.stop()}
        if profile is not None:
            entry["ui_thread_top"] = _top_functions(profile)
        if sampled is not None:
            entry["ui_samples"] = sampled
        if args.qt_events and hasattr(app, "take"):
            app.enabled = False
            entry["qt_events"] = app.take()
        if phase == "cells":
            entry["cells_summary"] = _cell_summary(state)
            report["cells"] = {"digests": _cell_digests(state), "defects": _defect_boxes(state)}
            if args.dump_keys:
                wanted = {key.strip() for key in args.dump_keys.split(",") if key.strip()}
                dump = {
                    (_name(model_id), layer, key): _without_root(result)
                    for model_id, layer, key, result in _cell_layers(state)
                    if key in wanted
                }
                with open(str(args.output) + ".cells.pickle", "wb") as handle:
                    pickle.dump(dump, handle)
            report["dirt"] = _dirt_payload(presenter)
            entry["dirt_frames"] = {model: len(items) for model, items in report["dirt"]["regions"].items()}
        else:
            report["validation"] = _validation_payload(state)
            entry["metric_key"] = report["validation"]["metric_key"]
            entry["scored"] = sum(1 for value in report["validation"]["scores"].values() if value is not None)
        report["phases"][phase] = entry
        print(phase, json.dumps({k: v for k, v in entry.items() if k not in ("ui_thread_top", "ui_samples", "qt_events")}, ensure_ascii=False), flush=True)
    widget.close()
    app.processEvents()
    return report


def compare(before_path: Path, after_path: Path) -> int:
    before = json.loads(before_path.read_text(encoding="utf-8"))
    after = json.loads(after_path.read_text(encoding="utf-8"))
    problems = 0
    for section, key in (("cells", "digests"), ("dirt", "digests"), ("validation", "digests")):
        first = (before.get(section) or {}).get(key) or {}
        second = (after.get(section) or {}).get(key) or {}
        if section == "validation":
            first, second = {"all": first}, {"all": second}
        for group in sorted(set(first) | set(second)):
            a, b = first.get(group) or {}, second.get(group) or {}
            differ = [frame for frame in sorted(set(a) | set(b)) if a.get(frame) != b.get(frame)]
            status = "same" if not differ else f"{len(differ)} differ, e.g. {differ[:5]}"
            problems += bool(differ)
            print(f"{section}/{group}: {len(a)} vs {len(b)} frames: {status}")
    for phase in sorted(set(before.get("phases", {})) | set(after.get("phases", {}))):
        a = before["phases"].get(phase, {})
        b = after["phases"].get(phase, {})
        print(
            f"{phase}: wall {a.get('wall_s')} -> {b.get('wall_s')} s; "
            f"UI max stall {a.get('ui', {}).get('max_ms')} -> {b.get('ui', {}).get('max_ms')} ms; "
            f"stalled {a.get('ui', {}).get('stalled_s')} -> {b.get('ui', {}).get('stalled_s')} s"
        )
    return 1 if problems else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", type=Path, nargs="?")
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--phases", default="cells,validation")
    parser.add_argument("--limit", type=int, default=0, help="frames per layer (0 = all)")
    parser.add_argument("--stride", type=int, default=1, help="every N-th frame of each layer")
    parser.add_argument("--offset", type=int, default=0, help="skip the first N frames of each layer")
    parser.add_argument("--layers", default="ones,zeros", help="layers to add: ones, zeros or both")
    parser.add_argument("--profile", action="store_true", help="cProfile (all threads of the window process)")
    parser.add_argument("--sample", action="store_true", help="sample where the UI thread is busy")
    parser.add_argument("--qt-events", action="store_true", help="time Qt events by receiver and type")
    parser.add_argument("--dump-keys", default="", help="comma-separated frame keys whose cell results are pickled beside OUT")
    parser.add_argument("--timeout", type=float, default=4 * 3600)
    parser.add_argument("--work-dir", type=Path, default=Path("D:/WORK/karakal_synthetic/perf_env"))
    parser.add_argument("--keep-cache", action="store_true")
    parser.add_argument("--src", type=Path, help="repository root of the code to run (default: this checkout)")
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("BEFORE", "AFTER"))
    args = parser.parse_args(argv)
    if args.compare:
        return compare(*args.compare)
    if args.dataset is None or args.output is None:
        parser.error("DATASET and OUT.json are needed")
    args.phases = [phase.strip() for phase in args.phases.split(",") if phase.strip()]
    args.layers = [layer.strip() for layer in args.layers.split(",") if layer.strip()]
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print("written", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
