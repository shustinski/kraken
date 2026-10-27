"""Compare HEAD and current readers on identical isolated real-event snapshots."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

from kraken_manager.infrastructure.filesystem import FilesystemEventStore, ProjectFileLock
from kraken_manager.infrastructure.filesystem.layout import FileProjectLayout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--segments", nargs="+", type=int, default=[256, 2048, 8192])
    args = parser.parse_args()
    if not args.segments or min(args.segments) < 1:
        parser.error("segment sizes must be positive")
    source = FileProjectLayout(args.catalog, args.project)
    if not source.events_dir.is_dir():
        raise FileNotFoundError("Existing source project event directory is required")
    output = args.output.resolve()
    if output.is_relative_to(source.catalog_root) or source.catalog_root.is_relative_to(output):
        raise ValueError("Benchmark output must be separate from the source catalog")
    output.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[1]
    baseline_bytes = subprocess.run(
        ["git", "show", "HEAD:src/kraken_manager/infrastructure/filesystem/event_store.py"],
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout
    baseline_path = output / "baseline_event_store.py"
    baseline_path.write_bytes(baseline_bytes)
    name = "kraken_manager.infrastructure.filesystem.audit_baseline_event_store"
    spec = importlib.util.spec_from_file_location(name, baseline_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    cases = []
    with ProjectFileLock(source.lock_path).hold(timeout=20):
        segments = FilesystemEventStore(args.catalog, args.project)._segment_paths()
        for count in sorted(set(min(size, len(segments)) for size in args.segments)):
            catalog = output / str(count)
            clone = FileProjectLayout(catalog, args.project)
            clone.ensure_directories()
            for _, _, path in segments[:count]:
                shutil.copyfile(path, clone.events_dir / path.name)
            cases.append((count, catalog))
    report = dict(
        baseline_revision=subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
        ).stdout.strip(),
        baseline_source_sha256=hashlib.sha256(baseline_bytes).hexdigest(),
        workload="current_revision + exact stream read + idempotency lookup; same immutable source bytes",
        limits=[
            "Read benchmark only; excludes blob copy, SQLite writes, fsync and UI",
            "Both readers share this machine and warmed OS file cache; not an SLA",
        ],
        cases=[],
    )
    for count, catalog in cases:
        current = FilesystemEventStore(catalog, args.project)
        baseline = module.FilesystemEventStore(catalog, args.project)
        events = list(current._iter_stored_unlocked())
        selected = next(event for event in reversed(events) if event.data.get("idempotency_key"))
        stream, key = selected.stream_id, selected.data["idempotency_key"]

        def query(store):
            return (
                store.current_revision(stream),
                tuple(str(e.event_id) for e in store.load_stream(stream)),
                tuple(str(e.event_id) for e in store.find_by_idempotency_key(args.project, key)),
            )

        start = time.perf_counter()
        actual = query(current)
        cold_seconds = time.perf_counter() - start
        assert query(baseline) == actual
        timings = dict(before=[], after=[])
        for _ in range(3):
            for label, reader in (("before", baseline), ("after", current)):
                start = time.perf_counter()
                assert query(reader) == actual
                timings[label].append(time.perf_counter() - start)
        medians = {key: statistics.median(value) for key, value in timings.items()}
        result = dict(
            segments=count,
            events=len(events),
            encoded_bytes=sum(path.stat().st_size for path in current.layout.events_dir.glob("*.jsonl")),
            source_identity=hashlib.sha256("".join(e.event_id for e in events).encode()).hexdigest(),
            current_cold_seconds=cold_seconds,
            medians_seconds=medians,
            samples_seconds=timings,
            speedup=medians["before"] / medians["after"],
        )
        report["cases"].append(result)
        (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
