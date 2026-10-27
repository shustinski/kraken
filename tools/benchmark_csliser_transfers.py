"""Compare legacy and verified transfer engines on identical read-only corpus inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    sys.path[:0] = [str(ROOT / "src"), str(ROOT / "plugins/csliser/src")]
    from csliser.application.use_cases import ExecuteTransferPlan, TransferExecutionOptions
    from csliser.domain.models import FileOperation, ProcessingConfig, SelectionMode, SourceFolder
    from csliser.domain.planner import build_operation_plan
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=32)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.pairs <= 64:
        parser.error("This bounded benchmark accepts 1..64 pairs")
    report = args.report.resolve()
    report.relative_to(ROOT / ".codex_tmp_scratch")
    report.mkdir(exist_ok=False)
    source_dirs = (args.corpus / "jpg_metal", args.corpus / "cif_metal")
    files = [folder / f"{i:04}{extension}" for folder, extension in zip(source_dirs, (".jpg", ".cif")) for i in range(args.pairs)]
    before = {str(p): sha(p) for p in files}
    total_bytes = sum(p.stat().st_size for p in files)
    if shutil.disk_usage(report).free < total_bytes*5 + 256*1024**2:
        raise RuntimeError("Insufficient bounded benchmark workspace")
    legacy_path = "plugins/csliser/src/csliser/application/use_cases.py"
    legacy_source = subprocess.check_output(["git", "show", "HEAD:" + legacy_path], cwd=ROOT)
    if b"shutil.copyfile(item.source, item.destination)" not in legacy_source:
        raise RuntimeError("HEAD does not contain the expected legacy transfer engine; do not substitute a different baseline")
    (report / "legacy-use_cases.py").write_bytes(legacy_source)
    module = types.ModuleType("csliser_legacy_transfer_benchmark")
    sys.modules[module.__name__] = module
    exec(compile(legacy_source, str(report / "legacy-use_cases.py"), "exec"), module.__dict__)
    trials = []
    for trial, engine in enumerate(("legacy", "verified", "verified", "legacy")):
        destination = report / f"trial-{trial}-{engine}"
        plan = build_operation_plan(ProcessingConfig(sources=tuple(SourceFolder(p, (ext,)) for p, ext in zip(source_dirs, (".jpg", ".cif"))),
            frame_expression=f"0-{args.pairs-1}", selection_mode=SelectionMode.FULL_RANGE, operation=FileOperation.COPY, destination=destination))
        assert len(plan.operations) == len(files)
        for operation in plan.operations:
            operation.destination.resolve().relative_to(report)
            assert not operation.destination.exists()
        options = TransferExecutionOptions(max_workers=4)
        executor = ExecuteTransferPlan() if engine == "verified" else module.ExecuteTransferPlan()
        started = time.perf_counter()
        result = executor.execute(plan, options=options)
        elapsed = time.perf_counter()-started
        assert result.ok and result.completed == len(files)
        assert all(sha(op.destination) == before[str(op.source)] for op in plan.operations)
        trials.append({"engine": engine, "seconds": elapsed, "bytes": total_bytes, "files": len(files), "mib_per_second": total_bytes/1024**2/elapsed})
    assert {str(p): sha(p) for p in files} == before
    summary = {engine: statistics.median(t["seconds"] for t in trials if t["engine"] == engine) for engine in ("legacy", "verified")}
    receipt = {"status": "passed", "scope": "transfer execution only; same plans, 4 workers, distinct empty destinations; warm-cache order ABBA, no cache eviction",
        "pairs": args.pairs, "legacy_source_sha256": hashlib.sha256(legacy_source).hexdigest(), "source_files_unchanged": True,
        "training_started": False, "trials": trials, "median_seconds": summary,
        "verified_to_legacy_time_ratio": summary["verified"]/summary["legacy"]}
    (report / "receipt.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
