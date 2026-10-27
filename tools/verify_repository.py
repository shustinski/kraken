"""Run test modules in isolated processes and retain every failure as evidence.

Examples::

    python tools/verify_repository.py tests --jobs 2
    python tools/verify_repository.py plugins/krona/tests --timeout 180

Process isolation prevents plugins' Qt/application globals and module patches
from contaminating unrelated modules. This does not turn a failed test into a
skip or certify an untested deployment.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
CHILD = """
import os, sys
if os.name == 'nt':
    import ctypes
    ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002)
    from kraken_core.process_lifetime import bind_child_process_lifetime
    bind_child_process_lifetime()
import pytest
raise SystemExit(pytest.main(sys.argv[1:]))
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--report", type=Path, default=ROOT / ".test_runtime" / "verification")
    args = parser.parse_args()
    if args.jobs < 1 or args.timeout <= 0:
        parser.error("jobs and timeout must be positive")
    files = set()
    for path in args.paths:
        path = path.resolve()
        if path.is_file():
            files.add(path)
        elif path.is_dir():
            files.update(path.rglob("test_*.py"))
            files.update(path.rglob("*_test.py"))
        else:
            parser.error(f"Test path does not exist: {path}")
    files = sorted(files)
    if not files:
        parser.error("No test modules found")
    report = args.report.resolve()
    report.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.setdefault("QT_QPA_PLATFORM", "offscreen")
    paths = [ROOT / "src", *(ROOT / "plugins").glob("*/src"), ROOT]
    environment["PYTHONPATH"] = os.pathsep.join(map(str, paths))
    startup = None
    if os.name == "nt":
        startup = subprocess.STARTUPINFO()
        startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startup.wShowWindow = subprocess.SW_HIDE

    def run(entry):
        index, path = entry
        prefix = f"{index:04d}-{path.stem}"
        started = time.monotonic()
        child_environment = dict(environment)
        cache = report / (prefix + "-cache")
        cache.mkdir(exist_ok=True)
        temporary = cache / "tmp"
        temporary.mkdir(exist_ok=True)
        child_environment["TMP"] = str(temporary)
        child_environment["TEMP"] = str(temporary)
        child_environment["TMPDIR"] = str(temporary)
        child_environment["KARAKAL_CACHE_DIR"] = str(cache / "karakal")
        child_environment["XDG_CACHE_HOME"] = str(cache)
        for variable, folder in (
            ("LOCALAPPDATA", "local"),
            ("APPDATA", "roaming"),
            ("XDG_CONFIG_HOME", "config"),
            ("XDG_STATE_HOME", "state"),
        ):
            scoped = cache / folder
            scoped.mkdir(exist_ok=True)
            child_environment[variable] = str(scoped)
        command = [
            sys.executable,
            "-c",
            CHILD,
            str(path),
            "-q",
            "--tb=short",
            f"--basetemp={cache / 'pytest-temp'}",
            "-o",
            f"cache_dir={cache / 'pytest-cache'}",
            f"--junitxml={report / (prefix + '.xml')}",
        ]
        log = report / (prefix + ".log")
        with log.open("w", encoding="utf-8") as stream:
            try:
                result = subprocess.run(
                    command,
                    cwd=ROOT,
                    env=child_environment,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=args.timeout,
                    startupinfo=startup,
                )
                code = result.returncode
                status = "passed" if code == 0 else "failed"
            except subprocess.TimeoutExpired:
                code, status = None, "timeout"
        return {
            "file": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
            "exit_code": code,
            "status": status,
            "seconds": round(time.monotonic() - started, 3),
            "log": str(log),
            "xml": str(report / (prefix + ".xml")),
        }

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        pending = {pool.submit(run, entry): entry for entry in enumerate(files, 1)}
        for future in concurrent.futures.as_completed(pending):
            result = future.result()
            results.append(result)
            print(f"{len(results)}/{len(files)} {result['status']}: {result['file']}", flush=True)
            (report / "results.json").write_text(
                json.dumps(sorted(results, key=lambda item: item["file"]), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
    return int(any(item["status"] != "passed" for item in results))


if __name__ == "__main__":
    raise SystemExit(main())
