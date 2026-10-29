"""Offline native Linux CSliser file-safety contracts, excluding Qt/installers."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "python@sha256:35caba83af2fe6da9a5d922f9b0156f21d1246a48278c58a08bd253303ba42e8"


def inside():
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--find-links=/input/wheels", "--no-cache-dir",
        "--target=/tmp/site", "pytest==9.0.3"], check=True, timeout=60)
    environment = dict(os.environ, PYTHONPATH="/input/source/src:/input/source/plugins/csliser/src:/tmp/site", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    modules = ["test_planner", "test_ranges", "test_use_cases", "test_transfer_safety", "test_removal_recovery"]
    status = subprocess.run([sys.executable, "-m", "pytest", "-q", "--tb=short", "--basetemp=/tmp/csliser-tests", "-o", "cache_dir=/tmp/pytest-cache",
        "--junitxml=/evidence/results.xml", *(f"/input/source/plugins/csliser/tests/unit/{name}.py" for name in modules)],
        env=environment, timeout=90).returncode
    return status


def host(args):
    output = args.report.resolve()
    output.relative_to(ROOT / ".codex_tmp_scratch")
    output.mkdir(exist_ok=False)
    inputs = output / "input"
    inputs.mkdir()
    shutil.copyfile(__file__, inputs / "check.py")
    cache = ROOT / ".codex_tmp_scratch/production-audit-20261026/linux-core-fixed-offline/input/wheelhouse"
    wheels = inputs / "wheels"
    wheels.mkdir()
    for path in cache.glob("*.whl"):
        if path.name.split("-", 1)[0].casefold() in {"pytest", "iniconfig", "packaging", "pluggy", "pygments"}:
            shutil.copyfile(path, wheels / path.name)
    assert any(wheels.glob("pytest-9.0.3-*.whl"))
    sources = [ROOT / "plugins/csliser/__main__.py", *(ROOT / "src/kraken_core").rglob("*.py"), *(ROOT / "plugins/csliser/src").rglob("*.py"), *(ROOT / "plugins/csliser/tests").rglob("*.py")]
    manifest = {}
    for path in sorted(sources):
        relative = path.relative_to(ROOT)
        target = inputs / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        manifest[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output / "source-manifest.json").write_text(json.dumps(manifest, indent=2))
    name = "kraken-csliser-audit-" + uuid4().hex[:12]
    with (output / "container.log").open("w") as log:
        status = subprocess.run(["docker", "run", "--rm", "--pull=never", "--name", name, "--network=none", "--read-only", "--cpus=1", "--memory=512m",
            "--pids-limit=64", "--cap-drop=ALL", "--security-opt=no-new-privileges", "--tmpfs=/tmp:rw,size=256m",
            "--env=PYTHONDONTWRITEBYTECODE=1", "--mount", f"type=bind,source={inputs},target=/input,readonly",
            "--mount", f"type=bind,source={output},target=/evidence", IMAGE, "python", "-u", "/input/check.py", "--inside"],
            stdout=log, stderr=subprocess.STDOUT, timeout=150).returncode
    receipt = {"status": "passed" if status == 0 else "failed", "exit_code": status, "image": IMAGE, "container": name,
        "network": "none", "scope": "native source file operations, no Qt/installer certification", "training_started": False,
        "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)
    return status


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inside", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    raise SystemExit(inside() if args.inside else host(args))
