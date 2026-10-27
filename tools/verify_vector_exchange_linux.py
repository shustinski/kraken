"""Windows/Linux exchange round trip in an isolated, resource-bounded container."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import tomllib
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "python@sha256:35caba83af2fe6da9a5d922f9b0156f21d1246a48278c58a08bd253303ba42e8"


def inside():
    started = time.monotonic()
    output = Path("/evidence")
    with (output / "installation.log").open("w") as log:
        result = subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--no-cache-dir",
            "--only-binary=:all:", "--no-deps", "--timeout=20", "--retries=1", "--target=/tmp/site", "-r", "/input/requirements.txt"],
            stdout=log, stderr=subprocess.STDOUT, timeout=120)
    if result.returncode:
        raise RuntimeError("Locked Linux binary dependency installation failed")
    sys.path[:0] = ["/input/source/src", "/tmp/site"]
    import numpy
    import shapely
    from kraken_manager.infrastructure.vector_exchange import VectorExchangeReader, write_vector_exchange
    input_path = Path("/input/windows.jsonl")
    with VectorExchangeReader(input_path) as reader:
        records = reader.blocks()
        def load(column, row):
            record = next(records)
            if column == reader.columns-1 and row == reader.rows-1:
                if next(records, None) is not None or not reader.verified:
                    raise RuntimeError("Input stream was not completely verified")
            return {**{k: reader.header[k] for k in ("job_id", "output_version_id", "output_sha256")},
                "features": record["features"], "conflicts": record["conflicts"], **({"coverage": record["coverage"]} if "coverage" in record else {})}
        receipt = write_vector_exchange(Path("/tmp/linux.jsonl"), reader.header, load)
        assert reader.verified
    source_digest = hashlib.sha256(input_path.read_bytes()).hexdigest()
    output_digest = hashlib.sha256(Path("/tmp/linux.jsonl").read_bytes()).hexdigest()
    assert source_digest == output_digest
    shutil.copyfile("/tmp/linux.jsonl", output / "linux.jsonl")
    receipt.update(status="passed", platform=sys.platform, python=sys.version, numpy=numpy.__version__, shapely=shapely.__version__,
        geos=shapely.geos_version_string, input_sha256=source_digest, output_sha256=output_digest,
        byte_identical=True, training_started=False, seconds=time.monotonic()-started)
    (output / "linux-receipt.json").write_text(json.dumps(receipt, indent=2))


def host(args):
    report = args.report.resolve()
    report.relative_to(ROOT / ".codex_tmp_scratch")
    report.mkdir(exist_ok=False)
    inputs = report / "input"
    inputs.mkdir()
    shutil.copyfile(args.input, inputs / "windows.jsonl")
    shutil.copyfile(__file__, inputs / "check.py")
    manifest = {}
    for package in ("kraken_core", "kraken_manager"):
        for path in sorted((ROOT / "src" / package).rglob("*.py")):
            relative = path.relative_to(ROOT)
            target = inputs / "source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            manifest[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    (report / "source-manifest.json").write_text(json.dumps(manifest, indent=2))
    packages = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))["package"]
    required = {p["name"]: p["version"] for p in packages if p["name"] in {"numpy", "shapely"}}
    assert set(required) == {"numpy", "shapely"}
    (inputs / "requirements.txt").write_text("\n".join(f"{k}=={v}" for k, v in sorted(required.items())) + "\n")
    name = "kraken-vector-exchange-" + uuid4().hex[:12]
    with (report / "container.log").open("w") as log:
        result = subprocess.run(["docker", "run", "--rm", "--pull=never", "--name", name, "--read-only", "--cpus=1", "--memory=768m",
            "--pids-limit=64", "--cap-drop=ALL", "--security-opt=no-new-privileges", "--tmpfs=/tmp:rw,exec,size=512m",
            "--env=PYTHONDONTWRITEBYTECODE=1", "--env=OPENBLAS_NUM_THREADS=1", "--env=OMP_NUM_THREADS=1",
            "--mount", f"type=bind,source={inputs},target=/input,readonly", "--mount", f"type=bind,source={report},target=/evidence",
            IMAGE, "python", "-u", "/input/check.py", "--inside"], stdout=log, stderr=subprocess.STDOUT, timeout=180)
    receipt = {"container": name, "image": IMAGE, "exit_code": result.returncode, "requirements": required,
        "scope": "source snapshot reader and geometry exchange, not a wheel/desktop installer acceptance"}
    if result.returncode == 0:
        sys.path.insert(0, str(ROOT / "src"))
        from kraken_manager.infrastructure.vector_exchange import VectorExchangeReader
        with VectorExchangeReader(report / "linux.jsonl") as reader:
            count = sum(1 for _ in reader.blocks())
            assert reader.verified
        receipt.update(status="passed", windows_readback_verified=True, blocks=count,
            sha256=hashlib.sha256((report / "linux.jsonl").read_bytes()).hexdigest())
    else:
        receipt["status"] = "failed"
    (report / "receipt.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)
    return result.returncode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inside", action="store_true")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--report", type=Path)
    arguments = parser.parse_args()
    if arguments.inside:
        inside()
    else:
        if arguments.input is None or arguments.report is None:
            parser.error("--input and --report are required")
        raise SystemExit(host(arguments))
