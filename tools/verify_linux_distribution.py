"""Install a supplied Kraken wheel in an isolated Linux container and retain evidence.

The host supplies a source/test snapshot and locked public dependency constraints.
The repository and user containers are never mounted or modified by the test.
This is a headless Linux container gate, not a desktop installer certification.
"""

from __future__ import annotations

import argparse
from email.parser import BytesParser
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
import tomllib
from uuid import uuid4
import zipfile


ROOT = Path(__file__).resolve().parents[1]
IMAGE = "python@sha256:35caba83af2fe6da9a5d922f9b0156f21d1246a48278c58a08bd253303ba42e8"
MODULES = (
    "test_filesystem_lock_recovery",
    "test_filesystem_event_read_cache",
    "test_manager_filesystem_events",
    "test_manager_filesystem_blob",
    "test_storage_contract_filesystem",
    "test_security_native_junctions",
    "test_security_paths_packages",
    "test_manager_domain",
    "test_manager_domain_application",
    "test_agent_execution_safety",
    "test_agent_lease_recovery",
    "test_agent_execution_modes",
    "test_queue_resource_configuration",
    "test_electrical_graph",
    "test_multilayer_registration",
    "test_artifact_lineage",
    "test_plugin_protocol_v1",
    "test_analysis_protocol",
    "test_analysis_run_protocol",
)
CORE_MODULES = tuple(name for name in MODULES if name not in {"test_electrical_graph", "test_multilayer_registration"}) + (
    "test_server_services", "test_server_route_contracts",
)
PROFILES = {
    "full": {"extras": "agent,desktop,server,postgres,reports,packages,registration,electrical", "modules": MODULES},
    "core": {"extras": "agent,server,packages", "modules": CORE_MODULES},
}


def export_cached_wheels(caches, destination):
    """Reuse complete pip HTTP-cache wheel bodies without trusting cache filenames."""
    destination.mkdir()
    manifest = []
    for cache in caches:
        for path in sorted(cache.resolve().rglob("*.body")):
            if not zipfile.is_zipfile(path):
                continue
            with zipfile.ZipFile(path) as archive:
                metadata_names = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
                if len(metadata_names) != 1 or archive.testzip() is not None:
                    continue
                prefix = metadata_names[0].removesuffix("METADATA")
                metadata = BytesParser().parsebytes(archive.read(metadata_names[0]))
                wheel = BytesParser().parsebytes(archive.read(prefix + "WHEEL"))
                tag = wheel.get("Tag", "")
                name = str(metadata["Name"]).replace("-", "_")
                filename = f"{name}-{metadata['Version']}-{tag}.whl"
                if Path(filename).name != filename or any(c in filename for c in "\\/:") or tag.count("-") != 2:
                    raise ValueError("Unsafe or invalid wheel cache metadata")
                target = destination / filename
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                    raise ValueError(f"Conflicting cached wheel: {filename}")
                shutil.copyfile(path, target)
                manifest.append({"filename": filename, "sha256": digest, "cache_body": str(path)})
    return manifest


def run_logged(command, log, **options):
    with Path(log).open("w", encoding="utf-8") as stream:
        return subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, **options).returncode


def inside(args):
    source = Path("/tmp/kraken-linux-audit")
    source.mkdir()
    with tarfile.open("/input/source.tar.gz") as archive:
        archive.extractall(source, filter="data")
    output = Path("/evidence")
    profile = PROFILES[args.profile]
    report = {"status": "running", "platform": sys.platform, "python": sys.version, "image": IMAGE,
              "profile": args.profile, "extras": profile["extras"], "test_modules": profile["modules"],
              "offline": args.offline,
              "scope": "headless core/server contracts; excludes desktop/Qt, numerical algorithms and PostgreSQL"
                       if args.profile == "core" else "headless full dependency profile"}
    started = time.monotonic()
    try:
        wheel = next(Path("/input").glob("*.whl"))
        report["wheel_sha256"] = hashlib.sha256(wheel.read_bytes()).hexdigest()
        extras = profile["extras"]
        installation = run_logged(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--cache-dir",
                str(output / "pip-cache"),
                "--only-binary=:all:",
                "--find-links=/input/wheelhouse",
                *(["--no-index"] if args.offline else ["--timeout=20", "--retries=1"]),
                "--constraint",
                "/input/constraints.txt",
                f"{wheel}[{extras}]",
                "pytest",
                "--report",
                str(output / "pip-installation.json"),
            ],
            output / "installation.log",
            timeout=args.installation_timeout,
        )
        report["installation_exit_code"] = installation
        if installation:
            raise RuntimeError("Linux wheel installation failed; see installation.log")
        dependency_check = run_logged(
            [sys.executable, "-m", "pip", "check"], output / "dependency-check.log", timeout=30,
        )
        report["dependency_check_exit_code"] = dependency_check
        if dependency_check:
            raise RuntimeError("Installed Linux dependency set is inconsistent")
        import kraken_manager

        report["installed_module"] = kraken_manager.__file__
        assert "site-packages" in report["installed_module"]
        report["entrypoints"] = {}
        for entrypoint in ("kraken-agent", "kraken-server"):
            executable = shutil.which(entrypoint)
            if executable is None:
                raise RuntimeError(f"Installed entrypoint missing: {entrypoint}")
            status = run_logged([executable, "--help"], output / f"{entrypoint}-help.log", timeout=30)
            report["entrypoints"][entrypoint] = {"executable": executable, "help_exit_code": status}
            if status:
                raise RuntimeError(f"Installed entrypoint failed: {entrypoint}")
        environment = dict(os.environ, QT_QPA_PLATFORM="offscreen",
                           PYTHONPATH=os.pathsep.join((str(source), str(source / "plugins/krona/src"))),
                           PYTHONDONTWRITEBYTECODE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
        report["krona_source_snapshot"] = "plugins/krona/src"
        subprocess.run(
            [sys.executable, "-m", "pip", "freeze", "--all"], stdout=(output / "installed.txt").open("w"), check=True
        )
        tests = run_logged(
            [
                sys.executable,
                "-m",
                "pytest",
                *(f"tests/{name}.py" for name in profile["modules"]),
                "-o",
                "pythonpath=.",
                "-o",
                "cache_dir=/tmp/kraken-pytest-cache",
                "-q",
                "--tb=short",
                f"--junitxml={output / 'tests.xml'}",
            ],
            output / "tests.log",
            cwd=source,
            env=environment,
            timeout=600,
        )
        report["tests_exit_code"] = tests
        if tests:
            raise RuntimeError("Linux regression tests failed; see tests.log")
        report["status"] = "passed"
        return 0
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["seconds"] = round(time.monotonic() - started, 3)
        (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def host(args):
    report = args.report.resolve()
    report.mkdir(parents=True, exist_ok=False)
    inputs = report / "input"
    inputs.mkdir()
    shutil.copyfile(__file__, inputs / "verify_linux_distribution.py")
    shutil.copyfile(args.wheel, inputs / args.wheel.name)
    cached = export_cached_wheels(args.pip_cache, inputs / "wheelhouse")
    (report / "cached-wheels.json").write_text(json.dumps(cached, indent=2), encoding="utf-8")
    if args.wheelhouse:
        for wheel in args.wheelhouse.resolve().glob("*.whl"):
            target = inputs / "wheelhouse" / wheel.name
            if target.exists() and target.read_bytes() != wheel.read_bytes():
                raise ValueError(f"Conflicting supplied wheel: {wheel.name}")
            shutil.copyfile(wheel, target)
    packages = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))["package"]
    constraints = {}
    for package in packages:
        if "registry" in package.get("source", {}):
            name, version = package["name"], package["version"]
            assert name not in constraints or constraints[name] == version
            constraints[name] = version
    (inputs / "constraints.txt").write_text("\n".join(f"{k}=={v}" for k, v in sorted(constraints.items())) + "\n")
    manifest = {}
    with tarfile.open(inputs / "source.tar.gz", "w:gz") as archive:
        paths = [
            ROOT / "pyproject.toml",
            ROOT / "README.md",
            *(ROOT / "src").rglob("*.py"),
            *(ROOT / "tests").rglob("*.py"),
            *(ROOT / "src").rglob("*.json"),
            *(ROOT / "plugins/krona/src").rglob("*.py"),
        ]
        for path in sorted(paths):
            relative = path.relative_to(ROOT).as_posix()
            archive.add(path, arcname=relative)
            manifest[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    (report / "source-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    name = "kraken-linux-audit-" + uuid4().hex[:12]
    (report / "container.json").write_text(json.dumps({"name": name, "image": IMAGE}), encoding="utf-8")
    return run_logged(
        [
            "docker",
            "run",
            "--rm",
            "--name",
            name,
            *(["--network=none"] if args.offline else []),
            "--mount",
            f"type=bind,source={inputs},target=/input,readonly",
            "--mount",
            f"type=bind,source={report},target=/evidence",
            IMAGE,
            "python",
            "-u",
            "/input/verify_linux_distribution.py",
            "--inside",
            "--profile", args.profile,
            "--installation-timeout", str(args.installation_timeout),
            *(["--offline"] if args.offline else []),
        ],
        report / "container.log",
        timeout=args.installation_timeout + 900,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inside", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--profile", choices=PROFILES, default="full")
    parser.add_argument("--pip-cache", type=Path, action="append", default=[], help="Read-only reuse of complete cached wheels")
    parser.add_argument("--wheelhouse", type=Path, help="Optional directory of previously downloaded wheels")
    parser.add_argument("--offline", action="store_true", help="Disable pip indexes and container networking")
    parser.add_argument("--installation-timeout", type=int, default=2400)
    args = parser.parse_args()
    if args.inside:
        raise SystemExit(inside(args))
    if args.wheel is None or args.report is None:
        parser.error("--wheel and --report are required")
    raise SystemExit(host(args))
