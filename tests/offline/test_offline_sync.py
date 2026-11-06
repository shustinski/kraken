"""Behavioral tests for the Windows offline export/import workflow."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SYNC = ROOT / "scripts" / "offline" / "sync.ps1"


def _git_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "GIT_AUTHOR_NAME": "Offline Test",
            "GIT_AUTHOR_EMAIL": "offline@example.com",
            "GIT_COMMITTER_NAME": "Offline Test",
            "GIT_COMMITTER_EMAIL": "offline@example.com",
        }
    )
    return env


def _run(args: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=_git_env(),
    )


def _git(repo: Path, *args: str) -> None:
    completed = _run(["git", "-C", str(repo), *args])
    assert completed.returncode == 0, completed.stderr


def _init_repo(path: Path, *, filename: str = "README.md", content: str = "base\n") -> None:
    path.mkdir(parents=True)
    _run(["git", "init", "-b", "main", str(path)])
    (path / filename).write_text(content, encoding="utf-8")
    _git(path, "add", filename)
    _git(path, "commit", "-m", "base")


def _pyproject(path: Path, dependencies: list[str] | None = None) -> None:
    deps = dependencies or []
    rendered = ", ".join(f'"{item}"' for item in deps)
    (path / "offline_sample.py").write_text("VALUE = 1\n", encoding="utf-8")
    (path / "pyproject.toml").write_text(
        "\n".join(
            [
                "[build-system]",
                'requires = ["setuptools>=69"]',
                'build-backend = "setuptools.build_meta"',
                "",
                "[project]",
                'name = "offline-sample"',
                'version = "0.0.1"',
                'requires-python = ">=3.14"',
                f"dependencies = [{rendered}]",
                "",
                "[tool.setuptools]",
                'py-modules = ["offline_sample"]',
                "",
            ]
        ),
        encoding="utf-8",
    )
    locked = _run(["uv", "lock", "--directory", str(path)])
    assert locked.returncode == 0, locked.stdout + locked.stderr
    _git(path, "add", "pyproject.toml", "uv.lock", "offline_sample.py")
    _git(path, "commit", "-m", "project")


def _sync(mode: str, transfer: Path, repo: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return _run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SYNC),
            "-Mode",
            mode,
            "-TransferPath",
            str(transfer),
            "-RepoRoot",
            str(repo),
            "-ToolMode",
            "Validate",
            *extra,
        ]
    )


def test_export_requires_the_script_to_exist() -> None:
    assert SYNC.is_file()


def test_import_missing_transfer_directory(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    missing = tmp_path / "no-such-transfer"
    completed = _sync("import", missing, repo)
    assert completed.returncode != 0
    assert "OFFLINE_ERROR: missing-transfer" in completed.stdout + completed.stderr


def test_import_invalid_manifest(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    transfer = tmp_path / "transfer"
    transfer.mkdir()
    (transfer / "manifest.json").write_text("{", encoding="utf-8")
    completed = _sync("import", transfer, repo)
    assert completed.returncode != 0
    assert "OFFLINE_ERROR: invalid-manifest" in completed.stdout + completed.stderr


def test_import_rejects_incomplete_manifest(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    transfer = tmp_path / "transfer"
    transfer.mkdir()
    (transfer / "manifest.json").write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
    (transfer / "manifest.sha256").write_text("abc", encoding="utf-8")
    completed = _sync("import", transfer, repo)
    assert completed.returncode != 0
    assert "OFFLINE_ERROR: invalid-manifest" in completed.stdout + completed.stderr


def test_roundtrip_preserves_uncommitted_work_and_supports_repeat_import(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    transfer = tmp_path / "transfer"
    _init_repo(source, content="one\n")
    _init_repo(destination, content="one\n")
    (source / "README.md").write_text("two\n", encoding="utf-8")
    _git(source, "add", "README.md")
    _git(source, "commit", "-m", "two")
    (destination / "README.md").write_text("local edit\n", encoding="utf-8")
    (destination / "notes.txt").write_text("keep me\n", encoding="utf-8")

    exported = _sync("export", transfer, source)
    assert exported.returncode == 0, exported.stdout + exported.stderr
    manifest = json.loads((transfer / "manifest.json").read_text(encoding="utf-8"))
    listed = {item["path"] for item in manifest["files"]}
    assert not any(".venv" in path.split("/") for path in listed)
    assert (transfer / "git" / "repo.bundle").is_file()

    first = _sync("import", transfer, destination, "-ProjectEnvironment", str(tmp_path / "venv"))
    assert first.returncode == 0, first.stdout + first.stderr
    assert (destination / "README.md").read_text(encoding="utf-8") == "local edit\n"
    assert (destination / "notes.txt").read_text(encoding="utf-8") == "keep me\n"
    head = _run(["git", "-C", str(destination), "rev-parse", "HEAD"])
    assert head.stdout.strip() != _run(["git", "-C", str(source), "rev-parse", "HEAD"]).stdout.strip()
    remote = _run(["git", "-C", str(destination), "rev-parse", "refs/remotes/kraken-offline/main"])
    assert remote.returncode == 0
    assert remote.stdout.strip() == _run(["git", "-C", str(source), "rev-parse", "HEAD"]).stdout.strip()
    assert "git merge kraken-offline/main" in first.stdout + first.stderr

    second = _sync("import", transfer, destination, "-ProjectEnvironment", str(tmp_path / "venv"))
    assert second.returncode == 0, second.stdout + second.stderr
    assert (destination / "README.md").read_text(encoding="utf-8") == "local edit\n"


def test_divergent_branches_are_not_reset(tmp_path: Path) -> None:
    source = tmp_path / "source"
    other = tmp_path / "other"
    _init_repo(source, content="base\n")
    shutil.copytree(source, other)
    (source / "from-source.txt").write_text("source\n", encoding="utf-8")
    _git(source, "add", "from-source.txt")
    _git(source, "commit", "-m", "source")
    (other / "from-other.txt").write_text("other\n", encoding="utf-8")
    _git(other, "add", "from-other.txt")
    _git(other, "commit", "-m", "other")
    local_head = _run(["git", "-C", str(other), "rev-parse", "HEAD"]).stdout.strip()

    transfer = tmp_path / "transfer"
    exported = _sync("export", transfer, source)
    assert exported.returncode == 0, exported.stderr
    imported = _sync("import", transfer, other, "-ProjectEnvironment", str(tmp_path / "venv"))
    combined = imported.stdout + imported.stderr
    assert imported.returncode == 0, combined
    assert "OFFLINE_INFO: diverged" in combined
    assert _run(["git", "-C", str(other), "rev-parse", "HEAD"]).stdout.strip() == local_head
    assert (other / "from-other.txt").is_file()
    assert not (other / "from-source.txt").exists()


def test_corrupted_bundle_and_missing_file_fail(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    _init_repo(source)
    _init_repo(destination)
    transfer = tmp_path / "transfer"
    exported = _sync("export", transfer, source)
    assert exported.returncode == 0, exported.stderr

    bundle = transfer / "git" / "repo.bundle"
    data = bytearray(bundle.read_bytes())
    data[min(80, len(data) - 1)] ^= 0xFF
    bundle.write_bytes(data)
    corrupted = _sync("import", transfer, destination, "-ProjectEnvironment", str(tmp_path / "venv-a"))
    assert corrupted.returncode != 0
    assert "OFFLINE_ERROR: checksum" in corrupted.stdout + corrupted.stderr

    exported_again = _sync("export", tmp_path / "transfer-2", source)
    assert exported_again.returncode == 0, exported_again.stderr
    second = tmp_path / "transfer-2"
    manifest = json.loads((second / "manifest.json").read_text(encoding="utf-8"))
    victim = second / manifest["files"][0]["path"]
    victim.unlink()
    interrupted = _sync("import", second, destination, "-ProjectEnvironment", str(tmp_path / "venv-b"))
    assert interrupted.returncode != 0
    assert "OFFLINE_ERROR: missing-file" in interrupted.stdout + interrupted.stderr


def test_incompatible_python_and_platform_are_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    _init_repo(source)
    _init_repo(destination)
    transfer = tmp_path / "transfer"
    exported = _sync("export", transfer, source)
    assert exported.returncode == 0, exported.stderr
    manifest_path = transfer / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["python_requires"] = ">=3.99"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    digest = _run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            f"(Get-FileHash -Algorithm SHA256 -Path '{manifest_path}').Hash.ToLower()",
        ]
    )
    (transfer / "manifest.sha256").write_text(digest.stdout.strip(), encoding="utf-8")
    rejected = _sync("import", transfer, destination, "-ProjectEnvironment", str(tmp_path / "venv"))
    assert rejected.returncode != 0
    assert "OFFLINE_ERROR: python-incompatible" in rejected.stdout + rejected.stderr

    manifest["python_requires"] = ">=3.14"
    manifest["platform"]["architecture"] = "arm64"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    digest = _run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            f"(Get-FileHash -Algorithm SHA256 -Path '{manifest_path}').Hash.ToLower()",
        ]
    )
    (transfer / "manifest.sha256").write_text(digest.stdout.strip(), encoding="utf-8")
    platform = _sync("import", transfer, destination, "-ProjectEnvironment", str(tmp_path / "venv"))
    assert platform.returncode != 0
    assert "OFFLINE_ERROR: platform-incompatible" in platform.stdout + platform.stderr


def test_missing_git_objects_do_not_change_head(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    _init_repo(source, content="base\n")
    _init_repo(destination, content="unrelated\n")
    (source / "README.md").write_text("next\n", encoding="utf-8")
    _git(source, "add", "README.md")
    _git(source, "commit", "-m", "next")
    base = _run(["git", "-C", str(source), "rev-parse", "HEAD~1"])
    tip = _run(["git", "-C", str(source), "rev-parse", "HEAD"])
    assert base.returncode == 0, base.stderr
    assert tip.returncode == 0, tip.stderr
    bundle = tmp_path / "incremental.bundle"
    created = _run(["git", "-C", str(source), "bundle", "create", str(bundle), "HEAD", "--not", "HEAD~1"])
    assert created.returncode == 0, created.stderr
    before = _run(["git", "-C", str(destination), "rev-parse", "HEAD"]).stdout.strip()
    completed = _run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SYNC),
            "-Mode",
            "import-bundle",
            "-TransferPath",
            str(bundle),
            "-RepoRoot",
            str(destination),
        ]
    )
    assert completed.returncode != 0
    assert "OFFLINE_ERROR: missing-git-objects" in completed.stdout + completed.stderr
    assert _run(["git", "-C", str(destination), "rev-parse", "HEAD"]).stdout.strip() == before


def test_submodules_and_missing_lfs_objects_fail_export(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / ".gitmodules").write_text('[submodule "vendor"]\n\tpath = vendor\n\turl = ../vendor\n', encoding="utf-8")
    failed = _sync("export", tmp_path / "transfer", repo)
    assert failed.returncode != 0
    assert "OFFLINE_ERROR: submodules" in failed.stdout + failed.stderr

    lfs_repo = tmp_path / "lfs"
    _init_repo(lfs_repo)
    installed = _run(["git", "-C", str(lfs_repo), "lfs", "install", "--local"])
    tracked = _run(["git", "-C", str(lfs_repo), "lfs", "track", "*.bin"])
    if installed.returncode != 0 or tracked.returncode != 0:
        pytest.skip("git lfs is not available")
    (lfs_repo / "payload.bin").write_bytes(b"lfs-payload")
    _git(lfs_repo, "add", ".gitattributes", "payload.bin")
    _git(lfs_repo, "commit", "-m", "lfs")
    pointer = _run(["git", "-C", str(lfs_repo), "show", "HEAD:payload.bin"])
    assert "git-lfs.github.com" in pointer.stdout, pointer.stdout + pointer.stderr
    (lfs_repo / "payload.bin").write_text(pointer.stdout, encoding="utf-8")
    lfs_dir = lfs_repo / ".git" / "lfs" / "objects"
    if lfs_dir.exists():
        shutil.rmtree(lfs_dir)
    lfs_export = _sync("export", tmp_path / "lfs-transfer", lfs_repo)
    assert lfs_export.returncode != 0
    assert "OFFLINE_ERROR: lfs-missing" in lfs_export.stdout + lfs_export.stderr


def test_paths_with_spaces_and_non_ascii_characters(tmp_path: Path) -> None:
    parent = tmp_path / "папка с пробелом"
    source = parent / "репо"
    destination = parent / "копия"
    transfer = parent / "перенос данных"
    _init_repo(source, filename="заметка.txt", content="данные\n")
    _init_repo(destination, filename="заметка.txt", content="данные\n")
    (source / "заметка.txt").write_text("новые данные\n", encoding="utf-8")
    _git(source, "add", "заметка.txt")
    _git(source, "commit", "-m", "unicode")
    exported = _sync("export", transfer, source)
    assert exported.returncode == 0, exported.stdout + exported.stderr
    imported = _sync("import", transfer, destination, "-ProjectEnvironment", str(parent / "venv files"))
    assert imported.returncode == 0, imported.stdout + imported.stderr
    assert (destination / "заметка.txt").read_text(encoding="utf-8") == "данные\n"


def test_verify_installs_from_transfer_cache_and_empty_cache_fails(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _init_repo(source)
    _pyproject(source, ["packaging==26.1"])
    transfer = tmp_path / "transfer"
    exported = _sync("export", transfer, source)
    assert exported.returncode == 0, exported.stdout + exported.stderr
    assert (transfer / "uv-cache").is_dir()
    environment = tmp_path / "fresh venv"
    verified = _sync("verify", transfer, source, "-ProjectEnvironment", str(environment))
    combined = verified.stdout + verified.stderr
    assert verified.returncode == 0, combined
    assert "OFFLINE_INFO: offline-install-ok" in combined
    assert (environment / "Lib" / "site-packages" / "packaging").is_dir()
    report_path = Path(str(transfer) + ".verify-report.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["network_isolation"]["os_firewall"] is False
    assert report["network_isolation"]["proxy_connections"] == 0
    assert report["cache_dir"] == str((transfer / "uv-cache").resolve())

    empty_cache = tmp_path / "empty-cache"
    empty_cache.mkdir()
    failed = _sync(
        "verify",
        transfer,
        source,
        "-ProjectEnvironment",
        str(tmp_path / "broken-venv"),
        "-CachePath",
        str(empty_cache),
    )
    assert failed.returncode != 0
    assert "OFFLINE_ERROR: offline-install" in failed.stdout + failed.stderr


def test_repeated_export_reuses_dependency_cache(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _init_repo(source)
    _pyproject(source)
    transfer = tmp_path / "transfer"
    first = _sync("export", transfer, source)
    assert first.returncode == 0, first.stderr
    second = _sync("export", transfer, source)
    assert second.returncode == 0, second.stderr
    assert "OFFLINE_INFO: dependencies-reused" in second.stdout + second.stderr


def test_multi_branch_bundle_clones_the_exported_head(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _init_repo(source, content="base\n")
    _git(source, "checkout", "-b", "other")
    (source / "other.txt").write_text("other\n", encoding="utf-8")
    _git(source, "add", "other.txt")
    _git(source, "commit", "-m", "other")
    _git(source, "checkout", "main")
    transfer = tmp_path / "transfer"
    exported = _sync("export", transfer, source)
    assert exported.returncode == 0, exported.stdout + exported.stderr
    clone = tmp_path / "clone"
    cloned = _run(["git", "clone", str(transfer / "git" / "repo.bundle"), str(clone)])
    assert cloned.returncode == 0, cloned.stderr
    assert (clone / "README.md").read_text(encoding="utf-8") == "base\n"
    source_head = _run(["git", "-C", str(source), "rev-parse", "HEAD"]).stdout.strip()
    clone_head = _run(["git", "-C", str(clone), "rev-parse", "HEAD"]).stdout.strip()
    assert clone_head == source_head


def test_script_does_not_invoke_destructive_git_commands() -> None:
    text = (ROOT / "scripts" / "offline" / "KrakenOffline.psm1").read_text(encoding="utf-8").lower()
    assert "reset --hard" not in text
    assert "clean -fd" not in text
    assert "push --force" not in text
    assert "checkout --" not in text
