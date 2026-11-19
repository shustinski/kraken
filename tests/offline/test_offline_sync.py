"""The offline script copies a git bundle and does not reset local work."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

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


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
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


def _init(path: Path, content: str = "base\n") -> None:
    path.mkdir(parents=True)
    _run(["git", "init", "-b", "main", str(path)])
    (path / "README.md").write_text(content, encoding="utf-8")
    _git(path, "add", "README.md")
    _git(path, "commit", "-m", "base")


def _sync(mode: str, transfer: Path, repo: Path) -> subprocess.CompletedProcess[str]:
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
        ]
    )


def test_import_keeps_local_commits(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    _init(source, "one\n")
    _init(destination, "one\n")
    (source / "README.md").write_text("two\n", encoding="utf-8")
    _git(source, "add", "README.md")
    _git(source, "commit", "-m", "two")
    (destination / "README.md").write_text("local edit\n", encoding="utf-8")

    exported = _sync("export", tmp_path / "transfer", source)
    assert exported.returncode == 0, exported.stdout + exported.stderr
    assert (tmp_path / "transfer" / "repo.bundle").is_file()

    imported = _sync("import", tmp_path / "transfer", destination)
    assert imported.returncode == 0, imported.stdout + imported.stderr
    assert (destination / "README.md").read_text(encoding="utf-8") == "local edit\n"
    remote = _run(["git", "-C", str(destination), "rev-parse", "refs/remotes/kraken-offline/main"])
    source_head = _run(["git", "-C", str(source), "rev-parse", "HEAD"])
    assert remote.stdout.strip() == source_head.stdout.strip()
    assert "git merge kraken-offline/<branch>" in imported.stdout + imported.stderr
