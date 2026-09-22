"""Child processes must end when the controlling console process ends."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest


def test_hidden_child_exits_when_main_process_exits(tmp_path: Path) -> None:
    if os.name != "nt":
        pytest.skip("process job objects are Windows-specific")
    child_pid_path = tmp_path / "child.pid"
    script = tmp_path / "main_lifetime.py"
    script.write_text(
        textwrap.dedent(
            f"""
            import subprocess
            import sys
            from pathlib import Path

            from kraken_core.process_lifetime import bind_child_process_lifetime

            bind_child_process_lifetime()
            child = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(120)"],
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            Path({str(child_pid_path)!r}).write_text(str(child.pid), encoding="ascii")
            """
        ),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")
    completed = subprocess.run(
        [sys.executable, str(script)],
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    child_pid = _read_pid(child_pid_path)
    try:
        _wait_until_dead(child_pid)
    finally:
        _kill_pid(child_pid)


def test_child_exits_when_parent_process_is_killed(tmp_path: Path) -> None:
    if os.name != "nt":
        pytest.skip("console process tree is Windows-specific")
    child_pid_path = tmp_path / "child.pid"
    parent_pid_path = tmp_path / "parent.pid"
    script = tmp_path / "parent_lifetime.py"
    script.write_text(
        textwrap.dedent(
            f"""
            import subprocess
            import sys
            import time
            from pathlib import Path

            from kraken_core.process_lifetime import bind_child_process_lifetime

            bind_child_process_lifetime()
            child = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(120)"],
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            Path({str(child_pid_path)!r}).write_text(str(child.pid), encoding="ascii")
            time.sleep(120)
            """
        ),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")
    launcher = subprocess.Popen(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                f"""
                import subprocess
                import sys
                import time
                from pathlib import Path

                child = subprocess.Popen([sys.executable, {str(script)!r}])
                Path({str(parent_pid_path)!r}).write_text(str(child.pid), encoding="ascii")
                time.sleep(120)
                """
            ),
        ],
        env=env,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    try:
        parent_pid = _read_pid(parent_pid_path)
        child_pid = _read_pid(child_pid_path)
        launcher.kill()
        launcher.wait(timeout=10)
        _wait_until_dead(parent_pid)
        _wait_until_dead(child_pid)
    finally:
        launcher.kill()
        for pid_path in (parent_pid_path, child_pid_path):
            if pid_path.is_file():
                _kill_pid(_read_existing_pid(pid_path))


def _read_pid(path: Path) -> int:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if path.is_file() and path.stat().st_size:
            return int(path.read_text(encoding="ascii"))
        time.sleep(0.05)
    raise AssertionError(f"pid file was not written: {path}")


def _read_existing_pid(path: Path) -> int:
    try:
        return int(path.read_text(encoding="ascii"))
    except (OSError, ValueError):
        return 0


def _wait_until_dead(pid: int) -> None:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if not _is_alive(pid):
            return
        time.sleep(0.05)
    raise AssertionError(f"process {pid} was still running")


def _is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    result = subprocess.run(
        ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
        check=False,
        capture_output=True,
        text=True,
    )
    return str(pid) in result.stdout


def _kill_pid(pid: int) -> None:
    if pid > 0 and _is_alive(pid):
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], check=False, capture_output=True)
