"""D23 regression pins: ``migrations/env.py`` logger-configuration gate.

``migrations/env.py`` configures logging as::

    if config.config_file_name is not None and config.attributes.get("configure_logger", True):
        fileConfig(config.config_file_name)

The gate is the whole D23 mechanism: the alembic CLI (which passes ``alembic.ini``
and expects its logging sections) keeps working, while programmatic embedders
(pytest's ``database`` fixture, ``kraken_server.run_migrations`` callers) opt out
with ``Config(..., attributes={"configure_logger": False})`` so ``fileConfig``
never replaces their root-logger handlers (pytest capture, the qt-harness ring
buffer — see defect D23).

These tests execute ``env.py`` for real but in a throwaway interpreter with a
stubbed ``alembic.context`` in offline mode: no database connection is opened,
no migration script runs, and the pytest process' own logging is never touched.
A marker handler installed on the probe's root logger before ``env.py`` runs is
the observation instrument: retained means ``fileConfig`` did not run, removed
means it did (``fileConfig`` clears every existing root handler).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

kraken_defect = "D23"

REPO_ROOT = Path(__file__).resolve().parents[1]

# Runs migrations/env.py against a stubbed alembic.context (offline mode, no DB)
# with a marker handler pre-installed on the root logger, and prints what the
# caller's logging survived. ``mode`` selects how the caller constructs Config.
_PROBE = r'''
import contextlib
import json
import logging
import runpy
import sys
import types

from alembic.config import Config

repo = sys.argv[1]
mode = sys.argv[2]
ini = repo + "/alembic.ini"
if mode == "opt-out":
    cfg = Config(ini, attributes={"configure_logger": False})
elif mode == "explicit-on":
    cfg = Config(ini, attributes={"configure_logger": True})
elif mode == "no-attributes":
    cfg = Config(ini)
else:
    raise SystemExit("unknown mode: " + mode)

marker = logging.NullHandler()
logging.getLogger().addHandler(marker)

calls = []
fake = types.ModuleType("alembic.context")
fake.config = cfg
fake.is_offline_mode = lambda: True
fake.configure = lambda **kwargs: calls.append("configure")
fake.begin_transaction = contextlib.nullcontext
fake.run_migrations = lambda: calls.append("run_migrations")
sys.modules["alembic.context"] = fake
import alembic
alembic.context = fake

runpy.run_path(repo + "/migrations/env.py", run_name="__alembic_env_probe__")
root = logging.getLogger()
print(json.dumps({
    "marker_retained": any(handler is marker for handler in root.handlers),
    "calls": calls,
}))
'''


def _run_env_probe(tmp_path: Path, mode: str) -> dict:
    """Execute env.py in a disposable interpreter and return the probe report."""
    script = tmp_path / "env_logger_probe.py"
    script.write_text(_PROBE, encoding="utf-8")
    env = {**os.environ, "KRAKEN_DATABASE_URL": "sqlite://"}
    result = subprocess.run(
        [sys.executable, str(script), REPO_ROOT.as_posix(), mode],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    assert result.returncode == 0, (
        f"env.py probe ({mode}) failed with {result.returncode}:\n{result.stdout}\n{result.stderr}"
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_opt_out_preserves_caller_logging_and_runs_migrations(tmp_path):
    """The D23 fix contract: ``configure_logger: False`` skips fileConfig.

    A programmatic embedder's root-logger handler survives and ``env.py``
    still reaches ``run_migrations`` (the opt-out gates logging setup only,
    not the migration run).
    """
    report = _run_env_probe(tmp_path, "opt-out")
    assert report["marker_retained"] is True, (
        "fileConfig ran despite attributes={'configure_logger': False}: "
        "an embedder's root handlers would be wiped (D23 regression)"
    )
    assert report["calls"] == ["configure", "run_migrations"]


def test_default_config_preserves_cli_logger_setup(tmp_path):
    """CLI behaviour is preserved: a plain ``Config("alembic.ini")`` (no
    attributes) still applies fileConfig — the alembic command line keeps its
    historic logging configuration. This is also the sharp edge programmatic
    callers must avoid: without the opt-out the caller's root handlers are
    replaced (the D23 failure mode, by design for the CLI)."""
    report = _run_env_probe(tmp_path, "no-attributes")
    assert report["marker_retained"] is False, (
        "fileConfig no longer runs for a default Config: alembic CLI logging "
        "behaviour changed"
    )
    assert report["calls"] == ["configure", "run_migrations"]


def test_explicit_configure_logger_true_still_clobbers(tmp_path):
    """The gate defaults to True and an explicit True behaves identically:
    passing an ``alembic.ini`` path always configures logging unless the
    caller explicitly opts out. Embedders must pass the opt-out themselves."""
    report = _run_env_probe(tmp_path, "explicit-on")
    assert report["marker_retained"] is False, (
        "attributes={'configure_logger': True} must behave like the CLI default"
    )
    assert report["calls"] == ["configure", "run_migrations"]
