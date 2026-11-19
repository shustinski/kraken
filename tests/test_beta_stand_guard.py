"""Pure-function tests for the beta-stand reserved database/schema name guard.

``tools/quality/beta_stand.py`` provisions and drops the stand's dedicated
PostgreSQL object (``kraken_beta_test`` by default). One wrong ``-DatabaseName``
must never be able to terminate connections to, or DROP/TRUNCATE-CASCADE, the
working database (``kraken_local``) or the working ``public`` schema, so the
validator runs BEFORE any connection is opened, on both entry points
(``db-provision``, ``db-drop``) and on the standalone ``check-database-name``
subcommand the PowerShell scripts use for their preconditions.

These tests exercise the validator itself - no live database is needed.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tools.quality.beta_stand import validate_stand_name

BASE = 'kraken_local'
START_SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'start-beta-stand.ps1'


@pytest.mark.parametrize('mode', ['database', 'schema'])
@pytest.mark.parametrize('name', [
    'kraken_beta_test',
    'kraken_beta_test_2',
    'beta_stand_w2',
])
def test_dedicated_names_are_accepted(name: str, mode: str) -> None:
    assert validate_stand_name(name, BASE, mode) == ''


@pytest.mark.parametrize('mode', ['database', 'schema'])
@pytest.mark.parametrize(('name', 'fragment'), [
    # The working database itself: the reviewer's exact attack,
    # `-DatabaseName kraken_local`, must be refused.
    ('kraken_local', 'base database'),
    # Mixed-case look-alikes never even reach the reserved-name comparison:
    # only lowercase identifiers are acceptable for the stand's own object.
    ('KRAKEN_LOCAL', 'identifier'),
    ('POSTGRES', 'identifier'),
    # Maintenance/template databases.
    ('postgres', 'reserved'),
    ('template0', 'reserved'),
    ('template1', 'reserved'),
    ('templates_dev', 'reserved'),
    # Not safe PostgreSQL identifiers (the guard also protects the quoted DDL).
    ('', 'identifier'),
    ('kraken-beta', 'identifier'),
    ('kraken beta', 'identifier'),
    ('kraken"; drop table projects;--', 'identifier'),
    ('1kraken', 'identifier'),
])
def test_reserved_and_unsafe_names_are_rejected_in_every_mode(
    name: str, mode: str, fragment: str,
) -> None:
    reason = validate_stand_name(name, BASE, mode)
    assert reason, f'{name!r} must be refused in {mode} mode'
    assert fragment.lower() in reason.lower()


@pytest.mark.parametrize('name', [
    'public',
    'PUBLIC',
    'information_schema',
    'pg_catalog',
    'pg_toast',
    'pg_temp_7',
])
def test_schema_mode_protects_the_schema_namespace(name: str) -> None:
    assert validate_stand_name(name, BASE, 'schema'), \
        f'{name!r} must be refused as a stand schema'


@pytest.mark.parametrize('name', ['public', 'information_schema', 'pg_catalog'])
def test_schema_namespace_names_are_the_schema_mode_extra(name: str) -> None:
    assert validate_stand_name(name, BASE, 'schema')
    # As a *database* name these are not schema-namespace objects - dropping a
    # database called "public" can never cascade into kraken_local's tables.
    # The start script nevertheless pre-checks with the strictest mode (schema)
    # because provisioning may still fall back to schema mode.
    assert validate_stand_name(name, BASE, 'database') == ''


def test_start_script_checks_database_name_before_restart_teardown() -> None:
    """Ordering pin: ``-Restart`` must not tear down before the name guard.

    ``stop-beta-stand.ps1`` drops the stand's database/schema. A refused
    ``-DatabaseName`` (e.g. ``kraken_local``) must fail with ``FAIL
    preconditions`` *before* that teardown is invoked, so the script text
    itself is pinned: the ``check-database-name`` guard must appear before
    the restart teardown invocation.
    """
    text = START_SCRIPT.read_text(encoding='utf-8')
    assert "'check-database-name'" in text
    assert "& (Join-Path $PSScriptRoot 'stop-beta-stand.ps1')" in text
    guard = text.index("'check-database-name'")
    teardown = text.index("& (Join-Path $PSScriptRoot 'stop-beta-stand.ps1')")
    assert guard < teardown, (
        'start-beta-stand.ps1 must run the -DatabaseName check-database-name '
        'guard before the -Restart stop-beta-stand.ps1 teardown'
    )
