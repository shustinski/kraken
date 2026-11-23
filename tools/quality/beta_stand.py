"""CLI helper for the permanent beta stand (scripts/start-beta-stand.ps1).

Windows PowerShell 5.1 cannot drive DPAPI secret files, TOML or the Kraken
HTTP contracts by itself (see the PowerShell notes in
scripts/quality-gates.ps1: inline ``python -c`` mangles quotes, so the logic
lives in tools/quality modules). Secret VALUES are never printed: stdout
carries statuses, identifiers and masked URLs only; tokens and database URLs
travel through DPAPI secret files under ``.test_runtime\\beta-stand``.

Subcommands (each is idempotent; the start script owns the ordering):

* ``render-server-config``  build the stand's server.toml from
  ``config/local/server.toml`` with a dedicated database secret name and an
  optional Blob Gateway executable.
* ``check-database-name``   offline guard used by both PowerShell scripts:
  refuse reserved/working database and schema names BEFORE any DDL.
* ``db-provision``          create (or reset) the dedicated PostgreSQL
  database and write its DPAPI URL secret. ``--mode schema`` is the
  documented fallback when the app role lacks CREATEDB (separate SCHEMA in
  the base database instead of a separate DATABASE).
* ``db-drop``               teardown counterpart of ``db-provision``.
* ``protect``               stdin -> DPAPI secret file.
* ``login``                 register-or-sign-in a local beta account via
  POST /api/v1/auth/accounts (loopback is never LAN-gated) and protect the
  session token.
* ``ensure-project``        create the protected sentinel project once and
  reuse it afterwards.
* ``probe-machine-token``   POST /api/v1/agent/lease with the machine token
  and report the HTTP status (teardown expects 401 after revoke).
* ``env-json``              write the caller environment document
  (KRAKEN_TEST_POSTGRES_URL + KRAKEN_LIVE_WORKER_CONFIG).
* ``health``                plain GET health probe for URL arguments.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import SQLAlchemyError

from kraken_server.configuration import protect_secret, unprotect_secret

_IDENTIFIER = re.compile(r'^[a-z_][a-z0-9_$]*$')

# Reserved database names: the stand may never terminate connections to or
# DROP these (the working base URL database is added per invocation).
_RESERVED_DATABASES = frozenset({'postgres'})
# Reserved schema names for the schema-fallback mode; ``pg_`` prefixes cover
# pg_catalog/pg_toast/pg_temp_*.
_RESERVED_SCHEMAS = frozenset({'public', 'information_schema'})


def validate_stand_name(name: str, base_database: str, mode: str) -> str:
    """Return '' when ``name`` is a safe dedicated stand object, else a reason.

    Runs BEFORE any database connection: one wrong ``-DatabaseName`` must
    never reach ``DROP DATABASE``/``DROP SCHEMA ... CASCADE`` on the working
    database (``kraken_local``) or the working ``public`` schema. ``mode`` is
    'database' or 'schema'; schema mode additionally protects the schema
    namespace (public, information_schema, pg_*).
    """
    if not _IDENTIFIER.match(name or ''):
        return f"unsafe PostgreSQL identifier: {name!r}"
    lowered = name.lower()
    if base_database and lowered == base_database.lower():
        return f"refuses to reuse the base database {base_database!r}; the stand needs a dedicated name"
    if lowered in _RESERVED_DATABASES or lowered.startswith('template'):
        return f"{name!r} is a reserved PostgreSQL database"
    if mode == 'schema':
        if lowered in _RESERVED_SCHEMAS:
            return f"{name!r} is a reserved schema and must not be dropped or reset"
        if lowered.startswith('pg_'):
            return f"schema names beginning with pg_ are reserved ({name!r})"
    return ''


def _check_database_name(args: argparse.Namespace) -> int:
    # Precondition probe for the PowerShell scripts: validates the name
    # against the base URL's database WITHOUT connecting anywhere.
    base_database = make_url(unprotect_secret(Path(args.source_secret))).database or 'postgres'
    reason = validate_stand_name(args.database, base_database, args.mode)
    if reason:
        print(json.dumps({'ok': False, 'database': args.database, 'mode': args.mode, 'reason': reason}))
        return 1
    print(json.dumps({'ok': True, 'database': args.database, 'mode': args.mode}))
    return 0


def _quoted(identifier: str) -> str:
    if not _IDENTIFIER.match(identifier):
        raise SystemExit(f"Unsafe PostgreSQL identifier: {identifier!r}")
    return f'"{identifier}"'


def _maintenance_engine(url_text: str, database: str):
    url = make_url(url_text)
    if database:
        url = url.set(database=database)
    return create_engine(url, isolation_level='AUTOCOMMIT', pool_pre_ping=True)


def _terminate_connections(engine, database: str) -> None:
    with engine.connect() as connection:
        connection.execute(
            text('select pg_terminate_backend(pid) from pg_stat_activity '
                 "where datname = :name and pid <> pg_backend_pid()"),
            {'name': database},
        )


def _db_provision(args: argparse.Namespace) -> int:
    source = unprotect_secret(Path(args.source_secret))
    base_database = make_url(source).database or 'postgres'
    # The guard precedes every connection/DDL statement (see validate_stand_name).
    if reason := validate_stand_name(args.database, base_database, args.mode):
        raise SystemExit(f'db-provision: {reason}')
    target_url = make_url(source).set(database=args.database)
    if args.mode == 'schema':
        # Documented fallback: a dedicated SCHEMA inside the base database
        # (used only when the role cannot CREATE DATABASE). The base schema
        # data of the working database stays untouched.
        schema = _quoted(args.database)
        with _maintenance_engine(source, base_database).connect() as connection:
            exists = connection.execute(
                text('select 1 from information_schema.schemata where schema_name = :name'),
                {'name': args.database},
            ).scalar()
            if exists and args.reset:
                connection.execute(text(f'drop schema {schema} cascade'))
                exists = None
            if not exists:
                connection.execute(text(f'create schema {schema}'))
        schema_url = target_url.set(database=base_database).set(
            query={**target_url.query, 'options': f'-csearch_path={args.database}'})
        # str(URL) would mask the password as ***; render the real one.
        protect_secret(schema_url.render_as_string(hide_password=False), Path(args.destination_secret))
        print(json.dumps({'mode': 'schema', 'database': base_database,
                          'schema': args.database,
                          'status': 'reset' if args.reset and exists else ('reused' if exists else 'created')}))
        return 0
    engine = _maintenance_engine(source, 'postgres')
    try:
        with engine.connect() as connection:
            exists = connection.execute(
                text('select 1 from pg_database where datname = :name'), {'name': args.database}).scalar()
            status = 'reused'
            if exists and args.reset:
                _terminate_connections(engine, args.database)
                connection.execute(text(f'drop database {_quoted(args.database)}'))
                exists = None
                status = 'reset'
            if not exists:
                connection.execute(text(f'create database {_quoted(args.database)}'))
                status = 'created' if status != 'reset' else status
    except SQLAlchemyError as error:
        raise SystemExit(f'db-provision: {str(error).splitlines()[0][:300]}') from error
    protect_secret(target_url.render_as_string(hide_password=False), Path(args.destination_secret))
    print(json.dumps({'mode': 'database', 'database': args.database, 'status': status}))
    return 0


def _db_drop(args: argparse.Namespace) -> int:
    source = unprotect_secret(Path(args.source_secret))
    base_database = make_url(source).database or 'postgres'
    if args.schema:
        # Same guard as provisioning: teardown must never drop a reserved name.
        if reason := validate_stand_name(args.schema, base_database, 'schema'):
            raise SystemExit(f'db-drop: {reason}')
        with _maintenance_engine(source, base_database).connect() as connection:
            connection.execute(text(f'drop schema if exists {_quoted(args.schema)} cascade'))
        print(json.dumps({'mode': 'schema', 'database': base_database,
                          'schema': args.schema, 'status': 'dropped'}))
        return 0
    if reason := validate_stand_name(args.database, base_database, 'database'):
        raise SystemExit(f'db-drop: {reason}')
    engine = _maintenance_engine(source, 'postgres')
    with engine.connect() as connection:
        _terminate_connections(engine, args.database)
        connection.execute(text(f'drop database if exists {_quoted(args.database)}'))
    print(json.dumps({'mode': 'database', 'database': args.database, 'status': 'dropped'}))
    return 0


def _render_server_config(args: argparse.Namespace) -> int:
    base_path = Path(args.base).resolve(strict=True)
    with base_path.open('rb') as stream:
        base = tomllib.load(stream)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gateway_enabled = args.gateway_executable.lower() != 'none'
    lines: list[str] = [
        '# Generated by tools/quality/beta_stand.py - do not edit by hand.',
        '# Beta stand server config derived from ' + str(base_path),
        '# (working config/local/server.toml is never modified).',
        '',
        '[server]',
        f'host = {json.dumps(str(base.get("server", {}).get("host", "127.0.0.1")))}',
        f'port = {int(base.get("server", {}).get("port", 8080))}',
        f'project_access_mode = {json.dumps(str(base.get("server", {}).get("project_access_mode", "acl")))}',
        f'log_level = {json.dumps(str(base.get("server", {}).get("log_level", "low")))}',
        '',
        '[database]',
        f'url_secret = {json.dumps(args.database_url_secret_name)}',
        f'pool_size = {int(base.get("database", {}).get("pool_size", 10))}',
        f'max_overflow = {int(base.get("database", {}).get("max_overflow", 10))}',
        '',
        '[storage]',
    ]
    storage = base.get('storage', {})
    for key in ('blob_root', 'source_root', 'derived_root'):
        lines.append(f'{key} = {json.dumps(str(storage.get(key)))}')
    lines.append('')
    lines.append('[blob_gateway]')
    lines.append(f'enabled = {"true" if gateway_enabled else "false"}')
    if gateway_enabled:
        gateway = base.get('blob_gateway', {})
        lines.append(f'public_url = {json.dumps(str(gateway.get("public_url", "http://127.0.0.1:8081")))}')
        lines.append(f'bind = {json.dumps(str(gateway.get("bind", "127.0.0.1:8081")))}')
        lines.append(f'executable = {json.dumps(str(Path(args.gateway_executable).resolve()))}')
        lines.append(f'secret = {json.dumps(args.gateway_secret_name)}')
        lines.append(f'ticket_lifetime_seconds = {int(gateway.get("ticket_lifetime_seconds", 900))}')
    identity = base.get('identity', {})
    if identity:
        lines.append('')
        lines.append('[identity]')
        for key in ('ldap_url', 'ldap_base_dn', 'ldap_bind_dn', 'ldap_bind_secret'):
            if key in identity:
                lines.append(f'{key} = {json.dumps(str(identity.get(key)))}')
        lines.append(f'ldap_allow_insecure = {"true" if identity.get("ldap_allow_insecure") else "false"}')
        lines.append(f'ldap_timeout = {float(identity.get("ldap_timeout", 5.0))}')
        lines.append(f'lan_cidrs = {json.dumps(list(identity.get("lan_cidrs", [])))}')
    out_path.write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
    # Copy the DPAPI secret files that the rendered config references by
    # name (machine-scoped DPAPI blobs decrypt for any local process).
    for name in (args.gateway_secret_name, str(identity.get('ldap_bind_secret') or '')):
        if not name:
            continue
        source = base_path.parent / name
        if source.is_file():
            (out_path.parent / name).write_bytes(source.read_bytes())
    print(json.dumps({'rendered': str(out_path.resolve()), 'gateway_enabled': gateway_enabled}))
    return 0


def _protect(args: argparse.Namespace) -> int:
    value = sys.stdin.buffer.read().decode('utf-8').lstrip('\ufeff').rstrip('\r\n')
    if not value:
        raise SystemExit('protect: empty input')
    protect_secret(value, Path(args.destination))
    print(json.dumps({'protected': str(Path(args.destination).resolve())}))
    return 0


def _remote():
    from kraken_hub.infrastructure.remote.client import RemoteHttpClient

    return RemoteHttpClient


def _login(args: argparse.Namespace) -> int:
    http = _remote()(args.server_url)
    token: str | None = None
    payload: dict[str, Any] = {}
    from kraken_hub.infrastructure.remote.parts.extract_review_archive import RemoteServerError

    try:
        payload = http.request('POST', '/api/v1/auth/accounts', token='',
                               payload={'username': args.username, 'password': args.password,
                                        'display_name': args.display_name})
    except RemoteServerError as error:
        # 400/403: registration refused; 409: the username is already taken
        # (e.g. occupied by a non-local principal) - all fall through to the
        # ordinary session login below.
        if error.status not in {400, 403, 409}:
            raise
    if not payload:
        payload = http.request('POST', '/api/v1/auth/sessions', token='',
                               payload={'username': args.username, 'password': args.password})
    token = str(payload.get('access_token', ''))
    if not token:
        raise SystemExit('login: server returned no access_token')
    protect_secret(token, Path(args.destination_secret))
    print(json.dumps({'provider': payload.get('provider'),
                      'principal_id': (payload.get('principal') or {}).get('id'),
                      'expires_at': payload.get('expires_at')}))
    return 0


def _ensure_project(args: argparse.Namespace) -> int:
    from kraken_hub.application.workspace import REMOTE_STORAGE_PROFILE
    from kraken_hub.infrastructure.remote.parts.extract_review_archive import RemoteServerError

    http = _remote()(args.server_url)
    token = unprotect_secret(Path(args.token_secret))

    def listed_id() -> str:
        listing = http.request('GET', '/api/v1/projects?include_archived=true', token=token)
        for item in listing.get('items', []):
            if str(item.get('name')) == args.name:
                return str(item.get('project_id'))
        return ''

    if args.project_id:
        try:
            project = http.request('GET', f"/api/v1/projects/{args.project_id}", token=token)
            if project:
                print(json.dumps({'project_id': str(project.get('project_id') or args.project_id),
                                  'status': 'reused'}))
                return 0
        except RemoteServerError as error:
            if error.status not in (403, 404):
                raise
            if error.status == 403:
                # A re-created beta account cannot access the old principal's
                # sentinel. Keep it untouched and create a new isolated one.
                args.name = f"{args.name}-{uuid.uuid4().hex[:8]}"
    known = listed_id()
    if known:
        print(json.dumps({'project_id': known, 'status': 'reused'}))
        return 0
    try:
        payload = http.request(
            'POST', '/api/v1/projects', token=token,
            payload={'name': args.name, 'width': args.width, 'height': args.height,
                     'orientation': 'y_down', 'storage_profile_id': REMOTE_STORAGE_PROFILE.id,
                     'layer_template': False},
            headers={'Idempotency-Key': str(uuid.uuid4())})
    except RemoteServerError as error:
        # 422 "project name already taken": a previous run created it while
        # it was invisible to this (possibly stale) server session.
        if error.status != 422 or not (known := listed_id()):
            raise
        print(json.dumps({'project_id': known, 'status': 'recovered'}))
        return 0
    print(json.dumps({'project_id': str(payload.get('project_id') or payload.get('id')),
                      'status': 'created'}))
    return 0


def _probe_machine_token(args: argparse.Namespace) -> int:
    from kraken_hub.infrastructure.remote.parts.extract_review_archive import RemoteServerError

    http = _remote()(args.server_url)
    token = unprotect_secret(Path(args.token_secret))
    status: int
    try:
        http.request('POST', '/api/v1/agent/lease', token=token,
                     payload={'capabilities': [args.capability]})
        status = 200
    except RemoteServerError as error:
        status = int(error.status or 0)
    print(json.dumps({'status': status, 'expected': args.expect}))
    return 0 if status == args.expect else 1


def _unprotect(args: argparse.Namespace) -> int:
    # Used only to feed a secret into a child process environment (the agent
    # worker reads KRAKEN_AGENT_TOKEN); never call from an interactive shell.
    sys.stdout.write(unprotect_secret(Path(args.source)))
    return 0


def _env_json(args: argparse.Namespace) -> int:
    # The plaintext URL goes to the git-ignored env json only, never stdout.
    document = {
        'KRAKEN_TEST_POSTGRES_URL': unprotect_secret(Path(args.database_url_secret)),
        'KRAKEN_LIVE_WORKER_CONFIG': str(Path(args.live_config).resolve()),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({'wrote': str(out.resolve())}))
    return 0


def _health(args: argparse.Namespace) -> int:
    try:
        with urllib.request.urlopen(args.url, timeout=args.timeout) as response:  # noqa: S310
            body = response.read().decode('utf-8', errors='replace')
            print(json.dumps({'status': response.status, 'body': body[:200]}))
            return 0 if response.status == 200 else 1
    except urllib.error.HTTPError as error:
        print(json.dumps({'status': error.code, 'body': ''}))
        return 1
    except OSError as error:
        print(json.dumps({'status': 0, 'body': str(error.reason)}))
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest='command', required=True)

    sub = commands.add_parser('render-server-config')
    sub.add_argument('--base', required=True)
    sub.add_argument('--out', required=True)
    sub.add_argument('--database-url-secret-name', required=True)
    sub.add_argument('--gateway-executable', required=True, help='absolute path, or "none"')
    sub.add_argument('--gateway-secret-name', default='blob-gateway.secret')
    sub.set_defaults(handler=_render_server_config)

    sub = commands.add_parser('check-database-name')
    sub.add_argument('--source-secret', required=True)
    sub.add_argument('--database', required=True)
    sub.add_argument('--mode', choices=('database', 'schema'), default='schema')
    sub.set_defaults(handler=_check_database_name)

    sub = commands.add_parser('db-provision')
    sub.add_argument('--source-secret', required=True)
    sub.add_argument('--destination-secret', required=True)
    sub.add_argument('--database', required=True)
    sub.add_argument('--reset', action='store_true')
    sub.add_argument('--mode', choices=('database', 'schema'), default='database')
    sub.set_defaults(handler=_db_provision)

    sub = commands.add_parser('db-drop')
    sub.add_argument('--source-secret', required=True)
    sub.add_argument('--database', required=True)
    sub.add_argument('--schema', default='')
    sub.set_defaults(handler=_db_drop)

    sub = commands.add_parser('protect')
    sub.add_argument('--destination', required=True)
    sub.set_defaults(handler=_protect)

    sub = commands.add_parser('login')
    sub.add_argument('--server-url', required=True)
    sub.add_argument('--username', required=True)
    sub.add_argument('--password', required=True)
    sub.add_argument('--display-name', default='Beta Stand')
    sub.add_argument('--destination-secret', required=True)
    sub.set_defaults(handler=_login)

    sub = commands.add_parser('ensure-project')
    sub.add_argument('--server-url', required=True)
    sub.add_argument('--token-secret', required=True)
    sub.add_argument('--name', required=True)
    sub.add_argument('--project-id', default='')
    sub.add_argument('--width', type=int, default=2)
    sub.add_argument('--height', type=int, default=1)
    sub.set_defaults(handler=_ensure_project)

    sub = commands.add_parser('probe-machine-token')
    sub.add_argument('--server-url', required=True)
    sub.add_argument('--token-secret', required=True)
    sub.add_argument('--capability', default='frames.vectorize.v1')
    sub.add_argument('--expect', type=int, default=401)
    sub.set_defaults(handler=_probe_machine_token)

    sub = commands.add_parser('unprotect')
    sub.add_argument('--source', required=True)
    sub.set_defaults(handler=_unprotect)

    sub = commands.add_parser('env-json')
    sub.add_argument('--database-url-secret', required=True)
    sub.add_argument('--live-config', required=True)
    sub.add_argument('--out', required=True)
    sub.set_defaults(handler=_env_json)

    sub = commands.add_parser('health')
    sub.add_argument('--url', required=True)
    sub.add_argument('--timeout', type=float, default=2.0)
    sub.set_defaults(handler=_health)

    args = parser.parse_args()
    return args.handler(args)


if __name__ == '__main__':
    raise SystemExit(main())
