"""[identity] LDAP configuration: TOML parsing, validation, environment propagation."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from kraken_server.configuration import ServerConfig, protect_secret, write_config

pytest_plugins = ["tests.test_admin_project_workflows"]

LDAP_VARS = (
    "KRAKEN_LDAP_URL",
    "KRAKEN_LDAP_BASE_DN",
    "KRAKEN_LDAP_USER_FILTER",
    "KRAKEN_LDAP_BIND_DN",
    "KRAKEN_LDAP_BIND_SECRET_FILE",
    "KRAKEN_LDAP_START_TLS",
    "KRAKEN_LDAP_ALLOW_INSECURE",
    "KRAKEN_LDAP_CA_FILE",
    "KRAKEN_LDAP_TIMEOUT",
)


def _toml(tmp_path: Path, identity_extra: str) -> Path:
    """A minimal valid server.toml (write_config) plus an [identity] block."""
    path = write_config(
        tmp_path / "server.toml",
        database_url="postgresql+psycopg://unused:unused@127.0.0.1/unused",
        blob_root=tmp_path / "blobs",
        source_root=tmp_path / "source",
        derived_root=tmp_path / "derived",
    ).path
    with path.open("a", encoding="utf-8") as stream:
        stream.write("\n[identity]\n" + identity_extra + "\n")
    return path


def test_load_parses_ldap_section(tmp_path: Path) -> None:
    secret = tmp_path / "ldap-bind.secret"
    protect_secret("bind-password-value", secret)
    ca_file = tmp_path / "internal-ca.pem"
    ca_file.write_bytes(b"certificate")
    path = _toml(
        tmp_path,
        "\n".join(
            (
                'ldap_url = "ldaps://DC.ad.inme:636"',
                'ldap_base_dn = "DC=ad,DC=inme"',
                'ldap_user_filter = "(&(objectClass=user)(sAMAccountName={username}))"',
                'ldap_bind_dn = "CN=kraken,CN=Users,DC=ad,DC=inme"',
                f'ldap_bind_secret = {json.dumps(secret.name)}',
                f'ldap_ca_file = {json.dumps(str(ca_file))}',
                "ldap_start_tls = false",
                "ldap_allow_insecure = false",
                "ldap_timeout = 2.5",
                'lan_cidrs = ["10.1.0.0/16", "192.168.0.0/24"]',
            )
        ),
    )
    config = ServerConfig.load(path)
    assert config.ldap_url == "ldaps://DC.ad.inme:636"
    assert config.ldap_base_dn == "DC=ad,DC=inme"
    assert config.ldap_user_filter == "(&(objectClass=user)(sAMAccountName={username}))"
    assert config.ldap_bind_dn == "CN=kraken,CN=Users,DC=ad,DC=inme"
    assert config.ldap_bind_secret == secret.resolve()
    assert config.ldap_start_tls is False
    assert config.ldap_allow_insecure is False
    assert config.ldap_ca_file == ca_file.resolve()
    assert config.ldap_timeout == 2.5
    assert config.lan_cidrs == ("10.1.0.0/16", "192.168.0.0/24")


def test_load_rejects_incomplete_ldap(tmp_path: Path) -> None:
    path = _toml(
        tmp_path,
        "\n".join(
            (
                'ldap_url = "ldaps://DC.ad.inme:636"',
                'ldap_bind_dn = "CN=kraken,CN=Users,DC=ad,DC=inme"',
            )
        ),
    )
    with pytest.raises(ValueError, match="LDAP identity configuration is incomplete"):
        ServerConfig.load(path)


def test_load_rejects_plain_without_starttls(tmp_path: Path) -> None:
    path = _toml(
        tmp_path,
        "\n".join(
            (
                'ldap_url = "ldap://dc.ad.inme:389"',
                'ldap_base_dn = "DC=ad,DC=inme"',
                'ldap_bind_dn = "CN=kraken,CN=Users,DC=ad,DC=inme"',
            )
        ),
    )
    with pytest.raises(ValueError, match="plain ldap"):
        ServerConfig.load(path)


def test_load_rejects_bad_cidr(tmp_path: Path) -> None:
    path = _toml(tmp_path, 'lan_cidrs = ["10.0.0.0/33"]')
    with pytest.raises(ValueError, match="Invalid lan_cidrs entry: 10.0.0.0/33"):
        ServerConfig.load(path)


def test_apply_to_environment_sets_and_pops(tmp_path: Path) -> None:
    secret = tmp_path / "ldap-bind.secret"
    protect_secret("super-secret-bind-password", secret)
    path = _toml(
        tmp_path,
        "\n".join(
            (
                'ldap_url = "ldaps://DC.ad.inme:636"',
                'ldap_base_dn = "DC=ad,DC=inme"',
                'ldap_bind_dn = "CN=kraken,CN=Users,DC=ad,DC=inme"',
                f'ldap_bind_secret = {json.dumps(secret.name)}',
                "ldap_start_tls = true",
                'lan_cidrs = ["10.0.0.0/8"]',
            )
        ),
    )
    with patch.dict(os.environ, {}, clear=False):
        ServerConfig.load(path).apply_to_environment()
        assert os.environ["KRAKEN_LDAP_URL"] == "ldaps://DC.ad.inme:636"
        assert os.environ["KRAKEN_LDAP_BASE_DN"] == "DC=ad,DC=inme"
        assert os.environ["KRAKEN_LDAP_BIND_DN"] == "CN=kraken,CN=Users,DC=ad,DC=inme"
        assert os.environ["KRAKEN_LDAP_BIND_SECRET_FILE"] == str(secret.resolve())
        assert os.environ["KRAKEN_LDAP_START_TLS"] == "1"
        assert os.environ["KRAKEN_LDAP_ALLOW_INSECURE"] == "0"
        assert os.environ["KRAKEN_LDAP_TIMEOUT"] == "5.0"
        assert json.loads(os.environ["KRAKEN_LAN_CIDRS"]) == ["10.0.0.0/8"]
        assert "KRAKEN_LDAP_USER_FILTER" not in os.environ
        assert "KRAKEN_LDAP_CA_FILE" not in os.environ
        # Security pin: the environment carries the secret file path, never the password.
        assert "super-secret-bind-password" not in os.environ.values()

        for name in LDAP_VARS:
            os.environ[name] = "stale"
        plain = _toml(tmp_path / "plain", "")
        ServerConfig.load(plain).apply_to_environment()
        for name in LDAP_VARS:
            assert name not in os.environ
        assert json.loads(os.environ["KRAKEN_LAN_CIDRS"]) == []


def _composition_env(monkeypatch, tmp_path: Path, *, secret_file: Path | None) -> None:
    """Env for a real postgresql_composition() run on the test schema (database fixture)."""
    monkeypatch.setenv("KRAKEN_BLOB_ROOT", str(tmp_path / "composition-blobs"))
    monkeypatch.setenv("KRAKEN_SOURCE_ROOT", str(tmp_path / "source"))
    monkeypatch.setenv("KRAKEN_DERIVED_ROOT", str(tmp_path / "derived"))
    monkeypatch.setenv("KRAKEN_LDAP_URL", "ldap://DC.ad.inme:389")
    monkeypatch.setenv("KRAKEN_LDAP_BASE_DN", "DC=ad,DC=inme")
    monkeypatch.setenv("KRAKEN_LDAP_BIND_DN", "CN=kraken,CN=Users,DC=ad,DC=inme")
    if secret_file is None:
        monkeypatch.delenv("KRAKEN_LDAP_BIND_SECRET_FILE", raising=False)
    else:
        monkeypatch.setenv("KRAKEN_LDAP_BIND_SECRET_FILE", str(secret_file))


def test_partial_ldap_config_disables_domain_auth_without_failing(database, monkeypatch, tmp_path):
    """Spec §1: URL + bind DN present but the bind secret file absent — no startup
    failure, domain_auth stays None and the login path behaves exactly as today."""
    _composition_env(monkeypatch, tmp_path, secret_file=None)
    from kraken_server.composition import postgresql_composition

    # A direct handler on the composition logger. Note: the database fixture's
    # alembic run executes migrations/env.py, whose fileConfig() call disables
    # every logger that already exists — so re-enable the target logger first;
    # neither pytest's caplog nor a bare handler sees a disabled logger.
    captured: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = captured.append
    logger = logging.getLogger("kraken_server.composition")
    logger.addHandler(handler)
    old_level = logger.level
    old_disabled = logger.disabled
    logger.setLevel(logging.WARNING)
    logger.disabled = False
    try:
        composed = postgresql_composition()
    finally:
        logger.setLevel(old_level)
        logger.disabled = old_disabled
        logger.removeHandler(handler)
    assert composed["domain_auth"] is None
    assert any("domain authentication is disabled" in record.getMessage() for record in captured)


def test_ldap_configured_but_ldap3_missing_fails_startup(database, monkeypatch, tmp_path):
    """Deploy-time failure with a clear message when ldap3 is absent (not a 500 at login)."""
    _composition_env(monkeypatch, tmp_path, secret_file=None)
    import importlib.util

    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        return None if name == "ldap3" else real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    from kraken_server.composition import postgresql_composition

    with pytest.raises(RuntimeError, match="ldap3 package is not installed"):
        postgresql_composition()
