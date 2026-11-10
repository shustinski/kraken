"""[identity] LDAP configuration: TOML parsing, validation, environment propagation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from kraken_server.configuration import ServerConfig, protect_secret, write_config

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
