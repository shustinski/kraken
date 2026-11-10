"""Direct LDAP (AD) sign-in: service-account search plus a user rebind, at login time only."""

from __future__ import annotations

import ssl
from dataclasses import dataclass
from typing import Any, Protocol

DEFAULT_USER_FILTER = "(&(objectClass=user)(objectCategory=person)(sAMAccountName={username}))"


class LdapUnavailable(RuntimeError):
    """The LDAP server or a search could not be used; not a credentials verdict."""


class LdapAuthenticationError(RuntimeError):
    """The LDAP server rejected the service-account bind."""


@dataclass(frozen=True, slots=True)
class LdapIdentity:
    username: str
    display_name: str
    email: str | None
    avatar_url: str | None


class LdapTransport(Protocol):
    """Transport seam so the client stays testable without ldap3 or a network.

    - ``connect``: service-account bind; ``LdapAuthenticationError`` on reject,
      ``LdapUnavailable`` on network/TLS failure.
    - ``search_user``: ``(dn, attrs)`` for exactly one match, ``None`` for no
      match, ``LdapUnavailable`` if ambiguous or on error.
    - ``user_bind``: rebind as the found user; ``True`` only on success.
    - ``close``: release the connection; must not raise (called from ``finally``).
    """

    def connect(self) -> None: ...

    def search_user(self, search_filter: str) -> tuple[str, dict[str, Any]] | None: ...

    def user_bind(self, dn: str, password: str) -> bool: ...

    def close(self) -> None: ...


def _escape(value: str) -> str:
    """Escape an LDAP filter value per RFC 4515 (same set as ldap3, plus '&' as \\26).

    Kept pure Python so the client and its unit tests do not require the
    ``server`` extra; the real transport passes the resulting filter through
    with ``auto_escape=False`` so escaping happens exactly once, here.
    """
    escaped = value.replace("\\", "\\5c")
    escaped = escaped.replace("*", "\\2a")
    escaped = escaped.replace("(", "\\28")
    escaped = escaped.replace(")", "\\29")
    escaped = escaped.replace("\x00", "\\00")
    escaped = escaped.replace("&", "\\26")
    return escaped


def resolve_ldap_tls(url: str, *, start_tls: bool, allow_insecure: bool) -> str:
    """Map an LDAP URL to a transport mode: ``"ssl" | "starttls" | "plain"``.

    Plain ``ldap://`` is rejected unless the deployment explicitly opts in
    with ``start_tls`` (cleartext then upgraded via StartTLS) or
    ``allow_insecure`` (cleartext, never recommended).
    """
    if "://" not in url:
        raise ValueError(f"unsupported LDAP URL: {url!r}")
    scheme = url.split("://", 1)[0].lower()
    if scheme == "ldaps":
        return "ssl"
    if scheme == "ldap":
        if start_tls:
            return "starttls"
        if allow_insecure:
            return "plain"
        raise ValueError("plain ldap:// requires start_tls=True or allow_insecure=True")
    raise ValueError(f"unsupported LDAP URL scheme: {scheme}")


class LdapClient:
    """Verifies domain credentials: search for the user, then rebind as the user.

    ``issuer`` is the configured URL; the broker stamps it onto the LDAP
    principal. With no injected ``transport``, a fresh ``Ldap3Transport`` is
    built per ``verify()`` call, so one login never shares a socket with another.
    """

    def __init__(
        self,
        url: str,
        *,
        base_dn: str,
        bind_dn: str,
        bind_password: str,
        user_filter: str = DEFAULT_USER_FILTER,
        start_tls: bool = False,
        allow_insecure: bool = False,
        ca_file: str | None = None,
        timeout: float = 5.0,
        transport: LdapTransport | None = None,
    ) -> None:
        # Fail early (ValueError) on a plain ldap:// URL without an explicit opt-in.
        resolve_ldap_tls(url, start_tls=start_tls, allow_insecure=allow_insecure)
        self.issuer = url
        self.base_dn = base_dn
        self.bind_dn = bind_dn
        self.bind_password = bind_password
        self.user_filter = user_filter
        self.start_tls = start_tls
        self.allow_insecure = allow_insecure
        self.ca_file = ca_file
        self.timeout = timeout
        self._transport = transport

    def verify(self, username: str, password: str) -> LdapIdentity | None:
        transport = self._transport if self._transport is not None else self._build_transport()
        transport.connect()
        try:
            search_filter = self.user_filter.format(username=_escape(username))
            found = transport.search_user(search_filter)
            if found is None:
                return None
            if (
                not isinstance(found, tuple)
                or len(found) != 2
                or not isinstance(found[0], str)
                or not isinstance(found[1], dict)
            ):
                raise LdapUnavailable("LDAP search returned an ambiguous or malformed result")
            dn, attrs = found
            if not transport.user_bind(dn, password):
                return None
            return self._identity(username, attrs)
        finally:
            transport.close()

    def _build_transport(self) -> LdapTransport:
        return Ldap3Transport(
            self.issuer,
            base_dn=self.base_dn,
            bind_dn=self.bind_dn,
            bind_password=self.bind_password,
            start_tls=self.start_tls,
            allow_insecure=self.allow_insecure,
            ca_file=self.ca_file,
            timeout=self.timeout,
        )

    @staticmethod
    def _identity(username: str, attrs: dict[str, Any]) -> LdapIdentity:
        display_name = str(attrs.get("displayName") or attrs.get("cn") or username)
        email = attrs.get("mail")
        email = None if email is None or "@" not in str(email) else str(email)
        # Binary thumbnailPhoto is not converted in v1; avatar URL storage is a future feature.
        return LdapIdentity(username=username, display_name=display_name, email=email, avatar_url=None)


class Ldap3Transport:
    """ldap3 implementation of the seam; one short-lived connection per verify()."""

    def __init__(
        self,
        url: str,
        *,
        base_dn: str,
        bind_dn: str,
        bind_password: str,
        start_tls: bool = False,
        allow_insecure: bool = False,
        ca_file: str | None = None,
        timeout: float = 5.0,
    ) -> None:
        try:
            from ldap3 import NONE, Server, Tls
        except ImportError as exc:  # pragma: no cover - depends on the server extra
            raise RuntimeError("Install Kraken with the 'server' extra to use LDAP sign-in") from exc
        self.base_dn = base_dn
        self.bind_dn = bind_dn
        self.bind_password = bind_password
        self.timeout = timeout
        self._mode = resolve_ldap_tls(url, start_tls=start_tls, allow_insecure=allow_insecure)
        tls = None if self._mode == "plain" else Tls(validate=ssl.CERT_REQUIRED, ca_certs_file=ca_file)
        # Server parses the URL (scheme, host, port); construction touches no network.
        self._server = Server(url, get_info=NONE, connect_timeout=timeout, tls=tls)
        self._connection: Any | None = None

    def connect(self) -> None:
        from ldap3 import Connection
        from ldap3.core.exceptions import (
            LDAPBindError,
            LDAPCommunicationError,
            LDAPResponseTimeoutError,
            LDAPStartTLSError,
        )

        self._teardown()
        connection = Connection(
            self._server,
            user=self.bind_dn,
            password=self.bind_password,
            auto_bind=False,
            auto_escape=False,  # filter values are already escaped by LdapClient._escape
            receive_timeout=self.timeout,
        )
        self._connection = connection
        try:
            if self._mode == "starttls":
                if not connection.start_tls(read_server_info=False):
                    raise LdapUnavailable("LDAP StartTLS negotiation failed")
            if not connection.bind(read_server_info=False):
                raise LdapAuthenticationError("LDAP service-account bind was rejected")
        except (LDAPBindError, LDAPCommunicationError, LDAPResponseTimeoutError, LDAPStartTLSError, OSError) as exc:
            self._teardown()
            raise LdapUnavailable("LDAP server is unavailable") from exc
        except (LdapAuthenticationError, LdapUnavailable):
            # Bind/StartTLS reported failure via return value: release the
            # connection before the typed error reaches the client.
            self._teardown()
            raise

    def search_user(self, search_filter: str) -> tuple[str, dict[str, Any]] | None:
        from ldap3.core.exceptions import (
            LDAPCommunicationError,
            LDAPResponseTimeoutError,
        )

        connection = self._connection
        if connection is None:
            raise LdapUnavailable("LDAP transport is not connected")
        try:
            ok = connection.search(
                self.base_dn,
                search_filter,
                attributes=("sAMAccountName", "displayName", "cn", "mail"),
                auto_escape=False,
            )
        except (LDAPCommunicationError, LDAPResponseTimeoutError, OSError) as exc:
            raise LdapUnavailable("LDAP search failed") from exc
        if not ok:
            raise LdapUnavailable("LDAP search was rejected by the server")
        entries = connection.entries
        if not entries:
            return None
        if len(entries) > 1:
            raise LdapUnavailable("LDAP search matched more than one user")
        entry = entries[0]
        attrs: dict[str, str] = {}
        for name in entry:
            values = entry[name].value
            if values:
                attrs[str(name)] = str(values[0])
        return str(entry.dn), attrs

    def user_bind(self, dn: str, password: str) -> bool:
        from ldap3.core.exceptions import (
            LDAPBindError,
            LDAPCommunicationError,
            LDAPResponseTimeoutError,
        )

        connection = self._connection
        if connection is None:
            raise LdapUnavailable("LDAP transport is not connected")
        try:
            return bool(connection.rebind(dn, password, read_server_info=False))
        except (LDAPBindError, LDAPCommunicationError, LDAPResponseTimeoutError, OSError) as exc:
            raise LdapUnavailable("LDAP user bind failed") from exc

    def close(self) -> None:
        self._teardown()

    def _teardown(self) -> None:
        connection = self._connection
        self._connection = None
        if connection is None:
            return
        # Best effort: the session may already be broken, and close() runs from
        # the client's finally block where a secondary error would mask the
        # exception in flight.
        try:
            connection.unbind()
        except Exception:  # noqa: BLE001 - cleanup path
            pass
        try:
            connection.strategy.close()
        except Exception:  # noqa: BLE001 - cleanup path
            pass


__all__ = [
    "DEFAULT_USER_FILTER",
    "Ldap3Transport",
    "LdapAuthenticationError",
    "LdapClient",
    "LdapIdentity",
    "LdapTransport",
    "LdapUnavailable",
    "resolve_ldap_tls",
]
